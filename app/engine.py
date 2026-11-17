"""Single-process micro-batch inference engine with runtime order reclamation.

Concurrency model
-----------------
* Everything except the blocking model forward pass runs on the FastAPI
  asyncio event loop, so no locks are needed for the shared state below.
* Exactly one worker task forms batches and exactly one executor thread runs
  the TorchScript model, therefore at most ONE batch is ever actually
  inferring at any moment.
* Capacity accounting: an order occupies a slot while it is queued AND while
  its batch is computing (even if it was cancelled or expired mid-flight).
  The slot is released only when the batch compute actually finishes.

State machine per order
-----------------------
queued -> processing -> completed
queued -> cancelled            (cancelled before entering a batch; never runs)
queued -> expired              (deadline passed while queued; never runs)
processing -> cancelled        (marked immediately, result discarded, slot held)
processing -> expired          (deadline passed mid-flight; result discarded)
processing -> failed           (model raised; defensive only)
Late results are written back ONLY to orders still in `processing` whose
deadline has not passed, so a late result can never revive a cancelled or
expired order.

Receipt retention (runtime reclamation)
---------------------------------------
A terminal order is *settled* as soon as it no longer occupies the queue or
an actual computation:

* completed / failed            -> settled when their batch compute ends
* cancelled / expired queued    -> settled immediately (slot already freed)
* cancelled / expired in-flight -> settled only when the whole batch compute
                                   actually ends (slot held until then)

Settling drops the image tensor reference and moves the order from the live
table into the receipt store. A receipt keeps status/result queryable for
`receipt_ttl_ms` (default 60 s) and the store keeps at most `receipt_max`
(default 128) entries; overflow evicts oldest-settled first. A background
sweeper settles expired queued orders and reclaims timed-out receipts even
when no requests arrive. Once a receipt is evicted, lookups return 404.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

import torch

log = logging.getLogger("batch-engine")

QUEUED = "queued"
PROCESSING = "processing"
COMPLETED = "completed"
CANCELLED = "cancelled"
EXPIRED = "expired"
FAILED = "failed"


# eq=False: orders are compared by identity. The default dataclass __eq__
# would compare tensors element-wise and raise on multi-element tensors.
@dataclass(eq=False)
class Order:
    id: str
    # None once the batch tensor is stacked, and always None on a receipt:
    # settled orders must not reference image tensors.
    tensor: Optional[torch.Tensor]
    deadline_ms: int
    deadline_at: float  # time.monotonic() at which the order expires
    state: str = QUEUED
    result: Optional[dict] = None
    batch_id: Optional[str] = None
    created_at: float = field(default_factory=time.monotonic)
    settled_at: Optional[float] = None  # monotonic time of settlement; receipts only


class BatchEngine:
    def __init__(
        self,
        model_path: str,
        max_capacity: int = 16,
        batch_size: int = 4,
        batch_wait_ms: int = 50,
        inference_delay_ms: int = 0,
        receipt_ttl_ms: int = 60_000,
        receipt_max: int = 128,
        sweep_interval_ms: int = 1000,
    ) -> None:
        self.max_capacity = max_capacity
        self.batch_size = batch_size
        self.batch_wait_s = batch_wait_ms / 1000.0
        self.inference_delay_s = inference_delay_ms / 1000.0
        self.receipt_ttl_s = receipt_ttl_ms / 1000.0
        self.receipt_max = receipt_max
        self.sweep_interval_s = sweep_interval_ms / 1000.0

        # Single-threaded CPU inference keeps results deterministic and makes
        # the "one batch at a time" guarantee easy to reason about.
        torch.set_num_threads(1)
        self.model = torch.jit.load(model_path, map_location="cpu")
        self.model.eval()

        self._pending: deque[Order] = deque()
        self._orders: dict[str, Order] = {}  # live orders: queued or in-flight
        # Settled orders (receipts). Insertion order == settlement order,
        # which is the eviction order required by the retention policy.
        self._receipts: OrderedDict[str, Order] = OrderedDict()
        self._inflight = 0  # orders inside the batch currently being computed
        self._not_empty = asyncio.Event()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="infer")
        self._worker: Optional[asyncio.Task] = None
        self._sweeper: Optional[asyncio.Task] = None
        # Counters for /stats and tests.
        self.batches_started = 0
        self.last_batch_size = 0
        self.settled_total = 0

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    async def start(self) -> None:
        self._worker = asyncio.create_task(self._run(), name="batch-worker")
        self._sweeper = asyncio.create_task(self._sweep_loop(), name="receipt-sweeper")

    async def stop(self) -> None:
        for task in (self._worker, self._sweeper):
            if task is not None:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._executor.shutdown(wait=False)

    # ------------------------------------------------------------------ #
    # settlement: terminal order -> receipt (event-loop thread only)
    # ------------------------------------------------------------------ #
    def _settle(self, o: Order) -> None:
        """Move a terminal order that occupies nothing into the receipt store."""
        if o.settled_at is not None:
            return  # already settled (e.g. a cancel raced the batch gate)
        o.tensor = None  # receipts never reference image tensors
        o.settled_at = time.monotonic()
        self._orders.pop(o.id, None)
        self._receipts[o.id] = o
        self.settled_total += 1
        # Cap: evict oldest-settled first (insertion order == settle order).
        while len(self._receipts) > self.receipt_max:
            self._receipts.popitem(last=False)

    def _expire_receipts(self) -> None:
        """Drop receipts older than the TTL. Runs even with no traffic."""
        if not self._receipts:
            return
        cutoff = time.monotonic() - self.receipt_ttl_s
        # Receipts are ordered by settled_at, so stop at the first fresh one.
        while self._receipts:
            o = next(iter(self._receipts.values()))
            if o.settled_at is None or o.settled_at > cutoff:
                break
            self._receipts.popitem(last=False)

    async def _sweep_loop(self) -> None:
        """Background reclamation, independent of incoming requests:
        settle queued orders whose deadline passed and reclaim receipts
        whose retention window has ended."""
        while True:
            await asyncio.sleep(self.sweep_interval_s)
            self._purge_expired_pending()
            self._expire_receipts()

    # ------------------------------------------------------------------ #
    # helpers (event-loop thread only)
    # ------------------------------------------------------------------ #
    def _purge_expired_pending(self) -> None:
        """Settle queued orders whose deadline passed; they never enter the model."""
        now = time.monotonic()
        kept: list[Order] = []
        expired: list[Order] = []
        for o in self._pending:
            if o.state == QUEUED and o.deadline_at <= now:
                o.state = EXPIRED
                expired.append(o)
            else:
                kept.append(o)
        if len(kept) != len(self._pending):
            self._pending.clear()
            self._pending.extend(kept)
        for o in expired:
            self._settle(o)  # queued expiry frees the slot at once

    def _refresh_state(self, o: Order) -> None:
        """Lazy expiry when an order is observed."""
        now = time.monotonic()
        if o.state == QUEUED and o.deadline_at <= now:
            try:
                self._pending.remove(o)
            except ValueError:
                pass
            o.state = EXPIRED
            self._settle(o)  # queued expiry frees the slot at once
        elif o.state == PROCESSING and o.deadline_at <= now:
            # Terminal state is visible immediately, but the slot stays
            # occupied until the batch compute actually ends: settle later.
            o.state = EXPIRED

    def capacity_used(self) -> int:
        self._purge_expired_pending()
        return len(self._pending) + self._inflight

    def stats(self) -> dict:
        self._purge_expired_pending()
        counts: dict[str, int] = {}
        # Copy: refreshing a live order may settle it (mutating _orders).
        for o in list(self._orders.values()):
            self._refresh_state(o)
            counts[o.state] = counts.get(o.state, 0) + 1
        for o in self._receipts.values():
            counts[o.state] = counts.get(o.state, 0) + 1
        return {
            "capacity_used": len(self._pending) + self._inflight,
            "max_capacity": self.max_capacity,
            "queued": len(self._pending),
            "inflight": self._inflight,
            "batches_started": self.batches_started,
            "last_batch_size": self.last_batch_size,
            "receipts": len(self._receipts),
            "settled_total": self.settled_total,
            "orders": counts,
        }

    # ------------------------------------------------------------------ #
    # public API used by the HTTP layer
    # ------------------------------------------------------------------ #
    def submit(self, tensor: torch.Tensor, deadline_ms: int) -> Optional[Order]:
        """Accept an image, or return None when at capacity (16 by default)."""
        if self.capacity_used() >= self.max_capacity:
            return None
        order = Order(
            id=uuid.uuid4().hex[:12],
            tensor=tensor,
            deadline_ms=deadline_ms,
            deadline_at=time.monotonic() + deadline_ms / 1000.0,
        )
        self._orders[order.id] = order
        self._pending.append(order)
        self._not_empty.set()
        return order

    def get(self, order_id: str) -> Optional[Order]:
        o = self._orders.get(order_id)
        if o is not None:
            self._refresh_state(o)  # may settle a queued order that just expired
            return o
        return self._receipts.get(order_id)

    def cancel(self, order_id: str) -> Optional[Order]:
        o = self._orders.get(order_id)
        if o is None:
            # Already settled (or unknown): the terminal state is final.
            return self._receipts.get(order_id)
        self._refresh_state(o)
        if o.state == QUEUED:
            try:
                self._pending.remove(o)
            except ValueError:
                pass
            o.state = CANCELLED
            self._settle(o)  # queued cancel frees the slot at once
        elif o.state == PROCESSING:
            # Mark immediately. The slot stays occupied until the batch
            # compute actually ends; no new batch is started because of it.
            o.state = CANCELLED
        return o

    # ------------------------------------------------------------------ #
    # batching worker
    # ------------------------------------------------------------------ #
    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            await self._not_empty.wait()

            # Batch-formation window: the idle model waits at most
            # `batch_wait_s` for more work, up to `batch_size` images.
            batch: list[Order] = []
            window_end = loop.time() + self.batch_wait_s
            while len(batch) < self.batch_size:
                self._purge_expired_pending()
                while self._pending and len(batch) < self.batch_size:
                    batch.append(self._pending.popleft())
                if len(batch) >= self.batch_size:
                    break
                remaining = window_end - loop.time()
                if remaining <= 0:
                    break
                self._not_empty.clear()
                try:
                    await asyncio.wait_for(self._not_empty.wait(), remaining)
                except asyncio.TimeoutError:
                    break

            # Final gate: cancelled/expired orders never enter the model.
            now = time.monotonic()
            kept: list[Order] = []
            for o in batch:
                if o.state == QUEUED and o.deadline_at > now:
                    kept.append(o)
                else:
                    # Popped from the queue but no longer runnable: it
                    # occupies nothing, so settle it right away.
                    if o.state == QUEUED:
                        o.state = EXPIRED
                    self._settle(o)
            batch = kept
            if not batch:
                continue

            batch_id = uuid.uuid4().hex[:8]
            for o in batch:
                o.state = PROCESSING
                o.batch_id = batch_id
            self._inflight = len(batch)
            self.batches_started += 1
            self.last_batch_size = len(batch)
            tensors = torch.stack([o.tensor for o in batch])
            # The stacked batch is a copy; per-order tensors are no longer
            # needed, so release that memory while the batch computes.
            for o in batch:
                o.tensor = None

            try:
                scores = await loop.run_in_executor(self._executor, self._infer, tensors)
            except Exception:  # noqa: BLE001 - defensive: model failure must not kill the worker
                log.exception("inference failed for batch %s", batch_id)
                for o in batch:
                    if o.state == PROCESSING:
                        o.state = FAILED
                self._inflight = 0
                # Compute actually ended: every order is terminal and its
                # slot is released, so settle the whole batch.
                for o in batch:
                    self._settle(o)
                continue

            now = time.monotonic()
            for o, sc in zip(batch, scores):
                if o.state != PROCESSING:
                    # Cancelled or expired mid-flight: result is discarded and
                    # the late result must NOT revive the order.
                    continue
                if o.deadline_at <= now:
                    o.state = EXPIRED
                    continue
                o.state = COMPLETED
                predicted = max(range(len(sc)), key=lambda i: sc[i])
                o.result = {"predicted_class": predicted, "scores": sc}
            self._inflight = 0
            # The batch compute actually ended: every order in it is terminal
            # and releases its slot, so settle them all into receipts.
            for o in batch:
                self._settle(o)

    # ------------------------------------------------------------------ #
    # runs on the single executor thread — never touches shared state
    # ------------------------------------------------------------------ #
    def _infer(self, batch_tensor: torch.Tensor) -> list[list[float]]:
        if self.inference_delay_s > 0:
            time.sleep(self.inference_delay_s)  # demo: simulate a slow model
        with torch.inference_mode():
            logits = self.model(batch_tensor)
            probs = torch.softmax(logits, dim=1)
        return [[float(v) for v in row] for row in probs]
