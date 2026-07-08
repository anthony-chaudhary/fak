<!--
  How-to: consume the fak change feed correctly (#3173).
  Concept page: ../explainers/change-data-capture-for-agents.md
  Reference consumer: ../../pkg/fakclient/changes_consumer.go
-->

# Consuming the fak change feed

fak exposes what agents change and refute as a **log-based change feed**, the same
shape a database CDC consumer expects. Read it by **offset**, not by polling for
diffs: hold a cursor, ask for everything after it, apply, advance. This page is the
consumer contract — the five habits a correct reader needs — and points at a
reference consumer that already encodes all five so you don't reinvent them.

> New to why fak has a change feed at all? Start with the concept page,
> [Change Data Capture for Agents](../explainers/change-data-capture-for-agents.md).
> This page is the how-to that sits under its "How do I consume the feed?" section.

## The two feeds

| Feed | Endpoint | Source | Use it for |
|---|---|---|---|
| **Live coherence bus** | `GET /v1/fak/changes?since=<cursor>` | cache mutations + integrity revocations, ordered by a shared `Seq` | reacting *now* — evict a private cache, re-plan when a peer changed shared state |
| **Durable audit journal** | `GET /v1/fak/events?since=<cursor>` | the hash-chained decision ledger | completeness and audit — replay from any point, reconcile a gap, verify the chain |

Both drain by the same `?since=` cursor and return `{events, cursor}`. The live bus
is bounded (a ring that ages out); the journal is durable. When the bus reports a
gap, the journal is the authority you reconcile against.

## The consumer contract

### 1. Resume from a cursor
Start at `0`. Each response carries the `cursor` to pass as the next `since`. The
cursor is the `Seq` you have consumed *through* — persist it (a file, a row, a
watermark) and reload it on restart so a restart pages forward instead of
re-reading the window. `since=0` means "everything still retained," not "from the
beginning of time" — the bus only holds its window.

### 2. Delivery is at-least-once — dedupe by identity
The same event may be delivered more than once (a retry after a handler error, an
overlapping drain). Make application idempotent: dedupe by the event's identity —
`Seq` for ordering, or the semantic key (`Witness` for a revocation) for effect.
Applying the same change twice must be a no-op. The reference consumer dedupes by
`Seq`: an event at or below the cursor is skipped, so your handler sees each advance
exactly once even when the wire repeats it.

### 3. A retention gap means re-sync, not panic
The live bus is a bounded ring. A consumer that lapses past the retained window
cannot see the events that aged off — it observes a **`Seq` gap** (the first event
returned sits above `since + 1`). Treat a gap as "you *might* have missed changes;
reconcile against the durable journal (`GET /v1/fak/events`) before assuming
continuity." It is a conservative signal — the shared `Seq` counter is monotone but
not contiguous per consumer (one counter spans the mutation and revocation buses,
and drains are principal-scoped), so a gap can be benign. Over-reporting a gap is
safe; missing one is not.

### 4. Fail without losing your place
If applying an event fails, stop the drain **before** advancing past it and leave
the cursor where it was. The next drain re-presents the failed event (that is the
at-least-once contract working for you), so a transient failure retries instead of
silently skipping a change.

### 5. Visibility is principal-scoped
A drain returns only what the requesting principal may see: its own changes plus
principal-less **global** broadcasts, never a peer tenant's. Set the principal on
the client (`WithPrincipal`) to scope your feed; an unscoped/admin client sees
everything. Don't assume the `Seq` you see are contiguous — the gaps are often just
another tenant's events you were never shown (see habit 3).

## Reference consumer

`pkg/fakclient` ships a small, resumable, at-least-once reader that encodes all five
habits — [`changes_consumer.go`](../../pkg/fakclient/changes_consumer.go). The value
is the contract it demonstrates, not the line count:

```go
client := fakclient.New("http://127.0.0.1:8080", fakclient.WithPrincipal("tenant-a"))

// Resume from a persisted cursor (0 on first run).
cs := fakclient.NewConsumerAt(client, savedCursor)

for {
    n, err := cs.Drain(ctx, func(ctx context.Context, ev fakclient.ChangeEvent) error {
        return apply(ev) // idempotent: safe if ev is re-presented
    })
    if err != nil {
        // handler failed; cursor is left before the failing event — retry next Drain
        break
    }
    if cs.Gapped() {
        reconcileAgainstJournal(ctx, client) // GET /v1/fak/events — the durable authority
    }
    persist(cs.Cursor()) // durable across restarts
    if n == 0 {
        wait() // caught up to head
    }
}
```

- `NewConsumerAt(client, cursor)` / `Cursor()` — habit 1 (resume).
- `Drain` dedupes by `Seq` — habit 2 (idempotent delivery).
- `Gapped()` — habit 3 (a conservative re-sync prompt; reconcile against the journal).
- A handler error stops the drain with the cursor left before the failure — habit 4.
- `WithPrincipal` scopes the feed — habit 5.

The consumer holds only a cursor and a sticky gap flag — no goroutine, no network
state — so it is cheap to build and safe to persist. Drive it from a single loop
(it is not safe for concurrent `Drain` calls).

## See also

- [Change Data Capture for Agents](../explainers/change-data-capture-for-agents.md) — the concept page and the CDC ↔ fak seam map.
- [`pkg/fakclient/changes_consumer.go`](../../pkg/fakclient/changes_consumer.go) — the reference consumer.
- [Verify, don't trust](../explainers/verify-dont-trust.md) — the read-model side: a done-claim graded against git, not the worker's word.
