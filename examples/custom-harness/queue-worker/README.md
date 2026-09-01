# BYO-queue worker adapter

This example shows the smallest useful queue boundary for a durable worker built on the public
`github.com/anthony-chaudhary/fak/pkg/harnesskit` semantic envelope. The builder owns the queue
transport, leases, persistence, retries, and domain effects; fak supplies stable semantic event
contracts and governance.

Run it:

```bash
go run ./examples/custom-harness/queue-worker
```

The sample uses an in-memory queue and store so it is deterministic and stdlib-only. Replace
`Queue` with an adapter for the broker you already operate and replace `Store` with durable
transactions.

## Contract demonstrated

- `JobID`, `RunID`, `InputID`, and each `Envelope.EventID` stay stable across redelivery.
- `Receive(ctx, capacity, lease)` provides bounded credit/backpressure; the worker never asks for
  more than its configured capacity.
- Delivery is at least once. `Retry` deliberately redelivers the same logical job and IDs.
- Input, job, and event effects are idempotent. A duplicate delivery can be acknowledged without
  repeating committed domain effects.
- `ApplyEvent` is the semantic commit: in production, persist the domain effect, event idempotency
  key, and exclusive run cursor atomically in one database transaction.
- Broker acknowledgement is separate and later: `Ack` removes the transport message only after
  semantic effects and the job commit succeed. A crash between those boundaries causes safe
  redelivery, not an exactly-once promise.
- Transient failures retry; poison deliveries move to a dead-letter destination after the bounded
  attempt count.
- Cancellation flows through `context.Context`; canceled work is not acknowledged.

## Deliberate boundary

fak is **not** a broker or workflow scheduler. Broker-native tuning—visibility timeouts, lease
renewal, partitions, ordering, consumer groups, retry delays, and dead-letter retention—remains
external and builder-owned. A real transport can feed jobs containing public harnesskit semantic
envelopes, but this example does **not** claim that a live fak sidecar/server command exists.

For production, also make input admission (`InputID`) atomic with its durable effect, renew long
leases using the selected broker's API, record cancellation outcomes, and add telemetry around
queue age, retry count, dead-letter volume, semantic commit latency, and ack failures.
