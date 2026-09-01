# Replay-safe semantic event consumer

This stdlib-only example consumes the public `pkg/harnesskit` semantic envelopes without importing provider SDK objects or `internal/` packages. Run it with:

```bash
go run ./examples/custom-harness/event-consumer
go test ./examples/custom-harness/event-consumer
```

The demonstration requests bounded credit, commits a first batch, simulates a disconnect by redelivering the original records, and resumes from the sequence **after** its committed semantic cursor. It also proves duplicate event and input handling, safe ignore-and-advance for an additive unknown event, and non-persistence of private or secret payloads.

## Production boundary

A real transport or sidecar should feed the same `harnesskit.Envelope` values into this consumer. This repository currently exposes projection of captured JSONL through `fak harness protocol project`; this example does **not** claim that a live public sidecar command exists.

Keep two acknowledgements distinct:

1. **Semantic cursor acknowledgement** records that the builder-owned projection/effect and its fak event sequence committed.
2. **Broker or transport acknowledgement** tells Kafka, NATS, SQS, or another transport that it may stop redelivering a message.

In production, commit the projection or idempotent domain effect and semantic cursor in one database transaction (or use an outbox/inbox equivalent). Only then acknowledge the broker message. Delivery is at least once; effect keys such as `EventID`, `CallID`, and `InputID` make redelivery safe. Do not promise exactly-once delivery across independent systems.

Sensitivity is an envelope-level policy signal, not permission to store payloads. This example counts private/secret records and advances their cursor without retaining their payload. A production consumer should apply its own authorized redaction or encrypted-storage policy before persistence.
