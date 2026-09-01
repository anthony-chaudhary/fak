# Portable replay/eval fixture

This minimal example captures one semantic fak run as versioned, language-neutral JSON, scrubs it, writes and reads it, then replays it offline into three products:

- `reference`: a stock/reference transcript projection;
- `ops-console`: a custom operational timeline;
- `pickup-card`: a custom UI/product card.

It imports only the standard library and public `pkg/harnesskit` protocol types. It does **not** import command, internal, or provider SDK types and does not claim a live sidecar or server command exists.

## Contract and ownership

fak owns stable run/input/event identity, public envelope semantics, sensitivity labels, compatibility, and witnessed outcomes. A builder owns fixture storage, product projections, evaluation policy, external queues/brokers, and domain-specific scrubbing rules. Broker acknowledgement is not a semantic cursor/effect commit. Delivery may be at least once; this example makes no exactly-once claim.

The JSON includes provenance and effective configuration, semantic input, public events, recorded tool outcomes, checkpoints, explicit time/randomness/concurrency/provider nondeterminism, expected product projections, expected outcome, and a machine-readable scrub report.

Strict comparison checks every projected byte of meaning. Tolerant comparison removes only the exact JSON-pointer patterns declared by the fixture (`/*/meta/observed_at` and `/*/meta/sample` here); behavioral and UI mutations still fail.

## Privacy boundary

The generic example scrubber detects its demonstrated key/token and email patterns, records replacement counts and paths, verifies the serialized bytes, and refuses an unsafe write. It is **not** a universal privacy or compliance system: builders must add domain-specific detectors, review sensitivity labels, minimize captures, control access/retention, and test against their real data.

## Run

```bash
go test -count=1 ./examples/custom-harness/replay-fixture
go run ./examples/custom-harness/replay-fixture
```

This is the smallest portable spine, not the final universal fixture format, broker, workflow scheduler, live replay service, or complete eval framework.
