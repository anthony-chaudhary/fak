# Micro-context S4d: bounded multi-turn continuation

**Status:** observed deterministic 1,000-context contract witness, 2026-08-06; resolves #5818.

Descriptor v2 extends the process-free S4a descriptor without changing its Host/Gateway runtime. Each logical context now carries a positive turn budget and a stable continuation token. The adapter accounts one completed gateway call per turn, preserves assistant history between calls, finishes only when the output contract matches, and refuses both an unmatched final turn and any call beyond the budget. Descriptor v1 remains accepted and exactly one turn.

Between turns, `DescriptorAgent.Freeze` serializes only mutable continuation state: logical identity, continuation token, turns used, last result, and assistant history. `HibernationStore.Wake` independently re-freezes after `Thaw` and refuses a non-byte-identical restore. The immutable base and descriptor remain outside the parked payload.

## Captured witness

```powershell
go run ./cmd/microcontextdemo -multi-turn-descriptor experiments/microcontext/s4d-local-multi-turn-descriptor-1000x3-pass-2026-08-06.json -contexts 1000 -workers 16 -turns 3
go run ./cmd/microcontextdemo -verify-multi-turn-descriptor experiments/microcontext/s4d-local-multi-turn-descriptor-1000x3-pass-2026-08-06.json
```

The ledger reconciles exactly:

- 1,000 logical contexts × 3 budgeted turns = 3,000 expected and 3,000 accounted gateway turns;
- 1,000/1,000 output contracts completed, with zero failures;
- 1,000 stable continuation tokens and zero trace mismatches;
- 1,000 mid-task parks and 1,000 byte-verified restores;
- one immutable base installation over 16 bounded Host workers;
- 201,000 total parked bytes and 2.685 s wall time for this deterministic local fixture.

The verifier rejects a missing turn, continuation mismatch, failed context, missing restore, non-positive rate, or non-v2 descriptor provenance.

## Claim boundary

This witnesses exact scheduler-level turn accounting, continuation identity through the existing Gateway context, and hibernate-safe between-turn state. The 372.44 verified completions/s value is deterministic fixture throughput. It is **not** model inference throughput, model quality, prefix-cache benefit, KV residency, or per-context process/RSS evidence. Tool/effect policy remains the separately shipped S4c seam.
