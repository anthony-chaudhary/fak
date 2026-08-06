# Micro-context S4c: parallel effect safety

**Status:** observed controlled-kernel adversarial fixture, 2026-08-06.

`EffectCoordinator` composes existing fak seams rather than trusting parallel model turns:

1. each intent names its context, required tool authority, resource, operation, and idempotency token;
2. explicit per-context tools default-deny absent authority;
3. resource ownership is nonblocking, so a conflict is quarantined/refused while disjoint work continues;
4. `internal/idempotency` serializes check/apply and persists landed results across process-store reopen;
5. an independently supplied readback must observe the external effect before the result is marked verified.

```powershell
go run ./cmd/microcontextdemo -effects-witness experiments/microcontext/s4-local-effects-2026-08-06.json
go run ./cmd/microcontextdemo -verify-effects experiments/microcontext/s4-local-effects-2026-08-06.json
```

Observed fixture: two disjoint writes completed concurrently; one same-resource conflict was refused; one context lacking write authority was denied; three physical writes landed; retry after reopening the durable store replayed the prior result without a fourth apply; all four successful/replayed outcomes passed independent file readback. Four admitted intents were journaled.

The effect coordinator is separate from model slots, so waiting on a resource does not hold an inference slot. The fixture establishes local file-effect behavior and durable dedupe, not arbitrary shell/database/message safety, distributed transactions, or power-loss atomicity between an external effect and ledger append. Backends need their own idempotency/readback contracts.
