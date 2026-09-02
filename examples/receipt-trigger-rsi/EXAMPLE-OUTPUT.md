# Captured run — `examples/receipt-trigger-rsi`

Real runs, offline, no key/model/GPU. Identical input prints an identical
decision receipt (verified over consecutive runs) — the signature and effect id
are derived from the receipt's own fields, not from wall-clock noise.

## 1. Selfcheck

```console
$ go run ./examples/receipt-trigger-rsi -selfcheck
selfcheck: PASS (routing, signatures, recursion, duplicate, stale, unknown schema)
```

## 2. Classify one typed receipt

```console
$ echo '{"schema":"fak-guard-crash/1","reason":"TERMINAL_CRASH","producer":"guard","produced_at":"2026-08-31T11:59:00Z","effect_key":"panic:ctxmmu","capacity":1,"expected_value":5,"evidence_refs":["receipt:abc"]}' | go run ./examples/receipt-trigger-rsi -now 2026-08-31T12:00:00Z
{
  "schema": "fak-receipt-trigger-decision/1",
  "decision": "RUN",
  "reason": "MATCHED_RECEIPT_READY",
  "consumer": "guard-crash-rsi",
  "authority": "authoritative",
  "signature": "e7e353db6254403a0b26720c",
  "effect_id": "effect:1708be52472c3bf4ffa2e15e",
  "outcome_link": "outcome:e7e353db6254403a0b26720c",
  "evidence_refs": [
    "receipt:abc"
  ]
}
```

## What the capture proves

- **The gate chain admits exactly the narrow case.** A well-formed, fresh,
  non-duplicate `fak-guard-crash/1` receipt with capacity and expected value
  routes `RUN` to the `guard-crash-rsi` consumer with `authoritative` authority.
- **The decision is a pure function of the input.** Same receipt in, same
  signature `e7e353db6254403a0b26720c`, same effect id, same `outcome_link` out —
  so a later outcome receipt can join deterministically.
- **Nothing mutates.** The demo reads one JSON object and prints one decision —
  no launch, no issue write, no side effect.
