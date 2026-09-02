# Receipt-triggered RSI example

This zero-network example turns a **bounded typed receipt** into a deterministic proposal for one narrowly matched development loop. It is deliberately not an event bus: it reads one JSON object, applies admission gates, prints one decision receipt, and performs no launch, issue write, or other mutation.

Requires: Go 1.26+ (the repository toolchain). No API key, network, model, or GPU.

```powershell
go run ./examples/receipt-trigger-rsi -selfcheck
```

Or classify a receipt:

```powershell
'{"schema":"fak-guard-crash/1","reason":"TERMINAL_CRASH","producer":"guard","produced_at":"2026-08-31T11:59:00Z","effect_key":"panic:ctxmmu","capacity":1,"expected_value":5,"evidence_refs":["receipt:abc"]}' | go run ./examples/receipt-trigger-rsi -now 2026-08-31T12:00:00Z
```

The output uses `RUN`, `SKIP`, `DEFER`, `MERGE`, or `REROUTE`, a closed stable reason, a normalized signature, an effect identity for deduplication, and an `outcome_link` that a later outcome receipt can join without embedding logs or secrets.

## What you'll see

A captured run is in [`EXAMPLE-OUTPUT.md`](EXAMPLE-OUTPUT.md): the `selfcheck: PASS (...)` line and the `RUN` decision receipt for the guard-crash example above. Both invocations complete in under a second on a warm build cache (the first `go run` compiles the package once), and the identical input prints the identical decision — the signature and effect id are pure functions of the receipt fields.

## Safety contract

Input is an explicit bounded schema: schema, reason, producer, timestamp, effect key, recursion marker, capacity, expected-value integer, sample count, and bounded evidence references. JSON unknown fields are rejected. Ambient logs, environments, prompts, credentials, and arbitrary payloads never enter the trigger context.

Gate precedence is: known contract → recursion → freshness → duplicate effect → capacity → heuristic minimum sample → expected value → run. Unknown contracts reroute to a contract audit rather than guessing. This demo renders only; production consumers retain their own capability and witness gates.

## Opportunity map

| Producer event | Narrow consumer | Signal class | Why direct routing helps |
|---|---|---|---|
| Terminal guard crash/failure receipt | guard-crash RSI | Authoritative | Starts bounded root-cause work from the exact typed failure. |
| Repeated `BLOCKED_BY_GUARD` receipts | guard audit | Authoritative after recurrence | Finds a structural wedge instead of retrying the denied call. |
| Quality-constrained benchmark regression | bisect/profile/SOTA-check packet | Authoritative | Preserves the measured envelope and moves directly to diagnosis. |
| Producer/consumer schema mismatch | trigger contract audit | Authoritative | Repairs drift before consumers silently miss events. |
| Recurring intent/trajectory cluster | harness-garden proposal | Heuristic | Converts repeated operator work into a bounded improvement candidate. |
| CI base-red diagnosis | quarantine/rebase/owner routing | Authoritative | Avoids blind retries against a known broken base. |
| Resource-pressure receipt | retention/tree-doctor action | Authoritative | Routes measured pressure to bounded cleanup rather than broad deletion. |
| Repeated test-failure signature | reproducer/flake triage | Heuristic until reproduced | Debounces noise before spending investigation capacity. |
| Native engine mismatch/fallback receipt | native-inference invariant audit | Authoritative | Detects evidence that ran outside the required fak-native path. |
| Trigger outcome receipt | cadence/value tuning | Authoritative outcome | Measures yield and regret so negative-net triggers can be suppressed. |

**Authoritative triggers** come from typed failures, measured regressions, schema mismatches, or capacity transitions and may request a bounded read-only consumer immediately. **Heuristic triggers** infer patterns such as recurring intent or suspected flakes; they require debounce, a minimum sample, and proposal-only behavior until independently witnessed.

The example complements the shipped guard-failure path and generic `fak superloop trigger` receipt. It does not implement live guard resume, canonical loop registration, durable outcome ledgers, fleet status, or shared receipt hardening; those remain separate issue scopes.
