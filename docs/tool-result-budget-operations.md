# Operate tool-result budgets safely

The normative contract is [`tool-result-budget-policy-v1.md`](decisions/tool-result-budget-policy-v1.md). Pin the policy artifact by immutable digest in the harness or deployment receipt; a mutable path alone is not evidence of which policy ran.

## Roll out: off → observe → enforce

1. **Off:** preserve current delivery while capturing an unbudgeted replay baseline.
2. **Observe:** evaluate the pinned contract and emit decisions/metrics without changing delivered bytes. Compare requested, effective, delivered, and continuation use against the baseline.
3. **Enforce:** enable only after replay shows bounded quality and net cost. Keep the same digest and corpus so the comparison is attributable.

Rollback means returning to `observe` or `off` with a receipt naming the prior digest and reason; do not silently edit the active artifact.

## Author the contract

Declare `kind: fak/tool-result-budget`, semantic `version`, stable `name`, `mode`, one `default` budget, and the narrowest required per-tool `rules`. Every result has one initial byte/item cap and one continuation cap. Exempt only outputs whose complete atomic delivery is correctness-critical; record the tool, reason, owner, and expiry. An exemption is policy, not prompt prose.

Branches and private payloads stay private: receipts carry digest, counts, decision, and continuation identity—not hidden branch content or truncated bytes. A continuation token must bind to the original result identity, policy digest, cursor, and remaining budget; reject replay under a different artifact or result.

## Replay and drift

Run the deterministic spine:

```bash
go run ./cmd/resultbudgetdemo --selfcheck --pretty
```

For rollout replay, freeze representative tool-result fixtures, run the same corpus in off/observe/enforce, and archive each JSON receipt. Treat artifact-digest changes, corpus changes, new exemptions, continuation failures, or material requested-minus-effective shifts as drift requiring review. Requested-minus-effective is an estimate, not a net-true saving: include continuation calls, latency, failures, and task quality.

## Metrics and alerts

Track decisions by policy digest and tool: requested bytes/items, effective cap, delivered amount, truncations, continuation requests/success/failure, exemptions, task success, latency, and total provider/tool cost. Alert on rising continuation failure, repeated truncation of one tool, exemption growth, quality regression, or cost increasing after enforcement.

## Safety boundary

Prompt skills may ask an agent to summarize or request less output. They are advisory and can be ignored. Hard policy runs at the result-delivery seam, deterministically limits bytes/items, issues bounded continuations, and records the decision. Never claim prompt guidance as enforcement.
