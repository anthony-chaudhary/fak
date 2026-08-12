# fak capability map: spend fewer tokens and turns

fak's current product focus is **agent efficiency**: keep stable prompt work reusable, avoid
unnecessary model round trips, route each call to an appropriate model, and let operators
control long-running sessions without injecting another conversational turn. The security
floor remains shipped and indexed, but it is a supporting property of the same tool-call
boundary rather than the lead story here.

This page is the short, outcome-first index. For the complete research and design inventory,
use the [innovations index](INNOVATIONS-INDEX.md); for the full documentation catalog, use
[docs/INDEX.md](INDEX.md).

## Start with the largest avoidable cost: extra turns

A model turn costs more than its visible answer. It can resend a long resident prefix, incur
provider latency, and create yet more context for every later turn. fak therefore measures and
removes **turn tax**, not only individual token strings.

| Capability | What it changes | Try or inspect it |
|---|---|---|
| Turn-tax meter | Attributes avoidable round trips to forced policy retry, deterministic tool work, duplicate/replay work, and other turn kinds. | `go run ./cmd/turntaxdemo -selfcheck`; [turn-tax measurement](../internal/turntaxmeter/) |
| Fused deterministic work | Completes kernel-known work at the tool boundary instead of paying for another model turn. | [fused-turn architecture](../internal/fusedturn/) |
| Live turn control | Budgets turns/tokens/context and pauses, resumes, throttles, steers, or stops a served session out of band—without spending a prompt turn to ask the model to control itself. | `fak help session`; [operator control plane](operator-control-plane.md) |
| Same-trace ablation | Replays one frozen trace with cache levers on/off so savings are attributable rather than anecdotal. | `fak help ablate`; `fak ablate --help` |

The browserless turn-tax demo is the smallest captured spine: its self-check drives the real
`turnbench` path and includes an anti-inflation control where a happy path must report zero
saved turns. The shipped fixture currently witnesses **9 avoided turns** (5 forced-turn + 4
elision) in its synthetic airline scenario; that is a reproducible fixture result, not a
universal workload claim.

## Reuse prompt work instead of repaying for it

| Capability | What it changes | Try or inspect it |
|---|---|---|
| Stable-prefix reuse (vDSO) | Keeps shared setup/provider-cache-compatible prefixes reusable across turns. | [vDSO cache quickstart](explainers/cache.md) |
| Managed context (ctxmmu) | Sheds stale turns while preserving a stable prefix and the information needed to continue. | [managed context](managed-context-continuous-usage.md) |
| Resume pricing | Prices full replay versus cut/reset after cache TTL expiry instead of blindly replaying a large session. | `fak resume plan --resident-tokens 250000 --idle-seconds 7200 --json` |
| Portable session state | Makes a long-running session inspectable and recoverable rather than forcing a restart from raw transcript. | [session image](notes/PORTABLE-SESSION-IMAGE-AND-SNAPSHOT-2026-06-24.md); `fak help resume` |
| Cache-value accounting | Reports reused tokens, effective cost, and attributable savings in operator terms. | `fak info --once`; [cache-value rollup](cache-value-rollup.md) |

## Spend expensive inference only where it helps

| Capability | What it changes | Try or inspect it |
|---|---|---|
| Per-call model routing | Routes calls by task/turn requirements rather than pinning a whole session to one expensive model. | [model routing](model-routing.md); `fak model --help` |
| Model ladder and acceptance | Separates candidate selection from measured acceptance so a cheaper route is not called a win merely because it is cheaper. | [model operations](model-production-readiness-inventory.md) |
| Token/cache observability | Makes savings visible during operation rather than leaving optimization buried in internal counters. | `fak info --once`; `fak ps`; `fak help ablate` |

## Supporting floor: keep optimization bounded

The same checkpoint also enforces a default-deny capability floor and records decisions. That
matters because a faster or cheaper agent that can silently widen its authority is not a
net-true improvement. Security details remain fully discoverable in the
[security and policy guide](fak/security.md), but new readers should begin with the efficiency paths above.

## Where to go next

- **Evaluate token efficiency:** [awesome-token-efficiency](awesome-token-efficiency.md) maps
  the broader field and defines honest measurement boundaries.
- **Understand cache economics:** [cache frontier operating plan](CACHE-FRONTIER-OPERATING-PLAN.md)
  and [cache-value rollup](cache-value-rollup.md).
- **Build an integration:** [integration guides](integrations/) and `fak help serve`.
- **Browse everything shipped or proposed:** [innovations index](INNOVATIONS-INDEX.md), which
  retains maturity labels so an idea is not mistaken for a product claim.

