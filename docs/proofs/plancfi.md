# D6 · plancfi

`plancfi` is **control-flow integrity for an agent's PLAN**: a stateful adjudicator that treats an agent's *sequence of tool calls* as its control flow and the operator-approved *plan* as its call graph. Binary CFI pins indirect jumps to a precomputed target set and traps a ROP/JOP gadget; `plancfi` pins each tool call to an approved plan transition and traps an *unplanned gadget* — e.g. a prompt-injection–induced `send_email` exfil call that the booking plan never approved. A plan is declared per `TraceID` out-of-band (`Ledger.Declare`); the adjudicator enforces it in-band. With no plan declared, CFI is inactive and `Defer`s (opt-in per session). A conforming call also `Defer`s (no objection — other gates decide). A deviating call returns `RequireApproval` (escalate to a human) by default, or `Deny` in strict mode.

For this module, **"correct" is regime D — decision-procedure soundness**: (1) the gate must be *sound* — it may raise no objection (Defer) only for a call that is an approved transition from the current plan state, so an off-plan gadget can never slip through as conforming; and (2) the underlying plan automaton must be *deterministic* — the verdict and next state are a pure function of (plan, mode, current position, tool), with no nondeterminism and a well-defined (monotone) state advance. Both are witnessed below by deterministic stdlib tests run natively on this macOS node (go1.26 darwin/arm64).

---

## THEOREM 1 — soundness: a Defer is admitted only for an approved transition; off-plan gadgets are trapped

**THEOREM.** For every `ToolCall c` whose `c.TraceID` has a declared plan, `Adjudicate` returns `VerdictDefer` (raises no objection) **only if** `c.Tool` is an approved transition from the current plan state — membership in the `AllowedSet`, or an allowed `Sequence` step (the next step, the current, or any prior). Any other tool (an off-plan gadget) yields `OnDeviation` (`RequireApproval` by default, or `Deny` in strict mode) carrying `Reason=TrustViolation`. CFI is opt-in: a `nil` call or an undeclared trace always `Defer`s and never constrains an unplanned flow.

**REGIME.** D — decision-procedure soundness (never admit what the plan forbids; fail closed).

**PROOF.** The only no-objection (`Defer`) exits in `Adjudicate` are at `fak/internal/plancfi/plancfi.go:154` (nil call **or** `!Declared` — CFI inactive) and `fak/internal/plancfi/plancfi.go:157` (`conforms()==true`). Every tool that reaches `fak/internal/plancfi/plancfi.go:161` returns `a.OnDeviation` with `Reason=abi.ReasonTrustViolation`, a deviation claim, and `Meta`. `conforms()` (`plancfi.go:113`) returns `true` **only** when: the trace has no plan (`plancfi.go:117` — unreachable from `Adjudicate`, already gated by `Declared` at `:153`); or `AllowedSet` membership via `Plan.has()` (`plancfi.go:120` → linear scan `plancfi.go:65`); or, in `Sequence` mode, `c.Tool == Tools[i]` for some `i ∈ [0, pos+1]` (`plancfi.go:125`). Every other tool falls to `return false` (`plancfi.go:133`) and is routed to the deviation branch. So an off-plan gadget cannot obtain a `Defer` — soundness holds. `OnDeviation` defaults to `VerdictRequireApproval` (`plancfi.go:147`), registered `FallbackDeny` (`plancfi.go:176`), so an unaware downstream fold fails closed.

**WITNESS.**
```
(go test ./internal/plancfi/ -count=1 -timeout 120s \
  -run 'TestDeviationEscalates|TestStrictModeDenies|TestSessionIsolation|TestSequenceMode|TestConformingCallDefers|TestNoPlanDefers' -v)
```
`TestDeviationEscalates` — `send_email` (not in the airline plan) → `VerdictRequireApproval`, `Meta[plancfi]=deviation`, `Meta[tool]=send_email`, non-empty `WitnessPayload.Claim`. `TestStrictModeDenies` — `OnDeviation=Deny` ⇒ `delete_everything` → `VerdictDeny`. `TestSessionIsolation` — `send_email` Defers on the unplanned trace but escalates on the planned trace. `TestSequenceMode`'s `dev("z")` — an unlisted tool is non-Defer. `TestConformingCallDefers` / `TestNoPlanDefers` pin the admit side: only approved tools (and the no-plan / nil case) Defer.

**VERDICT.** **PROVEN** — 2026-06-20, six tests green (`ok github.com/anthony-chaudhary/fak/internal/plancfi 0.263s`), run natively on the macOS node (go1.26 darwin/arm64).

**DOS.** bound at ship.

---

## THEOREM 2 — the plan automaton is deterministic

**THEOREM.** The plan automaton is deterministic: the verdict and the next state (the `Sequence` `pos`) are a pure function of (declared plan, mode, current `pos`, `c.Tool`). Identical inputs produce the identical transition on every run — no RNG, wall-clock, or network is consulted — and `Sequence` progress advances **monotonically** (`pos` never regresses on a re-read or a prior step).

**REGIME.** D — decision-procedure soundness (the composition/state order is well-defined).

**PROOF.** `conforms()` (`fak/internal/plancfi/plancfi.go:113`) reads only `st.plan`, `st.plan.Mode`, `st.pos`, and the `tool` argument; it invokes no `rand`, no `time`, no I/O. `AllowedSet` is a deterministic linear membership scan (`plancfi.go:65`). `Sequence` is a bounded deterministic loop over `i ∈ [0, min(pos+1, len−1)]` (`plancfi.go:125`), and the sole state mutation is the monotone advance `if i > st.pos { st.pos = i }` (`plancfi.go:127`) — `pos` can only increase, so re-reading a prior step is a deterministic no-op on state. The whole transition is serialized under `l.mu` (`plancfi.go:114`), so even concurrent callers observe a well-defined per-trace state sequence. `Adjudicate` (`plancfi.go:152`) adds only deterministic nil/`Declared` guards. Hence same (plan, mode, pos, tool) ⇒ same (verdict, pos′).

**WITNESS.**
```
(go test ./internal/plancfi/ -count=1 -timeout 120s -run 'TestSequenceMode' -v)
```
`TestSequenceMode` drives a fixed call sequence on plan `{a,b,c}` / `Sequence` and asserts each concrete transition: `ok(a)`=step0, `ok(b)`=step1, `ok(a)`=a prior step still Defers (pos stays at 1 — no regression), `dev(z)`=unlisted deviates, `ok(c)`=next step Defers. The exact, repeatable verdict at every step is the determinism witness; `-count=1` reproduces it bit-for-bit.

**VERDICT.** **PROVEN** — 2026-06-20, `TestSequenceMode` green (`ok github.com/anthony-chaudhary/fak/internal/plancfi 0.184s`), run natively on the macOS node. Residual: no test *re-runs* an identical call stream twice asserting verdict-equality; a `testing/quick` property over (plan, tool-stream) replayed twice would strengthen this from a structural+single-trace witness to a property witness. The transition function being pure and side-effect-free over its named inputs makes that an upgrade, not a gap that changes the verdict.

**DOS.** bound at ship.

---

### Mechanism map

| Property | file:line |
|---|---|
| Adjudicate entry / nil+Declared guard | `fak/internal/plancfi/plancfi.go:152`, `:154` |
| Defer on conform | `fak/internal/plancfi/plancfi.go:156` |
| Deviation → OnDeviation + TrustViolation | `fak/internal/plancfi/plancfi.go:161` |
| conforms() (membership + sequence) | `fak/internal/plancfi/plancfi.go:113` |
| AllowedSet linear scan | `fak/internal/plancfi/plancfi.go:65` |
| Sequence bounded loop | `fak/internal/plancfi/plancfi.go:125` |
| Monotone pos advance | `fak/internal/plancfi/plancfi.go:127` |
| RequireApproval registered FallbackDeny | `fak/internal/plancfi/plancfi.go:176` |

Implementing commit: `5415366` (*"fak/plancfi+harvest: plan-CFI adjudicator, RequireApproval verdict, LabelRow harvest"*).
