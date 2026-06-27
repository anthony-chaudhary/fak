---
title: "The Verification Ladder: epic roadmap (smallest-sufficient-rung adjudication)"
description: "Eight file-grounded epics that build the lazy-escalation half of fak's verification doctrine — a first-class INDETERMINATE verdict + lazy fold, per-claim risk-to-rung selection, complain->shadow->enforce promotion, result-side ladder parity, smallest-sufficient shipgate evidence, result-side ShareScope confinement, rung-decision telemetry, and a declarative throttle rung."
---

# The Verification Ladder — epic roadmap

> _Snapshot against `HEAD = 936e994` (2026-06-24). Companion to the doctrine [verification-ladder-doctrine.md](verification-ladder-doctrine.md). Each epic below was drafted from a code inventory and then **adversarially citation-checked** — a skeptic opened every cited file and refuted any "not-built" claim that was actually built. The four epics that had `file:line` drift (mostly `internal/abi/registry.go` `FoldRank`, which is at line 865, not the ~841 the draft guessed) were corrected before landing. Citations are point-in-time; chase the symbol, not the line._

## How to read this

Restrictiveness is already solved — fak's adjudicator is a most-restrictive-wins, cost-ordered, fail-closed reference monitor (epic #492, closed). These epics add the *flexible, granular, smallest-sufficient* half: stop at the cheapest rung that conclusively decides, let a cheap rung say "I cannot decide — climb," pick the rung by risk, and make which-rung-decided observable so an operator can tune the ladder the way `dos enforce-tune` tunes policy knobs.

**Sequencing.** Epic 1 is the keystone: it adds `VerdictIndeterminate` + the lazy chain fold the rest of the ladder leans on. Epics 2 (risk-class rung selection) and 4 (result-side parity) build directly on the fold semantics; Epics 3, 5, 6, 7, 8 are independently shippable in any order. None re-gates the non-forgeable keep-bit through `dos improve` (a known live trap) and none duplicates the in-flight model-routing spine (#595/#603) or its residency *call* gate.

**Index of epics**

1. feat(kernel): VerdictIndeterminate + lazy chain fold — short-circuit on the first conclusive rung, escalate only on a residual abstain
2. feat(adjudicator): per-claim risk-class RungProfile — restrict-only, data-driven rung selection (default the smallest sufficient rung)
3. feat(adjudicator): per-tool complain → shadow → enforce promotion ledger (AppArmor/eBPF analogue)
4. feat(kernel): result-side ladder parity — honor Deny/RequireWitness on admitResult
5. feat(shipgate): smallest-sufficient-evidence keep-bit (graduated EvidenceProfile, default all-three)
6. feat(engine): result-side ShareScope ceiling enforcement — confine a result to its declared scope on the share/readmit path
7. feat(observability): rung-decision telemetry — make the adjudication ladder OBSERVABLE (a labeled /metrics counter over the verdict stream)
8. feat(adjudicator): policy-manifest rate_limit + retry-after on WAIT (declarative throttle rung)

---

## Epic 1

# feat(kernel): VerdictIndeterminate + lazy chain fold

## Problem

The doctrine is "verify at the smallest rung that can establish the property; escalate only when that rung cannot." The kernel has the lattice that makes a fold order-independent, but it does **not** have the two halves that make it *lazy*:

1. **The chain fold never short-circuits.** `Fold` walks every adjudicator in the chain even after a rank-100 `Deny` is already maximal-among-all-possible-verdicts. A deny decided by a cheap name-map rung (rung 1) still pays for every costlier downstream rung in the same `Submit`. The per-*call* short-circuit exists inside one `Adjudicate` (each sub-rung early-returns at `internal/adjudicator/decide.go:262-374`), but the per-*chain* one does not (`internal/kernel/kernel.go:185-194`).

2. **There is no "I could not decide cheaply" signal.** The only abstention is `VerdictDefer` ("I have no opinion"), which folds to `DEFAULT_DENY` when nothing allows (`internal/kernel/kernel.go:195-197`). A cheap heuristic rung that is *uncertain* — not "no opinion", but "a costlier rung must look before we commit" — has no verdict to emit. It must either falsely deny (fail closed, over-restrictive) or falsely defer (fail open to the next link, which may allow). Neither preserves the property "do not commit on a guess."

These are the same gap from two directions: the fold can't stop early because it can't tell a *conclusive* verdict from a *provisional* one, and a cheap rung can't say "stop guessing, climb."

## Current state (read at HEAD)

- **The chain fold runs the whole chain, no early break.** `Fold` loops `for _, a := range chain`, skips `Defer`, and keeps the max-`FoldRank` verdict, with no break once a rank-100 `Deny` is seen — worst case is always O(full chain): `internal/kernel/kernel.go:178-199`.
- **The closed verdict kinds, in iota order:** `Allow, Deny, Transform, Quarantine, RequireWitness, Defer`, then `6..1023 reserved for additive CORE kinds`, then `VerdictReservedMax = 1023`: `internal/abi/types.go:207-216`.
- **The restrictiveness lattice is a constant switch:** `Allow=0, Defer=1, Transform=2, Quarantine=3, RequireWitness=4, Deny=100`; a registered kind falls back to its snapshot `foldRank`, else `100` (fail-closed): `internal/abi/registry.go:861-884` (switch body `865-879`).
- **`FoldExplain` mirrors `Fold` byte-identically** and adds the per-rung trace; it makes the same strict-greater max-rank, ties-to-first selection and the same empty/all-defer synthesis: `internal/kernel/explain.go:70-127`. Its returned verdict is asserted "byte-identical to `Fold`" (doc `explain.go:70-73`).
- **`Submit`'s verdict switch special-cases only the 5 closed kinds**; the `default` branch holds any other kind as a fail-closed deny-as-value: `internal/kernel/kernel.go:319-363`.
- **The result-side fold also never short-circuits** (`internal/kernel/kernel.go:252-259`) and acts only on `Quarantine`/`Transform`, defaulting everything else to admitted (`kernel.go:260-282`).
- **`Disposition` maps a closed reason set to RETRYABLE/WAIT/ESCALATE/TERMINAL**; default is TERMINAL: `internal/kernel/kernel.go:478-489`. Reason vocabulary is closed at `internal/abi/reasons.go:10-27`.
- **The per-call rung order inside `Adjudicate` is fixed in code**, each rung early-returning: `internal/adjudicator/decide.go:255-374`; `defaultDeny` resolves the fail-closed tail: `decide.go:377-389`.

## Design — the graduated rungs

The fold defaults to the **smallest rung that can conclusively decide**, and climbs only on a residual abstain.

**Rung A (DEFAULT — smallest sufficient): conclusive at the first rung, short-circuit.**
`Fold` gains an early-exit: it stops the moment the running `best` verdict's `FoldRank` is provably **maximal-among-all-possible-verdicts**. Because `Deny` is pinned at `100` and `FoldRank` is bounded, a `Deny` (or any kind whose rank equals the ceiling) cannot be raised by any remaining rung, so the remaining rungs are pure wasted work. This changes the **cost** (O(first-conclusive) on the deny path) without changing the **verdict** (the lattice fold is order-independent; a maximal verdict is the same whether or not later rungs run). This is the doctrine's "smallest rung": a call a name/path/arg rung conclusively denies (`internal/adjudicator/decide.go:262-331`) never pays the rest of the chain.

**Rung B (escalation, only on a residual abstain): `VerdictIndeterminate`.**
Add `VerdictIndeterminate` as an **additive core kind** in the reserved `6..1023` band (`internal/abi/types.go:214`), with a `FoldRank` between `Defer (1)` and `Transform (2)` (`internal/abi/registry.go:866-879`). Semantics, distinct from both `Defer` and a fail-closed `Deny`:

- A cheap rung returns `Indeterminate` to mean **"I could not conclusively decide; a costlier rung MUST be consulted before commit."**
- In the fold, `Indeterminate` is **not** skipped like `Defer` (it counts as `sawNonDefer`), but it is **not committable**: if the resolved `best` is `Indeterminate`, the fold has not established a verdict. The kernel **climbs** — re-resolves by consulting the next-costlier rung(s) rather than committing the residual.
- **The explicit escalation trigger:** a residual `Indeterminate` after the in-process chain has folded. It resolves to whichever costlier rung produces a conclusive verdict; if no costlier rung is available (the chain is exhausted and the residual is still `Indeterminate`), it **fails closed to `Deny`** — never fails open. An `Indeterminate` ranked below `Transform`/`Deny` means a conclusive `Deny` or `Transform` from any rung still wins outright (the fold never gets stuck at `Indeterminate` when something conclusive exists).

This keeps the v0.1 behavior byte-identical for any chain that emits no `Indeterminate` (the closed set is unchanged; a worker that never produces the new kind sees the old fold), and it makes "climb" expressible without falsely denying or falsely deferring.

**Rung C (observability, not a behavior change): surface which rung decided.**
`FoldExplain` already computes the winning rung and rank (`internal/kernel/explain.go:94-122`); it must mirror the rung-A short-circuit **byte-identically** (same returned verdict; the trace simply records that fold stopped early). No `/metrics` work in this epic.

## DOS + Linux analogue

- **DOS:** `Indeterminate` is the kernel-side analogue of `dos verify` returning INDETERMINATE rather than VERIFIED/REFUTED — "the cheap check abstained, bind the costlier evidence before you trust it," not a self-reported pass.
- **Linux:** the short-circuit is `iptables`/LSM-style first-match-wins on an ordered chain — once a terminal rule (a `Deny`) matches, the rest of the chain is not evaluated; `Indeterminate` is the `MODULE_DEFER`/`-EAGAIN` "ask a more authoritative layer" return, distinct from `-EPERM` (deny) and "no match" (defer).

## Acceptance criteria

1. `Fold` returns a **byte-identical verdict** to the current implementation on every chain that emits no `Indeterminate` (a property test over generated chains asserts old-vs-new equality).
2. On a chain where a rung at index `i` returns a maximal-rank verdict (`Deny`), no adjudicator at index `> i` is called (a counting/spy adjudicator proves the early-exit).
3. `FoldExplain` returns a verdict byte-identical to `Fold` **with the short-circuit applied**, and its `Rungs` trace contains exactly the rungs that ran (not the skipped tail).
4. A chain whose only non-defer verdict is `Indeterminate`, with **no** costlier rung available, folds to `Deny` (fail-closed — never `Allow`, never the bare `Indeterminate`).
5. A chain with an `Indeterminate` from a cheap rung **and** a conclusive `Allow`/`Deny`/`Transform` from another rung resolves to the conclusive verdict by lattice rank (the `Indeterminate` never wins over a conclusive kind).
6. `FoldRank(VerdictIndeterminate)` sits strictly between `FoldRank(VerdictDefer)` and `FoldRank(VerdictTransform)`; `Submit` does not dispatch an `Indeterminate` (it is held, never enqueued).
7. `make ci` / `scripts/ci.ps1` green; no new `os/exec` on the decide path (the architest `TestHotPathHasNoExec` / interpreter-free invariants stay green).

## Child issues (each independently shippable on the trunk)

1. **`feat(abi): add VerdictIndeterminate kind + FoldRank position`** — add the iota kind in the reserved core band (`internal/abi/types.go:214`) and its rank between `Defer` and `Transform` in the `FoldRank` switch (`internal/abi/registry.go:866-879`, inserting between the `VerdictDefer` and `VerdictTransform` cases at lines 870-871). Pure additive; no fold-behavior change yet. **Test:** `TestFoldRankIndeterminateBetweenDeferAndTransform` in `internal/abi/registry_test.go`.

2. **`feat(kernel): short-circuit Fold once the best verdict is maximal`** — add the early-exit to `Fold` (`internal/kernel/kernel.go:185-194`) keyed on the ceiling rank (`Deny`=100). Verdict-preserving. **Test:** `TestFoldShortCircuitsOnMaximalDeny` (spy adjudicator counts calls) + `TestFoldVerdictUnchangedUnderShortCircuit` (property: old-vs-new equal over generated chains) in `internal/kernel/kernel_test.go`.

3. **`feat(kernel): FoldExplain mirrors the short-circuit byte-identically`** — apply the same early-exit in `FoldExplain` (`internal/kernel/explain.go:94-122`) so the returned verdict matches `Fold` and the trace records only the rungs that ran. **Test:** `TestFoldExplainMatchesFoldUnderShortCircuit` in `internal/kernel/explain_test.go`.

4. **`feat(kernel): resolve a residual Indeterminate by climbing, else fail closed`** — make `Fold` (and `FoldExplain`) treat `Indeterminate` as committed-only-if-conclusive: a residual `Indeterminate` with nothing costlier folds to `Deny`/`DEFAULT_DENY`; a conclusive kind always wins. **Test:** `TestIndeterminateResidualFailsClosed` + `TestConclusiveBeatsIndeterminate` in `internal/kernel/kernel_test.go`.

5. **`feat(kernel): hold (never dispatch) an Indeterminate in Submit`** — confirm the `Submit` switch (`internal/kernel/kernel.go:319-363`) holds an `Indeterminate` via the fail-closed `default` branch (no enqueue, surfaced as deny-as-value). **Test:** `TestSubmitHoldsIndeterminate` in `internal/kernel/kernel_test.go`.

## Honesty boundary

- **Stays heuristic:** *which* rung emits `Indeterminate` and when. This epic adds the verdict and the lazy fold; it does **not** make any existing rung start returning it. The adjudicator's per-call rung order remains hard-coded (`internal/adjudicator/decide.go:255-374`) and data-driven/risk-based rung selection is **out of scope** — no rung is converted to emit `Indeterminate` here.
- **Out of scope (named, not built):**
  - Per-claim risk-based rung selection / reorderable rungs.
  - The result-side fold's missing top rungs and its lack of short-circuit (`internal/kernel/kernel.go:252-282`).
  - A WAIT/`EVIDENCE_PENDING` outcome for the witness gate; `Indeterminate` here is a *fold* signal, not a retry-after.
  - A `/metrics` rung-decision distribution (the trace is computed in `FoldExplain` but stays off the hot path; surfacing it is a separate observability epic).
  - Disposition routing for `Indeterminate` (the `Disposition` switch at `internal/kernel/kernel.go:478-489` is untouched; a held `Indeterminate` surfaces as a deny-as-value via the existing default branch).
- **The short-circuit is provably free of verdict change** because the lattice fold is order-independent and `Deny` is the rank ceiling — this is the load-bearing structural claim the tests in child issue #2 must pin, not narrate.

---

## Epic 2

# Epic: per-claim risk-class RungProfile (default the smallest sufficient rung)

## Problem

The v0.1 reference monitor runs the SAME fixed sub-rung sequence on every call that reaches it. A `read_*` default-deny pays the decode-args, three SELF_MODIFY rungs, the arg-predicate scan, and (if opted in) the lint rung before it ever lands on `defaultDeny`, exactly as a whole-file write into the kernel spine does. The order is hard-coded in `Adjudicate` (`internal/adjudicator/decide.go:255-374`) and the only sub-rung an operator can turn off is `LintWrites` (the opt-in bool at `decide.go:71`).

This makes the doctrine's "default to the smallest sufficient rung, escalate only on high-risk" a **fixed sequence**, not a **per-claim policy**. The seccomp-BPF / Rosetta data-program model the package doc gestures at (`decide.go:16-18`: "cheaper pre-flight rungs run first; order does not change the verdict, only the work done") is realized only at the CHAIN level via tool-scoped pruning, never WITHIN one `Adjudicate`. You cannot say "a low-risk read needs only the name rung" vs "a write into a guarded tree needs the arg-predicate + lint rungs"; the rungs are not data-driven, not reorderable, and not droppable per risk class.

The fix must be **restrict-only by construction**: a profile may DROP a rung for a low-risk class (skip work the class cannot benefit from) but may NEVER skip a refusal rung for a write / self-modify / ship class — those always run. The floor can only narrow, never widen.

## Current state (read at HEAD)

- The fixed sub-rung sequence lives in `Adjudicate`: explicit `Deny` map (`internal/adjudicator/decide.go:262`), decode-args-once (`decide.go:267`), file-write SELF_MODIFY (`decide.go:272-281`), shell SELF_MODIFY (`decide.go:292-299`), synth-tool SELF_MODIFY (`decide.go:309-316`), authored-script ledger note (`decide.go:320`), arg-predicates (`decide.go:327-331`), opt-in lint (`decide.go:342-351`), redact Transform (`decide.go:354-361`), affirmative allow (`decide.go:364-371`), `defaultDeny` (`decide.go:374`). Each refusal rung early-returns — this IS a genuine single-call short-circuit.
- The write-class predicate `writeShaped` is a fixed verb-substring list (`internal/adjudicator/decide.go:212-219`).
- The low-risk-read predicate `lowRiskReadShaped` is a fixed name-prefix list (`internal/adjudicator/decide.go:224-235`), reused only by `defaultDeny` (`decide.go:378`) under `PostureAdmitAndLog`.
- `Policy` is a flat struct with no rung-selection field (`internal/adjudicator/decide.go:36-72`); `LintWrites` (`decide.go:71`) is the only per-rung dial and it is a global bool.
- `Posture` is a two-value enum, global to the policy (`internal/adjudicator/decide.go:76-87`).
- `Adjudicator` holds the policy + the `argByTool` index + the authored ledger (`internal/adjudicator/decide.go:121-139`); `New`/`SetPolicy` rebuild the index (`decide.go:142-152`).
- The chain `Fold` runs the WHOLE chain, no early break on a Deny (`internal/kernel/kernel.go:178-199`); the only chain-level work-saving is tool-scoped pruning, not early-exit.
- vDSO hit returns a blanket Allow before the adjudicator chain (`internal/kernel/kernel.go:300-314`).
- Registration is rank-100 (`internal/adjudicator/decide.go:668-672`).

## Design — graduated rungs, DEFAULT = smallest sufficient

Introduce a `RungProfile` keyed by a coarse **risk class** computed once per call from the tool name + decoded args. The class selects WHICH sub-rungs run and (within the restrict-only invariant) in what order. The profile is a field on `Policy` (`decide.go:36-72`); a nil/zero profile reproduces today's fixed sequence byte-for-byte (drop-in safe).

Risk classes (coarsest-first, computed by generalizing the two existing predicates):
1. **read-shaped** — `lowRiskReadShaped(tool)` true (`decide.go:224-235`) and not write-shaped.
2. **write** — `writeShaped(tool)` true (`decide.go:212-219`) OR a shell/synth path reaches `targetPath` (`decide.go:239-251`).
3. **self-modify-adjacent** — write class whose decoded target lands under a `SelfModifyGlobs` fragment (`decide.go:49`).
4. **ship** — out of scope for this leaf; the ship rung lives in `internal/shipgate` and folds at rank 40, not here. Named only so the class enum is complete.

Per-class rung selection, with the DEFAULT being the smallest set that can still establish the property:

- **read-shaped (DEFAULT = name rung alone):** run the `Deny` map (`decide.go:262`), affirmative allow (`decide.go:364`), and `defaultDeny` (`decide.go:374`). DROP the three SELF_MODIFY rungs (they are write-scoped — `writeShaped`/`commandSelfModify`/`synthToolSelfModify` already return empty for a read), the arg-predicate scan if the tool has no predicates (already O(0) via `argByTool`), and the lint rung. Dropping rungs a read can never trip is **work elision, not floor relaxation**: the dropped rungs are provably no-ops for the class.
- **write (DEFAULT = name + SELF_MODIFY + arg-predicate):** run `Deny`, all three SELF_MODIFY rungs (`decide.go:272-316`), arg-predicates (`decide.go:327`), allow, `defaultDeny`. Lint stays opt-in.
- **self-modify-adjacent (DEFAULT = full ladder, no drops):** every refusal rung runs including opt-in lint forced on. This class can never drop a refusal rung.

**Escalation trigger (explicit):** the profile is queried for a **DROP** decision per rung. The invariant `mustRun(class, rung)` returns true for every refusal rung whenever the class is write / self-modify-adjacent; a profile that tries to drop a `mustRun` rung is rejected at `SetPolicy` time (build the index, validate the profile, fail the policy load — never silently widen). A read-shaped call **escalates to the write ladder** the moment `writeShaped`/`targetPath` resolves a non-empty target — i.e. classification is itself fail-closed: ambiguous → treat as the higher-risk class.

Non-forgeability: the class is computed from the tool name + decoded args (`decide.go:267`), never from model-controlled `Meta` (same discipline as `lowRiskReadShaped`'s comment at `decide.go:222-223`). The profile can only be set via the validated `SetPolicy` path (`decide.go:147-152`).

## DOS + Linux analogue

- **DOS:** the per-claim version of `dos enforce-tune` — instead of one global posture knob, each risk class carries its own enforce/elide ladder, witnessed by which rung actually decided.
- **Linux:** seccomp-BPF — a per-syscall-class filter program decides which checks run; a higher-risk syscall class can only ADD predicates, never drop the mandatory deny path (the `no_new_privs` floor).

## Acceptance criteria (testable)

1. A nil/zero `RungProfile` produces byte-identical verdicts to HEAD for the existing corpus (regression: re-run `TestAdmitAndLogPostureAllowsOnlyReadShapedDefaultDeny`, `TestSelfModifyDeniedWithBoundedWitness`, `TestArgPredicatesAreRestrictOnly` unchanged).
2. A read-shaped default-deny with a profile that drops the SELF_MODIFY + lint rungs returns the SAME verdict as today (the dropped rungs were no-ops) — proven by a differential test: profiled verdict `==` full-ladder verdict for every read-shaped fixture.
3. A profile that attempts to DROP a refusal rung for a write / self-modify-adjacent class is REJECTED at `SetPolicy` (the floor cannot be widened) — a named test asserts the policy load fails.
4. A write into a `SelfModifyGlobs` tree is still denied SELF_MODIFY with the bounded-glob witness under EVERY profile (re-run `TestSelfModifyGuardsWitnessMachinery`, `TestSelfModifyGuardsShellWritePath`).
5. p50 for a read-shaped call under the read profile is `<=` today's p50 (work elision never regresses latency): extend `TestAdjudicateP50UnderOneMillisecond` / `BenchmarkDecide` with a read-profile arm.
6. Which rung decided (or that a rung was elided) is observable in the `FoldExplain` trace.

## Child issues (each independently shippable on the trunk)

1. **Risk-class function + tests.** Add `riskClass(tool string, args map[string]any) class` generalizing `writeShaped` (`decide.go:212`) and `lowRiskReadShaped` (`decide.go:224`); ambiguous → higher class (fail-closed). Test: `TestRiskClassFailsClosedOnAmbiguousTool` in `internal/adjudicator`.
2. **`RungProfile` type + `Policy` field + `SetPolicy` validation.** Add the profile struct and the `mustRun(class, rung)` invariant; reject a profile that drops a mandatory refusal rung. Test: `TestRungProfileCannotDropRefusalRungForWriteClass`.
3. **Wire profile into `Adjudicate` (drop-safe).** Gate each sub-rung on `profile.runs(class, rung)`; nil profile = today's sequence. Test: `TestNilRungProfileIsByteIdenticalToFixedSequence` (differential over the existing fixture corpus).
4. **Read-class default profile + latency arm.** Ship a `DefaultRungProfile()` that elides write-only rungs for the read class; wire it into `DefaultPolicy` only behind an explicit constructor (not the zero policy). Test: extend `BenchmarkDecide` with `BenchmarkDecideReadProfile` + assert read p50 `<=` baseline.
5. **Surface the deciding/elided rung in `FoldExplain`.** Tag each rung trace entry with `elided bool` so the rung-decision distribution is observable. Test: `TestFoldExplainMarksElidedRung` in `internal/kernel`.

## Honesty boundary

- **Stays heuristic:** the risk class is a NAME + arg-shape heuristic (it inherits `writeShaped`'s and `lowRiskReadShaped`'s name-substring/prefix lists at `decide.go:212-235` and `targetPath`'s key list at `decide.go:243`). It is not a semantic analysis of what the tool does. A write tool named without a verb (e.g. `apply_change`) classifies read-shaped unless it routes through shell — exactly today's `writeShaped` gap, NOT widened by this change. Classification is fail-closed (ambiguous → higher class), so the heuristic can only over-protect.
- **Stays unbuilt / out of scope:** (a) the **chain-level** short-circuit — `Fold` still runs every adjudicator (`kernel.go:185-194`); this leaf is per-call only. (b) An INDETERMINATE / "climb" verdict — the abstention set is still `VerdictDefer` folding to `DEFAULT_DENY`. (c) The **ship** risk class is named for completeness but its rung lives in `internal/shipgate` (rank 40), not adjudicated here. (d) Per-tenant / per-tool rate-class profiles — this keys on a coarse risk class, not on tenancy. (e) The vDSO blanket-Allow bypass (`kernel.go:300-314`) is unchanged: a profile cannot re-check a fast-path hit.

---

## Epic 3

# feat(adjudicator): per-tool complain → shadow → enforce promotion ledger

## Problem

fak's "complain mode" is a single global binary that only relaxes the read-prefix set. `Posture` is one `uint8` on the whole `Policy` (`internal/adjudicator/decide.go:36-39`, `decide.go:74-87`), and the only relaxation it buys is at `defaultDeny`: a `PostureAdmitAndLog` policy downgrades a `DEFAULT_DENY` to `Allow` **only** when the tool name passes `lowRiskReadShaped` — a hard-coded prefix list `read_/get_/search_/list_/lookup_/find_/calc` plus the literal `calculate` (`decide.go:221-235`, `decide.go:377-389`). There is no way to say "admit-and-log for tool `X` but fail-closed for `Y`," and the `would_deny` forensic record is only ever attached on that one read-shaped path — every other rung's deny (a `SELF_MODIFY` glob hit, an `ArgPredicate` violation, a write-shaped `DEFAULT_DENY`) just fails closed with no record of what a complain-mode operator would have learned.

AppArmor's actual value is not "complain mode exists" — it is **per-profile** complain plus a learning toolchain (`aa-logprof`) that promotes a profile to enforce *once the audit log is clean*. fak already has the symmetric machinery pointing the **other** direction: `rulesynth` mines the live admit stream for near-misses to GROW the deny floor (`internal/rulesynth/stream.go:91-102`, `internal/rulesynth/rulesynth.go:126-147`), and gates every synthesized rule through a non-forgeable keep-bit (`rulesynth.go:204-263`). What is missing is the inward dial: a **per-tool complain → enforce** posture and a **witnessed promotion ledger** that counts clean complain-mode events per tool, so an operator promotes a tool from complain to enforce on accumulated evidence, not a manual flip.

This epic adds that dial at the smallest sufficient rung: a probed tool runs in complain mode (admit-and-log, rung 2) until the ledger proves it safe to enforce.

## Current state (every claim read at HEAD)

- `Posture` is a single per-policy `uint8`; the zero value is fail-closed. `internal/adjudicator/decide.go:36-39` (the `Posture Posture` field on `Policy`), `decide.go:74-87` (the `PostureFailClosed`/`PostureAdmitAndLog` enum).
- Complain mode relaxes **only** read-shaped names. `decide.go:221-235` (`lowRiskReadShaped` — the fixed prefix list `read_/get_/search_/list_/lookup_/find_/calc` and the literal `calculate`, gated `false` for anything `writeShaped`).
- The `would_deny` record is attached **only** at `defaultDeny`, and only for `lowRiskReadShaped` under `PostureAdmitAndLog`. `decide.go:377-389` — the `Meta{posture:admit_and_log, would_deny:DEFAULT_DENY}` is built nowhere else. Every other rung returns a bare `VerdictDeny` with no forensic `Meta`: explicit `Deny` map (`decide.go:262-264`), file-write `SELF_MODIFY` (`decide.go:272-281`), shell `SELF_MODIFY` (`decide.go:292-299`), synth-tool `SELF_MODIFY` (`decide.go:309-316`), `ArgPredicate` violation (`decide.go:327-331`), `LintWrites` MALFORMED (`decide.go:342-351`).
- `Adjudicate` is the rank-100 fold and its rung order is fixed in code (`decide.go:255-374`); each rung early-returns, so a single call short-circuits, but the rung set is not data-driven.
- The chain-level fold does NOT short-circuit: `internal/kernel/kernel.go:178-199` (`Fold`) scans every adjudicator with no early break, keeping the max `FoldRank`.
- `rulesynth` already rides the live stream to mine near-misses: `Harvester.Emit` keys on `EvDecide` + `VerdictAllow` only (`internal/rulesynth/stream.go:91-102`); `NearMissCorpus` is the thread-safe append log (`stream.go:28-57`); `Detect` is the near-miss predicate against the real floor (`rulesynth.go:126-147`).
- `Validate` is the honesty gate that folds a synthesized rule into shipgate's non-forgeable keep-bit (`rulesynth.go:204-263`): metric = catches, suite-green = zero benign regression, truth-clean = catches its whole cluster; `shipgate.Evaluate(w)` sets the keep-bit, `ev.Kept()` reads it back.

## Design — graduated rungs (DEFAULT = smallest sufficient)

The dial is per-tool posture with three states, defaulting to the smallest rung that still records evidence:

- **Rung 2 — complain / admit-and-log (DEFAULT for a probed tool).** A tool placed in the per-tool complain set admits its `DEFAULT_DENY` calls and attaches a `would_deny` forensic record carrying the rung that WOULD have denied (not just `DEFAULT_DENY`). It pays the name rung plus the record — nothing more. This is the smallest rung that establishes "what would this tool have been refused for, and how often, cleanly?"
  - **Escalation trigger (complain → enforce):** the promotion ledger counts clean complain-mode `would_deny` events per tool. Promotion is offered ONLY when the tool's ledger shows N clean events (N operator-set, ledger-default conservative) with ZERO would-deny events whose rung is a hard-refusal class (`SELF_MODIFY`, `MALFORMED`, an `ArgPredicate` violation) — i.e. only `DEFAULT_DENY`-class events are promotable evidence. A would-deny on a hard-refusal rung resets the tool's clean run to zero and bars promotion (it fails closed, exactly as enforce already does).
- **Rung 1 — enforce (the fail-closed floor).** A tool not in the complain set behaves exactly as HEAD does: every provable refusal stands. Promotion moves a tool from the complain set to the enforce set; the ledger records the promotion as a witnessed decision.
- **Honesty boundary on promotion (rung 3a analogue).** Promotion does NOT auto-mutate the policy in-process. Like `rulesynth.ManifestDiff` (`rulesynth.go:265-269`), the ledger emits a reviewable promotion record an operator applies — a clean count is a *recommendation*, never a self-grading flip of the floor.

Default behavior is unchanged for any tool not named in the complain set: the global `Posture` zero value stays `PostureFailClosed`, and the per-tool complain set is empty by default, so a zero `Policy` is byte-for-byte the HEAD floor.

### DOS + Linux analogue

- **DOS analogue:** the same observe → shadow → enforce promotion `dos enforce-tune` runs over policy knobs (keep only measured net gains, escalate repeated non-keeps), applied here per-tool to the complain dial — promotion is a witnessed counter, not a manual flip.
- **Linux analogue:** AppArmor per-profile `complain` mode + `aa-logprof` promoting a profile to `enforce` once its audit log is clean; the per-rung `would_deny` record is the audit-log line.

## Acceptance criteria (testable)

1. A `Policy` can name a per-tool complain set; a tool in it admits-and-logs its `DEFAULT_DENY` calls while a tool NOT in it (and not read-shaped) still fails closed — both proven in one table test.
2. The `would_deny` record names the rung that WOULD have denied, distinguishing a `DEFAULT_DENY`-class event from a hard-refusal-class event; a hard-refusal-class call in complain mode STILL fails closed (no admit).
3. The promotion ledger counts clean complain-mode events per tool; a hard-refusal-class would-deny resets that tool's clean run to zero.
4. Promotion is offered only at N clean events with zero hard-refusal events, and emits a reviewable record (never auto-mutates `Policy` in-process).
5. A zero `Policy` and any policy with an empty complain set produce byte-identical verdicts to HEAD (no regression to the existing fail-closed floor or the existing global `lowRiskReadShaped` admit-and-log path).

## Child issues (each independently shippable on the trunk)

1. **Per-tool complain set on `Policy` + admit-and-log gated by it.** Add a per-tool complain set field to `Policy` (`decide.go:36-39`) and a helper `complainFor(tool)` that `defaultDeny` consults (`decide.go:377-389`) in addition to the existing global `Posture`/`lowRiskReadShaped` path, so a named tool admits-and-logs its `DEFAULT_DENY` even if it is not read-shaped. *Test:* `TestPerToolComplainAdmitsNamedToolDefaultDeny` in `internal/adjudicator` (a named tool admits with a `would_deny` record; an un-named non-read tool fails closed; zero-set is HEAD-identical).
2. **Generalize the `would_deny` record to carry the would-have-denied rung on any complain-mode admit.** Where complain mode admits a default-deny, set `Meta["would_deny"]` to the rung's reason name via the existing `abi.ReasonName` call already used at `decide.go:384`, so the record distinguishes `DEFAULT_DENY` from a hard-refusal class; keep hard-refusal-class calls (`SELF_MODIFY` at `decide.go:272-316`, `MALFORMED` at `decide.go:342-351`, `ArgPredicate` at `decide.go:327-331`) failing closed under complain mode. *Test:* `TestComplainWouldDenyNamesRungAndFailsClosedOnHardRefusal`.
3. **Promotion ledger that counts clean complain-mode events per tool.** A new `internal/adjudicator/promote.go` (or a sibling `internal/promote` package) with a thread-safe per-tool counter folded from `EvDecide` complain-mode admits — modeled on `rulesynth.NearMissCorpus`'s `sync.Mutex` append log (`stream.go:28-57`) and `Harvester.Emit`'s `EvDecide`-keyed fold (`stream.go:91-102`). A hard-refusal-class would-deny resets the tool's run to zero. *Test:* `TestPromotionLedgerCountsCleanResetsOnHardRefusal`.
4. **Promotion offer + reviewable record (no in-process mutation).** A `Ledger.Promotable(n)` that returns the tools at N clean events with zero hard-refusal events, emitting a reviewable promotion record an operator applies — mirroring `rulesynth.Candidate.ManifestDiff`'s "emit a diff and STOP, never self-grade" contract (`rulesynth.go:265-269`). *Test:* `TestPromotableRequiresCleanThresholdAndEmitsReviewableRecord`.

## Honesty boundary

- **Stays heuristic:** which tool is "low risk" is still name-based — the complain set is operator-declared, and the `lowRiskReadShaped` prefix list (`decide.go:221-235`) is unchanged. The ledger counts *clean would-deny events*, which is evidence of "the floor would have refused this and the operator chose to admit it," NOT proof the tool is safe; promotion is a recommendation an operator reviews.
- **Stays fail-closed:** hard-refusal rungs (`SELF_MODIFY`, `MALFORMED`, `ArgPredicate`) never admit under complain mode and never count as promotable evidence — complain mode only ever relaxes a `DEFAULT_DENY`, exactly the bound the existing global posture already respects (`decide.go:377-389`).
- **Out of scope (named, not built):** the chain-level fold short-circuit (`kernel.go:185-194` still scans the whole chain) is untouched; an `INDETERMINATE`/"climb" verdict distinct from `Defer`→`DEFAULT_DENY` is not introduced; the policy-manifest surface that would let an operator DECLARE the complain set in a manifest (rather than construct the `Policy` field directly) is deferred to the `internal/policy` manifest track; and promotion never auto-applies — there is no in-process self-mutation of the floor.

---

## Epic 4

# Problem

The kernel runs two folds over the same restrictiveness lattice: a call-side fold (`Fold` -> `Decide`/`Submit`) that acts on the full disposition set (Allow / Defer / Transform / RequireWitness / Deny), and a result-side fold (`admitResult`) that folds the `ResultAdmitter` chain over a produced result. The two are NOT symmetric. `admitResult` acts only on `VerdictQuarantine` and `VerdictTransform`; a result-side `VerdictDeny` or `VerdictRequireWitness` falls through to the `default` branch and is silently counted as `Admitted` (internal/kernel/kernel.go:280-282).

This is the seccomp anti-pattern on the result path: an admitter that conclusively wants to KILL a produced result (deny-as-value) or LIFT it to a handback gate is downgraded to ALLOW. A produced RESULT therefore does not get the same graduated rungs a CALL gets. The asymmetry is the gap; the fold machinery, the lattice, the deny-as-value builder, and the witness gate already exist on the call side and can be reused.

Doctrine framing: the default rung stays the SMALLEST one — structural admit that default-admits on an empty chain and acts only on a conclusive admitter verdict. This change does not add a costlier rung that always runs; it only teaches the existing result fold to honor the two verdict kinds it currently drops, climbing to the witness gate ONLY when an admitter conclusively emits RequireWitness.

# Current state (file:line, read at HEAD)

- `admitResult` folds the `ResultAdmitter` chain by `FoldRank` and default-admits when the chain is empty or `r == nil` (internal/kernel/kernel.go:246-259). The switch on the folded `best.Kind` has cases ONLY for `VerdictQuarantine` (kernel.go:261-270) and `VerdictTransform` (kernel.go:271-279). The `default` branch (kernel.go:280-282) does `atomic.AddInt64(&k.ctr.Admitted, 1)` and returns `best` — so a folded `VerdictDeny` or `VerdictRequireWitness` is recorded as admitted.
- `AdmitResult` is the exported dual used by the served/proxy path (internal/kernel/kernel.go:240-242); the in-process `Reap` path calls `admitResult` at internal/kernel/kernel.go:409. Both share the one switch, so both inherit the gap.
- The call-side already has the missing rungs: `DenyResult` builds the structured deny-as-value with reason + disposition + bounded witness disclosure (internal/kernel/kernel.go:462-473); `Disposition` maps a reason to RETRYABLE/WAIT/ESCALATE/TERMINAL (internal/kernel/kernel.go:478-489); `resolveWitness` drives the require-witness gate, first-confirmed-wins, UNWITNESSED fail-closed (internal/kernel/kernel.go:208-231).
- The lattice is shared: `FoldRank` ranks Allow=0 < Defer=1 < Transform=2 < Quarantine=3 < RequireWitness=4 < Deny=100 (internal/abi/registry.go:865-884); the result fold already consults it (internal/kernel/kernel.go:253,256), so a result-side `VerdictDeny`/`VerdictRequireWitness` ALREADY wins the fold correctly — it is only the post-fold switch that drops it.
- The `ResultAdmitter` interface is `Admit(ctx, c, r) Verdict` (internal/abi/registry.go:603-606) — it can already return any `VerdictKind`. Nothing structurally prevents an admitter from emitting Deny/RequireWitness today; the kernel just ignores it.
- Counters carry no result-side deny/witness tally (internal/kernel/kernel.go:28-36): `Quarantines` and `Admitted` exist; there is no `ResultDenies` field.
- Event vocabulary has `EvQuarantine` for a held result (internal/abi/events.go:13) but NO result-deny event kind; `EvDeny` (events.go:10) is documented as a CALL refusal.
- The only realized result-side restriction today is the IFC StampGate clamping a TAINTED result's scope DOWN to `ScopeAgent` (internal/ifc/ifc.go:497-498) and then emitting `VerdictDefer` (internal/ifc/ifc.go:511) — i.e. no shipped admitter emits Deny/RequireWitness yet, so this change is latent-safe (no behavior change until an admitter opts in).
- Existing coverage: `TestDirectAdmitResultEmitsQuarantine` (internal/kernel/kernel_test.go:251-267) exercises only the Quarantine case via the `quarantineAdmitter` fixture (internal/kernel/kernel_test.go:75-80). No test exercises a result-side Deny or RequireWitness.

# Design — graduated rungs (default = smallest sufficient)

DEFAULT (rung, smallest sufficient): result-side structural admit (the MMU fold). On an empty `ResultAdmitter` chain or `r == nil`, default-admit (`VerdictAllow`, By `default-admit`) — UNCHANGED (kernel.go:248-251). On a non-empty chain, fold by `FoldRank` and act on the conclusive winner. Quarantine and Transform keep their exact current behavior.

ESCALATE only on a conclusive admitter verdict:

- `VerdictDeny` (new case): the result is hard-refused as a deny-as-value. Reuse `DenyResult(c, best)` shape — set `r.Status = StatusError`, `r.Outcome = OutcomeCommitted`, stamp `r.Meta[\"admit\"] = \"denied\"`, `r.Meta[\"reason\"]`/`r.Meta[\"disposition\"]` from the verdict reason via `Disposition`, increment a NEW `ResultDenies` counter, emit a result-deny event. Trigger: an admitter conclusively refuses the produced bytes (e.g. a future exfil/structural gate). Smaller rungs are insufficient because Quarantine pages-out-but-keeps; a Deny means the result must NOT enter context at all.

- `VerdictRequireWitness` (new case): route the result to the witness gate, exactly as `Submit` routes a call (kernel.go:320-334). Call `resolveWitness(ctx, c, best)`; a CONFIRMED claim admits the result (fall through to the admit tally), an UNWITNESSED/refuted one fails closed to the Deny path above. Trigger: an admitter asserts the result carries a claim a non-author resolver can corroborate (e.g. a provenance/origin claim). With no resolver registered, fail-closed — identical to the call-side default.

The explicit escalation trigger is: the post-fold `best.Kind`. The fold is unchanged; only the dispatch of the folded verdict widens. Order-independence and the most-restrictive-wins property are inherited from the shared `FoldRank` fold, so no new lattice surface is introduced.

Non-goal in this epic: chain-level short-circuit. The result fold loop (kernel.go:254-259), like the call fold (kernel.go:185-194), still evaluates the whole chain even after a rank-100 Deny. That is a separate, named gap and stays out of scope.

# DOS + Linux analogue

- DOS: the result path gains the same closed verdict ladder the call path has — a produced result can now be refused-as-value (`dos`-style structured refusal with a reason from the closed vocabulary) or lifted to a witness gate, instead of fail-open-on-anything-not-Quarantine.
- Linux: seccomp/LSM on the RETURN path — today the result hook can only TAINT/PAGE (Quarantine) or REWRITE (Transform); this adds KILL (Deny) and a CONFIRM-before-return gate (RequireWitness), so the egress filter is no longer downgrading KILL to ALLOW.

# Acceptance criteria (testable)

1. A `ResultAdmitter` that folds to `VerdictDeny` causes `AdmitResult` to return Deny, set `r.Status = StatusError` with `r.Meta[\"admit\"] = \"denied\"` and a disposition derived from the reason, and increment the new `ResultDenies` counter — NOT `Admitted`.
2. A `ResultAdmitter` that folds to `VerdictRequireWitness` with NO registered resolver fails closed (result denied, UNWITNESSED), preserving the call-side default.
3. A `ResultAdmitter` that folds to `VerdictRequireWitness` whose claim a registered resolver CONFIRMS admits the result (no deny).
4. Empty chain / `r == nil` / Quarantine / Transform behavior is byte-identical to HEAD (regression-guarded): `TestDirectAdmitResultEmitsQuarantine` still passes unchanged, and `Admitted`/`Quarantines` tallies are unchanged for those paths.
5. The in-process `Reap` path (kernel.go:409) and the exported `AdmitResult` (kernel.go:240) produce identical effects on `r` for every new verdict kind (the dual is preserved).
6. `make ci` green; no new `os/exec` on the hot path (the witness resolver seam is the existing one).

# Child issues (each independently shippable on the trunk)

1. Add the result-side `VerdictDeny` case to `admitResult` + a `ResultDenies` counter. Reuses `Disposition`/the `DenyResult` meta shape; smallest first cut, no witness routing. Test: `TestAdmitResultDenyHardRefuses` (internal/kernel/kernel_test.go) — register a deny-emitting admitter fixture (sibling of `quarantineAdmitter`), assert returned Kind + `r.Meta[\"admit\"]==\"denied\"` + `ResultDenies==1` + `Admitted` unchanged.

2. Route a result-side `VerdictRequireWitness` through `resolveWitness`. Reuses the call-side gate verbatim. Test: `TestAdmitResultRequireWitnessFailsClosed` and `TestAdmitResultRequireWitnessConfirmedAdmits` (internal/kernel/kernel_test.go) — one with no resolver (deny/UNWITNESSED), one with a confirming `WitnessResolver` fixture (admit).

3. Add a result-deny event kind (`EvResultDeny`) to the closed vocabulary and emit it on the new Deny path; keep `EvQuarantine` for the held-but-kept case. Test: `TestAdmitResultDenyEmitsResultDeny` (internal/kernel/kernel_test.go) using the existing `recordEmitter` (kernel_test.go:62-73).

4. Regression proof of result/call ladder parity: a table test asserting that for each of {Allow, Quarantine, Transform, RequireWitness, Deny} folded on the result side, `AdmitResult` and the `Reap` path agree, and that the empty-chain default-admit is unchanged. Test: `TestResultLadderParity` (internal/kernel/kernel_test.go).

5. (Optional, ships last) Wire one real admitter to emit Deny: extend the IFC result gate so a result whose taint exceeds a policy ceiling can hard-refuse instead of only clamping scope to `ScopeAgent` (internal/ifc/ifc.go:497-498). Test: `TestStampGateDeniesOverCeiling` (internal/ifc/ifc_test.go). Gated behind a policy bool so the default floor is byte-unchanged.

# Honesty boundary

- STAYS HEURISTIC: which admitters emit Deny vs Quarantine is a policy/detector judgment, not a structural proof. This epic only makes the kernel HONOR a conclusive verdict; it does not make any detector's verdict correct.
- STAYS UNBUILT / OUT OF SCOPE: (a) chain-level short-circuit on the result fold — the loop still scans the whole admitter chain after a rank-100 Deny (kernel.go:254-259), same as the call fold; (b) the result-side ShareScope UPWARD ceiling (a `ScopeFleet`/`ScopeTenant` result is not provably confined on the share path; only the DOWNWARD clamp at ifc.go:497-498 exists) — a separate gap; (c) a result-side WAIT/EVIDENCE_PENDING outcome — RequireWitness here is binary confirmed/closed exactly like the call side, with no retry-after; (d) no shipped admitter emits RequireWitness on the result side yet, so child issue 2 lands the mechanism with a test fixture, not a production caller.
- LATENT-SAFE: until an admitter opts into emitting Deny/RequireWitness, the only realized result admitters today are Quarantine/Transform/Defer, so this change is a no-op on the current fleet (no behavior change to ship anxiety).

CITED SEAMS (confirm each is real at HEAD — open the file, check the line):
[\"internal/kernel/kernel.go:240-242\",\"internal/kernel/kernel.go:246-259\",\"internal/kernel/kernel.go:261-270\",\"internal/kernel/kernel.go:271-279\",\"internal/kernel/kernel.go:280-282\",\"internal/kernel/kernel.go:283-284\",\"internal/kernel/kernel.go:320-334\",\"internal/kernel/kernel.go:409\",\"internal/kernel/kernel.go:462-473\",\"internal/kernel/kernel.go:478-489\",\"internal/kernel/kernel.go:208-231\",\"internal/kernel/kernel.go:185-194\",\"internal/kernel/kernel.go:253\",\"internal/kernel/kernel.go:256\",\"internal/kernel/kernel.go:28-36\",\"internal/abi/registry.go:603-606\",\"internal/abi/registry.go:786-795\",\"internal/abi/registry.go:865-884\",\"internal/abi/events.go:10\",\"internal/abi/events.go:13\",\"internal/ifc/ifc.go:497-498\",\"internal/ifc/ifc.go:511\",\"internal/kernel/kernel_test.go:62-73\",\"internal/kernel/kernel_test.go:75-80\",\"internal/kernel/kernel_test.go:251-267\"]

\"NOT BUILT / RIGID TODAY\" CLAIMS (confirm each — grep the tree; a claim that \"X is not enforced\" is FALSE if any non-test code enforces it):
[\"admitResult acts only on VerdictQuarantine and VerdictTransform; a folded VerdictDeny or VerdictRequireWitness falls through the default branch and is silently counted as Admitted (internal/kernel/kernel.go:280-282).\",\"There is no result-side deny-as-value path: DenyResult/Disposition exist only on the call side (internal/kernel/kernel.go:462-489) and are not invoked by admitResult.\",\"resolveWitness is wired only into the call-side Submit path (internal/kernel/kernel.go:320-334); the result fold never routes a RequireWitness verdict to it.\",\"The Counters struct has no result-side deny or witness tally — only Quarantines and Admitted (internal/kernel/kernel.go:28-36).\",\"The EventKind vocabulary has no result-deny event; only EvQuarantine covers the result side (internal/abi/events.go:13), and EvDeny is documented as a call refusal (internal/abi/events.go:10).\",\"The result fold loop does not short-circuit — it scans the whole ResultAdmitter chain even after a rank-100 Deny is maximal (internal/kernel/kernel.go:254-259), mirroring the call fold (internal/kernel/kernel.go:185-194).\",\"No shipped ResultAdmitter emits Deny or RequireWitness today; the only realized result-side restriction is the IFC StampGate clamping a tainted result's scope down to ScopeAgent and then emitting Defer (internal/ifc/ifc.go:497-498, 511).\",\"The result-side ShareScope upward ceiling is unenforced: only the downward clamp to ScopeAgent exists (internal/ifc/ifc.go:497-498); nothing confines a ScopeFleet/ScopeTenant result on the share path.\",\"RequireWitness carries no WAIT/EVIDENCE_PENDING/retry-after outcome; it is binary confirmed/closed on both call and (proposed) result sides.\",\"No existing test exercises a result-side Deny or RequireWitness; coverage stops at the Quarantine case (internal/kernel/kernel_test.go:251-267).\"]

---

## Epic 5

# feat(shipgate): smallest-sufficient-evidence keep-bit (graduated EvidenceProfile, per-candidate-class)

## Problem

The RSI ship gate is **all-three-or-revert**. `Evaluate` sets the non-forgeable keep-bit as a flat conjunction — `improvedBit = w.improved() && w.SuiteGreen && w.TruthClean` (`internal/shipgate/shipgate.go:67`). There is no notion of *smallest sufficient evidence*: a docs-only candidate, a comment-only candidate, or a proof-carrying candidate that cannot move a runtime metric all REVERT, because at least one of the three signals is necessarily false even though the keep property they care about *is* established by a cheaper subset.

Worse, every candidate that the harness measures pays the full rung-6 cost regardless of class: `ApplyInWorktree` always shells out `git worktree add --detach` off HEAD (`internal/shipgate/shipgate.go:110-130`, the `exec.Command` at `:121`), then the harness runs the suite to fill `SuiteGreen`/`TruthClean` (the `Witness` built from `Measure` at `internal/rsiloop/rsiloop.go:212-219`). A change whose keep property is git-checkable alone (rung 5) still pays a ms-spawn worktree fork plus a full suite (rung 6) it does not need.

The doctrine says verification defaults to the **smallest rung that can establish the property**, escalating only on INDETERMINATE / high risk. The keep-bit today is pinned at the *top* rung for *every* candidate. This epic adds a graduated `EvidenceProfile` so a declared candidate class is satisfied by the smallest sufficient subset of `{strict-gain, suite-green, truth-clean}` — while keeping the full all-three AND as the **default** and leaving the non-forgeable bit unchanged.

The known live trap (from the RSI-maturity work) is **re-gating the keep-bit through `dos improve`**, which makes the keep decision a tautology and erodes non-forgeability. This design does NOT touch that seam: the profile only narrows *which measured signals are required* for a declared class; `improvedBit` is still set only inside `Evaluate` from a measured witness, and every class still requires non-forgeable measured inputs (no class can be satisfied by a candidate's say-so).

## Current state (every claim read at HEAD)

- The keep-bit is a flat three-way AND with no smaller-sufficient subset: `internal/shipgate/shipgate.go:67` (`w.improvedBit = w.improved() && w.SuiteGreen && w.TruthClean`).
- `improvedBit` is the unexported non-forgeable cell, set **only** inside `Evaluate`: declared `internal/shipgate/shipgate.go:52`; set `internal/shipgate/shipgate.go:66-72`; read back via `Kept()` `internal/shipgate/shipgate.go:74-76`.
- The `Witness` carries `SuiteGreen` and `TruthClean` as plain caller-supplied bools and a free-form `Metric` with `Before`/`After`/`LowerBetter`: `internal/shipgate/shipgate.go:45-53`.
- `improved()` is a strict inequality only — no epsilon / effect-size / class notion: `internal/shipgate/shipgate.go:55-61`.
- `Evaluate` only ever returns `KEEP` or `REVERT` — never `ESCALATE`; that path is reachable only through the consecutive-non-keep breaker `Gate.Record` (`internal/shipgate/shipgate.go:95-105`). The proof test asserts this: `internal/shipgate/proofs_witness_test.go:88-91`.
- The determinism / non-vacuity proofs the new profiles must extend: repeat sweep + non-vacuity at `internal/shipgate/proofs_witness_test.go:100-128` (the "saw both outcomes" assertion at `:125-127`), the concurrent sweep at `:135-164`, the `quick.Check` at `:170-191`. `genWitness` deliberately leaves `improvedBit` zero to prove a caller cannot pre-seed it: `internal/shipgate/proofs_witness_test.go:27-29`.
- Every candidate pays the worktree fork: `ApplyInWorktree` always runs `git worktree add --detach` via `exec.Command` (`internal/shipgate/shipgate.go:121`), torn down by `RemoveWorktree` (`internal/shipgate/shipgate.go:132-139`); the os/exec waiver rationale is `internal/shipgate/shipgate.go:9-12`.
- The harness fills `SuiteGreen`/`TruthClean` from `Measure` unconditionally and constructs the `Witness`: `internal/rsiloop/rsiloop.go:212-219`; `Evaluate` is called at `internal/rsiloop/rsiloop.go:220`; `RunObserved`'s observer is side-effect-only and never re-gates the keep-bit (`internal/rsiloop/rsiloop.go:180-183`).
- The in-band ship rung (`ShipAdjudicator`) is a separate mechanism — it lifts a ship-shaped call to `VerdictRequireWitness` (`internal/shipgate/adjudicate.go:51-65`) at rank 40 (`internal/shipgate/adjudicate.go:74`) and does NOT call `Evaluate`. This epic does not touch it.

## Design — the graduated rungs (default = smallest sufficient, explicit escalation trigger)

Add an `EvidenceProfile` that names, per declared candidate class, which of the three measured signals are *required* for that class. `Evaluate` keeps the same shape; the only change is which signals it folds for the declared class. `improvedBit` is still computed only inside `Evaluate`, still from the measured witness.

The candidate class is a closed enum carried on the `Witness` (a new exported `Class` field, defaulting to the zero value `ClassFull`). The profile maps a class to a required-signal set:

- **`ClassFull` (THE DEFAULT, smallest sufficient for an unknown/runtime candidate):** requires `improved() && SuiteGreen && TruthClean` — **byte-identical to today**. An unclassified candidate, a candidate that touches code, or any candidate the harness cannot prove is narrower lands here. This is the conservative default; the zero value is the safe value.
- **`ClassDocsOnly` (smallest sufficient = rung 5, git-evidence alone):** requires `TruthClean` only. A docs/comment-only change establishes its keep property from `dos verify` confirming the change landed clean; there is nothing for the suite or a runtime metric to corroborate, so requiring them is *over-gating*, not safety. **The harness must independently prove the candidate touched only doc paths before it may declare this class** — the class is a claim the harness verifies, not the candidate's say-so.
- **`ClassProofCarrying` (smallest sufficient = gain + git-evidence, suite irrelevant):** requires `improved() && TruthClean`. A change that ships a new proof/witness and a measured gain, where the suite green-ness is not the property in question (e.g. a determinism proof whose own gate is the gain), is satisfied without re-deriving suite-green as a separate bound.

**Escalation trigger (the climb rule):** a class downgrade is permitted only when the harness has *independently established* the class predicate (e.g. doc-only path set) AND none of the required signals is INDETERMINATE. If the harness **cannot prove** the narrower class, or any required signal is missing/unmeasurable, the class **falls back to `ClassFull`** (all-three) — fail-up, never fail-down. There is no per-candidate `ESCALATE` from `Evaluate` (that stays a property of the breaker, `Gate.Record` `internal/shipgate/shipgate.go:95-105`); the escalation here is *rung escalation* — a narrower class that cannot be proven climbs back to the full rung.

Non-forgeability is preserved structurally: the required-signal subset for a class is **always a subset of {improved, SuiteGreen, TruthClean}** — the profile can only *drop* a required signal, never *add* a forgeable one, and it can never set the keep-bit from a non-measured input. A new proof obligation asserts: for **every** class, `Kept()` is a pure function of the measured witness fields the profile names, and no class admits a keep-bit set from `Class` alone.

## DOS + Linux analogue

- **DOS:** the keep-bit is a rung-6 `dos verify`-grade fold; the `EvidenceProfile` is the per-class "smallest sufficient evidence" selector — a docs-only candidate is bound by git-evidence alone (rung 5) instead of being forced through the worktree-measure rung (rung 6), exactly as `dos verify` binds the commit not the narration.
- **Linux:** like a capability bounding set that *drops* privileges per task (`no_new_privs` / a monotone-shrink mask) — a class can only narrow the required-evidence set, never widen it, and an unprovable narrowing latches back to the full set.

## Acceptance criteria (testable)

1. The zero-value `Witness` (no `Class` set) yields a keep decision **byte-identical** to today's all-three AND for every input — proven by extending the existing determinism sweep and adding a "default class == legacy behavior" equivalence test.
2. A `ClassDocsOnly` witness with `TruthClean=true` KEEPs even when `SuiteGreen=false` and `improved()=false`; with `TruthClean=false` it REVERTs.
3. A `ClassProofCarrying` witness with `improved() && TruthClean` KEEPs even when `SuiteGreen=false`; missing either required signal REVERTs.
4. No class can produce `Kept()==true` from an unmeasured input: a property test shows the keep-bit is a pure function of the profile's named signals, and `Class` alone never sets it (extends `internal/shipgate/proofs_witness_test.go:27-29`).
5. The determinism / non-vacuity / concurrent / `quick.Check` proofs (`internal/shipgate/proofs_witness_test.go:100-191`) pass for **every** class, with the non-vacuity assertion (`:125-127`) extended to require both KEEP and REVERT *per class*.
6. `Evaluate` still never returns `ESCALATE` for any class (the `proofs_witness_test.go:88-91` invariant holds unchanged).
7. An unprovable narrower class falls back to `ClassFull` (the harness gate refuses to apply a docs-only profile to a candidate it cannot prove is docs-only).
8. The in-band ship rung (`internal/shipgate/adjudicate.go:51-65`, rank 40 at `:74`) is untouched; the keep-bit is never re-gated through `dos improve` (the `RunObserved` observer at `internal/rsiloop/rsiloop.go:180-183` stays side-effect-only).

## Child issues (each independently shippable on the trunk, each with a named test)

1. **`feat(shipgate): add EvidenceProfile + Witness.Class with ClassFull default == legacy AND`** — add the `Class` enum (`ClassFull` zero value), the profile map, and fold the per-class required-signal subset inside `Evaluate` (`internal/shipgate/shipgate.go:66-72`). Ships alone: behavior is unchanged for the default class. Test: `TestEvaluateClassFullEqualsLegacyAND` (every witness's keep-bit equals the old `improved() && SuiteGreen && TruthClean`).

2. **`feat(shipgate): ClassDocsOnly keeps on truth-clean alone`** — wire the docs-only required-signal subset `{TruthClean}`. Test: `TestEvaluateClassDocsOnlyTruthCleanSufficient` (KEEP on `TruthClean` alone, REVERT without it, suite/gain ignored).

3. **`feat(shipgate): ClassProofCarrying keeps on gain + truth-clean`** — wire `{improved, TruthClean}`. Test: `TestEvaluateClassProofCarryingSkipsSuite`.

4. **`test(shipgate): per-class determinism + non-forgeability proofs`** — extend `internal/shipgate/proofs_witness_test.go` so the repeat/concurrent/quick sweeps and the non-vacuity assertion (`:125-127`) run per class, and add a "keep-bit is a pure function of the profile's named signals; `Class` alone never sets it" property. Test: `TestEvaluateKeepBitPureOverProfileSignals`.

5. **`feat(rsiloop): harness proves candidate class before downgrading; unprovable class falls back to ClassFull`** — let the harness declare a class only after independently establishing its predicate (e.g. doc-only path set), and default to `ClassFull` otherwise, then pass it into the `Witness` built at `internal/rsiloop/rsiloop.go:212-219`. Test: `TestRunDocsOnlyClassFallsBackToFullWhenUnproven`.

6. **`feat(shipgate): skip worktree+suite when the declared class makes them irrelevant`** — when the proven class requires neither `improved()` nor `SuiteGreen`, the harness path takes the cheaper rung (no `ApplyInWorktree` fork / no suite run), gated on the class being harness-proven. Touches the harness measure path, not `Evaluate`. Test: `TestDocsOnlyClassSkipsWorktreeMeasure` (asserts no `git worktree add` is invoked for a proven docs-only candidate).

## Honesty boundary

- **Stays heuristic:** *which class a candidate belongs to* is a harness judgment (e.g. "are these paths docs-only?"). That predicate is heuristic and must fail UP to `ClassFull` when unprovable — the profile makes the *evidence* requirement structural, not the *classification*. A misclassified candidate is safe only because the fallback is the full rung.
- **Unchanged / not re-gated:** the keep-bit's non-forgeability (`internal/shipgate/shipgate.go:52`), the breaker's exclusive ownership of `ESCALATE` (`internal/shipgate/shipgate.go:95-105`, `proofs_witness_test.go:88-91`), and the rule that `Evaluate` is never re-gated through `dos improve` (the observe-only seam `internal/rsiloop/rsiloop.go:180-183`).
- **Out of scope (named, not built here):** the in-band ship rung `ShipAdjudicator` (`internal/shipgate/adjudicate.go:51-65`) keeps its binary ship-shaped/Defer shape — no per-ship "small ship needs less corroboration" tier is added. The `improved()` strict-inequality with no epsilon/effect-size/noise-band (`internal/shipgate/shipgate.go:55-61`) is left as-is; graduated *strength of gain* is a separate concern. No new candidate class beyond the three is added; an operator wanting more edits the profile, which is the intended seam.

---

## Epic 6

# Epic: result-side ShareScope ceiling enforcement

## Problem

`internal/abi/types.go:60-63` documents a load-bearing invariant for the cross-agent shared-result pool: *"a result is never shared more widely than its scope, and sharing a result shares its taint. Defaults (Tainted, ScopeAgent) mean 'never shared, untrusted' — the fail-closed baseline."*

That invariant has only a **downward** realization. The IFC stamp gate clamps a *tainted* result's scope DOWN to `ScopeAgent` (`internal/ifc/ifc.go:496-499`), and the KV-pool reuse gate refuses to alias a `ScopeAgent` cell across a tenant boundary on the READER side (`internal/cachemeta/pool.go:200-202`). But nothing on the kernel's result-admit path checks the **upward** bound: when a result is *tagged* `ScopeFleet`/`ScopeTenant`, no rung confirms that the share target is actually inside that boundary before the result enters context / is handed back. A mis-tagged or over-tagged result is admitted as-is.

This is the result-side dual of a gate fak already ships on the call side: the engine-residency gate denies a `ScopeTenant` call routed to a remote engine (`internal/engine/engine.go:233-249`, `sensitiveRoute` at `:251-268`). The seam shape — \"read the scope on the `abi.Ref`, refuse a boundary crossing, cite `ReasonTrustViolation`\" — is already proven and registered (`internal/engine/engine.go:222`, rank 12). This epic extends that exact shape from the call route to the **result share boundary** by adding a `ResultAdmitter`.

## Current state (every claim read at HEAD)

- The invariant text: `internal/abi/types.go:60-63` (doc), `internal/abi/types.go:71` (`Scope ShareScope` field on `Ref`).
- The closed `ShareScope` lattice: `internal/abi/types.go:93-100` — `ScopeAgent`(0, fail-closed) < `ScopeFleet` < `ScopeTenant`, declared \"CLOSED, additive\".
- The ONLY write to `r.Payload.Scope` anywhere in `internal/`: `internal/ifc/ifc.go:498`, a DOWN-clamp to `ScopeAgent` for tainted data inside `StampGate.Admit` (which itself returns `VerdictDefer`, `internal/ifc/ifc.go:511`). There is no upward-bound check.
- The proven call-side pattern to mirror: `internal/engine/engine.go:233-249` (`residencyGate.Adjudicate` → `VerdictDeny`/`ReasonTrustViolation`), `internal/engine/engine.go:251-268` (`sensitiveRoute` keys on `c.Args.Scope == abi.ScopeTenant` plus a sensitivity tag), registered at `internal/engine/engine.go:216-222` via `abi.RegisterAdjudicator(12, residencyGate{})`.
- The result-admit fold this epic plugs into: `internal/kernel/kernel.go:244-284` (`admitResult` folds `abi.ResultAdmittersFor(c)` by `FoldRank`, most-restrictive wins), exported as `AdmitResult` at `internal/kernel/kernel.go:240-242`, wired from the in-process Reap path and the gateway served path.
- The result-admit registration seam: `internal/abi/registry.go:439-449` (`RegisterResultAdmitter(rank, ra)`), the interface at `internal/abi/registry.go:603-606`, per-tool scoping via `CallScope` at `internal/abi/registry.go:711-727`, fold selection via `ResultAdmittersFor` at `internal/abi/registry.go:803-810`.
- The existing result-admit chain ranks at HEAD (the rank the new gate must order against): `normgate` rank 5 (`internal/normgate/normgate.go:320`), `ctxmmu` rank 10 (`internal/ctxmmu/mmu.go:580`), `StampGate` rank 20 (`internal/ifc/ifc.go:649`; the comment is explicit — "rank 20 > ctxmmu 10 > normgate 5", meaning higher ranks run later in the chain).
- The reason to cite already exists in the closed vocabulary: `internal/abi/reasons.go:16` — `ReasonTrustViolation` is documented as \"taint/scope violation (shared-result isolation)\". No new reason code is needed.
- A reader-side scope refusal that already exists but is NOT this gate: `internal/cachemeta/pool.go:195-213` (`PoolReuseVerdict`) refuses to alias a `ScopeAgent` cell across a tenant boundary, using a package-local `LookupReason` (`internal/cachemeta/cachemeta.go:264`, `ReasonScopeDenied = \"scope_denied\"`). This is a KV-cache placement gate keyed on a `MaterializationKey`, not the kernel result-admit path on `abi.Result.Payload.Scope`.

### What is NOT enforced today

- The kernel result-admit fold (`internal/kernel/kernel.go:260-282`) acts ONLY on `VerdictQuarantine` and `VerdictTransform`; a result-side `VerdictDeny`/`VerdictRequireWitness` falls through to the default `admitted++` branch (`internal/kernel/kernel.go:280-282`). The result ladder is narrower than the call ladder.
- No rung anywhere reads `r.Payload.Scope` to enforce the upward bound. The only `r.Payload.Scope` access in `internal/` is the down-clamp at `internal/ifc/ifc.go:498`.

## Design — the graduated rungs

The gate is a `ResultAdmitter` (the result-side dual of `residencyGate`) that reads `r.Payload.Scope` and the share target on the call's `Meta`, and confines the result to its declared boundary. It follows fak doctrine: the DEFAULT does no work, and the rung only engages when there is something to check.

**Rung 0 — DEFAULT, smallest sufficient (structural admit, no work):** a `ScopeAgent` result (the fail-closed zero value) is the private baseline; it is never shared, so there is nothing to confine. The gate returns its identity verdict (`VerdictAllow`/admit-as-is) immediately. A `nil` result also admits. This is the common case and costs a single enum compare.

**Rung 1 — boundary confinement (structural deny):** the rung engages ONLY when `r.Payload.Scope` is `ScopeFleet` or `ScopeTenant`. It reads the share target from the call `Meta` (the same place `sensitiveRoute` reads `c.Meta[\"sensitivity\"]`/`c.Meta[\"data_sensitivity\"]` at `internal/engine/engine.go:256-260`) — a target scope/partition tag. The gate is provable from values already on the call/result:
- If the result's declared scope is *wider* than the share target's boundary (e.g. a `ScopeAgent`-bounded share target receiving a `ScopeFleet`/`ScopeTenant` result, or a `ScopeFleet` target receiving a `ScopeTenant` result), that is a boundary crossing.
- A crossing is refused. The smallest sufficient refusal is **Quarantine** (`VerdictQuarantine`) — the result is held out of the wider context rather than leaked, exactly as `admitResult` already pages out a quarantined result (`internal/kernel/kernel.go:260-270`). The witness cites `ReasonTrustViolation` (`internal/abi/reasons.go:16`) with a bounded disclosure (the two scopes, never the payload).
- Quarantine is chosen over `VerdictDeny` deliberately: `admitResult` already HAS a quarantine branch, so this rung is enforceable today with zero kernel edits. (A result-side `VerdictDeny` would silently fall through at `internal/kernel/kernel.go:280-282` — that gap is tracked as a separate child issue below, NOT a prerequisite.)

**Escalation trigger (explicit):** the gate is INDETERMINATE — and must defer rather than admit-as-if-safe — when the result is tagged `ScopeFleet`/`ScopeTenant` but the call carries NO share-target tag the gate can read. In that case it cannot prove the share is in-bounds *or* out-of-bounds from local values. Per doctrine it returns the most-restrictive defensible verdict for a missing-evidence share: `VerdictQuarantine` with a distinct `share_target=unknown` meta note (fail-closed — a wider-than-Agent result with an unknowable target is confined, not admitted). It does NOT climb to require-witness/human: confining the result locally is sufficient and terminal; there is no out-of-band corroboration a person could add that confinement does not already achieve.

### DOS + Linux analogue

- **DOS:** the result-side dual of `dos_refuse_reasons`/`dos_check_reason` — a share that crosses a declared boundary is refused with a structured, closed-vocabulary reason (`TRUST_VIOLATION`) instead of leaking, the same way the call-side residency gate already refuses a tenant→remote route.
- **Linux:** an LSM/SELinux *type-transition* check on the read side — an object labeled at one domain cannot be relabeled into a wider domain on read/share; the default (private label) needs no check, the relabel is where the hook fires.

## Acceptance criteria (testable)

1. A `ScopeAgent` result (default) folds through the new gate to `VerdictAllow`/admit-as-is with no allocation and no boundary read — verifiable by a Go test asserting verdict identity. The no-op guarantee is proven by the new `TestScopeCeilingDefaultIsNoop` (child issue 2), which mirrors the fold-equivalence pattern of `internal/kernel/kernel_scope_test.go` (`TestScopedFoldEquivalentToFullChain`, a CallScope per-tool fold-equivalence proof — the existing reference for \"a self-Deferring rung does not perturb the fold\").
2. A `ScopeTenant`-tagged result whose call names a `ScopeFleet` (narrower) share target is Quarantined with `By` = the new gate id and `Reason == abi.ReasonTrustViolation`; the bytes do not enter the wider context (`r.Meta[\"admit\"] == \"quarantined\"` per `internal/kernel/kernel.go:269`).
3. A `ScopeFleet` result whose call names a `ScopeFleet`-or-wider share target is admitted (in-bounds share is NOT a regression).
4. A `ScopeFleet`/`ScopeTenant` result with NO readable share target is Quarantined (fail-closed) with a distinct `share_target=unknown` meta marker.
5. The witness never discloses the payload — only the declared scope and the target scope (mirrors the bounded disclosure discipline already used by `residencyGate`).
6. The gate is registered additively (no edit to the closed `VerdictKind`/`ReasonCode` sets) and the full kernel suite (`make ci`) stays green; the gate runs only on results, never on the call path.

## Child issues (each independently shippable on the trunk)

1. **`feat(ifc): add ScopeCeilingGate ResultAdmitter (rung 0 default-admit + rung 1 quarantine)`** — the core gate in `internal/ifc`, registered via `abi.RegisterResultAdmitter` at a rank ABOVE the rank-20 `StampGate` (e.g. rank 21), so the ceiling folds AFTER taint is stamped and the tainted-data down-clamp at `internal/ifc/ifc.go:498` has already run. (Note: rank 10 is `ctxmmu`, not StampGate; StampGate is rank 20 per `internal/ifc/ifc.go:649`.) Test: `TestScopeCeilingGateConfinesWiderShare` (table over Agent/Fleet/Tenant result × target combinations, asserting Allow vs Quarantine).
2. **`test(ifc): prove the default ScopeAgent path is a no-op`** — a dedicated proof that a `ScopeAgent` result folds verdict-identically with and without the gate registered (the \"smallest rung does no work\" guarantee). Test: `TestScopeCeilingDefaultIsNoop`.
3. **`feat(ifc): fail-closed quarantine on an unreadable share target`** — the INDETERMINATE escalation arm; a wider-than-Agent result with no target tag is confined. Test: `TestScopeCeilingUnknownTargetQuarantines`.
4. **`feat(kernel): handle a result-side VerdictDeny in admitResult`** — close the narrower-result-ladder gap at `internal/kernel/kernel.go:280-282` so a future result gate CAN hard-deny (not just quarantine); this gate stays on Quarantine but the kernel branch becomes available. Test: `TestAdmitResultHonorsResultSideDeny`. Independently shippable; not a prerequisite for issues 1-3.
5. **`docs(model-routing): document the result-side scope ceiling next to the call-side residency gate`** — one paragraph in the residency/scope doc tying the two duals together, edited via the generator if the surface is generated. Test: the existing doc-link/scrub gates in `make ci` stay green.

## Honesty boundary

- **Stays heuristic:** the share *target* boundary is read from call `Meta` tags (the same tag surface `sensitiveRoute` keys on at `internal/engine/engine.go:256-260`). A caller that mislabels its OWN target tag is outside this gate's reach — the gate proves \"declared result scope ≤ declared target scope\", not \"the target tag is truthful\". That is the same trust assumption the call-side residency gate already makes.
- **Stays unbuilt / out of scope:** result-side `VerdictRequireWitness` (corroborating a share against out-of-band evidence) is not added — the gate confines locally via Quarantine, which is sufficient and terminal. The KV-pool reader-side gate (`internal/cachemeta/pool.go:195-213`) is a separate placement path and is unchanged. No new `ReasonCode` or `VerdictKind` is introduced — the closed sets at `internal/abi/reasons.go:10-27` and the lattice stay frozen. The general \"result ladder is narrower than the call ladder\" gap is only *partially* closed (issue 4 adds the Deny branch; result-side require-witness routing remains deferred)."

---

## Epic 7

# Problem

fak's whole pitch is a graduated capability floor: a call is decided by the *cheapest rung that can establish the property*, and only climbs to a costlier rung when a cheaper one cannot conclude. But that discipline is currently **invisible at runtime**. An operator running a live gateway cannot answer the first question any graduated system raises: *which rung is actually deciding each call?* — is the deny path dominated by the name-level deny-map (rung 1), by the witness gate (rung 3), or by fail-closed default-deny? Is the vDSO (rung 0) carrying real load or hitting 0%? Are the expensive rungs load-bearing or dead weight?

The per-rung trace is already *computed*. `kernel.FoldExplain` (`internal/kernel/explain.go:74`) folds the same chain to the same winning verdict as `Fold`, and additionally records a per-rung `RungVerdict` with a `Winner bool` (`explain.go:38-48`). The single-call CLI surface is also already built: `fak preflight --explain` / `--json` prints that ladder for one synthetic call (`cmd/fak/main.go:360-369`). So the *forensic trace for one call you hand-pick* exists.

What does **not** exist is the *aggregate distribution over the live stream*. `FoldExplain` is documented as \"Built only off the hot path\" (`explain.go:53`) and nothing folds its output into a counter. The hot path emits `EvDecide` / `EvDeny` events (`internal/kernel/kernel.go:131,318,328,333`) carrying the resolved `Verdict`, but those events carry only the *winning* verdict's `By` / `Reason` — not which structural rung produced it. The gateway's `/metrics` exposes seven flat kernel counters — `fak_kernel_submits_total`, `..._denies_total`, `..._transforms_total`, etc. (`internal/gateway/metrics.go:435-441`) — but **none is broken down by winning rung or by reason**. So flexibility is computed, surfaced for a single hand-fed call, and aggregated nowhere.

This is distinct from routing observability (#603, \"which *model* answered\"). This is **which *adjudication rung* decided** — the call-side capability floor, not the engine route.

# Current state (every claim read at HEAD)

- `internal/kernel/explain.go:38-48` — `RungVerdict{Index, Rung, By, Kind, Reason, Claim, Rank, Deferred, Winner}` is the per-rung detail `Fold` discards.
- `internal/kernel/explain.go:53` — doc: the `Decision` trace is \"Built only off the hot path.\"
- `internal/kernel/explain.go:74-127` — `FoldExplain` returns `(abi.Verdict, Decision)`, byte-identical verdict to `Fold`, with `d.Rungs[bestIdx].Winner = true` (`explain.go:121`).
- `internal/kernel/kernel.go:178-194` — `Fold` (the hot path) keeps only the winning verdict; the per-rung detail is never retained.
- `internal/kernel/kernel.go:131,318,328,333,361` — `EvDecide` / `EvDeny` emitted carrying `&v` (the resolved verdict only — no rung breakdown).
- `internal/kernel/kernel.go:309-311` — a vDSO hit emits `EvVDSOHit` with `Verdict{Kind: VerdictAllow, By: \"vdso\"}`; no adjudication ran, so there is no rung to attribute.
- `internal/kernel/kernel.go:421-432` — `emit` fans an event only to `abi.EmittersFor(ev.Kind)`; cost is O(interested), so an observer that subscribes to only `EvDecide`/`EvDeny` adds nothing to the `EvSubmit`/`EvDispatch`/`EvComplete` path.
- `internal/abi/registry.go:707-709` — `EventSubscriber.Subscriptions() []EventKind` lets an Emitter scope itself; `internal/abi/registry.go:838-845` — `EmittersFor(kind)` returns only subscribers for that kind (universal `s.allEmitters` fallback if none declared).
- `internal/abi/types.go:322` — `Emitter interface{ Emit(ev Event) }`.
- `internal/harvest/harvest.go:104-134` — the existing reference Emitter: registered via `abi.RegisterEmitter`, folds the `EvDecide`/`EvDeny` stream into a corpus. The pattern this epic copies, except harvest does NOT re-fold the chain (it sets `RungPassed: -1`, `harvest.go:128` — \"unknown without an explicit ladder label\"), so it cannot tell you the winning rung.
- `cmd/fak/main.go:360-369` — `fak preflight --explain`/`--json` already calls `FoldExplain` (the call is at `main.go:361`) and prints `d.Text()` / `d.JSON()` for one synthetic call.
- `internal/gateway/metrics.go:434-441` — the seven flat `fak_kernel_*_total` counters; `writeCounter` (`metrics.go:757-760`) and `writeHelpType` (`metrics.go:752-755`) are the prometheus-text render helpers.
- `internal/gateway/metrics.go:434` / `internal/gateway/debug.go:120` — both read `s.k.Counters()` (the flat snapshot), which has no rung dimension.

# Design — graduated rungs (DEFAULT = the smallest sufficient one)

The property to establish: *the rung-decision distribution is observable.* Establish it at the **smallest** rung that can; escalate only on the explicit trigger.

**DEFAULT — Rung A (smallest sufficient): a passive counting Emitter over the verdict events the kernel already emits.**
A new observer (`internal/rungobs`) implements `abi.Emitter` + `EventSubscriber{EvDecide, EvDeny}`, registered via `abi.RegisterEmitter`. On each event it re-folds the call's chain with `FoldExplain` *inside the observer* (off the syscall's critical section — the observer already runs after the verdict is resolved and `emit` is O(interested)), reads the `Winner` rung's type + the verdict reason, and bumps an in-memory labeled counter keyed `(rung, kind, reason)`. This adds **zero** rungs to adjudication, never touches the deny/allow path, and is fully removable by not registering it. *Sufficient because* the question — \"which rung won, how often\" — is answerable from data the kernel already produces; no new decision logic is required.

**Escalation trigger → Rung B:** re-folding the chain in the observer is only correct if it is **provably the same fold** the hot path ran. If a future change makes the chain call-instance-specific (an injected `WithAdjudicators` chain the global registry can't reproduce), the observer's re-fold would diverge from the real verdict — an INDETERMINATE attribution. **Only then** escalate to carrying the winning-rung identity *on the event itself* (a new `Event.Rung` field or a typed `LabelRow` like the existing `EvRungLabel` path, `internal/abi/events.go:15-17`), so the counter reads ground truth instead of re-deriving it. This is a core ABI edit and is deliberately deferred until the re-fold is shown to diverge.

**Rung C (only on operator demand, already built):** the single-call ladder — `fak preflight --explain`. Already shipped (`cmd/fak/main.go:360-369`); this epic does NOT rebuild it, it points the new `/metrics` HELP text at it as the drill-down.

The DEFAULT is Rung A because it establishes observability with a passive O(interested) tap and no ABI change. We escalate to Rung B (event-carried rung identity) ONLY if the observer's re-fold is shown to diverge from the hot-path verdict.

# DOS + Linux analogue

- **DOS:** this is `dos enforce-tune`'s evidence side — the same way enforce-tune tunes policy knobs from false-deny vs held-catch *counts*, the rung-decision counter is the per-rung firing distribution you'd tune the ladder against (which rung's denies dominate).
- **Linux:** `/proc`-style passive accounting — the rung counter is to the adjudicator chain what per-syscall `seccomp` action counters (`SECCOMP_RET_*` audit counts) are to a seccomp filter: it tells you which filter rule is firing without changing what the filter does.

# Acceptance criteria (testable)

1. A registered `rungobs` Emitter, after N decided calls, reports a per-rung histogram whose totals reconcile exactly with `kernel.Counters().Denies + .Transforms + (allows)` for the same stream (no double-count, no drop) — asserted by a Go test.
2. For a stream the observer attributes a winning rung to, that rung equals `FoldExplain(...).Rungs[winner].Rung` for the same call (the observer agrees with the canonical trace) — asserted by a Go test.
3. A vDSO-served call (`EvVDSOHit`, no adjudication) is counted under a distinct `rung=\"vdso\"` bucket, never misattributed to a structural rung — asserted by a Go test.
4. Registering the observer adds **0 allocations** to `emit` for an event kind it does NOT subscribe to (it must declare `Subscriptions(){EvDecide, EvDeny}`), proven with `testing.AllocsPerRun`, mirroring `TestEmittersForZeroAlloc` (`internal/abi/registry_events_test.go:91-102`).
5. `/metrics` exposes `fak_kernel_decisions_total{rung,kind,reason}` as a prometheus counter with `# TYPE ... counter`, asserted by a `renderMetrics()` substring test mirroring `internal/gateway/metrics_test.go:45-47`.
6. The decide/deny hot path is byte-for-byte unchanged with the observer present vs absent (same verdict, same `Counters()` flat totals) — the observer is provably passive.

# Child issues (each independently shippable on the trunk)

1. **`internal/rungobs`: the passive rung-decision counter.** New leaf package: an `abi.Emitter`+`EventSubscriber{EvDecide,EvDeny,EvVDSOHit}` that folds the verdict stream into a `map[rungKey]int64` (rung, kind, reason), re-deriving the winner via `FoldExplain`. Snapshot accessor returns sorted rows. *Test:* `TestRungObsAttributesWinningRung` + `TestRungObsReconcilesWithCounters`. Must add the architest tier-map entry + `dos.toml` lanes by hand (new leaf).
2. **vDSO + zero-alloc correctness.** Subscribe to `EvVDSOHit`, count it as `rung=\"vdso\"`; declare `Subscriptions()` so the tap is O(interested). *Test:* `TestRungObsVDSOBucketDistinct` + `TestRungObsZeroAllocOnUnsubscribedKind` (mirrors `registry_events_test.go:91`).
3. **`/metrics` surface.** Wire the snapshot into `renderMetrics` as `fak_kernel_decisions_total{rung,kind,reason}` using a labeled-counter render (the `writeHelpType`+per-row pattern at `internal/gateway/metrics.go:752-760`), HELP text pointing at `fak preflight --explain` for per-call drill-down. *Test:* `TestMetricsRenderRungDecisions` in `internal/gateway` (mirrors `metrics_test.go:255`).
4. **`fak rungstats` CLI (passive read-out).** A command that runs a small fixed probe set through the real fold, registers the observer, and prints the rung distribution table — the offline counterpart of the live `/metrics` row, so the distribution is inspectable without a running gateway. *Test:* `TestCmdRungStatsTable` (golden table).
5. **Docs + honesty boundary.** A short `docs/` note (rung-decision telemetry) explaining the counter is a passive O(interested) tap that re-derives the winner and is NOT on the deny path, and that per-call drill-down is `fak preflight --explain`. *Tool:* none beyond the readme/scorecard gates already in CI.

# Honesty boundary

- **Re-derivation, not ground truth (heuristic-by-construction-but-currently-exact):** Rung A re-folds the chain in the observer rather than reading the rung off the event. Today the global registry chain is the same one the hot path folds (`internal/kernel/kernel.go:148-153` falls back to `abi.AdjudicatorsFor`), so the re-fold is exact. It would diverge only for an injected `WithAdjudicators` chain that the global registry cannot reproduce — that is the Rung-B escalation trigger, explicitly out of scope here.
- **vDSO is opaque to attribution:** a vDSO hit ran no adjudication (`internal/kernel/kernel.go:303-314`), so it can only ever be counted as the single `rung=\"vdso\"` bucket — the counter cannot say *which structural rung would have decided it*. That is correct, not a gap.
- **Out of scope (NOT built here):** carrying the winning-rung identity on the `Event` (the ABI edit, Rung B) is deferred until divergence is shown. This epic does NOT add a new verdict, does NOT make the per-call ladder data-driven/reorderable (the rung ORDER stays hard-coded in `internal/adjudicator/decide.go`), does NOT add chain-level short-circuit, and does NOT touch the all-three shipgate keep-bit. It is pure passive observability over the existing fold.
- **Counter is in-memory, process-local:** like the existing `fak_kernel_*_total` counters it resets on restart and is not persisted; durable rung-decision history is a separate concern.

---

## Epic 8

# feat(adjudicator): policy-manifest `rate_limit` + retry-after on WAIT

## Problem

fak ships a working throughput/cost governor — `internal/ratelimit` is wired into the defconfig (`internal/registrations/registrations.go:53`), registers at rank 8 (`internal/ratelimit/ratelimit.go:266`), and emits `Deny(RATE_LIMITED)` over a cap whose reason the kernel already maps to a `WAIT` disposition (`internal/kernel/kernel.go:482-483`). But the governor is *unreachable from the file an operator actually edits*. It is configured **only** by `FAK_RATELIMIT_*` env vars read once at process start (`internal/ratelimit/ratelimit.go:272-286`) or in-process via `SetLimit` (`ratelimit.go:117-121`). The policy manifest — the declarative `fak-policy/v1` floor that maps 1:1 to `adjudicator.Policy` (`internal/policy/policy.go:46-82`) — has **no `rate_limit` field**; the manifest seam was deliberately deferred (`internal/ratelimit/ratelimit.go:31-34`). So a per-tool/per-tenant cap cannot be expressed, reviewed, or diffed declaratively, and cannot hot-reload the way `--policy` reload already re-pushes IFC config (`cmd/fak/main.go:563-573`).

Second, the `WAIT` the throttle produces is a **bare category token**. `Disposition` returns the string `"WAIT"` (`internal/kernel/kernel.go:482-483`) and `DenyResult` packs only `verdict/reason/disposition/by` (+ an optional witness claim) into the deny-as-value meta (`internal/kernel/kernel.go:462-473`). The loop is told *to* back off but not *for how long* — unlike errno's `EAGAIN`, which the caller pairs with a concrete retry window. The limiter already knows the cap that fired (`cap`/`limit`/`key` in its deny meta, `internal/ratelimit/ratelimit.go:220-231`) but emits no back-off hint.

This is the soft mid-ladder errno-equivalent: a recoverable, self-describing throttle. Both halves default to the smallest rung that establishes the property — an inert limiter still Defers on every call (`ratelimit.go:151-153`), so nothing changes until a cap is declared.

## Current state (confirmed at HEAD)

- `internal/ratelimit/ratelimit.go:31-34` — doc explicitly defers the `fak-policy/v1 rate_limit:` field to a follow-up "so this enforcer lands without touching the policy schema while it is in flight."
- `internal/ratelimit/ratelimit.go:117-121` — `SetLimit(Limit, KeyMode)` is the in-process config seam; takes effect immediately under the same lock as `Adjudicate`.
- `internal/ratelimit/ratelimit.go:151-153` — an unlimited (`MaxCalls==0 && MaxCost==0`) limiter Defers on every call: fail-open, inert until a cap is set.
- `internal/ratelimit/ratelimit.go:178-185, 220-231` — over-cap path emits `Deny(RATE_LIMITED)` with bounded meta `{cap, limit, key}`; check-before-consume means a refused call does not advance the counter.
- `internal/ratelimit/ratelimit.go:272-286` — `configureFromEnv` is the ONLY non-test config path today (`FAK_RATELIMIT_MAX_CALLS / _MAX_COST / _KEY`).
- `internal/registrations/registrations.go:53` — the blank-import that activates the leaf.
- `internal/kernel/kernel.go:478-489` — `Disposition` switch; `ReasonRateLimited`/`ReasonLeaseHeld` → `"WAIT"`, default → `"TERMINAL"`. No duration is attached.
- `internal/kernel/kernel.go:462-473` — `DenyResult` builds the deny-as-value meta (`verdict/reason/disposition/by` + optional `witness`); this is where a retry-after would surface to the loop.
- `internal/abi/reasons.go:19,38` — `ReasonRateLimited` is in the closed 12-reason vocabulary.
- `internal/policy/policy.go:62-82` — `Manifest` struct: the on-disk schema, no `rate_limit` field. `DisallowUnknownFields` (`policy.go:166-174`) means adding the field is a real schema change, not a silently-tolerated key.
- `internal/policy/policy.go:113-120, 245-250` — `Runtime` is the boot-time bundle (`Adjudicator` + IFC `Sources/SafeSinks/AuthorizeRules`); `ToRuntime` builds it.
- `cmd/fak/main.go:563-573` — `reloadPolicy` calls `adjudicator.Default.SetPolicy(rt.Adjudicator)` then `applyRuntime(rt)`; this same function is the hot-reload path (`policyReloader`, `main.go:576-591`).
- `cmd/fak/main.go:766-769` — `applyRuntime` pushes manifest-derived config into singletons: `policy.ApplySources(rt)` and `ifc.ConfigureDefaultPolicy(ifcPolicy(rt))`. This is the EXACT precedent a `ratelimit.Default.SetLimit(...)` call follows.

## Design — graduated rungs, smallest sufficient is the DEFAULT

The property: *"a declared per-key cap is enforced declaratively, and an over-cap call gets an actionable back-off."* It is established at the cheapest rung that can; the limiter is one more rung in the fold that costs nothing until a cap is exceeded.

**DEFAULT rung — declarative inert limiter (rung 1, in-process structural):** Add `RateLimit` to `Manifest`/`Runtime`; `applyRuntime` calls `ratelimit.Default.SetLimit(...)`. With NO `rate_limit:` block the limiter stays unlimited and Defers on every call — byte-for-byte unchanged from today (`ratelimit.go:151-153`). The smallest sufficient rung: a structural in-process Defer that engages only when a *declared* key exceeds its *declared* cap. No witness, no escalation, no model turn.

**Escalation trigger → WAIT-with-retry-after (still rung 1, richer disposition):** when (and only when) a cap is *exceeded*, the deny climbs from a bare `WAIT` token to a `WAIT` carrying a `retry_after` hint in the deny-as-value meta. The limiter computes the hint from the cap it already tracks; `DenyResult` (`kernel.go:462-473`) surfaces it. This is not a new verdict kind and not a core-reason edit — it is additive meta on the existing `WAIT` path, so a loop that ignores it degrades to today's behavior.

**Escalation trigger → rung 7 human (UNCHANGED, out of scope):** repeated throttling is a load-shaping signal, not a per-call escalation. If a runaway loop keeps hitting `WAIT`, the existing consecutive-non-keep breaker (`internal/shipgate/shipgate.go:95-105`) is the human-escalation path; this epic adds NO new escalation route. A throttle stays recoverable by construction — that is the whole point of the WAIT/EAGAIN analogue.

## DOS + Linux analogue

- **DOS:** the manifest `rate_limit` is a declarative refusal knob in the closed vocabulary (`RATE_LIMITED`) — the same shape as `dos_refuse_reasons`: a verifiable, structured refusal an operator authors in a diffable file, not free text.
- **Linux:** `EAGAIN`/`EWOULDBLOCK` from a rate-limited syscall (and `RLIMIT_*` set declaratively in `limits.conf`) — a recoverable "try again," now carrying a concrete back-off the way `Retry-After` does over HTTP 429.

## Acceptance criteria (testable)

1. A `fak-policy/v1` manifest with a `rate_limit:` block parses, validates (unknown key-mode / negative cap fails loud), and round-trips through `FromPolicy`/`ToPolicy` — proven by a `policy` package test.
2. A manifest with NO `rate_limit:` block leaves `ratelimit.Default` inert: every call Defers, no `RATE_LIMITED` ever emitted — proven by an inert-default test.
3. `applyRuntime` (or its successor) installs the declared cap into `ratelimit.Default` via `SetLimit`, and `--policy` hot-reload re-applies a changed cap — proven by a `cmd/fak` (or `policy`) test that loads, mutates, reloads.
4. An over-cap call's `DenyResult` meta carries a non-empty `retry_after` whose value parses as a Go `time.Duration` (or integer ms), and a sub-cap call carries none — proven by a `kernel` test.
5. `Disposition(ReasonRateLimited)` still returns `"WAIT"` (no regression to the closed switch) — existing kernel tests stay green.
6. `make ci` / `scripts/ci.ps1` green; `architest` tier-map + `dos.toml` lanes unchanged (no new leaf package — edits land in existing `ratelimit`/`policy`/`kernel`).

## Child issues (each independently shippable on the trunk)

1. **`feat(policy): add `rate_limit` to the fak-policy/v1 Manifest`** — add `RateLimit *RateLimitRule` to `Manifest` (`policy.go:62-82`), a `RateLimitRule{MaxCalls, MaxCost, Key}` type, validation (exactly-one-of-meaningful, known key-mode), and round-trip in `FromPolicy`. Surface it in `Summary`. Test: `TestManifestRateLimitRoundTrip` + `TestManifestRateLimitRejectsUnknownKeyMode` in `internal/policy`.

2. **`feat(policy): carry rate_limit through Runtime and apply it at boot`** — thread the parsed rule into `Runtime` (`policy.go:113-120`) and call `ratelimit.Default.SetLimit(...)` from `applyRuntime` (`cmd/fak/main.go:766-769`), mirroring `ifc.ConfigureDefaultPolicy`. An empty/absent rule installs the unlimited (inert) Limit. Test: `TestApplyRuntimeInstallsRateCap` (and inert-on-absent) in `cmd/fak` or `internal/policy`.

3. **`feat(ratelimit): emit a retry-after hint on the over-cap deny`** — add the back-off duration to the limiter's deny meta (`ratelimit.go:220-231`), computed from the cap that fired. Pure addition to existing meta. Test: `TestLimiterDenyCarriesRetryAfter` in `internal/ratelimit`.

4. **`feat(kernel): surface retry_after on the WAIT deny-as-value`** — read the limiter's hint in `DenyResult` (`kernel.go:462-473`) and emit `meta["retry_after"]` only for `WAIT`-disposition denies that carry one. `Disposition` unchanged. Test: `TestDenyResultWaitCarriesRetryAfter` + `TestDenyResultNonWaitNoRetryAfter` in `internal/kernel`.

5. **`docs: document the declarative rate_limit knob + WAIT/retry-after contract`** — add the `rate_limit:` block to the policy-manifest reference and the dispatch-loop WAIT doc (the loop SHOULD honor `retry_after`). Doc-only; folds into the freshness/seo lanes by explicit path.

## Honesty boundary

- **Stays heuristic:** the cost proxy is argument byte length, not a real token count (`ratelimit.go:233-253`) unless the caller supplies `Meta["fak.ratelimit.cost"]`. `retry_after` is therefore an *estimate* derived from a cap, not a guaranteed admission time — it is advisory back-off, like HTTP `Retry-After`, not a reservation.
- **Stays as-is (out of scope):** no new verdict kind, no new core reason, no change to the `Disposition` switch's closed set (`kernel.go:478-489`). `WAIT` remains reachable only via `ReasonRateLimited`/`ReasonLeaseHeld`; an out-of-tree reason still falls to `TERMINAL` (the known gap), and this epic does NOT fix that.
- **Unbuilt / explicitly not built here:** windowed/decaying budgets and per-trace TTL eviction stay the lifecycle leaf's job (`ratelimit.go:36-40`, issue #12) — the per-key counter map is still a fixed-ceiling fail-open structure (`ratelimit.go:160-170`). No per-tool *graduated* posture and no observe→enforce promotion ledger; this epic only makes the existing binary cap *declarable*. The loop is *told* a retry-after but nothing in this epic *forces* the loop to honor it — enforcement of back-off remains the loop's responsibility.
- **Not asserted:** there is no `retry_after` field anywhere in the tree today (confirmed: the only `retry_after` string is in `tools/api_host_retry_packet.py:219`, unrelated) — this epic introduces it net-new on the WAIT path.
