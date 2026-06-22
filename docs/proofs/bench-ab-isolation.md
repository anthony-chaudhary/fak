---
title: "fak proof: A/B benchmark paired-replay isolation"
description: "Proof that fak's turn-tax A/B benchmark is a paired replay where only the toggled lever differs, so its measured delta is attributable and real."
---

# I2 · bench-ab-isolation

**Regime A (algebraic / structural).** The `bench` and `turnbench` packages run fak's
turn-tax A/B benchmark: they replay one frozen tool-call trace through the kernel twice,
toggling exactly one variable (the vDSO fast-path on/off, or a shared-vs-isolated world
epoch for the fleet/fan-out sweeps), and report the difference in *turns saved* as the
value the toggled lever buys. "Correct" for this module is **not** a numerical-accuracy
claim; it is a **measurement-hygiene** claim with two parts: (1) the two arms are a
*paired replay* — same seed and same trace, every arm built from a freshly-reset kernel,
so the **only** input that differs between arms is the toggled variable, and equal inputs
give a byte-identical result; and (2) the measured A/B delta is **attributable** — it
equals the toggled axis's own independently-counted kernel events exactly, and a
zero-perturbation run saves exactly 0, so the benchmark is reading real avoided work and
never a fixed per-call discount dressed up as a result. Both theorems are discharged by
deterministic tests run green here.

---

## Theorem 1 — an A/B ablation is a PAIRED replay; only the toggled variable differs

**THEOREM.** Each arm of the ablation replays the *identical* call sequence through one
freshly-reset kernel; the only thing that differs between arms is the toggled variable
(vDSO on/off, or the shared/isolated world epoch). Therefore a fixed
`(profile, N, subTurns, trials, seed)` yields a **byte-identical** result, and flipping
the toggle provably swaps the kernel path (on-arm `VDSOHits > 0`, off-arm `VDSOHits == 0`).

**REGIME.** A — structural / determinism invariant.

**PROOF.** The paired-replay invariant is `turnbench.replay`
(`fak/internal/turnbench/turnbench.go:431`): each arm is an *isolated session* that resets
the two pieces of process-global cross-call state at its start —
`vdso.Default.BumpWorld()` (`turnbench.go:438`) and `ifc.Default.Reset("")`
(`turnbench.go:439`) — then builds a fresh `kernel.New` and sets **only** the toggled flag
via `k.SetVDSO(vdsoOn)` (`turnbench.go:442`) before driving the same `t.Calls` loop
(`turnbench.go:452`). `Run`/`RunWithCalls` invoke `replay(…, true, …)` and
`replay(…, false, …)` on the *same* `*Trace` (`turnbench.go:523`-`528`), so the trace,
seeds, and engine are held fixed and only `vdsoOn` varies — a genuine paired replay. The
`bench` package's `RunArm` is the same discipline at the CLI seam. Reproducibility under a
fixed seed is structural: `RunFanoutCell` derives per-trial seeds deterministically from
`(seed, N, subTurns)` (`fak/internal/turnbench/fanout.go:323`) and `RunStochastic` is
seed-driven, so equal inputs give equal outputs.

**WITNESS.**
```
go test ./internal/bench/ ./internal/turnbench/ -count=1 -timeout 180s \
  -run 'TestFanoutDeterministic|TestStochastic_Determinism|TestRunArm_VDSOAblationChangesPath|TestParity' -v
```
`TestFanoutDeterministic` (`fanout_test.go:11`) — two `RunFanoutCell` calls with identical
`(FanoutResearch, 8, 4, 8, 0xF1EE, cm)` produce identical dedup distributions and
projection. `TestStochastic_Determinism` (`stochastic_test.go:52`) — `reflect.DeepEqual`
on two same-seed runs, and a different seed differs.
`TestRunArm_VDSOAblationChangesPath` (`bench_test.go:54`) — on-arm `VDSOHits > 0`, off-arm
`VDSOHits == 0`, `on > off`: the toggle is the only changed input and it swaps the path.
`TestParity_*` witness the A/B *card* structure (matched fak-vs-baseline arms of the same
model) and the oracle grading.

**VERDICT.** **PROVEN** — 2026-06-20, go1.26 darwin/arm64. `TestFanoutDeterministic` PASS
(0.11s), `TestStochastic_Determinism` PASS (0.20s),
`TestRunArm_VDSOAblationChangesPath` PASS (0.13s), `TestParity_*` PASS.

**DOS.** bound at ship.

---

## Theorem 2 — the turn-tax delta is attributed to the toggled axis, not noise

**THEOREM.** The isolation attributes the A/B delta to the **toggled axis**: the
on-minus-off turns-saved delta equals the live kernel counter for that axis **exactly**
(`Net.TurnsSaved − VDSOOffNet.TurnsSaved == VDSOHits`), and a zero-perturbation /
clean-happy-path run saves **exactly 0** — there is no fixed per-call discount that would
inflate the delta independent of real avoided work.

**REGIME.** A — conservation / attribution invariant.

**PROOF.** Attribution is exact by construction in `turnbench.Run`
(`fak/internal/turnbench/turnbench.go:513`): `Net = netFor(onClass.turnsSaved(), cm)` and
`VDSOOffNet = netFor(offClass.turnsSaved(), cm)` (`turnbench.go:548`-`550`), where
`onClass`/`offClass` come from the same paired replay differing only in `k.SetVDSO`. Both
arms keep the grammar (TRANSFORM) lever, so the only work the vDSO toggle removes from the
off arm is the tier-1/2/3 vDSO serves — hence `Net − VDSOOffNet` must equal exactly the
vDSO serve count, which is the live `k.Counters().VDSOHits` the kernel itself tallied
(`turnbench.go:490`). The no-noise-floor half rests on `turnsSaved` being a *sum of real
classified events* — `Grammar + vdsoTotal` (`turnbench.go:194`) with **no** per-call
constant — so a trace with no alias/dup/poison classifies every call as the control
`pass` (`turnbench.go:185`-`187`) and saves 0.

**WITNESS.**
```
go test ./internal/turnbench/ -count=1 -timeout 180s \
  -run 'TestRun_VDSOAblationIsARealPathSwap|TestRun_HappyPathSavesNothing|TestStochastic_BaseSavesNothing|TestStochastic_ZeroRateP50IsZero|TestStochastic_Monotonicity' -v
```
`TestRun_VDSOAblationIsARealPathSwap` (`turnbench_test.go:132`) asserts
`leverDelta := Net.TurnsSaved − VDSOOffNet.TurnsSaved == int(Counters.VDSOHits)` exactly —
the whole delta is accounted for by the toggled axis's own live count, leaving no
unattributed residue. `TestRun_HappyPathSavesNothing` (`turnbench_test.go:168`) and
`TestStochastic_BaseSavesNothing` (`stochastic_test.go:33`) pin a clean trace to
`Net.TurnsSaved == 0` with tokens/$/latency all 0. `TestStochastic_ZeroRateP50IsZero`
(`stochastic_test.go:100`) collapses the whole zero-rate distribution to 0
(`p50 == max == mean == 0`). `TestStochastic_Monotonicity` (`stochastic_test.go:70`)
confirms the delta is dose-responsive to the toggled perturbation rate (strictly higher
rate ⇒ strictly higher median saved).

**VERDICT.** **PROVEN** — 2026-06-20, go1.26 darwin/arm64.
`TestRun_VDSOAblationIsARealPathSwap` PASS (0.07s), `TestRun_HappyPathSavesNothing` PASS
(0.02s), `TestStochastic_BaseSavesNothing` PASS (0.09s),
`TestStochastic_ZeroRateP50IsZero` PASS (0.17s), `TestStochastic_Monotonicity` PASS
(0.64s).

**DOS.** bound at ship.

---

### Notes on scope and honesty

- The prompt named `TestParity` as a witness for Theorem 1. After reading its body, the
  `TestParity_*` family asserts the **card/verdict grading** of matched A/B arms (fak vs
  baseline of the same model, oracle-graded), not the seed-determinism of a paired replay.
  It corroborates the *A/B card structure* but the seed→byte-identical-result property is
  carried directly by `TestFanoutDeterministic` and `TestStochastic_Determinism`, and the
  toggle-swaps-the-path property by `TestRunArm_VDSOAblationChangesPath` /
  `TestRun_VDSOAblationIsARealPathSwap`. The doc credits each test to what its body
  actually checks.
- These witnesses are pure-Go, zero-dependency, no weights, no oracle cache — they run
  green natively on this macOS node and through WSL on the Windows host. Neither theorem is
  SCOPED-OUT.
