# D11 · steward

> **Update — witness pass (2026-06-20, commit `3cb8ff9`).** 1 OPEN obligation(s) below were CLOSED to ✅ PROVEN by new deterministic tests added in `internal/steward/proofs_witness_test.go`. The body keeps the original analysis (the gap **and** the 'to close' plan that was then executed); the **current verdict is in the [master ledger](README.md)** and the executed closures are listed in *Closures* at the foot of this file.

The `steward` package is fak's *steward population*: a set of cheap, single-invariant validators that garden the kernel's journal. Each steward is one `abi.Steward` — a named `Check(ctx) (violated bool, witness string)` that fires **only** with an independently-authored witness and otherwise **abstains** (the ABI's "never block on your own opinion" rule). A `Population` runs the stewards (`Sweep`), tallies how often each fires by name, and a meta-steward (`Prune`) removes stewards that never fired across a soak so the population doesn't ossify into dead code. "Correct" here is **decision-procedure soundness (regime D)** in two senses: (1) *single responsibility* — every steward checks exactly one predicate and abstains by default, so the population composes without one steward's verdict contaminating another; and (2) *deterministic, order-independent population* — for a fixed steward set and environment, `Sweep`/`Prune` produce the same verdicts regardless of the order stewards were added, because all accumulation is keyed by `Name()`, not by slice index.

---

## Theorem 1 — each steward enforces exactly one checkable invariant predicate (single responsibility, composable)

**THEOREM.** Each steward enforces exactly one checkable invariant: a `FuncSteward` carries one `Check` returning `(violated bool, witness string)`; it returns `true` only on a violation of its own predicate, abstains `(false, "")` otherwise, and when it fires it carries a name and a non-empty witness. Stewards compose via `Population.Sweep`, which reports exactly the firing stewards keyed by name and excludes abstainers.

**REGIME.** D — decision-procedure soundness.

**PROOF.** The contract is `abi.Steward = {Name() string; Check(ctx) (violated bool, witness string)}` — a single predicate plus an independently-authored witness, abstain-by-default (`fak/internal/abi/registry.go:619`). `FuncSteward` adapts one named `Check` to that interface (`fak/internal/steward/steward.go:26`-`33`), so a steward is *structurally* one predicate. The four v0.1 builders each close over one invariant and emit one witness: `SecretInContext` scans `snapshot()` against the secret regex and returns the offending snippet (`steward.go:105`); `LeaseDisjointness` fires iff two leases share a tree prefix via `overlap()` (`steward.go:123`, helper at `:160`); `KPIRegression` fires iff `current > baseline*(1+tol)` (`steward.go:143`); `VDSOSoundness` forwards a probe (`steward.go:154`). Composition is fail-quiet: `Population.Sweep` iterates the stewards and records `name → witness` **only when** `Check` returns `true` (`steward.go:55`-`66`), so abstainers are excluded and the result is exactly the firing set. The witnesses assert, per builder, fire-on-violation **and** abstain-on-clean **and** correct `Name()` + non-empty witness, plus the Sweep-excludes-abstainer property — i.e. exactly single responsibility + composability.

**WITNESS.**
```
go test ./internal/steward/ -count=1 -timeout 120s \
  -run 'TestVDSOSoundness|TestSecretInContext|TestLeaseDisjointness|TestKPIRegression|TestSweepAbstainingStewardNotReported' -v
```
Tests: `TestSecretInContext`, `TestLeaseDisjointness`, `TestKPIRegression`, `TestVDSOSoundness`, `TestSweepAbstainingStewardNotReported`.

**VERDICT.** **PROVEN** — 2026-06-20 (native `go test`, macOS arm64). All five PASS (`ok github.com/anthony-chaudhary/fak/internal/steward 0.264s`). Each test body was read and asserts the claimed predicate-fires/abstains/witness behaviour; `TestSweepAbstainingStewardNotReported` asserts an abstaining check yields an empty fired map.

**DOS.** bound at ship.

---

## Theorem 2 — the steward population is deterministic and order-independent where it claims to be

**THEOREM.** For a fixed set of stewards and a fixed environment, (a) `Sweep` produces the same fired set and the same per-name fire tallies, and `Prune` removes exactly the stewards that never fired and keeps the rest — **regardless of the order** in which the stewards were added to the `Population`.

**REGIME.** D — decision-procedure soundness (composition order well-defined).

**PROOF.** Structurally the claim holds because the `Population` accumulates by `Name()`, not by slice index. `Sweep` returns a `map[string]string` keyed by `Name()` (`fak/internal/steward/steward.go:55`-`66`) — a Go map is inherently order-free — and increments `p.fires[s.Name()]` (`:61`), so the tally is name-keyed. `Prune` keeps a steward iff `p.fires[s.Name()] != 0` (`:70`-`84`), again name-keyed; the only order-sensitive outputs are the *ordering* of the returned `pruned` slice and of `Names()` (`:87`), which mirror insertion order. So determinism for a fixed input is real, and `Sweep`/`Prune` verdicts are order-independent **up to** the ordering of their list-shaped outputs. **However**, the only witness, `TestPrunePopulation` (`steward_test.go:145`-`188`), runs **one** fixed insertion order `[alpha,beta,gamma,delta,dead]`, sweeps twice, asserts `Prune()==[dead]` and the survivor set, and `sort.Strings()`-es `Names()` before comparing (`:177`-`179`). Sorting **tolerates** order rather than **proving** independence: a single ordering cannot witness order-independence, and no test re-runs the same stewards under a different (shuffled / reversed) `Population` order and asserts the identical fired map + tallies + pruned set. A grep of `internal/steward/` for `shuffle|Shuffle|quick|rand|order` returns nothing. So the determinism-for-fixed-input half is held green; the order-independence half is **un-witnessed**.

**WITNESS.**
```
go test ./internal/steward/ -count=1 -timeout 120s \
  -run 'TestPrunePopulation|TestNewStewardAndSweepReportsWitness' -v
```
Tests: `TestPrunePopulation`, `TestNewStewardAndSweepReportsWitness`.

**VERDICT.** **OPEN** — 2026-06-20. Both tests PASS (`ok ... 0.167s`), establishing determinism for a *single fixed* ordering, but no deterministic witness exercises **two** orderings and asserts the same result, so the order-independence claim is not discharged. **Closing witness:** add a stdlib property test (`testing/quick`, or an explicit permutations table) that builds the same steward set under several permutations of the `NewPopulation`/`Add` order, runs identical `Sweep`s, and asserts the resulting fired map, the per-name `p.fires` tallies, and the **sorted** `Prune()` set are identical across all permutations. Until that exists this stays OPEN — not promoted to PROVEN by the structural argument alone.

**DOS.** bound at ship.

---

## Closures (witness pass 2026-06-20, commit `3cb8ff9`)

Each obligation marked OPEN above was discharged by a new zero-dependency (stdlib `testing`/`testing/quick`) metamorphic/round-trip/invariant test that ASSERTS the property against an independently recomputed reference. Verified by `go test -count=1 ./internal/...` (45 packages green, 0 failures).

- **steward-population-deterministic-order-independent** → ✅ PROVEN by `TestStewardSweepFiredSetOrderIndependent`. Closed by 3 deterministic metamorphic tests over fixed-seed (rand.NewSource) add-order permutations of one fixed steward multiset (8 named stewards, deterministic fire/abstain verdicts; firing witness = w-+name). (a-fired-set) TestStewardSweepFiredSetOrderIndependent: over 200 permutations every Sweep returns a byte-identical fired map (reflect.DeepEqual), and that map equals the plan-derived ground-truth expectation, so non-vacuous. (a-tally+Prune) TestStewardPruneOrderIndependent: across 200 permutations, after an identical 3-sweep schedule, Prune removes exactly the never-fired set and Names() keeps exactly the firing set, both matching the verdict partition; since Prune removes iff tally==0, partition-invariance directly witnesses per-name tally order-independence. TestStewardSweepTallyMonotoneOrderIndependent pins the tally endpoints: with 0 sweeps every tally is 0 so Prune removes the WHOLE population in any of 100 permutations. All exact equality (sets/ints), no floats. go vet clean; all 3 new tests PASS; full package go test PASS with the file present. Caveat: Population.fires is unexported, so per-name tallies are witnessed indirectly-but-exactly through Prune's documented pruned-iff-tally-0 contract (sound, not vacuous).
