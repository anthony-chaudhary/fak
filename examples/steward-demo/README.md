# fak kernel — single-invariant steward demo

**A steward is a single-invariant runtime checker that fires only with an
*independently-authored witness* — not the model's own claim.** It is the structural
answer to "how do you trust a runtime alert?": you don't trust the alert, you trust the
*witness chain* behind it. A steward's check returns `(violated, witness)` and raises
**only when both** (a) its one-sentence invariant pattern matches **and** (b) the witness
is non-empty — and a witness is produced by an *independent* scan/replay/registry, never
by the thing under test.

```
  a steward fires  ⟺  (a) invariant pattern matches   AND   (b) an INDEPENDENT witness corroborates
                        e.g. "sk-…" in the context           e.g. a separate regex scanner
                                                              authored the witness string
                        ── (a) alone is a self-accusation the model cannot make → ABSTAIN (suppressed) ──
```

This demo drives a small frozen scenario through a faithful, dependency-free re-enactment
of fak's real steward package (`internal/steward/steward.go`) and shows three things:

- a steward **RAISING** with an independent witness — a real alert;
- the **same apparent condition without a witness — SUPPRESSED** (the model can't
  self-accuse): the load-bearing witness-vs-no-witness distinction; and
- the **meta-steward pruning** a steward that never fires across a soak — dead-code
  detection on the invariant layer itself, so the population can't ossify.

## Run it

```bash
./examples/steward-demo/run.sh             # run the demo
./examples/steward-demo/run.sh --no-color  # plain output
```

Requires only **Python 3** (stdlib). No model, no network, no Go toolchain. It is
deterministic — the same `sample-steward.json` always yields the same verdicts. Exit code
is `0` when every steward matched the witness model and the meta-steward pruned exactly the
never-firing steward; CI-usable.

Windows users: run the `.sh` launcher from WSL or Git Bash, or call the script directly
(`python examples\steward-demo\demo.py --no-color`).

## What you see

> **Reading the output:** a `✓` means *the verdict matched the witness model* — so a `✓` on
> a `RAISE` means the steward correctly fired (a witness was present), and a `✓` on an
> `ABSTAIN` means it correctly stayed quiet (either the pattern didn't trip, or it tripped
> but no independent witness corroborated it).

```
STEWARDS  — one invariant each; raise needs an independently-authored witness
  no-secret-in-context  invariant: no secret-shaped bytes are ever admitted to the model's context window
      ✓ secret admitted, independently scanned -> RAISE            RAISE    witness="…sk-abcde…" by secretscan
      ✓ same pattern, NO independent scan authored a witness -> SUPPRESSED ABSTAIN  pattern matched but NO independent witness — self-accusation suppressed
      ✓ clean context, nothing to scan -> ABSTAIN                  ABSTAIN  invariant pattern did not trip
  …
META-STEWARD  — after a 3-sweep soak, prune any steward that NEVER fired
      keep  no-secret-in-context   fired 3× during the soak
      …
      prune clock-monotonic        fired 0× — guards a clock-skew invariant the platform made impossible …

summary: steward demo passed  ·  4 raises (each with an independent witness)  ·  1 un-witnessed match suppressed  ·  1 never-firing steward pruned
```

Full captured run: [`EXAMPLE-OUTPUT.md`](EXAMPLE-OUTPUT.md).

## The four shipped stewards (one invariant each)

| steward | the one-sentence invariant | the independent witness author |
|---|---|---|
| **secret-in-context** | no secret-shaped bytes are ever admitted to the model's context | a separate regex scanner over the live context snapshot (not the model's say-so) |
| **lease-disjointness** | no two live file-tree leases share a path prefix | the arbiter's lease-registry replay (not either worker) |
| **kpi-regression** | the current p50 never regresses past the committed baseline by more than `tol` | the scoreboard reading the *committed* baseline (not the run reporting its own number) |
| **vdso-soundness** | a cached-path result is bit-identical to a fresh pure recompute | a replayer that recomputes the same pure call side-by-side |

Each maps directly to a builder in [`../../internal/steward/steward.go`](../../internal/steward/steward.go):
`SecretInContext`, `LeaseDisjointness`, `KPIRegression`, `VDSOSoundness`.

## What a steward is NOT

A steward is **not a free-form monitor**. The design is *composability over generality*:

- **The invariant is one sentence.** A steward checks exactly one property. You compose a
  population of narrow checks; you do not write one clever check that "watches everything."
  A broad monitor's failure mode is silent drift — it is hard to say what it would even
  miss. A single-invariant steward's scope is auditable by reading one sentence.
- **The witness is structured, not prose.** The raise carries a concrete witness produced
  by an independent authority. There is no "the model thinks this looks bad." If no
  independent party authored a witness, the steward abstains — a pattern match alone is a
  self-accusation, and **the thing under test is never trusted to accuse itself.**
- **A never-firing steward is dead weight, and gets pruned.** The meta-steward
  (`Population.Prune`) removes any steward that never fired across a soak, so the invariant
  layer can't ossify into checks that can no longer trip.

## Why the witness chain is the trust anchor

A *runtime alert* is only as trustworthy as whoever raised it. If the agent under
observation could raise (or suppress) its own alert, the alert proves nothing — a
compromised or confused agent would simply not alarm. So a steward **refuses to fire on its
own opinion**: it reports a violation *only* with an independently-authored witness, else it
abstains (see the package doc on `internal/steward/steward.go:1`). That asymmetry is the
load-bearing property this demo shows — the `RAISE` and the `SUPPRESSED` rows in part 1
differ in exactly one thing: whether an *independent* scan authored the witness. The
invariant pattern is identical.

## The authoritative witness: the Go tests (units 87–92)

This demo is a runnable, dependency-free *re-enactment*; the **authoritative witness** that
the shipped feature behaves this way is the Go test suite. To run it directly:

```bash
go test ./internal/steward/ -v -run \
  'TestSecretInContext|TestLeaseDisjointness|TestKPIRegression|TestVDSOSoundness|TestPrunePopulation'
```

| unit | test | what it proves |
|---|---|---|
| 87 | `TestNewStewardAndSweepReportsWitness` | a firing steward is reported by `Sweep` **with its witness** |
| 88 | `TestVDSOSoundness` | vdso-soundness fires on a probe mismatch; abstains on a match |
| 89 | `TestSecretInContext` | secret-in-context fires on a secret-shaped token; abstains on clean bytes |
| 90 | `TestLeaseDisjointness` | lease-disjointness fires on a shared tree prefix; abstains when disjoint |
| 91 | `TestKPIRegression` | kpi-regression fires when current regresses past `baseline*(1+tol)`; abstains when steady |
| 92 | `TestPrunePopulation` | the meta-steward prunes **exactly** the steward that never fired |

A companion proof in [`../../internal/steward/proofs_witness_test.go`](../../internal/steward/proofs_witness_test.go)
shows `Sweep`/`Prune` are order-independent: the fired set and the kept/pruned partition do
not depend on the order stewards were added.

## Files

| file | what it is |
|---|---|
| `run.sh` | one-command launcher (Python 3, stdlib only) |
| `demo.py` | a faithful re-enactment of `internal/steward` over the fixture; CI-usable exit code |
| `sample-steward.json` | the bundled steward definitions + witnesses + a never-fire fixture for the meta-steward |
| `EXAMPLE-OUTPUT.md` | a captured run |

## Cross-references

- **`CLAIMS.md`** — "Stewards + RSI ship-gate": *Single-invariant stewards (secret-in-context,
  lease-disjointness, kpi-regression, vdso-soundness) that fire only with an
  independently-authored witness; a meta-steward prunes never-firing stewards. Witness:
  `steward` tests (units 87–92).*
- **`internal/steward/`** — the shipped package: [`steward.go`](../../internal/steward/steward.go)
  (the four builders + `Population.Sweep`/`Prune`), [`steward_test.go`](../../internal/steward/steward_test.go)
  (units 87–92), [`proofs_witness_test.go`](../../internal/steward/proofs_witness_test.go) (order-independence).
- **Distinct from the *adjudication-time* floors** — `../adjudication-demo` (call-side
  capability gate) and `../quarantine-demo` (result-side containment) refuse at decision
  time; stewards are *runtime invariant* checkers that need independent authorship to fire.
