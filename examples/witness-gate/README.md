# fak kernel — the generic require-witness gate

**Any adjudicator can lift a single high-stakes call to *require-witness*: the call is
refused unless an independent corroboration backs it. A claim must be *corroborated*, not
*asserted*.** This is the difference between "the model *said* it deployed" and "the
deploy was *independently confirmed* against evidence the model did not author."

```
  preflight ──fold the SAME chain `fak serve` uses──▶  shipgate adjudicator
   (deploy)                                                 │  ship-shaped call?
                                                            │     yes → RequireWitness  ← lift
                                                            ▼  (the capability monitor would ALLOW; the gate overrules it)
                                              kernel Submit path resolves the gate:
                                                 witness CONFIRMED → ALLOW
                                                 witness REFUTED   → DENY  (TRUST_VIOLATION)
                                                 witness ABSTAIN   → DENY  (UNWITNESSED)  ← fail-closed
```

This is the **generic** gate. The ship-release flow ([issue #221](https://github.com/anthony-chaudhary/fak/issues/221))
is one *specialization* of it; this example shows the general pattern any adopter can
apply to their own high-stakes call — a money transfer, an irreversible delete, a
production deploy.

## Run it

```bash
./examples/witness-gate/run.sh                  # build/locate fak, run the preflight verdicts
FAK_BIN=/path/to/fak ./examples/witness-gate/run.sh   # use a prebuilt binary instead of building
```

Pure adjudication — no network, no model, no GPU. `run.sh` folds the real chain over one
tool call three times; the verdicts are deterministic. A full captured run is in
[`EXAMPLE-OUTPUT.md`](EXAMPLE-OUTPUT.md).

Windows users: run the `.sh` launcher from WSL or Git Bash, or invoke `fak preflight`
directly (the commands below work verbatim in PowerShell with a prebuilt `fak.exe`).

## What you see

The capability floor here ([`policy.json`](policy.json)) *allows* a generic high-stakes
tool, `deploy`. On its own the capability monitor would return **ALLOW**. But `deploy` is
a ship-shaped action, so the witness rung lifts it:

```bash
$ fak preflight --policy examples/witness-gate/policy.json --tool deploy --args '{}'
verdict=WITNESS reason=NONE by=shipgate
```

`--explain` shows *why* — the fold takes the most-restrictive verdict, so the witness
rung (rank 40) overrules the monitor's ALLOW (rank 0):

```bash
$ fak preflight --policy examples/witness-gate/policy.json --tool deploy --args '{}' --explain
verdict: WITNESS   by: shipgate
explanation: deploy held pending an independent witness read-back.

decision chain (9 rung(s), most-restrictive wins):
   ...
=> [7] shipgate.ShipAdjudicator   WITNESS   by=shipgate   <- winner (rank 40)
   [8] adjudicator.Adjudicator    ALLOW     by=monitor
```

A tool the floor does *not* sanction never reaches the gate — it is refused at the
capability floor first:

```bash
$ fak preflight --policy examples/witness-gate/policy.json --tool confirm_transfer --args '{}'
verdict=DENY reason=DEFAULT_DENY by=monitor
```

## Two halves: the LIFT (here) and the RESOLUTION (kernel `Submit` path)

Be precise about what each command shows.

`fak preflight` folds the adjudicator chain to a verdict — and a require-witness verdict
folds to **`WITNESS`**: the *lift*. It is the adjudication-time decision that this call
may not be taken on faith. That is the part this demo runs with no server.

**Resolving** that lift — turning `WITNESS` into a concrete `ALLOW` or `DENY` — happens
on the kernel's `Submit` / serve path (`internal/kernel`, `resolveWitness`), where every
registered witness resolver is asked to corroborate the claimed effect against evidence
the agent did not author (git ancestry, a tracked path, a clean tree…). The resolution
is witnessed by the kernel tests rather than a CLI verb:

```bash
$ go test ./internal/kernel -run TestRequireWitness
```

| witness outcome | resolved verdict | reason | meaning |
|---|---|---|---|
| **CONFIRMED** | `ALLOW` | — | the effect was independently corroborated; the call proceeds |
| **REFUTED** | `DENY` | `TRUST_VIOLATION` | the agent's claim is provably false (a lie about the effect) |
| **ABSTAIN** / no resolver | `DENY` | `UNWITNESSED` | no evidence either way → fail-closed |

(Source: `internal/kernel/kernel_witness_test.go` — `TestRequireWitnessConfirmedOpensGate`,
`TestRequireWitnessRefutedStaysClosed`, `TestRequireWitnessAbstainFailsClosed`,
`TestRequireWitnessNoResolverPreservesV01`.)

## The four points

**1. The structural property — corroborate, not assert.** A `require-witness` verdict is
the difference between "the model *said* it did X" and "X was *independently corroborated*."
The witness resolver (`internal/witness`) re-checks the claimed effect against evidence the
agent did not author — `git merge-base --is-ancestor` for "the phase shipped", `git ls-files`
for "the file was added", `git status --porcelain` for "the tree is clean". The kernel
refuses the uncorroborated case; the corroborated case still has to pass every *other*
floor (the capability monitor, the egress rung, the self-modify globs).

**2. Relation to the ship-gate (#221).** The ship-gate is one specialization — the
shipgate adjudicator (`internal/shipgate`) lifts a fixed set of ship/release actions
(`ship`, `release`, `publish`, `deploy`, `ship_release`). This is the general pattern:
*any* call your floor allows can be lifted to require-witness by an adjudicator that
returns `VerdictRequireWitness` carrying the claimed effect. An adopter writes one small
adjudicator and registers it; the kernel's witness fold does the rest.

**3. `UNWITNESSED` is closed-vocabulary.** The deny carries a *named* reason from the
closed refusal vocabulary (see [`POLICY.md`](../../POLICY.md) and `internal/abi/reasons.go`):
`UNWITNESSED` for "no corroboration", `TRUST_VIOLATION` for a *refuted* claim. Because the
reason is a closed token and not free text, a deny-loopback can derive a disposition from
it — `TRUST_VIOLATION` derives `ESCALATE` (a provable lie is not the model's to retry),
while a bare `UNWITNESSED` (no evidence yet) is `TERMINAL` for this call. The disposition
mapping is `kernel.Disposition`.

**4. Honest scope.** The witness corroborates a *claim* against an independent source — it
checks that the claimed effect is **present**, not that it is **correct**. "The commit
exists" and "the tree is clean" are evidence the effect happened; they do not certify the
code is right. Correctness still has to be tested separately (the execution witness,
`exec:<json>`, runs fail-to-pass / pass-to-pass selectors for exactly that — a stronger,
opt-in rung). The gate raises the trust floor from *self-report* to *corroborated effect*;
it is not a proof of correctness.

## Files

| file | what it is |
|---|---|
| `run.sh` | one-command launcher: build/locate `fak`, run the three preflight verdicts |
| `policy.json` | the capability floor — allows the generic high-stakes `deploy` tool |
| `EXAMPLE-OUTPUT.md` | a captured run |

Related: [`CLAIMS.md`](../../CLAIMS.md) #75 (the effect-verifying witness gate);
[`internal/witness/`](../../internal/witness/) (the `dos_verify`-style resolver);
[`internal/shipgate/`](../../internal/shipgate/) (the ship-release specialization, #221);
[`POLICY.md`](../../POLICY.md) (the closed refusal vocabulary, including `UNWITNESSED`);
[`../dev-agent-policy.json`](../dev-agent-policy.json) (a floor whose `ship_release` is
witness-gated this way).
