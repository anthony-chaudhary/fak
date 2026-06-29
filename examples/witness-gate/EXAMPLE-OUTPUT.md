# Captured run — `examples/witness-gate`

A real run of [`run.sh`](run.sh) (here driven against a prebuilt `fak` binary on
Windows; the verdicts are produced by the in-process adjudicator chain, so they are
deterministic regardless of host). The `[witness-gate]` lines are the launcher's
narration on stderr; the rest is `fak preflight` output.

## 1. A generic high-stakes call is lifted to require-witness

```
$ fak preflight --policy examples/witness-gate/policy.json --tool deploy --args '{}'
fak: loaded capability floor from examples/witness-gate/policy.json
verdict=WITNESS reason=NONE by=shipgate
```

The policy *allows* `deploy` — the capability monitor returns ALLOW — yet the verdict
is `WITNESS` (`by=shipgate`). The witness rung overruled a permissive monitor and held
the call pending an independent read-back.

## 2. The decision trace makes the overrule explicit

```
$ fak preflight --policy examples/witness-gate/policy.json --tool deploy --args '{}' --explain
fak: loaded capability floor from examples/witness-gate/policy.json
tool: deploy   args: 2 bytes (sha 44136fa355b3)
verdict: WITNESS   by: shipgate
explanation: deploy held pending an independent witness read-back.

decision chain (9 rung(s), most-restrictive wins):
   [0] grammar.Rung               DEFER     by=grammar
   [1] ratelimit.Limiter          DEFER     by=ratelimit
   [2] preflight.Ladder           DEFER     by=preflight
   [3] engine.residencyGate       DEFER     by=engine-residency
   [4] plancfi.Adjudicator        DEFER     by=plancfi
   [5] ifc.SinkGate               DEFER     by=ifc-sink
   [6] gitgate.GitGate            DEFER     by=gitgate
=> [7] shipgate.ShipAdjudicator   WITNESS   by=shipgate   <- winner (rank 40)
   [8] adjudicator.Adjudicator    ALLOW     by=monitor
```

Rung `[8]` (the capability monitor) returns ALLOW; rung `[7]` returns WITNESS. The fold
takes the most-restrictive verdict, so WITNESS (rank 40) wins over ALLOW (rank 0): the
gate refuses to take the allow on faith.

## 3. An unsanctioned tool never reaches the gate

```
$ fak preflight --policy examples/witness-gate/policy.json --tool confirm_transfer --args '{}'
fak: loaded capability floor from examples/witness-gate/policy.json
verdict=DENY reason=DEFAULT_DENY by=monitor
```

`confirm_transfer` is not on the allow-list, so the capability floor refuses it with
`DEFAULT_DENY` (rank 100) before the witness rung is even consulted. The require-witness
gate is for calls the floor *would otherwise allow* — it adds corroboration on top of
the floor, it does not replace it.

## The resolution half (kernel `Submit` / serve path)

`fak preflight` folds the chain to the **lift** (`WITNESS`). Turning that into a
resolved verdict — `ALLOW` when corroborated, `DENY/UNWITNESSED` when not — runs on the
kernel's `Submit` path. It is witnessed by the kernel tests:

```
$ go test ./internal/kernel -run TestRequireWitness -v
=== RUN   TestRequireWitnessConfirmedOpensGate      # CONFIRMED  -> Allow
=== RUN   TestRequireWitnessRefutedStaysClosed      # REFUTED    -> Deny/TRUST_VIOLATION
=== RUN   TestRequireWitnessAbstainFailsClosed      # ABSTAIN    -> Deny/UNWITNESSED
=== RUN   TestRequireWitnessNoResolverPreservesV01  # no witness -> Deny (fail-closed)
PASS
```
