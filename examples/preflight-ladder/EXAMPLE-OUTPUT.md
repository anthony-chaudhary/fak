# Example output

A captured run of the five `fak preflight` witnesses in `run.sh` (fak `0.34.0`, default
built-in capability floor — the tau2 airline-demo tools). No model, no network, no GPU; the
verdicts are the kernel's, and every line reproduces with the one command shown above it.
Reproduce the whole walk: `./examples/preflight-ladder/run.sh` (add `--explain` for the
per-rung trace).

## The five witnesses (default one-line verdicts)

```
$ fak preflight --tool search_flights --args '{"origin":"SFO","destination":"JFK"}'
verdict=ALLOW reason=NONE by=monitor
    # 1. clean call — rungs 0+1 pass, preflight defers, the monitor admits it

$ fak preflight --tool search_flights --args '{"origin":"SFO",}'
verdict=DENY reason=MALFORMED by=preflight
    # 2. malformed JSON (trailing comma) — caught at RUNG-0 (static parse), by=preflight

$ fak preflight --tool search_flights --args '{origin:SFO}'
verdict=DENY reason=MALFORMED by=preflight
    # 3. unparseable args (unquoted) — also RUNG-0; the args are not valid JSON

$ fak preflight --tool delete_everything --args '{}'
verdict=DENY reason=DEFAULT_DENY by=monitor
    # 4. unknown tool — args ARE valid JSON, so rungs 0/1 defer; the MONITOR fail-closes
    #    on an unsanctioned tool (capability floor, a different layer from pre-flight)

$ fak preflight --tool search_flights --args '{"origin":"SFO"}'
verdict=ALLOW reason=NONE by=monitor
    # 5. unknown grammar — the grammar rung FAILS OPEN (defers), the args parse, the
    #    monitor decides. Pre-flight never invents a verdict for a call it can't reason about.
```

## Which rung fired — the `--explain` trace

`--explain` folds the same call to the same verdict but prints the full decision chain, so
you can see exactly which rung won and that the cheap refusal short-circuits the rest.

### Step 2 — the rung-0 catch (malformed args)

The pre-flight ladder is the winner at index `[2]`; every rung after it is `ELIDED` because a
max-rank verdict already decided the call. No model turn, no tool, no monitor ever ran.

```
$ fak preflight --tool search_flights --args '{"origin":"SFO",}' --explain
tool: search_flights   args: 17 bytes (sha 2c3ccb28820a)
verdict: DENY   reason: MALFORMED   by: preflight   disposition: RETRYABLE
explanation: search_flights denied by preflight: MALFORMED (RETRYABLE).

decision chain (9 rung(s), most-restrictive wins):
   [0] grammar.Rung               DEFER     by=grammar
   [1] ratelimit.Limiter          DEFER     by=ratelimit
=> [2] preflight.Ladder           DENY      MALFORMED by=preflight   <- winner (rank 100)
   [3] engine.residencyGate       ELIDED      (elided: a max-rank verdict decided before this rung)
   [4] plancfi.Adjudicator        ELIDED      (elided: a max-rank verdict decided before this rung)
   [5] ifc.SinkGate               ELIDED      (elided: a max-rank verdict decided before this rung)
   [6] gitgate.GitGate            ELIDED      (elided: a max-rank verdict decided before this rung)
   [7] shipgate.ShipAdjudicator   ELIDED      (elided: a max-rank verdict decided before this rung)
   [8] adjudicator.Adjudicator    ELIDED      (elided: a max-rank verdict decided before this rung)
```

### Step 1 — escalate-on-pass (clean call)

Rungs 0 and 1 pass, so `preflight.Ladder` **defers** and the call escalates all the way to
the authoritative monitor at index `[8]`, which admits it.

```
$ fak preflight --tool search_flights --args '{"origin":"SFO","destination":"JFK"}' --explain
tool: search_flights   args: 36 bytes (sha 45404a66891c)
verdict: ALLOW   by: monitor
explanation: search_flights allowed: an affirmative policy rung permitted it.

decision chain (9 rung(s), most-restrictive wins):
   [0] grammar.Rung               DEFER     by=grammar
   [1] ratelimit.Limiter          DEFER     by=ratelimit
   [2] preflight.Ladder           DEFER     by=preflight
   [3] engine.residencyGate       DEFER     by=engine-residency
   [4] plancfi.Adjudicator        DEFER     by=plancfi
   [5] ifc.SinkGate               DEFER     by=ifc-sink
   [6] gitgate.GitGate            DEFER     by=gitgate
   [7] shipgate.ShipAdjudicator   DEFER     by=shipgate
=> [8] adjudicator.Adjudicator    ALLOW     by=monitor   <- winner (rank 0)
```

## Rung 1 — proven in the unit tests, not in the standalone witness

The standalone `fak preflight` default floor installs no per-tool schema (the airline-demo
schemas are registered by the agent loop's `agent.Configure()`, which the witness verb does
not run), so rung 1 **defers** in the run above rather than producing a live DENY. The rung
is real and its catch is proven directly:

```
$ go test ./internal/preflight -run 'Rung1|Rung0'
ok  	github.com/anthony-chaudhary/fak/internal/preflight

  TestRung1SchemaTypeCheck          {"origin":123}, required origin:string => DENY/MALFORMED at rung 1
  TestRung1MissingRequiredFieldDenied  {"dest":"JFK"} missing required origin => DENY/MALFORMED at rung 1
  TestRung0FailureNeverReachesRung1   "{bad" short-circuits at rung 0, never reaching rung 1
```

The label-harvest shape of each catch (`rung_passed`/`rung_failed`, `verdict:"deny"`, `reason`)
is asserted by `TestNegativesRowFields` and `TestCatchRateMix` (units 50–51): a rung-0 catch is
`rung_passed:-1, rung_failed:0`; a rung-1 catch is `rung_passed:0, rung_failed:1`.
