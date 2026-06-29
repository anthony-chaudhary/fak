# Captured run — `admit_and_log` posture demo

Captured with `fak` 0.34.0 via `fak preflight` on the two policy files in this
directory. The `fak: loaded capability floor …` line is on stderr; the `verdict…`
lines are the adjudicated result.

## 1. `fail_closed` — read-shaped tool off the allow-list → `DEFAULT_DENY`

```
$ fak preflight --policy examples/admit-and-log/research-batch-fail-closed.json \
    --tool read_internal_wiki --args '{}'
fak: loaded capability floor from examples/admit-and-log/research-batch-fail-closed.json
verdict=DENY reason=DEFAULT_DENY by=monitor
```

## 2. `admit_and_log` — same call admitted, with the would-deny stamp

```
$ fak preflight --policy examples/admit-and-log/research-batch-policy.json \
    --tool read_internal_wiki --args '{}' --explain
fak: loaded capability floor from examples/admit-and-log/research-batch-policy.json
tool: read_internal_wiki   args: 2 bytes (sha 44136fa355b3)
verdict: ALLOW   by: monitor
posture: admit_and_log (would_deny: DEFAULT_DENY)
explanation: read_internal_wiki allowed under the admit-and-log posture (would otherwise be DEFAULT_DENY); forensic metadata recorded.

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

The same call as JSON (the form a journal folds — `args_digest` only, never raw args):

```
$ fak preflight --policy examples/admit-and-log/research-batch-policy.json \
    --tool read_internal_wiki --args '{}' --json
{
  "tool": "read_internal_wiki",
  "args_digest": "44136fa355b3",
  "args_bytes": 2,
  "verdict": "ALLOW",
  "by": "monitor",
  "posture": "admit_and_log",
  "would_deny": "DEFAULT_DENY",
  ...
  "explanation": "read_internal_wiki allowed under the admit-and-log posture (would otherwise be DEFAULT_DENY); forensic metadata recorded."
}
```

## 3. `admit_and_log` does NOT relax write-shaped / explicitly-denied calls

```
$ fak preflight --policy examples/admit-and-log/research-batch-policy.json \
    --tool upload_file --args '{}'
fak: loaded capability floor from examples/admit-and-log/research-batch-policy.json
verdict=DENY reason=POLICY_BLOCK by=monitor

$ fak preflight --policy examples/admit-and-log/research-batch-policy.json \
    --tool write_report --args '{}'
fak: loaded capability floor from examples/admit-and-log/research-batch-policy.json
verdict=DENY reason=DEFAULT_DENY by=monitor
```

`upload_file` is an explicit `deny` entry (`POLICY_BLOCK`); `write_report` is
write-shaped and not allow-listed, so it stays at the `DEFAULT_DENY` floor. The posture
relaxed neither.
