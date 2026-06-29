# Example output

Captured run of the four `fak preflight --explain` witnesses (binary `fak` v0.34.0,
capability floor [`policy.json`](policy.json)). Each refusal is a pure function of
`(policy, the proposed call)` — no model, key, or network — so the verdicts are
deterministic: the same call always yields the same reason and the same derived
disposition. Reproduce with the commands in [`README.md`](README.md).

## RETRYABLE — `MISROUTE`

```
$ fak preflight --policy examples/deny-as-value/policy.json --tool fix_my_args --args '{}' --explain
fak: loaded capability floor from examples/deny-as-value/policy.json
tool: fix_my_args   args: 2 bytes (sha 44136fa355b3)
verdict: DENY   reason: MISROUTE   by: monitor   disposition: RETRYABLE
explanation: fix_my_args denied by monitor: MISROUTE (RETRYABLE).

decision chain (9 rung(s), most-restrictive wins):
   [0] grammar.Rung               DEFER     by=grammar
   [1] ratelimit.Limiter          DEFER     by=ratelimit
   [2] preflight.Ladder           DEFER     by=preflight
   [3] engine.residencyGate       DEFER     by=engine-residency
   [4] plancfi.Adjudicator        DEFER     by=plancfi
   [5] ifc.SinkGate               DEFER     by=ifc-sink
   [6] gitgate.GitGate            DEFER     by=gitgate
   [7] shipgate.ShipAdjudicator   DEFER     by=shipgate
=> [8] adjudicator.Adjudicator    DENY      MISROUTE by=monitor   <- winner (rank 100)
```

## WAIT — `RATE_LIMITED`

```
$ fak preflight --policy examples/deny-as-value/policy.json --tool rate_limited_call --args '{}' --explain
fak: loaded capability floor from examples/deny-as-value/policy.json
tool: rate_limited_call   args: 2 bytes (sha 44136fa355b3)
verdict: DENY   reason: RATE_LIMITED   by: monitor   disposition: WAIT
explanation: rate_limited_call denied by monitor: RATE_LIMITED (WAIT).

decision chain (9 rung(s), most-restrictive wins):
   [0] grammar.Rung               DEFER     by=grammar
   ...
=> [8] adjudicator.Adjudicator    DENY      RATE_LIMITED by=monitor   <- winner (rank 100)
```

## ESCALATE — `SELF_MODIFY`

```
$ fak preflight --policy examples/deny-as-value/policy.json --tool write_kernel --args '{"path":"internal/kernel/kernel.go"}' --explain
fak: loaded capability floor from examples/deny-as-value/policy.json
tool: write_kernel   args: 36 bytes (sha 37d9480d8a64)
verdict: DENY   reason: SELF_MODIFY   by: monitor   disposition: ESCALATE
explanation: write_kernel denied by monitor: SELF_MODIFY (ESCALATE).

decision chain (9 rung(s), most-restrictive wins):
   ...
=> [8] adjudicator.Adjudicator    DENY      SELF_MODIFY by=monitor   <- winner (rank 100)
```

## TERMINAL — `POLICY_BLOCK`

```
$ fak preflight --policy examples/deny-as-value/policy.json --tool delete_account --args '{}' --explain
fak: loaded capability floor from examples/deny-as-value/policy.json
tool: delete_account   args: 2 bytes (sha 44136fa355b3)
verdict: DENY   reason: POLICY_BLOCK   by: monitor   disposition: TERMINAL
explanation: delete_account denied by monitor: POLICY_BLOCK (TERMINAL).

decision chain (9 rung(s), most-restrictive wins):
   ...
=> [8] adjudicator.Adjudicator    DENY      POLICY_BLOCK by=monitor   <- winner (rank 100)
```

## The loop-consumable form (`--json`)

A deny-loopback branches on the `disposition` field, not on prose. The same WAIT refusal
as machine-readable JSON (rungs elided for brevity — `args_digest` is the SHA, never the
raw args):

```json
{
  "tool": "rate_limited_call",
  "args_digest": "44136fa355b3",
  "args_bytes": 2,
  "verdict": "DENY",
  "reason": "RATE_LIMITED",
  "by": "monitor",
  "disposition": "WAIT",
  "explanation": "rate_limited_call denied by monitor: RATE_LIMITED (WAIT)."
}
```

## Contrast: an ALLOW carries no disposition

An affirmatively-allowed call (`search_web` is on the floor's allow-list) has no refusal to
derive a disposition from — the loop simply dispatches it:

```
$ fak preflight --policy examples/deny-as-value/policy.json --tool search_web --args '{"q":"x"}' --explain
verdict: ALLOW   by: monitor
explanation: search_web allowed: an affirmative policy rung permitted it.
```
