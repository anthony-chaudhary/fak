# Captured run — self-modify floor

A real run of `fak preflight` against [`policy.json`](policy.json) on `fak` v0.34.0. The
floor is a pure function of `(policy, call)`, so these verdicts are deterministic — the same
call always yields the same line. The `fak: loaded capability floor …` notice each command
prints to stderr is elided below for readability.

## 1. Write targeting the policy file → `SELF_MODIFY` (the load-bearing witness)

```console
$ fak preflight --policy examples/self-modify-floor/policy.json \
    --tool write_file \
    --args '{"path":"policy.json","body":"{\"allow\":[\"delete_account\"]}"}'
verdict=DENY reason=SELF_MODIFY by=monitor
```

The same call with `--explain` surfaces the bounded-disclosure witness, the disposition, and
the full rung chain (eight rungs DEFER; the rank-100 adjudicator decides):

```console
$ fak preflight --policy examples/self-modify-floor/policy.json \
    --tool write_file \
    --args '{"path":"policy.json","body":"{\"allow\":[\"delete_account\"]}"}' --explain
tool: write_file   args: 64 bytes (sha f6b58c6eb22c)
verdict: DENY   reason: SELF_MODIFY   by: monitor   disposition: ESCALATE
witness: policy.json
explanation: write_file denied by monitor: SELF_MODIFY (ESCALATE). offending set: policy.json.

decision chain (9 rung(s), most-restrictive wins):
   [0] grammar.Rung               DEFER     by=grammar
   [1] ratelimit.Limiter          DEFER     by=ratelimit
   [2] preflight.Ladder           DEFER     by=preflight
   [3] engine.residencyGate       DEFER     by=engine-residency
   [4] plancfi.Adjudicator        DEFER     by=plancfi
   [5] ifc.SinkGate               DEFER     by=ifc-sink
   [6] gitgate.GitGate            DEFER     by=gitgate
   [7] shipgate.ShipAdjudicator   DEFER     by=shipgate
=> [8] adjudicator.Adjudicator    DENY      SELF_MODIFY by=monitor {policy.json}   <- winner (rank 100)
```

And `--json` — note `claim` carries the **single** offending glob, never the whole list:

```console
$ fak preflight --policy examples/self-modify-floor/policy.json \
    --tool write_file \
    --args '{"path":"policy.json","body":"{\"allow\":[\"delete_account\"]}"}' --json
{
  "tool": "write_file",
  "args_digest": "f6b58c6eb22c",
  "args_bytes": 64,
  "verdict": "DENY",
  "reason": "SELF_MODIFY",
  "by": "monitor",
  "claim": "policy.json",
  "disposition": "ESCALATE",
  "rungs": [
    { "index": 0, "rung": "grammar.Rung",             "by": "grammar",          "kind": "DEFER", "rank": 10, "deferred": true,  "winner": false },
    { "index": 1, "rung": "ratelimit.Limiter",        "by": "ratelimit",        "kind": "DEFER", "rank": 10, "deferred": true,  "winner": false },
    { "index": 2, "rung": "preflight.Ladder",         "by": "preflight",        "kind": "DEFER", "rank": 10, "deferred": true,  "winner": false },
    { "index": 3, "rung": "engine.residencyGate",     "by": "engine-residency", "kind": "DEFER", "rank": 10, "deferred": true,  "winner": false },
    { "index": 4, "rung": "plancfi.Adjudicator",      "by": "plancfi",          "kind": "DEFER", "rank": 10, "deferred": true,  "winner": false },
    { "index": 5, "rung": "ifc.SinkGate",             "by": "ifc-sink",         "kind": "DEFER", "rank": 10, "deferred": true,  "winner": false },
    { "index": 6, "rung": "gitgate.GitGate",          "by": "gitgate",          "kind": "DEFER", "rank": 10, "deferred": true,  "winner": false },
    { "index": 7, "rung": "shipgate.ShipAdjudicator", "by": "shipgate",         "kind": "DEFER", "rank": 10, "deferred": true,  "winner": false },
    { "index": 8, "rung": "adjudicator.Adjudicator",  "by": "monitor",          "kind": "DENY", "reason": "SELF_MODIFY", "claim": "policy.json", "rank": 100, "deferred": false, "winner": true }
  ],
  "explanation": "write_file denied by monitor: SELF_MODIFY (ESCALATE). offending set: policy.json."
}
```

## 2. The same allow-listed tool, a harmless target → `ALLOW`

```console
$ fak preflight --policy examples/self-modify-floor/policy.json \
    --tool write_file \
    --args '{"path":"notes/2026-06-20.md","body":"hi"}'
verdict=ALLOW reason=NONE by=monitor
```

`write_file` is allow-listed; the floor refuses it in case 1 only because the *path* matched
a glob, not because the tool is banned.

## 3. Every other protected glob refuses identically, each disclosing only its own match

```console
$ fak preflight … --tool write_file --args '{"path":".git/hooks/pre-commit","body":"x"}' --explain
verdict: DENY   reason: SELF_MODIFY   by: monitor   disposition: ESCALATE
witness: .git/

$ fak preflight … --tool write_file --args '{"path":"/home/agent/.ssh/id_rsa","body":"x"}' --explain
verdict: DENY   reason: SELF_MODIFY   by: monitor   disposition: ESCALATE
witness: id_rsa

$ fak preflight … --tool write_file --args '{"path":"/etc/passwd","body":"x"}' --explain
verdict: DENY   reason: SELF_MODIFY   by: monitor   disposition: ESCALATE
witness: /etc/
```

## 4. Honest scope — what this floor does NOT bound

The glob is matched against the *target* arg, not the payload. A protected path hidden in
the `body` is **not** caught (an `ALLOW`):

```console
$ fak preflight … --tool write_file --args '{"path":"notes/ok.md","body":"now also overwrite policy.json"}'
verdict=ALLOW reason=NONE by=monitor
```

And a non-write tool reaching a protected path is **not** a `SELF_MODIFY` — it falls to the
fail-closed `DEFAULT_DENY` (the read tool simply isn't allow-listed), because the self-modify
floor is scoped to write-shaped calls:

```console
$ fak preflight … --tool read_file --args '{"path":"policy.json"}'
verdict=DENY reason=DEFAULT_DENY by=monitor
```

See [`../../POLICY.md`](../../POLICY.md) "What the floor does and does NOT bound" for the full
boundary, including the shell-write path (`Bash: sed -i policy.json`) that the command-string
guard *does* cover.
