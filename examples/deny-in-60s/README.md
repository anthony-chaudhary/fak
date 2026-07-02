# fak kernel — deny-in-60s: watch an irreversible call get refused

**One command, under a minute, no API key, no model, no GPU, no network.** An agent
proposes `drop_production_database` — an irreversible action. The fak kernel refuses it
under a default-deny floor, then the *same call* clears under a permissive floor. The only
thing that changed between the two runs is the policy file — proving the refusal is
**structural** (the floor never admitted the tool), not a model choosing to behave.

```
  agent ──proposes──▶  drop_production_database {"database":"prod"}
                              │
        policy.json (default-deny: the tool is never mentioned)
                              ▼
                    verdict: DENY   reason: DEFAULT_DENY
                              │
        policy-permissive.json (the tool is on the allow-list)
                              ▼
                    verdict: ALLOW
```

## Prerequisites

A built `fak` binary on `PATH` (or `FAK_BIN=/path/to/fak`) — one static Go binary,
nothing else. **No model, API key, GPU, or network**: the verdict is a pure function of
`(policy, the proposed call)` — the offline adjudication path `fak preflight` exposes —
so the demo is deterministic and CI-usable.

```bash
go install github.com/anthony-chaudhary/fak/cmd/fak@latest   # or: go build ./cmd/fak
```

## Run it

```bash
bash examples/deny-in-60s/run.sh
```

## Expected output

Two `fak preflight --explain` witnesses (the full run also prints each rung of the
decision chain; excerpt):

```
== 1/2 DENY — the default-deny floor (policy.json) never mentions drop_production_database ==
verdict: DENY   reason: DEFAULT_DENY   by: monitor   disposition: TERMINAL
explanation: drop_production_database denied by monitor: DEFAULT_DENY (TERMINAL).

== 2/2 ALLOW — the permissive floor (policy-permissive.json) lists it ==
verdict: ALLOW   by: monitor
explanation: drop_production_database allowed: an affirmative policy rung permitted it.
```

Or run the two witnesses by hand — this is all the script does:

```bash
fak preflight --policy examples/deny-in-60s/policy.json            --tool drop_production_database --args '{"database":"prod"}' --explain
fak preflight --policy examples/deny-in-60s/policy-permissive.json --tool drop_production_database --args '{"database":"prod"}' --explain
```

## Why this is the interesting kind of "no"

- **Nobody wrote a rule about `drop_production_database`.** [`policy.json`](policy.json)
  is two allow-list entries; the dangerous tool is refused because it was never
  *admitted* (`DEFAULT_DENY`), not because someone predicted it and blocklisted it. A
  blocklist fails on the attack you didn't foresee; a default-deny floor doesn't have to
  foresee it.
- **The diff between refuse and permit is one policy line.** Compare the two JSON files:
  [`policy-permissive.json`](policy-permissive.json) adds the tool to `allow`, and the
  identical call clears. Same binary, same call, no model in the loop — the floor is the
  policy, not the model's judgment.
- **The refusal is a structured value, not an error string.** `DEFAULT_DENY` is a code
  from the closed refusal vocabulary in [`POLICY.md`](../../POLICY.md), and the verdict
  carries a derived disposition (`TERMINAL`) an agent loop can branch on without another
  model turn — [`deny-as-value/`](../deny-as-value/README.md) walks that four-way map.

## Scope — what this does **not** claim

This is a **name-level** demo: the floor refuses the *tool*, which is the structural
default-deny gate at its simplest. It does not exercise argument-level rules
([`sql-analyst-policy.json`](../sql-analyst-policy.json) does), quarantine of untrusted
tool *results* ([`quarantine-demo/`](../quarantine-demo/README.md) does), or a real model
trying to evade the gate ([`adjudication-demo/`](../adjudication-demo/README.md) drives a
live model into it). No adoption or benchmark claim is made here — the demo proves one
thing: the deny/allow boundary is the policy file.

## Files

| file | what it is |
|---|---|
| `README.md` | this walkthrough |
| `run.sh` | the one command — both preflight witnesses back to back |
| `policy.json` | the default-deny floor — `drop_production_database` is never mentioned |
| `policy-permissive.json` | the same floor plus one allow line — the same call clears |

Related: [`../../POLICY.md`](../../POLICY.md) (the manifest format and the closed refusal
vocabulary); [`deny-as-value/`](../deny-as-value/README.md) (what a loop *does* with a
structured deny); [`adjudication-demo/`](../adjudication-demo/README.md) (the same gate
with a real model attacking it).
