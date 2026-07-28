# compose-ollama example output

**Provenance — read this first.** The three blocks below are *captured* from a real
`fak` v0.41.0 run on this checkout against this recipe's [`policy.json`](policy.json).
They are the part that matters and the part you can re-check in seconds: the capability
floor and the verdicts it returns need no Docker, no model, and no network.

The `docker compose up` bring-up itself is **not** captured here. It pulls ~2 GB of
images plus the model, and its container banners and timings are machine-specific —
reproducing them is what [`run.sh`](run.sh) is for.

## The capability floor validates offline

```text
$ fak policy --check policy.json
OK  examples/compose-ollama/policy.json  (manifest valid; every deny cites a closed-vocabulary reason)

posture            : fail_closed
allow (exact)      : 1 tool(s)
allow (prefix)     : read_, get_, search_, list_, lookup_, find_
deny (explicit)    : 2 tool(s)
                     exfiltrate -> SECRET_EXFIL
                     shell_rm_rf -> POLICY_BLOCK
egress posture     : default-allowed
subagent depth     : max_depth=8 (default — fail-closed)
```

## The verdicts the gate will return, before anything runs

```text
$ fak preflight --policy policy.json --tool read_file --args '{}' --explain
fak: loaded capability floor from examples/compose-ollama/policy.json
tool: read_file   args: 2 bytes (sha 44136fa355b3)
verdict: ALLOW   by: monitor
explanation: read_file allowed: an affirmative policy rung permitted it.

$ fak preflight --policy policy.json --tool exfiltrate --args '{}' --explain
tool: exfiltrate   args: 2 bytes (sha 44136fa355b3)
verdict: DENY   reason: SECRET_EXFIL   by: monitor   disposition: TERMINAL
explanation: exfiltrate denied by monitor: SECRET_EXFIL (TERMINAL).

$ fak preflight --policy policy.json --tool shell_rm_rf --args '{}' --explain
tool: shell_rm_rf   args: 2 bytes (sha 44136fa355b3)
verdict: DENY   reason: POLICY_BLOCK   by: monitor   disposition: RETRYABLE
explanation: shell_rm_rf denied by monitor: POLICY_BLOCK (RETRYABLE).
```

Each verdict is a pure function of `(policy.json, the proposed call)` — no model, no
key, no network — so it is byte-identical on every run and on every machine, and each
command exits `0`. That is the determinism this recipe claims. The **model's**
generations are not deterministic and are not claimed to be.

## The governed front door answers

```text
$ curl -s http://localhost:8080/healthz
{"engine":"inkernel","model":"mock","ok":true,"planner":"mock"}
```

Captured against a local keyless `fak serve` with no upstream wired, which is why
`model` reads `mock`; under [`compose.yaml`](compose.yaml) the same route reports the
Ollama-backed upstream. `/healthz` is the one route that answers unauthenticated —
under compose (`--require-key-env=FAK_GATEWAY_KEY`) every other route is refused
without `Authorization: Bearer $FAK_GATEWAY_KEY`. `run.sh` prints the status code that
refusal actually carries rather than asserting one here.
