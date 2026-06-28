# Author your own trace

`fak run --trace <file>` replays a recorded tool-call sequence through the kernel
and prints the verdict for each call. It needs **no model, no network, no GPU** —
the lowest-friction way to see the adjudicator floor decide. This is where
[`GETTING-STARTED.md` §2](../../GETTING-STARTED.md) points after the one built-in
trace; this walkthrough shows the schema and how to write your own.

Two ready-to-run traces live here:

| File | Shows | Witness |
|---|---|---|
| [`minimal.json`](minimal.json) | one ALLOW, one DENY (`DEFAULT_DENY`), one DENY (`POLICY_BLOCK`) | `./fak run --trace examples/trace-authoring/minimal.json` |
| [`with-poison.json`](with-poison.json) | the result-side quarantine path (a prompt-injection result + a secret-shaped result held out of context) | `./fak run --trace examples/trace-authoring/with-poison.json --engine mock` |

Run the commands from the repo root (`fak/`). Use `go run ./cmd/fak run …` if you
have not built the `fak` binary yet.

## The trace schema

A trace is one JSON object: a `slice_id` and an array of `calls`. Each call is one
tool invocation to replay, in order.

| Field | Where | Meaning |
|---|---|---|
| `slice_id` | top level | a label for the trace (appears in bench output and the workload hash) |
| `calls` | top level | the ordered list of tool calls to replay |
| `tool` | per call | the tool name — this is what the floor decides on (allow-set, allow-prefix, or deny map) |
| `args` | per call | the tool arguments as a JSON object (defaults to `{}` if omitted) |
| `meta` | per call | optional MCP-style hints as a string map, e.g. `{"readOnlyHint":"true","idempotentHint":"true","destructive":"true"}` |

The loader ignores any key it does not recognise, so you can annotate a trace
freely — this dir uses `_note` per call, the same way the reference fixture
[`testdata/tau2/tau2-smoke.json`](../../testdata/tau2/tau2-smoke.json) uses
`_provenance`.

`verdict` and `by` are **outputs, not inputs**. `fak run` prints them per call; you
do not write them into the trace:

```
[ 0] get_reservation_details      verdict=ALLOW     by=monitor   status=OK
```

- `verdict` — `ALLOW` / `DENY` / `TRANSFORM` / `QUARANTINE`.
- `by` — which rung decided (`monitor` = the policy adjudicator, `vdso` = the dedup
  fast path, `ifc-sink` = the egress-sink screen, `ctxmmu` = the result-side gate).
- `status` — `OK` if the engine ran the call, `ERR` if it was refused before running.

To see the **reason** behind a verdict (`DEFAULT_DENY` vs `POLICY_BLOCK` etc.),
preflight a single call:

```bash
./fak preflight --tool make_payment --args '{}'
# verdict=DENY reason=DEFAULT_DENY by=monitor
./fak preflight --tool shell_rm_rf --args '{}'
# verdict=DENY reason=POLICY_BLOCK by=monitor
```

`fak run` with no `--policy` uses the built-in adjudicator floor (the tau2 airline
demo tools — read-only `get_*`/`search_*`/`list_*`, plus a handful of named
booking tools; everything else is fail-closed). Dump it with `fak policy --dump`,
or point `--policy <manifest>` at one of the floors in [`../`](../README.md).

## Walkthrough: authoring `minimal.json`

Pick three tools that land on three different rungs of the default floor:

1. **ALLOW** — `get_reservation_details`. It is on the floor's allow-set (a
   read-only airline tool), so it is admitted and dispatched to the engine.
2. **DENY (`DEFAULT_DENY`)** — `make_payment`. It is on no allow-list and matches
   no allow-prefix, so the fail-closed floor refuses it: nothing affirmatively
   permitted it.
3. **DENY (`POLICY_BLOCK`)** — `shell_rm_rf`. It is on the floor's explicit deny
   map, so it is refused by an affirmative rule, not by the default.

Write each as a call with a `tool` and `args`:

```json
{
  "slice_id": "trace-authoring-minimal",
  "calls": [
    { "tool": "get_reservation_details", "args": {"reservation_id": "ABC123"} },
    { "tool": "make_payment",            "args": {"amount": 500, "to": "vendor-x"} },
    { "tool": "shell_rm_rf",             "args": {"path": "/srv/data"} }
  ]
}
```

Run it:

```bash
./fak run --trace examples/trace-authoring/minimal.json
```

```
[ 0] get_reservation_details      verdict=ALLOW     by=monitor   status=OK
[ 1] make_payment                 verdict=DENY      by=monitor   status=ERR
[ 2] shell_rm_rf                  verdict=DENY      by=monitor   status=ERR

summary: submits=3 vdso_hits=0 engine_calls=1 denies=2 transforms=0 quarantines=0
```

One call ran (`engine_calls=1`), two were refused before running (`denies=2`).
Swap in your own agent's tool names and arguments to see how the floor would
decide them — that is the whole loop.

## Walkthrough: the quarantine path (`with-poison.json`)

The floor in `minimal.json` decides on the **call** (tool + args). A second gate,
the result-side context-MMU ([`internal/ctxmmu`](../../internal/ctxmmu)), decides
on the **result** a tool returns: a prompt-injection payload or a secret-shaped
string is *quarantined* — held out of the agent's context — while a benign result
is admitted. [`with-poison.json`](with-poison.json) carries the same three result
shapes as [`testdata/poison.json`](../../testdata/poison.json) (an injection, a
secret leak, a benign control), authored as a replayable calls trace.

```bash
./fak run --trace examples/trace-authoring/with-poison.json --engine mock
```

```
[ 0] read_webpage                 verdict=ALLOW     by=monitor   status=OK
[ 1] read_file                    verdict=ALLOW     by=monitor   status=OK
[ 2] get_reservation_details      verdict=ALLOW     by=monitor   status=OK

summary: submits=3 vdso_hits=0 engine_calls=3 denies=0 transforms=0 quarantines=2
```

All three calls are *allowed* (the tools are read-shaped, so the floor admits
them), but two **results** are quarantined (`quarantines=2`): the injection and
the secret. The benign reservation result is admitted.

### Why `--engine mock`?

The context-MMU scans the bytes a tool **returns**, not the bytes you put in
`args`. The default `inkernel` engine generates its own result from a fused model,
so the poison you write into `args` never reaches the result and nothing is
quarantined:

```bash
./fak run --trace examples/trace-authoring/with-poison.json
# … summary: … quarantines=0     (runs clean; the engine regenerated the result)
```

The `mock` engine echoes the call's `args` straight back into the result, which is
what lets the poison reach the result-side gate and be quarantined. Use `--engine
mock` when you want a trace's `args` to *be* the result under test — the standard
way to author a result-side fixture without a live tool.

## Where to go next

- [`testdata/tau2/tau2-smoke.json`](../../testdata/tau2/tau2-smoke.json) — the
  larger reference fixture (12 calls from the tau2-airline demo, with `meta`
  hints), the trace `GETTING-STARTED.md` §2 runs.
- `fak policy --dump` — print the active floor; `--policy <manifest>` to swap in
  one of the floors in [`../`](../README.md) and re-run the same trace under it.
- `fak preflight --tool <t> --args <json> --explain` — the full per-rung decision
  trace for a single call.
