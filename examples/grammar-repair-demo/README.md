# Grammar Repair Demo

This demo shows the pre-flight grammar rung repairing a model-shaped positional
tool call into the named argument object the tool dispatch path expects. The
repair happens inside the kernel as a `TRANSFORM`; the model never sees an error
and does not spend a second turn re-emitting the same call with fixed JSON.

Run from the repo root after building `fak` or through `go run`.

## Positional Call Repaired

```bash
go run ./cmd/fak preflight \
  --policy examples/customer-support-readonly-policy.json \
  --tool create_support_ticket \
  --grammar-schema examples/grammar-repair-demo/create-support-ticket.schema.json \
  --args '{"_positional":["please help me"]}' \
  --show-dispatched-args
```

Expected stdout:

```text
verdict=TRANSFORM reason=MISROUTE by=grammar
dispatched_args={"body":"please help me"}
```

The call arrived with `_positional`, but the loaded grammar has exactly one
required parameter, `body`, so fak zips the value into a named argument object
before dispatch.

## Arity Mismatch Denied

```bash
go run ./cmd/fak preflight \
  --policy examples/customer-support-readonly-policy.json \
  --tool create_support_ticket \
  --grammar-schema examples/grammar-repair-demo/create-support-ticket.schema.json \
  --args '{"_positional":["a","b","c"]}' \
  --show-dispatched-args
```

Expected stdout:

```text
verdict=DENY reason=MISROUTE by=grammar
```

Three positional values cannot be mechanically mapped onto the one required
parameter, so fak refuses with `MISROUTE` instead of guessing.

## What This Saves

A conventional loop handles a bad tool-call shape by returning the tool error to
the model, waiting for another model turn, and hoping the corrected call uses the
right argument names. The grammar rung handles the arity-matched case in-syscall,
so the corrected arguments are available at the tool boundary immediately.

For the priced version of that saved retry turn, run `fak turntax`:

```bash
go run ./cmd/fak turntax
```

## When It Does Not Fire

An arity mismatch returns `DENY (MISROUTE)`, as shown above. An unknown grammar
fails open: the grammar rung abstains instead of refusing a tool shape it cannot
inspect, and the policy/adjudicator floor still decides whether the tool is
allowed.

The grammar registry is content-addressed. Loading the same parameter shape for
multiple tools stores one grammar by digest and points each tool name at that
entry, keeping the hot path cheap when many tools share the same argument shape.
