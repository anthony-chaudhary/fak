# fak kernel — the pre-flight ladder (rungs 0 and 1)

**Before a proposed tool call ever reaches a model turn or an execution, the kernel runs
it up a ladder of cheap, escalate-on-pass well-formedness checks — and a call that is
malformed or schema-invalid is refused at the bottom of the ladder, for free.** This demo
walks the two cheapest rungs, the ones with no model turn and no execution:

```
  a proposed tool call ─▶  rung 0  STATIC PARSE     are the args even valid JSON?      ─┐ catch here
                           rung 1  SCHEMA VALIDATION required fields + right types?     ─┘ ⇒ no turn, no exec
                              │ (every cheap rung deferred — nothing to refuse)
                              ▼
                           the authoritative monitor (capability floor: ALLOW / DEFAULT_DENY)
                              │ (admitted)
                              ▼
                           a model turn / the tool actually runs
```

`fak preflight` is the witness verb: it folds the rung ladder over **one** proposed call
and prints the verdict and which rung produced it — without running a model or a tool.
That is the whole point of pre-flight: a dead branch (a malformed or schema-violating call)
is caught at the cheapest possible layer, so it never spawns a process or burns a turn.

## The ladder, cheapest first, escalate-on-pass

A call is run up the rungs in order, and **escalation only happens on a pass**: rung 1
runs only if rung 0 passed; the authoritative monitor runs only if every cheap rung
deferred. The kernel folds the rungs with a most-restrictive-wins rule, so **a rung DENY
wins over a later ALLOW** — the cheap refusal is final.

| rung | name | what it catches | status |
|---|---|---|---|
| **0** | **static parse** | args that are not valid JSON | **SHIPPED** (`internal/preflight`, unit 47) |
| **1** | **JSON-Schema validation** | a required field missing, or present with the wrong type | **SHIPPED** (unit 48) |
| 2 | grammar | positional args auto-repaired to named (in-syscall TRANSFORM, no model turn); fail-open on unknown grammar | **SHIPPED**, deep-dive in **#227** (`internal/grammar`, units 52–57) |
| 3 | dry-run probe | offline escalation above rung 1 | **STUB** — plumbing only, not built in v0.1 |
| 4 | sandbox probe | sandboxed escalation above rung 1 | **STUB** — plumbing only, not built in v0.1 |

This demo focuses on **rungs 0 and 1** — the cheapest and the least-documented. The
grammar rung is demonstrated in its own walkthrough (**#227**); rungs 2/3 above rung 1 are
honestly STUB in v0.1 (`CLAIMS.md` #41): the static-parse and schema-check rungs are the
only escalation rungs implemented today.

## Run it

From the repository root (the launcher builds `fak`, or set `FAK_BIN` to a prebuilt one):

```bash
./examples/preflight-ladder/run.sh            # the five witnesses below, one line each
./examples/preflight-ladder/run.sh --explain  # add the per-rung decision trace to each
```

No model, no network, no GPU — it completes in well under a second. The captured output is
in [`EXAMPLE-OUTPUT.md`](EXAMPLE-OUTPUT.md). Every line is reproducible with one command;
cross-check any of them:

```bash
# clean call — every cheap rung passes/defers, the monitor admits it
fak preflight --tool search_flights --args '{"origin":"SFO","destination":"JFK"}'
# verdict=ALLOW reason=NONE by=monitor

# malformed args — rung 0 (static parse) refuses BEFORE anything else looks at the call
fak preflight --tool search_flights --args '{"origin":"SFO",}'
# verdict=DENY reason=MALFORMED by=preflight
```

(The witness verb on the installed binary is `fak preflight …`; from a source checkout it is
`go run ./cmd/fak preflight …`.)

## What you see — which rung fired each verdict

The default `run.sh` prints a one-line verdict per witness; `--explain` prints the full
decision chain so you can see exactly which rung won the fold:

```
1. clean call         search_flights {"origin":"SFO","destination":"JFK"}  -> ALLOW  by=monitor
2. malformed JSON     search_flights {"origin":"SFO",}                     -> DENY   by=preflight  (RUNG-0, MALFORMED)
3. unparseable args   search_flights {origin:SFO}                          -> DENY   by=preflight  (RUNG-0, MALFORMED)
4. unknown tool       delete_everything {}                                 -> DENY   by=monitor    (DEFAULT_DENY)
5. unknown grammar    search_flights {"origin":"SFO"}                      -> ALLOW  by=monitor    (grammar rung fails open)
```

Read it as the ladder doing its job:

- **Step 2 and 3** are caught at **rung 0** — `by=preflight`, `MALFORMED`. The args never
  even parse, so no later rung, no monitor, no model, and no tool ever sees the call. In the
  `--explain` trace, `preflight.Ladder` is the winner at index `[2]` and every rung after it
  is `ELIDED` — the cheap refusal short-circuited the whole rest of the chain.
- **Step 1** is the escalate-on-pass path: rung 0 and rung 1 both pass (the args parse and
  satisfy what is required), so `preflight.Ladder` **defers**, and the call escalates all the
  way to the authoritative monitor, which admits it (`by=monitor`, ALLOW).
- **Step 4** shows the ladder escalating *past* the cheap rungs on a different axis: `{}` is
  valid JSON, so rungs 0/1 have nothing to refuse and defer — and the **monitor** fail-closes
  on an unsanctioned tool (`DEFAULT_DENY`). Pre-flight is about call *shape*; the capability
  floor is about *which tool*. They are different layers, and this call shows both.
- **Step 5** is the honest fail-mode: the kernel has no grammar registered for this call, so
  the grammar rung **fails open** (defers) rather than inventing a verdict — pre-flight never
  refuses a call it cannot reason about. The args parse, and the monitor decides.

### An honest edge: rung 1 in the standalone witness

Rung 1 (schema validation) is **SHIPPED and unit-proven** — a required field with the wrong
type, or a missing required field, is refused `DENY/MALFORMED by=preflight` at rung 1, *after*
rung 0 passes. But the schema is a **per-tool input** the kernel must be told (`SetSchema`),
and the standalone `fak preflight` default capability floor installs **no** per-tool schema —
the airline-demo schemas are registered by the agent loop's `agent.Configure()`, which the
witness verb does not run. So in *this* demo rung 1 **defers** for every tool, and you see the
rung-0 catch and the monitor verdict, not a live rung-1 DENY.

The rung itself is real; its catch is proven directly in the unit tests rather than fabricated
from the CLI:

```
internal/preflight/preflight_test.go
  TestRung1SchemaTypeCheck         {"origin":123} with required origin:string => DENY/MALFORMED at rung 1
  TestRung1MissingRequiredFieldDenied  {"dest":"JFK"} missing required origin  => DENY/MALFORMED at rung 1
  TestRung0FailureNeverReachesRung1   "{bad" short-circuits at rung 0, never reaching rung 1
```

Run them with `go test ./internal/preflight`. The escalation ORDER — rung 0 before rung 1,
catch at the cheapest rung that can decide — is exactly what `TestRung0FailureNeverReachesRung1`
asserts.

## Hard-negative label harvesting

Every rung catch is recorded as a typed **hard negative**: a call that *passed* cheap rung
*k* and *failed* the next rung *k+1*. The kernel emits each one as a `LabelRow`
(`call_hash`, `rung_passed`, `rung_failed`, `verdict:"deny"`, `reason`) — so the refusals are
not just thrown away, they are **self-labeled training data** for the future syscall-tuned
model (`CLAIMS.md` #39, "hard-negative label harvesting"). A rung-0 catch labels
`rung_passed:-1, rung_failed:0`; a rung-1 catch labels `rung_passed:0, rung_failed:1`. The
ledger is bounded (oldest dropped first) so sustained malformed traffic — exactly the workload
pre-flight exists to catch — cannot grow it without bound. Witnessed by `TestNegativesRowFields`
and `TestCatchRateMix` (units 50–51).

## Scope and honesty

This demo shows exactly one layer of fak — the **cheapest** one. It does **not** claim:

- that rungs 2 (dry-run probe) and 3 (sandbox probe) are built — they are **STUB** in v0.1
  (`CLAIMS.md` #41); only rung-0 static-parse and rung-1 schema-check are implemented in
  `internal/preflight`.
- that the grammar rung is demonstrated here — it has its **own** walkthrough (**#227**); this
  is the whole-ladder view with rungs 0/1 as the headline.
- that the monitor's `DEFAULT_DENY` (step 4) is a pre-flight result — it is the *capability
  floor*, a different layer; pre-flight decides call **shape**, not which tool is sanctioned.

What it does show: **the malformed and the schema-invalid call are refused at the cheapest
rung that can decide them — before any model turn and before any execution — and each refusal
becomes a labeled hard negative.**

## Files

| file | what it is |
|---|---|
| `run.sh` | the launcher: builds `fak`, runs the five `preflight` witnesses, prints each verdict |
| `README.md` | this walkthrough |
| `EXAMPLE-OUTPUT.md` | a captured run (default one-liners + the rung-0 `--explain` trace) |

Related: `CLAIMS.md` #39 (rungs 0/1 + hard-negative harvesting), #40 (grammar rung), #41
(rungs 2/3 STUB); issue **#227** (the grammar-rung deep-dive); `internal/preflight/` (the
ladder) and `internal/grammar/` (the grammar rung). The capability-gate layer this escalates
into has its own demo: [`../adjudication-demo/`](../adjudication-demo/).
