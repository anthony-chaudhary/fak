# fak kernel — grammar auto-repair (positional→named) demo

**A tool call arrives with the wrong argument *shape*; the kernel fixes it in-syscall as a
TRANSFORM — and the model never sees an error.** This is the **cost-saving rung** of the
pre-flight ladder. A weaker model emits a call with positional args, or with a synonym key
for a required param (`from` instead of `from_currency`); fak maps it to the named
parameters the tool expects **without burning a model turn**. A SOTA loop spends an extra
error-code turn fixing the same call — the model emits a bad call, gets the error back, and
re-emits a corrected one. fak deletes that round-trip.

```
  bad-shape tool call ──▶  grammar rung (the first/cheapest adjudicator)
  {"from":"USD",...}            │  grammar registered for this tool?  no  → Defer (FAIL-OPEN)
                               │  already well-formed?              yes → Defer (nothing to do)
                               │  positional, arity matches?        yes → TRANSFORM (zip → named)
                               │  alias closes the gap?             yes → TRANSFORM (rename → canonical)
                               ▼  otherwise (can't repair mechanically) → Deny(MISROUTE)
                          dispatched call carries the REPAIRED args — no model turn spent
```

This is the runnable witness for **[`CLAIMS.md`](../../CLAIMS.md) #40**. The
[`adjudication-demo`](../adjudication-demo/README.md) shows the call-side capability gate
(ALLOW/DENY); this demo shows the grammar rung that sits *in front* of it and repairs
recoverable shape errors instead of refusing them.

## Run it

The end-to-end witness is the offline A/B loop — it registers the `convert_currency`
grammar, drives the same deterministic planner over the same task twice (kernel-mediated
vs. naive), and reports the in-syscall repair:

```bash
fak agent --offline --log agent-trace.log
```

No model, no network, no key — the offline planner is deterministic, so the kernel verdicts
reproduce exactly. (A prebuilt binary works too: `C:\Users\USER\bin\fak.exe agent --offline`
on Windows, or build from a clean checkout — see "Building" below.)
Expected runtime: with a built binary, the offline witness completes in seconds.

## What you see

The headline rows from the run (full capture in [`EXAMPLE-OUTPUT.md`](EXAMPLE-OUTPUT.md)):

```
metric                        now(base)          fak
--------------------------   ----------   ----------
model turns                           9            7
tool errors (-> retries)              1            0
in-syscall repairs                  n/a            1
```

and the per-call trace (`--log agent-trace.log`) shows the verdict directly:

```
[fak      turn 5] convert_currency       args={"from":"USD","to":"EUR","amount":240}
          verdict=TRANSFORM by=grammar                          <- repaired in-syscall, no model turn
[baseline turn 6] convert_currency       args={"from":"USD","to":"EUR","amount":240}   <- tool ERROR
[baseline turn 7] convert_currency       args={"from_currency":"USD","to_currency":"EUR","amount":240}  <- the retry turn fak deleted
```

The model emitted `{"from":"USD","to":"EUR","amount":240}` on both arms. On the fak arm the
grammar rung renamed `from`→`from_currency` and `to`→`to_currency` **in-syscall** and the
call went through. On the baseline arm the tool rejected the synonym keys, and the model
spent **turn 7** re-emitting the canonical call. That extra turn is the whole cost story.

## The cost story (cross-link to `fak turntax`)

`fak turntax` prices the saved turn. Run it and read the grammar lever:

```bash
fak turntax
```

```
  grammar repair (TRANSFORM)  : 2   -> saved 2 baseline reparse turns
  ...
  grammar-repair [turn-tax] turns=2    TRANSFORM in-syscall (alias->canonical)
```

Two saved model turns. At a frontier model's per-turn price that is real dollars and
latency — `fak turntax`'s cost-sensitivity table prices the same fixed turn count from
`$0.032` (local-fast) to `$0.124` (frontier) per session slice. The grammar rung is one of
the more concrete before/after wins fak offers, and `turntax` is where it shows up on the
bill.

## When it does NOT fire (both honest)

The rung repairs only what is *mechanically* recoverable — it never guesses:

| input | verdict | why |
|---|---|---|
| positional, **arity matches** the grammar | `TRANSFORM` | zip positional → named 1:1, deterministic param order |
| **alias** closes the well-formedness gap (`from`→`from_currency`) | `TRANSFORM` | rename synonym → canonical, but only if it actually makes every required param present |
| positional, **arity mismatch** (3 values, 1 param) | `Deny(MISROUTE)` | can't zip; refuse with a **model-fixable** disposition rather than guess the mapping |
| **no grammar** registered for the tool | `Defer` (FAIL-OPEN) | never over-refuse a tool whose shape fak can't inspect |

The two failure modes are the honest part. `MISROUTE` is a deny the model can *recover*
from (re-emit a correct call) — fak refuses to silently dispatch a wrong-shaped call. And a
tool with no registered grammar **fails open**: fak does not refuse what it cannot inspect.
The testdata files (`testdata/*.json`) carry the exact before/after shapes for each row.

## Content-addressed grammar dedup

Grammars are hashed by content (`grammar.go`, `digest()`): two tools with an identical
argument shape — or the same tool seen twice — reuse a single deduped grammar entry, so the
rung stays cheap on the hot path. The rung is the **first** adjudicator in the fold
(`abi.RegisterAdjudicator(5, …)`), i.e. the cheapest, run before any trust or effect rung.

## The standalone `fak preflight` path (honest scope)

The issue sketched a standalone `fak preflight --tool … --args '{"_positional":[…]}'`
witness. That is **not yet** the witness here: `fak preflight` adjudicates against the
built-in floor and does **not** call `agent.Configure()`, which is where the
`convert_currency` grammar (and its aliases) is registered — so a bare `fak preflight
--tool convert_currency …` returns `DEFAULT_DENY by=monitor` (the tool isn't on the
preflight allow-list) and never reaches the grammar rung. The end-to-end rung is therefore
witnessed through **`fak agent`**, which does register the grammar. Wiring a grammar into
the standalone `preflight` path (e.g. a `--grammar schema.json` flag) would make the
issue's exact one-liner work and is a clean follow-on.

The positional→named half of the rung is witnessed directly by the unit tests:

```bash
go test ./internal/grammar/ -run 'TestAdjudicatePositional'
#   TestAdjudicatePositionalRepairable    {"_positional":["alice"]} -> {"name":"alice"}  TRANSFORM
#   TestAdjudicatePositionalUnrepairable  {"_positional":["a","b","c"]} (arity 3 vs 1)   Deny(MISROUTE)
```

## Building (if you don't have a prebuilt `fak`)

From a clean checkout of the repo root (the Go module root):

```bash
go build -o fak ./cmd/fak
./fak agent --offline --log agent-trace.log
```

Do not run `go build`/`go test` from a dirty shared tree; build from a fresh
`git archive HEAD` extract if needed.

## Files

| file | what it is |
|---|---|
| `README.md` | this walkthrough |
| `EXAMPLE-OUTPUT.md` | the captured `fak agent --offline` + `fak turntax` run |
| `testdata/aliased-call.json` | the `from`/`to` → canonical alias-repair shape (the end-to-end case) |
| `testdata/positional-call.json` | the positional→named arity-matched zip shape |
| `testdata/arity-mismatch-call.json` | the unrepairable arity-mismatch → `Deny(MISROUTE)` shape |

Related: [`../adjudication-demo/`](../adjudication-demo/README.md) (the call-side capability
gate this rung sits in front of), [`../../CLAIMS.md`](../../CLAIMS.md) #40 (the claim),
[`../../docs/proofs/grammar.md`](../../docs/proofs/grammar.md) (the soundness proof),
`internal/grammar/grammar.go` and `internal/agent/inject.go` (the Go witnesses behind it).
