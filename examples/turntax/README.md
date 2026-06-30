# fak kernel — turn-tax adoption walkthrough (price your own workload's turn tax)

**When a model emits a malformed, duplicate, or poisoned tool call, a SOTA agent loop
spends an *extra model turn* to recover — it gets the error back and re-emits.** fak's
preflight ladder handles those classes *in-syscall* and the model never spends that turn.
`fak turntax` replays a class-labeled trace through the **real kernel** and prices the
delta — per lever, with the safety floor on a separate axis. This walkthrough shows you how
to point it at **your own** workload.

This is the *measurement* companion to the [`grammar-repair`](../grammar-repair/README.md)
demo: that one shows the *mechanism* that deletes one turn (a positional→named grammar
repair); this one shows how to *count* the turns a whole workload saves and put a dollar
figure on them.

> Note on the [issue #330](https://github.com/anthony-chaudhary/fak/issues/330) text: the
> ticket sketched a `run.sh` and a Go-build prerequisite. This example runs against the
> prebuilt `fak` binary directly (no `run.sh`, no Go toolchain needed at run time) and the
> witness module is `internal/turnbench` — not the `cmd/turntaxdemo`/`cmd/fak/main.go:66`
> the ticket guessed at. The four-file deliverable, the four call classes, the per-lever
> breakdown, the separate-axis safety floor, and the happy-path-saves-0 control are all
> exactly as the ticket asked.

## Run it

`fak turntax` needs no model, no network, and no key — it replays a recorded trace through
the kernel, so the verdicts are deterministic and reproduce exactly. Two runs witness the
two halves of the honesty story (the captured output is in
[`EXAMPLE-OUTPUT.md`](EXAMPLE-OUTPUT.md)):

```bash
# 1. the real workload — all four classes fire
fak turntax --trace examples/turntax/sample-trace.json

# 2. the anti-inflation control — a clean happy path saves exactly 0
fak turntax --trace examples/turntax/sample-trace-happy.json
```

On Windows with the prebuilt binary: `C:\Users\USER\bin\fak.exe turntax --trace
examples\turntax\sample-trace.json`. Run from the repo root so the relative `--trace` path
resolves. (`--trace PATH` loads a trace file directly; the sibling `--suite NAME` form looks
for `testdata/turntax/<NAME>.json` relative to your working directory.)

## What you feed in: a class-labeled trace

The input is one JSON object per tool call, in turn order. Each call carries the `tool`
name, its `args`, the `meta` tool-hints the kernel reads (`readOnlyHint`, `idempotentHint`,
`destructive`), and a documentation `class` — the disposition you *expect*. The bundled
[`sample-trace.json`](sample-trace.json) is a 14-call airline-support session; one row:

```json
{ "tool": "convert_currency",
  "args": {"from": "USD", "to": "EUR", "amount": 240},
  "meta": {"readOnlyHint": "true", "idempotentHint": "true"},
  "class": "grammar",
  "note": "model emits the alias keys from/to, NOT canonical from_currency/to_currency" }
```

The `class` label is **not trusted**. `fak turntax` derives the *actual* class from the live
kernel verdict and cross-checks it against your label — a mislabeled call is caught by the
consistency guard, not silently scored. (The label also does not enter the workload hash, so
relabeling a call can't change the priced workload.)

## The four call classes

Every call falls into one of four buckets. Two save turns; two are the safety floor.

| class | what the model did | what fak does in-syscall | the saved turn |
|---|---|---|---|
| **happy** (`pass`) | a clean, first-occurrence, allowed call | allow → engine round-trip | **none** — both arms pay it (the control) |
| **malformed** (`grammar`) | emitted an alias / wrong-shape arg (`from` vs `from_currency`) | `TRANSFORM`: rename → canonical, dispatch the repaired call | the baseline's error+reparse turn |
| **duplicate** (`dedup` / `pure` / `static`) | re-issued a read, or called a pure/constant tool | vDSO local serve (tier-2 cache / tier-1 compute / tier-3 static) | the baseline's redundant round-trip turn |
| **poison** (`quarantine` / `deny`) | pulled in an injected doc, or fired a destructive op | context-MMU quarantine / capability-floor deny | **none** — this is the safety floor, a *separate axis* |

## What you read out: the per-lever breakdown

The headline is **decomposed honestly** into the two ways a turn gets saved (the captured
run is in [`EXAMPLE-OUTPUT.md`](EXAMPLE-OUTPUT.md)):

```
-- NET turn-tax --
  turns saved   : 9   (forced 5 = grammar+dedup; elision 4 = pure+static)

-- ablation levers --
  grammar-repair [turn-tax]    turns=2    TRANSFORM in-syscall (alias->canonical)
  vdso           [turn-tax]    turns=7    3-tier local serve (pure / content-cache / static)
  quarantine     [safety-floor] turns=0   context-MMU result admission (poison paged out)
  deny           [safety-floor] turns=0   capability-floor adjudication (deny-as-value)
  NET (1-shot)   [turn-tax]    turns=9    all levers
```

- **forced vs elision** is the honest decomposition. A *forced* turn is one the baseline is
  *forced* to spend re-emitting a call the kernel repaired or deduped (grammar + dedup). An
  *elision* turn is one the kernel *elides* entirely by serving the result locally (a pure
  computation or a static table). They are different mechanisms; the report keeps them
  apart instead of lumping them into one number.
- **the vDSO lever is a REAL on/off path swap**, not an estimate. The report runs the trace
  twice — vDSO on and vDSO off — and reports `turns_saved(on) − turns_saved(off)`, which
  must equal the kernel's `VDSOHits` counter (here `9 − 2 = 7 == VDSOHits 7`). This is what
  `TestRun_VDSOAblationIsARealPathSwap` pins.
- **per-lever turns map to kernel counters.** `grammar=2` is exactly `Counters.Transforms`;
  `vdso=7` is exactly `VDSOHits`. The fak side is **live kernel events, not modeled** —
  the only modeled part is the per-turn *price* (the cost knobs below), and that is applied
  to a turn count the kernel produced.

## The safety floor is on a separate axis

The poison classes (`quarantine`, `deny`) save **0 turns** by design — they are reported on
a different axis entirely:

```
-- safety floor (deterministic moat, NOT a turn count) --
  injections admitted   baseline=1  fak=0
  destructive executed  baseline=1  fak=0
```

This is the load-bearing honesty point. The turn-tax savings are **in addition to** the
security floor, **not a trade against it**. fak does not buy you cheaper turns by skipping a
safety check; the injected doc is quarantined and the destructive `delete_account` is denied
*regardless* of the turn count, and those events are never folded into the dollars-saved
number. A reader can take the turn savings and the floor separately and neither inflates the
other.

## The happy-path control saves exactly 0 (anti-inflation)

Run [`sample-trace-happy.json`](sample-trace-happy.json) — three clean calls, no alias, no
duplicate, no poison — and the kernel saves **nothing**:

```
  pass (allow+engine, control): 3   -> 0 saved (both arms pay it)
  turns saved                 : 0   (forced 0 = grammar+dedup; elision 0 = pure+static)
  dollars saved               : $0.00000
```

If this control ever printed a non-zero saving, the benchmark would be applying a fixed
per-call discount instead of measuring real avoided errors. That it reads `0` is what
`TestRun_HappyPathSavesNothing` enforces — it is the proof the headline isn't padded.

## Reading the cost — and where it actually pays off

The kernel fixes the *turn count*; the per-turn *price* is a transparent knob
(`--prompt-tokens`, `--completion-tokens`, `--turn-latency-ms`), so the report shows the
same 9 saved turns across three price regimes:

```
  local-fast (400ms, 600tok)     $0.03240   3.60s
  hosted-flash (1.5s, 1200tok)   $0.04860  13.50s
  frontier (4s, 4000tok)         $0.12420  36.00s
```

**An honest caveat the tool prints itself.** This sample slice is deliberately error-rich
(9 saved turns out of 14 calls). On a real production trajectory the addressable malformed/
duplicate rate is *small* — the `--breakeven` sweep estimates a real-world airline-support
rate near `0.7%`, where the per-session turn-saving is fractional. Run it to see the curve:

```bash
fak turntax --trace examples/turntax/sample-trace.json --breakeven
```

```
     h   mean_turns   tok/sess     $/sess
 0.007       0.3250        429    0.00176   <- real-world rate (small)
 0.100       5.5750       7359    0.03011
 0.500      27.1800      35878    0.14677   <- the demo slice is the far high end
```

So price your *own* trace, not the demo: capture a representative session, label its calls,
and read the per-lever turns. At low error rates the turn-tax is a modest cost win and **the
safety floor is the reason to run the kernel** — which is exactly why the report keeps the
two axes apart.

## Honest scope

This walkthrough does not claim the bundled trace is representative of every production
agent workload, and it does not prove a safety guarantee by itself. It is a pricing witness
for turn-tax accounting; the safety-floor witnesses are linked separately below.

## Files

| file | what it is |
|---|---|
| `README.md` | this walkthrough |
| `run.sh` | one-command replay of the workload trace and the happy-path control |
| `sample-trace.json` | the bundled 14-call class-labeled session (all four classes fire) |
| `sample-trace-happy.json` | the clean 3-call control that must save 0 |
| `EXAMPLE-OUTPUT.md` | the captured `fak turntax` runs (real workload + happy control + break-even) |

Cross-links: [`../grammar-repair/`](../grammar-repair/README.md) (the *mechanism* that saves
one turn), [`../../CLAIMS.md`](../../CLAIMS.md) §"Turn-tax benchmark" (the claim), and the
Go witnesses behind it: `internal/turnbench/turnbench.go` with the green tests
`TestRun_HappyPathSavesNothing` and `TestRun_VDSOAblationIsARealPathSwap`
(`internal/turnbench/turnbench_test.go`). The repo's top-level `TURN-TAX-RESULTS.md` carries
the writeup of the canonical `turntax-airline` slice this sample mirrors.
