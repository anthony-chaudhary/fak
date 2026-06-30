# fak — fleet / multi-agent shared-prompt reuse demo

The project is named **fleet** because its headline is reuse across many agents:
*the first worker pays for the shared setup; everyone after reads it for free.*
This is the runnable companion to that claim (issue #351) — the smallest thing you
can run to **see** the reuse curve, not just read the number in the top-level README.

The existing demos (`adjudication-demo/`, `cmd/simpledemo/`) are single-agent; the
benchmark harnesses (`cmd/fleetbench`, `cmd/fanbench`, `cmd/sessionbench`) measure the
property but are measurement tools, not "here is how you'd wire up a small fleet"
walkthroughs. This fills that gap.

```
  worker 1 ─┐
  worker 2 ─┤  same shared system prompt (setup)  ──▶  fak serve  (one kernel)
  worker N ─┘  + a small per-worker task δ               │  setup admitted ONCE,
                                                          ▼  reused by every later worker
                                                     model / engine
```

## What it shows

For a fleet of `N` workers that share one prompt prefix (system prompt + tool catalog
+ house rules) and differ only by a small per-worker task `δ`, at `N = 1, 2, 5`:

```
metric                          naive re-send    fak shared-prompt
total prompt bytes sent              N × full          1 × full + (N-1) × δ
model turns re-processing setup      N × setup         1 × setup
injection in shared context          per-worker        walled at first admit
```

The bytes/turns columns are an **exact accounting of the real request bodies** the
demo builds — not a stochastic measurement and not a projection. A naive re-send loop
makes the model re-ingest the whole setup once per worker (`N × setup`); behind fak the
shared prefix is admitted once and each later worker re-processes only its own `δ`. That
is arithmetic over the prompt structure, so the curve is the same on any box, GPU or not.

The **live half** (when a `fak serve` is reachable — `run.sh` provides one) drives the
`N` workers through the real kernel, same shared prompt each, to prove the wiring: `N`
agents, one kernel, one shared prefix. See [`EXAMPLE-OUTPUT.md`](EXAMPLE-OUTPUT.md) for a
captured run at `N = 1 / 2 / 5`.

## Run it

```bash
./examples/fleet-reuse-demo/run.sh            # build kernel, serve (offline mock), run the demo
./examples/fleet-reuse-demo/run.sh --offline  # the accounting only — no kernel, runs anywhere
FAK_DEMO_N=1,2,5,10 ./examples/fleet-reuse-demo/run.sh   # choose the worker counts
```

**Prerequisites — a strict subset of `adjudication-demo/`.** Just **Go** (to build
`fak`) and **Python 3** (stdlib only). No ollama, no model download, no GPU: the kernel
runs in front of its **offline mock planner**, because the reuse curve is an exact byte
accounting, not a model measurement. On Windows, run the `.sh` launcher from WSL or Git
Bash. The accounting view (`--offline`) needs only Python 3.
Expected runtime: the offline accounting run completes in seconds and is deterministic;
the live wiring path is deterministic for the committed N=1/2/5 fixture after the build.

## The three honesty caveats (reproduced verbatim from the README)

This demo deliberately does **not** over-claim. The top-level
[`README.md`](../../README.md) states the headline with its fences, and they are
reproduced here verbatim:

> On a 50-turn × 5-agent run, `fak`'s reuse does **~4.1× less work than a tuned
> warm-cache stack** (and ~60× less than the naive re-send loop)

> ~60× = headline session wall-time vs naive stateless; ~1.5–4× = realistic gain
> vs tuned warm-cache stack.

So: **the ~60× is only against the naive re-send pattern**, and **the ~4× is against a
tuned warm-cache stack** (vLLM / SGLang / provider prompt-caching). The small-`N` curve
this demo prints lands in exactly that ~1.5–4× region — it is the measured curve for
this small `N`, **not** projected to the frontier-scale "agent city" numbers the README
marks as design targets rather than measurements.

The third fence — **the reuse win is self-host only** — is reproduced verbatim from the
same honesty section (carried in
[`docs/benchmarks/TURN-TAX-RESULTS.md`](../../docs/benchmarks/TURN-TAX-RESULTS.md) §3
and the archived front page):

> The reuse win is self-host only: an app that just *calls* a frontier API gets the
> safety floor but not the savings.

In other words: if your agent calls a frontier API you do not host, you still get fak's
safety floor (the deny-by-structure capability gate), but the prompt-reuse savings shown
here require self-hosting the engine fak serves in front of.

## Why this is honest, not a simulation

- The **bytes** column counts the actual UTF-8 bytes of the request bodies the demo
  builds — both the naive `N × full` total and the fak `full + Σδ` total. Change the
  shared prompt or the per-worker tasks and the numbers move accordingly.
- The **setup-turns** column is the structural property of shared-prefix reuse: the
  setup is re-ingested `N` times by a naive loop and once behind fak.
- The **injection** row is the same property applied to a *poisoned* shared-context
  entry: a naive loop replays it into every worker's context (`N×`); fak holds a flagged
  result out of context at the first admit, so it is walled once and never replayed. The
  live quarantine mechanism itself is proven in [`../adjudication-demo/`](../adjudication-demo/)
  and [`../session-reload/`](../session-reload/); here it is the structural count.
- The offline mock planner reports no provider cache-read counter, so the live half
  proves the **wiring** (N workers, one kernel, one shared prefix); the bytes/turns table
  is the exact reuse **accounting**. Neither is projected to scale.

This demo does not claim frontier-provider prompt caching is replaced by fak; it shows the
self-hosted shared-prefix accounting and the kernel wiring for the small worker counts in
the captured run.

## Files

| file | what it is |
|---|---|
| `run.sh` | one-command launcher: build kernel → serve (offline mock) → run the demo → teardown |
| `demo.py` | the demo itself (stdlib only); prints the reuse curve, drives the fleet through `fak serve`; CI-usable exit code |
| `demo_test.py` | unit test pinning the reuse-curve accounting (no kernel needed) |
| `EXAMPLE-OUTPUT.md` | a captured run at N = 1 / 2 / 5 |

Related: top-level [`README.md`](../../README.md) "30-second picture" · the measurement
harness [`cmd/fleetbench`](../../cmd/fleetbench/) · the cost model
[`docs/benchmarks/TURN-TAX-RESULTS.md`](../../docs/benchmarks/TURN-TAX-RESULTS.md).
