# fak kernel — fanbench adoption walkthrough (one master goal → N sub-agents)

**`fanbench` sweeps the orchestrator→worker topology — one lead decomposes a single goal,
fans out to N sub-agents, folds their results — from N=1 to N=1024, and reports what that
fan-out *structure* buys.** Two numbers, kept strictly apart (this is the honesty line):

- **MEASURED on the real kernel** — `cross_uplift = shared − isolated`: the cross-agent
  tool-result dedup the interleaved fan-out gets that the same N sub-agents run *apart*
  cannot. It is a real `k.Syscall` tier-2 path-swap, ablated SHARED-world (the whole
  fan-out in one world epoch) vs ISOLATED-world (each sub-agent solo, its own world), the
  same discipline as `fleetbench`. Plus the **exact** shared-prefix KV-reuse geometry
  `(N−1)·prefix_tokens` — the prefill the kernel never redoes because
  `model.NewBatchFromPrefix` prefills the master-goal prefix ONCE and clones it
  bit-identical into all N sub-agents.
- **MODELED by a transparent, knobbed cost model** (`FanoutCostModel`) — the token
  multiplier, the prompt-cache `tax_clawed_back` (~62% at the N≈256 plateau), the
  critical-path latency, and the saturation knee. Priced at documented Anthropic
  prompt-cache multiples (read 0.1× / write 1.25×) and reported *apart* from the measured
  halves, never blended.

```
  one master goal  ──►  lead decomposes it  ──►  N sub-agents gather in ONE world epoch  ──►  lead folds
                                                  └─ shared master-goal prefix prefilled ONCE, cloned N× ─┘
                                                  └─ siblings' tool reads dedup across the fan-out (cross_uplift) ─┘

  cross_uplift = (calls deleted with siblings present)  −  (calls deleted running each sub-agent SOLO)
               = the fan-out-ONLY bonus.  N=1 has no sibling, so cross_uplift is EXACTLY 0 (anti-inflation control).
```

## The honesty guardrail — read this first

Verbatim from [`CLAIMS.md`](../../CLAIMS.md) §"Fan-out benchmark":

> HONESTY GUARDRAIL: this prefix-reuse number is the **reuse-vs-no-reuse /
> vs-stateless-consumer** ablation (the win over a stack that re-sends the master-goal
> prefix per sub-agent, the common framework default), NOT a head-to-head win over a tuned
> shared-prefix engine (SGLang/RadixAttention/vLLM-APC also prefill the prefix once);
> fanning out to N=1 is even a small net LOSS (orchestration + cache-write overhead).

In one line: **fanbench measures how much of the naive N× prefix tax the reuse removes —
not that fak beats SGLang.** Those tuned engines occupy the *same* prefix-reuse lever; on
raw GPU throughput they stay ahead and the repo does not claim otherwise (the three-axis
table in [`FANOUT-BENCH-RESULTS.md`](../../docs/benchmarks/FANOUT-BENCH-RESULTS.md) §2 keeps
fak's reuse geometry strictly apart from the tuned-engine wall-clock, where a CPU
head-to-head actually has **llama.cpp ahead of fak**). `cross_uplift` is a fak-vs-fak
SHARED-vs-ISOLATED ablation (the fan-out's benefit over running the sub-agents apart), not
a vs-competitor claim.

## Run it

```bash
./examples/fanbench/run.sh                       # sweep N=1,8,64,512 (research profile, P=2048)
./examples/fanbench/run.sh --scale --grid canonical   # the D-001 acceptance ladder (N=1/100/500/1000)
./examples/fanbench/run.sh --profile no-share    # the anti-inflation control: 0 uplift at every N
```

`run.sh` drives the **real** harness — `go run ./cmd/fanbench …` — and writes a
`fanout.json` / `fanout.csv` artifact next to itself. It needs the **Go toolchain** (the
bench is a Go program in `cmd/fanbench`; it is **not** part of the `fak` binary — `fak
fanbench` is an unknown verb). On a box without Go the script prints the exact command plus
the published numbers and exits `0` — see [Run it without a Go toolchain](#run-it-without-a-go-toolchain).

Windows: run the `.sh` from WSL or Git Bash, or call `go run ./cmd/fanbench` directly from
the repo root.

## What you put in — fanbench is profile + grid driven, NOT a workload file

A common first wrong guess (and an inaccuracy in older write-ups): fanbench does **not**
read a `sample-workload.json` of prompts. Its inputs are flags that *generate* the
synthetic call stream and price the cost model. The three that shape your sweep:

| flag | what it controls | the bundled walkthrough uses |
|---|---|---|
| `--profile {research\|write-heavy\|no-share}` | the sub-agent read/write mix | `research` (read-heavy; cleanest positive uplift) |
| `--agents 1,8,64,512` *or* `--agent-max N --grid {log\|full\|canonical}` | the fan-out widths N to sweep | `1,8,64,512` |
| `--prefix P` *or* `--prefixes smoke,small,…` | the shared master-goal prefix tokens P | `2048` |

The one real **file** fanbench ingests is an optional model config — `--model-config
<config.json>` reads `max_position_embeddings` (or `model_max_length`) so the `big`/`max`
prefix presets fill a real model's context window. The bundled
[`sample-model-config.json`](sample-model-config.json) is exactly that file, so the example
is self-contained:

```bash
go run ./cmd/fanbench --profile research --agents 1,8,64,512 \
  --prefixes big --model-config examples/fanbench/sample-model-config.json
```

To **sweep your own fan-out**, pick the profile that matches your sub-agents (do they read
shared sources, or write?), set `--agents` to the widths you care about, and set `--prefix`
to your real system+goal+tool-schema token count. Everything else (cache multiples, decode,
fold budget) has documented defaults you can override.

## What you read out — the `cross_uplift` curve and the prefix geometry

`run.sh`'s per-cell progress line (to stderr) carries the headline columns:

```
[  3/  4] P=2048   N=64   T=4  calls=260    shared=191   isolated=134   cross=58     tax_back=61.4% speedup= 29.0  …
```

and the `fanout.csv` / `fanout.json` artifacts carry one row per cell. The published
research-goal surface (16 seeded trials, sub-turns=4, P=2048 — from
[`FANOUT-BENCH-RESULTS.md`](../../docs/benchmarks/FANOUT-BENCH-RESULTS.md) §1) is what your
own run reproduces for this shape:

| N | calls | shared_saved | isolated_saved | **cross_uplift** | prefix_tokens_saved = (N−1)·P | tax_clawed_back (modeled) | parallel_speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1   | 8    | 2    | 2    | **0**    | 0         | **0%** (net loss) | 1.0  |
| 8   | 36   | 22   | 16   | **+6**   | 14,336    | ~58%              | 6.0  |
| 64  | 260  | 191  | 134  | **+58**  | 129,024   | 61.4%             | 29.0 |
| 512 | 2052 | 1573 | 1072 | **+501** | 1,046,528 | 61.7%             | 61.9 |

(N=8/512 interpolate the published 1/4/16/64/256/1024 ladder; run it to get the exact cells
for your grid.)

### How to interpret the `cross_uplift` curve

- **It grows ~linearly with N — that growth IS the fan-out's value.** Each new sibling adds
  more tool reads the shared world can dedup. At N=1024 the interleaved fan-out deletes
  3,155 of 4,100 calls (77%), of which **+1,005 is the cross-agent bonus** the same 1,024
  sub-agents run solo could not get.
- **N=1 is EXACTLY 0 — the anti-inflation control.** A lone worker has no sibling to share
  with, so `cross_uplift = 0` by construction. The `no-share` profile is the partner
  control: identically 0 at *every* N (a non-zero value there would be a harness bug,
  asserted by `TestFanoutNoShareZeroUplift`). Run `./run.sh --profile no-share` to see the
  whole curve sit on 0.
- **Fanning out to N=1 is a small NET LOSS.** `token_mult` is 1.26 (with reuse) and
  `net_tokens_saved = −512`: the orchestration fold + cache-write overhead cost *more* than
  just doing the goal yourself. The levers only pay once there are siblings to amortize
  across — surfaced honestly, not hidden.
- **`tax_clawed_back` is a separate, MODELED number — it saturates ~61.7%.** That is the
  prompt-cache economics (`1.25 + (N−1)·0.1` vs naive `N·1.0`), not the measured dedup.
  Its analytic asymptote is `1 − 0.9P/(P+S+D+fold) ≈ 0.618`, and the curve lands on it. Do
  not blend it with `cross_uplift`.
- **`prefix_tokens_saved` is exact geometry, not a model** — `(N−1)·P` prefill the SHARED
  arm never recomputes. It is the one number with zero modeling in it.

## Run it without a Go toolchain

The bench is a Go program (`cmd/fanbench`), not part of the `fak` binary, so it needs the Go
toolchain. With Go installed, from the repo root:

```bash
go run ./cmd/fanbench --profile research --agents 1,8,64,512 --prefix 2048 \
  --out examples/fanbench/fanout.json --csv examples/fanbench/fanout.csv
```

Without Go, the published numbers are the witness — the research-goal `cross_uplift` curve
climbs `0 → +6 → +58 → +501` across N=1,8,64,512, with the N=1 control at exactly **0** and
`(N−1)·P` prefill saved growing to **1,046,528 tokens at N=512**. A captured run is in
[`EXAMPLE-OUTPUT.md`](EXAMPLE-OUTPUT.md).

## The authoritative witness — the Go tests

This example is a runnable *adoption walkthrough*; the **authoritative witness** that the
shipped bench behaves this way is the Go test suite:

```bash
go test ./cmd/fanbench       # TestPrefixReuseFanoutWitness (N clones bit-identical to a full prefill),
                             # TestRunScaleWritesReport, TestBuildAgentGrid, TestBuildPrefixGrid
go test ./internal/turnbench # TestFanoutNoShareZeroUplift, TestFanoutSingleAgentNoUplift,
                             # TestFanoutResearchPositiveUplift, TestFanoutDeterministic
```

`cmd/fanbench/TestPrefixReuseFanoutWitness` is the soundness rung: it prefills the
master-goal prefix once, clones it into N=64 sub-agents via `NewBatchFromPrefix`, and
asserts every clone's suffix decode is `max|Δ| = 0` against an independent full prefill — so
cross-agent prefix reuse provably never changes a sub-agent's result.

## Files

| file | what it is |
|---|---|
| `run.sh` | one-command launcher — runs `go run ./cmd/fanbench …` at N=1,8,64,512; passes flags through |
| `sample-model-config.json` | a bundled model config (`max_position_embeddings`) — the one real *file* input fanbench reads, for the `big`/`max` prefix presets |
| `EXAMPLE-OUTPUT.md` | a captured run with the `cross_uplift` curve and the N=1 zero-uplift control |

## Cross-references

- **`CLAIMS.md` §"Fan-out benchmark"** — the `[SHIPPED]` measured halves (`cross_uplift`,
  `(N−1)·P` geometry, the witnesses) and the `[SIMULATED]` cost model with the honesty
  guardrail quoted above.
- **The dense results ledger**:
  [`docs/benchmarks/FANOUT-BENCH-RESULTS.md`](../../docs/benchmarks/FANOUT-BENCH-RESULTS.md)
  — the full N=1…1024 surface (§1), the D-001 acceptance grid (§1), the three-axis
  over-claim guardrail and the tuned-engine wall-clock (§2), the prefix-scale sweep, the
  saturation knee (§3), and the task-quality litmus (§5).
- **The MEASURED live capstone**: [`cmd/fanrun`](../../cmd/fanrun/) — actually *runs* N real
  agent sessions on one goal and wall-clocks the wave (1,024 agents in 364 ms on a no-GPU
  box; `vdso_fills` flat at 3 for every N). The modeled curve here, made concrete.
- **The sibling fleet demo**:
  [`examples/fleet-reuse-demo/`](../fleet-reuse-demo/README.md) — A *independent* agents
  sharing a prompt (the `fleetbench` topology). fanbench is the distinct
  *orchestrator-fans-out-to-N-workers* shape with the SHARED-vs-ISOLATED ablation.
- **The code**: [`cmd/fanbench/`](../../cmd/fanbench/) (the runner),
  [`internal/turnbench/fanout.go`](../../internal/turnbench/) (the sweep engine),
  [`internal/bench/fanscale.go`](../../internal/bench/) (the D-001 `--scale` harness).

> **A note on two paths.** The issue and `CLAIMS.md` refer to the results ledger as
> top-level `FANOUT-BENCH-RESULTS.md`. In this tree it lives at
> **`docs/benchmarks/FANOUT-BENCH-RESULTS.md`**; the links above resolve to the real path.
> The `sample-workload.json` named in older deliverable drafts does not exist as a fanbench
> input — fanbench is profile/grid driven (see "What you put in" above); the bundled real
> file input is [`sample-model-config.json`](sample-model-config.json).
