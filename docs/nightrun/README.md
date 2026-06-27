# Run it all night — the data-collection center of excellence

`fak nightrun` is the one door an operator or an agent uses to answer the only
question that matters for unattended data collection:

> **What is the single most important datum I can collect on THIS box, right
> now — and the exact command to collect it?**

…and then to collect the whole feasible queue on a loop, recording what was
gathered so the next night picks up where this one left off.

## Why this exists

fak already had the *parts* of overnight benchmarking, but no spine that made
"run it all night" trivial:

- **the menu** — `internal/benchcatalog` knows *what* 35 benchmarks exist and the
  cold-start cost of each, but not what to run next or whether *this* box can.
- **the results grid** — `experiments/benchmark/catalog.json` records what each
  remote bench-node *has* run.
- **a next() brain** — `tools/bench_plan.py` ranks the grid, but it plans for a
  fixed roster of *remote* bench-nodes, is plan-only, and can't answer "what can
  the box I'm sitting on collect tonight."

`nightrun` is the missing operator/agent front door over those parts. It is
**local-capability-aware** (it probes the box it runs on, so it never proposes a
CUDA benchmark on a Mac or an HW-gated witness on a box with no GPU),
**loop-closing** (a durable ledger records what was gathered so next() skips
fresh data and resurfaces stale data), and **unified** (the candidate set spans
both the benchmark grid *and* a curated backlog of the named, still-open measured
witnesses the project is actually blocked on).

## Run it

```bash
go run ./cmd/fak nightrun caps      # the probed box fact-sheet (gpu/weights/datasets/creds)
go run ./cmd/fak nightrun next      # the single most important feasible datum + the command
go run ./cmd/fak nightrun plan      # the whole ranked queue (feasible first, then why-blocked)
go run ./cmd/fak nightrun run       # DRY-RUN: print what it would collect, write nothing
go run ./cmd/fak nightrun run --apply --loop [--max N]   # collect the night for real
go run ./cmd/fak nightrun ledger    # the durable collection history
```

All of these take `--json` for an agent, and `--now <stamp>` to evaluate
deterministically as-of a fixed time.

## How next() ranks

Each feasible task is scored by a blend that sums to 1.0:

| signal | weight | reads |
|---|---|---|
| novelty | 0.45 | a datum never collected on **this** box — a first-ever measurement |
| value | 0.35 | the task's importance class (frontier > witness > regression > coverage > smoke) |
| staleness | 0.20 | for an already-collected datum, how far past its re-check interval it has drifted |

Infeasible tasks are kept in `plan` (so you can see *why* the box can't run them)
but always sort after every feasible task, and `next` only ever returns a
feasible one. A fixed `--now` + box + ledger yields byte-identical output.

## The two backlog sources

1. **the benchmark grid** — every entry in `internal/benchcatalog` becomes a
   task; its cold-start `Need` maps to a capability requirement and its level
   seeds a value.
2. **the curated open-witness backlog** — `internal/nightrun/backlog.go`'s
   `witnessTasks`: the named, still-open measured data the project is blocked on
   (e.g. the on-box GLM-5.2 load re-measure, the 7B Q8 Metal decode kernel
   bandwidth, the H200 GLM-5.2/vLLM throughput, the credentialed Terminal-Bench
   run). Each is a **task** — work to do — never a result, so it cannot overclaim.

Add a one-off datum without recompiling via the operator overlay
`experiments/nightrun/backlog.json` (a JSON array of tasks, additive over the
built-ins). Promote a durable, recurring datum into `witnessTasks` so it ships in
the binary.

## The honesty boundary

- `next` / `plan` / `caps` are pure reads — they never run anything.
- `run` is **dry-run by default**; only `--apply` executes real commands.
- A task the box can't run is **never selected**, so the loop can never claim to
  have collected HW-gated data on hardware that can't produce it.
- An `--apply` ledger row records what was **observed** (exit status, artifact
  path, a best-effort parsed number only when one is actually present) — never a
  fabricated number. A failed run is recorded as `failed`, with its artifact.

## The durable ledger

`docs/nightrun/collected.jsonl` is append-only — one `fak-nightrun-collect/1` row
per collected (or attempted) datum, so it is durable trunk evidence of what the
fleet has gathered, not a regenerable build artifact:

| field | meaning |
|---|---|
| `date` / `box` / `generated_at` | when, and on which machine, the datum was collected |
| `task_id` / `value` | which task, and its importance class |
| `command` | the exact command that ran |
| `outcome` | `collected` / `failed` / `dry-run` / `skipped` (observed, never asserted) |
| `artifact` | the captured-output path, when any |
| `number` | the first parsed unit-bearing token (e.g. `17.73 tok/s`), best-effort — the artifact is authoritative |

`--apply` extends the ledger; don't hand-edit it. Commit the one file by path:

```bash
git commit -s -- docs/nightrun/collected.jsonl -m "docs(nightrun): record collection tick (fak nightrun)"
```

The committed ledger is the durable record. The per-run captured-output logs under
`experiments/nightrun/<box>/` are local evidence only — they are gitignored (raw
stdout, regenerable, and a raw-hostname box dir must never reach the public tree).
The operator overlay `experiments/nightrun/backlog.json` is the exception: it is a
durable, shareable input and is committed (it already carries a real frontier
datum — the GLM-5.2 CPU-serve throughput, enqueued without a recompile).

## Not to be confused with

- **`fak loop`** re-runs a prompt/slash-command on a wall-clock interval.
  `nightrun` iterates over *data-collection tasks*, not prompts.
- **`internal/witness` / `dos verify`** resolve a shipped *claim* from git
  evidence. `nightrun`'s "acceptance" is the artifact that proves a datum was
  *gathered* — a different thing.
- **`fak cadence`** trends project *progress* (scores/work/releases). `nightrun`
  trends *data collection*.

The agent-facing entry point is the `/run-it-all-night` skill, which wraps these
commands with the operating discipline.
