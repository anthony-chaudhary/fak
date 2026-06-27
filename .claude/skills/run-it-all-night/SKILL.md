---
name: run-it-all-night
description: The agent door to unattended data collection — answer "what is the single most important datum I can collect on THIS box right now?" and then collect it on a loop. Wraps `fak nightrun`, which probes the local box (gpu/weights/datasets/creds), ranks the feasible-here collection tasks (the benchmark grid PLUS the curated open-witness backlog) by novelty × value × staleness, and closes the loop into a durable ledger (docs/nightrun/collected.jsonl). Use when the operator says "run it all night", "collect the next most important data", "what benchmark should I run on this machine", "start an overnight data-collection run", or when an agent on a fresh box (a Mac verify node, an A100, an H200) needs to know — without reading the whole repo — what data is worth gathering here.
allowed-tools: Read, Bash, Write
metadata:
  opencode: claude-only   # the commit-by-explicit-path discipline + the honesty boundary are load-bearing and not portable per-skill
---

# /run-it-all-night — the next() data-collection door

> Wraps the `fak nightrun` leaf (`internal/nightrun`). The whole point is one
> trivial answer for an operator OR an agent sitting on any box: **what is the
> single most important datum to collect HERE, right now, and the exact command
> to collect it** — then collect the whole feasible queue unattended, recording
> what was gathered into a durable ledger so the next night picks up where this
> one left off.

This is the front door over fak's existing collection parts (the benchmark
catalog `internal/benchcatalog`, the results grid `experiments/benchmark/`, the
remote-grid planner `tools/bench_plan.py`). It adds the piece those lacked: a
**local-capability-aware**, **loop-closing** selector that never proposes work
the box can't do.

## The honesty boundary (load-bearing — do not cross)

- `next` / `plan` / `caps` are **pure reads**. They never run anything.
- `run` is **DRY-RUN by default**: it prints what it *would* execute and writes
  nothing. Only `--apply` executes real commands.
- A task the box cannot run is **never selected**, so the loop can never claim to
  have collected HW-gated data on hardware that can't produce it.
- An `--apply` row records what was **OBSERVED** (exit status, artifact path, a
  parsed headline number only if one is actually present) — never a fabricated
  number. Report outcomes faithfully: if a run failed, say so with the artifact.

## Run it

1. **Orient the box.** `fak nightrun caps` — the one-line fact-sheet (gpu,
   weights, datasets, net, creds). This is *why* a task is or isn't feasible here.

2. **Ask next().** `fak nightrun next` — the single most important feasible datum,
   with the exact `run:` command and the `done:` acceptance criterion. For an
   agent, `fak nightrun next --json` gives the structured task to act on.

3. **See the whole night.** `fak nightrun plan` — the ranked queue (feasible
   first, then blocked tasks with the capability they wait on). Use this to decide
   whether tonight is worth a loop.

4. **Collect.** Preview first: `fak nightrun run` (dry-run). Then, when the
   operator approves executing real commands on this box:
   - one task: `fak nightrun run --apply`
   - the night: `fak nightrun run --apply --loop [--max N]`
   Each task is attempted at most once per invocation (a failing task can't spin
   the loop); every attempt appends an OBSERVED row to the ledger.

5. **Show what was gathered.** `fak nightrun ledger` — the durable collection
   history (newest first), and `--json` for the raw rows.

## Enqueue a new datum (no recompile)

When a NEW measurement becomes the most important thing to collect but isn't in
the built-in backlog yet, add it to the operator overlay
`experiments/nightrun/backlog.json` (a JSON array of tasks; `id` + `run` +
`value` + `requires` + `acceptance`). `fak nightrun` reads it additively over the
built-ins. Prefer this over editing the Go registry for a one-off; promote a
durable, recurring datum into `internal/nightrun/backlog.go` (the `witnessTasks`
registry) so it ships in the binary.

A task is **work to do**, never a result — it cannot overclaim. Keep the
`acceptance` honest (the artifact/number that proves the datum was gathered) and
point `doc` at the canonical issue/methodology.

## Committing (shared trunk)

If this pass changes tracked files (the overlay, the witness registry, the
ledger), commit **only those paths** on the trunk:

```
git commit -s -- experiments/nightrun/backlog.json   # or the specific paths you changed
```

Use a Conventional-Commits subject ending in a `(fak nightrun)` trailer. Never
`git add -A` (shared multi-session tree). The `collected.jsonl` ledger is durable
trunk evidence — append to it via `fak nightrun run --apply`, don't hand-edit.

## When NOT to use

- To re-run a prompt/slash-command on a wall-clock interval → that's `/loop`
  (`fak loop`), not nightrun. nightrun iterates over *data-collection tasks*.
- To verify a shipped CLAIM from git evidence → that's `internal/witness` /
  `dos verify`. nightrun's "acceptance" is the artifact that proves a datum was
  *gathered*, a different thing from a witness over a claim.
