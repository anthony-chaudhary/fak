---
name: trajectory-garden
description: One repeatable gardening pass over a trajectory corpus — the JSONL of per-turn Turn rows a fak trajectory.Recorder exports. Uses the `fak traj` toolkit (the data plane + the simhash reference vector-similarity primitive + the pluggable scorer seam) to find the trajectories worth a human's attention — near-duplicate queries the lexical ranker misses, cost outliers, and traces the kernel kept refusing — then PROPOSES prune candidates (it never deletes). The reference application of fak's trajectory observability primitives: proof that a trivial skill can build memory/trajectory gardening ON TOP of the defaults, and the starting point you fork to add your own scorers. Use after a recording run, when a memory store has bloated with redundant work, or on a /loop cadence to keep an agent's trajectory memory lean and its bad-query clusters visible.
---

# trajectory-garden — find the bad/redundant trajectories, propose the prune, never delete

> **What this does.** An agent that records its trajectories accumulates two kinds
> of cruft: *redundant* work (the same question asked five different ways) and
> *bad* trajectories (a loop the kernel kept refusing, a turn that burned a huge
> context budget). Left ungardened, both rot a memory store — the redundancy
> inflates retrieval cost, the bad trajectories poison few-shot recall. This pass
> turns "the memory feels messy" into a **concrete, sourced work-list**: the exact
> turns to look at, ranked worst-first, with a proposed prune you approve before
> anything is removed.

The shape: **export/locate the corpus → score it with the reference scorers →
read the findings worst-first → propose prunes (`fak traj gc`) → hand the operator
the prune list to approve.** Nothing here deletes a user's trajectory data — `fak`
ships the *observability*, the gardening *decision* stays with you.

This skill is also the **reference application** of fak's trajectory primitives
(`internal/simhash`, `internal/trajectory`, `internal/trajhook`, the `fak traj`
verbs). If you want a *different* analysis — a semantic-embedding dedupe, a
per-tool cost report, a regression detector over two corpora — fork the steps
below and swap the scorer. That swap is the whole point: fak gives you the data
and the seam, you write the policy.

---

## The one rule that overrides everything: PROPOSE, never delete

`fak traj gc` emits prune *candidates*. This skill surfaces them and stops. The
operator (or a downstream store with its own retention policy) decides what to
actually remove. fak never deletes trajectory data, and neither does this pass —
a wrong auto-prune is unrecoverable, and the cost of a missed prune is just a
slightly larger corpus. Always err toward keeping.

---

## What a "corpus" is

A trajectory corpus is the JSONL a `trajectory.Recorder` exports — one
`trajectory.Turn` per line (trace id, seq, query, tool, verdict, reason, taint,
token/byte cost, optional simhash embedding). Three ways one exists:

1. **A live fak kernel** with recording enabled (`FAK_TRAJECTORY=1`, and
   `FAK_TRAJECTORY_EMBED=1` to stamp query vectors): the recorder folds the
   adjudication stream into Turn rows, exportable via the gateway / a front door.
2. **An offline run** that drove a `trajectory.Recorder` and called `ExportTo`.
3. **The shipped demo** — `examples/trajectory/sample-corpus.jsonl` — to dry-run
   the verbs with no live kernel.

If no corpus exists yet, say so and point the operator at enabling
`FAK_TRAJECTORY`; do not fabricate one.

---

## The pass, step by step

Run from the repo root. Replace `CORPUS` with the corpus path (use the shipped
demo to rehearse).

```bash
CORPUS=examples/trajectory/sample-corpus.jsonl

# 1. Score the whole corpus with the three reference scorers, worst-first.
go run ./cmd/fak traj score --corpus "$CORPUS"

# 2. Look up whether a specific NEW query is redundant with past work.
go run ./cmd/fak traj similar --corpus "$CORPUS" --query "look up the refund policy" --k 5

# 3. Map the redundancy: cluster near-duplicate queries.
go run ./cmd/fak traj cluster --corpus "$CORPUS"

# 4. Propose prune candidates (later near-duplicates). JSON for a downstream store.
go run ./cmd/fak traj gc --corpus "$CORPUS" --json
```

`score` is the headline. It runs three reference scorers and lists findings
worst-first:

| label             | what it flags                                              | act on it by |
|-------------------|------------------------------------------------------------|--------------|
| `duplicate_query` | a turn whose query near-duplicates an EARLIER one (simhash)| prune the later one, or cache the result |
| `cost_outlier`    | a turn in the expensive tail of the token distribution     | investigate why it was so big; compress its inputs |
| `high_deny_rate`  | a TRACE the kernel refused on ≥50% of turns                 | read the trace — a confused or adversarial loop |

---

## Read the findings, then decide

For each finding worth acting on:

- **`duplicate_query`** → the later turn is redundant. If its result is cached,
  the future ask is a free hit; if not, it is a prune candidate. `fak traj gc`
  lists exactly these, each with the earlier turn it duplicates (`keep`) and the
  cosine. Hand the operator the `prune` keys; do not remove them yourself.
- **`cost_outlier`** → not cruft, a *signal*. A 9800-token turn is where the
  context budget went. Surface it; the fix is usually upstream (compress the tool
  output, page out the raw evidence) not a prune.
- **`high_deny_rate`** → a trajectory the kernel kept refusing. Read the trace's
  queries. A tight cluster of destructive variants ("delete rows" / "drop
  records" / "truncate") is an agent fighting the policy floor — worth a human
  look, never an auto-prune.

---

## Know the primitive's ceiling (and when to swap it out)

`simhash` is a **reference** vector-similarity primitive: a deterministic,
dependency-free hashing-trick sketch over word/char n-grams. It is honest about
its limits — near-duplicates that share most *words* score ~0.70–0.78, but two
queries that mean the same thing in *different vocabulary* ("delete every row" vs
"drop all records") score only ~0.35, below the default 0.70 duplicate threshold.
That is the floor of a lexical-feature hash, and it is by design: fak does not
ship a learned semantic model.

So: if your corpus has lots of same-intent / different-vocabulary redundancy that
the reference scorer misses, that is the signal to **swap in real embeddings** —
build a `simhash.Index` (or your own `[]float32` vectors) from a sentence
embedder and feed them through the same `fak traj` machinery, or register your own
`trajhook.Scorer`. The seam is built for exactly this; the reference scorer is the
floor, not the ceiling.

---

## Extend it (the reason this skill exists)

To add a new analysis, register a `trajhook.Scorer` (per-turn, sees the whole
corpus) or `trajhook.CorpusScorer` (whole-corpus) — a pure function from
`trajectory.Turn`(s) to `trajhook.Finding`s — and run it over the corpus. No core
edit, no ABI change: this is the application-layer seam. The three reference
scorers in `internal/trajhook/trajhook.go` (`DuplicateQuery`, `CostOutlier`,
`DenyRate`) are the worked examples you copy.

---

## What to report

A short operator note, not a wall:

- corpus path + turn/trace counts,
- the worst 5–10 findings (label, where, why),
- the prune candidates (count + the keys), framed as a **proposal to approve**,
- one line on whether the reference embedder is discriminating well on this corpus
  or it is time to swap in semantic vectors.

Stop there. The prune decision is the operator's.
