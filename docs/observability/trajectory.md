---
title: "fak trajectory observability primitives"
description: "How fak records agent turns, compares them by meaning, and lets custom scorers analyze trajectories without changing the kernel ABI."
---

# Trajectory observability — the data plane, the similarity primitive, and the seam

fak does not ship a trajectory-analysis product. It ships the three **primitives**
an analysis is built from, so you (or a trivial agent skill) can write your own
semantic, trajectory, memory, cache, or planner optimization on top of the kernel's
defaults — without forking the kernel.

The kernel already adjudicates every tool call and fans a typed lifecycle event to
any registered observer. What it lacked was an *analysis-shaped* view of that stream
and a way to compare turns by meaning rather than exact tokens. These three leaves
close that gap, each opt-in and each additive to the frozen ABI:

| primitive | package | what it gives you |
|---|---|---|
| **data plane** | `internal/trajectory` | a typed, exportable per-turn record folded from the kernel's event stream |
| **reference vector similarity** | `internal/simhash` | a deterministic, dependency-free embed + cosine + top-k, to find near-duplicates the lexical ranker misses |
| **scorer seam** | `internal/trajhook` | a pluggable registry of `Turn → Finding` scorers — attach your own analysis with no core edit |

The CLI surface is `fak traj` (`similar` / `cluster` / `score` / `gc` / `export`);
the reference application is the `trajectory-garden` skill.

---

## 1. The data plane — what a turn is

A **`trajectory.Turn`** is one analysis-shaped record of an agent action: the trace
it belongs to, its order within the trace, the human-meaningful query that drove it,
the tool and the kernel's verdict, the result taint, the digest identities, the
per-turn token/byte cost, and — optionally — a deterministic `simhash` embedding of
the query. It is deliberately *different* from a [decision-journal](https://github.com/anthony-chaudhary/fak/blob/main/internal/journal/journal.go)
row: the journal is the tamper-evident audit ledger (a verdict over a digest); a
`Turn` is the analysis surface (the query text, the cost, the cache shape, the
embedding). One proves what the kernel decided; the other lets you find the bad
trajectories.

The export schema is stable JSONL, one Turn per line:

```json
{"trace_id":"sess-a","seq":1,"query":"search the knowledge base for the refund policy","tool":"search_kb","verdict":"ALLOW","token_estimate":320}
{"trace_id":"sess-a","seq":2,"query":"refund the customer's last payment","tool":"refund_payment","verdict":"DENY","reason":"POLICY_BLOCK","token_estimate":210}
```

### How a turn is recorded without touching the ABI

A `trajectory.Recorder` is an `abi.Emitter`. The kernel already fans every lifecycle
transition to registered emitters, and `abi.Event` carries an **OPEN `Fields` map**
plus the call's **OPEN `Meta` map** — so the producer (the gateway / agent loop)
stamps the query text and per-turn cost into those open channels, and the Recorder
folds them into a Turn. No ABI field is added; the recorder reads only what is
already there, defaulting cleanly when a field is absent.

Recording is **off by default** (a benchmark should not pay to record). Turn it on
with an env toggle, exactly like the audit journal:

```bash
FAK_TRAJECTORY=1         # enable the recorder
FAK_TRAJECTORY_EMBED=1   # also stamp a simhash embedding on each turn's query
```

Programmatically, `trajectory.Enable(embedQueries)` registers one Recorder and
returns it; `trajectory.Default()` is the process-global instance a front door reads
and exports.

---

## 2. The reference vector-similarity primitive

`internal/simhash` is the answer to "find the bad trajectories or bad queries in a
useful way" — a **reference** vector similarity over tokens, not a learned model:

```go
v := simhash.Embed("delete every row in the production table") // deterministic, L2-normalized
c := simhash.Cosine(v, simhash.Embed("drop all rows in prod")) // similarity in [-1, 1]

var ix simhash.Index
ix.AddText("q1", "refund the last payment", "")
matches := ix.TopK(simhash.Embed("issue a refund"), 5)          // k nearest by cosine
```

It is a hashing-trick sketch over word unigrams/bigrams and character 3-grams:
deterministic (same text → same vector, on any platform, no RNG, no model),
dependency-free, and cheap. That makes it good enough to catch near-duplicate
queries and outlier trajectories on day one.

**Know its ceiling.** Because the features are lexical, two queries that mean the
same thing in *different vocabulary* score low:

| query A | query B | cosine |
|---|---|---|
| search the knowledge base for the refund policy | look up the refund policy in the knowledge base | **0.75** |
| please refund the customer's last payment | refund the customers last payment please | **0.70** |
| delete every row in the production users table | drop all records from the prod users table | **0.35** |

Shared-vocabulary paraphrases cluster ~0.70–0.78; same-intent / different-word pairs
fall to ~0.35. The default duplicate threshold (0.70) is calibrated to that reality —
it catches the real redundancy without firing on distinct work. When your corpus has
heavy same-intent/different-vocabulary redundancy, that is the signal to **swap in
real embeddings**: build a `simhash.Index` from a sentence embedder's `[]float32`
(the Index machinery is model-agnostic) or register your own scorer. The reference
primitive is the floor, not the ceiling — and the swap is the whole point.

---

## 3. The scorer seam — attach your own analysis

`internal/trajhook` is the application-layer extension point. A `Scorer` is a pure
function from a `trajectory.Turn` (with the whole corpus as context) to zero or more
`Finding`s; a `CorpusScorer` scores the whole corpus at once. You register named
scorers into a `Registry` and run them — the same "register a driver, don't edit the
core" discipline the kernel uses for `abi.Emitter`, lifted to the analysis layer
where no ABI is involved at all.

```go
reg := trajhook.NewRegistry()
reg.Register("my_regression", func(t trajectory.Turn, corpus []trajectory.Turn) []trajhook.Finding {
    // your analysis here — flag t against corpus and return findings
    return nil
})
findings := reg.Run(recorder.Turns()) // worst-first
```

Three **reference** scorers ship as worked examples (`trajhook.Default()`):

- **`duplicate_query`** — a turn whose query near-duplicates an *earlier* one
  (simhash cosine ≥ threshold). The redundancy a lexical ranker misses.
- **`cost_outlier`** — a turn in the expensive tail of the token distribution. Where
  the context budget went.
- **`high_deny_rate`** — a *trace* the kernel refused on ≥50% of turns. A confused or
  adversarial loop worth a human look.

They are examples, not policy. fak deliberately does **not** ship a learned
"bad-trajectory classifier" — that judgment is application-specific. What fak ships
is the substrate that makes one a few lines to write.

---

## 4. The CLI — `fak traj`

Gardening verbs over an exported corpus. Every verb reads a corpus file; none
mutates it (`gc` *proposes*, it never deletes — fak never removes a user's
trajectory data).

```bash
CORPUS=examples/trajectory/sample-corpus.jsonl

fak traj score   --corpus "$CORPUS"                          # run the reference scorers, worst-first
fak traj similar --corpus "$CORPUS" --query "issue a refund" # k most-similar past queries
fak traj cluster --corpus "$CORPUS"                          # group near-duplicate queries
fak traj gc      --corpus "$CORPUS" --json                   # propose prune candidates
fak traj export  --corpus "$CORPUS"                          # re-emit normalized JSONL
```

`examples/trajectory/sample-corpus.jsonl` is a shipped demo corpus to rehearse the
verbs with no live kernel.

---

## 5. The reference application — the `trajectory-garden` skill

`.claude/skills/trajectory-garden/` is a trivial agent skill that drives `fak traj
score` + `gc` to find redundant / bad / expensive trajectories and propose prunes.
It is the proof of the thesis: a relatively trivial skill can do real memory /
trajectory gardening *on top of* the primitives — work that wasn't possible before
fak expressed the data plane, the similarity primitive, and the seam. Fork it to
build your own analysis; swap the scorer and you have a different tool.

---

## Why this altitude

The goal was never for fak to ship the semantic layer. It was to express the
**primitives** at the kernel boundary — first-class data visibility and hooks — so
the semantic, trajectory, memory, cache, and planner optimizations are something you
and others write above the defaults, for a core use case or a one-off alike. The
data is typed and exportable; the similarity is a deterministic reference you can
replace; the seam takes your scorer with no core edit. That is the observability
layer; the analysis is yours to build.
