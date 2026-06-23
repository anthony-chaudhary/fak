---
name: industry-score
description: One repeatable pass that keeps fak's competitive story honest AND complete — graded industry-first, not from what fak happened to measure. Runs the industry scorecard (tools/industry_scorecard.py) over a modular data directory (tools/industry_scorecard.data/): a researched taxonomy of the dimensions the LLM-serving / agent-infra field competes on (vLLM/SGLang/TensorRT-LLM/llama.cpp), the current SOTA bar on each with a dated source, and fak's honest position — mostly named gaps. It drives two numbers, coverage (of the field) and parity-debt (honesty of the rows), and updates on two cadences: as the industry moves (a new dimension drops coverage; --stale lists SOTA bars due a re-check) and as fak moves (a benchmark turns a no-claim into a measured row). Regenerates the modular doc folder docs/industry-scorecard/ and commits only the scorecard lane by explicit path. The OUTWARD-facing counterpart of repo-hygiene/code-quality/appeal. Use after a benchmark lands, when a competitor ships a number, when the field adds a dimension, or on a /loop cadence.
---

# industry-score — keep the competitive map complete and honest, and prove it

> **What this does.** A buyer, a reviewer, or a skeptic asks one question the
> inward scorecards never answer: *where does fak actually stand against the
> industry?* The trap is to answer only with the comparisons fak happened to run —
> a highlight reel that silently omits the 90% of the field it has no number on.
> This pass refuses that. It starts from a **researched taxonomy of the dimensions
> the LLM-serving / agent-infra field competes on** (continuous batching, FP8/FP4
> quant, speculative decoding, disaggregated prefill, multi-LoRA, tokens-per-watt,
> …), pins the current SOTA bar on each with a **dated source**, and positions fak
> honestly on every one — and for most, the honest answer is a **named gap**, not a
> win. It makes "our competitive map is complete and honest" a **repeatable,
> provable pass** instead of a prose claim that drifts, inflates, and goes stale.

The shape: **run the scorecard → close the worst coverage gap or retire the worst
parity-debt → keep the SOTA bars current (--stale) → verify sources → regenerate
the doc folder → commit only the scorecard lane.**

Two numbers are driven:

- **coverage** — of the industry dimensions that matter, how many the scorecard has
  considered and positioned fak against. Toward **100%**. A dimension with no fak
  position row at all is `coverage_debt`.
- **parity-debt** — of the comparisons that exist, how many are dishonest,
  incomplete, or unsourced. Kept at **0**.

It runs `tools/industry_scorecard.py` over the **modular data directory**
`tools/industry_scorecard.data/` (one concern per file) and folds into the unified
`tools/scorecard_control_pane.py` via `corpus.parity_debt`.

---

## The data directory (modular — one concern per file)

```
tools/industry_scorecard.data/
  _meta.json          meta (as_of, fak_version, freshness windows) + the category
                      vocabulary (id → doc group). A row/dim with an undeclared
                      category is MALFORMED — this is the data-defined closed vocab.
  _taxonomy.json      the INDUSTRY-FIRST dimension catalog: each dimension that
                      matters, its why-it-matters, the current SOTA bar, the leading
                      systems, and a source_url + source_date (the freshness anchor).
  _competitors.json   the SOTA-system registry, each with a last_reviewed date.
  rows-<group>.json   fak's position rows, grouped by doc group (serving, memory,
                      decoding, numerics, distributed, models, agent, cost,
                      operability, security). Each row's dim_id links to a taxonomy
                      dimension; most are honest no-claim gaps.
```

Edit a file, then re-run the tool. The generated doc folder `docs/industry-scorecard/`
(index + one page per group + the dimension catalog + the update process) is rebuilt
by `--markdown-dir`; never hand-edit it.

---

## The one rule that overrides everything: never invent a number or a win

This pass **records** evidence; it never **manufactures** it.

- **No new fak number.** Every fak figure must already exist in
  [`BENCHMARK-AUTHORITY.md`](../../../BENCHMARK-AUTHORITY.md) (a commit + a committed
  artifact) or a results doc. If a benchmark hasn't been run, the honest row is a
  `no-claim` gap — **that is a valid, complete position**, and most dimensions are
  exactly this. Coverage rewards positioning a dimension honestly; it never rewards a
  guess.
- **No new competitor number.** Every SOTA bar must cite a real `source_url` +
  `source_date` (vendor docs, a paper, a leaderboard, an on-box bench).
- **No win against a naive baseline.** State the gain vs the tuned/SOTA system
  (`competitor_class` `sota`/`tuned`), never a re-prefill strawman (`naive` is refused).
- **No regime is all-wins.** fak's committed artifacts contain the losses (the
  single-stream CPU decode/prefill trails, the fak-gateway 0.75× of raw SGLang on
  concurrent throughput, the 7B model-size ceiling). They are first-class `trails`
  rows, not buried fences.
- **Every verdict matches its ratio.** `lead` needs fak clearly ahead; `parity` ~±5%
  or a published band; `trails` clearly behind; an oracle can only be `parity`; a
  ceiling can never be `lead`; a `no-claim` only on a genuinely unbuilt
  (`stub`/`projected`/`in-flight`) axis. A capability lead/loss with no clean ratio is
  marked `qualitative: true` and must carry a `comparison_note`.

If a fix would require changing a measured number or inventing a win, **stop** — that
is out of scope. Re-measure through the benchmark-governance process first.

---

## Step 1 — Run the scorecard (it builds both work-lists)

From the repo root:

```bash
python tools/industry_scorecard.py                 # the scorecard + standing + both work-lists
python tools/industry_scorecard.py --json          # machine payload (the loop / control-pane)
python tools/industry_scorecard.py --gaps          # the COVERAGE backlog (what to position / measure)
python tools/industry_scorecard.py --stale         # the INDUSTRY-DRIFT backlog (SOTA bars due a re-check)
python tools/industry_scorecard.py --verify-sources # fak numbers still match their committed artifacts
```

It grades the honesty KPIs (structure · completeness · honesty · traceability) into
**parity-debt**, computes **coverage** of the taxonomy, blends them into a composite
score (0–100, A–F), and prints the data-derived standing (▲ lead · ≈ parity · ▼ trails
· ○ honest gap). It is read-only over the data; the only writes are the doc folder
under `--markdown-dir`.

## Step 2 — Pick the worst-first move

The scorecard names it. There are two failure modes and the verdict tells you which:

- **`coverage_debt` > 0** → the map is incomplete. Run `--gaps`; for each unpositioned
  industry dimension add an honest fak position row to the right `rows-<group>.json`
  (linking `dim_id` to the taxonomy). **Most will be `no-claim` gaps** — that is the
  honest, complete answer, not a cop-out. Reuse an existing committed number only where
  a head-to-head genuinely exists.
- **`parity_debt` > 0** → a row that exists is dishonest. Retire it worst-group-first
  using the table below.

| Defect | KPI / group | The honest fix |
|---|---|---|
| **Win vs a NAIVE baseline** | baseline_sota / honesty | Restate the gain vs the tuned/SOTA alternative, or drop the row. |
| **Verdict ≠ ratio** (the overclaim) | verdict_consistency / honesty | Set the verdict the numbers imply. Oracle → `parity`; ceiling → never `lead`; `no-claim` only on an unbuilt axis. |
| **Contracted regime missing / a loss hidden** | axis_coverage / completeness | Add the regime's row — especially the `trails` losses the contract requires (single-stream, real head-to-heads where fak trails). |
| **Blank competitor** | competitor_named / completeness | Name the concrete SOTA / next-best system. |
| **Non-comparable, undisclosed** | apples_disclosed / honesty | Set `apples_to_apples:false` AND add a `comparison_note` stating what differs. |
| **Measured verdict, no number** | well_formed / structure | Add the fak/competitor numbers; or, for a real capability lead/loss with no ratio, set `qualitative:true` + a `comparison_note`. |
| **Shipped but untraced** | fak_traced / traceability | Add a `fak_commit`, `fak_artifact`, or `fak_doc`. |
| **Competitor number unsourced** | competitor_sourced / traceability | Add `competitor_source` (paper / vendor doc / on-box bench). |

After a batch, **re-run the scorecard** and watch the number fall; that loop is the
method.

## Step 3 — Keep it current (the two update cadences)

This is what stops the scorecard rotting — in both directions.

**As the industry moves.** New techniques appear and published SOTA bars age.
- A **new dimension** the field starts competing on → add it to `_taxonomy.json`. It is
  immediately uncovered, so `coverage` drops and it shows in `--gaps` until fak is
  positioned on it. (This is the mechanism that surfaces "we're missing this.")
- A **stale SOTA bar** → `--stale` lists every dimension whose `source_date` (or
  competitor `last_reviewed`) is past the `industry_review_window_days` window. Re-check
  it on the web, update the number + bump the date. Advisory, never parity-debt — a
  number doesn't become false the day it crosses the window, it wants a look. For a
  larger sweep, fan out a workflow of web-grounded researchers (one per group) to
  refresh the SOTA bars + sources.

**As fak moves.** A benchmark lands → turn the relevant `no-claim` into a measured
`lead`/`parity`/`trails`, citing the commit/artifact, and bump `measured_on`. The
`freshness` KPI flags fak numbers older than `fresh_window_days` (advisory) to
re-confirm when a bench node is free.

## Step 4 — Verify sources

Run `python tools/industry_scorecard.py --verify-sources` — no present artifact may
`MISMATCH` (an absent one, on a bench node, skips; fine off-box). For any number you
hand-transcribed, spot-check it against the cited doc (or have a subagent re-read it).
A wrong number in an honesty instrument is self-defeating.

## Step 5 — Prove the drop, regenerate the doc folder

```bash
python tools/industry_scorecard.py --json > /tmp/after.json
python tools/industry_scorecard.py --compare /tmp/before.json    # parity-debt + coverage delta
python tools/industry_scorecard.py --markdown-dir docs/industry-scorecard --stamp $(date +%F)
```

State the before/after (e.g. "coverage 21% → 100%; parity-debt 0; standing 3 lead ·
11 parity · 8 trails · 65 honest gap"). Regenerate the committed doc folder from the
data — never hand-edit a page. Optionally re-run `python tools/scorecard_control_pane.py`
to confirm the portfolio still folds.

## Step 6 — Commit only the scorecard lane, by explicit path

This is a shared trunk; commit *your* lane, never a peer's work:

```bash
git commit -s -F <msgfile> -- \
  tools/industry_scorecard.py tools/industry_scorecard_test.py \
  tools/industry_scorecard.data docs/industry-scorecard
```

- **Stage by explicit path, never `git add -A`.** Stage and commit in one shell call.
- **Data + generated-doc diff → a `docs(benchmark): …` or `chore(scorecard): …`
  subject**, lead with a verb, end with the `(fak tools)` trailer. A code-effect prefix
  on a data/doc diff overclaims.
- **On Windows, pass the message via a file** (`-F`), not a here-string.
- **If a peer's `MERGE_HEAD` is set**, wait for it to clear; commit by explicit path.
- **Stay on the trunk (`main`)**; push promptly.

---

## Scope: what this pass touches, and what it must not

- **In scope:** the modular data dir `tools/industry_scorecard.data/` (taxonomy,
  competitor registry, position rows) and the generated `docs/industry-scorecard/`
  folder. Also `tools/industry_scorecard.py` + its test when a KPI needs tuning.
- **Out of scope: the fak NUMBERS.** Those are produced + committed through the
  benchmark-governance process ([`BENCHMARK-GOVERNANCE.md`](../../../BENCHMARK-GOVERNANCE.md))
  → recorded in [`BENCHMARK-AUTHORITY.md`](../../../BENCHMARK-AUTHORITY.md). This pass
  cites them; it does not measure or edit them. `CLAIMS.md` is the capability ledger,
  not a scorecard surface.

## When to run this

- After a new benchmark lands — turn a `no-claim` into a measured row.
- When a competitor ships a number that moves a SOTA bar (`--stale` finds the aging ones).
- When the field starts competing on a **new dimension** — add it to the taxonomy.
- On a `/loop` cadence to keep the front-door competitive map complete, honest, and current.

The scorecard is the competitive story's checking-layer, the way `refresh-readme` is the
README's freshness layer and `quality-score` is the code's. Same discipline: numbers you
can move, and prove you moved.
