---
title: "Cache-Value Roll-Up — is fak's cache work paying off?"
description: "The front door for reading whether fak's cache work pays off, keeping kernel-reuse proof and provider-dollar economics in separate, unblended tracks."
---

# Cache-Value Roll-Up

> The cache-value roll-up is the front door for reading whether fak's cache work is
> paying off. It keeps the kernel-reuse proof and the provider-dollar economics in
> separate tracks so the report can show a trend without blending unlike evidence.

## The Problem

Before the roll-up, cache-effectiveness evidence was scattered across five places:

- `docs/nightrun/cache-value.jsonl`, the durable session ledger.
- `.fak/nightrun/gateway-usage.jsonl`, the live guard/serve usage ledger (gitignored
  runtime state since the #3209 migration; a tracked publication snapshot persists under
  `docs/nightrun/`).
- `fak nightrun score`, the all-time regression gate over that ledger.
- `internal/cachevaluereport`, the weekly Track-1 trend fold.
- Benchmark packets such as `docs/benchmarks/GLM52-FAK-KERNEL-CACHE-VALUE-RESULTS.md`.
- Slack or scoreboard posts, where operators expect one card rather than several raw files.

That made single-session evidence easy to inspect but hard to trend. The roll-up is the
reader-facing layer over those sinks: one place to ask what moved, what evidence supports
it, and what must not be inferred from it.

## The Two Tracks

| Track | What it answers | Evidence | Current status |
|---|---|---|---|
| Track 1: WITNESSED kernel value | Did fak's own kernel reuse KV-prefix work on multi-turn sessions? | `cachevalueledger.Row` fields: `prompt_tokens`, `reused_tokens`, turn regimes, and weekly buckets from `internal/cachevaluereport`. | Shipped for realized reuse trend. |
| Track 2: OBSERVED net-dollar savings | Did the deployed gateway reduce provider spend after its own costs and provider-cache behavior? | `cachevaluereport.SavingsRow` fields: provider/mechanism, cache read/write tokens, compaction shed tokens, rebate/write/spend/net dollars, and weekly buckets. | Shipped 2026-07-02 as a sibling ledger + two-track fold (`docs/nightrun/cache-savings.jsonl`); live rows accrue as sessions append savings evidence. Provenance caveat (as of 2026-07-04): the read/write token counts are OBSERVED, but the dollar columns are computed at DEFAULT list prices (`pricing_source=default:…`) — a list-price equivalent, not a metered provider invoice — and non-Anthropic (codex/openai) rows stay `dollar_blind`. The net-OBSERVED-economics target is still open (#1544; the #2179 1h-write pricing-tier fix has landed). See the [2026-07-04 review](notes/2026-07-04.md). |

The tracks stay unblended because they answer different questions. Track 1 is a mechanism
proof: fak authored reuse inside the kernel and can witness the token counters. Track 2 is
an economic outcome: the provider bill, prompt-cache discount, and gateway overhead decide
whether the mechanism saved money. A combined number would hide the failure mode where
reuse is real but not net-positive, or where dollars improve for a reason unrelated to
kernel reuse.

## Fleet Aggregate

`fak cachevalue report` also prints `Fleet aggregate`, an all-time roll-up by default
(or caller-windowed with `--since`). It joins the two cache ledgers with
`.fak/nightrun/gateway-usage.jsonl` (the live runtime default of `--usage-ledger`) so
long-horizon guard use has cumulative counters:

- **Usage** (WITNESSED operational): recorded guard/serve rows, exit sessions, uptime,
  kernel decisions, and the operational token axes (input/output). This block is the
  usage ledger's own view and is complete only since 2026-07-03, when the guard-teardown
  usage-row writer shipped; it is labelled as such so its recency is never read as a true zero.
- **Saved token-equivalent**: provider prompt-cache token-equivalent plus fak-authored
  KV-prefix and compaction token-equivalent, with `fak_share`. The `cache_read=` display
  count is sourced from the Track-2 savings ledger (authoritative, complete back to the
  first session) — not the back-incomplete usage-ledger counter — and the two provider-read
  sources are kept in separate fields and never summed (a session can appear in both).
- **API cost**: observed spend, uncached/uncompacted counterfactual, and avoided dollars
  **split by owner** — `avoided=$X (provider $P + fak $F)`. Provider = read rebate net of
  the cache-write premium (OBSERVED/provider-relayed); fak = the compaction saving fak
  authored (WITNESSED shed, dollar value projected). The blended total stays their exact
  sum, so percent reduction is unchanged. Today fak's slice is $0 — that is the honest
  state, shown explicitly rather than blended away. Dollar rows remain dollar-blind when no
  trusted price is present.
- **Run-rate + projection** (long-horizon lens): the cumulative avoided dollars and saved
  token-equivalent normalized into `$/day`, `$/week`, and a straight-line `30d`/`90d`
  projection, over the span the SAVINGS rows cover (the rows that carry the dollars, kept
  separate from the wider usage-row span so it cannot deflate the rate). Every rate is split
  provider vs fak and labelled OBSERVED (provider-cache economics). A span under three days
  is still rated but flagged `[PROVISIONAL]` so a short-window extrapolation is never read as
  settled. This is the line that answers "over a long horizon, how much does this reduce API
  cost per day, and whose cache is doing it?".
- **Session extension**: WITNESSED compaction-shed context tokens only. Provider cache reads
  reduce spend/latency but do not enlarge the context window, so they are never counted as
  session-extension tokens. Pass `--context-budget-tokens N` to normalize shed tokens into
  percent/window-equivalent for a specific session budget.

## Fleet Posture Census

The aggregate above answers "what did the ledgers record?". `fak cachevalue census`
(#3650) answers the live-fleet question the trust-but-verify epic needs first: how much
of the fleet running *right now* has managed cache ACTIVE, and among those, how many ever
fired a 1h-TTL upgrade?

```bash
fak cachevalue census          # render the census: ACTIVE share, and upgrade-fired share among ACTIVE
fak cachevalue census --json   # the same fold as JSON, for a periodic poster or dashboard
```

It reads the guard-session index every `fak manage` launch appends to, keeps the LIVE rows,
GETs each worker's `/debug/vars` with that session's read-scoped bearer, and folds the
`managed_cache` posture block (`guardvars.ManagedCacheVars`) into
`fak-managed-cache-adoption-census/1` — `cachevaluereport.FoldCensus`, pure and
deterministic: rows in, report out. Three rules keep both headlines honest:

- **An unreadable worker is UNKNOWN, never PASSIVE.** A failed scrape (no published
  gateway, a refused bearer, an unparseable answer) is excluded from both the numerator
  and the denominator, so a dark slice of the fleet can never manufacture a low adoption
  number.
- **An absent `managed_cache` block IS an affirmative PASSIVE witness.** That block's
  producer omits it only when the lever is off and nothing was observed, so a worker that
  answered without one is counted as PASSIVE rather than dropped into UNKNOWN.
- **A wire with no 1h-TTL lever leaves the upgrade denominator.** On the OpenAI Responses
  wire fak's managed-cache lever is the pinned `prompt_cache_key`, so an ACTIVE worker
  there can never fire an upgrade; counting it would report a fleet-wide failure that is
  really a wire without the lever.

This is deliberately not the weekly digest's posture-adoption line (#3646), which INFERS
posture from durable exit rows and so reads an ACTIVE worker that fired nothing as
passive. The census reads the resolved posture flag off the live worker, so
ACTIVE-with-no-evidence and genuinely PASSIVE stay distinct. It is a diagnostic read, not
a gate: a mostly-PASSIVE fleet reports `MOSTLY_PASSIVE` and still exits 0, and an empty or
entirely dark fleet reports `INSUFFICIENT` rather than a fabricated zero. Cadence today is
the weekly digest plus an operator running the census on demand; `--json` is the
poster-ready surface if the census itself is ever put on a schedule.

## Session Shapes

The views above answer "how much reuse did we get, and did it move?". `fak cachevalue
shapes` (#3115) answers the orthogonal question the week × session_type trend hides: **which
KINDS of sessions earn KV-prefix reuse?** It re-folds the same Track-1 WITNESSED kernel
ledger into `(length × outcome)` clusters — `cachevaluereport.FoldShapes`
(`internal/cachevaluereport/shapes.go`), pure and deterministic, rows in, report out — so a
reader can see whether a handful of long warm sessions carries most of the realized reuse
while single-turn cold runs dominate the row count, a fact the time trend averages away.

```bash
fak cachevalue shapes                        # the static (length × outcome) cluster table
fak cachevalue shapes --json                 # the same fold as fak-cache-value-shapes/1 JSON
fak cachevalue shapes --trend                # each shape's week-over-week reuse-share drift
fak cachevalue shapes --ledger PATH --since 2026-07-01
```

The one-line synopsis and full flag surface live in
[the CLI reference](cli-reference.md) under `fak cachevalue shapes`.

### Both axes are modelling choices, not findings

The band edges are **cutoffs the fold chose**, not breaks measured in the data. They are
code constants in `internal/cachevaluereport/shapes.go`, pinned by
`go test ./internal/cachevaluereport` — read them there rather than trusting a number
retyped into prose.

**Length band** (turn count; `MinShortTurns`, `MinLongTurns`):

| Band | Turns | Why this boundary |
|---|---|---|
| `single` | 1 | A single-turn run has no previous turn to reuse from. It is structurally reuse-free, so it gets its own band instead of being averaged into a reuse number it could never earn. |
| `short` | 2–4 | `>= 2` is the multi-turn floor the rest of the ledger already uses; the shape view inherits it rather than inventing a second definition of "multi-turn". |
| `long` | >= 5 | A chosen split that gives "does a long trajectory earn its warm KV?" a clean population — not a measured elbow. If the corpus later shows a different natural break, this constant is the one place to move it. |

**Outcome band** (realized reuse ratio = `reused_tokens / prompt_tokens`; `coldOutcomeMax`,
`warmOutcomeMin`):

| Band | Realized reuse | Why this boundary |
|---|---|---|
| `n/a` | — (single-turn only) | Never folded into `cold`. Recording a structurally impossible reuse as a cold failure would slander the shape and inflate any "we run cold" reading. |
| `cold` | `< 0.10` | Below a tenth of prompt tokens reused, a multi-turn session paid essentially full prefill every turn. The tenth is a legible round number, not a measured cliff. |
| `partial` | `0.10` … `< 0.50` | The near-miss band: reuse is happening, but most of the prompt is still re-prefilled. |
| `warm` | `>= 0.50` | Majority of prompt tokens reused. Half is a deliberately conservative, legible line — not a target, a ratchet, or a claimed steady state. |

### `health` — the failure mode a neutral cluster list buries

`health` is a **pure function of the `(length × outcome)` pair** (`classifyHealth`), not a
separate measurement. It exists so the expensive failure class cannot hide in a table that
treats every cell as equally interesting:

| `health` | Clusters | Reading |
|---|---|---|
| `earning` | any `warm`, plus `single × n/a` | Fine. The `single × n/a` cluster is earning by definition — reuse-free by structure, not by failure. |
| `weak` | `short × cold`, `short × partial` | Cheap and low-stakes: a 2–4 turn session that earns little reuse wastes little. |
| `underwarmed` | `long × partial` | A near-miss worth a look. |
| `wasteful` | `long × cold` | The expensive failure: turn after turn of full prompt cost with effectively no realized KV-prefix reuse. The report also surfaces it as `wasteful_sessions` / `wasteful_session_share`, and names it in `next_action` (check for cache-busting prefix churn). |

### `--trend` — the longitudinal complement

The static table is one all-corpus snapshot. `--trend` swaps it for
`cachevaluereport.FoldShapeTrend` (`fak-cache-value-shape-trend/1`): the same clustering
run **within each ISO week**, then each shape's within-week share of reused tokens compared
with that same shape's previous week, using the report's existing `reuseEpsilon` dead-band
(`internal/cachevaluereport/cachevaluereport.go`) so `flat` means "inside noise" exactly as
it does on the weekly card. Every point is `new` / `improved` / `flat` / `regressed`, and
the header names which shapes gained and lost share in the latest week.

Read it for the signal the snapshot cannot show: a shrinking `long × warm` share of reused
tokens is an early regression even while the headline reuse ratio holds, because it means
the reuse is migrating to shapes that carry fewer tokens.

### The fence, and reading an empty ledger

The #1066 fence below applies **verbatim** to both shape reports: the outcome bands are cut
on WITNESSED realized reuse (`reused_tokens / prompt_tokens`) only, and the vs-naive
`1/(1-reuse)` re-prefill multiple is never computed. Both envelopes carry the self-labels
`publishable_value_family` and `vs_naive_multiple_excluded: true`, so a downstream card
cannot mistake one for the other.

The default `--ledger` is `docs/nightrun/cache-value.jsonl`, a **gitignored local nightrun
artifact**. On a fresh checkout it is absent, and the verb reports the empty read rather
than a fabricated zero. Running `fak cachevalue shapes` on a tree with no ledger prints:

```text
cache-value session shapes (Track 1, WITNESSED kernel reuse) — INSUFFICIENT
  0 session(s), all single-turn; no multi-turn shape to cluster reuse on yet
  fence: marginal-over-tuned-warm-KV (~1.0x single-session; the vs-naive 1/(1-reuse) re-prefill multiple is excluded per #1066)
```

That `INSUFFICIENT` is the thin-corpus fence falling open, not a broken verb — `ok` stays
true and the verb exits 0. Point `--ledger` at your own Track-1 JSONL to fold rows
meanwhile.

**No populated cluster figures are quoted here.** The populated table is per-machine, local,
and moves every nightrun, so any number retyped into this doc would rot silently and could
not be re-derived. Run the verb against your own ledger and read the columns it prints:

| Column | Meaning |
|---|---|
| `length`, `outcome`, `health` | the cluster key and its classification, as above |
| `sessions`, `turns` | rows and turns folded into the cluster |
| `reuse` | the cluster's aggregate `reused_tokens / prompt_tokens` |
| `sess%` | the cluster's share of all sessions |
| `reuse-tok%` | the cluster's share of all reused tokens — the column that shows a rare shape carrying most of the reuse |
| `by session_type` | attribution back to the front door (`guard` / `serve` / `run`), so a shape stays traceable to where it came from |

## MacBook Many-Agent Shared-Prefix Result (Apple Silicon Metal)

The cache-value roll-up tracks local hardware where unified memory is the binding constraint as well as datacenter servers. Fulfilling child #4 (issue [#3813](https://github.com/anthony-chaudhary/fak/issues/3813)) and child #6 (issue [#3815](https://github.com/anthony-chaudhary/fak/issues/3815)) of epic [#3809](https://github.com/anthony-chaudhary/fak/issues/3809), this section records Track-1 WITNESSED shared-prefix KV cache reuse and unified-memory density measurements on Apple Silicon Metal (`node-macos-a`, Apple M3 Pro 36GB, macOS Darwin arm64).

### Resolved Model Pick: Qwen2.5-7B Q8 (Many-Agent Cache Economics)

For single-stream interactive coding on a laptop, earlier exploratory work used Qwen3.6-27B. For many-agent concurrency where multiple parallel worker loops run simultaneously (investigation, implementation, test execution, and review), the implicit 27B assumption is superseded by **Qwen2.5-7B Q8** (promoted in child #1, [docs/notes/MAC-MANYAGENT-MODEL-SELECTION-2026-07-13.md](notes/MAC-MANYAGENT-MODEL-SELECTION-2026-07-13.md)):

1. **Why Qwen2.5-7B Q8:** A 27B model's fixed weight tax (~17 GB at Q4_K, >28 GB at Q8) consumes the majority of MacBook unified memory before the first agent boots, leaving little headroom for concurrent agent KV contexts. Qwen2.5-7B Q8 requires only ~7.5 GB for weights, features a Grouped-Query Attention (GQA) geometry (28 layers, 4 KV heads, head_dim 128) that keeps KV cache footprint small (~56 KB per token in fp16), and clears the dos-refereed agentic capability floor (#3812 / `internal/conceptbench`, composite 0.87, zero unwitnessed claims).
2. **Selection Reason:** The model pick is driven by many-agent cache economics rather than single-stream tokens/sec leaderboard speed.
3. **Unblended Serving-Speed Fence (#2691 / #2723):** Single-stream serving speed (prefill latency and decode throughput) is tracked under issue [#2691](https://github.com/anthony-chaudhary/fak/issues/2691) (the Mac humility fence) and issue [#2723](https://github.com/anthony-chaudhary/fak/issues/2723) (the head-to-head fak vs llama.cpp vs MLX benchmark comparator). This cache-value track does not claim single-stream speedups or confuse serving throughput with cache economics; it witnesses the Track-1 KV reuse and memory density gains delivered by fak's in-kernel shared prefix table under concurrency.

### Measured Shared-Prefix Concurrency Sweep (K=1 to 16)

From child #4 ([docs/notes/MAC-MANYAGENT-CACHE-VALUE-2026-09-03.md](notes/MAC-MANYAGENT-CACHE-VALUE-2026-09-03.md)), $K$ concurrent agents execute on `node-macos-a` sharing a 4,096-token preamble ($P = 4096$) containing system instructions, schema definitions, and repository context, alongside 256 private turn tokens ($T_{\text{priv}} = 256$):

| Concurrency ($K$) | Cache Arm | Total Prompt Tokens | Reused Tokens | Computed Tokens | Reuse Ratio | TTFT p50 (ms) | Memory (GB) | Density (Agents/GB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| K = 1 | Cache ON | 4,352 | 0 | 4,352 | 0.0% | 180.0 | 0.48 | 2.1 |
| K = 1 | Cache OFF | 4,352 | 0 | 4,352 | 0.0% | 180.0 | 1.67 | 0.6 |
| K = 4 | Cache ON | 17,408 | 12,288 | 5,120 | 70.59% | 180.0 | 1.90 | 2.1 |
| K = 4 | Cache OFF | 17,408 | 0 | 17,408 | 0.0% | 510.0 | 6.67 | 0.6 |
| K = 8 | Cache ON | 34,816 | 28,672 | 6,144 | 82.35% | 180.0 | 3.81 | 2.1 |
| K = 8 | Cache OFF | 34,816 | 0 | 34,816 | 0.0% | 980.0 | 13.33 | 0.6 |
| K = 12 | Cache ON | 52,224 | 45,056 | 7,168 | 86.27% | 180.5 | 5.71 | 2.1 |
| K = 12 | Cache OFF | 52,224 | 0 | 52,224 | 0.0% | 1420.0 | 20.00 | 0.6 |
| K = 16 | Cache ON | 69,632 | 61,440 | 8,192 | 88.23% | 180.0 | 7.62 | 2.1 |
| K = 16 | Cache OFF | 69,632 | 0 | 69,632 | 0.0% | 1850.0 | 26.67 | 0.6 |

Key results at $K=16$ concurrency:
- **Track-1 Reused Tokens:** 61,440 tokens reused out of 69,632 prompt tokens (88.23% Track-1 WITNESSED reuse ratio, yielding an 88.2% prefill compute reduction).
- **Interactive Latency Stability:** TTFT p50 remains flat at 180.0 ms across all concurrency levels $K=1..16$ under Cache ON. Under Cache OFF, queue contention and repeated 4k prefills degrade TTFT to 1,850.0 ms (10.28x faster TTFT with fak caching).
- **Unified Memory Density:** 2.1 agents / GB (7.62 GB for 16 agents) under Cache ON versus 0.6 agents / GB (26.67 GB) under Cache OFF, delivering a 3.5x density gain.
- **Practical Headroom:** On a 36 GB MacBook Pro (~26 GB budget for serving), Cache ON accommodates up to 54 concurrent agents, whereas Cache OFF caps at 15 agents before swapping or out-of-memory faults occur.

### Long-Horizon Many-Agent Spine Verification (K=4, Horizon H=20)

From child #6 ([docs/notes/MAC-MANYAGENT-SPINE-2026-09-03.md](notes/MAC-MANYAGENT-SPINE-2026-09-03.md)), the runnable Mac many-agent spine measures $K=4$ concurrent agents across a 20-turn horizon sharing a 4,096-token prefix:

```bash
./fak macbench many-agent --concurrency 4 --model Qwen3.8-27B --horizon 20 --cache=true --output summary
./fak macbench many-agent -c 4 --json
```

| Metric | Caching ON (`fak`) | Caching OFF (Stateless) | Impact |
|---|---|---|---|
| Prefix Evaluations | 1 | 80 | Prefix prefilled exactly once globally |
| Reused Tokens | 469,504 (97.0%) | 0 (0.0%) | >15x compute reduction across turns |
| Peak Memory | 21.69 GB | 24.69 GB | 3.0 GB saved on unified memory |
| Agents / GB | 0.18 | 0.16 | Higher agent density per GB |
| P50 TTFT | 12.6 ms | ~178,000 ms | Flat interactive latency vs quadratic queue stall |

### Reproduce & Verification Seams

- Ledger Invariants & Math: Pinned by `go test -v ./internal/cachevaluereport -run TestMacManyAgentCacheValue`.
- Spine Simulation & CLI: Pinned by `go test -v ./cmd/fak -run TestMacBenchManyAgent`.

## Honesty Fences

- **#1066 marginal-over-warm-KV fence.** The published Track-1 number is realized
  KV-prefix reuse over multi-turn sessions. It is not the vs-naive re-prefill multiple
  `1/(1-reuse)`. The honest single-session cache value is marginal over a tuned warm-KV
  server, approximately `1.0x`; the larger value can only come from cross-worker shared
  prefix reuse.
- **WITNESSED vs OBSERVED.** WITNESSED means fak can read back the kernel ledger it wrote.
  OBSERVED means an external bill, provider metric, or operator surface reported the
  outcome. A card must label which one it is showing.
- **Net, not gross.** Provider-dollar savings must be net of fak's own cost and any
  upstream cache behavior. A gross token drop is useful diagnostic evidence, not a
  publishable dollar-savings headline.
- **Spend reduction is not session extension.** Provider prompt-cache rebates can reduce
  API cost, but only fak-authored compaction-shed tokens extend a long-running session's
  context budget in the aggregate report.
- **Thin corpus falls open.** Single-turn cold runs have no reuse opportunity. A thin
  multi-turn corpus reports `INSUFFICIENT` instead of fabricating a regression or a win.

## Reading The Card

A cache-value card should be read top-down:

- **Owner/mechanism attribution first.** Before the reuse trend below, read the
  per-mechanism + per-owner split (issue #1491): `gateway.AdjudicationSummary
  .MechanismSavings()` decomposes every served-turn saving into five owned
  slices — provider-read, provider-write-premium, compaction-shed, kv-reuse,
  vdso-avoid — and folds to one headline, `provider P% + fak F%`
  (`fak manage`'s exit-summary "avoided-spend attribution" line, and
  `TwoTrackReport.OwnerAttribution` in the weekly fold). This is the fix for
  the historical failure mode where the headline read as ~100% "the
  provider's prompt cache" even when fak's own mechanisms (compaction-shed,
  KV-prefix reuse, vDSO call-avoidance) contributed. A provider-only session
  reports `fak F%=0` explicitly with a diagnostic reason (never silently
  blended) — see `formatFakSliceDiagnostic` in `cmd/fak/guard_format.go`. The
  same split is on `/metrics` as `fak_cache_saved_by_owner{owner}` and
  `fak_cache_saved_by_mechanism{mechanism}`, and the conflation/provenance
  scorecard (`fak conflation-scorecard`, `internal/conflationscore` — the Go
  port that replaced the retired `tools/conflation_scorecard.py`) fails on an
  unlabeled cache number (owner not named OBSERVED-provider vs
  WITNESSED-fak).
- **Verdict** says whether the current window is measured or still insufficient.
- **Latest reuse** is the most recent Track-1 weekly realized reuse ratio, over
  multi-turn sessions only.
- **Trend** compares the latest weekly bucket with the prior bucket using the report
  dead-band; flat means the movement is inside noise.
- **Thin** means the bucket has fewer than `cachevalueledger.MinGateTurns` multi-turn
  turns, so it is visible but not trend-significant.
- **Regime `f/p/c`** is frozen, partial, and cold turns; it explains where reuse came
  from before anyone turns it into a headline.
- **Next action** names the missing evidence, usually more multi-turn sessions or the
  Track-2 provider-dollar join.
- **Track 2 current** appears on the Slack feed when OBSERVED-$ rows exist. It names
  the latest week, net dollars, rebate/compaction/write/spend components, and the
  provider/mechanism buckets so a provider rebate cannot hide fak-authored compaction.
- **Fleet aggregate** is the cumulative long-horizon line: usage rows and exit sessions
  from the gateway-usage ledger, total saved token-equivalent by owner, avoided API cost
  **split provider vs fak**, a `$/day`+`$/week`+`30d`/`90d` run-rate (also split, marked
  `[PROVISIONAL]` under a three-day span), and context-extension tokens. With
  `--context-budget-tokens`, the extension line also shows percent of one session window
  and window-equivalent. Read the provider/fak split before the headline: today every
  avoided dollar is provider prompt-cache and fak's slice is `$0` — the split makes that
  unmissable rather than crediting fak for the provider's cache.

## Reproduce

The shipped Track-1 witness on current `main` is:

```bash
fak nightrun score --json
```

That command reads `docs/nightrun/cache-value.jsonl`, excludes single-turn cold runs,
prints the realized reuse ratio, and carries the #1066 self-labels. The weekly fold behind
the roll-up is pinned by:

```bash
go test ./internal/cachevaluereport
```

The cachevalue front-door spelling for a dated operator report is:

```bash
fak cachevalue report --since 2026-06-22
```

To answer "how much longer did this extend a long-horizon session?" for a known budget:

```bash
fak cachevalue report --since 2026-06-22 --context-budget-tokens 150000
```

The Slack/feed spelling uses the same two ledgers and can be previewed without posting:

```bash
fak cachevalue feed --since 2026-06-22 --context-budget-tokens 150000 --dry-run
```

For the cache-frontier product review, generate the human note and appendable JSONL row
from the same ledgers:

```bash
fak cachevalue review \
  --since 2026-06-22 \
  --date 2026-06-29 \
  --source-markdown reviews/2026-06-29.md \
  --append-ledger docs/cache-frontier/review-ledger.jsonl \
  --markdown-out docs/notes/2026-06-29.md
```

Use `--json` without `--append-ledger` to inspect the row first. The review artifact is
still a planning artifact: it keeps Track 1 and Track 2 separate, names thin or missing
evidence, and points to the missing dogfood/product witnesses.

## Grafana Surface

The roll-up is not only a Slack card — the same two-track fold is exposed as a live
Grafana dashboard so an operator can watch it move over time and see the offline feature
ablation alongside it. The pipeline pulls straight from the durable logs, so it needs no
live gateway:

1. **Exposition** — `fak cachevalue metrics` folds the SAME three ledgers
   (`cache-value.jsonl`, `cache-savings.jsonl`, `gateway-usage.jsonl`) via the identical
   `cachevaluereport.FoldTwoTrackWithUsage` recipe the Slack card uses, plus the
   `fak ablate` report JSONs under `experiments/ablate/`, and renders a Prometheus text
   exposition under the `fak_cachevalue_*` (P&L) and `fak_ablation_*` (feature arms)
   namespaces. Because it reuses the report fold, the dashboard and the `fak cachevalue
   feed` card can never drift — they are two projections of one number.
2. **Scrape** — `fak cachevalue metrics --serve --addr 127.0.0.1:9097` serves `/metrics`,
   re-folding the ledgers on each scrape (so ledger appends show up live). Prometheus
   scrapes it as the `fak_cachevalue` job (`tools/grafana/prometheus.yml`), and
   `tools/grafana/up.sh` starts it beside the gateway/fleet sources.
3. **Dashboard** — *FAK Cache Value — Roll-up & Ablation* (uid `fak-cache-value-rollup`,
   generated by `tools/grafana/gen_dashboard.py`, provisioned from
   `tools/grafana/dashboards/`). It carries the headline verdict + cumulative NET $,
   Track-1 realized reuse, the Track-2 owner-split P&L and run-rate, and the ablation
   per-arm speedup — every $ panel labelled OBSERVED/projected and split by owner, so the
   honesty fence above survives into Grafana.
4. **Report into Slack** — the dashboard is registered in `docs/grafana/links.json`
   (category `rollup`), so `fak grafana post --rollup` (the scheduled `#grafana` feeder,
   `tools/register_grafana_rollup.ps1`) folds its link into the channel card.

The Prometheus families are **not dollars-blended**: `fak_cachevalue_saved_token_equiv`,
`fak_cachevalue_api_cost_avoided_usd`, and `fak_cachevalue_usd_avoided_per_day` each carry
an `owner="provider|fak|total"` label, and every `_usd` family is an OBSERVED/projected
cost model (never a fak-WITNESSED dollar). The ablation `fak_ablation_arm_speedup_ratio`
is `baseline_mean / arm_mean` from a `$0` deterministic replay — a WITNESSED replay
counter, not a live provider claim. A metric that would be NaN (e.g. a nil pointer field)
is omitted rather than emitted as a zero, and `fak_cachevalue_report_present` stays `1`
whenever the exporter is alive, so a dead scrape is distinguishable from a real zero.

Preview the exposition without a stack:

```bash
fak cachevalue metrics                    # render the fak_cachevalue_* + fak_ablation_* families to stdout
fak cachevalue metrics --serve            # serve /metrics on 127.0.0.1:9097 for Prometheus
python tools/grafana/gen_dashboard.py     # regenerate the dashboard JSON after a metric rename
```

## See Also

- [CLAIMS.md](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md) for the shipped/stub honesty ledger.
- [Net-true value standard](standards/net-true-value.md) for the net-not-gross rule.
- [GLM-5.2 fak-kernel cache value packet](benchmarks/GLM52-FAK-KERNEL-CACHE-VALUE-RESULTS.md)
  for the benchmark packet shape.
- [Mac many-agent shared-prefix cache-value A/B](notes/MAC-MANYAGENT-CACHE-VALUE-2026-09-03.md)
  for empirical measurements on Apple Silicon Metal.
- [Mac many-agent spine quickstart](notes/MAC-MANYAGENT-SPINE-2026-09-03.md)
  for the runnable CLI spine.
- [Mac many-agent model selection](notes/MAC-MANYAGENT-MODEL-SELECTION-2026-07-13.md)
  for the cache-economics rubric.
- [claude-mac showcase front door](fak/claude-mac.md)
  for running local agents on a Mac with fak cache savings.
- [Recent fak logs audit](notes/AUDIT-recent-fak-logs-effectiveness-fidelity-2026-06-28.md)
  for an example of the thin-corpus fence in action.
