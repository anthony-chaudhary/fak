# Context-window baseline — Claude Code sessions

_Generated: 2026-06-23 · 20 heaviest sessions · 2,249,698 estimated context tokens (≈4 chars/tok)._

> Auto-generated snapshot — regenerate with `python tools/ctxwin.py baseline --md docs/notes/CTXWIN-CONTEXT-WINDOW-BASELINE-<date>.md`. Don't hand-edit. Numbers are from this box's own sessions; aggregate-only (no transcript content or paths).

## What the window is made of

| block kind | % of context |
|---|---|
| result | 62.2% |
| tool_use | 32.3% |
| text | 5.5% |
| thinking | 0.0% |

**Tool results are 62.2% of the window.** Result mass by tool:

| tool | % of result mass |
|---|---|
| Read | 60.6% |
| Bash | 26.7% |
| mcp__playwright__browser_snapshot | 2.3% |
| Agent | 2.1% |
| Grep | 1.8% |
| Edit | 1.6% |
| TaskOutput | 0.8% |
| Write | 0.5% |

## How much is reducible — by RISK tier

| tier | risk | % of context | reduction |
|---|---|---|---|
| character noise (ANSI/whitespace/repeated lines) | none | 0.14% | 1.001× |
| **stale reads** (a Read the agent later Edited/Wrote) | low (recoverable) | **13.37%** | **1.154×** |
| exact duplicates | none | 0.75% | 1.008× |
| **LOW-RISK cumulative** (noise+stale+exact-dedup, zero relevance guessing) | low | **14.3%** | **1.166×** |

- **Line-level noise** (informational, overlaps the tiers above): cat-n line-number prefixes 2.52% (Edit-targeting *signal*, not free to drop) + cross-result repeated lines 2.78% + character noise 0.14% ≈ **5.45%** combined. So "strip the obvious noise" is ~5.45%, not nothing — but still **far below the stale-read share**. The dominant low-risk lever is **whole stale results** (files the agent read, then changed): superseded by construction, recoverable, zero relevance guessing.
- **Per-item windowing (bounded-loss)** carries the rest of the way — cap each surviving tool_result OR tool_use input to budget B, keep head+tail, elide the middle to a recoverable pointer:

| budget B (tok) | windowing saves | reduction |
|---|---|---|
| 2000 | 21.6% | 1.275× |
| 1000 | 33.5% | 1.504× |
| 700 | 40.1% | 1.67× |
| 500 | 46.4% | 1.867× |

- **Recency:** reducible mass by quartile (oldest→newest) = [16.9, 17.4, 47.2, 18.6]%; the oldest half holds **34.2%** — old items are the safest to collapse.

- **Windowing headroom by tool** (% of context held over a 700-tok cap — who to target with `--per-tool`): Read 26.5%, Write 5.3%, Bash 3.2%, mcp__playwright__browser_snapshot 1.1%, Agent 1.0%, Workflow 1.0%, Edit 0.9%, TaskOutput 0.4%.
- **Error-shaped results:** 0.87% of the window across 150 results carry a structural `is_error` flag or a head error marker. `--error-collapse hyper` reclaims most of this by collapsing a failed-call body to one line — **lossy, not file-recoverable**, so it is OFF in every profile but `aggressive`/`hyper`.

## Tuning menu — what each profile does to THIS corpus

The reducer is a dial. `off` keeps everything; `hyper` collapses a 30k error to one line. `recoverable` is the honest cost — the fraction of removed bytes with a direct re-fetch handle, which falls as the lossy error tier kicks in.

| profile | reduction | recoverable | lossy tok removed | error-collapse |
|---|---|---|---|---|
| `off` | 1.0× | 100% | 0 | off |
| `conservative` | 1.1651× | 100% | 0 | off |
| `balanced` | 2.0001× | 83% | 0 | off |
| `aggressive` | 3.0017× | 75% | 15,131 | head |
| `hyper` | 4.0018× | 70% | 16,458 | oneline |

> Compose finer with `--error-collapse off|head|oneline` and `--per-tool Bash=200 --per-tool TodoWrite=off`. The lossy levers stay OFF unless you ask; `recoverable` is MEASURED per profile, never asserted.

## Self-reduce to 2×

Low-risk tiers first (noise + stale + exact-dedup), then a uniform per-result budget of **541.5 tok** → **2.0001× reduction**. The low-risk tiers alone reach 1.1651×; the rest is bounded windowing. Every result keeps a real head+tail or a pointer (**nothing is dropped to zero**), and **83% of removed bytes have a direct re-fetch handle** (file-backed or redundant); the remainder is the elided middle of non-file results, recoverable via the production CAS page-back.

> Estimate-over-estimate ratios (≈4 chars/tok). This is the empirical pass over REAL Claude Code `.jsonl` transcripts that the `ctxplan` planned-view note (`docs/notes/O1-TURN-CONTEXT-PLANNER-2026-06-23.md`) names as its missing measurement: the stale-read and recency tiers are the resident-redundancy signals `ctxplan`'s benefit model exploits, and "recoverable" here is `ctxplan`'s **Faithful** (every elided span keeps a page-back handle). The result-level windowing maps to `internal/ctxmmu` Transform (oversize→recoverable CAS pointer). Both must be applied prefix-stably at the gateway seam (`fak guard`, residual #555) so the ~90% prompt cache survives.

