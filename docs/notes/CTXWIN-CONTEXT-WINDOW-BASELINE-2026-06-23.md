# Context-window baseline — Claude Code sessions

_Generated: 2026-06-23 · 20 heaviest sessions · 2,447,916 estimated context tokens (≈4 chars/tok)._

> Auto-generated snapshot — regenerate with `python tools/ctxwin.py baseline --md docs/notes/CTXWIN-CONTEXT-WINDOW-BASELINE-<date>.md`. Don't hand-edit. Numbers are from this box's own sessions; aggregate-only (no transcript content or paths).

## What the window is made of

| block kind | % of context |
|---|---|
| result | 56.1% |
| tool_use | 33.6% |
| text | 10.3% |
| thinking | 0.0% |

**Tool results are 56.1% of the window.** Result mass by tool:

| tool | % of result mass |
|---|---|
| Read | 56.2% |
| Bash | 26.2% |
| PowerShell | 3.2% |
| Agent | 2.7% |
| Grep | 2.0% |
| TaskOutput | 1.9% |
| Edit | 1.7% |
| mcp__playwright__browser_snapshot | 1.5% |

## How much is reducible — by RISK tier

| tier | risk | % of context | reduction |
|---|---|---|---|
| character noise (ANSI/whitespace/repeated lines) | none | 0.23% | 1.002× |
| **stale reads** (a Read the agent later Edited/Wrote) | low (recoverable) | **13.21%** | **1.152×** |
| exact duplicates | none | 0.74% | 1.007× |
| **LOW-RISK cumulative** (noise+stale+exact-dedup, zero relevance guessing) | low | **14.2%** | **1.165×** |

- **Line-level noise** (informational, overlaps the tiers above): cat-n line-number prefixes 2.14% (Edit-targeting *signal*, not free to drop) + cross-result repeated lines 3.25% + character noise 0.23% ≈ **5.62%** combined. So "strip the obvious noise" is ~5.62%, not nothing — but still **far below the stale-read share**. The dominant low-risk lever is **whole stale results** (files the agent read, then changed): superseded by construction, recoverable, zero relevance guessing.
- **Per-item windowing (bounded-loss)** carries the rest of the way — cap each surviving tool_result OR tool_use input to budget B, keep head+tail, elide the middle to a recoverable pointer:

| budget B (tok) | windowing saves | reduction |
|---|---|---|
| 2000 | 18.3% | 1.223× |
| 1000 | 29.3% | 1.414× |
| 700 | 35.4% | 1.547× |
| 500 | 41.3% | 1.703× |

- **Recency:** reducible mass by quartile (oldest→newest) = [19.5, 18.0, 42.8, 19.8]%; the oldest half holds **37.4%** — old items are the safest to collapse.

## Self-reduce to 2×

Low-risk tiers first (noise + stale + exact-dedup), then a uniform per-result budget of **391.0 tok** → **2.0003× reduction**. The low-risk tiers alone reach 1.1626×; the rest is bounded windowing. Every result keeps a real head+tail or a pointer (**nothing is dropped to zero**), and **78% of removed bytes have a direct re-fetch handle** (file-backed or redundant); the remainder is the elided middle of non-file results, recoverable via the production CAS page-back.

> Estimate-over-estimate ratios (≈4 chars/tok). This is the empirical pass over REAL Claude Code `.jsonl` transcripts that the `ctxplan` planned-view note (`docs/notes/O1-TURN-CONTEXT-PLANNER-2026-06-23.md`) names as its missing measurement: the stale-read and recency tiers are the resident-redundancy signals `ctxplan`'s benefit model exploits, and "recoverable" here is `ctxplan`'s **Faithful** (every elided span keeps a page-back handle). The result-level windowing maps to `internal/ctxmmu` Transform (oversize→recoverable CAS pointer). Both must be applied prefix-stably at the gateway seam (`fak guard`, residual #555) so the ~90% prompt cache survives.

