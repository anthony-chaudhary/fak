---
name: trajectory-audit
description: Sweep recent Claude Code session transcripts (.jsonl) for token-weighted cost/efficiency problems visible only across runs — machine-wide input:output ratio, prompt-cache / KV reuse, per-session distributions (tool calls, I:O, cache-hit, read-only fraction), the global tool mix, and the heaviest sessions by output tokens — plus the behavioral stuck/churn lens (#2365): per-tool error rates, shell timeout kills, foreground sleep-polls, Edit/Write read-discipline churn, repeated identical failure signatures, and per-file mutation churn. Wraps the project's auditor `tools/session_audit.py` (EXACT token accounting from the transcript usage records). Use when the operator says "audit recent claude trajectories/chats/sessions", "where is the token/cost going", "what are the heaviest sessions", "which sessions are stuck/looping/churning", or wants cross-session efficiency or behavior numbers. Read-only — emits a dated report, never edits code.
disable-model-invocation: false
user-invocable: true
allowed-tools: Read, Grep, Glob, Bash
argument-hint: "[--since-days N] [--md OUT | --json OUT] [--all]   (drill: deep <session.jsonl>)"
output_root: none
metadata:
  opencode: claude-only   # #422: read-only allowed-tools boundary is load-bearing and Claude-only — exclude from the opencode skills.paths scan
---

# /trajectory-audit — cross-session token & cache-efficiency sweep

> Wraps `tools/session_audit.py` (exact token accounting from the transcript
> `message.usage` records). Two lenses now run on every sweep:
> the *token-weighted* lens — exact I:O ratio, cache reuse, cost (split by billing
> bucket), tool mix, heaviest sessions — and the *behavioral* stuck/churn lens
> (#2365): per-tool call/error counts + error rate, shell timeout kills (exit 143 /
> "timed out"), foreground sleep-polls (`sleep`/`Start-Sleep` command prefix),
> Edit/Write read-discipline churn, repeated identical failure signatures (≥3×),
> and per-file mutation churn (≥5 edits of one file — the rewrite-loop smell).
> **Honest scope:** it still does **not** flag loops of *successful* identical
> calls (read-loops / glob-storms — only failures and mutations are looped-checked),
> and it does **not** join the transcript to any external run-id spine.
>
> **Billing buckets — the one cost rule.** Each model's tokens land on its vendor's
> invoice: `claude-*` is the Anthropic bucket; a `gemini-*` / `gpt-*` / local model
> is a **different bill entirely**. The auditor (a) breaks cost out per provider and
> **never sums across buckets**, (b) refuses to price a non-Claude model at Claude
> rates — an unknown model is reported with its tokens but **no fabricated cost**
> (add its card to `PRICING` + `PROVIDER_BUCKETS` to price it), and (c) treats
> `<synthetic>` (harness-injected) as **non-billed, $0**. A single blended "cost"
> across providers is meaningless — read the per-bucket / per-model tables, not just
> the headline.

Reads the most recent Claude Code session transcripts and rolls up **what only
shows up across many runs**: where the tokens and cost actually went, how much of
the ingested context was prompt-cache reuse vs fresh billed input, and which
sessions were the expensive ones. The transcripts carry EXACT token accounting
(`message.usage`), so every number here is ground truth — only the cost line uses
an assumed price table (flagged in the output; edit `PRICING` in the tool).

The auditor discovers transcripts under the Claude Code projects root
(`~/.claude/projects` by default; the tool's own defaults govern which roots and
namespace prefix it scans). Pass `--root` only if your transcripts live elsewhere.

## Inputs

```
/trajectory-audit [--since-days N] [--md OUT] [--json OUT] [--all] [--include-subagents]
/trajectory-audit deep <path-to-session.jsonl>     # follow ONE trajectory top-to-bottom
```

- `--since-days N` — only sessions touched in the last N days (default: all discovered).
- `--md OUT` / `--json OUT` — write the report to a file instead of stdout. Use
  `--md` for the human report; `--json` when the window is large and you want to
  skim programmatically.
- `--all` — drop the namespace filter (audit every project, not just the current repo family).
- `deep <session.jsonl>` — single-trajectory drill-down (the user asks in order,
  per-turn token spend, plus the behavior line: errors / timeout kills /
  sleep-polls / churn / worst repeats) — use when a heaviest-session or
  behavioral-offender row needs a root cause.

## Procedure

### Step 0 — Discover, then run the rollup (one read per session, no polling)

This skill **never** tails or re-reads logs in a loop — that is the very
anti-pattern it audits for. The tool does a single sequential read of each file.

```bash
# what's in the window (cheap):
python tools/session_audit.py discover --since-days 7

# the rollup, written to a dated report:
python tools/session_audit.py audit --since-days 7 --md trajectory-audit-$(date +%Y%m%d).md
```

For a large window prefer `--json` and skim, rather than rendering a giant table.
A bare report name lands at the repo root; pass a path under a gitignored dir
(e.g. `.dos/audits/`) if you don't want it tracked.

### Step 1 — Read the rollup, not the raw sessions

The report gives you, in order:

1. **Machine-wide totals (EXACT)** — output tokens (the real work), fresh billed
   input, cache-read (prompt-cache / KV reuse), the **I:O ratio**, the
   **cache-read share** of all ingested context, web search/fetch, multi-iteration
   count, **Anthropic-billed cost** (assumed-price, flagged), plus a line for any
   **other billing bucket** present and the non-billed `<synthetic>` turn count.
2. **Cost by billing bucket (provider)** — the answer to "is this Claude money or
   Gemini money?". Never sum across rows; an unpriced bucket shows "— (no card)".
3. **Per-model breakdown** — the tier split (opus vs sonnet vs haiku), so a blended
   number can be read as opus-heavy vs haiku-heavy. This is where you confirm *which*
   model drove the cost before you quote a figure.
4. **Per-namespace rollup** — which project the spend lands in, with its top model.
5. **Per-session distributions** — median/p90 of tool-calls, output tokens, I:O
   ratio, cache-hit fraction, read-only fraction. The tails are where the waste is.
6. **Global tool mix** — a tool dominating the call count is the first thing to question.
7. **Behavioral lens — stuck/churn detectors** — per-tool error rates, timeout
   kills, sleep-polls, Edit/Write churn, then the two worst-offender tables:
   sessions with a repeated identical failure (≥3× the same signature = a stuck
   loop) and sessions with file churn (≥5 mutations of one file = a rewrite-loop
   smell). These rows are the behavior audit the token lens can't see — each one
   names a session worth a `deep` drill.
8. **Top 15 sessions by output tokens** — the fastest path to the expensive runs.

The `trend` subcommand carries the same detectors per time bucket (`err%` /
`t/o` / `slp` / `chrn` columns, plus a `behavior` object in `--json` rows), so a
regression in fleet behavior is visible week-over-week, not just in one sweep.

Do **not** open the individual `.jsonl` files unless a heaviest-session row needs
a root cause the rollup can't name. If you do, run `deep <session>` once — don't
re-read it in a loop.

### Step 2 — Separate signal from expected

- A high cache-read share is **good** (it's KV reuse the harness already captures),
  not waste — a healthy long-running machine sits in the ~90%+ band. A *low*
  cache-read share on a long session is the smell worth chasing.
- A `--include-subagents` run counts sidechain/workflow transcripts that are
  normally uncounted; note them but don't double-count against the main thread.
- A high I:O ratio is expected for read/research-heavy work; flag it only when it
  pairs with a heavy session whose output was small (lots of context in, little
  produced).
- **Quote cost per bucket, never blended.** Before you state a dollar figure, read
  the per-bucket and per-model tables: say "Anthropic-billed, ~99% opus-4-8", not a
  lone total. If a non-Anthropic bucket appears with "— (no card)", say its cost is
  **unknown** (a separate invoice), not zero — zero would imply it was free.
  `<synthetic>` is genuinely $0 (never hit a vendor) and must not be counted as work.

### Step 3 — Surface the 0–2 that matter

End with a short operator summary: the headline machine-wide numbers (I:O,
cache-read share, **cost named by bucket + dominant model**), the single heaviest
session and its likely cause, the single worst behavioral offender (a repeated
identical failure or a file-churn loop) if one crossed the threshold, and any
distribution tail that crosses a sane bar.
Don't dump the whole table into chat — link the report file.

## Output

- Dated report: stdout by default, or the `--md` / `--json` path you pass.
- A 3–5 line operator summary in chat.
- Writes **no code** and edits **no plan/memory file** — read-only by construction.

## Notes

- Token figures are exact; the cost line uses an ASSUMED price table — edit
  `PRICING` in `tools/session_audit.py` to match the current card. `PRICING` is
  **Anthropic-only** and matched by model substring (`opus`/`sonnet`/`haiku`/`fable`).
- To price another vendor, add its rate card to `PRICING` **and** its model
  substrings to `PROVIDER_BUCKETS` (the bucket map). Without both, that vendor's
  sessions are reported with exact tokens but shown as "— (no card)" — never folded
  into the Anthropic total and never mispriced at Opus rates.
- `<synthetic>` / `?` / blank models are in `NONBILLED_MODELS` → always $0.
