---
name: trajectory-audit
description: Sweep recent Claude Code session transcripts (.jsonl) for token-weighted cost/efficiency problems visible only across runs — machine-wide input:output ratio, prompt-cache / KV reuse, per-session distributions (tool calls, I:O, cache-hit, read-only fraction), the global tool mix, and the heaviest sessions by output tokens. Wraps the project's auditor `tools/session_audit.py` (EXACT token accounting from the transcript usage records). Use when the operator says "audit recent claude trajectories/chats/sessions", "where is the token/cost going", "what are the heaviest sessions", or wants cross-session efficiency numbers. Read-only — emits a dated report, never edits code.
disable-model-invocation: false
user-invocable: true
allowed-tools: Read, Grep, Glob, Bash
argument-hint: "[--since-days N] [--md OUT | --json OUT] [--all]   (drill: deep <session.jsonl>)"
output_root: none
---

# /trajectory-audit — cross-session token & cache-efficiency sweep

> Wraps `tools/session_audit.py` (exact token accounting from the transcript
> `message.usage` records). **Honest scope:** this is the *token-weighted* lens —
> exact I:O ratio, cache reuse, cost, tool mix, heaviest sessions. It does **not**
> flag read-loops / shell-poll / glob-storms, and it does **not** join the
> transcript to any external run-id spine. Treat the numbers as a cost/efficiency
> census, not a behavior audit.

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
  per-turn token spend) — use only when a heaviest-session row needs a root cause.

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
   count, and estimated cost (assumed-price, flagged).
2. **Per-namespace rollup** — which project the spend lands in.
3. **Per-session distributions** — median/p90 of tool-calls, output tokens, I:O
   ratio, cache-hit fraction, read-only fraction. The tails are where the waste is.
4. **Global tool mix** — a tool dominating the call count is the first thing to question.
5. **Top 15 sessions by output tokens** — the fastest path to the expensive runs.

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

### Step 3 — Surface the 0–2 that matter

End with a short operator summary: the headline machine-wide numbers (I:O,
cache-read share, cost), the single heaviest session and its likely cause, and any
distribution tail that crosses a sane bar. Don't dump the whole table into chat —
link the report file.

## Output

- Dated report: stdout by default, or the `--md` / `--json` path you pass.
- A 3–5 line operator summary in chat.
- Writes **no code** and edits **no plan/memory file** — read-only by construction.

## Notes

- Token figures are exact; the cost line uses an ASSUMED price table — edit
  `PRICING` in `tools/session_audit.py` to match the current card.
