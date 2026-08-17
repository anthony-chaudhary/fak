---
name: trajectory-audit
description: Sweep recent Claude Code session transcripts (.jsonl) for token-weighted cost/efficiency problems visible only across runs — machine-wide input:output ratio, prompt-cache / KV reuse, the cache-CREATE burst / suffix-cache-thrash lens (#3069) that a flattering read-share hides, per-session distributions (tool calls, I:O, cache-hit, read-only fraction), the global tool mix, and the heaviest sessions by output tokens — plus the behavioral stuck/churn lens (#2365): per-tool error rates, shell timeout kills, foreground sleep-polls, Edit/Write read-discipline churn, repeated identical failure signatures, per-file mutation churn, and transcript-native hook outcomes/latency (including non-blocking errors and cancellations). Wraps the project's auditor `tools/session_audit.py` (EXACT token accounting from the transcript usage records). Use when the operator says "audit recent claude trajectories/chats/sessions", "where is the token/cost going", "what are the heaviest sessions", "which sessions are stuck/looping/churning", or wants cross-session efficiency or behavior numbers. Read-only — emits a dated report, never edits code.
metadata:
  generated-by: fak project-assets sync
  canonical: ../../../.claude/skills/trajectory-audit/SKILL.md
---

# Canonical project skill adapter

Load and follow [`../../../.claude/skills/trajectory-audit/SKILL.md`](../../../.claude/skills/trajectory-audit/SKILL.md). This generated discovery adapter contains no maintained workflow body.

## Portability contract

- The linked canonical `SKILL.md` is the single semantic workflow body for Claude, Codex, and fak-native loaders.
- This adapter changes discovery only; it must not fork, summarize, or translate the workflow.
- Harness-native invocation, permissions, hooks, model routing, and worker launch remain typed adapters outside the semantic body.
