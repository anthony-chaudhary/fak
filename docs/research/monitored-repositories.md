# Monitored harness repositories

**Checked:** 2026-08-14  
**Machine source:** [`monitored-repositories.json`](monitored-repositories.json)  
**Refresh view:** `fak study-monitor --due-days 14` (add `--json` for automation)

This is the durable discovery queue for external harness repositories. It complements dated deep-study notes: the registry records candidates before a full study, pins what was actually checked, and keeps `last_checked` visible so a scout can distinguish a fresh lead from a stale one. Exact `owner/name` is the deduplication key.

## Current ranking

| Rank | Repository | Why it is hot and helpful now | Discovery evidence |
|---:|---|---|---|
| 1 | [`ruvnet/ruflo`](https://github.com/ruvnet/ruflo) | The most direct meta-harness overlap: coding-agent swarms, memory, skills, MCP, hooks, and Codex/Claude integrations. | 67,866 stars; 47 commits from 2026-08-07 through check time; pushed 2026-08-14. |
| 2 | [`Untrivial-ai/agent-orchestrator`](https://github.com/Untrivial-ai/agent-orchestrator) | The closest fleet-control analogue: planning, worktree-isolated agents, CI repair, review, and merge-conflict handling. | 9,518 stars; at least 100 commits in the seven-day window; pushed 2026-08-14. |
| 3 | [`langchain-ai/open-swe`](https://github.com/langchain-ai/open-swe) | An asynchronous coding-agent harness with sandbox, task lifecycle, desktop, evaluation, and session seams. | 10,553 stars; 80 commits in the seven-day window; pushed 2026-08-14. |
| 4 | [`EveryInc/compound-engineering-plugin`](https://github.com/EveryInc/compound-engineering-plugin) | A cross-harness Codex/Claude/Cursor plugin whose plan-review-work-compound loop is directly comparable to fak's skill and evidence workflows. | 24,259 stars; 20 commits in the seven-day window; pushed 2026-08-14. |
| 5 | [`obra/superpowers`](https://github.com/obra/superpowers) | The strongest adoption signal for a portable skills, hooks, planning, testing, and subagent-development methodology not already tracked by exact repository name. | 272,168 stars; release activity at pinned revision; pushed 2026-08-13. |

The GitHub API's 100-item page cap means “at least 100” is deliberate. Stars are only a discovery signal. A deep study still has to pin source, inspect implementation seams, check the license, query fak first, and independently witness any claimed gap.

## Maintenance contract

1. Run `fak study-monitor --due-days 14` before selecting a scout lead.
2. Search this registry and tracked notes by exact `owner/name`; do not rediscover an existing source under a nickname.
3. On every meaningful check, update `last_checked`, `checked_revision`, `stars_at_check`, and `last_push_at_check` together.
4. Use `candidate` for an unstudied lead, `studied` only with `study_note`, `watch` when no present borrow survives, and `dismissed` only with the reason retained in `why`.
5. A completed `study-repo` pass updates this row in the same commit as its dated study note. A `scout-loop` pass selects due candidates before opening a fresh outward search.

This list intentionally does not claim that these repositories contain a borrowable gap. It proves only that they were fresh, relevant, previously untracked by exact source identity, and worth the next disciplined study pass on 2026-08-14.
