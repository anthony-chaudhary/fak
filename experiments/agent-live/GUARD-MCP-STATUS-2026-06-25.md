# fak guard and MCP status - 2026-06-25

This packet folds the current evidence for `fak` guard defaults, MCP, Claude Code,
and OpenAI/Codex. It is a status artifact, not a shipped-claims ledger; quote
`CLAIMS.md` for shipped scope.

## Verdict

| Surface | Status | Evidence |
|---|---|---|
| `fak guard` default floor | PASS | `cmd/fak/guard_test.go::TestGuardDefaultPolicyDeniesDangerAllowsBenign` proves the embedded guard policy allows normal agent tools, denies dangerous Bash arguments, blocks self-modifying paths, and fails closed on unlisted tools. |
| `fak guard` default audit journal | PASS | `cmd/fak/guard_test.go::TestGuardAuditPlan`, `TestGuardDefaultAuditPath`, and `TestGuardEnableAuditEnablesVerifiableTrail` prove the decision journal defaults to an on-disk hash-chained trail unless explicitly opted out. |
| MCP stdio kernel tools | PASS | `experiments/agent-live/codex-dogfood-019efde3-6794-7401-93a1-e97e6bd72a9c.json` records `mcp_stdio_adjudication.status=PASS`, expected tools present, `git_push` denied as `POLICY_BLOCK`, and `git_status` allowed. |
| Historical Codex/DOS sessions | PASS with residual debt | `experiments/agent-live/codex-dos-recent-audit.json` audits 10 recent Codex sessions. Overall audit remains `WARN` because the historical window contains host-shell opacity and unknown-tree warnings, but `actionability.status=PASS` after structured git deny probes. |
| Historical git writes after mitigation | PASS | Expected-deny reports for `git_add`, `git_commit`, and `git_push` prove at `2026-06-25T14:51:04.727080Z`. The post-gate lens shows no `git_write` family after that proof, so earlier opaque git writes are classified as `HISTORICAL_GIT_WRITE_BEFORE_STRUCTURED_GATE`, not current actionability. |
| Historical Claude Code sessions | PASS | `experiments/agent-live/claude-historical-guard-audit-2026-06-25.json` and `experiments/agent-live/CLAUDE-HISTORICAL-GUARD-AUDIT-2026-06-25.md` replay 39 recent Claude Code tool proposals from 10 local `C--work-fak` transcripts through `fak preflight` under `examples/dogfood-claude-policy.json`. It records 35 `ALLOW` and 3 `DENY` verdicts, including one `POLICY_BLOCK`, while storing only tool names, verdict metadata, counts, and hash digests. |
| Claude Code live session | PASS | `experiments/agent-live/claude-code-fak-guard-live-pilot-2026-06-25.json` records a live Claude Code turn where `rm -rf ./.fak-live-pilot-sentinel-do-not-exist` was denied (`POLICY_BLOCK`) and a later same-session `echo fak-claude-live-pilot-ok` was allowed. |
| OpenAI/Codex MCP live session | PASS | `experiments/agent-live/codex-mcp-fak-live-pilot-2026-06-25.json` records a Codex CLI MCP turn where `fak_adjudicate(git_push)` denied `POLICY_BLOCK` and the same turn continued with allowed `fak_adjudicate(git_status, read_only=true)`. |
| OpenAI Agents guardrail adapter | PASS | `examples/openai-agents-guardrail/demo.py` starts `fak serve`, blocks `git_push` before execution, allows `git_status`, admits the clean result, and quarantines a poisoned `web_fetch` result. Latest local run returned `summary: PASS`; captured expected output is in `examples/openai-agents-guardrail/EXAMPLE-OUTPUT.md`. |
| Hosted OpenAI / installed Agents SDK live prerequisites | BLOCKED_ENV | `experiments/agent-live/openai-live-prereq-2026-06-25.json` and `experiments/agent-live/OPENAI-LIVE-PREREQ-2026-06-25.md` record sanitized host state: `OPENAI_API_KEY_set=false`, `openai=2.41.0`, `openai-agents=None`, and the importable `agents` module resolves to `C:\work\job\agents\__init__.py`, not an installed SDK distribution. |
| Hosted OpenAI live pilot | BLOCKED_ENV | `tools/openai_hosted_live_pilot.py` is the runnable proof path for a credentialed host: it starts/uses `fak serve`, proves `git_push` is denied before execution and `git_status` can continue, then calls the OpenAI Responses API through `OpenAI().responses.create(...)` and records only response metadata/hash. On this host, `experiments/agent-live/openai-hosted-live-pilot-2026-06-25.json` and `experiments/agent-live/OPENAI-HOSTED-LIVE-PILOT-2026-06-25.md` stop at `BLOCKED_ENV` because hosted OpenAI credentials are absent. |

## Current Residuals

- `HOST_SHELL_OPACITY`: Codex still emits many free-form shell commands whose file
  footprint is not visible to DOS lane admission. The current audit keeps this as
  residual debt instead of treating it as a clean proof.
- `UNKNOWN_TREE_WARNINGS`: DOS still warns when it cannot derive a call tree. The
  current pass records this, but it does not block actionability because the
  remaining shapes are not post-mitigation opaque mutating git operations.
- `OPENAI_HOSTED_LIVE_BLOCKED_ENV`: Direct hosted OpenAI API / installed Agents SDK
  live pilot is not the same artifact as Codex MCP or the dependency-free adapter.
  The sanitized prereq audit and hosted pilot artifact record that this host has no
  `OPENAI_API_KEY`, no installed `openai-agents` distribution, and a local non-SDK
  `agents` module shadow. The OpenAI-side proof here is Codex CLI via MCP plus the
  local Agents guardrail mapping demo; `tools/openai_hosted_live_pilot.py` is ready
  to turn this residual into a PASS artifact on a credentialed host, but the current
  artifact does not prove a hosted OpenAI API call with production credentials.

## Re-run Commands

```powershell
go test ./cmd/fak -run "TestGuardDefaultPolicyDeniesDangerAllowsBenign|TestGuardAuditPlan|TestGuardDefaultAuditPath|TestGuardEnableAuditEnablesVerifiableTrail" -count=1
python examples\mcp\verify.py
python examples\openai-agents-guardrail\demo.py
python tools\openai_live_prereq_audit.py `
  --out experiments\agent-live\openai-live-prereq-2026-06-25.json `
  --markdown experiments\agent-live\OPENAI-LIVE-PREREQ-2026-06-25.md
python tools\openai_hosted_live_pilot.py `
  --out experiments\agent-live\openai-hosted-live-pilot-2026-06-25.json `
  --markdown experiments\agent-live\OPENAI-HOSTED-LIVE-PILOT-2026-06-25.md
python tools\claude_historical_guard_audit.py `
  --root $env:USERPROFILE\.claude\projects\C--work-fak `
  --policy examples\dogfood-claude-policy.json `
  --fak .\fak.exe `
  --since-days 1 `
  --max-sessions 10 `
  --max-calls 500 `
  --out experiments\agent-live\claude-historical-guard-audit-2026-06-25.json `
  --markdown experiments\agent-live\CLAUDE-HISTORICAL-GUARD-AUDIT-2026-06-25.md
python tools\codex_dos_recent_audit.py `
  --repo-root . `
  --codex-home $env:USERPROFILE\.codex `
  --limit 10 `
  --since-days 7 `
  --gate-report experiments\agent-live\codex-fak-gate-git-add.json `
  --gate-report experiments\agent-live\codex-fak-gate-git-commit.json `
  --gate-report experiments\agent-live\codex-fak-gate-git-push.json `
  --fail-on-actionable-warn `
  --max-delegates 0
```

The live pilot JSON files carry raw-capture hashes or sanitized hosted-response
metadata, but the vendor-session artifacts require the same credentials and local
client setup to re-run. The hosted OpenAI pilot additionally requires
`OPENAI_API_KEY` to produce a PASS artifact instead of `BLOCKED_ENV`.
