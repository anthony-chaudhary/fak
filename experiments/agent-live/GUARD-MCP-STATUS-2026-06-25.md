# fak guard and MCP status - 2026-06-25

This packet folds the current evidence for `fak` guard defaults, MCP, Claude Code,
and OpenAI/Codex. It is a status artifact, not a shipped-claims ledger; quote
`CLAIMS.md` for shipped scope.

Scope note: the Codex evidence below is the MCP/DOS hook path, not proof that the
current Codex process was launched under `fak guard -- codex`. On this host the
working Codex controls are the configured `fak` MCP server plus native DOS hooks;
the guard-wrapper and hosted OpenAI paths are separate pilots.

Machine-readable rollup: `experiments/agent-live/guard-mcp-status-audit-2026-06-25.json`
contains the same check verdicts and default-on blocker queue, and can be rendered
with `go run ./cmd/fak console guard --guard-json experiments\agent-live\guard-mcp-status-audit-2026-06-25.json`.
The queue evidence also carries bounded settlement-plan rows with marker-relative
paths, counts, origin labels, settlement actions, and count-shape transcript tags.

## Verdict

| Surface | Status | Evidence |
|---|---|---|
| `fak guard` default floor | PASS | `cmd/fak/guard_test.go::TestGuardDefaultPolicyDeniesDangerAllowsBenign` proves the embedded guard policy allows normal agent tools, denies dangerous Bash arguments, blocks self-modifying paths, and fails closed on unlisted tools. |
| `fak guard` default audit journal | PASS | `cmd/fak/guard_test.go::TestGuardAuditPlan`, `TestGuardDefaultAuditPath`, and `TestGuardEnableAuditEnablesVerifiableTrail` prove the decision journal defaults to an on-disk hash-chained trail unless explicitly opted out. |
| MCP stdio kernel tools | PASS | `experiments/agent-live/codex-dogfood-019efde3-6794-7401-93a1-e97e6bd72a9c.json` records `mcp_stdio_adjudication.status=PASS`, expected tools present, `git_push` denied as `POLICY_BLOCK`, and `git_status` allowed. |
| Historical Codex/DOS sessions | WARN - StopFailure review required | `experiments/agent-live/codex-dos-recent-audit.json` audits 20 recent Codex sessions. Overall audit remains `WARN`: Codex stream stop blocks/failures are 0, but the workspace StopFailure API-wall breaker rollup found 76 one-day failures across 60 nonzero markers; 3 markers are recently active with 3 consecutive failures, 43 markers are stale-active with 48 consecutive failures, and 14 nonzero markers have healed to `consecutive=0`, so `actionability.status=WARN`. |
| Historical git writes after mitigation | PASS | Expected-deny reports for `git_add`, `git_commit`, and `git_push` prove at `2026-06-25T22:38:12.037688Z`. The post-gate lens shows no `git_write` family after that proof, so earlier opaque git writes are classified as `HISTORICAL_GIT_WRITE_BEFORE_STRUCTURED_GATE`, not current actionability. |
| Historical Claude Code sessions | PASS + friction surfaced | `experiments/agent-live/claude-historical-guard-audit-2026-06-25.json` and `experiments/agent-live/CLAUDE-HISTORICAL-GUARD-AUDIT-2026-06-25.md` now scan 20 recent `C--work-fak` transcripts across 10 local `.claude*` account roots. They found 230 tool proposals and replayed 227 unique tool shapes through `fak preflight` under `examples/dogfood-claude-policy.json`, recording 181 `ALLOW` and 46 `DENY` verdicts, including 7 `POLICY_BLOCK`. The same artifact surfaces count-only friction tags: `HOOK_OR_API_WALL_FEEDBACK` in 20 sessions, `HOST_PERMISSION_INTERRUPT` in 20, `TOOL_ERROR_RECOVERY` in 19, `DENY_OR_BLOCKED_FEEDBACK` in 17, `SHELL_HEAVY_SESSION` in 11, and `LARGE_RESULT` in 9, while storing no prompts, tool arguments, tool results, full user paths, or raw transcript text. |
| Claude Code live session | PASS | `experiments/agent-live/claude-code-fak-guard-live-pilot-2026-06-25.json` records a live Claude Code turn where `rm -rf ./.fak-live-pilot-sentinel-do-not-exist` was denied (`POLICY_BLOCK`) and a later same-session `echo fak-claude-live-pilot-ok` was allowed. |
| OpenAI/Codex MCP live session | PASS | `experiments/agent-live/codex-mcp-fak-live-pilot-2026-06-25.json` records a Codex CLI MCP turn where `fak_adjudicate(git_push)` denied `POLICY_BLOCK` and the same turn continued with allowed `fak_adjudicate(git_status, read_only=true)`. |
| OpenAI Agents guardrail adapter | PASS | `examples/openai-agents-guardrail/demo.py` starts `fak serve`, blocks `git_push` before execution, allows `git_status`, admits the clean result, and quarantines a poisoned `web_fetch` result. Latest local run returned `summary: PASS`; captured expected output is in `examples/openai-agents-guardrail/EXAMPLE-OUTPUT.md`. |
| Hosted OpenAI / installed Agents SDK live prerequisites | PARTIAL | `experiments/agent-live/openai-live-prereq-2026-06-25.json` and `experiments/agent-live/OPENAI-LIVE-PREREQ-2026-06-25.md` record sanitized host state: `codex_login_ready=true`, `platform_api_ready=false`, `OPENAI_API_KEY_set=false`, `openai=2.41.0`, `openai-agents=None`, and the importable `agents` module resolves to `C:\work\job\agents\__init__.py`, not an installed SDK distribution. |
| Hosted OpenAI live pilot | PASS | `tools/openai_hosted_live_pilot.py --auth-mode codex-login` starts/uses `fak serve`, proves `git_push` is denied before execution and `git_status` can continue, then runs `codex exec --ephemeral` with the existing ChatGPT/OAuth login. `experiments/agent-live/openai-hosted-live-pilot-2026-06-25.json` and `experiments/agent-live/OPENAI-HOSTED-LIVE-PILOT-2026-06-25.md` record `auth_source=codex_login`, `codex_exec_exit_code=0`, `contains_expected_marker=true`, and only hosted-output hashes/event counts. |

## Current Residuals

- `HOST_SHELL_OPACITY` debt: Codex still emits many free-form shell commands whose file
  footprint is not visible to DOS lane admission. In the current audit this is visible
  as shell-shape debt, but not the active `actionability.residual`; the active WARN is
  dominated by StopFailure API-wall state.
- `HISTORICAL_GIT_WRITE_BEFORE_STRUCTURED_GATE`: the one-day window still contains
  opaque `git_write` shell families before the fresh structured deny probes. The
  post-gate lens is empty, so this is historical debt, not current actionability.
- `STOPFAILURE_API_WALL_BREAKER`: `.dos/stop-failures` contains 76 one-day API-wall
  failures across 60 nonzero session markers. Of those, 3 markers are recently active
  within the 6-hour live threshold with 3 consecutive failures, 43 markers are
  stale-active with 48 consecutive failures, and 14 nonzero markers are healed to
  `consecutive=0`. The recent origin split is 2 Claude-transcript-only markers
  and 1 marker with both DOS stream and Claude transcript linkage; the stale
  origin split is 31 Claude-transcript-only markers, 8 DOS-stream+Claude markers,
  and 4 marker-only records. The non-destructive settlement classifier marks 3
  active markers as `RECENT_REVIEW`, 39 stale markers as `STALE_RESET_CANDIDATE`,
  and 4 marker-only stale records as `STALE_MARKER_ONLY_ARCHIVE_CANDIDATE`. The
  blocker queue includes bounded recent-review and stale-settlement rows with
  `.dos/stop-failures/<session>.json` marker paths. The top recent, stale, and
  settlement-plan sessions carry only sanitized count-shape transcript evidence
  (`HOOK_OR_API_WALL_FEEDBACK`, `HOST_PERMISSION_INTERRUPT`, `DENY_OR_BLOCKED_FEEDBACK`,
  `TOOL_ERROR_RECOVERY`, `SHELL_HEAVY_SESSION`, `LARGE_TOOL_RESULT`) and no prompts,
  command bodies, tool output, or model text.
- `CLAUDE_FRICTION_SHAPES`: the all-account Claude replay is policy-clean enough to
  pass, but the transcripts show operational friction: 508 hook/API-wall marker lines,
  421 permission marker lines, 429 deny/blocked marker lines, 1527 error-recovery
  marker lines, and a maximum sanitized result length of 64,944 characters. These are
  count-shape signals only; session ids are hashed and root names are account labels.
- `OPENAI_AGENTS_SDK_NOT_INSTALLED`: The hosted OpenAI proof now passes through the
  existing Codex ChatGPT/OAuth login (`auth_source=codex_login`), not a Platform API
  key. The remaining OpenAI residual is narrower: this host still has no installed
  `openai-agents` distribution, and the local `agents` import resolves to a non-SDK
  shadow package. The dependency-free guardrail adapter remains the local mapping
  proof for Agents-style behavior.

## Default-On Blocker Queue

Ranked from current actionability blocker to historical/external debt:

1. `WORKSPACE_RECENT_STOPFAILURE_API_WALL` (`ACTIVE`, workspace DOS): 3 recent
   markers still carry 3 consecutive StopFailure counts within the 6-hour live
   threshold. Current origin split is 2 Claude-transcript-only markers and 1
   marker with both DOS stream and Claude transcript linkage. Settlement class:
   3 `RECENT_REVIEW`. Top recent sessions are mapped only by session id, counts,
   marker-relative path, origin, settlement class, project label, and count-shape
   tags. Next action: inspect those recent sessions before clearing or rotating their nonzero
   `consecutive` state.
2. `CODEX_HOST_SHELL_OPACITY` (`ACTIVE_DEBT`, Codex hooks): post-repair evidence still
   has 2,290 `shell_no_write_target_detected` calls. Next action: prefer path-visible
   host tools or structured tool payloads so DOS can assign file-tree footprints.
3. `CLAUDE_ALL_ACCOUNT_OPERATIONAL_FRICTION` (`ACTIVE_DEBT`, Claude Code): across 20
   recent transcripts, the all-account replay shows hook/API-wall and permission
   friction in all 20 summarized sessions, plus tool-error recovery in 19,
   deny/blocked feedback in 17, shell-heavy sessions in 11, and large results in 9.
   Next action: triage top friction sessions by tag and reduce hook/API-wall and
   permission interruptions first.
4. `WORKSPACE_STALE_STOPFAILURE_MARKERS` (`STALE_DEBT`, workspace DOS): 43 stale-active
   markers still carry 48 consecutive StopFailure counts outside the 6-hour live
   threshold. Their origin split is 31 Claude-transcript-only markers, 8
   DOS-stream+Claude markers, and 4 marker-only records. Settlement classes:
   39 `STALE_RESET_CANDIDATE` and 4 `STALE_MARKER_ONLY_ARCHIVE_CANDIDATE`. Next
   action: use the bounded settlement plan's marker-relative paths to reset or
   archive those stale markers so old breaker state no longer obscures live
   fak-by-default actionability.
5. `HISTORICAL_OPAQUE_GIT_WRITE_BEFORE_GATE` (`HISTORICAL`, Codex hooks): opaque
   `git_write` appears in the one-day post-repair window, but not after the structured
   git gate proof at `2026-06-25T22:38:12.037688Z`. Next action: keep structured git
   gates in place; this only becomes current actionability if `git_write` reappears
   after the gate timestamp.
6. `OPENAI_AGENTS_SDK_NOT_INSTALLED` (`EXTERNAL_PREREQ`, hosted OpenAI): the hosted
   Codex-login pilot passes, but the installed Agents SDK path remains unavailable on
   this host. Next action: install the SDK only when that hosted Agents surface is the
   target.

## Re-run Commands

```powershell
go test ./cmd/fak -run "TestGuardDefaultPolicyDeniesDangerAllowsBenign|TestGuardAuditPlan|TestGuardDefaultAuditPath|TestGuardEnableAuditEnablesVerifiableTrail" -count=1
python examples\mcp\verify.py
python examples\openai-agents-guardrail\demo.py
python tools\openai_live_prereq_audit.py `
  --out experiments\agent-live\openai-live-prereq-2026-06-25.json `
  --markdown experiments\agent-live\OPENAI-LIVE-PREREQ-2026-06-25.md
python tools\openai_hosted_live_pilot.py `
  --auth-mode codex-login `
  --out experiments\agent-live\openai-hosted-live-pilot-2026-06-25.json `
  --markdown experiments\agent-live\OPENAI-HOSTED-LIVE-PILOT-2026-06-25.md
python tools\claude_historical_guard_audit.py `
  --all-accounts `
  --namespace C--work-fak `
  --policy examples\dogfood-claude-policy.json `
  --fak .\fak.exe `
  --since-days 1 `
  --max-sessions 20 `
  --max-calls 500 `
  --out experiments\agent-live\claude-historical-guard-audit-2026-06-25.json `
  --markdown experiments\agent-live\CLAUDE-HISTORICAL-GUARD-AUDIT-2026-06-25.md
python tools\codex_dos_recent_audit.py `
  --repo-root . `
  --codex-home $env:USERPROFILE\.codex `
  --limit 20 `
  --since-days 1 `
  --gate-report experiments\agent-live\codex-fak-gate-git-add.json `
  --gate-report experiments\agent-live\codex-fak-gate-git-commit.json `
  --gate-report experiments\agent-live\codex-fak-gate-git-push.json `
  --max-delegates 0
python tools\guard_mcp_status_audit.py `
  --root . `
  --out experiments\agent-live\guard-mcp-status-audit-2026-06-25.json
go run ./cmd/fak console guard `
  --guard-json experiments\agent-live\guard-mcp-status-audit-2026-06-25.json `
  --width 190
```

The live pilot JSON files carry raw-capture hashes or sanitized hosted-response
metadata, but the vendor-session artifacts require the same credentials and local
client setup to re-run. The hosted OpenAI pilot uses the existing Codex
ChatGPT/OAuth login in `~/.codex/auth.json` for `--auth-mode codex-login`; the
direct Platform API-key route remains available with `--auth-mode api-key`.
