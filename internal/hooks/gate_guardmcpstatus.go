package hooks

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

// gate_guardmcpstatus.go — the GUARD_MCP_STATUS gate, a byte-faithful port of
// tools/guard_mcp_status_audit.py.
//
// It audits the guard/MCP proof packet against current evidence artifacts:
// - guard default-floor/default-journal tests are present in cmd/fak/guard_test.go;
// - MCP stdio denies git_push and allows git_status in codex dogfood;
// - structured git gates reject dangerous git operations;
// - the historical Codex/DOS audit states actionability after structured git gates;
// - the post-git-gate lens has no opaque git_write;
// - Claude Code and Codex MCP live pilot artifacts show deny + useful continuation;
// - OpenAI agents adapter proof, live prereqs, and hosted live pilot.
//
// Parity anchor: tools/guard_mcp_status_audit.py collect() and check_* functions.

const (
	guardMCPStatusSchema       = "fak-guard-mcp-status-audit/1"
	guardMCPStatusPacketPath   = "experiments/agent-live/GUARD-MCP-STATUS-2026-06-25.md"
	guardMCPDogfoodPath        = "experiments/agent-live/codex-dogfood-019efde3-6794-7401-93a1-e97e6bd72a9c.json"
	guardMCPDosAuditPath       = "experiments/agent-live/codex-dos-recent-audit.json"
	guardMCPClaudeHistJSONPath = "experiments/agent-live/claude-historical-guard-audit-2026-06-25.json"
	guardMCPClaudeHistMDPath   = "experiments/agent-live/CLAUDE-HISTORICAL-GUARD-AUDIT-2026-06-25.md"
	guardMCPClaudeLivePath     = "experiments/agent-live/claude-code-fak-guard-live-pilot-2026-06-25.json"
	guardMCPCodexMCPLivePath   = "experiments/agent-live/codex-mcp-fak-live-pilot-2026-06-25.json"
	guardMCPOpenAIAgentsOutput = "examples/openai-agents-guardrail/EXAMPLE-OUTPUT.md"
	guardMCPOpenAIAgentsDemo   = "examples/openai-agents-guardrail/demo.py"
	guardMCPOpenAIPrereqJSON   = "experiments/agent-live/openai-live-prereq-2026-06-25.json"
	guardMCPOpenAIPrereqMD     = "experiments/agent-live/OPENAI-LIVE-PREREQ-2026-06-25.md"
	guardMCPOpenAIHostedJSON   = "experiments/agent-live/openai-hosted-live-pilot-2026-06-25.json"
	guardMCPOpenAIHostedMD     = "experiments/agent-live/OPENAI-HOSTED-LIVE-PILOT-2026-06-25.md"
	guardMCPGuardTestPath      = "cmd/fak/guard_test.go"
)

type guardCheckResult struct {
	name   string
	file   string
	passed bool
	detail string
}

func readGuardJSON(root, rel string) (map[string]any, error) {
	p := filepath.Join(root, filepath.FromSlash(rel))
	b, err := os.ReadFile(p)
	if err != nil {
		return nil, err
	}
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		return nil, err
	}
	return m, nil
}

func readGuardText(root, rel string) (string, error) {
	p := filepath.Join(root, filepath.FromSlash(rel))
	b, err := os.ReadFile(p)
	if err != nil {
		return "", err
	}
	return string(b), nil
}

func guardMapVal(m map[string]any, k string) map[string]any {
	if m == nil {
		return nil
	}
	if v, ok := m[k].(map[string]any); ok {
		return v
	}
	return nil
}

func guardStringVal(m map[string]any, k string) string {
	if m == nil {
		return ""
	}
	if s, ok := m[k].(string); ok {
		return s
	}
	return ""
}

func guardBoolVal(m map[string]any, k string) bool {
	if m == nil {
		return false
	}
	if b, ok := m[k].(bool); ok {
		return b
	}
	return false
}

func guardSliceVal(m map[string]any, k string) []any {
	if m == nil {
		return nil
	}
	if s, ok := m[k].([]any); ok {
		return s
	}
	return nil
}

func guardStringSliceVal(m map[string]any, k string) []string {
	if m == nil {
		return nil
	}
	s, ok := m[k].([]any)
	if !ok {
		return nil
	}
	res := make([]string, 0, len(s))
	for _, item := range s {
		if str, ok := item.(string); ok {
			res = append(res, str)
		}
	}
	return res
}

func guardToInt(v any) int {
	if v == nil {
		return 0
	}
	switch n := v.(type) {
	case int:
		return n
	case int64:
		return int(n)
	case float64:
		return int(n)
	case string:
		i, _ := strconv.Atoi(n)
		return i
	default:
		return 0
	}
}

func guardVerdictKind(row map[string]any) string {
	if row == nil {
		return ""
	}
	if fv := guardMapVal(row, "fak_verdict"); fv != nil {
		if k := guardStringVal(fv, "kind"); k != "" {
			return k
		}
	}
	if fa := guardMapVal(row, "fak_audit"); fa != nil {
		if v := guardStringVal(fa, "verdict"); v != "" {
			return v
		}
	}
	if v := guardStringVal(row, "verdict"); v != "" {
		return v
	}
	if vd := guardMapVal(row, "verdict"); vd != nil {
		if k := guardStringVal(vd, "kind"); k != "" {
			return k
		}
	}
	return ""
}

func guardVerdictReason(row map[string]any) string {
	if row == nil {
		return ""
	}
	if fv := guardMapVal(row, "fak_verdict"); fv != nil {
		if r := guardStringVal(fv, "reason"); r != "" {
			return r
		}
	}
	if fa := guardMapVal(row, "fak_audit"); fa != nil {
		if r := guardStringVal(fa, "reason"); r != "" {
			return r
		}
	}
	if r := guardStringVal(row, "reason"); r != "" {
		return r
	}
	if vd := guardMapVal(row, "verdict"); vd != nil {
		if r := guardStringVal(vd, "reason"); r != "" {
			return r
		}
	}
	return ""
}

func checkGuardStatusPacket(root string, checks *[]guardCheckResult) {
	rel := guardMCPStatusPacketPath
	text, err := readGuardText(root, rel)
	if err != nil {
		*checks = append(*checks, guardCheckResult{
			name:   "status packet present",
			file:   rel,
			passed: false,
			detail: err.Error(),
		})
		return
	}
	required := []string{
		guardMCPDosAuditPath,
		guardMCPClaudeHistJSONPath,
		guardMCPClaudeHistMDPath,
		guardMCPClaudeLivePath,
		guardMCPCodexMCPLivePath,
		guardMCPOpenAIAgentsOutput,
		guardMCPOpenAIPrereqJSON,
		guardMCPOpenAIPrereqMD,
		guardMCPOpenAIHostedJSON,
		guardMCPOpenAIHostedMD,
		"actionability.status=",
		"post-gate lens shows no `git_write`",
		"Default-On Blocker Queue",
		"WORKSPACE_RECENT_STOPFAILURE_API_WALL",
		"WORKSPACE_STALE_STOPFAILURE_MARKERS",
		"CLAUDE_ALL_ACCOUNT_OPERATIONAL_FRICTION",
		"codex_login",
	}
	var missing []string
	for _, item := range required {
		if !strings.Contains(text, item) {
			missing = append(missing, item)
		}
	}
	ok := len(missing) == 0
	detail := "status packet names the evidence and residual interpretation"
	if !ok {
		detail = fmt.Sprintf("missing %v", missing)
	}
	*checks = append(*checks, guardCheckResult{
		name:   "status packet present",
		file:   rel,
		passed: ok,
		detail: detail,
	})
}

func checkGuardDefaultTests(root string, checks *[]guardCheckResult) {
	rel := guardMCPGuardTestPath
	text, err := readGuardText(root, rel)
	if err != nil {
		*checks = append(*checks, guardCheckResult{
			name:   "guard default tests present",
			file:   rel,
			passed: false,
			detail: err.Error(),
		})
		return
	}
	required := []string{
		"TestGuardDefaultPolicyDeniesDangerAllowsBenign",
		"TestGuardAuditPlan",
		"TestGuardDefaultAuditPath",
		"TestGuardEnableAuditEnablesVerifiableTrail",
	}
	var missing []string
	for _, name := range required {
		if !strings.Contains(text, name) {
			missing = append(missing, name)
		}
	}
	ok := len(missing) == 0
	detail := "all required guard default tests are present"
	if !ok {
		detail = fmt.Sprintf("missing %v", missing)
	}
	*checks = append(*checks, guardCheckResult{
		name:   "guard default tests present",
		file:   rel,
		passed: ok,
		detail: detail,
	})
}

func checkGuardMCPStdio(root string, checks *[]guardCheckResult) {
	rel := guardMCPDogfoodPath
	data, err := readGuardJSON(root, rel)
	if err != nil {
		*checks = append(*checks, guardCheckResult{
			name:   "mcp stdio adjudication",
			file:   rel,
			passed: false,
			detail: err.Error(),
		})
		return
	}
	chkMap := guardMapVal(data, "checks")
	mcp := guardMapVal(chkMap, "mcp_stdio_adjudication")
	deny := guardMapVal(mcp, "denies_publish")
	allow := guardMapVal(mcp, "allows_status")
	missingTools := guardSliceVal(mcp, "missing_tools")
	status := guardStringVal(mcp, "status")

	ok := status == "PASS" &&
		len(missingTools) == 0 &&
		guardStringVal(deny, "kind") == "DENY" &&
		guardStringVal(deny, "reason") == "POLICY_BLOCK" &&
		guardStringVal(allow, "kind") == "ALLOW"

	*checks = append(*checks, guardCheckResult{
		name:   "mcp stdio adjudication",
		file:   rel,
		passed: ok,
		detail: fmt.Sprintf("status=%s deny=%v allow=%v", status, deny, allow),
	})
}

func checkGuardGitGateReports(root string, checks *[]guardCheckResult) {
	gates := []struct {
		tool   string
		rel    string
		reason string
	}{
		{"git_add", "experiments/agent-live/codex-fak-gate-git-add.json", "DEFAULT_DENY"},
		{"git_commit", "experiments/agent-live/codex-fak-gate-git-commit.json", "DEFAULT_DENY"},
		{"git_push", "experiments/agent-live/codex-fak-gate-git-push.json", "POLICY_BLOCK"},
	}
	for _, g := range gates {
		data, err := readGuardJSON(root, g.rel)
		if err != nil {
			*checks = append(*checks, guardCheckResult{
				name:   "structured git gate " + g.tool,
				file:   g.rel,
				passed: false,
				detail: err.Error(),
			})
			continue
		}
		preflight := guardMapVal(data, "preflight")
		ok := guardStringVal(data, "tool") == g.tool &&
			guardStringVal(data, "status") == "DENIED_EXPECTED" &&
			guardBoolVal(data, "expect_deny") == true &&
			guardBoolVal(data, "executed") == false &&
			guardStringVal(preflight, "verdict") == "DENY" &&
			guardStringVal(preflight, "reason") == g.reason

		*checks = append(*checks, guardCheckResult{
			name:   "structured git gate " + g.tool,
			file:   g.rel,
			passed: ok,
			detail: fmt.Sprintf("status=%s reason=%s", guardStringVal(data, "status"), guardStringVal(preflight, "reason")),
		})
	}
}

func checkGuardHistoricalAudit(root string, checks *[]guardCheckResult) {
	rel := guardMCPDosAuditPath
	data, err := readGuardJSON(root, rel)
	if err != nil {
		*checks = append(*checks, guardCheckResult{
			name:   "historical codex/dos actionability",
			file:   rel,
			passed: false,
			detail: err.Error(),
		})
		return
	}
	action := guardMapVal(data, "actionability")
	gitGate := guardMapVal(data, "git_gate_evidence")
	postGate := guardMapVal(gitGate, "post_gate_command_shapes")
	postGateFamilies := guardMapVal(postGate, "shell_family_counts")
	summary := guardMapVal(data, "summary")
	residual := guardStringSliceVal(action, "residual")
	reasons := guardStringSliceVal(action, "reasons")
	postRepairShapes := guardMapVal(action, "post_repair_shell_shape_counts")

	var activeConsecutive int
	if v, exists := summary["codex_origin_stop_failure_current_live_consecutive_total"]; exists {
		activeConsecutive = guardToInt(v)
	} else if v, exists := summary["workspace_stop_failure_current_live_consecutive_total"]; exists {
		activeConsecutive = guardToInt(v)
	} else {
		activeConsecutive = guardToInt(summary["workspace_stop_failure_active_consecutive_total"])
	}
	stopFailureActive := activeConsecutive > 0

	hasAPIWall := false
	for _, r := range reasons {
		if strings.Contains(r, "StopFailure API-wall") {
			hasAPIWall = true
			break
		}
	}

	actionStatus := guardStringVal(action, "status")
	actionabilityOK := false
	if stopFailureActive {
		actionabilityOK = actionStatus == "WARN" && hasAPIWall
	} else {
		actionabilityOK = actionStatus == "PASS" && !hasAPIWall
	}

	reportStatus := guardStringVal(data, "status")
	reportStatusOK := false
	if stopFailureActive {
		reportStatusOK = reportStatus == "WARN"
	} else {
		reportStatusOK = reportStatus == "PASS" || reportStatus == "WARN"
	}

	hasGitWrite := false
	if v, exists := postGateFamilies["git_write"]; exists && guardToInt(v) > 0 {
		hasGitWrite = true
	}

	hasResidual := false
	for _, res := range residual {
		if res == "HISTORICAL_GIT_WRITE_BEFORE_STRUCTURED_GATE" {
			hasResidual = true
			break
		}
	}

	ok := reportStatusOK &&
		actionabilityOK &&
		guardStringVal(gitGate, "status") == "PASS" &&
		!hasGitWrite &&
		guardToInt(summary["workspace_stop_failures_total"]) > 0 &&
		guardToInt(postRepairShapes["shell_no_write_target_detected"]) > 0 &&
		hasResidual

	sort.Strings(reasons)
	sort.Strings(residual)
	*checks = append(*checks, guardCheckResult{
		name:   "historical codex/dos actionability",
		file:   rel,
		passed: ok,
		detail: fmt.Sprintf("audit=%s actionability=%s workspace_stop_failures=%v active_consecutive=%d reasons=%v residual=%v post_gate=%v",
			reportStatus, actionStatus, summary["workspace_stop_failures_total"], activeConsecutive, reasons, residual, postGateFamilies),
	})
}

func checkGuardClaudeHistorical(root string, checks *[]guardCheckResult) {
	relJSON := guardMCPClaudeHistJSONPath
	relMD := guardMCPClaudeHistMDPath
	data, err := readGuardJSON(root, relJSON)
	if err != nil {
		*checks = append(*checks, guardCheckResult{
			name:   "claude code historical replay",
			file:   relJSON,
			passed: false,
			detail: err.Error(),
		})
		return
	}
	md, err := readGuardText(root, relMD)
	if err != nil {
		*checks = append(*checks, guardCheckResult{
			name:   "claude code historical replay",
			file:   relMD,
			passed: false,
			detail: err.Error(),
		})
		return
	}
	verdicts := guardMapVal(data, "verdict_counts")
	reasons := guardMapVal(data, "reason_counts")
	shape := guardMapVal(data, "transcript_shape")
	tags := guardMapVal(shape, "evidence_tag_counts")
	remediation := guardMapVal(shape, "remediation_session_counts")
	topFriction := guardSliceVal(data, "top_friction_sessions")

	jsonBytes, _ := json.Marshal(data)
	serialized := string(jsonBytes) + md
	leakedTokens := []string{"rm -rf", "README.md", "tool_result", "secret result", "C:\\Users\\", "C:/Users/"}
	leakedPayload := false
	for _, token := range leakedTokens {
		if strings.Contains(serialized, token) {
			leakedPayload = true
			break
		}
	}

	rootLabels := guardSliceVal(data, "root_labels")
	sessionsDiscovered := guardToInt(data["sessions_discovered"])
	sessionsAudited := guardToInt(data["sessions_audited"])
	toolCallsSeen := guardToInt(data["tool_calls_seen"])
	uniqueReplayed := guardToInt(data["unique_tool_calls_replayed"])
	summarizedSessions := guardToInt(shape["summarized_sessions"])

	ok := guardStringVal(data, "schema") == "fak-claude-historical-guard-audit/1" &&
		guardStringVal(data, "status") == "PASS" &&
		guardBoolVal(data, "all_accounts") == true &&
		len(rootLabels) >= 2 &&
		sessionsDiscovered >= 1 &&
		sessionsAudited >= 1 &&
		toolCallsSeen >= 1 &&
		uniqueReplayed >= 1 &&
		guardToInt(verdicts["ALLOW"]) >= 1 &&
		guardToInt(verdicts["DENY"]) >= 1 &&
		guardToInt(reasons["POLICY_BLOCK"]) >= 1 &&
		summarizedSessions >= sessionsDiscovered &&
		guardToInt(tags["HOOK_OR_API_WALL_FEEDBACK"]) >= 1 &&
		guardToInt(tags["HOST_PERMISSION_INTERRUPT"]) >= 1 &&
		guardToInt(remediation["clear_hook_or_api_wall_feedback"]) >= 1 &&
		len(topFriction) >= 1 &&
		guardBoolVal(data, "truncated") == false &&
		strings.Contains(md, "status: **`PASS`**") &&
		strings.Contains(md, "Transcript Friction Signals") &&
		strings.Contains(md, "It never writes prompts, tool arguments, tool results, or raw transcript text.") &&
		!leakedPayload

	*checks = append(*checks, guardCheckResult{
		name:   "claude code historical replay",
		file:   relJSON,
		passed: ok,
		detail: fmt.Sprintf("status=%s sessions=%d calls=%d verdicts=%v tags=%v remediation=%v leaked_payload=%v",
			guardStringVal(data, "status"), sessionsAudited, toolCallsSeen, verdicts, tags, remediation, leakedPayload),
	})
}

func checkGuardClaudeLive(root string, checks *[]guardCheckResult) {
	rel := guardMCPClaudeLivePath
	data, err := readGuardJSON(root, rel)
	if err != nil {
		*checks = append(*checks, guardCheckResult{
			name:   "claude code live pilot",
			file:   rel,
			passed: false,
			detail: err.Error(),
		})
		return
	}
	danger := guardMapVal(data, "dangerous_attempt")
	useful := guardMapVal(data, "useful_continuation")
	final := guardMapVal(useful, "final_message")
	session := guardMapVal(data, "session")

	sameSession := guardStringVal(useful, "same_claude_session_id") != "" &&
		guardStringVal(useful, "same_claude_session_id") == guardStringVal(session, "claude_session_id")

	ok := guardStringVal(data, "status") == "PASS" &&
		guardVerdictKind(danger) == "DENY" &&
		guardVerdictReason(danger) == "POLICY_BLOCK" &&
		guardVerdictKind(useful) == "ALLOW" &&
		sameSession &&
		guardBoolVal(final, "useful_completed") == true

	*checks = append(*checks, guardCheckResult{
		name:   "claude code live pilot",
		file:   rel,
		passed: ok,
		detail: fmt.Sprintf("status=%s same_session=%v", guardStringVal(data, "status"), sameSession),
	})
}

func checkGuardCodexMCPLive(root string, checks *[]guardCheckResult) {
	rel := guardMCPCodexMCPLivePath
	data, err := readGuardJSON(root, rel)
	if err != nil {
		*checks = append(*checks, guardCheckResult{
			name:   "codex mcp live pilot",
			file:   rel,
			passed: false,
			detail: err.Error(),
		})
		return
	}
	danger := guardMapVal(data, "dangerous_attempt")
	useful := guardMapVal(data, "useful_continuation")
	final := guardMapVal(data, "final_message")

	ok := guardStringVal(data, "status") == "PASS" &&
		guardVerdictKind(danger) == "DENY" &&
		guardVerdictReason(danger) == "POLICY_BLOCK" &&
		guardVerdictKind(useful) == "ALLOW" &&
		guardBoolVal(final, "denied_attempt") == true &&
		guardBoolVal(final, "useful_continued") == true

	*checks = append(*checks, guardCheckResult{
		name:   "codex mcp live pilot",
		file:   rel,
		passed: ok,
		detail: fmt.Sprintf("status=%s denied=%v continued=%v", guardStringVal(data, "status"), guardBoolVal(final, "denied_attempt"), guardBoolVal(final, "useful_continued")),
	})
}

func checkGuardOpenAIAgentsAdapter(root string, checks *[]guardCheckResult) {
	relDemo := guardMCPOpenAIAgentsDemo
	relOutput := guardMCPOpenAIAgentsOutput

	fi, err := os.Stat(filepath.Join(root, filepath.FromSlash(relDemo)))
	demoExists := err == nil && !fi.IsDir()

	text, err := readGuardText(root, relOutput)
	if err != nil {
		*checks = append(*checks, guardCheckResult{
			name:   "openai agents adapter proof",
			file:   relOutput,
			passed: false,
			detail: err.Error(),
		})
		return
	}
	required := []string{
		"input guardrail blocks git_push",
		"behavior=reject_content verdict=DENY reason=POLICY_BLOCK executed=false",
		"input guardrail allows git_status",
		"output guardrail admits git_status result",
		"output guardrail quarantines web_fetch result",
		"verdict=QUARANTINE reason=SECRET_EXFIL",
		"summary: PASS",
	}
	var missing []string
	for _, item := range required {
		if !strings.Contains(text, item) {
			missing = append(missing, item)
		}
	}
	ok := demoExists && len(missing) == 0
	detail := "demo and captured output prove deny/run/quarantine mapping"
	if !ok {
		detail = fmt.Sprintf("demo_exists=%v missing=%v", demoExists, missing)
	}
	*checks = append(*checks, guardCheckResult{
		name:   "openai agents adapter proof",
		file:   relOutput,
		passed: ok,
		detail: detail,
	})
}

func checkGuardOpenAILivePrereq(root string, checks *[]guardCheckResult) {
	relJSON := guardMCPOpenAIPrereqJSON
	relMD := guardMCPOpenAIPrereqMD
	data, err := readGuardJSON(root, relJSON)
	if err != nil {
		*checks = append(*checks, guardCheckResult{
			name:   "openai hosted live prereqs",
			file:   relJSON,
			passed: false,
			detail: err.Error(),
		})
		return
	}
	md, err := readGuardText(root, relMD)
	if err != nil {
		*checks = append(*checks, guardCheckResult{
			name:   "openai hosted live prereqs",
			file:   relMD,
			passed: false,
			detail: err.Error(),
		})
		return
	}
	blockers := guardStringSliceVal(data, "blockers")
	sort.Strings(blockers)

	jsonBytes, _ := json.Marshal(data)
	serialized := string(jsonBytes) + md
	leakTokens := []string{"sk-", "OPENAI_API_KEY_value", "access_token_value", "refresh_token_value", "id_token_value"}
	secretLeak := false
	for _, token := range leakTokens {
		if strings.Contains(serialized, token) {
			secretLeak = true
			break
		}
	}

	status := guardStringVal(data, "status")
	schema := guardStringVal(data, "schema")
	ok := false

	if guardBoolVal(data, "codex_login_ready") == true {
		authSources := guardMapVal(data, "auth_sources")
		hasNoAPIKeyBlocker := true
		for _, b := range blockers {
			if b == "OPENAI_API_KEY is not set" {
				hasNoAPIKeyBlocker = false
				break
			}
		}
		ok = schema == "fak-openai-live-prereq-audit/1" &&
			(status == "PARTIAL" || status == "READY") &&
			guardBoolVal(data, "hosted_openai_ready") == true &&
			guardBoolVal(authSources, "codex_login") == true &&
			hasNoAPIKeyBlocker &&
			strings.Contains(md, fmt.Sprintf("status: **`%s`**", status)) &&
			strings.Contains(md, "It never writes API key values") &&
			strings.Contains(md, "Codex token values") &&
			!secretLeak
	} else {
		reqBlockers := []string{
			"OPENAI_API_KEY is not set",
			"openai-agents distribution is not installed",
			"importable agents module is not an installed OpenAI Agents SDK distribution",
		}
		hasAll := true
		for _, req := range reqBlockers {
			found := false
			for _, b := range blockers {
				if b == req {
					found = true
					break
				}
			}
			if !found {
				hasAll = false
				break
			}
		}
		ok = schema == "fak-openai-live-prereq-audit/1" &&
			status == "BLOCKED_ENV" &&
			guardBoolVal(data, "hosted_openai_ready") == false &&
			guardBoolVal(data, "agents_sdk_ready") == false &&
			hasAll &&
			strings.Contains(md, "status: **`BLOCKED_ENV`**") &&
			strings.Contains(md, "It never writes API key values") &&
			!secretLeak
	}

	*checks = append(*checks, guardCheckResult{
		name:   "openai hosted live prereqs",
		file:   relJSON,
		passed: ok,
		detail: fmt.Sprintf("status=%s blockers=%v secret_leak=%v", status, blockers, secretLeak),
	})
}

func checkGuardOpenAIHostedLivePilot(root string, checks *[]guardCheckResult) {
	relJSON := guardMCPOpenAIHostedJSON
	relMD := guardMCPOpenAIHostedMD
	data, err := readGuardJSON(root, relJSON)
	if err != nil {
		*checks = append(*checks, guardCheckResult{
			name:   "openai hosted live pilot",
			file:   relJSON,
			passed: false,
			detail: err.Error(),
		})
		return
	}
	md, err := readGuardText(root, relMD)
	if err != nil {
		*checks = append(*checks, guardCheckResult{
			name:   "openai hosted live pilot",
			file:   relMD,
			passed: false,
			detail: err.Error(),
		})
		return
	}
	status := guardStringVal(data, "status")
	schema := guardStringVal(data, "schema")
	jsonBytes, _ := json.Marshal(data)
	serialized := string(jsonBytes) + md
	secretLeak := strings.Contains(serialized, "sk-") || strings.Contains(serialized, "OPENAI_API_KEY_value")
	blockers := guardStringSliceVal(data, "blockers")
	sort.Strings(blockers)
	prereqs := guardMapVal(data, "prereqs")

	ok := false
	if status == "BLOCKED_ENV" {
		reqBlockers := []string{
			"OPENAI_API_KEY is not set",
			"openai-agents distribution is not installed",
			"importable agents module is not an installed OpenAI Agents SDK distribution",
		}
		hasAll := true
		for _, req := range reqBlockers {
			found := false
			for _, b := range blockers {
				if b == req {
					found = true
					break
				}
			}
			if !found {
				hasAll = false
				break
			}
		}
		ok = schema == "fak-openai-hosted-live-pilot/1" &&
			guardBoolVal(prereqs, "hosted_openai_ready") == false &&
			hasAll &&
			strings.Contains(md, "status: **`BLOCKED_ENV`**") &&
			!secretLeak
	} else if status == "PASS" {
		guard := guardMapVal(data, "guard")
		hosted := guardMapVal(data, "hosted_openai")
		danger := guardMapVal(guard, "dangerous_attempt")
		useful := guardMapVal(guard, "useful_continuation")
		authSource := guardStringVal(hosted, "auth_source")
		if authSource == "" {
			authSource = guardStringVal(data, "auth_source")
		}
		hostedProofOK := false
		if authSource == "codex_login" {
			_, hasSHA := hosted["output_text_sha256"]
			hostedProofOK = guardToInt(hosted["codex_exec_exit_code"]) == 0 &&
				guardBoolVal(hosted, "contains_expected_marker") == true &&
				hasSHA
		} else if authSource == "platform_api_key" {
			_, hasSHA := hosted["output_text_sha256"]
			hostedProofOK = guardBoolVal(hosted, "response_id_present") == true &&
				guardBoolVal(hosted, "contains_expected_marker") == true &&
				hasSHA
		}
		hostedJSON, _ := json.Marshal(hosted)
		noRawResponseText := !strings.Contains(string(hostedJSON), "raw hosted OpenAI response text")

		dangerVerdict := guardMapVal(danger, "verdict")
		usefulVerdict := guardMapVal(useful, "verdict")

		dangerKind := guardStringVal(dangerVerdict, "kind")
		if dangerKind == "" {
			dangerKind = guardVerdictKind(danger)
		}
		dangerReason := guardStringVal(dangerVerdict, "reason")
		if dangerReason == "" {
			dangerReason = guardVerdictReason(danger)
		}
		usefulKind := guardStringVal(usefulVerdict, "kind")
		if usefulKind == "" {
			usefulKind = guardVerdictKind(useful)
		}

		ok = schema == "fak-openai-hosted-live-pilot/1" &&
			guardStringVal(guard, "status") == "PASS" &&
			guardStringVal(hosted, "status") == "PASS" &&
			dangerKind == "DENY" &&
			dangerReason == "POLICY_BLOCK" &&
			guardBoolVal(danger, "executed") == false &&
			usefulKind == "ALLOW" &&
			hostedProofOK &&
			noRawResponseText &&
			!secretLeak
	}

	*checks = append(*checks, guardCheckResult{
		name:   "openai hosted live pilot",
		file:   relJSON,
		passed: ok,
		detail: fmt.Sprintf("status=%s blockers=%v secret_leak=%v", status, blockers, secretLeak),
	})
}

// runGuardMCPStatusAudit runs all 11 status checks over workspace root, matching
// guard_mcp_status_audit.py collect().
func runGuardMCPStatusAudit(root string) []guardCheckResult {
	var checks []guardCheckResult
	checkGuardStatusPacket(root, &checks)
	checkGuardDefaultTests(root, &checks)
	checkGuardMCPStdio(root, &checks)
	checkGuardGitGateReports(root, &checks)
	checkGuardHistoricalAudit(root, &checks)
	checkGuardClaudeHistorical(root, &checks)
	checkGuardClaudeLive(root, &checks)
	checkGuardCodexMCPLive(root, &checks)
	checkGuardOpenAIAgentsAdapter(root, &checks)
	checkGuardOpenAILivePrereq(root, &checks)
	checkGuardOpenAIHostedLivePilot(root, &checks)
	return checks
}

// gateGuardMCPStatusTree is the GUARD_MCP_STATUS tree hygiene gate.
func gateGuardMCPStatusTree(t *TrackedTree) ([]Finding, error) {
	checks := runGuardMCPStatusAudit(t.Root)
	var findings []Finding
	for _, c := range checks {
		if !c.passed {
			findings = append(findings, Finding{
				Gate:   "GUARD_MCP_STATUS",
				File:   c.file,
				Detail: c.name + ": " + c.detail,
			})
		}
	}
	return findings, nil
}
