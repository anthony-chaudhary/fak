package main

import (
	"encoding/json"
	"fmt"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"
)

func buildTUIGuardReport(artifacts []tuiGuardArtifact, at time.Time) tuiGuardReport {
	rows := []tuiGuardRow{}
	sources := make([]tuiGuardSource, 0, len(artifacts))
	for _, artifact := range artifacts {
		schema := tuiGuardString(artifact.Raw, "schema")
		status := tuiGuardEffectiveSourceStatus(artifact)
		sources = append(sources, tuiGuardSource{
			Path:   artifact.Path,
			Schema: schema,
			Status: status,
		})
		rows = append(rows, tuiGuardRowsForArtifact(artifact)...)
	}
	for i := range rows {
		rows[i].Tags, rows[i].Attention = scoreTUIGuardRow(rows[i])
	}
	sort.SliceStable(rows, func(i, j int) bool {
		if rows[i].Rank > 0 && rows[j].Rank > 0 && rows[i].Rank != rows[j].Rank {
			return rows[i].Rank < rows[j].Rank
		}
		if rows[i].Attention != rows[j].Attention {
			return rows[i].Attention > rows[j].Attention
		}
		if rows[i].Artifact != rows[j].Artifact {
			return rows[i].Artifact < rows[j].Artifact
		}
		return rows[i].Kind < rows[j].Kind
	})
	counts := countTUIGuard(rows, sources)
	status := tuiGuardStatus(counts)
	return tuiGuardReport{
		Schema:  tuiGuardSchema,
		At:      at.UTC().Format(time.RFC3339),
		Source:  tuiGuardSourceLabel(artifacts),
		Status:  status,
		Counts:  counts,
		Actions: tuiGuardActions(counts),
		Rows:    rows,
		Sources: sources,
	}
}

func tuiGuardRowsForArtifact(artifact tuiGuardArtifact) []tuiGuardRow {
	raw := artifact.Raw
	schema := tuiGuardString(raw, "schema")
	if preflight := tuiGuardMap(raw["preflight"]); preflight != nil {
		return []tuiGuardRow{tuiGuardPreflightProbeRow(artifact, preflight)}
	}
	if schema == "fak-guard-mcp-status-audit/1" || raw["default_blockers"] != nil {
		return tuiGuardStatusAuditRows(artifact)
	}
	if schema == "fak-codex-dos-recent-audit/1" || (raw["actionability"] != nil && raw["codex_hook_fast_path"] != nil) {
		return tuiGuardCodexRecentRows(artifact)
	}
	if schema == "fak-claude-historical-guard-audit/1" || raw["verdict_counts"] != nil || raw["non_allow_samples"] != nil {
		return tuiGuardHistoricalRows(artifact)
	}
	if schema == "fak-vendor-live-pilot/1" || raw["dangerous_attempt"] != nil {
		return tuiGuardVendorRows(artifact)
	}
	if schema == "fak-combined-dogfood/1" || raw["floor"] != nil {
		return tuiGuardCombinedRows(artifact)
	}
	if checks, ok := raw["checks"].([]any); ok {
		return tuiGuardCheckRows(artifact, checks)
	}
	row := tuiGuardGenericRow(artifact)
	if row.Kind == "" {
		return nil
	}
	return []tuiGuardRow{row}
}

func tuiGuardEffectiveSourceStatus(artifact tuiGuardArtifact) string {
	status := tuiGuardString(artifact.Raw, "status")
	if tuiGuardStatusClass(status) == "fail" {
		return status
	}
	schema := tuiGuardString(artifact.Raw, "schema")
	if schema == "fak-guard-mcp-status-audit/1" || artifact.Raw["default_blockers"] != nil {
		if blockers, ok := artifact.Raw["default_blockers"].([]any); ok && len(blockers) > 0 {
			return "WARN"
		}
	}
	return status
}

func tuiGuardPreflightProbeRow(artifact tuiGuardArtifact, preflight map[string]any) tuiGuardRow {
	raw := artifact.Raw
	status := tuiGuardString(raw, "status")
	verdict := tuiGuardString(preflight, "verdict")
	reason := tuiGuardString(preflight, "reason")
	expected := ""
	if tuiGuardBool(raw, "expect_deny") {
		expected = "expected deny"
		if want := tuiGuardString(raw, "expect_reason"); want != "" {
			expected += " " + want
		}
	}
	detail := strings.TrimSpace(strings.Join(nonEmptyTUI([]string{
		tuiGuardString(raw, "command_label"),
		tuiGuardString(raw, "policy"),
		expected,
	}), "  "))
	return tuiGuardRow{
		Artifact: tuiGuardArtifactName(artifact.Path),
		Kind:     "preflight-probe",
		Tool:     tuiGuardString(raw, "tool"),
		Verdict:  verdict,
		Reason:   reason,
		By:       tuiGuardString(preflight, "by"),
		Status:   status,
		Detail:   detail,
		Count:    1,
	}
}

func tuiGuardStatusAuditRows(artifact tuiGuardArtifact) []tuiGuardRow {
	raw := artifact.Raw
	artifactName := tuiGuardArtifactName(artifact.Path)
	summary := tuiGuardMap(raw["summary"])
	status := tuiGuardEffectiveSourceStatus(artifact)
	rows := []tuiGuardRow{
		{
			Artifact: artifactName,
			Kind:     "guard-status-audit",
			Status:   status,
			Detail: strings.Join(nonEmptyTUI([]string{
				fmt.Sprintf("checks=%d/%d", tuiGuardInt(summary, "passed"), tuiGuardInt(summary, "total")),
				fmt.Sprintf("failed=%d", tuiGuardInt(summary, "failed")),
				fmt.Sprintf("default_blockers=%d", tuiGuardInt(summary, "default_blockers")),
				fmt.Sprintf("active_default_blockers=%d", tuiGuardInt(summary, "active_default_blockers")),
			}), "  "),
			Count: 1,
		},
	}
	if blockers, ok := raw["default_blockers"].([]any); ok {
		for _, blocker := range blockers {
			m := tuiGuardMap(blocker)
			if m == nil {
				continue
			}
			rows = append(rows, tuiGuardDefaultBlockerRow(artifactName, m))
			rows = append(rows, tuiGuardSettlementRows(artifactName, m)...)
		}
	}
	if checks, ok := raw["checks"].([]any); ok {
		for _, check := range checks {
			m := tuiGuardMap(check)
			if m == nil || strings.EqualFold(tuiGuardString(m, "status"), "PASS") {
				continue
			}
			rows = append(rows, tuiGuardRow{
				Artifact: artifactName,
				Kind:     "check",
				Status:   tuiGuardString(m, "status"),
				Detail:   strings.TrimSpace(tuiGuardString(m, "name") + "  " + tuiGuardString(m, "detail")),
				Count:    1,
			})
		}
	}
	return rows
}

func tuiGuardDefaultBlockerRow(artifactName string, blocker map[string]any) tuiGuardRow {
	evidence := tuiGuardMap(blocker["evidence"])
	code := tuiGuardString(blocker, "code")
	status := tuiGuardString(blocker, "status")
	return tuiGuardRow{
		Artifact: artifactName,
		Kind:     "default-blocker",
		Tool:     tuiGuardString(blocker, "surface"),
		Reason:   code,
		Status:   status,
		Detail:   tuiGuardDefaultBlockerDetail(blocker, evidence),
		Count:    tuiGuardDefaultBlockerCount(evidence),
		Rank:     tuiGuardInt(blocker, "rank"),
	}
}

func tuiGuardSettlementRows(artifactName string, blocker map[string]any) []tuiGuardRow {
	evidence := tuiGuardMap(blocker["evidence"])
	if evidence == nil {
		return nil
	}
	rows := []tuiGuardRow{}
	for _, key := range []string{"recent_review_plan", "stale_settlement_plan"} {
		items, ok := evidence[key].([]any)
		if !ok {
			continue
		}
		for _, item := range items {
			m := tuiGuardMap(item)
			if m == nil {
				continue
			}
			rows = append(rows, tuiGuardRow{
				Artifact: artifactName,
				Kind:     "settlement-candidate",
				Tool:     tuiGuardString(blocker, "surface"),
				Reason:   tuiGuardString(m, "settlement_action"),
				Status:   "WARN",
				Detail: strings.Join(nonEmptyTUI([]string{
					"code=" + tuiGuardString(blocker, "code"),
					"session=" + tuiGuardString(m, "session_id"),
					"path=" + tuiGuardString(m, "marker_path"),
					fmt.Sprintf("total=%d", tuiGuardInt(m, "total")),
					fmt.Sprintf("consecutive=%d", tuiGuardInt(m, "consecutive")),
					fmt.Sprintf("age_seconds=%d", tuiGuardInt(m, "age_seconds")),
					"origin=" + tuiGuardString(m, "origin"),
					"project=" + tuiGuardString(m, "transcript_project"),
					"evidence=" + tuiGuardCompactJSON(m["evidence_tags"]),
				}), "  "),
				Count: maxTUI(1, tuiGuardInt(m, "consecutive")),
				Rank:  tuiGuardInt(blocker, "rank"),
			})
			if len(rows) >= 8 {
				return rows
			}
		}
	}
	return rows
}

func tuiGuardDefaultBlockerDetail(blocker, evidence map[string]any) string {
	bits := []string{
		"code=" + tuiGuardString(blocker, "code"),
		"surface=" + tuiGuardString(blocker, "surface"),
		fmt.Sprintf("rank=%d", tuiGuardInt(blocker, "rank")),
	}
	bits = appendGuardIntMapBits(bits, evidence,
		"active_settlement_action_counts",
		"recent_active_settlement_action_counts",
		"stale_active_settlement_action_counts")
	for _, key := range []string{
		"recent_active_consecutive_total",
		"active_consecutive_total",
		"stale_active_consecutive_total",
		"shell_no_write_target_detected",
		"sessions_audited",
		"tool_calls_seen",
		"max_result_chars",
	} {
		if n := tuiGuardInt(evidence, key); n > 0 {
			bits = append(bits, fmt.Sprintf("%s=%d", key, n))
		}
	}
	if tags := tuiGuardIntMap(evidence["evidence_tag_counts"]); len(tags) > 0 {
		bits = append(bits, "tags="+tuiGuardCompactJSON(tags))
	}
	bits = appendGuardIntMapBits(bits, evidence,
		"recent_active_origin_counts",
		"stale_active_origin_counts",
		"origin_counts")
	if blockers, ok := evidence["blockers"].([]any); ok && len(blockers) > 0 {
		bits = append(bits, "blockers="+tuiGuardCompactJSON(blockers))
	}
	if provedAt := tuiGuardString(evidence, "proved_at"); provedAt != "" {
		bits = append(bits, "proved_at="+provedAt)
	}
	if next := tuiGuardString(blocker, "next_action"); next != "" {
		bits = append(bits, "next="+next)
	}
	return strings.Join(nonEmptyTUI(bits), "  ")
}

// appendGuardIntMapBits appends "key=<compact-json>" for each named evidence key whose value
// decodes to a non-empty int map, returning the grown slice. It folds the repeated
// range-over-keys / non-empty-int-map / append shape into one named intent.
func appendGuardIntMapBits(bits []string, evidence map[string]any, keys ...string) []string {
	for _, key := range keys {
		if counts := tuiGuardIntMap(evidence[key]); len(counts) > 0 {
			bits = append(bits, key+"="+tuiGuardCompactJSON(counts))
		}
	}
	return bits
}

func tuiGuardDefaultBlockerCount(evidence map[string]any) int {
	for _, key := range []string{
		"recent_active_consecutive_total",
		"shell_no_write_target_detected",
		"stale_active_consecutive_total",
		"sessions_audited",
		"tool_calls_seen",
		"one_day_failures_total",
		"active_consecutive_total",
	} {
		if n := tuiGuardInt(evidence, key); n > 0 {
			return n
		}
	}
	for _, key := range []string{"post_repair_mutating_shell_family_counts", "evidence_tag_counts"} {
		total := 0
		for _, n := range tuiGuardIntMap(evidence[key]) {
			total += n
		}
		if total > 0 {
			return total
		}
	}
	if blockers, ok := evidence["blockers"].([]any); ok && len(blockers) > 0 {
		return len(blockers)
	}
	return 1
}

func tuiGuardHistoricalRows(artifact tuiGuardArtifact) []tuiGuardRow {
	raw := artifact.Raw
	artifactName := tuiGuardArtifactName(artifact.Path)
	rows := []tuiGuardRow{}
	for verdict, count := range tuiGuardIntMap(raw["verdict_counts"]) {
		rows = append(rows, tuiGuardRow{
			Artifact: artifactName,
			Kind:     "verdict-count",
			Verdict:  strings.ToUpper(verdict),
			Status:   tuiGuardString(raw, "status"),
			Detail:   tuiGuardHistoricalDetail(raw),
			Count:    count,
		})
	}
	for reason, count := range tuiGuardIntMap(raw["reason_counts"]) {
		rows = append(rows, tuiGuardRow{
			Artifact: artifactName,
			Kind:     "reason-count",
			Reason:   strings.ToUpper(reason),
			Status:   tuiGuardString(raw, "status"),
			Detail:   tuiGuardString(raw, "policy"),
			Count:    count,
		})
	}
	shape := tuiGuardMap(raw["transcript_shape"])
	if tags := tuiGuardIntMap(shape["evidence_tag_counts"]); len(tags) > 0 {
		for tag, count := range tags {
			rows = append(rows, tuiGuardRow{
				Artifact: artifactName,
				Kind:     "claude-friction",
				Reason:   strings.ToUpper(tag),
				Status:   "WARN",
				Detail: strings.Join(nonEmptyTUI([]string{
					fmt.Sprintf("sessions=%d", tuiGuardInt(shape, "summarized_sessions")),
					fmt.Sprintf("max_result_chars=%d", tuiGuardInt(shape, "max_result_chars")),
				}), "  "),
				Count: count,
			})
		}
	}
	if sessions, ok := raw["top_friction_sessions"].([]any); ok {
		for _, session := range sessions {
			m := tuiGuardMap(session)
			if m == nil {
				continue
			}
			rows = append(rows, tuiGuardRow{
				Artifact: artifactName,
				Kind:     "claude-friction-session",
				Status:   "WARN",
				Detail: strings.Join(nonEmptyTUI([]string{
					"session=" + tuiGuardString(m, "session_digest"),
					"root=" + tuiGuardString(m, "root_label"),
					fmt.Sprintf("tool_calls=%d", tuiGuardInt(m, "tool_calls")),
					fmt.Sprintf("marker_lines=%d", tuiGuardInt(m, "marker_lines")),
					fmt.Sprintf("max_result_chars=%d", tuiGuardInt(m, "max_result_chars")),
					"tags=" + tuiGuardCompactJSON(m["evidence_tags"]),
				}), "  "),
				Count: tuiGuardInt(m, "marker_lines"),
			})
			if len(rows) >= 18 {
				break
			}
		}
	}
	if samples, ok := raw["non_allow_samples"].([]any); ok {
		for _, sample := range samples {
			m := tuiGuardMap(sample)
			if m == nil {
				continue
			}
			rows = append(rows, tuiGuardRow{
				Artifact: artifactName,
				Kind:     "sample",
				Tool:     tuiGuardString(m, "tool"),
				Verdict:  tuiGuardString(m, "verdict"),
				Reason:   tuiGuardString(m, "reason"),
				By:       tuiGuardString(m, "by"),
				Status:   tuiGuardString(raw, "status"),
				Detail:   firstNonEmptyTUI(tuiGuardString(m, "claim"), tuiGuardString(m, "call_digest")),
			})
		}
	}
	return rows
}

func tuiGuardHistoricalDetail(raw map[string]any) string {
	bits := []string{}
	if policy := tuiGuardString(raw, "policy"); policy != "" {
		bits = append(bits, "policy="+policy)
	}
	if calls := tuiGuardInt(raw, "tool_calls_seen"); calls > 0 {
		bits = append(bits, fmt.Sprintf("tool_calls=%d", calls))
	}
	if sessions := tuiGuardInt(raw, "sessions_audited"); sessions > 0 {
		bits = append(bits, fmt.Sprintf("sessions=%d", sessions))
	}
	return strings.Join(bits, "  ")
}

func tuiGuardCodexRecentRows(artifact tuiGuardArtifact) []tuiGuardRow {
	raw := artifact.Raw
	artifactName := tuiGuardArtifactName(artifact.Path)
	status := tuiGuardString(raw, "status")
	summary := tuiGuardMap(raw["summary"])
	actionability := tuiGuardMap(raw["actionability"])
	hook := tuiGuardMap(raw["codex_hook_fast_path"])
	gitGate := tuiGuardMap(raw["git_gate_evidence"])
	workspaceStop := tuiGuardMap(raw["workspace_stop_failures"])

	rows := []tuiGuardRow{
		{
			Artifact: artifactName,
			Kind:     "codex-actionability",
			Status:   firstNonEmptyTUI(tuiGuardString(actionability, "status"), status),
			Detail: strings.Join(nonEmptyTUI([]string{
				fmt.Sprintf("sessions=%d/%d", tuiGuardInt(raw, "sessions_audited"), tuiGuardInt(raw, "codex_threads_discovered")),
				"reasons=" + tuiGuardCompactJSON(actionability["reasons"]),
				"residual=" + tuiGuardCompactJSON(actionability["residual"]),
				fmt.Sprintf("delegates=%d", tuiGuardInt(actionability, "delegate_count")),
			}), "  "),
			Count: 1,
		},
	}
	if hook != nil {
		rows = append(rows, tuiGuardRow{
			Artifact: artifactName,
			Kind:     "codex-hook-fast-path",
			Status:   tuiGuardString(hook, "status"),
			Detail: strings.Join(nonEmptyTUI([]string{
				"modes=" + tuiGuardCompactJSON(hook["codex_command_modes"]),
				"reason=" + tuiGuardString(hook, "reason"),
			}), "  "),
			Count: 1,
		})
	}
	if gitGate != nil {
		rows = append(rows, tuiGuardRow{
			Artifact: artifactName,
			Kind:     "codex-git-gate",
			Status:   tuiGuardString(gitGate, "status"),
			Detail: strings.Join(nonEmptyTUI([]string{
				"proved_at=" + tuiGuardString(gitGate, "proved_at"),
				"missing=" + tuiGuardCompactJSON(gitGate["missing"]),
			}), "  "),
			Count: 1,
		})
	}
	if workspaceStop != nil && (tuiGuardInt(workspaceStop, "markers") > 0 || tuiGuardInt(workspaceStop, "total_failures") > 0) {
		detail := []string{
			fmt.Sprintf("markers=%d", tuiGuardInt(workspaceStop, "markers")),
			fmt.Sprintf("failures=%d", tuiGuardInt(workspaceStop, "total_failures")),
			fmt.Sprintf("nonzero=%d", tuiGuardInt(workspaceStop, "nonzero_total_markers")),
			fmt.Sprintf("active=%d", tuiGuardInt(workspaceStop, "active_consecutive_markers")),
			fmt.Sprintf("active_consecutive=%d", tuiGuardInt(workspaceStop, "active_consecutive_total")),
			fmt.Sprintf("recent=%d", tuiGuardInt(workspaceStop, "recent_active_consecutive_markers")),
			fmt.Sprintf("recent_consecutive=%d", tuiGuardInt(workspaceStop, "recent_active_consecutive_total")),
			fmt.Sprintf("stale=%d", tuiGuardInt(workspaceStop, "stale_active_consecutive_markers")),
			fmt.Sprintf("stale_consecutive=%d", tuiGuardInt(workspaceStop, "stale_active_consecutive_total")),
			fmt.Sprintf("healed=%d", tuiGuardInt(workspaceStop, "healed_nonzero_markers")),
			fmt.Sprintf("zero=%d", tuiGuardInt(workspaceStop, "zero_total_markers")),
			fmt.Sprintf("max_consecutive=%d", tuiGuardInt(workspaceStop, "max_consecutive")),
		}
		for _, key := range []string{
			"active_settlement_action_counts",
			"recent_active_settlement_action_counts",
			"stale_active_settlement_action_counts",
			"origin_counts",
			"recent_active_origin_counts",
			"stale_active_origin_counts",
		} {
			if counts := tuiGuardIntMap(workspaceStop[key]); len(counts) > 0 {
				detail = append(detail, key+"="+tuiGuardCompactJSON(counts))
			}
		}
		rows = append(rows, tuiGuardRow{
			Artifact: artifactName,
			Kind:     "stopfailure-api-wall",
			Status:   tuiGuardString(workspaceStop, "status"),
			Detail:   strings.Join(detail, " "),
			Count:    1,
		})
		topStop := workspaceStop["top_recent_active"]
		if topStop == nil {
			topStop = workspaceStop["top_active"]
		}
		if topStop == nil {
			topStop = workspaceStop["top_nonzero"]
		}
		if topStop == nil {
			topStop = workspaceStop["recent"]
		}
		rows = append(rows, tuiGuardStopFailureSessionRows(artifactName, topStop)...)
	}
	if mutating := tuiGuardIntMap(actionability["post_repair_mutating_shell_family_counts"]); len(mutating) > 0 {
		rows = append(rows, tuiGuardRow{
			Artifact: artifactName,
			Kind:     "codex-mutating-shell",
			Status:   "WARN",
			Detail:   "families=" + tuiGuardCompactJSON(mutating),
			Count:    1,
		})
	}
	if residualsContain(actionability["residual"], "HOST_SHELL_OPACITY") || tuiGuardIntMap(actionability["post_repair_shell_shape_counts"])["shell_no_write_target_detected"] > 0 {
		rows = append(rows, tuiGuardRow{
			Artifact: artifactName,
			Kind:     "codex-shell-opacity",
			Status:   firstNonEmptyTUI(tuiGuardString(actionability, "status"), status),
			Detail: strings.Join(nonEmptyTUI([]string{
				"shapes=" + tuiGuardCompactJSON(actionability["post_repair_shell_shape_counts"]),
				"families=" + tuiGuardCompactJSON(actionability["post_repair_shell_family_counts"]),
				fmt.Sprintf("unknown_tree=%d", tuiGuardInt(summary, "unknown_tree_admission_warnings")),
			}), "  "),
			Count: 1,
		})
	}
	return rows
}

func tuiGuardStopFailureSessionRows(artifactName string, recent any) []tuiGuardRow {
	items, ok := recent.([]any)
	if !ok {
		return nil
	}
	rows := []tuiGuardRow{}
	for _, item := range items {
		m := tuiGuardMap(item)
		if m == nil {
			continue
		}
		total := tuiGuardInt(m, "total")
		if total <= 0 {
			continue
		}
		transcript := tuiGuardMap(m["transcript"])
		transcriptSummary := tuiGuardMap(m["transcript_summary"])
		rows = append(rows, tuiGuardRow{
			Artifact: artifactName,
			Kind:     "stopfailure-session",
			Status:   "WARN",
			Detail: strings.Join(nonEmptyTUI([]string{
				"session=" + tuiGuardString(m, "session_id"),
				fmt.Sprintf("total=%d", total),
				fmt.Sprintf("consecutive=%d", tuiGuardInt(m, "consecutive")),
				"mtime=" + tuiGuardString(m, "mtime"),
				"transcript=" + tuiGuardString(transcript, "status"),
				"account=" + tuiGuardString(transcript, "account"),
				"project=" + tuiGuardString(transcript, "project"),
				"evidence=" + tuiGuardCompactJSON(transcriptSummary["evidence_tags"]),
			}), "  "),
			Count: total,
		})
		if len(rows) >= 5 {
			break
		}
	}
	return rows
}

func tuiGuardVendorRows(artifact tuiGuardArtifact) []tuiGuardRow {
	raw := artifact.Raw
	artifactName := tuiGuardArtifactName(artifact.Path)
	rows := []tuiGuardRow{}
	if m := tuiGuardMap(raw["dangerous_attempt"]); m != nil {
		rows = append(rows, tuiGuardDecisionRow(artifactName, "dangerous-attempt", tuiGuardString(raw, "status"), m))
	}
	if m := tuiGuardMap(raw["useful_continuation"]); m != nil {
		rows = append(rows, tuiGuardDecisionRow(artifactName, "useful-continuation", tuiGuardString(raw, "status"), m))
	}
	return rows
}

func tuiGuardCombinedRows(artifact tuiGuardArtifact) []tuiGuardRow {
	raw := artifact.Raw
	floor := tuiGuardMap(raw["floor"])
	if floor == nil {
		return nil
	}
	artifactName := tuiGuardArtifactName(artifact.Path)
	status := tuiGuardString(raw, "verdict")
	denyReason := ""
	if strings.Contains(tuiGuardString(floor, "deny_response_excerpt"), "POLICY_BLOCK") {
		denyReason = "POLICY_BLOCK"
	}
	return []tuiGuardRow{
		{
			Artifact: artifactName,
			Kind:     "floor-deny",
			Tool:     tuiGuardString(floor, "deny_call"),
			Verdict:  "DENY",
			Reason:   denyReason,
			Status:   status,
			Detail:   fmt.Sprintf("pass=%v", tuiGuardBool(floor, "pass")),
			Count:    1,
		},
		{
			Artifact: artifactName,
			Kind:     "floor-allow",
			Tool:     tuiGuardString(floor, "allow_call"),
			Verdict:  "ALLOW",
			Status:   status,
			Detail:   fmt.Sprintf("pass=%v", tuiGuardBool(floor, "pass")),
			Count:    1,
		},
	}
}

func tuiGuardCheckRows(artifact tuiGuardArtifact, checks []any) []tuiGuardRow {
	rows := []tuiGuardRow{}
	for _, check := range checks {
		m := tuiGuardMap(check)
		if m == nil {
			continue
		}
		rows = append(rows, tuiGuardRow{
			Artifact: tuiGuardArtifactName(artifact.Path),
			Kind:     "check",
			Status:   tuiGuardString(m, "status"),
			Detail:   strings.TrimSpace(tuiGuardString(m, "name") + "  " + tuiGuardString(m, "detail")),
		})
	}
	return rows
}

func tuiGuardGenericRow(artifact tuiGuardArtifact) tuiGuardRow {
	raw := artifact.Raw
	verdict := firstNonEmptyTUI(tuiGuardString(raw, "verdict"), tuiGuardString(raw, "kind"))
	status := tuiGuardString(raw, "status")
	if verdict == "" && status == "" {
		return tuiGuardRow{}
	}
	return tuiGuardRow{
		Artifact: tuiGuardArtifactName(artifact.Path),
		Kind:     "artifact",
		Tool:     tuiGuardString(raw, "tool"),
		Verdict:  verdict,
		Reason:   tuiGuardString(raw, "reason"),
		By:       tuiGuardString(raw, "by"),
		Status:   status,
		Detail:   tuiGuardString(raw, "finding"),
		Count:    1,
	}
}

func tuiGuardDecisionRow(artifact, kind, status string, m map[string]any) tuiGuardRow {
	tool := tuiGuardString(m, "tool")
	if args := tuiGuardMap(m["arguments"]); args != nil {
		tool = firstNonEmptyTUI(tuiGuardString(args, "tool"), tool)
	}
	verdict, reason, by := "", "", ""
	if audit := tuiGuardMap(m["fak_audit"]); audit != nil {
		verdict = tuiGuardString(audit, "verdict")
		reason = tuiGuardString(audit, "reason")
		by = tuiGuardString(audit, "by")
	}
	if verdict == "" {
		if v := tuiGuardMap(m["fak_verdict"]); v != nil {
			verdict = firstNonEmptyTUI(tuiGuardString(v, "kind"), tuiGuardString(v, "verdict"))
			reason = tuiGuardString(v, "reason")
			by = tuiGuardString(v, "by")
		}
	}
	detail := strings.Join(nonEmptyTUI([]string{
		tuiGuardString(m, "mcp_tool"),
		tuiGuardString(m, "trace_id"),
		tuiGuardString(m, "assistant_visible_refusal"),
	}), "  ")
	return tuiGuardRow{
		Artifact: artifact,
		Kind:     kind,
		Tool:     tool,
		Verdict:  verdict,
		Reason:   reason,
		By:       by,
		Status:   status,
		Detail:   detail,
		Count:    1,
	}
}

func scoreTUIGuardRow(row tuiGuardRow) ([]string, int) {
	tags := []string{}
	score := 0
	status := strings.ToUpper(row.Status)
	verdict := strings.ToUpper(row.Verdict)
	reason := strings.ToUpper(row.Reason)
	switch verdict {
	case "DENY":
		tags = append(tags, "deny")
		score += 45
	case "ALLOW":
		tags = append(tags, "allow")
		score += 5
	case "TRANSFORM":
		tags = append(tags, "transform")
		score += 25
	case "QUARANTINE":
		tags = append(tags, "quarantine")
		score += 70
	}
	switch reason {
	case "POLICY_BLOCK":
		tags = append(tags, "policy-block")
		score += 25
	case "DEFAULT_DENY":
		tags = append(tags, "default-deny")
		score += 15
	case "TRUST_VIOLATION":
		tags = append(tags, "trust-violation")
		score += 30
	}
	if row.Kind == "sample" {
		tags = append(tags, "sample")
		score += 15
	}
	if row.Kind == "settlement-candidate" {
		tags = append(tags, "settlement")
		score += 30
	}
	if row.Kind == "default-blocker" {
		tags = append(tags, "blocker")
		score += 20
		switch {
		case status == "ACTIVE":
			tags = append(tags, "active")
			score += 160
		case strings.Contains(status, "ACTIVE"):
			tags = append(tags, "active")
			score += 60
		case strings.Contains(status, "STALE"):
			tags = append(tags, "stale")
			score += 40
		case strings.Contains(status, "HISTORICAL"):
			tags = append(tags, "historical")
			score += 15
		case strings.Contains(status, "EXTERNAL"):
			tags = append(tags, "external")
			score += 15
		}
		if strings.Contains(status, "DEBT") {
			tags = append(tags, "debt")
			score += 10
		}
	}
	if strings.Contains(status, "DENIED_EXPECTED") {
		tags = append(tags, "expected-deny")
		score += 5
	}
	if strings.Contains(status, "WARN") {
		tags = append(tags, "warn")
		score += 35
	}
	if status == "FAIL" || strings.Contains(status, "UNEXPECTED") || strings.Contains(status, "ERROR") || strings.Contains(status, "BLOCKED") {
		tags = append(tags, "unexpected")
		score += 100
	}
	if row.Count > 1 {
		score += minTUI(row.Count, 50)
	}
	if len(tags) == 0 {
		tags = append(tags, "status")
	}
	return tags, score
}

// tuiGuardStatus folds the guard-row counts into the headline status: any FAIL/Unexpected
// row -> "FAIL", else any WARN -> "WARN", else "PASS".
func tuiGuardStatus(counts tuiGuardCounts) string {
	switch {
	case counts.Fail > 0 || counts.Unexpected > 0:
		return "FAIL"
	case counts.Warn > 0:
		return "WARN"
	}
	return "PASS"
}

func countTUIGuard(rows []tuiGuardRow, sources []tuiGuardSource) tuiGuardCounts {
	c := tuiGuardCounts{Artifacts: len(sources), Rows: len(rows)}
	for _, source := range sources {
		switch tuiGuardStatusClass(source.Status) {
		case "pass":
			c.Pass++
		case "warn":
			c.Warn++
		case "fail":
			c.Fail++
		}
	}
	for _, row := range rows {
		n := row.Count
		if n <= 0 {
			if hasStringTUI(row.Tags, "unexpected") {
				c.Unexpected++
			}
			continue
		}
		switch strings.ToUpper(row.Verdict) {
		case "ALLOW":
			c.Allow += n
		case "DENY":
			c.Deny += n
		case "TRANSFORM":
			c.Transform += n
		case "QUARANTINE":
			c.Quarantine += n
		}
		switch strings.ToUpper(row.Reason) {
		case "POLICY_BLOCK":
			c.PolicyBlock += n
		case "DEFAULT_DENY":
			c.DefaultDeny += n
		}
		if hasStringTUI(row.Tags, "expected-deny") {
			c.Expected += n
		}
		if hasStringTUI(row.Tags, "unexpected") {
			c.Unexpected += n
		}
	}
	return c
}

func tuiGuardStatusClass(status string) string {
	status = strings.ToUpper(strings.TrimSpace(status))
	switch {
	case status == "":
		return ""
	case status == "PASS" || status == "OK" || status == "DENIED_EXPECTED":
		return "pass"
	case strings.Contains(status, "WARN"):
		return "warn"
	case strings.Contains(status, "FAIL") || strings.Contains(status, "ERROR") || strings.Contains(status, "BLOCKED") || strings.Contains(status, "UNEXPECTED"):
		return "fail"
	default:
		return "pass"
	}
}

func tuiGuardActions(counts tuiGuardCounts) []string {
	switch {
	case counts.Unexpected > 0 || counts.Fail > 0:
		return []string{"inspect failing or unexpected guard artifacts before treating the proof packet as current"}
	case counts.Warn > 0:
		return []string{"inspect WARN guard artifacts before treating fak-by-default actionability as clear"}
	case counts.Deny == 0 && counts.Quarantine == 0:
		return []string{"add a recent guard proof with at least one denial or quarantine"}
	case counts.PolicyBlock == 0:
		return []string{"capture a POLICY_BLOCK proof for destructive tool refusals"}
	default:
		return []string{"keep feeding recent guard artifacts into this pane; the denial surface is visible"}
	}
}

func renderTUIGuard(report tuiGuardReport, width int) string {
	var b strings.Builder
	fmt.Fprintf(&b, "fak console guard  at=%s  source=%s\n", report.At, report.Source)
	fmt.Fprintf(&b, "status=%s  artifacts=%d  pass=%d  warn=%d  fail=%d  rows=%d  allow=%d  deny=%d  quarantine=%d  policy_block=%d  default_deny=%d  expected=%d  unexpected=%d\n",
		report.Status, report.Counts.Artifacts, report.Counts.Pass, report.Counts.Warn,
		report.Counts.Fail, report.Counts.Rows, report.Counts.Allow, report.Counts.Deny,
		report.Counts.Quarantine, report.Counts.PolicyBlock, report.Counts.DefaultDeny,
		report.Counts.Expected, report.Counts.Unexpected)
	if len(report.Actions) > 0 {
		fmt.Fprintf(&b, "next: %s\n", trimTUI(report.Actions[0], maxTUI(20, width-6)))
	}
	if len(report.Rows) == 0 {
		fmt.Fprintln(&b, "\nno guard rows")
		return b.String()
	}
	fmt.Fprintln(&b, "\nRows")
	fmt.Fprintln(&b, "attention artifact                 kind                 tool             verdict reason         count tags")
	for _, row := range report.Rows {
		count := "-"
		if row.Count > 0 {
			count = strconv.Itoa(row.Count)
		}
		tags := displayTUITags(row.Tags, 4)
		if row.Detail != "" {
			tags += "  " + row.Detail
		}
		fmt.Fprintf(&b, "%9d %s %s %s %s %s %-5s %s\n",
			row.Attention,
			padRightTUI(trimTUI(row.Artifact, 24), 24),
			padRightTUI(trimTUI(row.Kind, 20), 20),
			padRightTUI(trimTUI(row.Tool, 16), 16),
			padRightTUI(trimTUI(row.Verdict, 7), 7),
			padRightTUI(trimTUI(row.Reason, 14), 14),
			count, trimTUI(tags, maxTUI(12, width-91)))
	}
	return b.String()
}

func tuiGuardSourceLabel(artifacts []tuiGuardArtifact) string {
	if len(artifacts) == 1 {
		return artifacts[0].Path
	}
	return fmt.Sprintf("%d artifacts", len(artifacts))
}

func tuiGuardArtifactName(path string) string {
	name := filepath.Base(path)
	if name == "." || name == string(filepath.Separator) {
		return path
	}
	return name
}

func tuiGuardMap(v any) map[string]any {
	m, ok := v.(map[string]any)
	if !ok {
		return nil
	}
	return m
}

func tuiGuardString(m map[string]any, key string) string {
	if m == nil {
		return ""
	}
	v := m[key]
	if v == nil {
		return ""
	}
	switch s := v.(type) {
	case string:
		return s
	case bool:
		return strconv.FormatBool(s)
	case float64:
		if s == float64(int64(s)) {
			return strconv.FormatInt(int64(s), 10)
		}
		return strconv.FormatFloat(s, 'f', -1, 64)
	default:
		return fmt.Sprintf("%v", s)
	}
}

func tuiGuardBool(m map[string]any, key string) bool {
	if m == nil {
		return false
	}
	b, ok := m[key].(bool)
	return ok && b
}

func tuiGuardInt(m map[string]any, key string) int {
	if m == nil {
		return 0
	}
	switch n := m[key].(type) {
	case float64:
		return int(n)
	case int:
		return n
	case string:
		i, _ := strconv.Atoi(strings.TrimSpace(n))
		return i
	default:
		return 0
	}
}

func tuiGuardIntMap(v any) map[string]int {
	out := map[string]int{}
	m, ok := v.(map[string]any)
	if !ok {
		return out
	}
	for k, raw := range m {
		out[k] = tuiGuardInt(map[string]any{"v": raw}, "v")
	}
	return out
}

func tuiGuardCompactJSON(v any) string {
	if v == nil {
		return "null"
	}
	b, err := json.Marshal(v)
	if err != nil || len(b) == 0 {
		return fmt.Sprintf("%v", v)
	}
	return string(b)
}

func residualsContain(v any, want string) bool {
	switch xs := v.(type) {
	case []any:
		for _, x := range xs {
			if strings.EqualFold(strings.TrimSpace(fmt.Sprint(x)), want) {
				return true
			}
		}
	case []string:
		for _, x := range xs {
			if strings.EqualFold(strings.TrimSpace(x), want) {
				return true
			}
		}
	case string:
		for _, x := range strings.Split(xs, ",") {
			if strings.EqualFold(strings.TrimSpace(x), want) {
				return true
			}
		}
	}
	return false
}
