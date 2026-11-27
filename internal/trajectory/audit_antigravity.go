package trajectory

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
)

var antigravityExitCodeRegex = regexp.MustCompile(`(?i)command exited with code ([0-9]+)`)

func parseAntigravityAuditRecord(record map[string]any, line int, rel string, state *auditParseState, refusals *[]AuditRefusalRow) {
	recordType, _ := record["type"].(string)
	switch recordType {
	case "PLANNER_RESPONSE":
		if model, ok := record["model"].(string); ok && model != "" {
			state.models[model] = struct{}{}
		}
		thinking, _ := record["thinking"].(string)
		content, _ := record["content"].(string)
		if strings.TrimSpace(content) != "" {
			state.mutationHypothesis = auditNormalizedHead(content, 200)
		}
		observeAuditAssistantTurn(thinking, content, state)
		parseAntigravityAuditUsage(record, line, rel, state, refusals)
		parseAntigravityToolCalls(record["tool_calls"], state)
	case "CHECKPOINT":
		state.repeatedFailuresAccumulated += auditRepeatedFailureCount(state.failureCounts)
		state.failureCounts = map[string]int{}
		event := AuditCompactionEvent{
			Line:     line,
			Unit:     "antigravity_checkpoint_utf8_bytes",
			Origin:   "antigravity_checkpoint",
			Coverage: "summary_only",
		}
		if summary, ok := record["content"].(string); ok {
			n := int64(len(summary))
			event.SummaryBytes = &n
		}
		state.row.CompactionEvents = append(state.row.CompactionEvents, event)
		if _, hasIn := record["input_tokens"]; hasIn {
			parseAntigravityAuditUsage(record, line, rel, state, refusals)
		}
	case "GENERIC":
		var call auditToolCall
		if len(state.antigravityPendingCalls) > 0 {
			call = state.antigravityPendingCalls[0]
			state.antigravityPendingCalls = state.antigravityPendingCalls[1:]
		}
		content, _ := record["content"].(string)
		errText, _ := record["error"].(string)
		status, _ := record["status"].(string)
		isErr := strings.TrimSpace(errText) != "" ||
			strings.EqualFold(status, "ERROR") ||
			auditPiNonzeroExit.MatchString(content)
		if isErr {
			if auditAntigravityBenignNegative(call, content) {
				state.row.ToolErrorsBenignNegative++
				return
			}
			failurePayload := content
			if strings.TrimSpace(failurePayload) == "" {
				failurePayload = errText
			}
			auditRecordToolFailure(state, call, failurePayload, line)
			return
		}
		if call.target == "" {
			auditAppendMutationEvent(state, "", QwenMutationWitness)
		}
	case "USER_INPUT", "SYSTEM_MESSAGE":
		// Non-usage envelope; user prompt matching is handled by auditUserText.
	}
}

func parseAntigravityAuditUsage(record map[string]any, line int, rel string, state *auditParseState, refusals *[]AuditRefusalRow) {
	state.usageSeen++
	input, err := auditRequiredInt(record, "input_tokens")
	if err != nil {
		*refusals = append(*refusals, newAuditRefusal(AuditSourceAntigravity, rel, line, "antigravity_usage_input_invalid", err.Error()))
		return
	}
	output, err := auditRequiredInt(record, "output_tokens")
	if err != nil {
		*refusals = append(*refusals, newAuditRefusal(AuditSourceAntigravity, rel, line, "antigravity_usage_output_invalid", err.Error()))
		return
	}
	cacheRead, err := auditOptionalInt(record, "cache_read_tokens")
	if err != nil {
		*refusals = append(*refusals, newAuditRefusal(AuditSourceAntigravity, rel, line, "antigravity_usage_cache_read_invalid", err.Error()))
		return
	}
	tokens := AuditTokens{
		InputTokens:     input,
		OutputTokens:    output,
		CacheReadTokens: cacheRead,
	}
	state.usageExact++
	state.row.Tokens.add(tokens)
	state.row.UsageRecords++
}

func parseAntigravityToolCalls(rawCalls any, state *auditParseState) {
	calls, _ := rawCalls.([]any)
	for _, raw := range calls {
		tc, _ := raw.(map[string]any)
		if tc == nil {
			continue
		}
		name, _ := tc["name"].(string)
		args := auditCanonicalArgs(tc["args"])
		target := auditMutationTarget(name, tc["args"])
		call := auditToolCall{name: name, args: args, target: target}
		state.antigravityPendingCalls = append(state.antigravityPendingCalls, call)
		state.row.ToolCalls++
		if target != "" {
			state.mutationCounts[target]++
			auditAppendMutationEvent(state, target, QwenMutationWrite)
		}
	}
}

func auditAntigravityBenignNegative(call auditToolCall, content string) bool {
	if call.name != "run_command" {
		return false
	}
	var decoded map[string]any
	if json.Unmarshal([]byte(call.args), &decoded) != nil {
		return false
	}
	cmdLine, _ := decoded["CommandLine"].(string)
	lower := strings.ToLower(strings.TrimSpace(cmdLine))
	if strings.Contains(lower, "select-string") || strings.HasPrefix(lower, "rg ") || strings.HasPrefix(lower, "grep ") {
		if m := antigravityExitCodeRegex.FindStringSubmatch(content); len(m) == 2 && m[1] == "1" {
			return true
		}
	}
	if strings.Contains(lower, "git ") && strings.Contains(lower, "diff") && strings.Contains(lower, "--quiet") {
		if m := antigravityExitCodeRegex.FindStringSubmatch(content); len(m) == 2 && m[1] == "1" {
			return true
		}
	}
	return false
}

// Antigravity distribution & tool-result helpers.

func classifyAntigravityDistributionEvents(typ string, root map[string]json.RawMessage) []auditClassifiedEvent {
	switch typ {
	case "USER_INPUT":
		return []auditClassifiedEvent{
			auditVisibleEvent("user_message", "", "", joinRaw(root["content"]), "USER_INPUT", "inferred_model_visible", "known_user_input"),
		}
	case "PLANNER_RESPONSE":
		var events []auditClassifiedEvent
		if thinking := joinRaw(root["thinking"]); len(thinking) > 0 {
			events = append(events, auditVisibleEvent("reasoning", "", "", thinking, "PLANNER_RESPONSE/thinking", "inferred_model_visible", "known_thinking"))
		}
		if content := joinRaw(root["content"]); len(content) > 0 {
			events = append(events, auditVisibleEvent("assistant_message", "", "", content, "PLANNER_RESPONSE/content", "inferred_model_visible", "known_assistant_content"))
		}
		var toolCalls []map[string]json.RawMessage
		if json.Unmarshal(root["tool_calls"], &toolCalls) == nil {
			for i, tc := range toolCalls {
				name := rawString(tc["name"])
				if name == "" {
					name = "unknown"
				}
				callBytes, _ := json.Marshal(tc)
				id := fmt.Sprintf("ag:%s:%d", rawString(root["step_index"]), i)
				events = append(events, auditVisibleEvent("tool_call", name, id, callBytes, auditToolSubtype("tool_use", name), "inferred_model_visible", "known_tool_call"))
			}
		}
		if len(events) == 0 {
			events = append(events, auditStorageEvent("PLANNER_RESPONSE/empty", "explicit_storage_only", "empty_planner_response"))
		}
		return events
	case "GENERIC":
		return []auditClassifiedEvent{
			auditVisibleEvent("tool_result", "", "", joinRaw(root["content"]), "GENERIC/tool_result", "inferred_model_visible", "known_tool_result"),
		}
	case "SYSTEM_MESSAGE", "CHECKPOINT":
		return []auditClassifiedEvent{
			auditStorageEvent(typ, "explicit_storage_only", "known_antigravity_envelope"),
		}
	default:
		return []auditClassifiedEvent{
			auditStorageEvent(firstNonempty(typ, "unknown"), "unresolved", "unknown_antigravity_subtype"),
		}
	}
}

func (d *auditDistribution) observeAntigravityToolResult(typ string, root map[string]json.RawMessage) {
	switch typ {
	case "PLANNER_RESPONSE":
		var toolCalls []map[string]json.RawMessage
		if json.Unmarshal(root["tool_calls"], &toolCalls) != nil {
			return
		}
		stepIdx := rawString(root["step_index"])
		for i, tc := range toolCalls {
			name := rawString(tc["name"])
			if name == "" {
				name = "unknown"
			}
			id := fmt.Sprintf("ag:%s:%d", stepIdx, i)
			d.antigravityPending = append(d.antigravityPending, id)
			d.observeResultCall(id, name, auditToolSubtype("tool_use", name))
		}
	case "GENERIC":
		id := ""
		if len(d.antigravityPending) > 0 {
			id = d.antigravityPending[0]
			d.antigravityPending = d.antigravityPending[1:]
		}
		content := rawString(root["content"])
		errText := rawString(root["error"])
		status := rawString(root["status"])
		explicitErr := strings.TrimSpace(errText) != "" || strings.EqualFold(status, "ERROR") || auditPiNonzeroExit.MatchString(content)
		res := auditToolResult{
			bytes:     int64(len(joinRaw(root["content"]))),
			status:    "success",
			truncated: strings.Contains(content, "output was truncated due to being too long"),
		}
		if explicitErr {
			res.status = "error"
		}
		if m := antigravityExitCodeRegex.FindStringSubmatch(content); len(m) == 2 {
			var code int64
			if _, err := fmt.Sscanf(m[1], "%d", &code); err == nil {
				res.exitKnown = true
				res.exitCode = code
			}
		}
		d.observeResult(id, res)
	}
}
