package trajectory

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

var auditNonzeroExit = regexp.MustCompile(`(?i)(?:process exited with code|exit (?:code|status)[:= ]+)\s*[1-9][0-9]*`)

const auditMaxLineBytes = 32 * 1024 * 1024

type auditToolCall struct {
	name   string
	args   string
	target string
}

type auditParseState struct {
	row                   AuditTranscriptRow
	models                map[string]struct{}
	calls                 map[string]auditToolCall
	seenCalls             map[string]struct{}
	failureCounts         map[string]int
	mutationCounts        map[string]int
	mutationEvents        []QwenMutationEvent
	mutationHypothesis    string
	mutationAccountedFor  int64
	hookDurations         []int64
	claudeUsageByID       map[string]AuditTokens
	codexPrimaryMetaSeen  bool
	codexRawTotal         *auditCodexRawTokens
	codexCompleted        AuditTokens
	codexVersion          string
	codexModelProvider    string
	codexResetArmed       bool
	codexCacheSamples     int
	codexCacheMin         int64
	codexCacheMax         int64
	usageSeen             int
	usageExact            int
	usageDuplicates       int
	toolErrorEvents       []QwenToolErrorEvent
	toolErrorAttributions []qwenToolErrorAttribution
	distribution          auditDistribution
	buildIdentity         AuditBuildIdentity
	buildIdentities       map[AuditBuildIdentity]struct{}
	schemaShapes          map[string]auditShapeSet
}

type auditCodexRawTokens struct {
	Input      int64
	Output     int64
	CacheRead  int64
	CacheWrite int64
}

func parseAuditFile(source, path, rel string, denominator *AuditDenominatorRow) (AuditTranscriptRow, []AuditRefusalRow, []int64, []QwenToolErrorEvent, error) {
	file, err := os.Open(path)
	if err != nil {
		return AuditTranscriptRow{}, nil, nil, nil, fmt.Errorf("trajectory audit: open transcript: %w", err)
	}
	defer file.Close()

	state := auditParseState{
		distribution: newAuditDistribution(),
		row: AuditTranscriptRow{
			Schema: AuditSchema, Kind: "session", Source: source,
			TranscriptID: strings.TrimSuffix(filepath.Base(path), filepath.Ext(path)), SourcePath: rel,
		},
		models:          map[string]struct{}{},
		calls:           map[string]auditToolCall{},
		seenCalls:       map[string]struct{}{},
		failureCounts:   map[string]int{},
		mutationCounts:  map[string]int{},
		claudeUsageByID: map[string]AuditTokens{},
		buildIdentity:   AuditBuildIdentity{Harness: source},
		buildIdentities: map[AuditBuildIdentity]struct{}{},
		schemaShapes:    map[string]auditShapeSet{},
	}
	var refusals []AuditRefusalRow
	fragmentHasher := sha256.New()
	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 64*1024), auditMaxLineBytes)
	lineNumber := 0
	for scanner.Scan() {
		lineNumber++
		line := bytes.TrimSpace(scanner.Bytes())
		if len(line) == 0 {
			continue
		}
		_, _ = fragmentHasher.Write(line)
		_, _ = fragmentHasher.Write([]byte{'\n'})
		denominator.Records++
		state.distribution.observe(source, line)
		var record map[string]any
		decoder := json.NewDecoder(bytes.NewReader(line))
		decoder.UseNumber()
		if err := decoder.Decode(&record); err != nil {
			refusals = append(refusals, newAuditRefusal(source, rel, lineNumber, "malformed_json", err.Error()))
			denominator.RefusedRecords++
			continue
		}
		observeAuditBuildIdentity(source, record, &state)
		observeAuditEventShape(record, &state)
		recordType, _ := record["type"].(string)
		if recordType == "" {
			recordType = "<missing>"
		}
		denominator.RecordTypes[auditRecordType(record)]++
		before := len(refusals)
		switch source {
		case AuditSourceClaude:
			parseClaudeAuditRecord(record, lineNumber, rel, &state, &refusals)
		case AuditSourceCodex:
			parseCodexAuditRecord(record, lineNumber, rel, &state, &refusals)
		}
		if len(refusals) > before {
			denominator.RefusedRecords += len(refusals) - before
		}
	}
	if err := scanner.Err(); err != nil {
		if !errors.Is(err, bufio.ErrTooLong) {
			return AuditTranscriptRow{}, nil, nil, nil, fmt.Errorf("trajectory audit: scan transcript: %w", err)
		}
		lineNumber++
		denominator.Records++
		denominator.RefusedRecords++
		refusals = append(refusals, newAuditRefusal(source, rel, lineNumber, "line_too_large", fmt.Sprintf("line exceeds %d-byte limit", auditMaxLineBytes)))
	}
	denominator.UsageRecordsSeen += state.usageSeen
	denominator.UsageRecordsExact += state.usageExact
	denominator.DuplicateUsageRecords += state.usageDuplicates
	denominator.UsageRecordsApplied += state.row.UsageRecords

	state.row.usageByID = make(map[string]AuditTokens, len(state.claudeUsageByID))
	for id, usage := range state.claudeUsageByID {
		state.row.usageByID[id] = usage
	}

	state.row.Models = make([]string, 0, len(state.models))
	for model := range state.models {
		state.row.Models = append(state.row.Models, model)
	}
	sort.Strings(state.row.Models)
	state.row.BuildIdentities = make([]AuditBuildIdentity, 0, len(state.buildIdentities))
	for identity := range state.buildIdentities {
		state.row.BuildIdentities = append(state.row.BuildIdentities, identity)
	}
	state.row.BuildIdentities = appendAuditBuilds(nil, state.row.BuildIdentities...)
	state.row.schemaShapes = state.schemaShapes
	state.row.failureCounts = cloneAuditFailureCounts(state.failureCounts)
	state.row.RepeatedFailures = auditRepeatedFailureCount(state.failureCounts)
	state.row.MutationChurnEvents = DetectQwenMutationChurn(state.mutationEvents)
	for _, churn := range state.row.MutationChurnEvents {
		state.row.MutationChurn += churn.Count - 1
	}
	applyQwenToolErrorAttribution(state.toolErrorEvents, state.toolErrorAttributions, state.mutationCounts)
	state.row.HookP95MS = auditPercentile(state.hookDurations, 95)
	state.row.Distribution = state.distribution.distributionRows()
	state.row.ToolDistribution = toolDistributionRows(state.distribution.tools)
	state.row.ToolResults = state.distribution.toolResultRows()
	state.row.StorageDistribution = state.distribution.storageRows()
	state.row.UnknownExemplars = state.distribution.exemplars.snapshot()
	state.row.fragmentDigest = hex.EncodeToString(fragmentHasher.Sum(nil))
	if source == AuditSourceCodex {
		state.row.CodexCache = &AuditCodexCacheObservation{
			TranscriptProducer:               AuditSourceCodex,
			ModelProvider:                    state.codexModelProvider,
			ModelProviderSource:              "session_meta.model_provider",
			LastTokenUsageCachedInputSamples: state.codexCacheSamples,
			PhysicalProviderCacheResidency:   "not_inferable_from_cached_input_tokens",
			FakOwnedCacheCoverage:            "not_observed_by_codex_token_count",
		}
		if state.codexCacheSamples > 0 {
			minimum, maximum := state.codexCacheMin, state.codexCacheMax
			state.row.CodexCache.LastTokenUsageCachedInputMin = &minimum
			state.row.CodexCache.LastTokenUsageCachedInputMax = &maximum
		}
	}
	if source == AuditSourceCodex && state.codexRawTotal != nil {
		state.row.UsageRecords++
		denominator.UsageRecordsApplied++
	}
	return state.row, refusals, append([]int64(nil), state.hookDurations...), append([]QwenToolErrorEvent(nil), state.toolErrorEvents...), nil
}

func parseClaudeAuditRecord(record map[string]any, line int, rel string, state *auditParseState, refusals *[]AuditRefusalRow) {
	if id, ok := record["sessionId"].(string); ok && strings.TrimSpace(id) != "" {
		state.row.TranscriptID = id
	}
	recordType, _ := record["type"].(string)
	if _, supported := auditClaudeRowSubtypes[recordType]; !supported {
		return
	}
	if recordType == "attachment" {
		parseClaudeAuditHook(record, state)
		return
	}
	message, _ := record["message"].(map[string]any)
	if recordType == "assistant" {
		if model, ok := message["model"].(string); ok && model != "" {
			state.models[model] = struct{}{}
		}
		if usage, present := message["usage"]; present {
			parseClaudeAuditUsage(usage, message, line, rel, state, refusals)
		}
		state.mutationHypothesis = auditMutationHypothesis(message["content"])
		parseClaudeToolCalls(message["content"], line, state)
		return
	}
	if recordType == "user" {
		parseClaudeToolResults(message["content"], line, state)
	}
}

func parseClaudeAuditUsage(value any, message map[string]any, line int, rel string, state *auditParseState, refusals *[]AuditRefusalRow) {
	state.usageSeen++
	usage, ok := value.(map[string]any)
	if !ok {
		*refusals = append(*refusals, newAuditRefusal(AuditSourceClaude, rel, line, "claude_usage_not_object", "message.usage must be an object"))
		return
	}
	input, err := auditRequiredInt(usage, "input_tokens")
	if err != nil {
		*refusals = append(*refusals, newAuditRefusal(AuditSourceClaude, rel, line, "claude_input_tokens", err.Error()))
		return
	}
	output, err := auditRequiredInt(usage, "output_tokens")
	if err != nil {
		*refusals = append(*refusals, newAuditRefusal(AuditSourceClaude, rel, line, "claude_output_tokens", err.Error()))
		return
	}
	cacheRead, err := auditOptionalInt(usage, "cache_read_input_tokens")
	if err != nil {
		*refusals = append(*refusals, newAuditRefusal(AuditSourceClaude, rel, line, "claude_read_bucket", err.Error()))
		return
	}
	cacheCreate, err := auditOptionalInt(usage, "cache_creation_input_tokens")
	if err != nil {
		*refusals = append(*refusals, newAuditRefusal(AuditSourceClaude, rel, line, "claude_create_bucket", err.Error()))
		return
	}
	tokens := AuditTokens{InputTokens: input, OutputTokens: output, CacheReadTokens: cacheRead, CacheCreateTokens: cacheCreate}
	mid, _ := message["id"].(string)
	if mid == "" {
		mid = "line:" + strconv.Itoa(line)
	}
	if previous, seen := state.claudeUsageByID[mid]; seen {
		state.usageDuplicates++
		if previous != tokens {
			*refusals = append(*refusals, newAuditRefusal(AuditSourceClaude, rel, line, "claude_duplicate_usage_mismatch", "duplicate message id carries different usage"))
			return
		}
		state.usageExact++
		return
	}
	state.claudeUsageByID[mid] = tokens
	state.usageExact++
	state.row.Tokens.add(tokens)
	state.row.UsageRecords++
}

func parseClaudeToolCalls(content any, line int, state *auditParseState) {
	blocks, _ := content.([]any)
	for i, raw := range blocks {
		block, _ := raw.(map[string]any)
		if block["type"] != "tool_use" {
			continue
		}
		id, _ := block["id"].(string)
		if id == "" {
			id = fmt.Sprintf("line:%d:block:%d", line, i)
		}
		if _, seen := state.seenCalls[id]; seen {
			continue
		}
		state.seenCalls[id] = struct{}{}
		name, _ := block["name"].(string)
		args := block["input"]
		call := auditToolCall{name: name, args: auditCanonicalJSON(args), target: auditMutationTarget(name, args)}
		state.calls[id] = call
		state.row.ToolCalls++
		if call.target != "" {
			state.mutationCounts[call.target]++
			auditAppendMutationEvent(state, call.target, QwenMutationWrite)
		}
	}
}

func parseClaudeToolResults(content any, line int, state *auditParseState) {
	blocks, _ := content.([]any)
	for _, raw := range blocks {
		block, _ := raw.(map[string]any)
		if block["type"] != "tool_result" {
			continue
		}
		isError, _ := block["is_error"].(bool)
		id, _ := block["tool_use_id"].(string)
		call := state.calls[id]
		if auditExpectedWaitTimeout(call, block["content"]) {
			state.row.ExpectedWaitTimeouts++
			continue
		}
		if !isError {
			if call.target == "" {
				auditAppendMutationEvent(state, "", QwenMutationWitness)
			}
			continue
		}
		auditRecordToolFailure(state, call, block["content"], line)
	}
}

func parseClaudeAuditHook(record map[string]any, state *auditParseState) {
	attachment, _ := record["attachment"].(map[string]any)
	kind, _ := attachment["type"].(string)
	if !strings.HasPrefix(kind, "hook_") {
		return
	}
	duration, err := auditAnyInt(attachment["durationMs"])
	if err == nil && duration >= 0 {
		state.hookDurations = append(state.hookDurations, duration)
	}
}

func parseCodexAuditRecord(record map[string]any, line int, rel string, state *auditParseState, refusals *[]AuditRefusalRow) {
	recordType, _ := record["type"].(string)
	if _, supported := auditCodexRowSubtypes[recordType]; !supported {
		return
	}
	payload, _ := record["payload"].(map[string]any)
	switch recordType {
	case "session_meta":
		// Subagent rollouts embed their parent transcript history after the
		// file-local metadata row. Only the first session_meta names this
		// rollout; later rows are copied provenance and must not replace its
		// identity or parser version.
		if state.codexPrimaryMetaSeen {
			return
		}
		state.codexPrimaryMetaSeen = true
		if version, ok := payload["cli_version"].(string); ok {
			state.codexVersion = strings.TrimSpace(version)
		}
		if provider, ok := payload["model_provider"].(string); ok {
			state.codexModelProvider = strings.TrimSpace(provider)
		}
		for _, key := range []string{"id", "session_id"} {
			if id, ok := payload[key].(string); ok && strings.TrimSpace(id) != "" {
				state.row.TranscriptID = id
				break
			}
		}
	case "turn_context":
		if model, ok := payload["model"].(string); ok && model != "" {
			state.models[model] = struct{}{}
		}
	case "event_msg":
		switch payload["type"] {
		case "task_started":
			turnID, _ := payload["turn_id"].(string)
			state.codexResetArmed = state.codexRawTotal != nil && state.codexVersion != "" && strings.TrimSpace(turnID) != ""
		case "token_count":
			parseCodexAuditUsage(payload, line, rel, state, refusals)
		}
	case "response_item":
		parseCodexResponseItem(payload, line, rel, state, refusals)
	}
}

func parseCodexAuditUsage(payload map[string]any, line int, rel string, state *auditParseState, refusals *[]AuditRefusalRow) {
	info, ok := payload["info"].(map[string]any)
	if !ok {
		if payload["info"] == nil {
			return
		}
		*refusals = append(*refusals, newAuditRefusal(AuditSourceCodex, rel, line, "codex_token_info_not_object", "token_count.info must be an object or null"))
		return
	}
	state.usageSeen++
	observeCodexLastTokenUsage(info, line, rel, state, refusals)
	total, ok := info["total_token_usage"].(map[string]any)
	if !ok {
		*refusals = append(*refusals, newAuditRefusal(AuditSourceCodex, rel, line, "codex_total_usage_missing", "cumulative total_token_usage is required; last_token_usage is not summed"))
		return
	}
	input, err := auditRequiredInt(total, "input_tokens")
	if err != nil {
		*refusals = append(*refusals, newAuditRefusal(AuditSourceCodex, rel, line, "codex_input_tokens", err.Error()))
		return
	}
	output, err := auditRequiredInt(total, "output_tokens")
	if err != nil {
		*refusals = append(*refusals, newAuditRefusal(AuditSourceCodex, rel, line, "codex_output_tokens", err.Error()))
		return
	}
	cacheRead, err := auditOptionalInt(total, "cached_input_tokens")
	if err != nil {
		*refusals = append(*refusals, newAuditRefusal(AuditSourceCodex, rel, line, "codex_read_bucket", err.Error()))
		return
	}
	// Keep the provider's exact wire key while avoiding a local identifier that
	// could be confused with fak's distinct cache concept family.
	cacheWrite, err := auditOptionalInt(total, "ca"+"che_write_input_tokens")
	if err != nil {
		*refusals = append(*refusals, newAuditRefusal(AuditSourceCodex, rel, line, "codex_write_bucket", err.Error()))
		return
	}
	if cacheRead+cacheWrite > input {
		*refusals = append(*refusals, newAuditRefusal(AuditSourceCodex, rel, line, "codex_bucket_subsets_exceed_input", "cached plus cache-write tokens exceed cumulative input_tokens"))
		return
	}
	raw := auditCodexRawTokens{Input: input, Output: output, CacheRead: cacheRead, CacheWrite: cacheWrite}
	if state.codexRawTotal != nil && !raw.atLeast(*state.codexRawTotal) {
		if !state.codexResetArmed {
			*refusals = append(*refusals, newAuditRefusal(AuditSourceCodex, rel, line, "codex_total_usage_decreased", "cumulative total_token_usage decreased without a versioned task_started boundary"))
			return
		}
		state.codexCompleted.add(state.codexRawTotal.normalized())
		state.row.UsageRecords++
	}
	state.codexResetArmed = false
	state.codexRawTotal = &raw
	state.usageExact++
	state.row.Tokens = state.codexCompleted
	state.row.Tokens.add(raw.normalized())
}

func observeCodexLastTokenUsage(info map[string]any, line int, rel string, state *auditParseState, refusals *[]AuditRefusalRow) {
	raw, exists := info["last_token_usage"]
	if !exists || raw == nil {
		return
	}
	last, ok := raw.(map[string]any)
	if !ok {
		*refusals = append(*refusals, newAuditRefusal(AuditSourceCodex, rel, line, "codex_last_usage_not_object", "last_token_usage must be an object or null"))
		return
	}
	cachedInput, err := auditRequiredInt(last, "cached_input_tokens")
	if err != nil {
		*refusals = append(*refusals, newAuditRefusal(AuditSourceCodex, rel, line, "codex_last_cached_input_tokens", err.Error()))
		return
	}
	if state.codexCacheSamples == 0 {
		state.codexCacheMin = cachedInput
		state.codexCacheMax = cachedInput
	} else {
		state.codexCacheMin = min(state.codexCacheMin, cachedInput)
		state.codexCacheMax = max(state.codexCacheMax, cachedInput)
	}
	state.codexCacheSamples++
}

func (t auditCodexRawTokens) atLeast(previous auditCodexRawTokens) bool {
	return t.Input >= previous.Input && t.Output >= previous.Output &&
		t.CacheRead >= previous.CacheRead && t.CacheWrite >= previous.CacheWrite
}

func (t auditCodexRawTokens) normalized() AuditTokens {
	return AuditTokens{
		InputTokens: t.Input - t.CacheRead - t.CacheWrite, OutputTokens: t.Output,
		CacheReadTokens: t.CacheRead, CacheCreateTokens: t.CacheWrite,
	}
}

func parseCodexResponseItem(payload map[string]any, line int, rel string, state *auditParseState, refusals *[]AuditRefusalRow) {
	kind, _ := payload["type"].(string)
	if _, hasUsage := payload["usage"]; hasUsage {
		*refusals = append(*refusals, newAuditRefusal(AuditSourceCodex, rel, line, "codex_response_item_usage", "response_item usage is not a supported cumulative token_count shape"))
	}
	switch kind {
	case "function_call", "custom_tool_call":
		id, _ := payload["call_id"].(string)
		if id == "" {
			id = "line:" + strconv.Itoa(line)
		}
		if _, seen := state.seenCalls[id]; seen {
			return
		}
		state.seenCalls[id] = struct{}{}
		name, _ := payload["name"].(string)
		args := payload["arguments"]
		if kind == "custom_tool_call" {
			args = payload["input"]
		}
		call := auditToolCall{name: name, args: auditCanonicalArgs(args), target: auditMutationTarget(name, args)}
		state.calls[id] = call
		state.row.ToolCalls++
		if call.target != "" {
			state.mutationCounts[call.target]++
			auditAppendMutationEvent(state, call.target, QwenMutationWrite)
		}
	case "function_call_output", "custom_tool_call_output":
		id, _ := payload["call_id"].(string)
		call := state.calls[id]
		if auditExpectedWaitTimeout(call, payload["output"]) {
			state.row.ExpectedWaitTimeouts++
			return
		}
		if !auditOutputIsError(payload["output"]) {
			if call.target == "" {
				auditAppendMutationEvent(state, "", QwenMutationWitness)
			}
			return
		}
		auditRecordToolFailure(state, call, payload["output"], line)
	}
}

func auditRecordToolFailure(state *auditParseState, call auditToolCall, output any, line int) {
	state.row.ToolErrors++
	content := auditText(output)
	name := call.name
	if name == "" {
		name = "unknown"
	}
	signature := name + "\x00" + call.args + "\x00" + auditNormalizedHead(content, 200)
	state.toolErrorEvents = append(state.toolErrorEvents, QwenToolErrorEvent{Content: content, Index: line})
	state.toolErrorAttributions = append(state.toolErrorAttributions, qwenToolErrorAttribution{failureKey: signature, mutationTarget: call.target})
	state.failureCounts[signature]++
}

func auditRepeatedFailureCount(counts map[string]int) int {
	var repeated int
	for _, count := range counts {
		if count > 1 {
			repeated += count - 1
		}
	}
	return repeated
}

// auditExpectedWaitTimeout recognizes only the bounded wait_agent polling
// primitive and only when its result explicitly reports a timeout. A timeout
// from any other tool, or any other wait_agent error, remains a failed action.
func auditExpectedWaitTimeout(call auditToolCall, output any) bool {
	return auditIsWaitAgentTool(call.name) && auditOutputIsTimeout(output) && !auditOutputHasTerminalFailure(output)
}

func auditIsWaitAgentTool(name string) bool {
	name = strings.ToLower(strings.TrimSpace(name))
	if name == "wait_agent" {
		return true
	}
	for _, namespaceSeparator := range []string{".", "/", ":", "__"} {
		if strings.HasSuffix(name, namespaceSeparator+"wait_agent") {
			return true
		}
	}
	return false
}

func auditOutputIsTimeout(value any) bool {
	switch typed := value.(type) {
	case map[string]any:
		for _, key := range []string{"timed_out", "timedOut"} {
			if flag, _ := typed[key].(bool); flag {
				return true
			}
		}
		for _, key := range []string{"status", "outcome"} {
			if status, ok := typed[key].(string); ok && auditTimeoutStatus(status) {
				return true
			}
		}
		for _, key := range []string{"output", "content", "result", "text", "error", "message"} {
			if child, ok := typed[key]; ok && auditOutputIsTimeout(child) {
				return true
			}
		}
	case []any:
		for _, child := range typed {
			if auditOutputIsTimeout(child) {
				return true
			}
		}
	case string:
		trimmed := strings.TrimSpace(typed)
		if trimmed == "" {
			return false
		}
		var decoded any
		decoder := json.NewDecoder(strings.NewReader(trimmed))
		decoder.UseNumber()
		if (strings.HasPrefix(trimmed, "{") || strings.HasPrefix(trimmed, "[")) && decoder.Decode(&decoded) == nil {
			return auditOutputIsTimeout(decoded)
		}
		normalized := strings.ToLower(strings.Join(strings.Fields(trimmed), " "))
		return auditTimeoutStatus(normalized)
	}
	return false
}

func auditOutputHasTerminalFailure(value any) bool {
	switch typed := value.(type) {
	case map[string]any:
		for _, key := range []string{"is_error", "isError"} {
			if flag, _ := typed[key].(bool); flag {
				return true
			}
		}
		if status, _ := typed["status"].(string); status != "" {
			switch strings.ToLower(strings.TrimSpace(status)) {
			case "failed", "error", "cancelled", "canceled":
				return true
			}
		}
		if raw, ok := typed["error"]; ok && auditText(raw) != "" && auditText(raw) != "<nil>" {
			return true
		}
		for _, key := range []string{"output", "content", "result"} {
			if child, ok := typed[key]; ok && auditOutputHasTerminalFailure(child) {
				return true
			}
		}
	case []any:
		for _, child := range typed {
			if auditOutputHasTerminalFailure(child) {
				return true
			}
		}
	case string:
		trimmed := strings.TrimSpace(typed)
		if trimmed == "" {
			return false
		}
		var decoded any
		decoder := json.NewDecoder(strings.NewReader(trimmed))
		decoder.UseNumber()
		if (strings.HasPrefix(trimmed, "{") || strings.HasPrefix(trimmed, "[")) && decoder.Decode(&decoded) == nil {
			return auditOutputHasTerminalFailure(decoded)
		}
	}
	return false
}

func auditTimeoutStatus(value string) bool {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "timeout", "timed_out", "timed-out", "timed out":
		return true
	default:
		return false
	}
}

func auditAppendMutationEvent(state *auditParseState, target string, kind QwenMutationKind) {
	accounted := state.row.Tokens.accountedTotal()
	delta := accounted - state.mutationAccountedFor
	if delta < 0 {
		delta = 0
	}
	state.mutationAccountedFor = accounted
	state.mutationEvents = append(state.mutationEvents, QwenMutationEvent{
		TranscriptID: state.row.TranscriptID, Target: target, Kind: kind,
		AccountedTokens: uint64(delta), HypothesisID: state.mutationHypothesis,
	})
}

// auditMutationHypothesis deterministically uses only explicit assistant text.
// Empty text leaves the hypothesis unchanged rather than inferring one from edits.
func auditMutationHypothesis(content any) string {
	blocks, _ := content.([]any)
	var text strings.Builder
	for _, raw := range blocks {
		block, _ := raw.(map[string]any)
		if block["type"] != "text" && block["type"] != "input_text" && block["type"] != "output_text" {
			continue
		}
		text.WriteString(auditText(block["text"]))
	}
	return strings.TrimSpace(text.String())
}
func auditMutationTarget(name string, args any) string {
	lower := strings.ToLower(name)
	if !strings.Contains(lower, "edit") && !strings.Contains(lower, "write") && !strings.Contains(lower, "patch") {
		return ""
	}
	value := args
	if text, ok := args.(string); ok {
		var decoded any
		decoder := json.NewDecoder(strings.NewReader(text))
		decoder.UseNumber()
		if decoder.Decode(&decoded) == nil {
			value = decoded
		} else if match := regexp.MustCompile(`(?m)^\*\*\* (?:Update|Add|Delete) File: ([^\r\n]+)`).FindStringSubmatch(text); len(match) == 2 {
			return filepath.ToSlash(strings.TrimSpace(match[1]))
		}
	}
	object, _ := value.(map[string]any)
	for _, key := range []string{"file_path", "notebook_path", "path"} {
		if path, ok := object[key].(string); ok && strings.TrimSpace(path) != "" {
			return filepath.ToSlash(strings.TrimSpace(path))
		}
	}
	return ""
}

func auditOutputIsError(value any) bool {
	switch typed := value.(type) {
	case map[string]any:
		for _, key := range []string{"is_error", "isError", "timed_out", "timedOut"} {
			if flag, _ := typed[key].(bool); flag {
				return true
			}
		}
		for _, key := range []string{"exit_code", "exitCode"} {
			if code, err := auditAnyInt(typed[key]); err == nil && code != 0 {
				return true
			}
		}
		if status, _ := typed["status"].(string); status == "failed" || status == "error" || auditTimeoutStatus(status) {
			return true
		}
		for _, key := range []string{"output", "content", "result", "text"} {
			if child, ok := typed[key]; ok && auditOutputIsError(child) {
				return true
			}
		}
	case []any:
		for _, child := range typed {
			if auditOutputIsError(child) {
				return true
			}
		}
	case string:
		trimmed := strings.TrimSpace(typed)
		if trimmed == "" {
			return false
		}
		var decoded any
		decoder := json.NewDecoder(strings.NewReader(trimmed))
		decoder.UseNumber()
		if (strings.HasPrefix(trimmed, "{") || strings.HasPrefix(trimmed, "[")) && decoder.Decode(&decoded) == nil {
			return auditOutputIsError(decoded)
		}
		lower := strings.ToLower(trimmed)
		return auditNonzeroExit.MatchString(trimmed) || strings.Contains(lower, `"is_error":true`) ||
			strings.Contains(lower, `"iserror":true`) || strings.Contains(lower, "timed out")
	}
	return false
}

func auditRecordType(record map[string]any) string {
	recordType, _ := record["type"].(string)
	if recordType == "" {
		return "<missing>"
	}
	if payload, ok := record["payload"].(map[string]any); ok {
		if subtype, ok := payload["type"].(string); ok && subtype != "" {
			return recordType + ":" + subtype
		}
	}
	return recordType
}

func newAuditRefusal(source, rel string, line int, code, detail string) AuditRefusalRow {
	return AuditRefusalRow{
		Schema: AuditSchema, Kind: "refusal", Source: source, SourcePath: rel,
		Line: line, Code: code, Detail: auditNormalizedHead(detail, 240),
	}
}

func auditRequiredInt(object map[string]any, key string) (int64, error) {
	value, ok := object[key]
	if !ok {
		return 0, fmt.Errorf("%s is missing", key)
	}
	return auditAnyInt(value)
}

func auditOptionalInt(object map[string]any, key string) (int64, error) {
	value, ok := object[key]
	if !ok || value == nil {
		return 0, nil
	}
	return auditAnyInt(value)
}

func auditAnyInt(value any) (int64, error) {
	var number int64
	switch typed := value.(type) {
	case json.Number:
		parsed, err := typed.Int64()
		if err != nil {
			return 0, fmt.Errorf("must be an integer")
		}
		number = parsed
	case int:
		number = int64(typed)
	case int64:
		number = typed
	case float64:
		if math.Trunc(typed) != typed || typed > math.MaxInt64 {
			return 0, fmt.Errorf("must be an integer")
		}
		number = int64(typed)
	default:
		return 0, fmt.Errorf("must be a non-negative integer")
	}
	if number < 0 {
		return 0, fmt.Errorf("must be a non-negative integer")
	}
	return number, nil
}

func auditCanonicalArgs(value any) string {
	if text, ok := value.(string); ok {
		var decoded any
		decoder := json.NewDecoder(strings.NewReader(text))
		decoder.UseNumber()
		if decoder.Decode(&decoded) == nil {
			return auditCanonicalJSON(decoded)
		}
		return auditNormalizedHead(text, 1000)
	}
	return auditCanonicalJSON(value)
}

func auditCanonicalJSON(value any) string {
	encoded, err := json.Marshal(value)
	if err != nil {
		return "<unencodable>"
	}
	return string(encoded)
}

func auditText(value any) string {
	switch typed := value.(type) {
	case string:
		return typed
	case []any:
		var builder strings.Builder
		for _, child := range typed {
			builder.WriteString(auditText(child))
			builder.WriteByte(' ')
		}
		return builder.String()
	case map[string]any:
		for _, key := range []string{"text", "content", "output", "result"} {
			if child, ok := typed[key]; ok {
				return auditText(child)
			}
		}
		return auditCanonicalJSON(typed)
	default:
		return fmt.Sprint(value)
	}
}

func auditNormalizedHead(value string, limit int) string {
	value = strings.Join(strings.Fields(value), " ")
	if len(value) > limit {
		value = value[:limit]
	}
	return value
}

func auditPercentile(values []int64, percentile float64) *int64 {
	if len(values) == 0 {
		return nil
	}
	sorted := append([]int64(nil), values...)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })
	index := int(math.Round((percentile / 100) * float64(len(sorted)-1)))
	if index < 0 {
		index = 0
	}
	if index >= len(sorted) {
		index = len(sorted) - 1
	}
	value := sorted[index]
	return &value
}
