package agent

// message_compact.go — structured prompt-shrink passes for the typed []Message / []ToolDef wire.
//
// While the Anthropic passthrough rewrites raw JSON bytes (req.Raw) via byte-splicing to preserve
// the client's cache_control prefix, native / in-kernel planning decodes from []Message and []ToolDef
// and renders ChatML via internal/tokenizer. Mutating req.Raw has zero effect on the in-kernel ChatML
// tokenizer.
//
// This file provides the typed-messages counterparts for all three prompt-shrink levers:
//   - CompactMessages / CompactMessagesWithOptions: history compaction under a resident budget
//     while preserving the leading system prompt (the RadixAttention prefix anchor), hoisting
//     standing goals ([fak:goal]), creating originating-task tombstones, and avoiding orphaning
//     tool calls / tool results.
//   - ElideStaleReadMessages: replaces Read tool_results whose file was superseded by a later in-session
//     Edit/Write/MultiEdit with a compact fak_context_restore marker.
//   - DeferColdToolDefs: filters cold tool definitions from the advertised schema list and provides
//     a ToolSearch tool so the model can search for them on demand.

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"path/filepath"
	"strings"
)

// EstimateMessageTokens estimates the token weight of a single chat message
// using the canonical ~4-characters-per-token heuristic plus framing overhead.
func EstimateMessageTokens(m Message) int {
	tokens := 4
	if len(m.Content) > 0 {
		tokens += (len(m.Content) + 3) / 4
	}
	if len(m.ReasoningContent) > 0 {
		tokens += (len(m.ReasoningContent) + 3) / 4
	}
	if len(m.Thinking) > 0 {
		tokens += (len(m.Thinking) + 3) / 4
	}
	for _, tc := range m.ToolCalls {
		tokens += (len(tc.Function.Name) + len(tc.Function.Arguments) + 16) / 4
	}
	if m.FunctionCall != nil {
		tokens += (len(m.FunctionCall.Name) + len(m.FunctionCall.Arguments) + 16) / 4
	}
	return tokens
}

// EstimateMessagesTokens estimates the total token weight of a slice of messages.
func EstimateMessagesTokens(messages []Message) int {
	total := 0
	for _, m := range messages {
		total += EstimateMessageTokens(m)
	}
	return total
}

// CompactMessages compacts a slice of messages under a resident token budget using default options.
func CompactMessages(messages []Message, budget int) ([]Message, CompactOutcome) {
	return CompactMessagesWithOptions(messages, CompactOptions{Budget: budget})
}

// CompactMessagesWithOptions compacts messages down to opts.Budget while:
//  1. Protecting leading system prompt messages (so the RadixAttention KV prefix stays identical).
//  2. Preserving the recent working-set window up to the budget.
//  3. Hoisting any standing goal pin ([fak:goal]) out of the dropped middle.
//  4. Ensuring assistant tool_calls and tool responses are not orphaned across the boundary.
//  5. Tombstoning the dropped originating user task with a content-addressed restore_id.
//  6. Emitting a synthetic stub message indicating the compacted turns.
func CompactMessagesWithOptions(messages []Message, opts CompactOptions) ([]Message, CompactOutcome) {
	budget := opts.Budget
	if budget <= 0 {
		return messages, CompactOutcome{Reason: CompactReasonUnderBudget}
	}
	if len(messages) < 3 {
		return messages, CompactOutcome{Reason: CompactReasonTooFewMsgs}
	}

	// 1. Find the protected system prefix at the start of messages.
	pfxEnd := -1
	for i := 0; i < len(messages); i++ {
		if messages[i].Role == RoleSystem || messages[i].Role == "system" {
			pfxEnd = i
		} else {
			break
		}
	}

	prefixTokens := 0
	if pfxEnd >= 0 {
		for i := 0; i <= pfxEnd; i++ {
			prefixTokens += EstimateMessageTokens(messages[i])
		}
	}
	compactStart := pfxEnd + 1
	if len(messages)-compactStart < 2 {
		return messages, CompactOutcome{Reason: CompactReasonTooFewMsgs, ProtectedPrefixTokens: prefixTokens}
	}

	// 2. Total compactible suffix tokens.
	suffixTokens := 0
	for i := compactStart; i < len(messages); i++ {
		suffixTokens += EstimateMessageTokens(messages[i])
	}
	if suffixTokens <= budget {
		return messages, CompactOutcome{
			Reason:                CompactReasonUnderBudget,
			ProtectedPrefixTokens: prefixTokens,
			SuffixTokens:          suffixTokens,
			AnchorStarved:         prefixTokens > budget,
		}
	}

	// 3. Choose the kept recent window backwards from the end.
	keepStart := len(messages)
	acc := 0
	for i := len(messages) - 1; i >= compactStart; i-- {
		cost := EstimateMessageTokens(messages[i])
		if acc+cost > budget && acc > 0 {
			break
		}
		acc += cost
		keepStart = i
	}

	// 4. Do not orphan tool calls / tool results:
	// If keepStart lands on a tool response, pull in the preceding assistant message.
	for keepStart > compactStart && (messages[keepStart].Role == RoleTool || messages[keepStart].Role == "tool" || messages[keepStart].ToolCallID != "") {
		keepStart--
	}
	if keepStart <= compactStart || keepStart >= len(messages) {
		return messages, CompactOutcome{
			Reason:                CompactReasonWindowNoDrop,
			ProtectedPrefixTokens: prefixTokens,
			SuffixTokens:          suffixTokens,
		}
	}

	// 5. Inspect the dropped range [compactStart, keepStart).
	goalIdx := -1
	for i := compactStart; i < keepStart; i++ {
		if isCompactGoalText(messages[i].Content) {
			goalIdx = i
		}
	}

	var restoreID, restoreExcerpt string
	var restoreBytes []byte
	if (messages[compactStart].Role == RoleUser || messages[compactStart].Role == "user") && goalIdx != compactStart {
		content := strings.TrimSpace(messages[compactStart].Content)
		excerpt := content
		runes := []rune(excerpt)
		if len(runes) > compactTombstoneCap {
			excerpt = string(runes[:compactTombstoneCap]) + "..."
		}
		sum := sha256.Sum256([]byte(content))
		restoreID = hex.EncodeToString(sum[:])
		restoreExcerpt = excerpt
		restoreBytes = []byte(content)
		if opts.RestoreStash != nil {
			opts.RestoreStash(restoreID, excerpt, restoreBytes)
		}
	}

	droppedCount := 0
	droppedTokens := 0
	for i := compactStart; i < keepStart; i++ {
		if i == goalIdx {
			continue // hoisted
		}
		droppedCount++
		droppedTokens += EstimateMessageTokens(messages[i])
	}

	stubContent := fmt.Sprintf("%s%d earlier turn(s) (%d estimated tokens) to stay within the resident budget; detail is omitted.", compactStubPrefix, droppedCount, droppedTokens)
	if restoreID != "" {
		stubContent += fmt.Sprintf("\n%s%s [restore_id=%s]", compactTombstonePrefix, restoreExcerpt, restoreID)
	}

	stubRole := RoleUser
	if keepStart < len(messages) && (messages[keepStart].Role == RoleUser || messages[keepStart].Role == "user") {
		stubRole = RoleAssistant
	}

	stubMsg := Message{
		Role:    stubRole,
		Content: stubContent,
	}

	out := make([]Message, 0, compactStart+2+len(messages)-keepStart)
	if compactStart > 0 {
		out = append(out, messages[:compactStart]...)
	}
	if goalIdx >= 0 {
		out = append(out, messages[goalIdx])
	}
	out = append(out, stubMsg)
	out = append(out, messages[keepStart:]...)

	return out, CompactOutcome{
		Reason:                CompactReasonNone,
		Dropped:               droppedCount,
		ShedTokens:            droppedTokens,
		ProtectedPrefixTokens: prefixTokens,
		SuffixTokens:          suffixTokens,
		RestoreID:             restoreID,
		RestoreExcerpt:        restoreExcerpt,
		RestoreBytes:          restoreBytes,
	}
}

// ElideStaleReadMessages elides Read tool results that have been superseded by subsequent
// Edit/Write/MultiEdit/NotebookEdit calls to the same file path, replacing their bodies
// with compact restore markers while preserving the recent working set tail.
func ElideStaleReadMessages(messages []Message, stash func(id, excerpt string, body []byte)) []Message {
	const defaultStaleProtectTail = 6
	if len(messages) <= defaultStaleProtectTail {
		return messages
	}
	out := append([]Message(nil), messages...)
	readPath := make(map[string]string)
	lastEdit := make(map[string]int)
	for i, msg := range messages {
		for _, call := range msg.ToolCalls {
			path := decodedToolPath(call.Function.Arguments)
			if path == "" {
				continue
			}
			switch strings.ToLower(call.Function.Name) {
			case "read":
				readPath[call.ID] = path
			case "edit", "write", "multiedit", "notebookedit":
				lastEdit[strings.ToLower(filepath.Clean(path))] = i
			}
		}
		if msg.FunctionCall != nil {
			path := decodedToolPath(msg.FunctionCall.Arguments)
			if path != "" {
				switch strings.ToLower(msg.FunctionCall.Name) {
				case "read":
					readPath[msg.ToolCallID] = path
				case "edit", "write", "multiedit", "notebookedit":
					lastEdit[strings.ToLower(filepath.Clean(path))] = i
				}
			}
		}
	}
	limit := len(messages) - defaultStaleProtectTail
	for i := 0; i < limit; i++ {
		msg := messages[i]
		if (msg.Role != RoleTool && msg.Role != "tool") || msg.ToolCallID == "" || msg.Content == "" {
			continue
		}
		path := readPath[msg.ToolCallID]
		if path == "" || lastEdit[strings.ToLower(filepath.Clean(path))] <= i {
			continue
		}
		body := []byte(msg.Content)
		sum := sha256.Sum256(body)
		id := hex.EncodeToString(sum[:])
		excerpt := strings.TrimSpace(msg.Content)
		if len(excerpt) > 160 {
			excerpt = excerpt[:160]
		}
		if stash != nil {
			stash(id, excerpt, body)
		}
		out[i].Content = fmt.Sprintf("...[fak: this Read of %s was superseded by a later in-session edit and its body was elided to stay within the context budget; recover via fak_context_restore restore_id=%s]...", path, id)
	}
	return out
}

func decodedToolPath(arguments string) string {
	var args map[string]json.RawMessage
	if json.Unmarshal([]byte(arguments), &args) != nil {
		return ""
	}
	for _, key := range []string{"file_path", "path", "notebook_path"} {
		var path string
		if json.Unmarshal(args[key], &path) == nil && strings.TrimSpace(path) != "" {
			return path
		}
	}
	return ""
}

// DefaultHotToolNames defines the default set of eager tools kept resident.
var DefaultHotToolNames = map[string]bool{
	"Bash": true, "Read": true, "Edit": true, "Write": true, "MultiEdit": true,
	"Glob": true, "Grep": true, "TodoWrite": true, "Task": true,
	"WebFetch": true, "WebSearch": true, "NotebookEdit": true,
	"ToolSearch": true,
}

// DeferColdToolDefs filters out cold tools from the advertised tools list and ensures
// ToolSearch is present so the model can discover deferred tools.
func DeferColdToolDefs(tools []ToolDef) ([]ToolDef, int) {
	if len(tools) == 0 {
		return tools, 0
	}
	var hot []ToolDef
	coldCount := 0
	hasSearch := false
	for _, t := range tools {
		name := t.Function.Name
		if strings.EqualFold(name, "ToolSearch") || strings.EqualFold(name, "tool_search") {
			hasSearch = true
			hot = append(hot, t)
			continue
		}
		if DefaultHotToolNames[name] {
			hot = append(hot, t)
		} else {
			coldCount++
		}
	}
	if coldCount == 0 {
		return tools, 0
	}
	if !hasSearch {
		hot = append(hot, ToolDef{
			Type: "function",
			Function: ToolDefFunction{
				Name:        "ToolSearch",
				Description: "Search and retrieve deferred tool definitions by name or query keyword.",
				Parameters:  json.RawMessage(`{"type":"object","properties":{"query":{"type":"string","description":"tool name or capability keyword to search"}},"required":["query"]}`),
			},
		})
	}
	return hot, coldCount
}

// TypedPromptShrinkConfig parameterizes the typed-messages prompt shrink pass.
type TypedPromptShrinkConfig struct {
	CompactHistoryBudget int
	ElideStaleReads      bool
	DeferColdTools       bool
	RestoreStash         func(id, excerpt string, body []byte)
}

// TypedPromptShrinkOutcome records what the prompt-shrink pass did.
type TypedPromptShrinkOutcome struct {
	Compacted         bool
	CompactOutcome    CompactOutcome
	StaleReadsElided  int
	ColdToolsDeferred int
}

// ApplyTypedPromptShrinkLevers runs all active prompt-shrink levers over typed messages and tools.
func ApplyTypedPromptShrinkLevers(messages []Message, tools []ToolDef, cfg TypedPromptShrinkConfig) ([]Message, []ToolDef, TypedPromptShrinkOutcome) {
	var outcome TypedPromptShrinkOutcome
	if cfg.ElideStaleReads {
		orig := messages
		messages = ElideStaleReadMessages(messages, cfg.RestoreStash)
		for i := range messages {
			if i < len(orig) && messages[i].Content != orig[i].Content {
				outcome.StaleReadsElided++
			}
		}
	}
	if cfg.CompactHistoryBudget > 0 {
		var compOutcome CompactOutcome
		messages, compOutcome = CompactMessagesWithOptions(messages, CompactOptions{
			Budget:       cfg.CompactHistoryBudget,
			RestoreStash: cfg.RestoreStash,
		})
		outcome.CompactOutcome = compOutcome
		outcome.Compacted = (compOutcome.Reason == CompactReasonNone)
	}
	if cfg.DeferColdTools {
		var deferred int
		tools, deferred = DeferColdToolDefs(tools)
		outcome.ColdToolsDeferred = deferred
	}
	return messages, tools, outcome
}
