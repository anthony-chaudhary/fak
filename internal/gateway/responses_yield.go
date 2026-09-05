package gateway

import (
	"encoding/json"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

const (
	// DefaultMaxSubturnTokens is the default resident context token threshold
	// at which the mid-turn token yield valve activates.
	DefaultMaxSubturnTokens = 160000

	// DefaultMaxSubturnToolCalls is the default threshold for continuous
	// sub-turn tool calls / results before the yield valve activates.
	DefaultMaxSubturnToolCalls = 30

	// SubturnYieldHeader is the response header set when the sub-turn yield valve intercepts.
	SubturnYieldHeader = "X-Fak-Subturn-Yield"

	// SubturnYieldMessage is the synthetic concluding prompt returned to trigger native context compaction.
	SubturnYieldMessage = "Context threshold reached (resident sub-turn token yield valve activated). Concluding current turn to trigger native context compaction. Please summarize progress and resume from the latest state."
)

// resolveSubturnYieldThresholds returns the configurable thresholds for sub-turn tokens and tool calls.
// Defaults to 160000 tokens and 30 tool calls, overridable via environment variables:
// FAK_RESPONSES_MAX_SUBTURN_TOKENS and FAK_RESPONSES_MAX_SUBTURN_TOOL_CALLS.
func resolveSubturnYieldThresholds() (maxTokens int, maxToolCalls int) {
	maxTokens = DefaultMaxSubturnTokens
	if v := strings.TrimSpace(os.Getenv("FAK_RESPONSES_MAX_SUBTURN_TOKENS")); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			maxTokens = n
		}
	} else if v := strings.TrimSpace(os.Getenv("FAK_RESPONSES_SUBTURN_TOKEN_THRESHOLD")); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			maxTokens = n
		}
	}
	maxToolCalls = DefaultMaxSubturnToolCalls
	if v := strings.TrimSpace(os.Getenv("FAK_RESPONSES_MAX_SUBTURN_TOOL_CALLS")); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			maxToolCalls = n
		}
	} else if v := strings.TrimSpace(os.Getenv("FAK_RESPONSES_SUBTURN_TOOL_CALL_THRESHOLD")); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			maxToolCalls = n
		}
	}
	return maxTokens, maxToolCalls
}

// activeSubturnMessages returns the slice of messages occurring after the last user message.
// If the last message is a user message, an empty slice is returned.
// If there are no user messages, all messages are returned.
func activeSubturnMessages(messages []agent.Message) []agent.Message {
	lastUserIdx := -1
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role == agent.RoleUser {
			lastUserIdx = i
			break
		}
	}
	if lastUserIdx >= 0 {
		return messages[lastUserIdx+1:]
	}
	return messages
}

// isUserRawInputItem reports whether an item in raw input JSON is a user message.
func isUserRawInputItem(it responsesInputItem) bool {
	if it.Type != "message" && it.Type != "" {
		return false
	}
	if it.Role == "" {
		return false
	}
	return responsesRole(it.Role) == agent.RoleUser
}

// activeSubturnRawItems returns the slice of raw input items occurring after the last user message.
// If the last item is a user message, an empty slice is returned.
// If there are no user messages, all items are returned.
func activeSubturnRawItems(items []responsesInputItem) []responsesInputItem {
	lastUserIdx := -1
	for i := len(items) - 1; i >= 0; i-- {
		if isUserRawInputItem(items[i]) {
			lastUserIdx = i
			break
		}
	}
	if lastUserIdx >= 0 {
		return items[lastUserIdx+1:]
	}
	return items
}

// countSubturnToolCalls counts the total tool results / function calls in the decoded messages
// for the active sub-turn (messages after the last user message).
func countSubturnToolCalls(messages []agent.Message) int {
	calls := 0
	results := 0
	for _, m := range activeSubturnMessages(messages) {
		calls += len(m.ToolCalls)
		if m.FunctionCall != nil {
			calls++
		}
		if m.Role == agent.RoleTool || m.ToolCallID != "" {
			results++
		}
	}
	if results > calls {
		return results
	}
	return calls
}

// countSubturnToolCallsFromRaw counts function_call and function_call_output items from raw input JSON
// for the active sub-turn (items after the last user message).
func countSubturnToolCallsFromRaw(raw json.RawMessage) int {
	b := trimLeadingWS(raw)
	if len(b) == 0 || b[0] != '[' {
		return 0
	}
	var items []responsesInputItem
	if err := json.Unmarshal(raw, &items); err != nil {
		return 0
	}
	calls := 0
	results := 0
	for _, it := range activeSubturnRawItems(items) {
		switch it.Type {
		case "function_call":
			calls++
		case "function_call_output":
			results++
		}
	}
	if results > calls {
		return results
	}
	return calls
}

// totalSubturnToolCalls returns the maximum tool call / result count observed across
// decoded messages and raw input for the active sub-turn.
func totalSubturnToolCalls(messages []agent.Message, rawInput json.RawMessage) int {
	c1 := countSubturnToolCalls(messages)
	c2 := countSubturnToolCallsFromRaw(rawInput)
	if c2 > c1 {
		return c2
	}
	return c1
}

// estimateSubturnTokens estimates prompt tokens for the resident context using
// messages, tool definitions, and raw input bytes.
func estimateSubturnTokens(messages []agent.Message, tools []agent.ToolDef, rawInput json.RawMessage) int {
	chars := servedPromptChars(messages, tools)
	tokens := (chars + 3) / 4
	if len(rawInput) > 0 {
		rawTokens := (len(rawInput) + 3) / 4
		if rawTokens > tokens {
			tokens = rawTokens
		}
	}
	if chars > 0 && tokens == 0 {
		tokens = 1
	}
	return tokens
}

// detectSubturnRepetitionRunaway identifies sub-turn runaway conditions:
// 1. Tool call count exceeding 2x the configured limit.
// 2. Continuous repetition of identical tool calls with identical arguments.
func detectSubturnRepetitionRunaway(messages []agent.Message, toolCallCount, maxToolCalls int) bool {
	if maxToolCalls > 0 && toolCallCount >= maxToolCalls*2 {
		return true
	}
	streak := 0
	var lastSig string
	for _, m := range activeSubturnMessages(messages) {
		for _, tc := range m.ToolCalls {
			sig := tc.Function.Name + ":" + tc.Function.Arguments
			if sig == lastSig && sig != "" {
				streak++
				if streak >= 5 {
					return true
				}
			} else {
				lastSig = sig
				streak = 1
			}
		}
	}
	return false
}

// lastAssistantMessageWasYield checks whether the last assistant message in messages
// (or raw input) was SubturnYieldMessage, indicating the previous turn already yielded
// to request context compaction/summary.
func lastAssistantMessageWasYield(messages []agent.Message, rawInput json.RawMessage) bool {
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role == agent.RoleAssistant {
			return strings.Contains(messages[i].Content, SubturnYieldMessage)
		}
	}
	if len(messages) == 0 && len(rawInput) > 0 {
		b := trimLeadingWS(rawInput)
		if len(b) > 0 && b[0] == '[' {
			var items []responsesInputItem
			if err := json.Unmarshal(rawInput, &items); err == nil {
				for i := len(items) - 1; i >= 0; i-- {
					it := items[i]
					if (it.Type == "message" || it.Type == "") && it.Role == "assistant" {
						return strings.Contains(string(it.Content), SubturnYieldMessage)
					}
					if it.Type == "function_call" {
						return false
					}
				}
			}
		}
	}
	return false
}

// shouldYieldResponsesSubturn checks whether the mid-turn token yield valve should activate.
// Yields when resident context tokens exceed the threshold AND tool call count exceeds the limit,
// or if a sub-turn repetition runaway is detected.
func shouldYieldResponsesSubturn(messages []agent.Message, tools []agent.ToolDef, rawInput json.RawMessage) (bool, int, int) {
	maxTokens, maxToolCalls := resolveSubturnYieldThresholds()
	toolCallCount := totalSubturnToolCalls(messages, rawInput)
	estTokens := estimateSubturnTokens(messages, tools, rawInput)

	if lastAssistantMessageWasYield(messages, rawInput) {
		return false, estTokens, toolCallCount
	}
	if estTokens >= maxTokens && toolCallCount >= maxToolCalls {
		return true, estTokens, toolCallCount
	}
	if detectSubturnRepetitionRunaway(messages, toolCallCount, maxToolCalls) {
		return true, estTokens, toolCallCount
	}
	return false, estTokens, toolCallCount
}

// makeSubturnYieldResponse constructs the synthetic completed response that instructs
// the harness/agent to trigger native context compaction.
func makeSubturnYieldResponse(reqModel string, admissions []ResultAdmission) responsesResponse {
	resp := responsesResponse{
		ID:           "resp_fak_" + itoa(uint64(time.Now().UnixNano())),
		Object:       "response",
		CreatedAt:    time.Now().Unix(),
		Model:        reqModel,
		Status:       "completed",
		FinishReason: "stop",
		Output: responsesOutputFromAssistant(agent.Message{
			Role:    agent.RoleAssistant,
			Content: SubturnYieldMessage,
		}),
		OutputText: SubturnYieldMessage,
		Usage:      responsesUsage{},
	}
	if len(admissions) > 0 {
		resp.Fak = fakExtFrom(nil, admissions)
	}
	return resp
}
