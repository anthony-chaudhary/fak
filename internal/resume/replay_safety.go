package resume

import (
	"encoding/json"
	"strings"
)

// EmittedBlockKind is the observable shape of one block emitted before a turn
// was interrupted. Keep this vocabulary closed: unknown kinds fail safe.
type EmittedBlockKind string

const (
	BlockText       EmittedBlockKind = "text"
	BlockImage      EmittedBlockKind = "image"
	BlockToolCall   EmittedBlockKind = "tool_call"
	BlockServerTool EmittedBlockKind = "server_tool"
	BlockThinking   EmittedBlockKind = "thinking"
	BlockToolResult EmittedBlockKind = "tool_result"
)

// EmittedBlock contains only fields needed to decide whether discarding and
// replaying a partial turn could duplicate an observable effect.
type EmittedBlock struct {
	Kind       EmittedBlockKind
	Text       string
	ToolCallID string
}

// ReplaySafe reports whether blocks can be discarded before retrying. Empty,
// thinking-only, and whitespace-only partials are safe. Everything else,
// including unknown future block kinds, is unsafe by default.
func ReplaySafe(blocks []EmittedBlock) bool {
	for _, block := range blocks {
		switch block.Kind {
		case BlockThinking:
			continue
		case BlockText:
			if strings.TrimSpace(block.Text) == "" {
				continue
			}
			return false
		default:
			return false
		}
	}
	return true
}

// RetryBranch identifies the existing error-classification branch requesting
// an automatic retry. Replay safety is an additional gate, never a substitute
// for this classification.
type RetryBranch string

const (
	RetryNotEligible       RetryBranch = "not_eligible"
	RetryableError         RetryBranch = "retryable_error"
	RetryClassifierRefusal RetryBranch = "classifier_refusal"
)

// PartialRetryAction is the typed outcome for an interrupted partial turn.
type PartialRetryAction string

const (
	PartialRetry            PartialRetryAction = "retry"
	PartialRetrySuppressed  PartialRetryAction = "suppress_retry"
	PartialPreserveContinue PartialRetryAction = "preserve_and_continue"
)

const (
	ReasonReplaySafe           = "REPLAY_SAFE_PARTIAL"
	ReasonReplayUnsafeOutput   = "REPLAY_UNSAFE_OUTPUT"
	ReasonRetryNotEligible     = "ERROR_NOT_RETRYABLE"
	ReasonCompletedToolEffects = "COMPLETED_TOOL_EFFECTS"
)

// PartialRetryDecision records both actuation and the reason operators need to
// distinguish an error-classification refusal from replay-unsafe output.
type PartialRetryDecision struct {
	Action PartialRetryAction
	Reason string
}

// DecidePartialRetry adds content-shaped replay safety to an already-classified
// retry branch. Completed tool calls are preserved and continued so their side
// effects are never replayed.
func DecidePartialRetry(branch RetryBranch, blocks []EmittedBlock) PartialRetryDecision {
	if completedToolEffects(blocks) {
		return PartialRetryDecision{Action: PartialPreserveContinue, Reason: ReasonCompletedToolEffects}
	}
	if branch != RetryableError && branch != RetryClassifierRefusal {
		return PartialRetryDecision{Action: PartialRetrySuppressed, Reason: ReasonRetryNotEligible}
	}
	if !ReplaySafe(blocks) {
		return PartialRetryDecision{Action: PartialRetrySuppressed, Reason: ReasonReplayUnsafeOutput}
	}
	return PartialRetryDecision{Action: PartialRetry, Reason: ReasonReplaySafe}
}

// HasDanglingToolCall reports whether any tool/server-tool call in blocks lacks a
// matching result — the one tail shape whose side effect cannot be proven absent: the
// call was emitted, the process died, and whether the tool ran is unknowable from the
// transcript. A call with an empty id can never be matched, so it counts as dangling.
func HasDanglingToolCall(blocks []EmittedBlock) bool {
	results := make(map[string]struct{})
	for _, block := range blocks {
		if block.Kind == BlockToolResult && block.ToolCallID != "" {
			results[block.ToolCallID] = struct{}{}
		}
	}
	for _, block := range blocks {
		if block.Kind != BlockToolCall && block.Kind != BlockServerTool {
			continue
		}
		if block.ToolCallID == "" {
			return true
		}
		if _, ok := results[block.ToolCallID]; !ok {
			return true
		}
	}
	return false
}

// EmittedBlocksFromContent maps one transcript message's content JSON onto the closed
// EmittedBlock vocabulary. A bare JSON string is one text block. Unknown block types keep
// their raw type token as the kind, which ReplaySafe fails safe on — a future block shape
// is unsafe until this vocabulary learns it. Malformed JSON folds to nil (no blocks
// proven), never an error: the callers' gates are fail-open on absent evidence.
func EmittedBlocksFromContent(raw json.RawMessage) []EmittedBlock {
	trimmed := strings.TrimSpace(string(raw))
	if trimmed == "" || trimmed == "null" {
		return nil
	}
	var plain string
	if json.Unmarshal(raw, &plain) == nil {
		if strings.TrimSpace(plain) == "" {
			return nil
		}
		return []EmittedBlock{{Kind: BlockText, Text: plain}}
	}
	var parts []struct {
		Type      string `json:"type"`
		Text      string `json:"text"`
		ID        string `json:"id"`
		ToolUseID string `json:"tool_use_id"`
	}
	if json.Unmarshal(raw, &parts) != nil {
		return nil
	}
	var out []EmittedBlock
	for _, p := range parts {
		switch p.Type {
		case "text":
			out = append(out, EmittedBlock{Kind: BlockText, Text: p.Text})
		case "thinking", "redacted_thinking":
			out = append(out, EmittedBlock{Kind: BlockThinking})
		case "image":
			out = append(out, EmittedBlock{Kind: BlockImage})
		case "tool_use", "mcp_tool_use":
			out = append(out, EmittedBlock{Kind: BlockToolCall, ToolCallID: p.ID})
		case "server_tool_use":
			out = append(out, EmittedBlock{Kind: BlockServerTool, ToolCallID: p.ID})
		case "tool_result", "mcp_tool_result", "web_search_tool_result", "code_execution_tool_result":
			out = append(out, EmittedBlock{Kind: BlockToolResult, ToolCallID: p.ToolUseID})
		default:
			out = append(out, EmittedBlock{Kind: EmittedBlockKind(p.Type)})
		}
	}
	return out
}

func completedToolEffects(blocks []EmittedBlock) bool {
	calls := make(map[string]struct{})
	results := make(map[string]struct{})
	for _, block := range blocks {
		if block.ToolCallID == "" {
			continue
		}
		switch block.Kind {
		case BlockToolCall:
			calls[block.ToolCallID] = struct{}{}
		case BlockToolResult:
			results[block.ToolCallID] = struct{}{}
		}
	}
	if len(calls) == 0 {
		return false
	}
	for id := range calls {
		if _, ok := results[id]; !ok {
			return false
		}
	}
	return true
}
