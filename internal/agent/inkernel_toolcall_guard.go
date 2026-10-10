package agent

import (
	"encoding/json"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// markInKernelDroppedToolCalls fails closed when no structured call survives the
// post-decode lift despite a tool-call marker. Only a final unclosed call cut off
// by the token budget is oversized; a closed but unparseable call is malformed.
func markInKernelDroppedToolCalls(comp *Completion, offered OfferedTools) {
	if comp == nil || len(comp.Message.ToolCalls) != 0 {
		return
	}
	if len(offered.names) == 0 {
		return
	}
	content := comp.Message.Content
	if _, rejected := liftTextToolCalls(comp.Message, offered); rejected {
		// Ignore recognized preserved blocks, but keep malformed neighboring
		// markers subject to the existing fail-closed guard.
		recognized := OfferedTools{names: map[string]struct{}{"_": {}}, extension: func(string) bool { return true }}
		content = LiftTextToolCalls(comp.Message, recognized).Content
	}
	opener := strings.LastIndex(content, "<tool_call>")
	if opener < 0 {
		return
	}
	comp.ToolCallsDropped = true
	// Forced-tool enforcement may already have classified this completion.
	if comp.ToolCallsDroppedReason != abi.ReasonNone {
		return
	}
	comp.ToolCallsDroppedReason = abi.ReasonMalformed
	if comp.FinishReason == "length" && strings.LastIndex(content, "</tool_call>") < opener {
		comp.ToolCallsDroppedReason = abi.ReasonOversize
	}
}

// normalizeInKernelToolCalls is the existing Complete tail, with an unknown-name
// refusal retained across forced-tool handling. The explicit forced fallback is
// otherwise unchanged and remains distinct from text lifting.
func normalizeInKernelToolCalls(comp *Completion, offered OfferedTools, choice json.RawMessage, tools []ToolDef, messages []Message) *Completion {
	comp = normalizeCompletionToolCalls(comp, offered)
	if comp == nil {
		return nil
	}
	if comp.ToolCallsDroppedReason != abi.ReasonUnknownTool && inKernelEffectiveToolName(choice, tools) != "" {
		comp = enforceForcedToolChoice(comp, choice, tools, messages)
	}
	markInKernelDroppedToolCalls(comp, offered)
	return comp
}
