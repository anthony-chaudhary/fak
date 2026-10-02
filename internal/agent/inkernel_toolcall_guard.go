package agent

import (
	"strings"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// markInKernelDroppedToolCalls fails closed when no structured call survives the
// post-decode lift despite a tool-call marker. Only a final unclosed call cut off
// by the token budget is oversized; a closed but unparseable call is malformed.
func markInKernelDroppedToolCalls(comp *Completion) {
	if comp == nil || len(comp.Message.ToolCalls) != 0 {
		return
	}
	content := comp.Message.Content
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
