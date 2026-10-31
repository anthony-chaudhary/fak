package gateway

// toolCallLogFinishReason admits only canonical metadata to refusal logs.
// Upstream finish strings are untrusted; unknown values never enter the log.
// The completion itself and its wire representation are left unchanged.
func toolCallLogFinishReason(reason string) string {
	switch reason {
	case "stop", "length", "tool_calls", "function_call", "content_filter":
		return reason
	default:
		return "unknown"
	}
}
