package abi

// InlineResult constructs a successful inline result for tool payload bytes.
func InlineResult(tool string, payload []byte) (*ToolCall, *Result) {
	call := &ToolCall{Tool: tool}
	body := append([]byte(nil), payload...)
	return call, &Result{
		Call:    call,
		Payload: Ref{Kind: RefInline, Inline: body, Len: int64(len(body))},
		Status:  StatusOK,
	}
}
