package main

import (
	"io"
	"strings"
	"testing"
)

// fak-test:runtime fast est=5ms lane=default
func TestOpsRunPiTerminalSequences(t *testing.T) {
	const stop = `{"type":"message_end","message":{"role":"assistant","content":[{"type":"text","text":"bounded answer"}],"stopReason":"stop"}}`
	const start = `{"type":"message_start","message":{"role":"assistant"}}`
	const toolUse = `{"type":"message_end","message":{"role":"assistant","stopReason":"toolUse"}}`
	const settled = `{"type":"agent_settled","aborted":false}`
	const aborted = `{"type":"agent_settled","aborted":true}`
	for _, tc := range []struct {
		name     string
		events   []string
		complete bool
	}{
		{name: "modern_completed", events: []string{start, stop, `{"type":"agent_end","willRetry":false}`, settled}, complete: true},
		{name: "user_message_only", events: []string{`{"type":"message_end","message":{"role":"user","stopReason":"stop"}}`, settled}},
		{name: "tool_result_only", events: []string{`{"type":"message_end","message":{"role":"toolResult","stopReason":"stop"}}`, settled}},
		{name: "assistant_start_only", events: []string{start, settled}},
		{name: "assistant_delta_only", events: []string{start, `{"type":"message_update","assistantMessageEvent":{"type":"text_delta","delta":"partial"}}`, settled}},
		{name: "assistant_end_without_reason", events: []string{`{"type":"message_end","message":{"role":"assistant","content":[{"type":"text","text":"partial"}]}}`, settled}},
		{name: "settled_without_answer", events: []string{settled}},
		{name: "stop_then_unfinished_assistant", events: []string{stop, start, settled}},
		{name: "stop_then_tool_use", events: []string{stop, start, toolUse, settled}},
		{name: "tool_use_then_final_stop", events: []string{start, toolUse, start, stop, settled}, complete: true},
		{name: "length_is_not_completion", events: []string{`{"type":"message_end","message":{"role":"assistant","stopReason":"length"}}`, settled}},
		{name: "stop_then_length", events: []string{stop, `{"type":"message_end","message":{"role":"assistant","stopReason":"length"}}`, settled}},
		{name: "aborted_after_stop", events: []string{stop, aborted}},
		{name: "assistant_error_after_stop", events: []string{stop, `{"type":"message_end","message":{"role":"assistant","stopReason":"error"}}`, settled}},
		{name: "invalid_stop_reason_after_stop", events: []string{stop, `{"type":"message_end","message":{"role":"assistant","stopReason":17}}`, settled}},
		{name: "invalid_aborted_type_after_stop", events: []string{stop, `{"type":"agent_settled","aborted":"false"}`}},
		{name: "stream_error_is_sticky", events: []string{`{"type":"error","message":"fixture failure"}`, start, stop, settled}},
		{name: "legacy_error_is_sticky", events: []string{`{"type":"run_error","message":"fixture failure"}`, stop, settled}},
		{name: "legacy_finish_then_tool_use", events: []string{`{"type":"finish"}`, toolUse, settled}},
		{name: "legacy_finish_then_aborted", events: []string{`{"type":"finish"}`, aborted}},
		{name: "legacy_assistant", events: []string{`{"type":"assistant"}`}, complete: true},
		{name: "legacy_message", events: []string{`{"type":"message"}`}, complete: true},
		{name: "legacy_result", events: []string{`{"type":"result"}`}, complete: true},
		{name: "legacy_run_result", events: []string{`{"type":"run_result"}`}, complete: true},
		{name: "legacy_finish", events: []string{`{"type":"finish"}`}, complete: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			stream := strings.Join(tc.events, "\n")
			for _, fragmented := range []bool{false, true} {
				events := &opsPiEvents{output: io.Discard}
				if fragmented {
					for i := range stream {
						if _, err := events.Write([]byte{stream[i]}); err != nil {
							t.Fatal(err)
						}
					}
				} else if _, err := io.WriteString(events, stream); err != nil {
					t.Fatal(err)
				}
				events.finishLine()
				if got := opsPiComplete(events); got != tc.complete {
					t.Errorf("complete=%v, want %v (fragmented=%v)", got, tc.complete, fragmented)
				}
			}
		})
	}
}
