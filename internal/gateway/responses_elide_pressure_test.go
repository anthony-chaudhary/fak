package gateway

import (
	"fmt"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/pkg/harnesskit"
)

// fak-test:runtime fast est=1s

const haloSlotWindow = 131072

// piReadTranscript mirrors a pi multi-file read: a user turn, then n assistant tool calls
// each answered by a read result of resultBytes.
func piReadTranscript(n, resultBytes int) []agent.Message {
	msgs := []agent.Message{
		{Role: agent.RoleSystem, Content: "pi system prompt"},
		{Role: agent.RoleUser, Content: "read these files"},
	}
	for i := 0; i < n; i++ {
		id := fmt.Sprintf("call_%d", i)
		msgs = append(msgs,
			agent.Message{Role: agent.RoleAssistant, ToolCalls: []agent.ToolCall{{ID: id, Type: "function", Function: agent.Func{Name: "read", Arguments: fmt.Sprintf(`{"path":"doc%d.md"}`, i)}}}},
			agent.Message{Role: agent.RoleTool, ToolCallID: id, Content: strings.Repeat(string(rune('a'+i%26)), resultBytes)},
		)
	}
	return msgs
}

func elidedCount(msgs []agent.Message) int {
	n := 0
	for _, m := range msgs {
		if strings.HasPrefix(m.Content, "...[fak: tool output elided") {
			n++
		}
	}
	return n
}

func haloServer(t *testing.T) *Server {
	t.Helper()
	srv := newTestServer(t)
	if srv.upstreamWindows == nil {
		srv.upstreamWindows = newUpstreamWindowCache()
	}
	srv.upstreamWindows.record(srv.model, haloSlotWindow)
	return srv
}

// The 2026-10-09 Halo probe: five ~52 KB reads (~65k resident) on a 131072 slot. The
// gateway elided the oldest read at turn 6, which broke the KV prefix and the deadline
// admission's prefix estimate. Below the client's own compaction trigger nothing is shed.
func TestElisionHoldsBelowClientCompactTrigger(t *testing.T) {
	srv := haloServer(t)
	env, err := harnesskit.DeriveContextEnvelope(harnesskit.ContextEnvelopeInput{ServedWindow: haloSlotWindow, Source: harnesskit.WindowServed})
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name  string
		msgs  []agent.Message
		limit int
	}{
		{"halo-probe-turn6", piReadTranscript(5, 52*1024), 0},
		{"at-client-trigger", piReadTranscript(8, env.CompactTrigger*4/8), env.CompactTrigger},
	} {
		if tc.limit > 0 && estimateMessageContentTokens(tc.msgs) < tc.limit {
			t.Fatalf("%s: fixture resident %d below trigger %d", tc.name, estimateMessageContentTokens(tc.msgs), tc.limit)
		}
		out := srv.maybeElideResponsesToolResults("t-pressure-"+tc.name, tc.msgs)
		if got := elidedCount(out); got != 0 {
			t.Fatalf("%s: elided %d tool results below the served-window shed line", tc.name, got)
		}
	}
}

// A client that did not compact and outgrew the envelope still gets shed.
func TestElisionFiresPastServedWindowEnvelope(t *testing.T) {
	srv := haloServer(t)
	msgs := piReadTranscript(10, 52*1024)
	if estimateMessageContentTokens(msgs) < haloSlotWindow {
		t.Fatalf("fixture resident %d not past the window", estimateMessageContentTokens(msgs))
	}
	out := srv.maybeElideResponsesToolResults("t-pressure-overflow", msgs)
	if got := elidedCount(out); got != 10-elideRecentKeepMsgs {
		t.Fatalf("elided %d, want %d", got, 10-elideRecentKeepMsgs)
	}
}
