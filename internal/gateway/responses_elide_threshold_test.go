package gateway

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

func TestResponsesElideThresholdMatchesAnthropicDefault(t *testing.T) {
	t.Setenv("FAK_RESPONSES_ELIDE_THRESHOLD", "")
	srv := newTestServer(t)
	if got := srv.responsesElideThreshold(); got != DocumentedElideResultBytes {
		t.Fatalf("responses threshold = %d, Anthropic reviewed threshold = %d", got, DocumentedElideResultBytes)
	}

	toolOutput := func(size int) []agent.Message {
		return []agent.Message{
			{Role: agent.RoleTool, ToolCallID: "old", Content: strings.Repeat("A", size)},
			{Role: agent.RoleTool, ToolCallID: "r1", Content: "r1"},
			{Role: agent.RoleTool, ToolCallID: "r2", Content: "r2"},
			{Role: agent.RoleTool, ToolCallID: "r3", Content: "r3"},
			{Role: agent.RoleTool, ToolCallID: "r4", Content: "r4"},
		}
	}
	for _, tc := range []struct {
		name   string
		size   int
		elided bool
	}{
		{"ordinary_4KiB", 4 << 10, false},
		{"oversized_20KiB", 20 << 10, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			msgs := toolOutput(tc.size)
			got := srv.maybeElideResponsesToolResults("threshold-test", msgs)[0].Content
			elided := strings.Contains(got, "fak_context_restore id=")
			if elided != tc.elided {
				t.Fatalf("size=%d elided=%t, want %t", tc.size, elided, tc.elided)
			}
		})
	}
}

func TestResponsesElideThresholdOverrideAndZeroFallback(t *testing.T) {
	srv := newTestServer(t)
	t.Setenv("FAK_RESPONSES_ELIDE_THRESHOLD", "8192")
	if got := srv.responsesElideThreshold(); got != 8192 {
		t.Fatalf("override = %d, want 8192", got)
	}
	t.Setenv("FAK_RESPONSES_ELIDE_THRESHOLD", "0")
	if got := srv.responsesElideThreshold(); got != DocumentedElideResultBytes {
		t.Fatalf("zero fallback = %d, want %d", got, DocumentedElideResultBytes)
	}
}
