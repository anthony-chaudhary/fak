package gateway

import (
	"context"
	"encoding/json"
	"testing"
)

func TestNativeMCPFilterABProof(t *testing.T) {
	s := newDeferServer(t, false)
	full := s.exposedToolDescriptors()
	active, receipt := s.toolsListView()
	if receipt.Mode != "active" || len(active) >= len(full) || receipt.SavedBytes <= 0 {
		t.Fatalf("active arm receipt=%+v active=%d full=%d", receipt, len(active), len(full))
	}

	t.Setenv("FAK_ABLATE_MCP_TOOL_FILTER", "1")
	control, controlReceipt := s.toolsListView()
	if controlReceipt.Mode != "bypass" || len(control) != len(full) || controlReceipt.SavedBytes != 0 {
		t.Fatalf("control arm receipt=%+v control=%d full=%d", controlReceipt, len(control), len(full))
	}

	held := []struct{ query, want string }{
		{"memory drivers", "fak_memory_drivers"},
		{"change context budget", "fak_context_change"},
		{"query available features", "fak_feature_query"},
	}
	for _, tc := range held {
		names := s.rankToolNamesByIntent(tc.query)
		if !hasToolName(names, tc.want) {
			t.Errorf("held query %q missed %q: %v", tc.query, tc.want, names)
		}
	}

	// Security parity: a tool excluded by --expose is neither rediscovered nor
	// callable in either arm. The optimization cannot widen the capability set.
	s.exposeAllow = func(name string) bool { return name != "fak_memory_run" }
	if hasToolName(s.rankToolNamesByIntent("run memory"), "fak_memory_run") {
		t.Fatal("filtered-out tool leaked through search")
	}
	params, _ := json.Marshal(map[string]any{"name": "fak_memory_run", "arguments": map[string]any{}})
	if _, rerr := s.callTool(context.Background(), params); rerr == nil {
		t.Fatal("filtered-out tool became callable")
	}
}
