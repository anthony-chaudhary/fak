package gateway

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/treestatus"
)

func TestMCPTreeStatusTool(t *testing.T) {
	srv := newTestServer(t)

	res := callMCPTool[treestatus.Report](t, srv, "fak_tree_status", map[string]any{
		"lane": "gateway",
	})
	if res.Branch == "" {
		t.Error("expected non-empty branch in tree status report")
	}
	if res.Head == "" {
		t.Error("expected non-empty head SHA in tree status report")
	}
}
