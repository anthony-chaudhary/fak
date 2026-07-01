package gateway

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/selfquery"
)

func TestMCPCapabilitiesListed(t *testing.T) {
	srv := newTestServer(t)
	list := resultMap(t, rpcRoundTrip(t, srv, "tools/list", ""))
	tools, ok := list["tools"].([]any)
	if !ok {
		t.Fatalf("tools/list malformed: %v", list)
	}
	names := map[string]bool{}
	for _, raw := range tools {
		tool := raw.(map[string]any)
		names[tool["name"].(string)] = true
	}
	if !names["fak_capabilities"] {
		t.Fatal("tools/list missing fak_capabilities")
	}
}

func TestMCPCapabilitiesEmptyQueryListsToolbelt(t *testing.T) {
	srv := newTestServer(t)
	root := writeMCPIndexRepo(t)

	resp := callMCPTool[selfquery.CapabilitiesResponse](t, srv, "fak_capabilities", map[string]any{
		"root": root,
	})
	names := map[string]bool{}
	for _, c := range resp.Cards {
		names[c.Name] = true
	}
	for _, want := range []string{"memory-driver:recall", "memory-driver:compact", "fak index lane", "fak_changes", "dos_arbitrate"} {
		if !names[want] {
			t.Fatalf("fak_capabilities empty query missing %s; got %v", want, names)
		}
	}
}

func TestMCPCapabilitiesCompactIntentReadyMemoryRunCall(t *testing.T) {
	srv := newTestServer(t)
	root := writeMCPIndexRepo(t)

	resp := callMCPTool[selfquery.CapabilitiesResponse](t, srv, "fak_capabilities", map[string]any{
		"root":  root,
		"query": "compact my context",
	})
	if len(resp.Cards) == 0 || resp.Cards[0].Name != "memory-driver:compact" {
		t.Fatalf("fak_capabilities compact intent top card = %+v, want memory-driver:compact first", resp.Cards)
	}
	names := map[string]bool{}
	for _, c := range resp.Cards {
		names[c.Name] = true
	}
	if !names["memory-driver:clean"] {
		t.Fatalf("fak_capabilities compact intent should also surface memory-driver:clean; got %v", names)
	}
	req := resp.Cards[0].Request
	if req.MCPTool != "fak_memory_run" || req.Executed {
		t.Fatalf("memory-driver:compact request = %+v, want ready unexecuted fak_memory_run call", req)
	}
}

func TestMCPCapabilitiesNegativeLimitInvalidParams(t *testing.T) {
	srv := newTestServer(t)
	root := writeMCPIndexRepo(t)
	params, _ := json.Marshal(map[string]any{
		"name":      "fak_capabilities",
		"arguments": map[string]any{"root": root, "limit": -1},
	})
	if _, rerr := srv.callTool(context.Background(), params); rerr == nil || rerr.Code != rpcInvalidParams {
		t.Fatalf("fak_capabilities with negative limit should be InvalidParams, got %+v", rerr)
	}
}
