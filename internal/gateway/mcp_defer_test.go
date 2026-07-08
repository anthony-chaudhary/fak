package gateway

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// mcp_defer_test.go is the #3231 witness: with deferral on, tools/list is
// schema-light (only the bootstrap set), yet every cold tool stays re-findable
// through fak_tools_search AND callable through tools/call — deferral hides the
// schema, never the route. With deferral off (the default) tools/list is the
// full registry, byte-for-byte the pre-#3231 surface.

func newDeferServer(t *testing.T, defer_ bool) *Server {
	t.Helper()
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterAdjudicator(0, toolAdj{})
	srv, err := New(Config{EngineID: "test", Model: "test-model", VDSO: true, DeferMCPTools: defer_})
	if err != nil {
		t.Fatalf("New(defer=%v): %v", defer_, err)
	}
	t.Cleanup(srv.Close)
	return srv
}

func hasToolName(names []string, want string) bool {
	for _, n := range names {
		if n == want {
			return true
		}
	}
	return false
}

// TestDeferMCPToolsBootstrapList: with deferral on, tools/list returns ONLY the
// bootstrap set — the hot core + the search tool — so the cold long tail is
// absent from the always-sent floor.
func TestDeferMCPToolsBootstrapList(t *testing.T) {
	srv := newDeferServer(t, true)
	names := toolsListNames(t, srv)

	// Bootstrap members present.
	for want := range bootstrapToolNames {
		if !hasToolName(names, want) {
			t.Errorf("deferred tools/list missing bootstrap tool %q; got %v", want, names)
		}
	}
	// The search tool must be present (it is how cold schemas are re-found).
	if !hasToolName(names, "fak_tools_search") {
		t.Fatalf("deferred tools/list missing fak_tools_search; got %v", names)
	}
	// A cold tool must be ABSENT from the resident list.
	if hasToolName(names, "fak_memory_drivers") {
		t.Fatalf("deferred tools/list unexpectedly includes the cold tool fak_memory_drivers; got %v", names)
	}
	// The bootstrap is strictly smaller than the full registry (the whole point).
	full := toolsListNames(t, newDeferServer(t, false))
	if len(names) >= len(full) {
		t.Fatalf("bootstrap list (%d) is not smaller than the full registry (%d)", len(names), len(full))
	}
	t.Logf("deferred tools/list = %d tools (bootstrap) vs %d full — cold tail deferred", len(names), len(full))
}

// TestDeferMCPToolsDefaultFullList: default (deferral off) tools/list returns the
// full registry — no regression for a server that has not opted in.
func TestDeferMCPToolsDefaultFullList(t *testing.T) {
	srv := newDeferServer(t, false)
	names := toolsListNames(t, srv)
	if !hasToolName(names, "fak_memory_drivers") {
		t.Fatalf("default tools/list should contain the full registry incl. fak_memory_drivers; got %v", names)
	}
}

// TestDeferredToolStillSearchable: a cold tool absent from a deferred tools/list
// is still surfaced by fak_tools_search, ranked through the selfquery catalog —
// so the model can fault its schema back in on demand.
func TestDeferredToolStillSearchable(t *testing.T) {
	srv := newDeferServer(t, true)
	resp, err := srv.toolsSearch(ToolsSearchRequest{Query: "memory drivers", DetailLevel: "name"})
	if err != nil {
		t.Fatalf("toolsSearch: %v", err)
	}
	found := false
	for _, tl := range resp.Tools {
		if n, _ := tl["name"].(string); n == "fak_memory_drivers" {
			found = true
		}
	}
	if !found {
		got := make([]string, 0, len(resp.Tools))
		for _, tl := range resp.Tools {
			if n, _ := tl["name"].(string); n != "" {
				got = append(got, n)
			}
		}
		t.Fatalf("fak_tools_search did not surface the deferred tool fak_memory_drivers for intent 'memory drivers'; got %v", got)
	}
}

// TestDeferredToolStillCallable: deferral hides the SCHEMA, never the ROUTE — a
// cold tool absent from tools/list still dispatches through tools/call.
func TestDeferredToolStillCallable(t *testing.T) {
	srv := newDeferServer(t, true)
	params, _ := json.Marshal(map[string]any{"name": "fak_memory_drivers", "arguments": map[string]any{}})
	_, rerr := srv.handleMethod(context.Background(), "tools/call", params)
	if rerr != nil {
		t.Fatalf("deferred tool fak_memory_drivers should still dispatch via tools/call, got rpc error: %v", rerr.Message)
	}
}
