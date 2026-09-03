package gateway

import (
	"context"
	"encoding/json"
	"strings"
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
	full := newDeferServer(t, false).exposedToolDescriptors()
	if len(names) >= len(full) {
		t.Fatalf("bootstrap list (%d) is not smaller than the full registry (%d)", len(names), len(full))
	}
	t.Logf("deferred tools/list = %d tools (bootstrap) vs %d full — cold tail deferred", len(names), len(full))
}

// TestDeferMCPToolsDefaultFullList: default (deferral off) tools/list returns the
// full registry — no regression for a server that has not opted in.
func TestDeferMCPToolsDefaultOnReceipt(t *testing.T) {
	s := &Server{}
	got, receipt := s.toolsListView()
	if receipt.Mode != "active" || receipt.Reason != "default_on" {
		t.Fatalf("receipt=%+v want active/default_on", receipt)
	}
	if receipt.ToolsAfter != len(got) || receipt.ToolsBefore <= receipt.ToolsAfter {
		t.Fatalf("receipt counts=%+v got=%d", receipt, len(got))
	}
	if receipt.SavedBytes <= 0 || receipt.DescriptorBytesBefore-receipt.DescriptorBytesAfter != receipt.SavedBytes {
		t.Fatalf("receipt byte proof invalid: %+v", receipt)
	}
}

func TestDeferMCPToolsAblationFailsOpen(t *testing.T) {
	t.Setenv("FAK_ABLATE_MCP_TOOL_FILTER", "1")
	s := &Server{}
	got, receipt := s.toolsListView()
	if receipt.Mode != "bypass" || receipt.Reason != "ablation" {
		t.Fatalf("receipt=%+v want bypass/ablation", receipt)
	}
	if len(got) != len(s.exposedToolDescriptors()) || receipt.SavedBytes != 0 {
		t.Fatalf("ablation did not restore full list: %+v", receipt)
	}
}

func TestDeferMCPToolsDisableMCPDeferFailsOpen(t *testing.T) {
	s := &Server{disableMCPDefer: true}
	got, receipt := s.toolsListView()
	if receipt.Mode != "bypass" || receipt.Reason != "ablation" {
		t.Fatalf("receipt=%+v want bypass/ablation", receipt)
	}
	if len(got) != len(s.exposedToolDescriptors()) || receipt.SavedBytes != 0 {
		t.Fatalf("disableMCPDefer did not restore full list: %+v", receipt)
	}
}

func TestDeferMCPToolsHiddenRecoveryFailsOpen(t *testing.T) {
	s := &Server{exposeAllow: func(name string) bool { return name != "fak_tools_search" }}
	got, receipt := s.toolsListView()
	if receipt.Mode != "bypass" || receipt.Reason != "recovery_tool_hidden" {
		t.Fatalf("receipt=%+v want bypass/recovery_tool_hidden", receipt)
	}
	if len(got) != len(s.exposedToolDescriptors()) || receipt.SavedBytes != 0 {
		t.Fatalf("hidden recovery path did not fail open: %+v", receipt)
	}
}

func TestToolsListExposesFilterReceipt(t *testing.T) {
	s := &Server{}
	result, rerr := s.handleMethod(context.Background(), "tools/list", nil)
	if rerr != nil {
		t.Fatal(rerr)
	}
	m := result.(map[string]any)
	meta := m["_meta"].(map[string]any)
	receipt := meta["fak/tool_filter"].(MCPToolFilterStatus)
	if receipt.Mode != "active" || receipt.SavedBytes <= 0 {
		t.Fatalf("wire receipt=%+v", receipt)
	}
}

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

// TestTrimmedHotDescriptionsStayDiscoverable witnesses the #4250 trade: the
// largest always-sent descriptions are leaner, but intent search must still
// rank each tool first from the terms that remain.
func TestTrimmedHotDescriptionsStayDiscoverable(t *testing.T) {
	srv := newDeferServer(t, true)
	for _, tc := range []struct {
		query string
		want  string
	}{
		{"trajectory SQL scoped view", "fak_trajquery"},
		{"managed context pressure headroom", "fak_context_value"},
		{"restore dropped context sha256", "fak_context_restore"},
		{"list dropped context spans", "fak_context_spans"},
		{"resume heal history retry", "fak_resume_history"},
	} {
		t.Run(tc.want, func(t *testing.T) {
			resp, err := srv.toolsSearch(ToolsSearchRequest{Query: tc.query, DetailLevel: "name"})
			if err != nil {
				t.Fatalf("toolsSearch(%q): %v", tc.query, err)
			}
			if len(resp.Tools) == 0 {
				t.Fatalf("toolsSearch(%q) returned no tools", tc.query)
			}
			if got, _ := resp.Tools[0]["name"].(string); got != tc.want {
				t.Fatalf("toolsSearch(%q) ranked %q first, want %q", tc.query, got, tc.want)
			}
		})
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

func TestFakReadDiscoverySchemaDocumentsReceipt(t *testing.T) {
	srv := newDeferServer(t, true)
	resp, err := srv.toolsSearch(ToolsSearchRequest{Query: "verified fresh file read outcome receipt", DetailLevel: "full"})
	if err != nil {
		t.Fatal(err)
	}
	var tool map[string]any
	for _, candidate := range resp.Tools {
		if candidate["name"] == "fak_read" {
			tool = candidate
			break
		}
	}
	if tool == nil {
		t.Fatalf("fak_read not discoverable: %v", resp.Tools)
	}
	description, _ := tool["description"].(string)
	for _, term := range []string{"executed_cold_read", "verified_fresh_reuse", "duration_ns", "witness", "typed errors"} {
		if !strings.Contains(description, term) {
			t.Fatalf("description omits %q: %s", term, description)
		}
	}
	schemaBytes, err := json.Marshal(tool["inputSchema"])
	if err != nil {
		t.Fatal(err)
	}
	schema := string(schemaBytes)
	for _, term := range []string{"file_path", "file_paths"} {
		if !strings.Contains(schema, term) {
			t.Fatalf("input schema omits %q: %s", term, schema)
		}
	}
	resident, _ := srv.toolsListView()
	found := false
	for _, candidate := range resident {
		if candidate["name"] == "fak_read" {
			found = true
		}
	}
	if !found {
		t.Fatal("fak_read must remain in bootstrap discovery")
	}
}
