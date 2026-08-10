package gateway

import (
	"context"
	"encoding/json"
	"reflect"
	"sort"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// newExposeServer builds an isolated echo-engine Server with the given --expose
// allowlist patterns, mirroring newTestServer's registrations. Not parallel-safe
// (mutates the global ABI registry).
func newExposeServer(t *testing.T, expose ...string) *Server {
	t.Helper()
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterAdjudicator(0, toolAdj{})
	srv, err := New(Config{EngineID: "test", Model: "test-model", VDSO: true, ExposeTools: expose})
	if err != nil {
		t.Fatalf("New(expose=%v): %v", expose, err)
	}
	t.Cleanup(srv.Close)
	return srv
}

// toolsListNames drives tools/list and returns the advertised tool names.
func toolsListNames(t *testing.T, srv *Server) []string {
	t.Helper()
	res, rerr := srv.handleMethod(context.Background(), "tools/list", nil)
	if rerr != nil {
		t.Fatalf("tools/list rpc error: %v", rerr.Message)
	}
	raw := res.(map[string]any)["tools"].([]map[string]any)
	names := make([]string, 0, len(raw))
	for _, td := range raw {
		if n, _ := td["name"].(string); n != "" {
			names = append(names, n)
		}
	}
	return names
}

func TestCompileToolExposeAllow(t *testing.T) {
	// Empty / blank-only input leaves NO allowlist (nil predicate = full surface).
	for _, raw := range [][]string{nil, {}, {"  ", ""}} {
		pred, err := compileToolExposeAllow(raw)
		if err != nil || pred != nil {
			t.Fatalf("compileToolExposeAllow(%q): err=%v, pred!=nil=%t, want (nil predicate, nil error)", raw, err, pred != nil)
		}
	}

	// Comma-split within one entry + surrounding whitespace: both names honored.
	pred, err := compileToolExposeAllow([]string{" fak_capabilities , fak_feature_query "})
	if err != nil {
		t.Fatalf("comma expose: %v", err)
	}
	for _, name := range []string{"fak_capabilities", "fak_feature_query"} {
		if !pred(name) {
			t.Errorf("comma expose: %q should be allowed", name)
		}
	}
	if pred("fak_syscall") {
		t.Error("comma expose: fak_syscall must NOT be allowed")
	}

	// Glob matches the runtime context family but not a sibling.
	pred, err = compileToolExposeAllow([]string{"fak_context_*"})
	if err != nil {
		t.Fatalf("glob expose: %v", err)
	}
	for _, name := range []string{"fak_context_change", "fak_context_restore", "fak_context_spans"} {
		if !pred(name) {
			t.Errorf("glob expose: %q should match fak_context_*", name)
		}
	}
	if pred("fak_capabilities") {
		t.Error("glob expose: fak_capabilities must not match fak_context_*")
	}

	// A pattern matching NO known tool fails loud (typo protection).
	if _, err := compileToolExposeAllow([]string{"fak_does_not_exist"}); err == nil ||
		!strings.Contains(err.Error(), "matches no known tool") {
		t.Fatalf("zero-match expose err = %v, want 'matches no known tool'", err)
	}
	// A malformed glob fails loud too.
	if _, err := compileToolExposeAllow([]string{"fak_["}); err == nil ||
		!strings.Contains(err.Error(), "not a valid glob") {
		t.Fatalf("malformed glob err = %v, want 'not a valid glob'", err)
	}
}

func TestExposeAllowlistFiltersDiscoveryAndGuardsCall(t *testing.T) {
	srv := newExposeServer(t, "fak_tools_search")
	want := map[string]bool{
		"fak_tools_search": true,
	}

	// 1. tools/list advertises ONLY the allowlisted set.
	got := toolsListNames(t, srv)
	if len(got) != len(want) {
		t.Fatalf("tools/list returned %d tools %v, want %d", len(got), got, len(want))
	}
	for _, n := range got {
		if !want[n] {
			t.Errorf("tools/list leaked hidden tool %q", n)
		}
	}

	// 2. A hidden tool is not callable, and the fault is byte-identical to a
	// non-existent tool's — hiding a tool never leaks that it exists.
	_, rerr := srv.handleMethod(context.Background(), "tools/call",
		json.RawMessage(`{"name":"fak_syscall","arguments":{"tool":"allow_x"}}`))
	if rerr == nil || rerr.Message != "unknown tool: fak_syscall" {
		t.Fatalf("hidden fak_syscall call: rerr=%v, want 'unknown tool: fak_syscall'", rerr)
	}

	// 3. An exposed tool still dispatches — and fak_tools_search's own progressive-
	// disclosure view honors the allowlist (it cannot surface a hidden tool either).
	res, rerr := srv.handleMethod(context.Background(), "tools/call",
		json.RawMessage(`{"name":"fak_tools_search","arguments":{"detail_level":"name"}}`))
	if rerr != nil {
		t.Fatalf("exposed fak_tools_search call: rpc error %v", rerr.Message)
	}
	var tsr ToolsSearchResponse
	decodeMCPText(t, res, &tsr)
	if len(tsr.Tools) != len(want) {
		t.Fatalf("fak_tools_search returned %d tools, want %d (the exposed set)", len(tsr.Tools), len(want))
	}
	for _, tool := range tsr.Tools {
		if n, _ := tool["name"].(string); !want[n] {
			t.Errorf("fak_tools_search leaked hidden tool %q", n)
		}
	}
}

func TestExposeEmptyExposesFullSurface(t *testing.T) {
	srv := newExposeServer(t) // no allowlist
	got := toolsListNames(t, srv)
	// Pin the exact inventory, not just its size: a count-only assertion lets a
	// rename (count unchanged) or a net-zero add+drop through silently. Adding,
	// dropping, or renaming an exposed tool must fail here and be re-pinned
	// deliberately, so the wire-visible MCP surface never drifts unreviewed.
	wantInventory := []string{
		"fak_adjudicate",
		"fak_admit",
		"fak_capabilities",
		"fak_changes",
		"fak_context_change",
		"fak_context_restore",
		"fak_context_spans",
		"fak_context_value",
		"fak_feature_query",
		"fak_memory_drivers",
		"fak_memory_explain",
		"fak_memory_run",
		"fak_read",
		"fak_resume_history",
		"fak_revoke",
		"fak_session_reset",
		"fak_syscall",
		"fak_tools_search",
		"fak_trajquery",
	}
	sort.Strings(got)
	if !reflect.DeepEqual(got, wantInventory) {
		t.Fatalf("exposed MCP tool inventory drifted:\n got: %v\nwant: %v", got, wantInventory)
	}
	present := false
	for _, n := range got {
		if n == "fak_syscall" {
			present = true
		}
	}
	if !present {
		t.Fatal("no-allowlist surface must include fak_syscall")
	}
	// The tool hidden in the allowlist test dispatches normally with no allowlist.
	if _, rerr := srv.handleMethod(context.Background(), "tools/call",
		json.RawMessage(`{"name":"fak_syscall","arguments":{"tool":"allow_x"}}`)); rerr != nil {
		t.Fatalf("no-allowlist fak_syscall must dispatch, got rpc error %v", rerr.Message)
	}
}

func TestRuntimeGatewayDoesNotExposeDevelopmentIndexTools(t *testing.T) {
	srv := newExposeServer(t)
	for _, name := range toolsListNames(t, srv) {
		if strings.HasPrefix(name, "fak_index_") {
			t.Fatalf("runtime gateway exposed development index tool %q", name)
		}
	}
	_, rerr := srv.handleMethod(context.Background(), "tools/call",
		json.RawMessage(`{"name":"fak_index_docs","arguments":{"query":"gateway"}}`))
	if rerr == nil || rerr.Message != "unknown tool: fak_index_docs" {
		t.Fatalf("development tool call = %v, want unknown-tool refusal", rerr)
	}
}

func TestNewRejectsBadExpose(t *testing.T) {
	for _, tc := range []struct {
		name    string
		expose  []string
		wantErr string
	}{
		{"zero-match", []string{"fak_nope_*"}, "matches no known tool"},
		{"malformed-glob", []string{"fak_["}, "not a valid glob"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			abi.ResetForTest()
			abi.RegisterRegionBackend(inlineBackend{})
			abi.RegisterEngine("test", echoEngine{})
			abi.RegisterAdjudicator(0, toolAdj{})
			srv, err := New(Config{EngineID: "test", Model: "m", VDSO: true, ExposeTools: tc.expose})
			if err == nil {
				srv.Close()
				t.Fatalf("New(expose=%v) succeeded, want error containing %q", tc.expose, tc.wantErr)
			}
			if !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("New err = %q, want substring %q", err.Error(), tc.wantErr)
			}
		})
	}
}
