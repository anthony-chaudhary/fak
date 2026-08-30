package gateway

import (
	"context"
	"encoding/json"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/selfquery"
)

func TestMCPCapabilitiesDiscoverable(t *testing.T) {
	srv := newTestServer(t)
	search, err := srv.toolsSearch(ToolsSearchRequest{Query: "query available capabilities", DetailLevel: "name"})
	if err != nil {
		t.Fatal(err)
	}
	for _, descriptor := range search.Tools {
		if descriptor["name"] == "fak_capabilities" {
			return
		}
	}
	t.Fatal("fak_tools_search missing fak_capabilities")
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
	for _, want := range []string{"memory-driver:recall", "memory-driver:compact", "fak_changes", "dos_arbitrate"} {
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

func TestMCPCapabilitiesExposesNativePerformanceStages(t *testing.T) {
	srv := newTestServer(t)
	root := writeMCPIndexRepo(t)
	tests := []struct {
		query   string
		detail  string
		command []string
	}{
		{"serve native model", "docs/model-engine-env.md", []string{"fak", "serve", "--gguf", "<model.gguf>", "--metal"}},
		{"benchmark native inference", "docs/model-engine-env.md", []string{"fak", "benchmarks", "describe", "modelbench"}},
		{"evaluate model quality", "docs/quality/output-quality-regression-runbook.md", []string{"fak", "quality", "run", "--json"}},
		{"profile native bottleneck", "docs/benchmarks/NATIVE-PERFORMANCE-HILLCLIMB.md", []string{"fak", "native-performance", "--profile-next", "profile.json"}},
		{"performance receipt", "docs/benchmarks/NATIVE-PERFORMANCE-REGRESSION-GATE.md", []string{"fak", "native-performance", "--gate", "gate-request.json"}},
	}
	for _, tc := range tests {
		t.Run(tc.query, func(t *testing.T) {
			resp := callMCPTool[selfquery.CapabilitiesResponse](t, srv, "fak_capabilities", map[string]any{
				"root": root, "query": tc.query, "limit": 1,
			})
			if len(resp.Cards) != 1 || resp.Cards[0].DetailRef != tc.detail {
				t.Fatalf("fak_capabilities(%q) = %#v, want detail %q first", tc.query, resp.Cards, tc.detail)
			}
			if !reflect.DeepEqual(resp.Cards[0].Request.Command, tc.command) {
				t.Fatalf("fak_capabilities(%q) command = %#v, want %#v", tc.query, resp.Cards[0].Request.Command, tc.command)
			}
		})
	}
}

// A successful capabilities lookup is stable and side-effect-free. Once the
// MCP boundary has returned it, identical immediate retries should reuse that
// result instead of reopening and rebuilding the catalog each time. The cache's
// execution/hit counters make the boundary observable without replacing the
// production loader: calls 2..5 must all be reuse hits.
func TestMCPCapabilitiesReusesIdenticalSuccessfulDiscovery(t *testing.T) {
	srv := newTestServer(t)
	root := writeMCPIndexRepo(t)
	args := map[string]any{"root": root, "query": "compact my context"}

	first := callMCPTool[selfquery.CapabilitiesResponse](t, srv, "fak_capabilities", args)
	for call := 2; call <= 5; call++ {
		got := callMCPTool[selfquery.CapabilitiesResponse](t, srv, "fak_capabilities", args)
		if !reflect.DeepEqual(got, first) {
			t.Fatalf("call %d reused response = %+v, want first response %+v", call, got, first)
		}
	}
	if executions, hits := srv.capabilitiesReuse.snapshot(); executions != 1 || hits != 4 {
		t.Fatalf("identical calls executed=%d reused=%d, want executed=1 reused=4", executions, hits)
	}
	refreshed := callMCPTool[selfquery.CapabilitiesResponse](t, srv, "fak_capabilities", args)
	if !reflect.DeepEqual(refreshed, first) {
		t.Fatalf("bounded refresh response = %+v, want first response %+v", refreshed, first)
	}
	if executions, hits := srv.capabilitiesReuse.snapshot(); executions != 2 || hits != 4 {
		t.Fatalf("call 6 executed=%d reused=%d, want bounded refresh at executed=2 reused=4", executions, hits)
	}

	// The reuse key includes every request field. A changed discovery request
	// must execute and return its own ranked response.
	changed := callMCPTool[selfquery.CapabilitiesResponse](t, srv, "fak_capabilities", map[string]any{
		"root": root, "query": "inspect policy",
	})
	if changed.Query != "inspect policy" {
		t.Fatalf("changed capabilities query = %q, want inspect policy", changed.Query)
	}
	if executions, hits := srv.capabilitiesReuse.snapshot(); executions != 3 || hits != 4 {
		t.Fatalf("changed request executed=%d reused=%d, want executed=3 reused=4", executions, hits)
	}
}
