package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
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

type allowCapabilitiesAdj struct{}

func (allowCapabilitiesAdj) Caps() []abi.Capability { return nil }
func (allowCapabilitiesAdj) Adjudicate(ctx context.Context, c *abi.ToolCall) abi.Verdict {
	if isCapabilitiesTool(c.Tool) {
		return abi.Verdict{Kind: abi.VerdictAllow, By: "test-policy"}
	}
	return abi.Verdict{Kind: abi.VerdictDefer, By: "test-policy"}
}

func TestSyscallPreservesCapabilitiesCards(t *testing.T) {
	srv := newTestServer(t)
	abi.RegisterAdjudicator(1, allowCapabilitiesAdj{})
	root := writeMCPIndexRepo(t)

	// Call fak_capabilities through guarded syscall
	args, _ := json.Marshal(map[string]any{"root": root, "query": "inspect policy"})
	wv, env, err := srv.syscall(context.Background(), "fak_capabilities", string(args), false, "", "trace-test-cap")
	if err != nil {
		t.Fatalf("syscall error: %v", err)
	}
	if wv.Kind != "ALLOW" {
		t.Fatalf("expected ALLOW verdict, got %q", wv.Kind)
	}
	if env == nil || env.Status != "OK" {
		t.Fatalf("expected env.Status = OK, got %+v", env)
	}
	if env.Meta["engine"] != "fak-mcp" {
		t.Fatalf("expected engine=fak-mcp, got %q", env.Meta["engine"])
	}

	var resp selfquery.CapabilitiesResponse
	if err := json.Unmarshal([]byte(env.Content), &resp); err != nil {
		t.Fatalf("failed to parse capability cards from result content: %v (raw content: %s)", err, env.Content)
	}
	if len(resp.Cards) == 0 {
		t.Fatalf("expected capability cards in result, got 0 cards (response: %+v)", resp)
	}
	if strings.Contains(env.Content, "generated_tokens") {
		t.Fatal("result content should not contain model-generation tokens")
	}
}

// TestMCPCompactDiscoveryRoundTrip witnesses the bounded task-scoped discovery round trip (#11552):
// 1. Initial name-only discovery reveals the compact query path (fak_capabilities) without dumping schemas.
// 2. Client requests one intent-matching card with actionable fields via query/limit.
// 3. Client retrieves only its selected tool schema with exact selected-tool lookup.
// 4. Client verifies bounded non-paged responses and exact equivalence to the full catalog schema.
func TestMCPCompactDiscoveryRoundTrip(t *testing.T) {
	srv := newTestServerWithConfig(t, Config{
		EngineID:        "test",
		DisableMCPDefer: true,
	})
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()
	root := writeMCPIndexRepo(t)

	allExposed := srv.exposedToolDescriptors()
	if len(allExposed) < 15 {
		t.Fatalf("expected large catalog with at least 15 tools, got %d", len(allExposed))
	}

	// 1. Initial name-only discovery reveals compact query path without requiring full catalog output.
	nameOnlyReq, _ := json.Marshal(map[string]any{
		"jsonrpc": "2.0",
		"id":      1,
		"method":  "tools/call",
		"params": map[string]any{
			"name": "fak_tools_search",
			"arguments": map[string]any{
				"detail_level": "name",
			},
		},
	})
	res1, err := http.Post(ts.URL+"/mcp", "application/json", bytes.NewReader(nameOnlyReq))
	if err != nil {
		t.Fatalf("name-only discovery POST error: %v", err)
	}
	defer res1.Body.Close()
	if res1.StatusCode != http.StatusOK {
		t.Fatalf("name-only status = %d, want 200", res1.StatusCode)
	}
	var rpcResp1 rpcResponse
	if err := json.NewDecoder(res1.Body).Decode(&rpcResp1); err != nil {
		t.Fatalf("decode name-only rpc response: %v", err)
	}
	if rpcResp1.Error != nil {
		t.Fatalf("name-only RPC error: %+v", rpcResp1.Error)
	}

	var searchResp ToolsSearchResponse
	decodeMCPResult(t, rpcResp1.Result, &searchResp)

	// Assert compact query path is revealed
	if searchResp.CompactQueryPath == "" {
		t.Fatal("name-only discovery missing compact_query_path in response")
	}
	queryTool := searchResp.CompactQueryPath
	if queryTool != CompactDiscoveryToolName {
		t.Fatalf("compact_query_path = %q, want %q", queryTool, CompactDiscoveryToolName)
	}

	// Assert bounded, non-paged, schema-free name list
	foundCap := false
	for _, td := range searchResp.Tools {
		name, _ := td["name"].(string)
		if name == "" {
			t.Fatalf("tool in name-only response has empty name: %+v", td)
		}
		if name == queryTool {
			foundCap = true
			if path, _ := td["compact_query_path"].(string); path != CompactDiscoveryToolName {
				t.Fatalf("tool %q missing compact_query_path attribute: %+v", name, td)
			}
		}
		if _, hasDesc := td["description"]; hasDesc {
			t.Fatalf("name-only response leaked description for tool %q", name)
		}
		if _, hasSchema := td["inputSchema"]; hasSchema {
			t.Fatalf("name-only response leaked inputSchema for tool %q", name)
		}
	}
	if !foundCap {
		t.Fatalf("name-only discovery tools missing %q", queryTool)
	}

	// 2. Request one intent-matching card via compact query path
	const intent = "compact my context"
	capReq, _ := json.Marshal(map[string]any{
		"jsonrpc": "2.0",
		"id":      2,
		"method":  "tools/call",
		"params": map[string]any{
			"name": queryTool,
			"arguments": map[string]any{
				"root":  root,
				"query": intent,
				"limit": 1,
			},
		},
	})
	res2, err := http.Post(ts.URL+"/mcp", "application/json", bytes.NewReader(capReq))
	if err != nil {
		t.Fatalf("capability card request error: %v", err)
	}
	defer res2.Body.Close()
	if res2.StatusCode != http.StatusOK {
		t.Fatalf("capability card request status = %d, want 200", res2.StatusCode)
	}
	var rpcResp2 rpcResponse
	if err := json.NewDecoder(res2.Body).Decode(&rpcResp2); err != nil {
		t.Fatalf("decode capability card rpc response: %v", err)
	}
	if rpcResp2.Error != nil {
		t.Fatalf("capability card RPC error: %+v", rpcResp2.Error)
	}

	var capResp selfquery.CapabilitiesResponse
	decodeMCPResult(t, rpcResp2.Result, &capResp)

	if len(capResp.Cards) != 1 {
		t.Fatalf("expected exactly 1 card with limit=1, got %d", len(capResp.Cards))
	}
	matchedCard := capResp.Cards[0]
	if matchedCard.Name != "memory-driver:compact" {
		t.Fatalf("matched card name = %q, want memory-driver:compact", matchedCard.Name)
	}

	// Assert actionable fields on matched card
	selectedTool := matchedCard.Request.MCPTool
	if selectedTool == "" {
		t.Fatalf("matched card missing actionable MCPTool in request: %+v", matchedCard.Request)
	}
	if matchedCard.Request.Arguments == nil || len(matchedCard.Request.Arguments) == 0 {
		t.Fatalf("matched card missing actionable arguments in request: %+v", matchedCard.Request)
	}
	if matchedCard.Request.Route != "mcp/tools-call" {
		t.Fatalf("matched card request route = %q, want mcp/tools-call", matchedCard.Request.Route)
	}
	if len(matchedCard.Request.Command) == 0 {
		t.Fatalf("matched card request command is empty: %+v", matchedCard.Request)
	}

	// 3. Retrieve only its selected schema via exact selected-tool schema lookup
	schemaReq, _ := json.Marshal(map[string]any{
		"jsonrpc": "2.0",
		"id":      3,
		"method":  "tools/call",
		"params": map[string]any{
			"name": "fak_tools_search",
			"arguments": map[string]any{
				"tool":         selectedTool,
				"detail_level": "full",
			},
		},
	})
	res3, err := http.Post(ts.URL+"/mcp", "application/json", bytes.NewReader(schemaReq))
	if err != nil {
		t.Fatalf("schema lookup error: %v", err)
	}
	defer res3.Body.Close()
	if res3.StatusCode != http.StatusOK {
		t.Fatalf("schema lookup status = %d, want 200", res3.StatusCode)
	}
	var rpcResp3 rpcResponse
	if err := json.NewDecoder(res3.Body).Decode(&rpcResp3); err != nil {
		t.Fatalf("decode schema lookup rpc response: %v", err)
	}
	if rpcResp3.Error != nil {
		t.Fatalf("schema lookup RPC error: %+v", rpcResp3.Error)
	}

	var schemaResp ToolsSearchResponse
	decodeMCPResult(t, rpcResp3.Result, &schemaResp)

	// Assert bounded non-paged response returning ONLY the selected tool
	if len(schemaResp.Tools) != 1 {
		t.Fatalf("expected exactly 1 tool schema returned, got %d", len(schemaResp.Tools))
	}
	retrievedTool := schemaResp.Tools[0]
	if retrievedTool["name"] != selectedTool {
		t.Fatalf("retrieved tool = %v, want %q", retrievedTool["name"], selectedTool)
	}
	retrievedSchema, ok := retrievedTool["inputSchema"]
	if !ok || retrievedSchema == nil {
		t.Fatalf("retrieved tool %q missing inputSchema: %+v", selectedTool, retrievedTool)
	}

	// 4. Assert equal selected-tool correctness against the large catalog
	var catalogSchema any
	for _, td := range allExposed {
		if td["name"] == selectedTool {
			catalogSchema = td["inputSchema"]
			break
		}
	}
	if catalogSchema == nil {
		t.Fatalf("selected tool %q not found in server catalog", selectedTool)
	}
	retrievedJSON, err := json.Marshal(retrievedSchema)
	if err != nil {
		t.Fatal(err)
	}
	var gotSchema, wantSchema any
	if err := json.Unmarshal(retrievedJSON, &gotSchema); err != nil {
		t.Fatal(err)
	}
	if rawBytes, ok := catalogSchema.(json.RawMessage); ok {
		if err := json.Unmarshal(rawBytes, &wantSchema); err != nil {
			t.Fatal(err)
		}
	} else {
		wantBytes, _ := json.Marshal(catalogSchema)
		if err := json.Unmarshal(wantBytes, &wantSchema); err != nil {
			t.Fatal(err)
		}
	}
	if !reflect.DeepEqual(gotSchema, wantSchema) {
		t.Fatalf("retrieved schema does not match catalog schema:\n got %#v\nwant %#v", gotSchema, wantSchema)
	}

	// 5. Verify explicit full-detail compatibility (returning all tools when no exact tool is requested)
	fullReq, _ := json.Marshal(map[string]any{
		"jsonrpc": "2.0",
		"id":      4,
		"method":  "tools/call",
		"params": map[string]any{
			"name": "fak_tools_search",
			"arguments": map[string]any{
				"detail_level": "full",
			},
		},
	})
	res4, err := http.Post(ts.URL+"/mcp", "application/json", bytes.NewReader(fullReq))
	if err != nil {
		t.Fatalf("full detail request error: %v", err)
	}
	defer res4.Body.Close()
	var rpcResp4 rpcResponse
	if err := json.NewDecoder(res4.Body).Decode(&rpcResp4); err != nil {
		t.Fatalf("decode full detail rpc response: %v", err)
	}
	if rpcResp4.Error != nil {
		t.Fatalf("full detail RPC error: %+v", rpcResp4.Error)
	}
	var fullResp ToolsSearchResponse
	decodeMCPResult(t, rpcResp4.Result, &fullResp)
	if len(fullResp.Tools) != len(allExposed) {
		t.Fatalf("full detail returned %d tools, want %d", len(fullResp.Tools), len(allExposed))
	}
}
