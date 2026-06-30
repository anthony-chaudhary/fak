package gateway

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func writeMCPIndexRepo(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	if err := os.MkdirAll(filepath.Join(root, "internal", "gateway"), 0o755); err != nil {
		t.Fatal(err)
	}
	dosToml := `[lanes.trees]
gateway = ["internal/gateway/**"] # OpenAI/MCP gateway surface
cmd = ["cmd/**"] # CLI commands
docs = ["docs/**"] # documentation
`
	indexMd := `# INDEX
- [Gateway](docs/gateway.md) - OpenAI and MCP front door.
- [Policies](POLICY.md) - capability-floor manifests.
`
	claimsMd := `# CLAIMS
## Gateway
- [SHIPPED] The ` + "`internal/gateway`" + ` MCP bridge exposes tool calls.
- [STUB] The ` + "`internal/gateway`" + ` future registry is not complete.
`
	viewsJSON := `{
  "version": 1,
  "default": "ready-leaves",
  "limit": 300,
  "views": [
    {"slug": "ready-leaves", "title": "Ready leaves", "query": "is:open no:assignee", "note": "the default what-to-work-on surface"},
    {"slug": "epics", "title": "Epics", "query": "is:open label:epic", "note": "decompose, do not dispatch"}
  ]
}`
	for path, body := range map[string]string{
		"dos.toml":                 dosToml,
		"INDEX.md":                 indexMd,
		"CLAIMS.md":                claimsMd,
		"docs/gateway.md":          "# Gateway\n",
		"POLICY.md":                "# Policy\n",
		".github/issue-views.json": viewsJSON,
	} {
		full := filepath.Join(root, filepath.FromSlash(path))
		if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(full, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return root
}

func callMCPTool[T any](t *testing.T, srv *Server, name string, args any) T {
	t.Helper()
	params, err := json.Marshal(map[string]any{"name": name, "arguments": args})
	if err != nil {
		t.Fatal(err)
	}
	res, rerr := srv.callTool(context.Background(), params)
	if rerr != nil {
		t.Fatalf("%s rpc error: %s", name, rerr.Message)
	}
	var out T
	decodeMCPText(t, res, &out)
	return out
}

func TestMCPIndexToolsMirrorDevIndex(t *testing.T) {
	srv := newTestServer(t)
	root := writeMCPIndexRepo(t)

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
	for _, want := range []string{"fak_index_lane", "fak_index_leaves", "fak_index_docs", "fak_index_claims", "fak_index_verbs", "fak_index_work"} {
		if !names[want] {
			t.Fatalf("tools/list missing %s", want)
		}
	}

	lane := callMCPTool[IndexLaneResponse](t, srv, "fak_index_lane", map[string]any{
		"root": root,
		"paths": []string{
			"internal/gateway/mcp.go",
			"README.md",
		},
	})
	if len(lane.Results) != 2 || lane.Results[0].Lane != "gateway" || lane.Results[0].Stamp != "(fak gateway)" {
		t.Fatalf("fak_index_lane response = %+v, want gateway stamp then no-lane root file", lane)
	}
	if lane.Results[1].Lane != "" || lane.Results[1].Stamp != "" {
		t.Fatalf("root file should not imply a lane/stamp: %+v", lane.Results[1])
	}

	leaves := callMCPTool[IndexLeavesResponse](t, srv, "fak_index_leaves", map[string]any{
		"root":  root,
		"query": "gateway",
		"limit": 1,
	})
	if len(leaves.Leaves) != 1 || leaves.Leaves[0].Name != "gateway" || leaves.Leaves[0].Status.Shipped != 1 {
		t.Fatalf("fak_index_leaves response = %+v, want gateway with shipped rollup", leaves)
	}

	docs := callMCPTool[IndexDocsResponse](t, srv, "fak_index_docs", map[string]any{
		"root":  root,
		"query": "gateway front door",
	})
	if len(docs.Docs) == 0 || docs.Docs[0].Path != "docs/gateway.md" {
		t.Fatalf("fak_index_docs response = %+v, want gateway doc first", docs)
	}

	claims := callMCPTool[IndexClaimsResponse](t, srv, "fak_index_claims", map[string]any{
		"root":  root,
		"query": "gateway",
	})
	if len(claims.Claims) == 0 || claims.Claims[0].Tag != "SHIPPED" || len(claims.Claims[0].Lanes) != 1 || claims.Claims[0].Lanes[0] != "gateway" {
		t.Fatalf("fak_index_claims response = %+v, want shipped gateway claim first", claims)
	}

	verbs := callMCPTool[IndexVerbsResponse](t, srv, "fak_index_verbs", map[string]any{
		"root":  root,
		"query": "guard",
		"limit": 1,
	})
	if len(verbs.Verbs) != 1 || verbs.Verbs[0].Name != "guard" {
		t.Fatalf("fak_index_verbs response = %+v, want guard verb", verbs)
	}

	work := callMCPTool[IndexWorkResponse](t, srv, "fak_index_work", map[string]any{
		"root": root,
	})
	if work.Default != "ready-leaves" || work.Limit != 300 || len(work.Views) != 2 {
		t.Fatalf("fak_index_work response = %+v, want ready-leaves default, limit 300, 2 views", work)
	}
	if work.Views[0].Slug != "ready-leaves" || work.Views[0].Query == "" {
		t.Fatalf("fak_index_work first view = %+v, want ready-leaves with a gh query", work.Views[0])
	}
}

func TestMCPIndexDocsRequiresQuery(t *testing.T) {
	srv := newTestServer(t)
	root := writeMCPIndexRepo(t)
	params, _ := json.Marshal(map[string]any{
		"name":      "fak_index_docs",
		"arguments": map[string]any{"root": root},
	})
	if _, rerr := srv.callTool(context.Background(), params); rerr == nil || rerr.Code != rpcInvalidParams {
		t.Fatalf("fak_index_docs without query should be InvalidParams, got %+v", rerr)
	}
}
