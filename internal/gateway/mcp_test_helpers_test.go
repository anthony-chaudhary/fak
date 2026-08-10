package gateway

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// Shared MCP test helpers. Repository index tools moved to fak-dev, but runtime
// feature/capability and context tool witnesses use the same call harness.
func writeMCPIndexRepo(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	if err := os.MkdirAll(filepath.Join(root, "internal", "gateway"), 0o755); err != nil {
		t.Fatal(err)
	}
	files := map[string]string{
		"dos.toml": `[lanes.trees]
gateway = ["internal/gateway/**"] # OpenAI/MCP gateway surface
cmd = ["cmd/**"] # CLI commands
docs = ["docs/**"] # documentation
`,
		"INDEX.md": `# INDEX
- [Gateway](docs/gateway.md) - OpenAI and MCP front door.
- [Policies](POLICY.md) - capability-floor manifests.
`,
		"CLAIMS.md":       "# CLAIMS\n## Gateway\n- [SHIPPED] The `internal/gateway` MCP bridge exposes tool calls.\n- [STUB] The `internal/gateway` future registry is not complete.\n",
		"docs/gateway.md": "# Gateway\n",
		"POLICY.md":       "# Policy\n",
		".github/issue-views.json": `{
  "version": 1,
  "default": "ready-leaves",
  "limit": 300,
  "views": [
    {"slug": "ready-leaves", "title": "Ready leaves", "query": "is:open no:assignee", "note": "the default what-to-work-on surface"},
    {"slug": "epics", "title": "Epics", "query": "is:open label:epic", "note": "decompose, do not dispatch"}
  ]
}`,
	}
	for path, body := range files {
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
