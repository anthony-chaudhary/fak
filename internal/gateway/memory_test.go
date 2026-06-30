package gateway

import (
	"context"
	"os"
	"path/filepath"
	"testing"
)

// #1431 criterion 2: the Codex memories backend is reachable through fak_memory_run (the MCP
// surface) — backend="codex" + codex_home reads the external home as a read-only recall layer,
// without expanding the gateway trust boundary (the cells are external/untrusted, gated as
// usual by the render path).
func TestMemoryRun_codexBackendRendersFixture(t *testing.T) {
	home := t.TempDir()
	dir := filepath.Join(home, "memories")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "note.md"), []byte("# codex memory\nremember the cache prefix\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	res, err := (&Server{logf: func(string, ...any) {}}).memoryRun(context.Background(), MemoryRequest{
		Driver: "render", Intent: "codex memory", Backend: "codex", CodexHome: home,
	})
	if err != nil {
		t.Fatalf("memoryRun(codex): %v", err)
	}
	if res.Stats.Rendered == 0 {
		t.Fatalf("expected the codex fixture cell to render via the MCP path; got %d rendered", res.Stats.Rendered)
	}
}

// backend=codex with no home resolved yields an EMPTY external corpus, never an error —
// untrusted external state must never crash the algebra (fail-closed).
func TestMemoryRun_codexEmptyHomeIsEmptyNotError(t *testing.T) {
	t.Setenv("CODEX_HOME", "")
	res, err := (&Server{logf: func(string, ...any) {}}).memoryRun(context.Background(), MemoryRequest{
		Driver: "render", Backend: "codex", CodexHome: "",
	})
	if err != nil {
		t.Fatalf("an unresolved codex home must not error: %v", err)
	}
	if res.Stats.Rendered != 0 {
		t.Fatalf("an empty codex corpus should render nothing; got %d", res.Stats.Rendered)
	}
}
