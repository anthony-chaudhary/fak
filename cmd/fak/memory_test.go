package main

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/memq"
)

// fixtureCodexHome writes a minimal Codex memories home (<home>/memories/<name>.md) and
// returns the home dir — the external generated state #1431's codex backend reads.
func fixtureCodexHome(t *testing.T, name, body string) string {
	t.Helper()
	home := t.TempDir()
	dir := filepath.Join(home, "memories")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	return home
}

// #1431: `fak memory run --backend codex --codex-home <fixture>` routes (via cmdMemoryRun's
// new flag dispatch) to the read-only Codex backend, which surfaces the home's memory files
// as external/untrusted recall cells.
func TestCodexMemoryBackend_readsFixture(t *testing.T) {
	home := fixtureCodexHome(t, "note.md", "# codex memory\nremember the --build flag\n")
	backend, label := codexMemoryBackend(home, false)
	if !strings.Contains(label, "codex memories") {
		t.Fatalf("the label must mark this as the codex backend; got %q", label)
	}
	cells, err := backend.Cells(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(cells) == 0 {
		t.Fatal("expected at least one codex memory cell from the fixture home")
	}
	// The safety property: every Codex cell is stamped external/untrusted (never an
	// authoritative team rule), so the result gate treats it as generated external state.
	if got := cells[0].Attrs["provenance"]; got != memq.CodexProvenance {
		t.Fatalf("codex cell must be stamped external/untrusted provenance; got %q", got)
	}
}

// #1431 acceptance: the codex backend runs through a driver and yields renderable cells
// (gated like any other result — the cells are external/untrusted, not authoritative rules).
func TestMemoryRun_codexBackendRendersFixtureCell(t *testing.T) {
	home := fixtureCodexHome(t, "note.md", "# codex memory\nkeep the cache-prefix preserved\n")
	backend, _ := codexMemoryBackend(home, false)
	d, ok := memq.Get("render")
	if !ok {
		t.Skip("no built-in 'render' driver registered")
	}
	q := d.Build(memq.Params{Intent: "find the codex note"})
	res, err := memq.Run(context.Background(), backend, q, memq.Caps{})
	if err != nil {
		t.Fatalf("memq.Run over the codex backend: %v", err)
	}
	if res.Stats.Rendered == 0 {
		t.Fatalf("expected the codex fixture cell to render, got %d rendered", res.Stats.Rendered)
	}
}

func TestResolveCodexHome_flagThenEnv(t *testing.T) {
	home, src := resolveCodexHome("/explicit/codex", true)
	if src != "flag" || home != "/explicit/codex" {
		t.Errorf("explicit --codex-home must win as the flag source; got home=%q source=%q", home, src)
	}
	t.Setenv("CODEX_HOME", "/env/codex")
	home, src = resolveCodexHome("", true)
	if src != "env" || home != "/env/codex" {
		t.Errorf("CODEX_HOME must resolve when no flag; got home=%q source=%q", home, src)
	}
}
