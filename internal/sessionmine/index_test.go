package sessionmine

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestMineIndexedReusesUnchangedFilesAndEmitsCrossingOnce(t *testing.T) {
	root := t.TempDir()
	codex := filepath.Join(root, "codex")
	claude := filepath.Join(root, "claude")
	os.MkdirAll(codex, 0755)
	os.MkdirAll(claude, 0755)
	codexBody := `{"timestamp":"2026-08-14T00:00:00Z","type":"response_item","payload":{"type":"function_call","name":"view_image","arguments":"SECRET"}}` + "\n" + `{"timestamp":"2026-08-14T00:00:01Z","type":"response_item","payload":{"type":"function_call","name":"view_image"}}` + "\n"
	claudeBody := `{"timestamp":"2026-08-14T00:00:00Z","type":"assistant","message":{"content":[{"type":"tool_use","name":"view_image","input":{"secret":"SECRET"}},{"type":"tool_use","name":"view_image"}]}}` + "\n"
	os.WriteFile(filepath.Join(codex, "one.jsonl"), []byte(codexBody), 0644)
	os.WriteFile(filepath.Join(claude, "two.jsonl"), []byte(claudeBody), 0644)
	index := filepath.Join(root, "state", "index.json")
	first, err := MineIndexed(Options{CodexRoot: codex, ClaudeRoot: claude, MinSupport: 2, Limit: 10}, index)
	if err != nil {
		t.Fatal(err)
	}
	if first.ParsedFiles != 2 || first.ReusedFiles != 0 || len(first.NewCandidates) != 1 {
		t.Fatalf("first=%+v", first)
	}
	second, err := MineIndexed(Options{CodexRoot: codex, ClaudeRoot: claude, MinSupport: 2, Limit: 10}, index)
	if err != nil {
		t.Fatal(err)
	}
	if second.ParsedFiles != 0 || second.ReusedFiles != 2 || len(second.NewCandidates) != 0 {
		t.Fatalf("second=%+v", second)
	}
	data := string(mustRead(t, index))
	for _, secret := range []string{"SECRET", root, "one.jsonl", "two.jsonl"} {
		if strings.Contains(data, secret) {
			t.Fatalf("index leaked %q: %s", secret, data)
		}
	}
	var state IndexState
	if err := json.Unmarshal([]byte(data), &state); err != nil {
		t.Fatal(err)
	}
	if state.Schema != indexSchema || len(state.Files) != 2 || len(state.Seen) != 1 {
		t.Fatalf("state=%+v", state)
	}
}

func TestMineIndexedRestartParsesOnlyChangedSource(t *testing.T) {
	root := t.TempDir()
	os.MkdirAll(filepath.Join(root, "codex"), 0755)
	source := filepath.Join(root, "codex", "one.jsonl")
	index := filepath.Join(root, "index.json")
	os.WriteFile(source, []byte(`{"timestamp":"2026-08-14T00:00:00Z","type":"response_item","payload":{"type":"function_call","name":"view_image"}}`+"\n"), 0644)
	if _, err := MineIndexed(Options{CodexRoot: filepath.Dir(source)}, index); err != nil {
		t.Fatal(err)
	}
	f, err := os.OpenFile(source, os.O_APPEND|os.O_WRONLY, 0644)
	if err != nil {
		t.Fatal(err)
	}
	f.WriteString(`{"timestamp":"2026-08-14T00:00:01Z","type":"response_item","payload":{"type":"function_call_output","output":"ok"}}` + "\n")
	f.Close()
	got, err := MineIndexed(Options{CodexRoot: filepath.Dir(source)}, index)
	if err != nil {
		t.Fatal(err)
	}
	if got.ParsedFiles != 1 || got.ReusedFiles != 0 || got.Report.Metrics.ToolResults != 1 {
		t.Fatalf("restart=%+v", got)
	}
}

func TestMineIndexedRejectsUnknownSchema(t *testing.T) {
	p := filepath.Join(t.TempDir(), "index.json")
	os.WriteFile(p, []byte(`{"schema":"future/9"}`), 0600)
	if _, err := MineIndexed(Options{}, p); err == nil {
		t.Fatal("expected schema rejection")
	}
}
func mustRead(t *testing.T, path string) []byte {
	t.Helper()
	b, e := os.ReadFile(path)
	if e != nil {
		t.Fatal(e)
	}
	return b
}

func TestMineIndexedUsesNewestSessionEndAsWatermark(t *testing.T) {
	root := t.TempDir()
	codex := filepath.Join(root, "codex")
	if err := os.MkdirAll(codex, 0755); err != nil {
		t.Fatal(err)
	}
	body := `{"timestamp":"2026-08-16T01:00:00Z","type":"response_item","payload":{"type":"function_call","name":"view_image"}}` + "\n" + `{"timestamp":"2026-08-17T03:00:00Z","type":"response_item","payload":{"type":"function_call_output","output":"ok"}}` + "\n"
	if err := os.WriteFile(filepath.Join(codex, "one.jsonl"), []byte(body), 0644); err != nil {
		t.Fatal(err)
	}
	index := filepath.Join(root, "index.json")
	if _, err := MineIndexed(Options{CodexRoot: codex}, index); err != nil {
		t.Fatal(err)
	}
	state, err := LoadIndex(index)
	if err != nil {
		t.Fatal(err)
	}
	if state.UpdatedAt != "2026-08-17T03:00:00Z" {
		t.Fatalf("watermark=%q", state.UpdatedAt)
	}
	if _, err := MineIndexed(Options{CodexRoot: codex}, index); err != nil {
		t.Fatal(err)
	}
	reused, err := LoadIndex(index)
	if err != nil {
		t.Fatal(err)
	}
	if reused.UpdatedAt != state.UpdatedAt {
		t.Fatalf("reused watermark=%q want %q", reused.UpdatedAt, state.UpdatedAt)
	}
}
