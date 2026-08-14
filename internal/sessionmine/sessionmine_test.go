package sessionmine

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestMineNormalizesCodexAndClaudeWithoutContent(t *testing.T) {
	root := t.TempDir()
	codex := filepath.Join(root, "codex")
	claude := filepath.Join(root, "claude")
	os.MkdirAll(codex, 0755)
	os.MkdirAll(claude, 0755)
	write := func(path, body string) {
		t.Helper()
		if err := os.WriteFile(path, []byte(body), 0644); err != nil {
			t.Fatal(err)
		}
	}
	write(filepath.Join(codex, "one.jsonl"), `{"timestamp":"2026-08-14T00:00:00Z","type":"response_item","payload":{"type":"message","role":"user","content":"SECRET"}}
{"timestamp":"2026-08-14T00:00:01Z","type":"response_item","payload":{"type":"function_call","name":"shell_command","arguments":"SECRET"}}
{"timestamp":"2026-08-14T00:00:02Z","type":"response_item","payload":{"type":"function_call","name":"view_image"}}
{"timestamp":"2026-08-14T00:00:03Z","type":"response_item","payload":{"type":"function_call_output","output":"ok SECRET"}}
`)
	write(filepath.Join(claude, "two.jsonl"), `{"timestamp":"2026-08-14T00:00:00Z","type":"user","message":{"content":"SECRET"}}
{"timestamp":"2026-08-14T00:00:01Z","type":"assistant","message":{"content":[{"type":"tool_use","name":"shell-command","input":{"secret":"SECRET"}},{"type":"tool_use","name":"view_image"}]}}
{"timestamp":"2026-08-14T00:00:02Z","type":"user","message":{"content":[{"type":"tool_result","content":"SECRET"}]}}
`)
	r, err := Mine(Options{CodexRoot: codex, ClaudeRoot: claude, MinSupport: 2, Limit: 10})
	if err != nil {
		t.Fatal(err)
	}
	if r.Metrics.Sessions != 2 || r.Metrics.ToolCalls != 4 {
		t.Fatalf("metrics=%+v", r.Metrics)
	}
	if len(r.Candidates) != 1 {
		t.Fatalf("candidates=%+v", r.Candidates)
	}
	c := r.Candidates[0]
	if c.ProviderSupport != 2 || c.SessionSupport != 2 || c.SuggestedLeaf != "shell_command-view_image" {
		t.Fatalf("candidate=%+v", c)
	}
	if r.Inputs.CodexRoot != "codex-sessions" || r.Inputs.ClaudeRoot != "claude-sessions" {
		t.Fatalf("input labels leaked or changed: %+v", r.Inputs)
	}
	if r.Sessions[0].ID == filepath.Join(codex, "one.jsonl") {
		t.Fatal("session id leaked path")
	}
}

func TestMineSinceAndMalformed(t *testing.T) {
	root := t.TempDir()
	p := filepath.Join(root, "old.jsonl")
	os.WriteFile(p, []byte("not-json\n"), 0644)
	old := time.Now().Add(-48 * time.Hour)
	os.Chtimes(p, old, old)
	r, err := Mine(Options{CodexRoot: root, Since: time.Now().Add(-time.Hour)})
	if err != nil {
		t.Fatal(err)
	}
	if r.Inputs.FilesScanned != 0 {
		t.Fatalf("inputs=%+v", r.Inputs)
	}
}

func TestShellClassificationIsBoundedAndCrossProvider(t *testing.T) {
	if got := classifyCodexShell(`{"command":"git status --short C:\\secret\\repo"}`); got != "git_status" {
		t.Fatalf("codex shell = %q", got)
	}
	if got := classifyClaudeShell(map[string]any{"command": "Get-Content C:\\secret\\token.txt"}); got != "read_file" {
		t.Fatalf("claude shell = %q", got)
	}
	if got := classifyShell("curl https://secret.example/private?q=token"); got != "shell_command" {
		t.Fatalf("unknown shell leaked classification: %q", got)
	}
}

func TestPercentileUsesNearestRank(t *testing.T) {
	values := []int64{10, 20, 30, 40}
	if got := percentile(values, 50); got != 20 {
		t.Fatalf("p50 = %d", got)
	}
	if got := percentile(values, 95); got != 40 {
		t.Fatalf("p95 = %d", got)
	}
}

func TestGeneratedAtIsInputDerived(t *testing.T) {
	since := time.Date(2026, 8, 14, 1, 2, 3, 0, time.FixedZone("local", -7*60*60))
	if got := generatedAt(since); got != "2026-08-14T08:02:03Z" {
		t.Fatalf("generated_at = %q", got)
	}
	if got := generatedAt(time.Time{}); got != "all" {
		t.Fatalf("all generated_at = %q", got)
	}
}
