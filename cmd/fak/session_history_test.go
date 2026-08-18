package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/sessionmine"
)

func TestRunSessionHistoryAggregateAndDrillDown(t *testing.T) {
	root := t.TempDir()
	codex := filepath.Join(root, "codex")
	claude := filepath.Join(root, "claude")
	os.MkdirAll(codex, 0755)
	os.MkdirAll(claude, 0755)
	os.WriteFile(filepath.Join(codex, "SECRET-a.jsonl"), []byte(`{"timestamp":"2026-08-17T20:00:00Z","type":"response_item","payload":{"type":"function_call","name":"view_image","arguments":"SECRET"}}`+"\n"), 0644)
	os.WriteFile(filepath.Join(claude, "SECRET-b.jsonl"), []byte(`{"timestamp":"2026-08-17T21:00:00Z","type":"assistant","message":{"content":[{"type":"tool_use","name":"view_image","input":{"secret":"SECRET"}}]}}`+"\n"), 0644)
	index := filepath.Join(root, "index.json")
	if _, err := sessionmine.MineIndexed(sessionmine.Options{CodexRoot: codex, ClaudeRoot: claude, MinSupport: 1}, index); err != nil {
		t.Fatal(err)
	}
	var out, errOut bytes.Buffer
	if code := runVCache(&out, &errOut, []string{"session-history", "--index", index}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	var aggregate sessionmine.HistoryReport
	if err := json.Unmarshal(out.Bytes(), &aggregate); err != nil {
		t.Fatal(err)
	}
	if aggregate.Metrics.Sessions != 2 || len(aggregate.Sessions) != 2 {
		t.Fatalf("aggregate=%+v", aggregate)
	}
	sessionID := aggregate.Sessions[0].ID
	out.Reset()
	if code := runVCache(&out, &errOut, []string{"session-history", "--index", index, "--session", sessionID}); code != 0 {
		t.Fatalf("detail code=%d stderr=%s", code, errOut.String())
	}
	var detail sessionmine.HistoryReport
	if err := json.Unmarshal(out.Bytes(), &detail); err != nil {
		t.Fatal(err)
	}
	if detail.Session == nil || detail.Session.ID != sessionID || len(detail.Session.Trajectory) != 1 {
		t.Fatalf("detail=%+v", detail)
	}
	rendered := out.String()
	for _, secret := range []string{"SECRET", root, ".jsonl"} {
		if strings.Contains(rendered, secret) {
			t.Fatalf("detail leaked %q: %s", secret, rendered)
		}
	}
}

func TestRunSessionHistoryRequiresIndex(t *testing.T) {
	var out, errOut bytes.Buffer
	if code := runSessionHistory(&out, &errOut, nil); code != 2 || !strings.Contains(errOut.String(), "--index is required") {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
}
