package framevisibility

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

func createBenchmarkCorpus(tb testing.TB, homes, sessionsPerHome, subagentsPerSession int) (string, string) {
	tb.Helper()
	root := tb.TempDir()
	project := "C--work-fak"

	for h := 0; h < homes; h++ {
		homeDir := filepath.Join(root, fmt.Sprintf(".claude-home-%d", h), "projects", project)
		for s := 0; s < sessionsPerHome; s++ {
			sessionID := fmt.Sprintf("session-%d-%d", h, s)
			sessionDir := filepath.Join(homeDir, sessionID)
			subagentsDir := filepath.Join(sessionDir, "subagents")
			if err := os.MkdirAll(subagentsDir, 0o755); err != nil {
				tb.Fatal(err)
			}

			masterPath := filepath.Join(homeDir, sessionID+".jsonl")
			masterContent := []byte(
				`{"type":"user","message":{"content":"please inspect the repository"}}` + "\n" +
					`{"type":"assistant","message":{"content":[{"type":"tool_use","name":"Task","input":{"prompt":"run subagent task"}}]}}` + "\n" +
					`{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"task-1","content":"subagent completed"}]}}` + "\n" +
					`{"type":"assistant","message":{"content":[{"type":"tool_use","name":"Bash","input":{"command":"git commit -m 'update'"}}]}}` + "\n" +
					`{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"bash-1","content":"[main 1234567] update"}]}}` + "\n" +
					`{"type":"assistant","message":{"content":[{"type":"tool_use","name":"Bash","input":{"command":"git status"}}]}}` + "\n" +
					`{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"bash-2","content":"error: failed","is_error":true}]}}` + "\n" +
					`{"type":"assistant","message":{"content":[{"type":"tool_use","name":"Edit","input":{"file_path":"main.go"}}]}}` + "\n" +
					`{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"edit-1","content":"ok"}]}}` + "\n" +
					`{"type":"assistant","message":{"content":[{"type":"text","text":"all tasks completed successfully"}]}}` + "\n",
			)
			if err := os.WriteFile(masterPath, masterContent, 0o600); err != nil {
				tb.Fatal(err)
			}

			for a := 0; a < subagentsPerSession; a++ {
				agentPath := filepath.Join(subagentsDir, fmt.Sprintf("agent-%d.jsonl", a))
				agentContent := []byte(
					`{"type":"user","message":{"content":"explore directory structure"}}` + "\n" +
						`{"type":"assistant","message":{"content":[{"type":"tool_use","name":"Read","input":{"file_path":"README.md"}}]}}` + "\n" +
						`{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"read-1","content":"# Project"}]}}` + "\n" +
						`{"type":"assistant","message":{"content":[{"type":"tool_use","name":"Grep","input":{"pattern":"TODO"}}]}}` + "\n" +
						`{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"grep-1","content":"none"}]}}` + "\n" +
						`{"type":"assistant","message":{"content":[{"type":"text","text":"investigation finished"}]}}` + "\n",
				)
				if err := os.WriteFile(agentPath, agentContent, 0o600); err != nil {
					tb.Fatal(err)
				}
			}
		}
	}
	return root, project
}

func BenchmarkFrameVisibility(b *testing.B) {
	root, project := createBenchmarkCorpus(b, 2, 2, 2)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		counts, rows, err := Fold(root, project)
		if err != nil {
			b.Fatalf("Fold failed: %v", err)
		}
		if counts.Sessions != 4 || len(rows) != 4 {
			b.Fatalf("unexpected counts: sessions=%d rows=%d", counts.Sessions, len(rows))
		}
	}
}

func TestBenchmarkFrameVisibility(t *testing.T) {
	root, project := createBenchmarkCorpus(t, 2, 2, 2)
	counts, rows, err := Fold(root, project)
	if err != nil {
		t.Fatalf("Fold failed: %v", err)
	}
	if counts.Homes != 2 || counts.Sessions != 4 || counts.SubFiles != 8 {
		t.Fatalf("unexpected structure counts: homes=%d sessions=%d subfiles=%d", counts.Homes, counts.Sessions, counts.SubFiles)
	}
	if len(rows) != 4 {
		t.Fatalf("expected 4 session rows, got %d", len(rows))
	}
	if counts.MasterEvents != 36 || counts.MasterRelevant != 20 {
		t.Fatalf("master metrics: events=%d relevant=%d, want 36/20", counts.MasterEvents, counts.MasterRelevant)
	}
	if counts.SubEvents != 48 || counts.SubVisible != 16 {
		t.Fatalf("subagent metrics: events=%d visible=%d, want 48/16", counts.SubEvents, counts.SubVisible)
	}
}
