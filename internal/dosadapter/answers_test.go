package dosadapter

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestSearchAnswers_FallbackWhenNoDocsTree(t *testing.T) {
	// Create an isolated temporary directory with zero docs/ or answers/ files.
	tempDir := t.TempDir()

	resp := SearchAnswers(tempDir, "how do I verify an AI agent actually did the work", 3)

	if resp.Count <= 0 {
		t.Fatalf("expected Count > 0 with fallback bundled index, got %d (note: %q)", resp.Count, resp.Note)
	}
	if len(resp.Results) == 0 {
		t.Fatalf("expected non-empty Results slice, got 0 items")
	}
	if resp.Note != "" {
		t.Fatalf("expected empty Note on successful match, got %q", resp.Note)
	}

	top := resp.Results[0]
	expectedSlug := "how-to-verify-an-ai-agent-actually-did-the-work"
	if top.Slug != expectedSlug {
		t.Errorf("expected top slug %q, got %q", expectedSlug, top.Slug)
	}
	if top.Score <= 0.0 {
		t.Errorf("expected score > 0.0, got %f", top.Score)
	}
	if top.Answer == "" {
		t.Errorf("expected non-empty Answer text")
	}
	if top.URL == "" {
		t.Errorf("expected non-empty URL")
	}
}

func TestSearchAnswers_ColloquialQueries(t *testing.T) {
	tempDir := t.TempDir()

	cases := []struct {
		query        string
		expectedSlug string
	}{
		{
			query:        "how do I stop two AI agents from overwriting each other's files",
			expectedSlug: "how-to-stop-two-ai-agents-overwriting-each-other",
		},
		{
			query:        "my AI writes tests that pass but assert nothing",
			expectedSlug: "ai-generated-tests-that-pass-but-test-nothing",
		},
		{
			query:        "is a cited legal case actually real",
			expectedSlug: "how-to-verify-a-cited-legal-case-exists",
		},
		{
			query:        "does my agent's commit message match what it actually changed",
			expectedSlug: "does-the-commit-message-match-what-changed",
		},
	}

	for _, tc := range cases {
		t.Run(tc.expectedSlug, func(t *testing.T) {
			resp := SearchAnswers(tempDir, tc.query, 3)
			if resp.Count == 0 || len(resp.Results) == 0 {
				t.Fatalf("no results returned for query %q", tc.query)
			}
			top := resp.Results[0]
			if top.Slug != tc.expectedSlug {
				t.Errorf("for query %q: expected top slug %q, got %q", tc.query, tc.expectedSlug, top.Slug)
			}
		})
	}
}

func TestSearchAnswers_RespectsK(t *testing.T) {
	tempDir := t.TempDir()
	resp := SearchAnswers(tempDir, "agent verification", 2)
	if len(resp.Results) > 2 {
		t.Errorf("expected at most 2 results when k=2, got %d", len(resp.Results))
	}
	if resp.Count != len(resp.Results) {
		t.Errorf("count %d does not match results length %d", resp.Count, len(resp.Results))
	}
}

func TestSearchAnswers_EmptyOrNonsense(t *testing.T) {
	tempDir := t.TempDir()

	for _, q := range []string{"", "   ", "zzzxqq nonsense token soup 12345"} {
		resp := SearchAnswers(tempDir, q, 3)
		if resp.Query != q {
			t.Errorf("expected Query %q, got %q", q, resp.Query)
		}
		if resp.Count != len(resp.Results) {
			t.Errorf("for query %q: count %d != len(results) %d", q, resp.Count, len(resp.Results))
		}
		if resp.Count == 0 && resp.Note == "" {
			t.Errorf("for empty query %q: expected non-empty explanatory Note", q)
		}
	}
}

func TestLoadAnswers_WorkspacePrecedence(t *testing.T) {
	tempDir := t.TempDir()
	answersDir := filepath.Join(tempDir, "docs", "answers")
	if err := os.MkdirAll(answersDir, 0755); err != nil {
		t.Fatalf("failed to create mock answers dir: %v", err)
	}

	customRow := AnswerRow{
		Slug:     "custom-workspace-answer",
		Question: "How do custom workspace answers take precedence?",
		Answer:   "Local workspace answers always outrank bundled fallbacks.",
		Path:     "docs/answers/custom.md",
		URL:      "https://example.com/custom.md",
		Queries:  []string{"custom workspace test"},
	}

	data, err := json.Marshal([]AnswerRow{customRow})
	if err != nil {
		t.Fatalf("failed to marshal custom row: %v", err)
	}

	indexPath := filepath.Join(answersDir, "index.jsonl")
	if err := os.WriteFile(indexPath, data, 0644); err != nil {
		t.Fatalf("failed to write custom index: %v", err)
	}

	rows := LoadAnswers(tempDir)
	if len(rows) != 1 {
		t.Fatalf("expected 1 custom row from workspace, got %d", len(rows))
	}
	if rows[0].Slug != "custom-workspace-answer" {
		t.Errorf("expected custom slug, got %q", rows[0].Slug)
	}
}
