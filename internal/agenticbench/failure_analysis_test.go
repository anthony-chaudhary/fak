package agenticbench

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestFailureAnalysisCardFrozenBundleValidatesAndRenders(t *testing.T) {
	root, card := loadFailureAnalysisFixture(t)
	before := readFailureAnalysisArtifacts(t, root, card)
	if len(failureAnalysisCategories) != 4 {
		t.Fatalf("top-level failure classes = %d, want 4", len(failureAnalysisCategories))
	}

	if err := ValidateFailureAnalysisCard(root, &card); err != nil {
		t.Fatalf("valid frozen card refused: %v", err)
	}
	jsonOutput, err := RenderFailureAnalysisJSON(root, &card)
	if err != nil {
		t.Fatalf("render JSON: %v", err)
	}
	markdown, err := RenderFailureAnalysisMarkdown(root, &card)
	if err != nil {
		t.Fatalf("render Markdown: %v", err)
	}

	for _, want := range []string{`"label": "ADVISORY"`, `"model": "gpt-5.6"`, `"prompt_version": "failure-analysis-v1"`} {
		if !bytes.Contains(jsonOutput, []byte(want)) {
			t.Fatalf("JSON output missing %q:\n%s", want, jsonOutput)
		}
	}
	for _, want := range []string{
		"# ADVISORY Failure Analysis Card",
		"## 1. Conclusion",
		"## 2. Task and constraints",
		"## 3. What the agent did right",
		"## 4. What the agent did wrong",
		"## 5. Scoring",
		"execution / output_format_error",
		"interaction_log.json#/entries/1/summary",
		"events.jsonl@event:2",
		"gpt-5.6 / failure-analysis-v1 / fak-test-r1",
	} {
		if !strings.Contains(markdown, want) {
			t.Fatalf("Markdown output missing %q:\n%s", want, markdown)
		}
	}

	after := readFailureAnalysisArtifacts(t, root, card)
	for path, want := range before {
		if !bytes.Equal(after[path], want) {
			t.Fatalf("advisory validation changed official run artifact %q", path)
		}
	}
}

func TestFailureAnalysisCardAdversarialFixturesFailClosed(t *testing.T) {
	root, original := loadFailureAnalysisFixture(t)
	data, err := os.ReadFile(filepath.Join("testdata", "failure-analysis", "invalid-cases.json"))
	if err != nil {
		t.Fatal(err)
	}
	var cases []struct {
		Name      string `json:"name"`
		Mutation  string `json:"mutation"`
		WantError string `json:"want_error"`
	}
	if err := json.Unmarshal(data, &cases); err != nil {
		t.Fatal(err)
	}
	if len(cases) < 4 {
		t.Fatalf("adversarial fixture count = %d, want at least 4", len(cases))
	}

	for _, tc := range cases {
		t.Run(tc.Name, func(t *testing.T) {
			card := cloneFailureAnalysisCard(t, original)
			switch tc.Mutation {
			case "unsupported_claim":
				card.Strengths[0].Evidence = nil
			case "escaping_path":
				card.Failures.Observations[0].Evidence[0].ArtifactPath = "../outside.json"
			case "stale_hash":
				card.Artifacts[0].SHA256 = "sha256:0000000000000000000000000000000000000000000000000000000000000000"
			case "unlabeled_inference":
				card.Failures.Inferences[0].Kind = ""
			case "unknown_category":
				card.Classification.Class = "knowledge"
			default:
				t.Fatalf("unknown fixture mutation %q", tc.Mutation)
			}
			err := ValidateFailureAnalysisCard(root, &card)
			if err == nil || !strings.Contains(err.Error(), tc.WantError) {
				t.Fatalf("err = %v, want substring %q", err, tc.WantError)
			}
		})
	}
}

func loadFailureAnalysisFixture(t *testing.T) (string, FailureAnalysisCard) {
	t.Helper()
	fixture := filepath.Join("testdata", "failure-analysis")
	data, err := os.ReadFile(filepath.Join(fixture, "card.json"))
	if err != nil {
		t.Fatal(err)
	}
	card, err := DecodeFailureAnalysisCard(data)
	if err != nil {
		t.Fatal(err)
	}
	return filepath.Join(fixture, "run"), card
}

func cloneFailureAnalysisCard(t *testing.T, card FailureAnalysisCard) FailureAnalysisCard {
	t.Helper()
	data, err := json.Marshal(card)
	if err != nil {
		t.Fatal(err)
	}
	var cloned FailureAnalysisCard
	if err := json.Unmarshal(data, &cloned); err != nil {
		t.Fatal(err)
	}
	return cloned
}

func readFailureAnalysisArtifacts(t *testing.T, root string, card FailureAnalysisCard) map[string][]byte {
	t.Helper()
	artifacts := make(map[string][]byte, len(card.Artifacts))
	for _, artifact := range card.Artifacts {
		data, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(artifact.ArtifactPath)))
		if err != nil {
			t.Fatal(err)
		}
		artifacts[artifact.ArtifactPath] = data
	}
	return artifacts
}
