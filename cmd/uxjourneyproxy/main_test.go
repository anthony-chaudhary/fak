package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func loadTestCorpus(t *testing.T) (corpus, []byte) {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata", "corpus.json"))
	if err != nil {
		t.Fatal(err)
	}
	var c corpus
	if err := json.Unmarshal(raw, &c); err != nil {
		t.Fatal(err)
	}
	return c, raw
}

func TestCorpusProducesCompleteDeterministicWitness(t *testing.T) {
	c, raw := loadTestCorpus(t)
	first, err := score(c, raw)
	if err != nil {
		t.Fatal(err)
	}
	second, err := score(c, raw)
	if err != nil {
		t.Fatal(err)
	}
	a, _ := json.Marshal(first)
	b, _ := json.Marshal(second)
	if !bytes.Equal(a, b) {
		t.Fatal("same corpus produced different output")
	}
	if got, want := len(first.Scripts), 42; got != want {
		t.Fatalf("scripts = %d, want %d", got, want)
	}
	if first.Vocabulary.Total < 12 || first.Vocabulary.Correct != first.Vocabulary.Total {
		t.Fatalf("vocabulary = %d/%d, want all cases correct", first.Vocabulary.Correct, first.Vocabulary.Total)
	}
}

func TestGoldenBaselineMatchesScorer(t *testing.T) {
	c, raw := loadTestCorpus(t)
	got, err := score(c, raw)
	if err != nil {
		t.Fatal(err)
	}
	gotJSON, err := json.MarshalIndent(got, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	gotJSON = append(gotJSON, '\n')
	want, err := os.ReadFile(filepath.Join("..", "..", "docs", "research", "managed-context-cognitive-load-baseline-2026-08-13.json"))
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(gotJSON, want) {
		t.Fatal("golden baseline drifted; regenerate it with uxjourneyproxy")
	}
}

func TestCorpusRejectsMissingPersonaAlternativeCell(t *testing.T) {
	c, _ := loadTestCorpus(t)
	c.Journeys[0].Scripts = c.Journeys[0].Scripts[:5]
	if err := validateCorpus(c); err == nil {
		t.Fatal("missing persona/alternative cell accepted")
	}
}

func TestVocabularyClassifierRejectsUnstatedMeaning(t *testing.T) {
	if got := classifyVocabulary("a familiar everyday noun without a product contrast"); got != "unknown" {
		t.Fatalf("classifier = %q, want unknown", got)
	}
}
