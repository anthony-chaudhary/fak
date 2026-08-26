package study

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func fixtureRecord() Record {
	return Record{Schema: Schema, Title: "borrow bounded lexical search", Observed: "2026-08-26", Sources: []Source{{URL: "https://example.test/repo", Revision: "abc123"}}, Observations: []string{"source pins candidate decisions"}, Candidates: []Candidate{{Effect: "reuse prior studies", Status: "ABSENT", Disposition: "DEFAULT", Evidence: []string{"internal/study/store.go"}, Issue: "https://github.com/anthony-chaudhary/fak/issues/8613"}}}
}

func TestReceiptPersistsAcrossFreshSearch(t *testing.T) {
	store := filepath.Join(t.TempDir(), "records")
	first, err := Add(store, fixtureRecord())
	if err != nil {
		t.Fatal(err)
	}
	if !first.Created {
		t.Fatal("first add was not created")
	}
	again, err := Add(store, fixtureRecord())
	if err != nil {
		t.Fatal(err)
	}
	if again.ID != first.ID || again.Created {
		t.Fatalf("idempotence failed: %#v %#v", first, again)
	}
	got, err := Search(store, "reuse prior", 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0].ID != first.ID || got[0].Record.Candidates[0].Issue == "" {
		t.Fatalf("fresh search lost receipt linkage: %#v", got)
	}
}

func TestUnavailableStorageIsExplicit(t *testing.T) {
	parent := t.TempDir()
	blocker := filepath.Join(parent, "file")
	if err := os.WriteFile(blocker, []byte("x"), 0600); err != nil {
		t.Fatal(err)
	}
	_, err := Add(filepath.Join(blocker, "records"), fixtureRecord())
	if err == nil || !strings.Contains(err.Error(), "storage unavailable") {
		t.Fatalf("expected explicit storage failure, got %v", err)
	}
}
