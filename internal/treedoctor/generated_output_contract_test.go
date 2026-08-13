package treedoctor

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestGeneratedCaptureNamesAreIgnoredOnlyAtLegacyRoot(t *testing.T) {
	repo := findRepositoryRoot(t)
	ignore, err := os.ReadFile(filepath.Join(repo, ".gitignore"))
	if err != nil {
		t.Fatal(err)
	}
	rules := string(ignore)
	for _, name := range []string{
		"dosloop.out", "dosloop.err", "tick.json", "router.json",
		"fak_help.txt", "ps5.out", "ps5.err", "$cover",
	} {
		if !strings.Contains(rules, "/"+name) {
			t.Errorf("legacy capture %q lacks a root-anchored compatibility ignore", name)
		}
	}
	if !strings.Contains(rules, "**/.st_*") {
		t.Error("session transaction captures lack their recursive compatibility ignore")
	}
}

func TestGeneratedOutputGuideDefaultsCapturesToScratch(t *testing.T) {
	repo := findRepositoryRoot(t)
	guide, err := os.ReadFile(filepath.Join(repo, "docs", "generated-output-defaults.md"))
	if err != nil {
		t.Fatal(err)
	}
	text := string(guide)
	for _, want := range []string{
		"tree-doctor --scratch-dir fleet-loop",
		"tree-doctor --scratch-path coverage/unit.cover",
		"tree-doctor --sweep-scratch --dry-run",
	} {
		if !strings.Contains(text, want) {
			t.Errorf("guide does not establish %q", want)
		}
	}
}

func findRepositoryRoot(t *testing.T) string {
	t.Helper()
	dir, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Fatal("repository root not found")
		}
		dir = parent
	}
}
