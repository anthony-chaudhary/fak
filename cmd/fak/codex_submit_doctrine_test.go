package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestCodexUserPromptSubmitDoctrineIsDiscoverable(t *testing.T) {
	root := filepath.Join("..", "..")
	read := func(rel string) string {
		t.Helper()
		raw, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(rel)))
		if err != nil {
			t.Fatalf("read %s: %v", rel, err)
		}
		return string(raw)
	}

	doc := read("docs/integrations/openai-codex.md")
	for _, want := range []string{
		"### UserPromptSubmit modes",
		"The capability floor is explicit:",
		"fak sessions codex-hook-install",
		"fak sessions codex-hook-install --hardened",
		"fak sessions codex-loop-hook --hardened",
	} {
		if !strings.Contains(doc, want) {
			t.Errorf("Codex doctrine is missing %q", want)
		}
	}

	const link = "integrations/openai-codex.md#userpromptsubmit-modes"
	if !strings.Contains(read("docs/index.md"), link) {
		t.Errorf("docs/index.md does not link the UserPromptSubmit doctrine")
	}
	if !strings.Contains(read("llms.txt"), "docs/"+link) {
		t.Errorf("llms.txt does not link the UserPromptSubmit doctrine")
	}
}
