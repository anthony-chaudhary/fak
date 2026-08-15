package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/harnessdiscover"
)

func TestHarnessDiscoverAndSelectCLI(t *testing.T) {
	root := t.TempDir()
	write := func(path, scope, id, capability string) {
		t.Helper()
		raw := `{"schema":"fak.harness-selection/v1alpha1","layers":[{"id":"` + id + `","scope":"` + scope + `","capabilities":["` + capability + `"]}]}`
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(raw), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	write(filepath.Join(root, "person.json"), "person", "alice", "terse-output")
	repo := filepath.Join(root, "legal", "briefs")
	write(filepath.Join(root, "legal", ".fak", "harness.json"), "repo", "legal-repo", "citations")
	if err := os.MkdirAll(repo, 0o755); err != nil {
		t.Fatal(err)
	}
	registry := harnessdiscover.Registry{Schema: harnessdiscover.Schema, DiscoverRepo: true, Sources: []harnessdiscover.Source{{ID: "alice", Scope: "person", Owner: "alice", Principals: []string{"alice"}, Root: ".", Path: "person.json", Trust: "local", RefreshPolicy: "manual"}}}
	raw, _ := json.Marshal(registry)
	regPath := filepath.Join(root, "registry.json")
	if err := os.WriteFile(regPath, raw, 0o600); err != nil {
		t.Fatal(err)
	}

	var discovered, errb bytes.Buffer
	if code := runHarness(&discovered, &errb, []string{"discover", "--registry", regPath, "--path", repo, "--principal", "alice"}); code != 0 {
		t.Fatalf("discover code=%d stderr=%s", code, errb.String())
	}
	for _, want := range []string{`"scope": "person"`, `"scope": "repo"`, `"owner": "alice"`, `"digest": "sha256:`, `"trust": "local"`} {
		if !strings.Contains(discovered.String(), want) {
			t.Fatalf("discovery missing %s:\n%s", want, discovered.String())
		}
	}

	var selected bytes.Buffer
	errb.Reset()
	if code := runHarness(&selected, &errb, []string{"select", "--discover", regPath, "--path", repo, "--principal", "alice"}); code != 0 {
		t.Fatalf("select code=%d stderr=%s", code, errb.String())
	}
	for _, want := range []string{`"alice"`, `"legal-repo"`, `"name": "terse-output"`, `"name": "citations"`} {
		if !strings.Contains(selected.String(), want) {
			t.Fatalf("selection missing %s:\n%s", want, selected.String())
		}
	}
}
