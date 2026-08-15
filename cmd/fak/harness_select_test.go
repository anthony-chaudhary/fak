package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestHarnessSelectCLIExplainsOverlap(t *testing.T) {
	dir := t.TempDir()
	manifest := filepath.Join(dir, "harnesses.json")
	raw := `{"schema":"fak.harness-selection/v1alpha1","layers":[
		{"id":"company","scope":"company","lock":["audit"]},
		{"id":"person","scope":"person","capabilities":["terse-output"]},
		{"id":"legal","scope":"domain","when":{"tags":["legal"]},"capabilities":["citations"]},
		{"id":"coding","scope":"domain","when":{"tags":["coding"]},"capabilities":["shell"]}
	]}`
	if err := os.WriteFile(manifest, []byte(raw), 0o600); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	code := runHarness(&out, &errb, []string{"select", "--manifest", manifest, "--path", "C:/matters/7", "--tag", "legal"})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errb.String())
	}
	got := out.String()
	for _, want := range []string{`"layers": [`, `"company"`, `"person"`, `"legal"`, `"name": "audit"`, `"locked": true`, `"name": "citations"`, `"action": "skip"`, `"layer": "coding"`} {
		if !strings.Contains(got, want) {
			t.Fatalf("output missing %s:\n%s", want, got)
		}
	}
	if strings.Contains(got, `"name": "shell"`) {
		t.Fatalf("coding capability leaked into legal selection:\n%s", got)
	}
}
