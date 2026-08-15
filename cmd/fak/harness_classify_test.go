package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestHarnessClassifyCLIExplicitAmbiguousAndRemembered(t *testing.T) {
	var out, errb bytes.Buffer
	if code := runHarness(&out, &errb, []string{"classify", "--path", "brief.go", "--task", "implement contract brief", "--task-domain", "legal"}); code != 0 {
		t.Fatalf("explicit code=%d stderr=%s", code, errb.String())
	}
	if !strings.Contains(out.String(), `"domain": "legal"`) || !strings.Contains(out.String(), `"source": "task-declaration"`) {
		t.Fatalf("explicit output=%s", out.String())
	}

	out.Reset()
	errb.Reset()
	if code := runHarness(&out, &errb, []string{"classify", "--path", "brief.go", "--task", "implement contract brief"}); code != 3 {
		t.Fatalf("ambiguous code=%d stderr=%s output=%s", code, errb.String(), out.String())
	}
	if !strings.Contains(out.String(), `"needs_decision": true`) || !strings.Contains(out.String(), `"choices"`) {
		t.Fatalf("ambiguous output=%s", out.String())
	}

	choice := filepath.Join(t.TempDir(), "choice.json")
	out.Reset()
	errb.Reset()
	if code := runHarness(&out, &errb, []string{"classify", "--path", "brief.go", "--task", "implement contract brief", "--choose", "legal", "--choice-out", choice, "--reason", "operator confirmed legal matter", "--ttl", "1h"}); code != 0 {
		t.Fatalf("choose code=%d stderr=%s", code, errb.String())
	}
	if raw, err := os.ReadFile(choice); err != nil || !strings.Contains(string(raw), `"scope": "ctx:`) {
		t.Fatalf("choice raw=%s err=%v", raw, err)
	}
	out.Reset()
	errb.Reset()
	if code := runHarness(&out, &errb, []string{"classify", "--path", "brief.go", "--task", "implement contract brief", "--choice-file", choice}); code != 0 {
		t.Fatalf("remembered code=%d stderr=%s", code, errb.String())
	}
	if !strings.Contains(out.String(), `"source": "remembered-choice"`) {
		t.Fatalf("remembered output=%s", out.String())
	}
}

func TestHarnessSelectClassifiesLegalWithoutCodingLeak(t *testing.T) {
	manifest := filepath.Join(t.TempDir(), "harness.json")
	raw := `{"schema":"fak.harness-selection/v1alpha1","layers":[{"id":"legal","scope":"domain","when":{"tags":["legal"]},"capabilities":["citations"]},{"id":"coding","scope":"domain","when":{"tags":["coding"]},"capabilities":["shell"]}]}`
	if err := os.WriteFile(manifest, []byte(raw), 0o600); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	code := runHarness(&out, &errb, []string{"select", "--manifest", manifest, "--path", "matter.docx", "--task", "draft deposition brief"})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errb.String())
	}
	if !strings.Contains(out.String(), `"name": "citations"`) || strings.Contains(out.String(), `"name": "shell"`) {
		t.Fatalf("output=%s", out.String())
	}
}
