package main

import (
	"bytes"
	"strings"
	"testing"
)

func TestRunHarnessHint(t *testing.T) {
	var stdout, stderr bytes.Buffer
	rc := runHarnessHint(&stdout, &stderr, []string{"--model", "gemini-3.8-flash", "--json"})
	if rc != 0 {
		t.Fatalf("expected rc 0, got %d, stderr: %s", rc, stderr.String())
	}
	out := stdout.String()
	if !strings.Contains(out, `"canonical_model": "gemini-3.8-flash"`) {
		t.Errorf("expected canonical model in json, got: %s", out)
	}
	if !strings.Contains(out, `"posture": "support_heavy"`) {
		t.Errorf("expected support_heavy posture, got: %s", out)
	}

	stdout.Reset()
	stderr.Reset()
	rc = runHarnessHint(&stdout, &stderr, []string{"--model", "gpt-4o"})
	if rc != 0 {
		t.Fatalf("expected rc 0, got %d, stderr: %s", rc, stderr.String())
	}
	if !strings.Contains(stdout.String(), "Model:        gpt-4o") {
		t.Errorf("expected model in text output, got: %s", stdout.String())
	}
}
