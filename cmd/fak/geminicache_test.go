package main

import (
	"bytes"
	"strings"
	"testing"
)

func TestGeminiCacheCLI_InspectAndAdmission(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runGeminiCache(&stdout, &stderr, []string{"--prefix", "stable prompt", "--check-admission", "--json"})
	if code != 0 {
		t.Fatalf("runGeminiCache returned %d; stderr: %s", code, stderr.String())
	}
	out := stdout.String()
	if !strings.Contains(out, `"schema": "fak/gemini-cached-content/v1"`) {
		t.Errorf("missing schema in json output: %s", out)
	}
	if !strings.Contains(out, `"status": "ready"`) {
		t.Errorf("missing ready status: %s", out)
	}

	stdout.Reset()
	stderr.Reset()
	code = runGeminiCache(&stdout, &stderr, []string{"--prefix", "stable prompt", "--check-admission", "--privacy-allowed=false"})
	if code != 1 {
		t.Fatalf("expected admission failure code 1, got %d", code)
	}
	if !strings.Contains(stdout.String(), "admission_refused") {
		t.Errorf("expected admission_refused in output: %s", stdout.String())
	}
}
