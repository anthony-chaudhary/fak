package devcmd

import (
	"bytes"
	"strings"
	"testing"
)

func TestRunAMDGPUFactsRejectsUnexpectedArguments(t *testing.T) {
	var stdout, stderr bytes.Buffer

	code := RunAMDGPUFacts(&stdout, &stderr, []string{"extra"})

	if code != 2 {
		t.Fatalf("RunAMDGPUFacts exit = %d, want 2", code)
	}
	if stdout.Len() != 0 {
		t.Errorf("stdout = %q, want empty", stdout.String())
	}
	if !strings.Contains(stderr.String(), "unexpected arguments: extra") {
		t.Errorf("stderr = %q, want unexpected-arguments diagnostic", stderr.String())
	}
}

func TestRunAMDGPUFactsRejectsUnknownFlag(t *testing.T) {
	var stdout, stderr bytes.Buffer

	code := RunAMDGPUFacts(&stdout, &stderr, []string{"--unknown"})

	if code != 2 {
		t.Fatalf("RunAMDGPUFacts exit = %d, want 2", code)
	}
	if stdout.Len() != 0 {
		t.Errorf("stdout = %q, want empty", stdout.String())
	}
	if !strings.Contains(stderr.String(), "flag provided but not defined") {
		t.Errorf("stderr = %q, want flag parsing diagnostic", stderr.String())
	}
}
