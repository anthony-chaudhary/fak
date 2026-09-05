package main

import (
	"bytes"
	"strings"
	"testing"
)

func TestRunSenseCheck(t *testing.T) {
	var stdout, stderr bytes.Buffer
	rc := runSenseCheck(strings.NewReader(""), &stdout, &stderr, []string{"--text", "All tests passed with zero failures.", "--json"})
	if rc != 0 {
		t.Fatalf("expected rc 0, got %d, stderr: %s", rc, stderr.String())
	}
	if !strings.Contains(stdout.String(), `"verdict": "CLEAN"`) {
		t.Errorf("expected clean verdict, got: %s", stdout.String())
	}
}
