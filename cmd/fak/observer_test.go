package main

import (
	"bytes"
	"strings"
	"testing"
)

func TestRunObserver(t *testing.T) {
	var stdout, stderr bytes.Buffer
	rc := runObserver(&stdout, &stderr, []string{"--json"})
	if rc != 0 {
		t.Fatalf("expected rc 0, got %d, stderr: %s", rc, stderr.String())
	}
	if !strings.Contains(stdout.String(), `"registered": true`) {
		t.Errorf("expected registered in json output, got: %s", stdout.String())
	}
}
