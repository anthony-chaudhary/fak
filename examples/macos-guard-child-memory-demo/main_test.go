package main

import (
	"bytes"
	"strings"
	"testing"
)

func TestSelfcheck(t *testing.T) {
	if err := runSelfcheck(); err != nil {
		t.Fatalf("runSelfcheck failed: %v", err)
	}
}

func TestRunSelfcheckFlag(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := run(&stdout, &stderr, []string{"-selfcheck"})
	if code != 0 {
		t.Fatalf("run -selfcheck exit code=%d, stderr=%s", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "selfcheck: PASS") {
		t.Fatalf("run -selfcheck output missing PASS marker: %s", stdout.String())
	}
}

func TestRunDefaultOutput(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := run(&stdout, &stderr, nil)
	if code != 0 {
		t.Fatalf("run default exit code=%d, stderr=%s", code, stderr.String())
	}
	out := stdout.String()
	for _, want := range []string{
		"macOS Default Guard Child-Memory Containment Demo",
		"compliant-child-tree",
		"runaway-child-tree",
		"CHILD_TREE_RSS_LIMIT",
		"fak.guard.child-resource.v1",
		"selfcheck: PASS",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("run default output missing %q:\n%s", want, out)
		}
	}
}

func TestRunJSONOutput(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := run(&stdout, &stderr, []string{"-json"})
	if code != 0 {
		t.Fatalf("run -json exit code=%d, stderr=%s", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), `"schema": "fak.guard.child-resource.v1"`) {
		t.Fatalf("run -json output missing receipt schema:\n%s", stdout.String())
	}
}
