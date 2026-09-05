package main

import (
	"bytes"
	"io"
	"reflect"
	"testing"
)

func TestReleaseNextSubcommandDispatchesHelper(t *testing.T) {
	oldNext := releaseRunNext
	defer func() { releaseRunNext = oldNext }()

	var gotArgs []string
	releaseRunNext = func(stdout, stderr io.Writer, args []string) int {
		gotArgs = append([]string(nil), args...)
		return 0
	}

	var out, errb bytes.Buffer
	rc := runRelease(&out, &errb, []string{"next", "--json", "--check"})
	if rc != 0 {
		t.Fatalf("exit = %d, want 0", rc)
	}
	if !reflect.DeepEqual(gotArgs, []string{"--json", "--check"}) {
		t.Fatalf("args = %#v, want [--json --check]", gotArgs)
	}
}

func TestReleaseNextInvokesReleaseNextScript(t *testing.T) {
	oldScript := releaseRunScript
	defer func() { releaseRunScript = oldScript }()

	var gotScript string
	var gotArgs []string
	releaseRunScript = func(root, script string, args []string, stdout, stderr io.Writer) int {
		gotScript = script
		gotArgs = append([]string(nil), args...)
		return 42
	}

	var out, errb bytes.Buffer
	rc := runReleaseNext(&out, &errb, []string{"--sync"})
	if rc != 42 {
		t.Fatalf("exit = %d, want 42", rc)
	}
	if gotScript != "release_next.py" {
		t.Fatalf("script = %q, want release_next.py", gotScript)
	}
	if !reflect.DeepEqual(gotArgs, []string{"--sync"}) {
		t.Fatalf("args = %#v, want [--sync]", gotArgs)
	}
}
