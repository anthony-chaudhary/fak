package main

import (
	"bytes"
	"strings"
	"testing"
)

func TestRunCommitHelpDocumentsBuildCheckTimeout(t *testing.T) {
	var out, errb bytes.Buffer
	code := runCommit(&out, &errb, []string{"--help"})
	if code != 2 {
		t.Fatalf("help exit = %d, want usage exit 2", code)
	}
	for _, want := range []string{
		"build-check-timeout",
		"default 4m",
		"controls prospective validation, not advisory-lock waiting or earlier build/materialization phases",
	} {
		if !strings.Contains(errb.String(), want) {
			t.Fatalf("commit help missing %q; got:\n%s", want, errb.String())
		}
	}
}

func TestRunCommitRejectsNonPositiveBuildCheckTimeout(t *testing.T) {
	for _, flagVal := range []string{"--build-check-timeout=0", "--build-check-timeout=0s", "--build-check-timeout=-1s"} {
		var out, errb bytes.Buffer
		code := runCommit(&out, &errb, []string{flagVal, "--path", "a.go", "-m", "fix: test (fak a)"})
		if code != 2 || !strings.Contains(errb.String(), "fak commit: --build-check-timeout must be greater than zero") {
			t.Fatalf("flag %q: code=%d stderr=%q, want greater than zero refusal", flagVal, code, errb.String())
		}
	}
}

func TestRunCommitRejectsMalformedOrOverflowBuildCheckTimeout(t *testing.T) {
	for _, flagVal := range []string{"--build-check-timeout=notaduration", "--build-check-timeout=99999999999999999999h"} {
		var out, errb bytes.Buffer
		code := runCommit(&out, &errb, []string{flagVal, "--path", "a.go", "-m", "fix: test (fak a)"})
		if code != 2 {
			t.Fatalf("flag %q: code=%d, want usage exit 2", flagVal, code)
		}
	}
}
