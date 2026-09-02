package main

// Smoke tests for `fak session-audit posttool` (#10662) against the committed
// regression fixture in internal/codexlifecycle/testdata/posttool/issue-10662.

import (
	"bytes"
	"path/filepath"
	"strings"
	"testing"
)

func postToolFixtureRoot(t *testing.T) string {
	t.Helper()
	root, err := filepath.Abs(filepath.Join("..", "..", "internal", "codexlifecycle", "testdata", "posttool", "issue-10662"))
	if err != nil {
		t.Fatal(err)
	}
	return root
}

func TestSessionAuditPosttoolJSONFixtureSmoke(t *testing.T) {
	var stdout, stderr bytes.Buffer
	rc := runSessionAuditPosttool(&stdout, &stderr, []string{"--root", postToolFixtureRoot(t), "--json"})
	if rc != 0 {
		t.Fatalf("rc = %d, stderr: %s", rc, stderr.String())
	}
	out := stdout.String()
	for _, want := range []string{
		`"schema": "fak-codex-posttool/1"`,
		`"spans": 211`,
		`"tail_skipped": 1`,
		`"key": "gte201"`,
		`"key": "50k_100k"`,
	} {
		if !strings.Contains(out, want) {
			t.Errorf("json output missing %s", want)
		}
	}
}

func TestSessionAuditPosttoolTextFixtureSmoke(t *testing.T) {
	var stdout, stderr bytes.Buffer
	rc := runSessionAuditPosttool(&stdout, &stderr, []string{"--root", postToolFixtureRoot(t)})
	if rc != 0 {
		t.Fatalf("rc = %d, stderr: %s", rc, stderr.String())
	}
	out := stdout.String()
	for _, want := range []string{"gap_s", "tool_s", "band 50k_100k", "ordinal 1_20", "ordinal gte201"} {
		if !strings.Contains(out, want) {
			t.Errorf("text output missing %q:\n%s", want, out)
		}
	}
}

func TestSessionAuditPosttoolUsageListsSubcommand(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if rc := runSessionAudit(&stdout, &stderr, []string{"help"}); rc != 0 {
		t.Fatalf("rc = %d", rc)
	}
	if !strings.Contains(stderr.String()+stdout.String(), "session-audit posttool") {
		t.Error("session-audit usage does not list the posttool subcommand")
	}
}
