package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestIssueCreateDryRunDoesNotInvokeRunner(t *testing.T) {
	called := false
	runner := func(args []string) (string, string, bool) {
		called = true
		return "https://example.test/issues/1", "", true
	}
	var out, errb bytes.Buffer
	code := runIssueCreateWith(&out, &errb, []string{
		"--title", "feat: per-session activity cell",
		"--body", "add a pane row",
		"--dry-run",
	}, runner)
	if code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, errb.String())
	}
	if called {
		t.Fatalf("--dry-run must never invoke the runner")
	}
	if !strings.Contains(out.String(), "gh issue create") {
		t.Fatalf("dry-run output missing rendered gh argv: %s", out.String())
	}
}

func TestIssueCreateBuildsExpectedGHArgs(t *testing.T) {
	var calls [][]string
	runner := func(args []string) (string, string, bool) {
		calls = append(calls, args)
		return "https://example.test/issues/9", "", true
	}
	var out, errb bytes.Buffer
	code := runIssueCreateWith(&out, &errb, []string{
		"--title", "feat: thing",
		"--body", "body text",
		"--labels", "agent-handoff,next-step",
		"--repo", "owner/repo",
	}, runner)
	if code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, errb.String())
	}
	if len(calls) != 1 {
		t.Fatalf("runner calls = %d, want 1", len(calls))
	}
	joined := strings.Join(calls[0], " ")
	for _, want := range []string{"issue create", "--title feat: thing", "--body body text", "--label agent-handoff", "--label next-step", "--repo owner/repo"} {
		if !strings.Contains(joined, want) {
			t.Fatalf("gh args missing %q: %v", want, calls[0])
		}
	}
	if got := strings.TrimSpace(out.String()); got != "https://example.test/issues/9" {
		t.Fatalf("stdout = %q, want the issue URL", got)
	}
}

func TestIssueCreateBodyFileReadsContent(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "body.md")
	if err := os.WriteFile(path, []byte("body from file"), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	var calls [][]string
	runner := func(args []string) (string, string, bool) {
		calls = append(calls, args)
		return "https://example.test/issues/2", "", true
	}
	var out, errb bytes.Buffer
	code := runIssueCreateWith(&out, &errb, []string{
		"--title", "t",
		"--body-file", path,
	}, runner)
	if code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, errb.String())
	}
	if !strings.Contains(strings.Join(calls[0], " "), "--body body from file") {
		t.Fatalf("gh args missing file body: %v", calls[0])
	}
}

func TestIssueCreateRequiresTitleAndBody(t *testing.T) {
	cases := [][]string{
		{"--body", "b"},
		{"--title", "t"},
		{"--title", "t", "--body", "b", "--body-file", "x"},
	}
	for _, argv := range cases {
		var out, errb bytes.Buffer
		code := runIssueCreateWith(&out, &errb, argv, func(args []string) (string, string, bool) {
			t.Fatalf("runner must not be called for invalid flags: %v", args)
			return "", "", false
		})
		if code != 2 {
			t.Fatalf("argv=%v exit=%d, want 2 (stderr=%s)", argv, code, errb.String())
		}
	}
}

func TestIssueCreateReportsGHFailure(t *testing.T) {
	runner := func(args []string) (string, string, bool) {
		return "", "HTTP 422: validation failed", false
	}
	var out, errb bytes.Buffer
	code := runIssueCreateWith(&out, &errb, []string{"--title", "t", "--body", "b"}, runner)
	if code != 1 {
		t.Fatalf("exit=%d, want 1", code)
	}
	if !strings.Contains(errb.String(), "validation failed") {
		t.Fatalf("stderr missing gh failure detail: %s", errb.String())
	}
}

func TestIssueCreateJSONOutput(t *testing.T) {
	runner := func(args []string) (string, string, bool) {
		return "https://example.test/issues/3", "", true
	}
	var out, errb bytes.Buffer
	code := runIssueCreateWith(&out, &errb, []string{"--title", "t", "--body", "b", "--json"}, runner)
	if code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, errb.String())
	}
	var got issueCreateResult
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatalf("parse json: %v\n%s", err, out.String())
	}
	if !got.OK || got.URL != "https://example.test/issues/3" {
		t.Fatalf("result = %+v", got)
	}
}
