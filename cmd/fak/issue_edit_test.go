package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

// captureGH is a fake issueCreateRunner: it records every gh argv it is handed
// and returns a canned result, so a test asserts on the built argv without ever
// invoking real gh.
type captureGH struct {
	calls [][]string
	out   string
	fail  bool
}

func (c *captureGH) run(args []string) (string, string, bool) {
	dup := append([]string(nil), args...)
	c.calls = append(c.calls, dup)
	if c.fail {
		return "", "gh boom", false
	}
	return c.out, "", true
}

func decodeEdit(t *testing.T, b []byte) issueEditResult {
	t.Helper()
	var r issueEditResult
	if err := json.Unmarshal(b, &r); err != nil {
		t.Fatalf("decode issueEditResult: %v\n%s", err, b)
	}
	return r
}

func TestIssueEditDryRunRendersArgvWithoutCallingGH(t *testing.T) {
	var out, errb bytes.Buffer
	gh := &captureGH{out: "https://x/42"}
	code := runIssueEditWith(&out, &errb, []string{"--issue", "42", "--title", "New title", "--dry-run", "--json"}, gh.run)
	if code != 0 {
		t.Fatalf("exit = %d, want 0; stderr=%s", code, errb.String())
	}
	if len(gh.calls) != 0 {
		t.Fatalf("dry-run must not call gh, got %v", gh.calls)
	}
	r := decodeEdit(t, out.Bytes())
	if !r.DryRun || !r.OK || r.Issue != 42 {
		t.Fatalf("result = %+v", r)
	}
	if strings.Join(r.Args, " ") != "issue edit 42 --title New title" {
		t.Fatalf("args = %v", r.Args)
	}
}

func TestIssueEditLiveCallsRunnerWithBody(t *testing.T) {
	var out, errb bytes.Buffer
	gh := &captureGH{out: "https://x/42"}
	code := runIssueEditWith(&out, &errb, []string{"--issue", "42", "--body", "fixed body", "--json"}, gh.run)
	if code != 0 {
		t.Fatalf("exit = %d, want 0; stderr=%s", code, errb.String())
	}
	if len(gh.calls) != 1 {
		t.Fatalf("expected exactly one gh call, got %v", gh.calls)
	}
	if strings.Join(gh.calls[0], " ") != "issue edit 42 --body fixed body" {
		t.Fatalf("gh argv = %v", gh.calls[0])
	}
	r := decodeEdit(t, out.Bytes())
	if r.DryRun || !r.OK || r.URL != "https://x/42" {
		t.Fatalf("result = %+v", r)
	}
}

func TestIssueEditAddRemoveLabelArgv(t *testing.T) {
	var out, errb bytes.Buffer
	gh := &captureGH{}
	code := runIssueEditWith(&out, &errb, []string{"--issue", "7", "--add-label", "a,b", "--remove-label", "c", "--dry-run", "--json"}, gh.run)
	if code != 0 {
		t.Fatalf("exit = %d, want 0; stderr=%s", code, errb.String())
	}
	r := decodeEdit(t, out.Bytes())
	if strings.Join(r.Args, " ") != "issue edit 7 --add-label a --add-label b --remove-label c" {
		t.Fatalf("args = %v", r.Args)
	}
}

func TestIssueEditRejectsMissingIssue(t *testing.T) {
	var out, errb bytes.Buffer
	if code := runIssueEditWith(&out, &errb, []string{"--title", "x"}, (&captureGH{}).run); code != 2 {
		t.Fatalf("exit = %d, want 2", code)
	}
}

func TestIssueEditRejectsNoChange(t *testing.T) {
	var out, errb bytes.Buffer
	if code := runIssueEditWith(&out, &errb, []string{"--issue", "5"}, (&captureGH{}).run); code != 2 {
		t.Fatalf("exit = %d, want 2 for nothing-to-change", code)
	}
}

func TestIssueEditRejectsBodyAndBodyFileTogether(t *testing.T) {
	var out, errb bytes.Buffer
	if code := runIssueEditWith(&out, &errb, []string{"--issue", "5", "--body", "a", "--body-file", "f"}, (&captureGH{}).run); code != 2 {
		t.Fatalf("exit = %d, want 2 for body XOR body-file", code)
	}
}

func TestIssueEditReportsGHFailure(t *testing.T) {
	var out, errb bytes.Buffer
	gh := &captureGH{fail: true}
	if code := runIssueEditWith(&out, &errb, []string{"--issue", "5", "--body", "x"}, gh.run); code != 1 {
		t.Fatalf("exit = %d, want 1 on gh failure", code)
	}
}
