package main

import (
	"bytes"
	"strings"
	"testing"
)

func TestParseGhIssueUnwitnessedInput(t *testing.T) {
	raw := []byte(`{
		"number": 1816,
		"state": "OPEN",
		"comments": [
			{"author": {"login": "alice"}, "body": "still working on it"},
			{"author": {"login": "bob"}, "body": "this is done now"}
		]
	}`)
	in, err := parseGhIssueUnwitnessedInput(raw)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if in.IssueNumber != 1816 || !in.Open {
		t.Fatalf("want issue_number=1816 open=true, got %+v", in)
	}
	if len(in.Comments) != 2 || in.Comments[1].Author != "bob" {
		t.Fatalf("want 2 comments, latest by bob, got %+v", in.Comments)
	}
}

func TestParseGhIssueUnwitnessedInput_ClosedIssue(t *testing.T) {
	raw := []byte(`{"number": 42, "state": "CLOSED", "comments": []}`)
	in, err := parseGhIssueUnwitnessedInput(raw)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if in.Open {
		t.Fatalf("want open=false for a CLOSED issue, got %+v", in)
	}
}

func TestParseGhIssueUnwitnessedInput_BadJSON(t *testing.T) {
	if _, err := parseGhIssueUnwitnessedInput([]byte("not json")); err == nil {
		t.Fatal("want an error for invalid JSON")
	}
}

func TestRunDispatchUnwitnessedClaim_MissingIssueFlag(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runDispatchUnwitnessedClaim(&stdout, &stderr, []string{"--json"})
	if code != 2 {
		t.Fatalf("want exit 2 for a missing --issue, got %d (stderr=%s)", code, stderr.String())
	}
	if !strings.Contains(stderr.String(), "--issue") {
		t.Errorf("stderr should name the missing flag, got %q", stderr.String())
	}
}
