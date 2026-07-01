package main

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/issuesmallness"
)

const dispatchIssueSmallnessSingleBody = `## Goal
Add a filing-time lint for unresolved template artifacts.

## Done condition
The issue filer rejects bodies containing the known corruption pattern.

## Witness
A regression test fails on a malformed-body fixture and passes on a normal body.
`

const dispatchIssueSmallnessThreeBody = `## Goal
- Fix the login redirect bug on the accounts page.
- Add a new weekly throughput dashboard for the operator.
- Rewrite the onboarding documentation from scratch.

## Done condition
All three items above land.

## Witness
A fixture proves the three-task bundle fails.
`

func TestDispatchIssueSmallnessBodyFileJSONFailsThreeTasks(t *testing.T) {
	var stdout, stderr strings.Builder
	code := runDispatchIssueSmallnessLint(&stdout, &stderr, strings.NewReader(dispatchIssueSmallnessThreeBody), []string{"--body-file", "-", "--json"})
	if code != 1 {
		t.Fatalf("exit = %d, want 1 (stderr: %s)", code, stderr.String())
	}
	var got map[string]any
	if err := json.Unmarshal([]byte(stdout.String()), &got); err != nil {
		t.Fatalf("bad json: %v\n%s", err, stdout.String())
	}
	if got["schema"] != issuesmallness.Schema || got["mode"] != "body-file" || got["verdict"] != issuesmallness.Fail {
		t.Fatalf("payload = %#v, want body-file fail schema", got)
	}
	if got["count"] != float64(3) || got["witness_count"] != float64(1) {
		t.Fatalf("count/witness_count = %v/%v, want 3/1", got["count"], got["witness_count"])
	}
}

func TestDispatchIssueSmallnessOpenReportUsesDryRunRows(t *testing.T) {
	oldFetch := dispatchIssueSmallnessFetchOpenIssues
	dispatchIssueSmallnessFetchOpenIssues = func(limit int) ([]issuesmallness.Issue, error) {
		if limit != 7 {
			t.Fatalf("limit = %d, want 7", limit)
		}
		return []issuesmallness.Issue{
			{Number: 1, Title: "clean one", Body: dispatchIssueSmallnessSingleBody},
			{Number: 2, Title: "bundled one", Body: dispatchIssueSmallnessThreeBody},
		}, nil
	}
	t.Cleanup(func() { dispatchIssueSmallnessFetchOpenIssues = oldFetch })

	var stdout, stderr strings.Builder
	code := runDispatchIssueSmallnessLint(&stdout, &stderr, strings.NewReader(""), []string{"--open", "--limit", "7", "--json"})
	if code != 1 {
		t.Fatalf("exit = %d, want 1 because report contains a fail (stderr: %s)", code, stderr.String())
	}
	var got issuesmallness.OpenReport
	if err := json.Unmarshal([]byte(stdout.String()), &got); err != nil {
		t.Fatalf("bad json: %v\n%s", err, stdout.String())
	}
	if got.Scanned != 2 || got.Counts[issuesmallness.Pass] != 1 || got.Counts[issuesmallness.Fail] != 1 {
		t.Fatalf("report = %#v, want one pass and one fail", got)
	}
	if len(got.Flagged) != 1 || got.Flagged[0].Number != 2 {
		t.Fatalf("flagged = %#v, want issue 2", got.Flagged)
	}
}

func TestDispatchIssueSmallnessIssueModeFetchesBody(t *testing.T) {
	oldFetch := dispatchIssueSmallnessFetchIssueBody
	dispatchIssueSmallnessFetchIssueBody = func(number int) (string, error) {
		if number != 42 {
			t.Fatalf("number = %d, want 42", number)
		}
		return dispatchIssueSmallnessSingleBody, nil
	}
	t.Cleanup(func() { dispatchIssueSmallnessFetchIssueBody = oldFetch })

	var stdout, stderr strings.Builder
	code := runDispatchIssueSmallnessLint(&stdout, &stderr, strings.NewReader(""), []string{"--issue", "42", "--json"})
	if code != 0 {
		t.Fatalf("exit = %d, want 0 (stderr: %s)", code, stderr.String())
	}
	var got map[string]any
	if err := json.Unmarshal([]byte(stdout.String()), &got); err != nil {
		t.Fatalf("bad json: %v\n%s", err, stdout.String())
	}
	if got["mode"] != "issue" || got["issue"] != float64(42) || got["verdict"] != issuesmallness.Pass {
		t.Fatalf("payload = %#v, want issue 42 pass", got)
	}
}

func TestDispatchIssueSmallnessRequiresOneMode(t *testing.T) {
	var stdout, stderr strings.Builder
	code := runDispatchIssueSmallnessLint(&stdout, &stderr, strings.NewReader(""), nil)
	if code != 2 {
		t.Fatalf("exit = %d, want 2", code)
	}
	if !strings.Contains(stderr.String(), "choose exactly one") {
		t.Fatalf("stderr = %q, want mode error", stderr.String())
	}
}

func TestRunDispatchRoutesIssueSmallnessAlias(t *testing.T) {
	oldFetch := dispatchIssueSmallnessFetchOpenIssues
	dispatchIssueSmallnessFetchOpenIssues = func(limit int) ([]issuesmallness.Issue, error) {
		return []issuesmallness.Issue{{Number: 1, Title: "clean one", Body: dispatchIssueSmallnessSingleBody}}, nil
	}
	t.Cleanup(func() { dispatchIssueSmallnessFetchOpenIssues = oldFetch })

	var stdout, stderr strings.Builder
	code := runDispatch(&stdout, &stderr, []string{"smallness", "--open"})
	if code != 0 {
		t.Fatalf("exit = %d, want 0 (stderr: %s)", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "1 pass") {
		t.Fatalf("stdout = %q, want human dry-run report", stdout.String())
	}
}
