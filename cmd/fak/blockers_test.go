package main

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const (
	blockerLabelProbe = "api --silent repos/o/r/labels/blocked"
	blockerIssueQuery = "issue list --repo o/r --state open --label blocked --limit 200 --json number,title,url,assignees,labels"
)

func TestRunBlockersSourceSuccessfulEmptyQueryFeedsClear(t *testing.T) {
	dir := t.TempDir()
	issuesPath := filepath.Join(dir, "issues.json")
	statusPath := filepath.Join(dir, "blockers-source.json")
	gh := scriptedBlockerGH(t,
		blockerGHStep{args: blockerLabelProbe},
		blockerGHStep{args: blockerIssueQuery, out: "[]\n"},
		blockerGHStep{args: blockerLabelProbe},
	)

	var sourceOut, sourceErr bytes.Buffer
	code := runBlockersSourceWithGH(&sourceOut, &sourceErr, []string{
		"--repo", "o/r",
		"--label", "blocked",
		"--issues-out", issuesPath,
		"--status-out", statusPath,
	}, gh)
	if code != 0 {
		t.Fatalf("successful source acquisition exit = %d, stderr=%s", code, sourceErr.String())
	}
	if !strings.Contains(sourceOut.String(), `blocker source OK: 0 open "blocked" issue(s)`) {
		t.Fatalf("source success output omitted the witnessed zero count:\n%s", sourceOut.String())
	}

	var stdout, stderr bytes.Buffer
	code = runBlockersFeed(&stdout, &stderr, []string{
		"--issues", issuesPath,
		"--source-status", statusPath,
		"--label", "blocked",
		"--source", "ci",
		"--dry-run",
	})
	if code != 0 {
		t.Fatalf("successful empty query exit = %d, stderr=%s", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "no standing blockers") {
		t.Fatalf("successful empty query did not render the clear card:\n%s", stdout.String())
	}
}

func TestRunBlockersSourceFailuresStayUnknown(t *testing.T) {
	tests := []struct {
		name  string
		label string
		steps []blockerGHStep
		want  string
	}{
		{
			name:  "empty label",
			label: " ",
			want:  "configured label is empty",
		},
		{
			name:  "missing label before query",
			label: "blocked",
			steps: []blockerGHStep{{args: blockerLabelProbe, err: errors.New("HTTP 404")}},
			want:  "could not be resolved before the issue query",
		},
		{
			name:  "query failure",
			label: "blocked",
			steps: []blockerGHStep{
				{args: blockerLabelProbe},
				{args: blockerIssueQuery, err: errors.New("authentication required")},
			},
			want: "gh issue list failed",
		},
		{
			name:  "unusable payload",
			label: "blocked",
			steps: []blockerGHStep{
				{args: blockerLabelProbe},
				{args: blockerIssueQuery, out: "null\n"},
			},
			want: "returned an unusable payload",
		},
		{
			name:  "label disappears during query",
			label: "blocked",
			steps: []blockerGHStep{
				{args: blockerLabelProbe},
				{args: blockerIssueQuery, out: "[]\n"},
				{args: blockerLabelProbe, err: errors.New("HTTP 404")},
			},
			want: "could not be resolved after the issue query",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			issuesPath := filepath.Join(dir, "issues.json")
			statusPath := filepath.Join(dir, "blockers-source.json")
			gh := scriptedBlockerGH(t, tt.steps...)

			var stdout, stderr bytes.Buffer
			code := runBlockersSourceWithGH(&stdout, &stderr, []string{
				"--repo", "o/r",
				"--label", tt.label,
				"--issues-out", issuesPath,
				"--status-out", statusPath,
			}, gh)
			if code == 0 {
				t.Fatalf("source failure unexpectedly succeeded:\n%s", stdout.String())
			}
			if !strings.Contains(stderr.String(), "blocker source UNKNOWN") || !strings.Contains(stderr.String(), tt.want) {
				t.Fatalf("source failure stderr = %q, want UNKNOWN with %q", stderr.String(), tt.want)
			}
			raw, err := os.ReadFile(statusPath)
			if err != nil {
				t.Fatalf("read UNKNOWN marker: %v", err)
			}
			if !strings.Contains(string(raw), `"status":"UNKNOWN"`) || !strings.Contains(string(raw), tt.want) {
				t.Fatalf("failure marker = %s, want UNKNOWN with %q", raw, tt.want)
			}
			if _, err := os.Stat(issuesPath); !os.IsNotExist(err) {
				t.Fatalf("source failure wrote an issue payload: stat err=%v", err)
			}
		})
	}
}

func TestRunBlockersFeedUnknownSourceFailsBeforeRendering(t *testing.T) {
	dir := t.TempDir()
	statusPath := filepath.Join(dir, "blockers-source.json")
	writeBlockersFixture(t, statusPath, `{"status":"UNKNOWN","reason":"configured label could not be resolved"}`)

	var stdout, stderr bytes.Buffer
	code := runBlockersFeed(&stdout, &stderr, []string{
		"--issues", filepath.Join(dir, "missing-issues.json"),
		"--source-status", statusPath,
		"--dry-run",
	})
	if code == 0 {
		t.Fatalf("UNKNOWN source unexpectedly succeeded:\n%s", stdout.String())
	}
	if stdout.Len() != 0 || strings.Contains(strings.ToLower(stdout.String()), "no standing blockers") {
		t.Fatalf("UNKNOWN source rendered an all-clear:\n%s", stdout.String())
	}
	got := stderr.String()
	if !strings.Contains(got, "blocker source UNKNOWN") || !strings.Contains(got, "configured label could not be resolved") {
		t.Fatalf("UNKNOWN source failure omitted its typed reason:\n%s", got)
	}
	if strings.Contains(got, "missing-issues.json") {
		t.Fatalf("issue payload was read before the UNKNOWN source marker was checked:\n%s", got)
	}
}

func TestRunBlockersFeedRejectsUnusableIssuePayloads(t *testing.T) {
	tests := []struct {
		name string
		body *string
		want string
	}{
		{name: "missing flag", want: "--issues is required"},
		{name: "blank", body: blockerString(" \n\t"), want: "--issues payload is empty"},
		{name: "null", body: blockerString("null"), want: "null is UNKNOWN"},
		{name: "malformed", body: blockerString("{"), want: "parse --issues payload"},
		{name: "missing issue fields", body: blockerString("[{}]"), want: "issue[0] has no positive number"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var args []string
			if tt.body != nil {
				path := filepath.Join(t.TempDir(), "issues.json")
				writeBlockersFixture(t, path, *tt.body)
				args = append(args, "--issues", path)
			}
			args = append(args, "--dry-run")

			var stdout, stderr bytes.Buffer
			code := runBlockersFeed(&stdout, &stderr, args)
			if code == 0 {
				t.Fatalf("unusable payload unexpectedly succeeded:\n%s", stdout.String())
			}
			if stdout.Len() != 0 || strings.Contains(strings.ToLower(stdout.String()), "no standing blockers") {
				t.Fatalf("unusable payload rendered an all-clear:\n%s", stdout.String())
			}
			if !strings.Contains(stderr.String(), tt.want) {
				t.Fatalf("stderr = %q, want %q", stderr.String(), tt.want)
			}
		})
	}
}

func TestBlockersFeedWorkflowChecksSourceMarkerBeforeRendering(t *testing.T) {
	path := filepath.Join("..", "..", ".github", "workflows", "blockers-feed.yml")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	got := string(raw)
	for _, want := range []string{
		"continue-on-error: true",
		"go run ./cmd/fak blockers source",
		"--issues-out issues.json",
		"--status-out blockers-source.json",
		"--source-status blockers-source.json",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("workflow lost fail-closed source contract %q", want)
		}
	}
	for _, forbidden := range []string{
		`echo "[]" > issues.json`,
		"treating as no standing blockers",
	} {
		if strings.Contains(got, forbidden) {
			t.Fatalf("workflow still contains false-all-clear fallback %q", forbidden)
		}
	}
	if strings.Count(got, "--source-status blockers-source.json") != 2 {
		t.Fatalf("both render and post must check the source marker; workflow has %d checks",
			strings.Count(got, "--source-status blockers-source.json"))
	}
}

func writeBlockersFixture(t *testing.T, path, body string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}

type blockerGHStep struct {
	args string
	out  string
	err  error
}

func scriptedBlockerGH(t *testing.T, steps ...blockerGHStep) blockerFeedGHRunner {
	t.Helper()
	next := 0
	t.Cleanup(func() {
		if next != len(steps) {
			t.Errorf("gh script consumed %d of %d step(s)", next, len(steps))
		}
	})
	return func(args ...string) ([]byte, error) {
		t.Helper()
		if next >= len(steps) {
			t.Fatalf("unexpected gh call: %s", strings.Join(args, " "))
		}
		step := steps[next]
		next++
		if got := strings.Join(args, " "); got != step.args {
			t.Fatalf("gh call %d = %q, want %q", next, got, step.args)
		}
		return []byte(step.out), step.err
	}
}

func blockerString(s string) *string { return &s }
