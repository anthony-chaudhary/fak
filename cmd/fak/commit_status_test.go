package main

import (
	"bytes"
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/commitlane"
)

func withCommitStatusFn(t *testing.T, fn func(context.Context, commitlane.Options) (commitlane.Report, error)) {
	t.Helper()
	prev := commitStatusFn
	commitStatusFn = fn
	t.Cleanup(func() { commitStatusFn = prev })
}

func TestRunCommitCommandDispatchesStatus(t *testing.T) {
	withCommitStatusFn(t, func(_ context.Context, opts commitlane.Options) (commitlane.Report, error) {
		if opts.Dir != "repo" {
			t.Fatalf("Dir = %q, want repo", opts.Dir)
		}
		return commitlane.Report{
			Schema:     commitlane.Schema,
			OK:         true,
			Verdict:    commitlane.VerdictClear,
			Reason:     "clear",
			NextAction: "commit lane is clear",
			CommitLock: commitlane.CommitLock{Path: "repo/.git/fak-commit.lock"},
			IndexLock:  commitlane.IndexLock{Path: "repo/.git/index.lock"},
		}, nil
	})
	var out, errb bytes.Buffer
	code := runCommitCommand(&out, &errb, []string{"status", "--dir", "repo"})
	if code != 0 {
		t.Fatalf("want exit 0, got %d stderr=%q", code, errb.String())
	}
	if !strings.Contains(out.String(), "commit lane: clear") || !strings.Contains(out.String(), "queue: none observed") {
		t.Fatalf("human status output missing expected lines:\n%s", out.String())
	}
}

func TestRunCommitStatusJSON(t *testing.T) {
	withCommitStatusFn(t, func(_ context.Context, opts commitlane.Options) (commitlane.Report, error) {
		return commitlane.Report{
			Schema:  commitlane.Schema,
			OK:      false,
			Verdict: commitlane.VerdictStale,
			CommitLock: commitlane.CommitLock{
				Path:      "repo/.git/fak-commit.lock",
				Present:   true,
				HolderPID: 123,
				Stale:     true,
			},
		}, nil
	})
	var out, errb bytes.Buffer
	code := runCommitStatus(&out, &errb, []string{"--json"})
	if code != 0 {
		t.Fatalf("want exit 0, got %d stderr=%q", code, errb.String())
	}
	var rep commitlane.Report
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("invalid json: %v\n%s", err, out.String())
	}
	if rep.Schema != commitlane.Schema || rep.Verdict != commitlane.VerdictStale || rep.CommitLock.HolderPID != 123 {
		t.Fatalf("json report = %+v", rep)
	}
}
