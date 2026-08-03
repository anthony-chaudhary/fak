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

// TestRunCommitStatusOffersScopedIndexChurnClear pins the OFFER half of #5339 at the verb
// boundary: `fak commit status` names the no-op staged deletions, prints the scoped
// `git restore --staged` for exactly those paths, says plainly that it did not run it, and
// never lets a real staged deletion into the offered command.
func TestRunCommitStatusOffersScopedIndexChurnClear(t *testing.T) {
	audit := commitlane.ClassifyStagedDeletions([]commitlane.StagedDeletionFact{
		{Path: "internal/a/noop.go", OnDisk: true, DiskHash: "aaaa", HeadHash: "aaaa"},
		{Path: "internal/b/keep.go", OnDisk: true, DiskHash: "bbbb", HeadHash: "bbbb"},
		{Path: "internal/c/really_deleted.go", OnDisk: false, HeadHash: "cccc"},
	})
	withCommitStatusFn(t, func(context.Context, commitlane.Options) (commitlane.Report, error) {
		return commitlane.Report{
			Schema:     commitlane.Schema,
			OK:         true,
			Verdict:    commitlane.VerdictClear,
			IndexChurn: &audit,
		}, nil
	})
	var out, errb bytes.Buffer
	if code := runCommitStatus(&out, &errb, nil); code != 0 {
		t.Fatalf("want exit 0, got %d stderr=%q", code, errb.String())
	}
	got := out.String()
	for _, want := range []string{
		"index churn: 2 of 3 staged deletion(s) are no-ops",
		"internal/a/noop.go",
		"1 other staged deletion(s) left alone",
		"clear only these (not run for you): git restore --staged -- internal/a/noop.go internal/b/keep.go",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("status output missing %q:\n%s", want, got)
		}
	}
	remedy := got[strings.Index(got, "clear only these"):]
	if strings.Contains(remedy, "really_deleted.go") {
		t.Fatalf("the offered clear must never name a real staged deletion:\n%s", remedy)
	}
}

// TestRunCommitStatusReportsNoChurnWhenAllDeletionsAreReal confirms an index full of
// GENUINE staged deletions produces no remedy at all — the case where a naive detector
// would offer to un-stage a peer's real work.
func TestRunCommitStatusReportsNoChurnWhenAllDeletionsAreReal(t *testing.T) {
	audit := commitlane.ClassifyStagedDeletions([]commitlane.StagedDeletionFact{
		{Path: "internal/a/gone.go", OnDisk: false, HeadHash: "aaaa"},
		{Path: "internal/b/edited.go", OnDisk: true, DiskHash: "bbbb", HeadHash: "cccc"},
	})
	withCommitStatusFn(t, func(context.Context, commitlane.Options) (commitlane.Report, error) {
		return commitlane.Report{Schema: commitlane.Schema, OK: true, Verdict: commitlane.VerdictClear, IndexChurn: &audit}, nil
	})
	var out, errb bytes.Buffer
	if code := runCommitStatus(&out, &errb, nil); code != 0 {
		t.Fatalf("want exit 0, got %d stderr=%q", code, errb.String())
	}
	got := out.String()
	if !strings.Contains(got, "index churn: none (2 staged deletion(s), all real or unproven)") {
		t.Fatalf("status output should report no churn:\n%s", got)
	}
	if strings.Contains(got, "git restore --staged") {
		t.Fatalf("no churn must offer no remedy:\n%s", got)
	}
}
