package auditreason

import (
	"reflect"
	"testing"
)

// TestClassifyResumeCommitPushHalfApplied is the #2086 fixture: a commit+push
// sequence killed on timeout AFTER the local commit landed but BEFORE the push
// reached the remote must report which half applied and a safe resume command,
// instead of the bare exit-143 that hides it.
func TestClassifyResumeCommitPushHalfApplied(t *testing.T) {
	steps := []ResumeStep{
		{Name: "commit", Applied: true, AppliedMsg: "commit created locally", ResumeCmd: "git commit -s -- <paths>"},
		{Name: "push", Applied: false, PendingMsg: "push NOT sent", ResumeCmd: "git push"},
	}
	got := ClassifyResume("commit-push", ToolFailureTimeout, steps)

	if got.Complete {
		t.Fatal("half-applied sequence must not be Complete")
	}
	if !reflect.DeepEqual(got.AppliedSoFar, []string{"commit"}) {
		t.Fatalf("applied_so_far = %v, want [commit]", got.AppliedSoFar)
	}
	if !reflect.DeepEqual(got.Pending, []string{"push"}) {
		t.Fatalf("pending = %v, want [push]", got.Pending)
	}
	if got.SafeResumeCmd != "git push" {
		t.Fatalf("safe_resume_cmd = %q, want %q", got.SafeResumeCmd, "git push")
	}
	if got.TreeState != "commit created locally; push NOT sent" {
		t.Fatalf("tree_state = %q", got.TreeState)
	}
	if got.Token != string(ToolFailureTimeout) {
		t.Fatalf("token = %q, want %q", got.Token, ToolFailureTimeout)
	}
	if !got.Retryable {
		t.Fatal("a half-applied op with an observed safe resume must be Retryable")
	}
}

// TestClassifyResumeComplete pins the both-halves-landed case: a sequence killed
// only after every effect applied reports Complete, no pending step, no resume.
func TestClassifyResumeComplete(t *testing.T) {
	steps := []ResumeStep{
		{Name: "commit", Applied: true, AppliedMsg: "commit created locally"},
		{Name: "push", Applied: true, AppliedMsg: "push confirmed on origin/main"},
	}
	got := ClassifyResume("commit-push", ToolFailureTimeout, steps)
	if !got.Complete {
		t.Fatalf("all-applied sequence must be Complete: %+v", got)
	}
	if len(got.Pending) != 0 {
		t.Fatalf("pending = %v, want empty", got.Pending)
	}
	if got.SafeResumeCmd != "" {
		t.Fatalf("safe_resume_cmd = %q, want empty for a complete op", got.SafeResumeCmd)
	}
	if got.Retryable {
		t.Fatal("a complete op needs no resume, so it is not Retryable")
	}
}

// TestClassifyResumeNothingApplied pins the earliest-kill case: the commit half
// never landed, so the resume starts at the commit, not the push.
func TestClassifyResumeNothingApplied(t *testing.T) {
	steps := []ResumeStep{
		{Name: "commit", Applied: false, PendingMsg: "changes still uncommitted (commit NOT created)", ResumeCmd: "git commit -s -- <paths>"},
		{Name: "push", Applied: false, PendingMsg: "push NOT sent (no commit to push yet)", ResumeCmd: "git push"},
	}
	got := ClassifyResume("commit-push", ToolFailureTimeout, steps)
	if got.Complete {
		t.Fatal("nothing-applied sequence must not be Complete")
	}
	if len(got.AppliedSoFar) != 0 {
		t.Fatalf("applied_so_far = %v, want empty", got.AppliedSoFar)
	}
	if got.SafeResumeCmd != "git commit -s -- <paths>" {
		t.Fatalf("safe_resume_cmd = %q, want the commit resume (first pending step)", got.SafeResumeCmd)
	}
}
