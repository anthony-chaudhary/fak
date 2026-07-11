package main

import (
	"fmt"
	"io"
	"os/exec"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/auditreason"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// runResumePlan is the `fak check-tool-failure --resume` mode: instead of a bare
// exit-143, report the partial state of a long-running mutating op that was
// killed on timeout — which steps already applied, the resulting tree state, and
// the single safe command that resumes without double-applying (#2086).
func runToolResumePlan(stdout, stderr io.Writer, op, dir string, asJSON bool) int {
	var (
		steps []auditreason.ResumeStep
		err   error
	)
	switch strings.ToLower(strings.TrimSpace(op)) {
	case "", "commit-push", "commit+push":
		op = "commit-push"
		steps, err = commitPushSteps(dir)
	default:
		fmt.Fprintf(stderr, "fak check-tool-failure: unknown --op %q (want commit-push)\n", op)
		return 2
	}
	if err != nil {
		fmt.Fprintf(stderr, "fak check-tool-failure: %v\n", err)
		return 3
	}

	report := auditreason.ClassifyResume(op, auditreason.ToolFailureTimeout, steps)
	if asJSON {
		return encodeJSONOrFail(stdout, stderr, report, "fak check-tool-failure")
	}
	state := "incomplete"
	if report.Complete {
		state = "complete"
	}
	fmt.Fprintf(stdout, "op: %s  (%s, %s)\n", report.Op, report.Token, state)
	fmt.Fprintf(stdout, "  applied so far: %s\n", joinOrNone(report.AppliedSoFar))
	fmt.Fprintf(stdout, "  pending: %s\n", joinOrNone(report.Pending))
	if strings.TrimSpace(report.TreeState) != "" {
		fmt.Fprintf(stdout, "  tree state: %s\n", report.TreeState)
	}
	if report.SafeResumeCmd != "" {
		fmt.Fprintf(stdout, "  safe resume: %s\n", report.SafeResumeCmd)
	}
	fmt.Fprintf(stdout, "  retryable: %v\n", report.Retryable)
	return 0
}

// commitPushSteps reads dir's live git state after a killed `git commit && git
// push` sequence and builds the ordered commit->push step pair. It distinguishes
// three post-kill tree states:
//   - working tree still dirty  => the commit never landed (commit pending)
//   - clean tree, HEAD ahead of its upstream => commit landed locally, push pending
//   - clean tree, HEAD == upstream => both halves landed (complete)
//
// A repo with a commit but no tracking ref is read as "commit landed, nothing
// pushed" — you cannot have pushed without an upstream to push to.
func commitPushSteps(dir string) ([]auditreason.ResumeStep, error) {
	if _, ok := gitTrim(dir, "rev-parse", "--git-dir"); !ok {
		return nil, fmt.Errorf("not a git repository: %s", dir)
	}

	status, _ := gitTrim(dir, "status", "--porcelain")
	dirty := strings.TrimSpace(status) != ""
	_, hasHead := gitTrim(dir, "rev-parse", "HEAD")
	upstream, hasUpstream := gitTrim(dir, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")

	ahead := 0
	switch {
	case hasUpstream:
		if c, ok := gitTrim(dir, "rev-list", "--count", "@{u}..HEAD"); ok {
			ahead, _ = strconv.Atoi(c)
		}
	case hasHead:
		// A commit exists but no upstream to push to => nothing has been pushed.
		ahead = 1
	}

	commitApplied := hasHead && !dirty
	pushApplied := commitApplied && hasUpstream && ahead == 0

	commit := auditreason.ResumeStep{Name: "commit", Applied: commitApplied, ResumeCmd: "git commit -s -- <paths>"}
	if commitApplied {
		commit.AppliedMsg = "commit created locally"
	} else {
		commit.PendingMsg = "changes still uncommitted (commit NOT created)"
	}

	push := auditreason.ResumeStep{Name: "push", Applied: pushApplied, ResumeCmd: "git push"}
	switch {
	case pushApplied:
		push.AppliedMsg = "push confirmed on " + upstream
	case commitApplied:
		push.PendingMsg = "push NOT sent"
	default:
		push.PendingMsg = "push NOT sent (no commit to push yet)"
	}

	return []auditreason.ResumeStep{commit, push}, nil
}

// gitTrim runs `git -C dir args...` and returns its trimmed stdout. ok is false
// on any error (a non-git dir, an unset upstream via @{u}, etc.), which the
// caller reads as "that fact is not established" rather than a hard failure.
func gitTrim(dir string, args ...string) (string, bool) {
	full := append([]string{"-C", dir}, args...)
	cmd := exec.Command("git", full...)
	windowgate.ConfigureBackgroundCommand(cmd)
	out, err := cmd.Output()
	if err != nil {
		return "", false
	}
	return strings.TrimSpace(string(out)), true
}
