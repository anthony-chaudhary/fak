package main

import (
	"fmt"
	"io"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/hooks"
)

// runCommitPreview lints a proposed commit (message + the paths it would touch) and reports the
// verdict WITHOUT running git. Exit 0 when nothing blocking was found, 1 otherwise.
//
// It also surfaces the ADVISORY core-lock fold (issue #1682): the exact paths the commit would
// touch are classified against the shipped soft-contract taxonomy, and any coherence-bearing
// surface produces a path-scoped warning naming the witness command that would clear it. The
// warnings are WARNING MODE — they never change the exit code; only the lint verdict decides it.
func runCommitPreview(stdout, stderr io.Writer, message string, paths []string, root, expectedBranch string, asJSON, requireIssue bool) int {
	rep := hooks.LintCommitMessageWithOptions(message, paths, root, requireIssue)
	coreLockWarns := auditCoreLockPaths(paths)
	if asJSON {
		payload := struct {
			hooks.CommitLintReport
			ExpectedBranch   string            `json:"expected_branch,omitempty"`
			CoreLockWarnings []coreLockWarning `json:"core_lock_warnings"`
			CoreLockWarnMode string            `json:"core_lock_warn_mode"`
		}{
			CommitLintReport: rep,
			ExpectedBranch:   expectedBranch,
			CoreLockWarnings: coreLockWarns,
			CoreLockWarnMode: coreLockModeWarning,
		}
		if err := writeIndentedJSON(stdout, payload); err != nil {
			fmt.Fprintf(stderr, "fak commit: %v\n", err)
			return 1
		}
	} else {
		renderPreview(stdout, rep, expectedBranch)
		renderCoreLockWarnings(stdout, coreLockWarns)
	}
	if rep.OK {
		return 0
	}
	return 1
}

func renderPreview(w io.Writer, r hooks.CommitLintReport, expectedBranch string) {
	if r.OK {
		fmt.Fprintln(w, "commit-preview OK — subject is witness-gradeable and bindable")
	} else {
		fmt.Fprintf(w, "commit-preview: %d blocking issue(s)\n", len(r.Issues))
	}
	fmt.Fprintf(w, "  score    : %d/100 (%s)\n", r.Score, r.Grade)
	fmt.Fprintf(w, "  subject  : %s\n", r.Subject)
	fmt.Fprintf(w, "  gradeable: %v   stamp: %s", r.Gradeable, r.StampKind)
	if r.Leaf != "" {
		fmt.Fprintf(w, " (fak %s, recognized=%v)", r.Leaf, r.LeafRecognized)
	}
	fmt.Fprintln(w)
	if len(r.PathLanes) > 0 {
		fmt.Fprintf(w, "  path lane: %s\n", strings.Join(r.PathLanes, ", "))
	}
	if expectedBranch != "" {
		fmt.Fprintf(w, "  expected branch: %s\n", expectedBranch)
	}
	fmt.Fprintf(w, "  issue link: resolving=%v", r.IssueResolving)
	if len(r.IssueRefs) > 0 {
		refs := make([]string, len(r.IssueRefs))
		for i, n := range r.IssueRefs {
			refs[i] = fmt.Sprintf("#%d", n)
		}
		fmt.Fprintf(w, " (refs %s)", strings.Join(refs, ", "))
	}
	fmt.Fprintln(w)
	if r.Generation != "" {
		fmt.Fprintf(w, "  generation: %s\n", r.Generation)
	}
	for _, is := range r.Issues {
		fmt.Fprintf(w, "  ✗ %s\n", is)
	}
	for _, n := range r.Notes {
		fmt.Fprintf(w, "  · %s\n", n)
	}
	if !r.OK && r.SuggestedSubject != "" {
		fmt.Fprintf(w, "  → suggested subject: %s\n", r.SuggestedSubject)
	} else if !r.OK && r.SuggestTrailer != "" {
		fmt.Fprintf(w, "  → suggested trailer: %s\n", r.SuggestTrailer)
	}
}

func firstCommitLine(message string) string {
	for _, line := range strings.Split(message, "\n") {
		line = strings.TrimSpace(line)
		if line != "" {
			return line
		}
	}
	return ""
}
