package main

import (
	"fmt"
	"io"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/hooks"
)

// runCommitPreview lints a proposed commit (message + the paths it would touch) and reports the
// verdict WITHOUT running git. Exit 0 when nothing blocking was found, 1 otherwise.
func runCommitPreview(stdout, stderr io.Writer, message string, paths []string, root string, asJSON, requireIssue bool) int {
	rep := hooks.LintCommitMessageWithOptions(message, paths, root, requireIssue)
	if asJSON {
		if err := writeIndentedJSON(stdout, rep); err != nil {
			fmt.Fprintf(stderr, "fak commit: %v\n", err)
			return 1
		}
	} else {
		renderPreview(stdout, rep)
	}
	if rep.OK {
		return 0
	}
	return 1
}

func renderPreview(w io.Writer, r hooks.CommitLintReport) {
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
	fmt.Fprintf(w, "  issue link: resolving=%v", r.IssueResolving)
	if len(r.IssueRefs) > 0 {
		refs := make([]string, len(r.IssueRefs))
		for i, n := range r.IssueRefs {
			refs[i] = fmt.Sprintf("#%d", n)
		}
		fmt.Fprintf(w, " (refs %s)", strings.Join(refs, ", "))
	}
	fmt.Fprintln(w)
	for _, is := range r.Issues {
		fmt.Fprintf(w, "  ✗ %s\n", is)
	}
	for _, n := range r.Notes {
		fmt.Fprintf(w, "  · %s\n", n)
	}
	if !r.OK && r.SuggestTrailer != "" {
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
