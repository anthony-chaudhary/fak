package main

import (
	"fmt"
	"io"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/hooks"
	"github.com/anthony-chaudhary/fak/internal/modver"
)

// runCommitPreview lints a proposed commit (message + the paths it would touch) and reports the
// verdict WITHOUT running git. Exit 0 when nothing blocking was found, 1 otherwise.
//
// It also surfaces the ADVISORY core-lock fold (issue #1682): the exact paths the commit would
// touch are classified against the shipped soft-contract taxonomy, and any coherence-bearing
// surface produces a path-scoped warning naming the witness command that would clear it. The
// warnings are WARNING MODE — they never change the exit code; only the lint verdict decides it.
func runCommitPreview(stdout, stderr io.Writer, message string, paths []string, root, expectedBranch string, asJSON, requireIssue bool) int {
	paths = commitPreviewEffectPaths(root, paths)
	rep := hooks.LintCommitMessageWithOptions(message, paths, root, requireIssue)
	coreLockWarns := auditCoreLockPaths(paths)
	// The modver cross-check (#2495): name the modules whose rev a commit of exactly these
	// staged paths would bump, so a stamp/lane mismatch ("subject stamps leaf X, but these
	// paths bump modules Y,Z") is visible at the one fixable moment. Pure path→module
	// projection (no git, no Snapshot), so it adds no latency beyond the string walk.
	bumpedModules := modver.ModulesForPaths(paths)
	if asJSON {
		payload := struct {
			hooks.CommitLintReport
			ExpectedBranch   string            `json:"expected_branch,omitempty"`
			CoreLockWarnings []coreLockWarning `json:"core_lock_warnings"`
			CoreLockWarnMode string            `json:"core_lock_warn_mode"`
			BumpedModules    []string          `json:"bumped_modules"`
			RequiredPaths    []string          `json:"required_paths"`
		}{
			CommitLintReport: rep,
			ExpectedBranch:   expectedBranch,
			CoreLockWarnings: coreLockWarns,
			CoreLockWarnMode: coreLockModeWarning,
			BumpedModules:    bumpedModules,
			RequiredPaths:    paths,
		}
		if err := writeIndentedJSON(stdout, payload); err != nil {
			fmt.Fprintf(stderr, "fak commit: %v\n", err)
			return 1
		}
	} else {
		renderPreview(stdout, rep, expectedBranch)
		renderModuleAdvisory(stdout, bumpedModules)
		if len(paths) > 0 {
			fmt.Fprintf(stdout, "  required paths: %s\n", strings.Join(paths, ", "))
		}
		renderCoreLockWarnings(stdout, coreLockWarns)
	}
	if rep.OK {
		return 0
	}
	return 1
}

func commitPreviewEffectPaths(root string, paths []string) []string {
	out := append([]string(nil), paths...)
	for _, path := range paths {
		rel := filepath.ToSlash(path)
		if filepath.IsAbs(path) {
			if r, err := filepath.Rel(root, path); err == nil {
				rel = filepath.ToSlash(r)
			}
		}
		if strings.HasPrefix(rel, "docs/notes/") && strings.HasSuffix(strings.ToLower(rel), ".md") {
			base := filepath.Base(rel)
			if len(base) >= 14 && base[len(base)-14] == '-' && !commitPreviewContainsPath(out, "INDEX.md") {
				out = append(out, "INDEX.md")
			}
		}
	}
	return out
}

func commitPreviewContainsPath(values []string, want string) bool {
	for _, value := range values {
		if filepath.ToSlash(value) == want {
			return true
		}
	}
	return false
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

// renderModuleAdvisory prints the version-everything cross-check (#2495): the modules whose rev a
// commit of exactly these staged paths would bump. It is ADVISORY — a lens beside the stamp/lane
// doctor ("subject stamps leaf X; these paths bump modules Y,Z"), never an exit-code gate. The
// module set is a pure, git-free projection of the staged paths (modver.ModulesForPaths), so it
// adds no latency beyond the string walk. Paths under no tracked keyspace bump no module, so the
// line is omitted rather than printed empty.
func renderModuleAdvisory(w io.Writer, bumped []string) {
	if len(bumped) == 0 {
		return
	}
	fmt.Fprintf(w, "  bumps modules: %s\n", strings.Join(bumped, ", "))
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
