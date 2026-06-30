package main

import (
	"fmt"
	"io"
	"os/exec"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/corelockaudit"
	"github.com/anthony-chaudhary/fak/internal/corelocks"
	"github.com/anthony-chaudhary/fak/internal/hooks"
)

// cmd/fak/corelockwarn.go — the ADVISORY core-lock surface for `fak hygiene` and `fak commit
// --preview` (issue #1682). It consumes the shipped, read-only fold in internal/corelockaudit
// (#1680) and the declarative taxonomy in internal/corelocks (#1681) AS-IS: this file only
// classifies a changed-path set and renders the warn-verdict findings as advisory output.
//
// Mode is WARNING — never blocking. A soft-contract / shadow-learn / coherence-bearing change
// produces a concise warning naming the exact witness command that would clear it, but the
// caller's exit code is untouched: the audit cannot fail a commit or a hygiene run in this
// phase. Ordinary leaf edits (open-leaf, the advisory-ok classes) stay quiet — only warn-verdict
// findings are surfaced.

// coreLockMode is the advisory/enforcement mode a finding was surfaced under. This phase only
// ever emits "warning"; the field exists so later metrics can distinguish a warned surface from
// an enforced one without a schema change.
const (
	coreLockModeWarning  = "warning"
	coreLockModeEnforced = "enforced"
)

// coreLockWarning is one advisory warning row, suitable for later metrics: the lock id, its
// class, the data-only reason token, the witness command(s) that would clear it, the paths that
// tripped it, and the mode it was surfaced under (always "warning" in this phase).
type coreLockWarning struct {
	LockID  string   `json:"lock_id"`
	Class   string   `json:"class"`
	Reason  string   `json:"reason"`
	Witness []string `json:"witness"`
	Paths   []string `json:"paths"`
	Mode    string   `json:"mode"`
}

// coreLockWarnings folds a corelockaudit.Report down to the advisory rows worth surfacing: the
// warn-verdict findings only. open-leaf and the advisory-ok classes are dropped so ordinary leaf
// edits stay quiet. The result is deterministic (findings are pre-sorted by the audit).
func coreLockWarnings(rep corelockaudit.Report) []coreLockWarning {
	var out []coreLockWarning
	for _, f := range rep.Findings {
		if f.Verdict != corelockaudit.VerdictWarn {
			continue
		}
		out = append(out, coreLockWarning{
			LockID:  f.LockID,
			Class:   f.Class,
			Reason:  f.ReasonToken,
			Witness: append([]string{}, f.RequiredWitnesses...),
			Paths:   append([]string{}, f.Paths...),
			Mode:    coreLockModeWarning,
		})
	}
	return out
}

// auditCoreLockPaths classifies a changed-path set against the shipped taxonomy and returns the
// advisory warnings. A taxonomy-load failure or an empty path set yields no warnings and no error
// — the advisory surface fails OPEN (it must never wedge a hygiene run or a commit preview).
func auditCoreLockPaths(paths []string) []coreLockWarning {
	if len(paths) == 0 {
		return nil
	}
	tax, err := corelocks.LoadFixture()
	if err != nil {
		// Fail open: a malformed shipped taxonomy is the corelocks package's own test problem,
		// never a reason to block this advisory pass.
		return nil
	}
	return coreLockWarnings(corelockaudit.Audit(tax, paths))
}

// renderCoreLockWarnings writes the advisory warnings in the human style hygiene/commit-preview
// use. It returns the number of warnings written so the caller can stay quiet when there are
// none. It NEVER changes the caller's exit code — these are advisory.
func renderCoreLockWarnings(w io.Writer, warnings []coreLockWarning) int {
	if len(warnings) == 0 {
		return 0
	}
	fmt.Fprintf(w, "core-lock advisory: %d soft-contract surface warning(s) (advisory — NOT blocking)\n", len(warnings))
	for _, cw := range warnings {
		fmt.Fprintf(w, "  ⚠ [%s] %s", cw.Class, cw.Reason)
		if len(cw.Paths) > 0 {
			fmt.Fprintf(w, " — %s", strings.Join(cw.Paths, ", "))
		}
		fmt.Fprintln(w)
		if len(cw.Witness) > 0 {
			fmt.Fprintf(w, "    witness to clear: %s\n", strings.Join(cw.Witness, "; "))
		}
	}
	return len(warnings)
}

// emitHygieneJSON renders the hygiene result as JSON: the blocking gate findings (the existing
// shape) PLUS the advisory core-lock warnings under a separate key, so a metrics consumer can read
// lock id / class / reason / witness / mode without confusing an advisory warning for a blocking
// finding. The warnings carry mode="warning" — they never affect the exit code.
func emitHygieneJSON(stdout, stderr io.Writer, findings []hooks.Finding, warnings []coreLockWarning) {
	if findings == nil {
		findings = []hooks.Finding{}
	}
	if warnings == nil {
		warnings = []coreLockWarning{}
	}
	payload := map[string]any{
		"findings":            findings,
		"count":               len(findings),
		"core_lock_warnings":  warnings,
		"core_lock_warn_mode": coreLockModeWarning,
	}
	if err := writeIndentedJSON(stdout, payload); err != nil {
		fmt.Fprintf(stderr, "fak hygiene: %v\n", err)
	}
}

// changedTreePaths returns the union of staged and unstaged working-tree changes (repo-relative,
// de-duplicated, sorted) for the whole-tree hygiene advisory. It is the "what has this checkout
// touched" set the warning mode classifies. A git failure yields no paths (fail open) — the
// advisory surface must never wedge hygiene.
func changedTreePaths(root string) []string {
	seen := map[string]bool{}
	for _, args := range [][]string{
		{"diff", "--name-only", "--cached"},
		{"diff", "--name-only"},
	} {
		cmd := exec.Command("git", args...)
		if root != "" {
			cmd.Dir = root
		}
		out, err := cmd.Output()
		if err != nil {
			continue
		}
		for _, line := range strings.Split(strings.ReplaceAll(string(out), "\r\n", "\n"), "\n") {
			p := strings.TrimSpace(line)
			if p != "" {
				seen[p] = true
			}
		}
	}
	paths := make([]string, 0, len(seen))
	for p := range seen {
		paths = append(paths, p)
	}
	sort.Strings(paths)
	return paths
}
