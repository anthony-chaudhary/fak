package godfileceiling

import (
	"bufio"
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// Invariant: godfile ceiling enforces max lines and file size boundaries fail-closed without allocations in core evaluation loops.
// Contract: Evaluate and Repin are pure, deterministic functions that perform no filesystem I/O.
// Contract: Ratchet operations are strictly monotonic; caps can only decrease and new offenders cannot be admitted into baseline.

// HardCeiling is the global LOC ceiling: a tracked .go file not in the Baseline may not
// exceed it. It matches tools/code_quality_scorecard.py's FILE_HARD_MAX so the gate and
// the scorecard call the same thing a god-file.
const HardCeiling = 1500

// ExcludeDirs are the path segments whose .go files are NOT first-party shipped code and
// so are not measured — testdata fixtures, vendored/generated trees, and the
// agent-machinery checkout dirs that hold full repo COPIES (.claude worktrees, .fak/.dos/
// .tmp checkouts). A copy is identical to its tracked source, so counting it would
// double-grade the same file. Kept in sync with code_quality_scorecard.py's
// GO_EXCLUDE_DIRS.
var ExcludeDirs = map[string]bool{
	".git": true, ".claude": true, ".fak": true, ".dos": true, ".tmp": true,
	"node_modules": true, "testdata": true, "vendor": true, "__pycache__": true,
}

// Excluded reports whether a repo-relative path lies under an excluded tree.
func Excluded(rel string) bool {
	for _, seg := range strings.Split(filepath.ToSlash(rel), "/") {
		if ExcludeDirs[seg] {
			return true
		}
	}
	return false
}

// LineCount returns the physical line count of text, matching the scorecard's
// len(text.splitlines()): the number of "\n"-separated lines, with a final line that has
// no trailing newline still counted.
func LineCount(text []byte) int {
	if len(text) == 0 {
		return 0
	}
	n := bytes.Count(text, []byte{'\n'})
	if text[len(text)-1] != '\n' {
		n++ // a final line without a trailing newline is still a line
	}
	return n
}

// Violation represents a single tracked source file exceeding either the global LOC ceiling or its pinned ratchet baseline.
type Violation struct {
	// Path is the repo-relative slash-delimited path to the violating file.
	Path string
	// Lines is the measured physical line count of the file.
	Lines int
	// Cap is the maximum line count allowed (HardCeiling for a new god-file, or the pinned cap for a grown pin).
	Cap int
	// Over is the number of lines exceeding Cap (Lines - Cap).
	Over int
	// Kind classifies the violation: "new-god-file" or "grew-past-cap".
	Kind string
}

// Shrunk records a pinned god-file whose physical line count has dropped below its baseline cap.
type Shrunk struct {
	// Path is the repo-relative slash-delimited path to the shrunk file.
	Path string
	// Lines is the measured physical line count of the file.
	Lines int
	// Cap is the previous baseline cap for the file.
	Cap int
	// Under is the delta below Cap (Cap - Lines), indicating ratcheting potential.
	Under int
}

// Verdict represents the deterministic evaluation result of applying god-file ceiling rules to a measured tree.
type Verdict struct {
	// OK indicates whether all measured files comply with the ceiling and ratchet rules without violations.
	OK bool
	// NFiles is the total number of non-excluded source files evaluated.
	NFiles int
	// NPinned is the total number of files tracked in the baseline cap map.
	NPinned int
	// Violations enumerates files breaching either the global ceiling or pinned caps, sorted deterministically.
	Violations []Violation
	// Shrunk lists baseline files that have decreased in line count and can be ratcheted down.
	Shrunk []Shrunk
	// StalePins lists baseline paths that are no longer present or are now excluded.
	StalePins []string
}

// Evaluate applies the two rules to a measured {path: lines} tree against caps (the pinned
// baseline). Pure: no I/O, deterministic, sorted output — the unit-testable core.
func Evaluate(measured map[string]int, caps map[string]int) Verdict {
	var grew, newGod []Violation
	var shrunk []Shrunk

	paths := make([]string, 0, len(measured))
	for p := range measured {
		paths = append(paths, p)
	}
	sort.Strings(paths)

	for _, rel := range paths {
		n := measured[rel]
		if cap, pinned := caps[rel]; pinned {
			switch {
			case n > cap:
				grew = append(grew, Violation{rel, n, cap, n - cap, "grew-past-cap"})
			case n < cap:
				shrunk = append(shrunk, Shrunk{rel, n, cap, cap - n})
			}
		} else if n > HardCeiling {
			newGod = append(newGod, Violation{rel, n, HardCeiling, n - HardCeiling, "new-god-file"})
		}
	}

	var stale []string
	for rel := range caps {
		if _, ok := measured[rel]; !ok {
			stale = append(stale, rel)
		}
	}
	sort.Strings(stale)

	violations := append(append([]Violation{}, grew...), newGod...)
	return Verdict{
		OK:         len(violations) == 0,
		NFiles:     len(measured),
		NPinned:    len(caps),
		Violations: violations,
		Shrunk:     shrunk,
		StalePins:  stale,
	}
}

// ProposeBaseline recomputes the caps map from a measured tree: every file over
// HardCeiling, pinned at its current line count. This is the ratchet's candidate baseline.
func ProposeBaseline(measured map[string]int) map[string]int {
	caps := map[string]int{}
	for rel, n := range measured {
		if n > HardCeiling {
			caps[rel] = n
		}
	}
	return caps
}

// Repin validates a proposed baseline against the current one: RATCHET-DOWN ONLY. It
// returns the accepted new baseline and no refusals when every proposed cap is <= the old
// cap for a file already pinned; it refuses (non-empty refusals, nil baseline) if the
// proposal would RAISE any cap or pin a brand-new over-ceiling file — both would defeat
// the ratchet. A new god-file must be caught by Evaluate and split, never pinned.
func Repin(measured map[string]int, old map[string]int) (accepted map[string]int, refusals []string) {
	proposed := ProposeBaseline(measured)
	names := make([]string, 0, len(proposed))
	for rel := range proposed {
		names = append(names, rel)
	}
	sort.Strings(names)
	for _, rel := range names {
		n := proposed[rel]
		oldCap, was := old[rel]
		switch {
		case !was:
			refusals = append(refusals, fmt.Sprintf(
				"%s (%d lines) is a NEW file over the ceiling — the gate must catch it, not pin it; split the file instead of re-pinning",
				rel, n))
		case n > oldCap:
			refusals = append(refusals, fmt.Sprintf(
				"%s would RAISE its cap %d -> %d; the ratchet only goes down — split the file instead of re-pinning",
				rel, oldCap, n))
		}
	}
	if len(refusals) > 0 {
		return nil, refusals
	}
	return proposed, nil
}

// MeasureTree returns {rel: lines} for every counted tracked .go file under root. It shells
// to `git ls-files '*.go'` (tracked set) and counts physical lines, skipping ExcludeDirs.
// This is the impure shell; Evaluate/Repin hold the testable logic.
func MeasureTree(root string) (map[string]int, error) {
	cmd := exec.Command("git", "ls-files", "*.go")
	cmd.Dir = root
	windowgate.ConfigureBackgroundCommand(cmd)
	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("git ls-files: %w", err)
	}
	measured := map[string]int{}
	sc := bufio.NewScanner(bytes.NewReader(out))
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for sc.Scan() {
		rel := filepath.ToSlash(strings.TrimSpace(sc.Text()))
		if rel == "" || Excluded(rel) {
			continue
		}
		// _test.go files are graded by the tests KPI, not architecture — they churn
		// constantly (every new leaf appends a row to a shared *_test.go), so pinning
		// them reds the gate on unrelated growth. Match internal/hooks/gate_godfile.go
		// and tools/code_quality_scorecard.py, which both exclude tests from the
		// architecture corpus.
		if strings.HasSuffix(rel, "_test.go") {
			continue
		}
		data, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(rel)))
		if err != nil {
			// A tracked path we cannot read (deleted in the working tree) is skipped, not
			// fatal: Evaluate surfaces its baseline entry as a stale pin instead.
			continue
		}
		measured[rel] = LineCount(data)
	}
	if err := sc.Err(); err != nil {
		return nil, fmt.Errorf("scan git ls-files: %w", err)
	}
	return measured, nil
}

// FormatBaseline renders a caps map as the Go source of baseline.go — the ratchet's
// regenerate output. Sorted by path for a stable diff.
func FormatBaseline(caps map[string]int) string {
	names := make([]string, 0, len(caps))
	for rel := range caps {
		names = append(names, rel)
	}
	sort.Strings(names)
	var b strings.Builder
	b.WriteString("// Code generated from a MeasureTree->Repin->FormatBaseline pass. DO NOT EDIT by hand.\n")
	b.WriteString("// Regenerate only to TIGHTEN after a god-file shrinks; never to raise a cap.\n\n")
	b.WriteString("package godfileceiling\n\n")
	b.WriteString("// Invariant: baseline caps are monotonically non-increasing and pin only first-party files exceeding HardCeiling.\n\n")
	b.WriteString("// Baseline pins today's god-files (> HardCeiling lines) at their current LOC. A\n")
	b.WriteString("// pinned file may only shrink; an unpinned file may not exceed HardCeiling. See doc.go.\n")
	b.WriteString("var Baseline = map[string]int{\n")
	for _, rel := range names {
		b.WriteString(fmt.Sprintf("\t%q: %d,\n", rel, caps[rel]))
	}
	b.WriteString("}\n")
	return b.String()
}
