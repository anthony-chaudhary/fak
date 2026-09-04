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
//
// Postcondition: Returns true if any path segment matches ExcludeDirs, false otherwise.
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
//
// Postcondition: Returns total newline count plus one if the final line lacks a trailing newline.
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

// Violation records an observed source file exceeding HardCeiling or its pinned ratchet baseline limit.
type Violation struct {
	Path  string
	Lines int
	Cap   int    // the ceiling it broke: HardCeiling for a new god-file, the pinned cap for a grown pin
	Over  int    // Lines - Cap
	Kind  string // "new-god-file" or "grew-past-cap"
}

// Shrunk identifies a pinned baseline file whose current physical line count has dropped below its cap.
type Shrunk struct {
	Path  string
	Lines int
	Cap   int
	Under int // Cap - Lines
}

// Verdict encapsulates the evaluation outcome of repository files checked against the hard ceiling and baseline.
type Verdict struct {
	OK         bool
	NFiles     int
	NPinned    int
	Violations []Violation // grew-past-cap first, then new-god-file — the failing set
	Shrunk     []Shrunk    // pinned files now under their cap (ratchet-down opportunities)
	StalePins  []string    // baseline entries whose file no longer exists / is now excluded
}

// Evaluate applies the two rules to a measured {path: lines} tree against caps (the pinned
// baseline). Pure: no I/O, deterministic, sorted output — the unit-testable core.
//
// Precondition: measured maps repo-relative file paths to physical line counts; caps provides pinned ratchet thresholds.
// Postcondition: Returns an OK verdict only when zero violations are detected across all measured files.
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
//
// Postcondition: Returns a candidate map containing only files whose line count exceeds HardCeiling.
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
//
// Precondition: The old baseline map must provide existing caps for all previously admitted god-files.
// Invariant: Ratchet caps can only decrease monotonically and new over-ceiling files are strictly refused.
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
//
// Precondition: Root directory must be an initialized git repository containing tracked source files.
// Postcondition: Returns physical line counts for all first-party non-test Go source files under root.
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
		// Exclude test files from god-file measurements; they churn per package test additions.
		if strings.HasSuffix(rel, "_test.go") {
			continue
		}
		data, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(rel)))
		if err != nil {
			// Deleted working-tree files are skipped; Evaluate flags them as stale pins.
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
//
// Postcondition: Returns formatted Go source code declaring Baseline with lexicographically sorted paths.
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
	b.WriteString("// Baseline pins today's god-files (> HardCeiling lines) at their current LOC. A\n")
	b.WriteString("// pinned file may only shrink; an unpinned file may not exceed HardCeiling. See doc.go.\n")
	b.WriteString("var Baseline = map[string]int{\n")
	for _, rel := range names {
		b.WriteString(fmt.Sprintf("\t%q: %d,\n", rel, caps[rel]))
	}
	b.WriteString("}\n")
	return b.String()
}
