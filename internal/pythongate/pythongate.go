package pythongate

import (
	"fmt"
	"os/exec"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// ReasonNewPythonTool is the closed-vocabulary refusal code for a tools/*.py that is not
// in the grandfathered baseline — the structured, machine-checkable form the ratchet
// refuses with instead of free text.
const ReasonNewPythonTool = "NEW_PYTHON_TOOL"

// Offense is one tracked tools/*.py path that is NOT in the grandfathered baseline:
// a new Python tool the de-Python ratchet refuses.
type Offense struct {
	Path string // repo-relative path, e.g. "tools/new_thing.py"
}

// String renders the offense as a one-line port-it-to-Go report carrying the reason code.
func (o Offense) String() string {
	return fmt.Sprintf("%s is a NEW python tool; port it to Go instead (%s)", o.Path, ReasonNewPythonTool)
}

// ScanTree lists the tracked tools/*.py paths in repoRoot (via `git ls-files tools/*.py`,
// cwd=repoRoot) and returns one Offense per path that is NOT in the grandfathered
// baseline, sorted for stable output. A path in the baseline that no longer exists is
// simply absent from git ls-files and produces nothing — the ratchet never complains
// about a tool that was ported away.
func ScanTree(repoRoot string) ([]Offense, error) {
	tracked, err := trackedPyTools(repoRoot)
	if err != nil {
		return nil, err
	}
	return offensesAgainst(tracked, baselineSet()), nil
}

// offensesAgainst is the pure ratchet core, split out so it can be unit-tested on a
// synthetic tracked set + allowlist: one Offense per tracked path NOT in allowed, sorted.
func offensesAgainst(tracked []string, allowed map[string]bool) []Offense {
	var offenses []Offense
	for _, p := range tracked {
		if !allowed[p] {
			offenses = append(offenses, Offense{Path: p})
		}
	}
	sort.Slice(offenses, func(i, j int) bool { return offenses[i].Path < offenses[j].Path })
	return offenses
}

// trackedPyTools shells out to git for the authoritative tracked-file list. Using
// git ls-files (rather than a filesystem walk) means an untracked scratch .py in tools/
// is correctly ignored — only files that would actually ship count.
func trackedPyTools(repoRoot string) ([]string, error) {
	cmd := exec.Command("git", "ls-files", "tools/*.py")
	windowgate.ConfigureBackgroundCommand(cmd)
	cmd.Dir = repoRoot
	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("git ls-files tools/*.py in %s: %w", repoRoot, err)
	}
	var paths []string
	for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		// Normalize to forward slashes; git already emits them, but be defensive.
		paths = append(paths, strings.ReplaceAll(line, "\\", "/"))
	}
	return paths, nil
}

// baselineSet materializes the grandfathered slice into a lookup set.
func baselineSet() map[string]bool {
	set := make(map[string]bool, len(grandfathered))
	for _, p := range grandfathered {
		set[p] = true
	}
	return set
}
