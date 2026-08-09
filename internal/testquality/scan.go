package testquality

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// skipDirs are never scanned. testdata is the important one and it cuts both
// ways: this package's own fixtures are deliberately broken tests, so a scan that
// walked them would permanently report findings against itself.
var skipDirs = map[string]bool{"testdata": true, "vendor": true, "node_modules": true}

// TrackedTestFiles lists the git-TRACKED *_test.go paths under root, sorted and
// slash-separated.
//
// Tracked, not walked. fak's checkout is shared by ~20 concurrent agent sessions,
// so the working tree carries files that are in nobody's commit; a gate whose
// verdict is a function of peer WIP refuses work the author did not do. It is the
// same argument internal/pythongate makes for `git ls-files tools/*.py`.
//
// An empty result is an ERROR, not a clean tree. A scan whose corpus went to zero
// (wrong root, a git that failed silently) reports zero findings, which is
// indistinguishable from a tree with none — the false-negative shape this whole
// package exists to refuse.
func TrackedTestFiles(root string) ([]string, error) {
	cmd := exec.Command("git", "ls-files", "-z", "*_test.go")
	windowgate.ConfigureBackgroundCommand(cmd)
	cmd.Dir = root
	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("git ls-files *_test.go in %s: %w", root, err)
	}
	var paths []string
	for _, p := range strings.Split(string(out), "\x00") {
		p = strings.TrimSpace(strings.ReplaceAll(p, "\\", "/"))
		if p == "" || IsSkipped(p) {
			continue
		}
		paths = append(paths, p)
	}
	if len(paths) == 0 {
		return nil, fmt.Errorf("no tracked *_test.go under %s: an empty corpus reports zero findings, "+
			"which is not distinguishable from a tree that has none", root)
	}
	sort.Strings(paths)
	return paths, nil
}

// IsSkipped reports whether a repo-relative path sits under a directory the scan
// never judges. Exported so the hooks gate filters the tracked tree the same way
// the tool does, instead of maintaining a second, drifting copy of the rule.
func IsSkipped(rel string) bool {
	for _, seg := range strings.Split(rel, "/") {
		if skipDirs[seg] {
			return true
		}
	}
	return false
}

// Scan analyzes every named file (repo-relative, slash-separated) under root and
// returns the findings in a deterministic order: by path, then by Analyze's
// within-file order.
//
// A file that cannot be read or parsed aborts the scan with an error. A partial
// scan reported as a result would understate the tree and, through the ratchet,
// silently lower the floor on regeneration.
func Scan(root string, files []string) ([]Finding, error) {
	sorted := append([]string(nil), files...)
	sort.Strings(sorted)
	var out []Finding
	for _, rel := range sorted {
		src, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(rel)))
		if err != nil {
			return nil, fmt.Errorf("read %s: %w", rel, err)
		}
		got, err := Analyze(rel, src)
		if err != nil {
			return nil, err
		}
		out = append(out, got...)
	}
	return out, nil
}

// ScanTree is TrackedTestFiles + Scan: the whole-tree corpus and its findings.
func ScanTree(root string) ([]Finding, []string, error) {
	files, err := TrackedTestFiles(root)
	if err != nil {
		return nil, nil, err
	}
	findings, err := Scan(root, files)
	if err != nil {
		return nil, nil, err
	}
	return findings, files, nil
}

// LoadBaseline reads and parses the tracked floor under root. A MISSING baseline
// is reported as such (nil, false, nil) rather than as an empty one: an absent
// floor makes every candidate "new", which is the honest reading but must be said
// out loud, because otherwise it looks like a sudden regression in the diff of
// whoever happens to run next. An unparseable baseline is a hard error — see
// ParseBaseline.
func LoadBaseline(root string) (Baseline, bool, error) {
	src, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(BaselineFile)))
	if os.IsNotExist(err) {
		return nil, false, nil
	}
	if err != nil {
		return nil, false, fmt.Errorf("read %s: %w", BaselineFile, err)
	}
	b, perr := ParseBaseline(src)
	if perr != nil {
		return nil, false, fmt.Errorf("%s: %w", BaselineFile, perr)
	}
	return b, true, nil
}
