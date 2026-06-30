package corelockaudit

import (
	"fmt"
	"os/exec"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// ChangedPaths returns the repo-relative paths changed between sinceRef and HEAD,
// by shelling to `git diff --name-only <sinceRef>...HEAD` in workspace. This is
// the thin I/O layer kept apart from the pure Audit fold so the classification
// is unit-testable without git.
//
// The `A...B` (three-dot) form diffs B against the merge-base of A and B, which
// is the right "what did this branch change" set for an audit against a base
// ref. Output is de-duplicated and sorted for determinism. A git failure is
// returned with its stderr so the caller can surface a concrete cause.
func ChangedPaths(workspace, sinceRef string) ([]string, error) {
	ref := strings.TrimSpace(sinceRef)
	if ref == "" {
		return nil, fmt.Errorf("corelockaudit: empty since-ref")
	}
	cmd := exec.Command("git", "diff", "--name-only", ref+"...HEAD")
	if workspace != "" {
		cmd.Dir = workspace
	}
	// Suppress the Windows console-window flash a windowless dispatch parent would
	// otherwise pop when this git probe runs (windowgate's UNSUPPRESSED_GO_EXEC gate).
	windowgate.ConfigureBackgroundCommand(cmd)
	out, err := cmd.Output()
	if err != nil {
		if ee, ok := err.(*exec.ExitError); ok {
			return nil, fmt.Errorf("corelockaudit: git diff failed: %s: %w",
				strings.TrimSpace(string(ee.Stderr)), err)
		}
		return nil, fmt.Errorf("corelockaudit: git diff failed: %w", err)
	}
	return splitChanged(string(out)), nil
}

// splitChanged turns newline-separated git output into a de-duplicated, sorted
// path slice. Split out so it is testable without git.
func splitChanged(raw string) []string {
	seen := map[string]bool{}
	var paths []string
	for _, line := range strings.Split(strings.ReplaceAll(raw, "\r\n", "\n"), "\n") {
		p := strings.TrimSpace(line)
		if p == "" || seen[p] {
			continue
		}
		seen[p] = true
		paths = append(paths, p)
	}
	sort.Strings(paths)
	return paths
}
