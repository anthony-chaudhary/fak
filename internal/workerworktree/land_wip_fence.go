package workerworktree

import (
	"os"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/wipfence"
)

// stripWorktreeWIPFences removes ephemeral //go:build wip_<slug> build constraints from
// files in wtPath prior to capturing the landing diff.
func stripWorktreeWIPFences(wtPath string, paths []string) {
	if len(paths) > 0 {
		for _, p := range paths {
			target := filepath.Join(wtPath, filepath.FromSlash(p))
			if fi, err := os.Stat(target); err == nil {
				if fi.IsDir() {
					_ = filepath.Walk(target, func(sub string, info os.FileInfo, werr error) error {
						if werr == nil && !info.IsDir() && strings.HasSuffix(sub, ".go") {
							_ = wipfence.Strip(sub)
						}
						return nil
					})
				} else if strings.HasSuffix(target, ".go") {
					_ = wipfence.Strip(target)
				}
			}
		}
		return
	}
	_ = filepath.Walk(wtPath, func(sub string, info os.FileInfo, werr error) error {
		if werr != nil {
			return nil
		}
		if info.IsDir() && (info.Name() == ".git" || info.Name() == "vendor") {
			return filepath.SkipDir
		}
		if !info.IsDir() && strings.HasSuffix(sub, ".go") {
			_ = wipfence.Strip(sub)
		}
		return nil
	})
}

// prospectiveVerdictStillApplies reports whether a GREEN prospective verdict taken
// against verifiedBase still describes a candidate re-seeded onto newBase. It is true
// only when a closure is declared AND the trunk delta between the two bases is readable
// AND touches no path inside that closure; every other case re-verifies.
func prospectiveVerdictStillApplies(git GitRunner, root, verifiedBase, newBase string, closure []string) bool {
	verifiedBase, newBase = strings.TrimSpace(verifiedBase), strings.TrimSpace(newBase)
	if len(closure) == 0 || verifiedBase == "" || newBase == "" {
		return false
	}
	if verifiedBase == newBase {
		return true
	}
	rc, out := run(git, root, []string{"diff-tree", "-r", "--name-only", "--no-commit-id", verifiedBase, newBase})
	if rc != 0 {
		return false
	}
	for _, line := range strings.Split(out, "\n") {
		changed := normalizeFenceGlob(line)
		if changed == "" {
			continue
		}
		for _, glob := range closure {
			if matchFenceGlob(normalizeFenceGlob(glob), changed) {
				return false
			}
		}
	}
	return true
}
