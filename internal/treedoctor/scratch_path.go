package treedoctor

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

const scratchNamespace = "_scratch"

// PrepareScratchDir creates and returns an absolute directory below the repository's
// ignored scratch namespace. rel is a logical producer/run name such as "fleet-loop"
// or "coverage/unit". Absolute paths and parent traversal are refused so callers
// cannot accidentally turn a scratch redirect back into a repository-root artifact.
func PrepareScratchDir(repoRoot, rel string) (string, error) {
	root, err := filepath.Abs(repoRoot)
	if err != nil {
		return "", fmt.Errorf("resolve repository root: %w", err)
	}
	clean, err := cleanScratchRelative(rel)
	if err != nil {
		return "", err
	}
	dir := filepath.Join(root, scratchNamespace, clean)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", fmt.Errorf("create scratch directory %s: %w", dir, err)
	}
	return dir, nil
}

// PrepareScratchPath creates the parent directory and returns an absolute file path
// below _scratch. rel must include both a producer directory and a filename, which
// keeps unrelated generated captures from accumulating in one flat bucket.
func PrepareScratchPath(repoRoot, rel string) (string, error) {
	clean, err := cleanScratchRelative(rel)
	if err != nil {
		return "", err
	}
	if filepath.Dir(clean) == "." {
		return "", fmt.Errorf("scratch path %q needs a producer directory (for example fleet-loop/tick.json)", rel)
	}
	dir, err := PrepareScratchDir(repoRoot, filepath.Dir(clean))
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, filepath.Base(clean)), nil
}

func cleanScratchRelative(rel string) (string, error) {
	rel = strings.TrimSpace(rel)
	if rel == "" {
		return "", fmt.Errorf("scratch location is empty")
	}
	if filepath.IsAbs(rel) {
		return "", fmt.Errorf("scratch location %q must be repository-relative", rel)
	}
	clean := filepath.Clean(filepath.FromSlash(rel))
	if clean == "." || clean == ".." || strings.HasPrefix(clean, ".."+string(filepath.Separator)) {
		return "", fmt.Errorf("scratch location %q escapes the scratch namespace", rel)
	}
	if first := strings.Split(clean, string(filepath.Separator))[0]; first == scratchNamespace {
		return "", fmt.Errorf("scratch location %q must be relative to _scratch, not include it", rel)
	}
	return clean, nil
}
