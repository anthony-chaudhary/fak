package safecommit

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/gitgate"
)

// ReasonCachedRemoveWorktreePresent is emitted when a requested path is staged as an
// index deletion while the working-tree file still exists. The common source is
// `git rm --cached <path>` followed by a path-scoped commit: `git commit -- <path>`
// reads the still-present worktree copy and clears the deletion instead of recording
// an untrack operation.
const ReasonCachedRemoveWorktreePresent = "CACHED_REMOVE_WORKTREE_PRESENT"

const cachedRemoveEnvVar = "FAK_CACHED_REMOVE_GUARD"

// cachedRemoveGuardMode reads FAK_CACHED_REMOVE_GUARD. Default is block; warn records the
// would-be refusal in Result.Detail and proceeds; off disables the guard for a deliberate
// one-shot workflow.
func cachedRemoveGuardMode() staleBaseMode {
	switch strings.ToLower(strings.TrimSpace(os.Getenv(cachedRemoveEnvVar))) {
	case "off", "0", "false":
		return staleBaseOff
	case "warn", "advisory":
		return staleBaseWarn
	default:
		return staleBaseBlock
	}
}

func checkCachedRemoveWorktreePresent(ctx context.Context, run Runner, dir string, paths []string) (detail string, fired bool) {
	diffArgs := append([]string{"diff", "--cached", "--name-status", "--"}, paths...)
	out, code, err := run(ctx, dir, diffArgs...)
	if err != nil || code != 0 {
		return "", false
	}

	deleted := cachedDeletedPaths(out, paths)
	if len(deleted) == 0 {
		return "", false
	}

	root := worktreeRoot(ctx, run, dir)
	var present []string
	for _, p := range deleted {
		if worktreePathExists(root, p) {
			present = append(present, p)
		}
	}
	if len(present) == 0 {
		return "", false
	}

	sort.Strings(present)
	present = dedupeStrings(present)
	return fmt.Sprintf(
		"path(s) are staged as an index deletion while the working-tree file still exists. "+
			"A pathspec commit would read or re-stage that worktree copy and clear the deletion instead of recording it. "+
			"To record a real deletion, remove or move the working file and rerun. "+
			"To intentionally untrack while keeping a local copy, add or keep a .gitignore rule and use an index-preserving untrack commit flow; "+
			"do not use a pathspec commit while the file exists. Paths: %s",
		strings.Join(present, ", "),
	), true
}

func cachedDeletedPaths(nameStatus string, requested []string) []string {
	var deleted []string
	for _, line := range strings.Split(nameStatus, "\n") {
		status, path := parseNameStatusDelete(line)
		if status != "D" {
			continue
		}
		p, ok := gitgate.CleanRepoPath(path)
		if !ok || !gitgate.CoveredByAnyTree(p, requested) {
			continue
		}
		deleted = append(deleted, p)
	}
	sort.Strings(deleted)
	return dedupeStrings(deleted)
}

func parseNameStatusDelete(line string) (status, path string) {
	line = strings.TrimSpace(line)
	if line == "" {
		return "", ""
	}
	fields := strings.Split(line, "\t")
	if len(fields) >= 2 {
		return strings.TrimSpace(fields[0]), strings.TrimSpace(fields[1])
	}
	fields = strings.Fields(line)
	if len(fields) >= 2 {
		return strings.TrimSpace(fields[0]), strings.TrimSpace(fields[1])
	}
	return "", ""
}

func worktreeRoot(ctx context.Context, run Runner, dir string) string {
	if strings.TrimSpace(dir) != "" {
		return dir
	}
	out, code, err := run(ctx, dir, "rev-parse", "--show-toplevel")
	if err == nil && code == 0 && strings.TrimSpace(out) != "" {
		return strings.TrimSpace(out)
	}
	return "."
}

func worktreePathExists(root, repoPath string) bool {
	if strings.TrimSpace(root) == "" {
		root = "."
	}
	_, err := os.Lstat(filepath.Join(root, filepath.FromSlash(repoPath)))
	return err == nil
}
