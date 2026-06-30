package safecommit

import (
	"context"
	"fmt"
	"os"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/gitgate"
)

// ReasonPreStagedPathOverlap is emitted when a requested path already has index changes
// before fak commit does its own path-scoped staging. In a shared trunk, that is a same-file
// ownership ambiguity: `git add --all -- <path>` would fold those pre-staged hunks into this
// commit even if they came from a peer.
const ReasonPreStagedPathOverlap = "PRESTAGED_PATH_OVERLAP"

const preStagedPathEnvVar = "FAK_PRESTAGED_PATH_GUARD"

// preStagedPathGuardMode reads FAK_PRESTAGED_PATH_GUARD. Default is block. warn records the
// would-be refusal in Result.Detail and proceeds; off disables the guard for an intentional
// one-shot staged-path commit.
func preStagedPathGuardMode() staleBaseMode {
	switch strings.ToLower(strings.TrimSpace(os.Getenv(preStagedPathEnvVar))) {
	case "off", "0", "false":
		return staleBaseOff
	case "warn", "advisory":
		return staleBaseWarn
	default:
		return staleBaseBlock
	}
}

func checkPreStagedPathOverlap(ctx context.Context, run Runner, dir string, paths []string) (detail string, fired bool) {
	statusArgs := append([]string{"status", "--porcelain", "--"}, paths...)
	out, code, err := run(ctx, dir, statusArgs...)
	if err != nil || code != 0 {
		return "", false
	}
	return preStagedPathOverlapFromStatus(out, paths)
}

func preStagedPathOverlapFromStatus(status string, paths []string) (detail string, fired bool) {
	var staged []string
	for _, line := range strings.Split(status, "\n") {
		path, isStaged := parsePreStagedPathLine(line)
		if path == "" || !isStaged || !gitgate.CoveredByAnyTree(path, paths) {
			continue
		}
		staged = append(staged, path)
	}
	if len(staged) == 0 {
		return "", false
	}
	sort.Strings(staged)
	staged = dedupeStrings(staged)
	return fmt.Sprintf(
		"requested path(s) already have staged changes before fak commit stages the pathspec. "+
			"In a shared tree this is same-file ownership ambiguity: those index hunks may belong to a peer, "+
			"and `git add --all -- <paths>` would fold them into this commit. "+
			"Fix: git restore --staged -- %s (worktree edits stay), then rerun fak commit after the staged hunks are yours. Paths: %s",
		strings.Join(staged, " "), strings.Join(staged, ", "),
	), true
}

func parsePreStagedPathLine(line string) (path string, staged bool) {
	if len(line) < 4 {
		return "", false
	}
	xy := line[:2]
	rest := line[3:]
	if xy == "??" {
		return strings.TrimSpace(rest), false
	}
	if strings.Contains(rest, " -> ") {
		parts := strings.Split(rest, " -> ")
		rest = parts[len(parts)-1]
	}
	return strings.TrimSpace(rest), xy[0] != ' ' && xy[0] != 'D'
}

func dedupeStrings(in []string) []string {
	if len(in) < 2 {
		return in
	}
	out := in[:1]
	for _, s := range in[1:] {
		if s != out[len(out)-1] {
			out = append(out, s)
		}
	}
	return out
}
