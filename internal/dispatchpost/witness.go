package dispatchpost

import (
	"context"
	"os/exec"
	"strings"
)

// HeadSHA returns the short HEAD sha of the git repo at dir, or "" if git is
// unavailable or dir is not a repo. It never errors to the caller: a missing witness
// is reported as an empty string (the card then omits the HEAD delta) rather than
// failing the dispatch — the run's outcome must surface even when git does not.
func HeadSHA(ctx context.Context, dir string) string {
	out, err := runGit(ctx, dir, "rev-parse", "--short", "HEAD")
	if err != nil {
		return ""
	}
	return strings.TrimSpace(out)
}

// CommitsBetween returns the `git log --oneline before..after` subjects — the commits
// the dispatch landed, the WITNESS for the result card. before/after are HEAD shas
// captured around the run. An empty or equal range yields nil (no commits landed),
// which the renderer surfaces honestly as "no commit landed". A git error yields nil
// too: the absence of a witness is reported as "no witnessed change", never inferred
// as success.
func CommitsBetween(ctx context.Context, dir, before, after string) []string {
	before = strings.TrimSpace(before)
	after = strings.TrimSpace(after)
	if before == "" || after == "" || before == after {
		return nil
	}
	out, err := runGit(ctx, dir, "log", "--oneline", "--no-decorate", before+".."+after)
	if err != nil {
		return nil
	}
	var commits []string
	for _, ln := range strings.Split(strings.TrimSpace(out), "\n") {
		ln = strings.TrimSpace(ln)
		if ln != "" {
			commits = append(commits, ln)
		}
	}
	return commits
}

// runGit runs a git subcommand in dir and returns its stdout. It is the single seam
// the witness helpers share; tests that need a deterministic delta exercise the
// renderer with a hand-built Result rather than shelling out.
func runGit(ctx context.Context, dir string, args ...string) (string, error) {
	cmd := exec.CommandContext(ctx, "git", args...)
	if dir != "" {
		cmd.Dir = dir
	}
	out, err := cmd.Output()
	if err != nil {
		return "", err
	}
	return string(out), nil
}
