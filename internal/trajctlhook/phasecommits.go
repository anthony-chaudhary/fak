package trajctlhook

// phasecommits.go — issue #3129, the producer half of the trajectory-control
// spine (epic #2533): the impure `git log` walk that reads the commit trailers a
// session writes and hands the pure fold (trajctl.PhaseCommitsFromTrailers) the
// commits to assemble EvidenceWindow.PhaseCommits.
//
// This closes the second dogfood gap
// (docs/notes/TRAJCTL-DOGFOOD-2026-07-07.md, miss #2): a wired turn-end sampler
// with no phase→commit bindings scores W3=0 forever. GitPhaseCommits reads the
// bindings from git so a live session's turn-end pass actually credits witnessed
// progress. It lives HERE, not in tier-1 trajctl, because it shells to git — the
// same seam GitEvidenceResolver already shells from; trajctl stays pure.

import (
	"os/exec"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/trajctl"
)

// PhaseCommitScanDefault is the number of most-recent commits GitPhaseCommits
// walks when the caller does not name a range. The turn-end producer only needs
// the recent history that could carry a plan-phase trailer; bounding the walk
// keeps the turn-boundary pass cheap regardless of repo size.
const PhaseCommitScanDefault = 500

// GitPhaseCommits walks the git history at root and folds every
// `Trajctl-Phase: <phase-id>` commit trailer into the phase→commit map an
// EvidenceWindow carries. It shells:
//
//	git -C <root> log --format=%H%x00%B%x00 -n <limit> HEAD
//
// and hands the pre-read (SHA, message) pairs to the pure
// trajctl.PhaseCommitsFromTrailers fold. limit <= 0 defaults to
// PhaseCommitScanDefault; a non-positive/zero limit never means "all of history"
// (an unbounded walk would be an unbounded turn-boundary cost).
//
// It is fail-soft by contract, matching the turn-boundary posture: a git that is
// absent, errors, or an empty/rootless repo yields a nil map (no bindings), so a
// caller scores 0 rather than crediting unverified work — never a panic, never a
// raised error. If root is empty git runs in the process cwd, matching plain
// `git` behavior.
func GitPhaseCommits(root string, limit int) map[string][]string {
	commits := gitTrailerCommits(root, limit)
	if len(commits) == 0 {
		return nil
	}
	return trajctl.PhaseCommitsFromTrailers(commits)
}

// LiveTurnEnd is the live-session producer entry point: it assembles the full
// turn-end evidence window from the git repo at root — the phase→commit bindings
// from commit trailers (GitPhaseCommits) AND the git resolver that re-verifies
// each candidate SHA (GitEvidenceResolver) — and drives RunTurnEnd against the
// ledger at path. This is the one call a running session's turn-end hook makes:
// it turns the wired-but-empty sampler (which scored W3=0 because nothing built
// PhaseCommits) into a producer that credits witnessed progress from the session's
// own commits.
//
// It is fail-open on the same contract as RunTurnEnd: a repo without git, an
// unreadable ledger, or no bound phase folds to a zero/empty sample rather than a
// panic. sessionPaths are the transcript files (if any) the W2 stall scorer
// analyzes; they may be nil. unixMillis stamps the produced rows (0 lets the
// append path stamp).
func LiveTurnEnd(path, root string, sessionPaths []string, unixMillis int64, stamp trajctl.Stamp) Result {
	return RunTurnEnd(path, WindowInput{
		PhaseCommits: GitPhaseCommits(root, 0),
		SessionPaths: sessionPaths,
		Resolve:      GitEvidenceResolver(root),
		UnixMillis:   unixMillis,
	}, stamp)
}

// gitTrailerCommits reads the recent commits at root as (SHA, full message)
// pairs via `git log --format=%H%x00%B%x00`. Fail-soft: any git error or missing
// binary yields nil. The NUL record separator makes the parse robust against
// multi-line commit bodies (the trailer lives in the body).
func gitTrailerCommits(root string, limit int) []trajctl.TrailerCommit {
	if limit <= 0 {
		limit = PhaseCommitScanDefault
	}
	args := []string{}
	if strings.TrimSpace(root) != "" {
		args = append(args, "-C", root)
	}
	args = append(args, "log", "--format=%H%x00%B%x00", "-n", strconv.Itoa(limit), "HEAD")
	out, err := exec.Command("git", args...).Output()
	if err != nil {
		return nil
	}
	parts := strings.Split(string(out), "\x00")
	var commits []trajctl.TrailerCommit
	// Records are emitted as SHA, NUL, body, NUL — so parts pair up as
	// [sha, body, sha, body, ...] with a trailing empty element.
	for i := 0; i+1 < len(parts); i += 2 {
		sha := strings.TrimSpace(parts[i])
		if sha == "" {
			continue
		}
		commits = append(commits, trajctl.TrailerCommit{SHA: sha, Message: parts[i+1]})
	}
	return commits
}
