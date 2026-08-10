package devcmd

import (
	"context"
	"fmt"
	"io"
	"os/exec"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/issuefanout"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// The witness-gathering shell behind `fak-dev issue fanout --coverage` (#2532).
//
// The leaf stays pure: everything here does is run the two witnesses (git for
// the leaf set and its spine artifacts, gh for the filed fan-out markers) and
// hand the raw rows to issuefanout.Coverage, which decides the standing. That
// split is what keeps the honesty meter honest — the scorecard reads git and the
// tracker, never a claim that the defaults were followed.

// fanoutCoverageGitTimeout bounds each git witness call. The scan is two reads
// over local history, so a slow answer means a wedged repo, not real work.
const fanoutCoverageGitTimeout = 60 * time.Second

// fanoutCoverageGHTimeout bounds the tracker export.
//
// It is much larger than the shared task-handoff gh timeout (30s) for the same
// reason DefaultCoverageScanCap is much larger than DefaultDedupeCap: a coverage
// export pulls thousands of issue bodies, not a bounded recent slice, and the
// short timeout turns that into a hard failure rather than a slow answer.
const fanoutCoverageGHTimeout = 5 * time.Minute

// fanoutCoverageDeps injects the two effectful runners so tests exercise the
// gather path without a real git or gh.
type fanoutCoverageDeps struct {
	git func(args []string) (stdout, stderr string, ok bool)
	gh  issueCreateRunner
}

// runFanoutCoverageGit runs one bounded git invocation in the current repo.
func runFanoutCoverageGit(args []string) (string, string, bool) {
	ctx, cancel := context.WithTimeout(context.Background(), fanoutCoverageGitTimeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, "git", args...)
	windowgate.ConfigureBackgroundCommand(cmd)
	var out, errb strings.Builder
	cmd.Stdout = &out
	cmd.Stderr = &errb
	err := cmd.Run()
	return out.String(), errb.String(), err == nil
}

// runFanoutCoverageGH runs the tracker export under the coverage timeout.
func runFanoutCoverageGH(args []string) (string, string, bool) {
	ctx, cancel := context.WithTimeout(context.Background(), fanoutCoverageGHTimeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, "gh", args...)
	windowgate.ConfigureBackgroundCommand(cmd)
	return runBoundedGHCmd(ctx, cmd, fanoutCoverageGHTimeout)
}

// gatherFanoutCoverage folds the real repo into the two-rate scorecard.
//
// The window is a git revision-range selector (`--since`), so the caller picks
// what "recently shipped" means rather than the tool baking in a horizon.
func gatherFanoutCoverage(since, repo string, scanCap int, deps fanoutCoverageDeps) (issuefanout.CoverageReport, error) {
	if scanCap <= 0 {
		scanCap = issuefanout.DefaultCoverageScanCap
	}
	git := deps.git
	if git == nil {
		git = runFanoutCoverageGit
	}

	// Witness 1: which internal leaves were ADDED inside the window.
	addedArgs := []string{"log", "--since=" + since, "--diff-filter=A", "--name-only", "--pretty=format:"}
	addedOut, addedErr, ok := git(addedArgs)
	if !ok {
		return issuefanout.CoverageReport{}, fmt.Errorf("git %s: %s", strings.Join(addedArgs, " "), strings.TrimSpace(addedErr))
	}

	// Witness 1b: which leaves already existed BEFORE the window. Without this
	// second read a leaf that merely gained a file inside the window counts as
	// new, inflating the spine denominator with leaves that predate the default.
	// The two reads use the same selector, so --since/--until partition history.
	beforeArgs := []string{"log", "--until=" + since, "--diff-filter=A", "--name-only", "--pretty=format:"}
	beforeOut, beforeErr, ok := git(beforeArgs)
	if !ok {
		return issuefanout.CoverageReport{}, fmt.Errorf("git %s: %s", strings.Join(beforeArgs, " "), strings.TrimSpace(beforeErr))
	}
	newLeaves := issuefanout.NewLeavesInWindow(
		strings.Split(addedOut, "\n"),
		strings.Split(beforeOut, "\n"),
	)

	// Witness 2: the tracked file list, which decides each leaf's spine artifacts.
	trackedOut, trackedErr, ok := git([]string{"ls-files"})
	if !ok {
		return issuefanout.CoverageReport{}, fmt.Errorf("git ls-files: %s", strings.TrimSpace(trackedErr))
	}
	witnesses := issuefanout.WitnessLeaves(newLeaves, strings.Split(trackedOut, "\n"))

	// Witness 3: the filed fan-out markers, read out of the tracker export. The
	// scan runs under the coverage cap, not --live's dedupe cap, and its size is
	// reported so a truncated export cannot pass as a measured rate.
	ghRun := deps.gh
	if ghRun == nil {
		ghRun = runFanoutCoverageGH
	}
	existing, err := fetchFanoutExisting(repo, scanCap, ghRun)
	if err != nil {
		return issuefanout.CoverageReport{}, err
	}

	return issuefanout.Coverage(
		witnesses,
		issuefanout.ExtractMarkerKeys(existing),
		issuefanout.CoverageScan{Issues: len(existing), Cap: scanCap},
	), nil
}

// emitFanoutCoverage gathers and renders the scorecard. It exits non-zero when
// either rate is short, so a pipeline can gate on the defaults staying real.
func emitFanoutCoverage(stdout, stderr io.Writer, since, repo string, scanCap int, asJSON bool, deps fanoutCoverageDeps) int {
	rep, err := gatherFanoutCoverage(since, repo, scanCap, deps)
	if err != nil {
		fmt.Fprintf(stderr, "fak-dev issue fanout --coverage: %v\n", err)
		return 2
	}
	if asJSON {
		if err := writeIndentedJSON(stdout, rep); err != nil {
			fmt.Fprintf(stderr, "fak-dev issue fanout: encode json: %v\n", err)
			return 1
		}
	} else {
		fmt.Fprint(stdout, issuefanout.RenderCoverage(rep))
	}
	if !rep.OK {
		return 1 // a rate is short — the honesty meter fails the gate
	}
	return 0
}
