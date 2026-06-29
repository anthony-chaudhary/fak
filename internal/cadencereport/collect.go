package cadencereport

// Live runners for `fak cadence`: scores/releases are measured by shelling to
// the existing Python control-pane folds, maturity is measured in-process, and
// work-done is read from git. Kept separate from the pure fold so
// cadencereport.go stays unit-testable without a process or a repo.

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gardenbundle"
	"github.com/anthony-chaudhary/fak/internal/hooks"
	maturityscore "github.com/anthony-chaudhary/fak/internal/maturity"
	"github.com/anthony-chaudhary/fak/internal/releasestale"
)

// ScoresArgv is the scorecard-control-pane fold, emitting the portfolio debt +
// trend the SCORES dimension distills. Plain --json (not --check) so the report
// reads the raw fold, never an exit-code-coupled view.
var ScoresArgv = []string{"tools/scorecard_control_pane.py", "--json"}

// ReleasesArgv is the release-status fold, offline (--skip-gh) and without the
// slow cut dry-run (--skip-cut-plan), so the RELEASES dimension is deterministic
// and network-free.
var ReleasesArgv = []string{"tools/release_status.py", "--json", "--skip-gh", "--skip-cut-plan"}

// Collect measures the original live dimensions. The scores/releases members run
// the Python folds; work is derived from git over the trailing window. A member
// that cannot run yields an errored dimension (never a silent zero).
func Collect(root, python string, timeout time.Duration, windowDays int) (Scores, Work, Releases) {
	if python == "" {
		python = defaultPython()
	}
	scoresPayload, scoresErr := RunPyEnvelope(root, ScoresArgv, python, timeout)
	releasesPayload, releasesErr := RunPyEnvelope(root, ReleasesArgv, python, timeout)
	return InterpretScores(scoresPayload, scoresErr),
		WorkFromGit(root, windowDays),
		withPublishStaleness(root, InterpretReleases(releasesPayload, releasesErr))
}

// CollectMaturity measures the feature-lifecycle scorecard in-process. Unlike
// SCORES/RELEASES it is already Go-native, so cadence can read it directly
// without shelling through the control-pane runner.
func CollectMaturity(root string) Maturity {
	return MaturityFromScorecard(maturityscore.Build(maturityscore.Options{Root: root}))
}

// withPublishStaleness layers the Go-native @latest-vs-HEAD lag onto a Releases
// dimension via the releasestale signal. It is the impure half (git is read here, off
// the hot path); the projection itself is the pure WithPublishStaleness. A no-tag /
// unreadable repo yields an Unknown verdict with zero lag, never a false "stale".
func withPublishStaleness(root string, r Releases) Releases {
	// versionFile is only used by releasestale to detect an untagged cut (not surfaced
	// in the cadence line), so passing "" here is fine — the lag itself is git-derived.
	f := releasestale.Gather(context.Background(), releasestale.RealRunner, root, "")
	p := releasestale.Compute(f, releasestale.DefaultThresholds(), root)
	return WithPublishStaleness(r, p.CommitsBehind, p.DaysBehind, p.Verdict)
}

// InterpretScoresFromFile reads a scorecard-control-pane JSON payload from path
// (or os.Stdin when path is "-") and folds it into the SCORES dimension via the
// SAME InterpretScores the live run uses — so a payload captured once (e.g. by
// the garden bundle) can drive `fak cadence --scores-from` instead of re-running
// the ~4-minute pane. A missing or garbled file degrades to an ERRORED SCORES
// dimension (Err set, OK=false), identical in shape to a failed live run, never a
// silent zero. The reader is injectable for testing the stdin path.
func InterpretScoresFromFile(path string, stdin io.Reader) Scores {
	var (
		data []byte
		err  error
	)
	if path == "-" {
		data, err = io.ReadAll(stdin)
	} else {
		data, err = os.ReadFile(path)
	}
	if err != nil {
		return InterpretScores(nil, "--scores-from "+path+": "+err.Error())
	}
	var payload map[string]any
	if jerr := json.Unmarshal(data, &payload); jerr != nil || payload == nil {
		return InterpretScores(nil, "--scores-from "+path+": not a scorecard control-pane JSON payload")
	}
	return InterpretScores(payload, "")
}

// CollectWithScores runs the WORK-DONE (git) and RELEASES (release-status)
// dimensions but takes the SCORES dimension as a pre-interpreted value, so the
// scorecard pane is NOT shelled when --scores-from supplied it. The default path
// (Collect) is unchanged, so the standalone command and the weekly cadence run
// are unaffected.
func CollectWithScores(root, python string, scores Scores, timeout time.Duration, windowDays int) (Scores, Work, Releases) {
	if python == "" {
		python = defaultPython()
	}
	releasesPayload, releasesErr := RunPyEnvelope(root, ReleasesArgv, python, timeout)
	return scores, WorkFromGit(root, windowDays), withPublishStaleness(root, InterpretReleases(releasesPayload, releasesErr))
}

// RunPyEnvelope runs a Python control-pane member and parses its JSON stdout. It
// returns the parsed payload (nil on any failure) and an error string (empty on
// success). Mirrors internal/gardenbundle.RunMember.
func RunPyEnvelope(root string, argv []string, python string, timeout time.Duration) (map[string]any, string) {
	if len(argv) == 0 {
		return nil, "empty argv"
	}
	script := filepath.Join(root, argv[0])
	if _, err := os.Stat(script); err != nil {
		return nil, "missing member script: " + argv[0]
	}
	args := append([]string{script}, argv[1:]...)
	cmd := exec.Command(python, args...)
	cmd.Dir = root
	var stdout, stderr strings.Builder
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	if err := cmd.Start(); err != nil {
		return nil, err.Error()
	}
	done := make(chan error, 1)
	go func() { done <- cmd.Wait() }()
	select {
	case <-time.After(timeout):
		_ = cmd.Process.Kill()
		<-done
		return nil, fmt.Sprintf("timed out after %ds", int(timeout.Seconds()))
	case <-done:
		var payload map[string]any
		if err := json.Unmarshal([]byte(stdout.String()), &payload); err == nil && payload != nil {
			return payload, ""
		}
		tail := lastLine(stderr.String())
		if tail == "" {
			tail = lastLine(stdout.String())
		}
		if len(tail) > 160 {
			tail = tail[:160]
		}
		return nil, "non-JSON output: " + tail
	}
}

// WorkFromGit derives the WORK-DONE dimension from git over the trailing window:
// the total commit count on HEAD, and the subset whose SUBJECT carries a real
// per-leaf ship-stamp. The ship count is NOT a bare `--grep=(fak ` count — that
// over-counts in three ways the real grammar doesn't: it matches a merge subject,
// a body-line / co-author-trailer mention anywhere in the message (not the
// anchored subject stamp), and it can't bucket by leaf. Instead we enumerate one
// `(sha, subject)` per non-merge commit and decide ship-ness per subject through
// hooks.StampOf — the SAME grammar the pre-commit lint binds to — so this counts
// exactly what `dos verify` can bind, and buckets each ship by its leaf.
func WorkFromGit(root string, windowDays int) Work {
	w := Work{WindowDays: windowDays}
	since := fmt.Sprintf("%d days ago", windowDays)

	commits, err := gitCount(root, []string{
		"rev-list", "--count", "--since=" + since, "HEAD",
	})
	if err != "" {
		w.Err = err
		return w
	}
	w.Commits = commits

	// One subject line per non-merge commit (%s is git's single-line subject, so a
	// multi-line body can't leak a body-only `(fak x)` into the count). git log over
	// HEAD is already deduped, so each reachable commit is counted at most once.
	subjects, gerr := gitShipSubjects(root, since)
	if gerr != "" {
		// The commit count already succeeded; a subject-enumeration failure is a
		// partial signal, not a measurement failure — keep commits authoritative,
		// leave ships at 0, do NOT set w.Err (matching the prior soft-degrade).
		w.Ships = 0
		return w
	}
	w.Ships, w.ByLane = shipsBySubjects(subjects)
	return w
}

// gitShipSubjects returns one subject line per non-merge commit in the trailing
// window on HEAD. Errors degrade to ("", errString) for the soft-fail path above.
func gitShipSubjects(root, since string) ([]string, string) {
	cmd := exec.Command("git", "log", "--no-merges", "--since="+since, "--format=%s", "HEAD")
	cmd.Dir = root
	out, err := cmd.Output()
	if err != nil {
		return nil, "git log failed: " + gitErr(err)
	}
	var subjects []string
	for _, line := range strings.Split(string(out), "\n") {
		s := strings.TrimSpace(line)
		if s != "" {
			subjects = append(subjects, s)
		}
	}
	return subjects, ""
}

// shipsBySubjects is the pure ship predicate over already-extracted subjects: a
// subject is a ship iff hooks.StampOf grades it trailer|direct (a real per-leaf
// stamp). A merge/bookkeeping/body-only subject (kind "none") and a release-bundle
// subject (kind "release", not a per-leaf ship) are excluded. An off-lane typo
// like `(fak gatway)` still counts — the count is grammar-based, not taxonomy-
// validated (that lane check is the pre-commit lint's job, not the ledger's).
// Kept pure (no git) so it is unit-testable without a repo.
func shipsBySubjects(subjects []string) (ships int, byLane map[string]int) {
	byLane = map[string]int{}
	for _, subj := range subjects {
		kind, leaf := hooks.StampOf(subj)
		if kind == "trailer" || kind == "direct" {
			ships++
			byLane[leaf]++
		}
	}
	if len(byLane) == 0 {
		byLane = nil
	}
	return ships, byLane
}

// gitCount runs a git subcommand expected to print a single integer and parses
// it. Returns (count, errString); errString is empty on success.
func gitCount(root string, args []string) (int, string) {
	cmd := exec.Command("git", args...)
	cmd.Dir = root
	out, err := cmd.Output()
	if err != nil {
		return 0, "git " + args[0] + " failed: " + gitErr(err)
	}
	n, perr := strconv.Atoi(strings.TrimSpace(string(out)))
	if perr != nil {
		return 0, "git " + args[0] + " emitted non-integer: " + strings.TrimSpace(string(out))
	}
	return n, ""
}

func gitErr(err error) string {
	if ee, ok := err.(*exec.ExitError); ok {
		s := strings.TrimSpace(string(ee.Stderr))
		if s != "" {
			return lastLine(s)
		}
	}
	return err.Error()
}

// HeadCommit returns the short HEAD commit of root, or "unknown". It shares the
// one implementation in gardenbundle rather than copying the git plumbing.
func HeadCommit(root string) string { return gardenbundle.HeadCommit(root) }

func defaultPython() string { return gardenbundle.DefaultPython() }

func lastLine(s string) string {
	s = strings.TrimRight(s, "\n")
	if s == "" {
		return ""
	}
	if i := strings.LastIndexByte(s, '\n'); i >= 0 {
		return strings.TrimSpace(s[i+1:])
	}
	return strings.TrimSpace(s)
}
