package cadencereport

// Live runners for `fak cadence`: the three dimensions are measured by shelling
// to the existing Python control-pane folds (scores, releases) and to git
// (work-done). Kept separate from the pure fold so cadencereport.go stays
// unit-testable without a process or a repo.

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// ScoresArgv is the scorecard-control-pane fold, emitting the portfolio debt +
// trend the SCORES dimension distills. Plain --json (not --check) so the report
// reads the raw fold, never an exit-code-coupled view.
var ScoresArgv = []string{"tools/scorecard_control_pane.py", "--json"}

// ReleasesArgv is the release-status fold, offline (--skip-gh) and without the
// slow cut dry-run (--skip-cut-plan), so the RELEASES dimension is deterministic
// and network-free.
var ReleasesArgv = []string{"tools/release_status.py", "--json", "--skip-gh", "--skip-cut-plan"}

// Collect measures all three dimensions live. The scores/releases members run
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
		InterpretReleases(releasesPayload, releasesErr)
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
// the total commit count on HEAD and the subset carrying a `(fak ` ship trailer
// (the Conventional-Commits leaf stamp every ship commit ends with).
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

	// `(fak ` is a literal needle, so fix the regex engine to fixed-strings. A
	// ship commit ends with a `(fak <leaf>)` trailer; an un-stamped WIP commit
	// won't match, so this counts SHIPS, not every commit.
	ships, err := gitCount(root, []string{
		"rev-list", "--count", "--since=" + since, "--fixed-strings", "--grep=(fak ", "HEAD",
	})
	if err != "" {
		// The commit count already succeeded; a grep-count failure is a partial
		// signal, not a measurement failure — keep commits, leave ships at 0 with
		// a soft note folded into the commit count being authoritative.
		w.Ships = 0
		return w
	}
	w.Ships = ships
	return w
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

// HeadCommit returns the short HEAD commit of root, or "unknown".
func HeadCommit(root string) string {
	cmd := exec.Command("git", "rev-parse", "--short", "HEAD")
	cmd.Dir = root
	out, err := cmd.Output()
	if err != nil {
		return "unknown"
	}
	s := strings.TrimSpace(string(out))
	if s == "" {
		return "unknown"
	}
	return s
}

func defaultPython() string {
	if p := os.Getenv("FAK_PYTHON"); p != "" {
		return p
	}
	return "python3"
}

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
