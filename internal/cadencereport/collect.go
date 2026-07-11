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
	"github.com/anthony-chaudhary/fak/internal/procguard"
	"github.com/anthony-chaudhary/fak/internal/releasestale"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
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
	f := releasestale.Gather(context.Background(), releasestale.RealRunner, releasestale.RealClock, root, "")
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
		// The folded scripts spawn `git` children; a single-PID kill orphans them (#3103).
		_, _ = procguard.KillPID(cmd.Process.Pid)
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

const (
	shipHoldLocalOnly            = "local_only"
	shipHoldPublishedUnwitnessed = "published_unwitnessed"
)

type shipCommit struct {
	SHA     string
	Subject string
	Leaf    string
}

type shipAuditResult struct {
	Witnessed bool
	Detail    string
}

type shipAuditFunc func(root, publishedRef string, commits []shipCommit) (map[string]shipAuditResult, string)

// WorkFromGit derives WORK-DONE from authoritative effects, not the local subject
// alone. Commits remains the trailing HEAD activity count for compatibility. Ships is
// narrower: the commit must be reachable from origin's trunk, carry the real stamp
// grammar, AND receive a diff-witnessed OK from `dos commit-audit`. Ship-stamped local
// commits and published-but-unwitnessed commits are retained as typed holds.
func WorkFromGit(root string, windowDays int) Work {
	return workFromGit(root, windowDays, auditPublishedShips)
}

func workFromGit(root string, windowDays int, audit shipAuditFunc) Work {
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

	head, gerr := gitShipCommits(root, since, "HEAD")
	if gerr != "" {
		w.Err = gerr
		return w
	}
	publishedRef, rerr := originTrunkRef(root)
	if rerr != "" {
		w.Err = rerr
		return w
	}
	published, perr := gitShipCommits(root, since, publishedRef)
	if perr != "" {
		w.Err = perr
		return w
	}

	publishedSet := make(map[string]bool, len(published))
	var publishedCandidates []shipCommit
	for _, commit := range published {
		publishedSet[commit.SHA] = true
		if commit.Leaf != "" {
			publishedCandidates = append(publishedCandidates, commit)
		}
	}
	for _, commit := range head {
		if commit.Leaf == "" || publishedSet[commit.SHA] {
			continue
		}
		w.Held = append(w.Held, ShipHold{
			SHA: shortCommit(commit.SHA), Leaf: commit.Leaf, Reason: shipHoldLocalOnly,
			Detail: "not reachable from " + publishedRef, Subject: commit.Subject,
		})
	}

	audits, aerr := audit(root, publishedRef, publishedCandidates)
	if aerr != "" {
		w.Err = aerr
	}
	w.ByLane = map[string]int{}
	for _, commit := range publishedCandidates {
		result, ok := audits[commit.SHA]
		if ok && result.Witnessed {
			w.Ships++
			w.ByLane[commit.Leaf]++
			continue
		}
		detail := "commit audit returned no diff-witnessed OK"
		if ok && result.Detail != "" {
			detail = result.Detail
		} else if aerr != "" {
			detail = aerr
		}
		w.Held = append(w.Held, ShipHold{
			SHA: shortCommit(commit.SHA), Leaf: commit.Leaf, Reason: shipHoldPublishedUnwitnessed,
			Detail: detail, Subject: commit.Subject,
		})
	}
	if len(w.ByLane) == 0 {
		w.ByLane = nil
	}
	return w
}

// gitShipSubjects is retained as the pure-subject compatibility helper for its grammar
// tests. Delivery credit no longer uses it; WorkFromGit needs SHA + publication + audit.
func gitShipSubjects(root, since string) ([]string, string) {
	commits, err := gitShipCommits(root, since, "HEAD")
	if err != "" {
		return nil, err
	}
	subjects := make([]string, len(commits))
	for i, commit := range commits {
		subjects[i] = commit.Subject
	}
	return subjects, ""
}

func gitShipCommits(root, since, ref string) ([]shipCommit, string) {
	cmd := exec.Command("git", "log", "--no-merges", "--since="+since, "--format=%H%x09%s", ref)
	windowgate.ConfigureBackgroundCommand(cmd)
	cmd.Dir = root
	out, err := cmd.Output()
	if err != nil {
		return nil, "git log " + ref + " failed: " + gitErr(err)
	}
	var commits []shipCommit
	for _, line := range strings.Split(string(out), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		sha, subject, ok := strings.Cut(line, "\t")
		if !ok || sha == "" || subject == "" {
			return nil, "git log " + ref + " emitted malformed commit row"
		}
		kind, leaf := hooks.StampOf(subject)
		if kind != "trailer" && kind != "direct" {
			leaf = ""
		}
		commits = append(commits, shipCommit{SHA: sha, Subject: subject, Leaf: leaf})
	}
	return commits, ""
}

func originTrunkRef(root string) (string, string) {
	cmd := exec.Command("git", "symbolic-ref", "--quiet", "--short", "refs/remotes/origin/HEAD")
	windowgate.ConfigureBackgroundCommand(cmd)
	cmd.Dir = root
	if out, err := cmd.Output(); err == nil {
		if ref := strings.TrimSpace(string(out)); ref != "" {
			return ref, ""
		}
	}
	// A local checkout created by `git push -u` may not have origin/HEAD, but its
	// branch upstream is still authoritative. This also covers master-named trunks.
	cmd = exec.Command("git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
	windowgate.ConfigureBackgroundCommand(cmd)
	cmd.Dir = root
	if out, err := cmd.Output(); err == nil {
		if ref := strings.TrimSpace(string(out)); strings.HasPrefix(ref, "origin/") {
			return ref, ""
		}
	}
	var lastErr error
	for _, ref := range []string{"origin/main", "origin/master"} {
		cmd = exec.Command("git", "rev-parse", "--verify", ref)
		windowgate.ConfigureBackgroundCommand(cmd)
		cmd.Dir = root
		if _, err := cmd.Output(); err == nil {
			return ref, ""
		} else {
			lastErr = err
		}
	}
	return "", "git origin trunk unavailable: " + gitErr(lastErr)
}

type dosAuditRow struct {
	SHA     string `json:"sha"`
	Verdict string `json:"verdict"`
	Witness string `json:"witness"`
	Reason  string `json:"reason"`
}

func auditPublishedShips(root, publishedRef string, commits []shipCommit) (map[string]shipAuditResult, string) {
	results := map[string]shipAuditResult{}
	if len(commits) == 0 {
		return results, ""
	}
	oldest := commits[len(commits)-1].SHA
	parentCmd := exec.Command("git", "rev-parse", "--verify", oldest+"^")
	windowgate.ConfigureBackgroundCommand(parentCmd)
	parentCmd.Dir = root
	parentOut, err := parentCmd.Output()
	if err != nil {
		return auditPublishedIndividually(root, commits)
	}
	rangeRef := strings.TrimSpace(string(parentOut)) + ".." + publishedRef
	rows, auditErr := runDOSCommitAudit(root, rangeRef)
	if auditErr != "" {
		return nil, auditErr
	}
	return foldDOSAuditRows(commits, rows), ""
}

func auditPublishedIndividually(root string, commits []shipCommit) (map[string]shipAuditResult, string) {
	results := map[string]shipAuditResult{}
	for _, commit := range commits {
		rows, err := runDOSCommitAudit(root, commit.SHA)
		if err != "" {
			return nil, err
		}
		for sha, result := range foldDOSAuditRows([]shipCommit{commit}, rows) {
			results[sha] = result
		}
	}
	return results, ""
}

func runDOSCommitAudit(root, ref string) ([]dosAuditRow, string) {
	cmd := exec.Command("dos", "commit-audit", "--json", ref)
	windowgate.ConfigureBackgroundCommand(cmd)
	cmd.Dir = root
	out, err := cmd.Output()
	if err != nil {
		if ee, ok := err.(*exec.ExitError); !ok || ee.ExitCode() != 1 {
			return nil, "dos commit-audit failed: " + gitErr(err)
		}
	}
	var rows []dosAuditRow
	if err := json.Unmarshal(out, &rows); err != nil {
		return nil, "dos commit-audit emitted invalid JSON: " + err.Error()
	}
	return rows, ""
}

func foldDOSAuditRows(commits []shipCommit, rows []dosAuditRow) map[string]shipAuditResult {
	results := map[string]shipAuditResult{}
	for _, commit := range commits {
		for _, row := range rows {
			if !strings.HasPrefix(commit.SHA, row.SHA) && !strings.HasPrefix(row.SHA, commit.SHA) {
				continue
			}
			detail := strings.TrimSpace(row.Verdict + " " + row.Witness + ": " + row.Reason)
			results[commit.SHA] = shipAuditResult{
				Witnessed: row.Verdict == "OK" && row.Witness == "diff-witnessed",
				Detail:    detail,
			}
			break
		}
	}
	return results
}

func shortCommit(sha string) string {
	if len(sha) > 12 {
		return sha[:12]
	}
	return sha
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
	windowgate.ConfigureBackgroundCommand(cmd)
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
