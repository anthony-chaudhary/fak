// Package commitlane reports the read-only state of the shared fak commit lane.
//
// It observes the two local files that normally block a path-scoped commit:
// <gitdir>/fak-commit.lock, owned by fak's safecommit path, and <gitdir>/index.lock,
// owned by git itself — plus the <gitdir>/next-index-<pid>.lock temp files git leaves
// behind when an index writer dies mid-write (#5338). It never removes any of them.
// Process inventory is best-effort and only used to make a lock owner or likely queue
// visible.
package commitlane

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/safecommit"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

const Schema = "fak-commit-lane-status/1"

const (
	VerdictClear   = "clear"
	VerdictBusy    = "busy"
	VerdictStale   = "stale"
	VerdictBlocked = "blocked"
	VerdictUnknown = "unknown"
)

const DefaultStaleIndexAge = 15 * time.Minute

// DefaultOwnerDeadIndexAge is the SHORT frozen window used when the orphan's creator is
// NAMED and provably dead (a sibling .git/fak-commit.lock holding a dead pid). The 15-minute
// DefaultStaleIndexAge exists because a bare .git/index.lock names nobody, so only a long
// freeze can stand in for "the holder is gone"; once the holder is proven dead by pid, that
// long wait buys nothing and is exactly the window in which the wedge reproduces itself
// (#5335: a killed `fak commit` orphans its own index.lock, and the lane stays blocked for
// the remaining grace period while a peer swarm keeps re-killing committers). One frozen
// minute plus a dead named creator is already stronger evidence than fifteen frozen minutes
// with no creator at all.
const DefaultOwnerDeadIndexAge = 60 * time.Second

// DefaultIndexLockSettleWindow bounds the pause between the two index.lock samples that
// witness whether the lock's mtime is ADVANCING. It is spent ONLY when a lock is actually
// present — the clear lane pays nothing — and it is what keeps the age gates honest: an
// advancing mtime means a live writer is holding this lock right now, however old its
// mtime looks, so no age-based reap may fire against it.
const DefaultIndexLockSettleWindow = 300 * time.Millisecond

type Runner func(ctx context.Context, dir string, args ...string) RunResult

type RunResult struct {
	Stdout string
	Stderr string
	Code   int
	Err    error
}

type ProbeLockFunc func(path string) safecommit.LockProbe

type FileStatFunc func(path string) FileFact

type ProcessListFunc func(ctx context.Context) ([]Process, error)

// GlobFunc lists the paths matching a shell pattern. It is the read-only seam behind
// the next-index-*.lock residue scan (default filepath.Glob) so tests can enumerate a
// fixture set without a real .git directory.
type GlobFunc func(pattern string) ([]string, error)

// PIDAliveFunc reports whether a pid is currently running. It is the seam behind the
// dead-owner half of the next-index residue evidence (default safecommit.ProcessAlive).
type PIDAliveFunc func(pid int) bool

type Options struct {
	Dir           string
	Runner        Runner
	ProbeLock     ProbeLockFunc
	Stat          FileStatFunc
	ProcessList   ProcessListFunc
	Glob          GlobFunc
	PIDAlive      PIDAliveFunc
	Now           func() time.Time
	StaleIndexAge time.Duration
	// Sleep is the seam behind the bounded pause between the two index.lock samples that
	// witness an ADVANCING mtime (default time.Sleep). Tests inject a no-op so they take the
	// real settle path with zero wall-clock wait — a settle witness proven by sleeping would
	// be untestable without making the suite hang-prone, which is the very failure this
	// package exists to bound.
	Sleep func(time.Duration)
	// SettleWindow bounds that pause (default DefaultIndexLockSettleWindow). It is spent only
	// when .git/index.lock is present.
	SettleWindow time.Duration
	// OwnerDeadIndexAge is the short frozen window that applies when the lock's creator is
	// named and proven dead (default DefaultOwnerDeadIndexAge). See FrozenHint.
	OwnerDeadIndexAge time.Duration
}

type Report struct {
	Schema     string     `json:"schema"`
	OK         bool       `json:"ok"`
	Verdict    string     `json:"verdict"`
	Reason     string     `json:"reason,omitempty"`
	NextAction string     `json:"next_action,omitempty"`
	RepoRoot   string     `json:"repo_root,omitempty"`
	GitDir     string     `json:"git_dir,omitempty"`
	CommitLock CommitLock `json:"commit_lock"`
	IndexLock  IndexLock  `json:"index_lock"`
	// NextIndexLocks is the observed .git/next-index-<pid>.lock residue: the temp files
	// git writes a new index into before renaming it over .git/index. A writer that dies
	// mid-write leaves one behind forever — git never reaps them, so they accumulate
	// alongside the recurring index.lock wedge (#5338). Observation only; nothing here
	// removes a file.
	NextIndexLocks []NextIndexLock `json:"next_index_locks,omitempty"`
	// IndexChurn is the no-op staged-deletion audit (#5339): paths the shared index has
	// staged for DELETION whose working-tree file is still present and byte-identical to
	// HEAD, i.e. `git rm --cached` residue with no commit effect. It is a hygiene finding,
	// not a lane state — it never feeds finalize's verdict, and nothing here (or in the CLI
	// renderer) executes the remedy it carries. Nil when the index is clean or the probe
	// could not run. See stageddelete.go.
	IndexChurn   *StagedDeletionAudit `json:"index_churn,omitempty"`
	Owner        *ProcessFact         `json:"owner,omitempty"`
	Queue        []ProcessFact        `json:"queue,omitempty"`
	LiveWriters  []ProcessFact        `json:"live_writers,omitempty"`
	ProcessProbe string               `json:"process_probe"`
	Errors       []string             `json:"errors,omitempty"`
}

type CommitLock struct {
	Path        string `json:"path"`
	Present     bool   `json:"present"`
	HolderPID   int    `json:"holder_pid,omitempty"`
	HolderAlive bool   `json:"holder_alive,omitempty"`
	Stale       bool   `json:"stale"`
}

type IndexLock struct {
	Path       string `json:"path"`
	Present    bool   `json:"present"`
	ModTime    string `json:"mod_time,omitempty"`
	AgeSeconds int64  `json:"age_seconds,omitempty"`
	StaleHint  bool   `json:"stale_hint,omitempty"`
	// FrozenHint is the SHORT freeze witness: the lock's mtime has not moved for at least
	// OwnerDeadIndexAge and did not move across the settle window either. On its own it
	// proves nothing (a one-minute freeze is not fifteen); it is the age half of the
	// owner-dead evidence, and only combines into a reap when a sibling fak-commit.lock
	// names a DEAD creator for this lock (#5335 item 3).
	FrozenHint bool `json:"frozen_hint,omitempty"`
	// Advancing is the live-holder witness: a second sample taken a bounded settle window
	// after the first saw the mtime move forward or the size change. An advancing lock is
	// being written RIGHT NOW, so it is never an orphan no matter how old its mtime reads —
	// this is the distinction every age-based reap here must preserve.
	Advancing bool `json:"advancing,omitempty"`
	// SettleMillis records the settle window actually spent, so a reclaim refusal (or a
	// reap) can be audited against the window that produced its Advancing verdict.
	SettleMillis int64  `json:"settle_millis,omitempty"`
	Detail       string `json:"detail,omitempty"`
}

// NextIndexLock is one observed .git/next-index-<pid>.lock temp file. Unlike index.lock
// its name carries the writing process's pid, so "dead owner" is provable DIRECTLY
// (OwnerAlive) rather than inferred from the absence of a matching live writer.
type NextIndexLock struct {
	Path       string `json:"path"`
	PID        int    `json:"pid,omitempty"`
	OwnerAlive bool   `json:"owner_alive,omitempty"`
	ModTime    string `json:"mod_time,omitempty"`
	AgeSeconds int64  `json:"age_seconds,omitempty"`
	StaleHint  bool   `json:"stale_hint,omitempty"`
}

type FileFact struct {
	Exists  bool
	ModTime time.Time
	Size    int64
	Err     string
}

type Process struct {
	PID       int    `json:"pid"`
	ParentPID int    `json:"parent_pid,omitempty"`
	Name      string `json:"name,omitempty"`
	Command   string `json:"command,omitempty"`
}

type ProcessFact struct {
	PID         int    `json:"pid"`
	ParentPID   int    `json:"parent_pid,omitempty"`
	Name        string `json:"name,omitempty"`
	Command     string `json:"command,omitempty"`
	Role        string `json:"role"`
	Match       string `json:"match,omitempty"`
	RepoMatched bool   `json:"repo_matched,omitempty"`
	Confidence  string `json:"confidence,omitempty"`
}

func Status(ctx context.Context, opts Options) (Report, error) {
	opts = normalizeOptions(opts)
	now := opts.Now()
	dir := strings.TrimSpace(opts.Dir)
	if dir == "" {
		dir = "."
	}
	rep := Report{
		Schema:       Schema,
		ProcessProbe: "not_run",
		OK:           false,
		Verdict:      VerdictUnknown,
	}

	root, ok, err := gitRead(ctx, opts.Runner, dir, "rev-parse", "--show-toplevel")
	if err != nil {
		return rep, err
	}
	if !ok || strings.TrimSpace(root) == "" {
		rep.Reason = "not inside a git work tree"
		rep.NextAction = "run from the fak checkout or pass --dir"
		return rep, nil
	}
	rep.RepoRoot = cleanPath(root)

	gitDir, ok, err := gitRead(ctx, opts.Runner, dir, "rev-parse", "--absolute-git-dir")
	if err != nil {
		return rep, err
	}
	if !ok || strings.TrimSpace(gitDir) == "" {
		rep.Reason = "could not resolve the repository git directory"
		rep.NextAction = "verify git is readable from the workspace root, then re-run"
		return rep, nil
	}
	rep.GitDir = cleanPath(gitDir)

	rep.CommitLock = probeCommitLock(filepath.Join(rep.GitDir, "fak-commit.lock"), opts.ProbeLock)
	rep.IndexLock = probeIndexLock(filepath.Join(rep.GitDir, "index.lock"), opts, now)
	nextIndex, nerr := probeNextIndexLocks(rep.GitDir, opts.Glob, opts.Stat, opts.PIDAlive, now, opts.StaleIndexAge)
	rep.NextIndexLocks = nextIndex
	if nerr != "" {
		rep.Errors = append(rep.Errors, "next-index scan: "+nerr)
	}
	// The no-op staged-deletion audit (#5339). It costs exactly ONE extra git read on a
	// clean index (the name-only diff comes back empty and the batched hash reads never
	// run), and it fails open and silent, so a probe that cannot run adds no warning and
	// no verdict change — the lane report must not grow noise because a hygiene check
	// misfired.
	if audit := ScanStagedDeletions(ctx, opts.Runner, opts.Stat, rep.RepoRoot); len(audit.Rows) > 0 {
		rep.IndexChurn = &audit
	}

	procs, perr := opts.ProcessList(ctx)
	if perr != nil {
		rep.ProcessProbe = "error"
		rep.Errors = append(rep.Errors, "process inventory: "+perr.Error())
	} else {
		rep.ProcessProbe = "ok"
		rep.Owner, rep.LiveWriters, rep.Queue = classifyProcesses(rep.RepoRoot, rep.CommitLock.HolderPID, procs)
	}

	finalize(&rep)
	return rep, nil
}

func normalizeOptions(opts Options) Options {
	if opts.Runner == nil {
		opts.Runner = RealGitRunner
	}
	if opts.ProbeLock == nil {
		opts.ProbeLock = safecommit.ProbeLock
	}
	if opts.Stat == nil {
		opts.Stat = StatFile
	}
	if opts.ProcessList == nil {
		opts.ProcessList = DefaultProcessList
	}
	if opts.Glob == nil {
		opts.Glob = filepath.Glob
	}
	if opts.PIDAlive == nil {
		opts.PIDAlive = safecommit.ProcessAlive
	}
	if opts.Now == nil {
		opts.Now = time.Now
	}
	if opts.StaleIndexAge == 0 {
		opts.StaleIndexAge = DefaultStaleIndexAge
	}
	if opts.Sleep == nil {
		opts.Sleep = time.Sleep
	}
	if opts.SettleWindow == 0 {
		opts.SettleWindow = DefaultIndexLockSettleWindow
	}
	if opts.OwnerDeadIndexAge == 0 {
		opts.OwnerDeadIndexAge = DefaultOwnerDeadIndexAge
	}
	return opts
}

func gitRead(ctx context.Context, run Runner, dir string, args ...string) (string, bool, error) {
	full := append([]string{"--no-optional-locks"}, args...)
	res := run(ctx, dir, full...)
	if res.Err != nil {
		return "", false, res.Err
	}
	if res.Code != 0 {
		return "", false, nil
	}
	return strings.TrimSpace(res.Stdout), true, nil
}

func RealGitRunner(ctx context.Context, dir string, args ...string) RunResult {
	cmd := exec.CommandContext(ctx, "git", args...)
	windowgate.ConfigureBackgroundCommand(cmd)
	if strings.TrimSpace(dir) != "" {
		cmd.Dir = dir
	}
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	err := cmd.Run()
	code := 0
	if err != nil {
		var exit *exec.ExitError
		if errors.As(err, &exit) {
			code = exit.ExitCode()
			err = nil
		} else {
			code = -1
		}
	}
	return RunResult{Stdout: stdout.String(), Stderr: stderr.String(), Code: code, Err: err}
}

func StatFile(path string) FileFact {
	info, err := os.Stat(path)
	if err == nil {
		return FileFact{Exists: true, ModTime: info.ModTime(), Size: info.Size()}
	}
	if errors.Is(err, os.ErrNotExist) {
		return FileFact{}
	}
	return FileFact{Err: err.Error()}
}

func probeCommitLock(path string, probe ProbeLockFunc) CommitLock {
	p := probe(path)
	return CommitLock{
		Path:        path,
		Present:     p.Exists,
		HolderPID:   p.HolderPID,
		HolderAlive: p.Alive,
		Stale:       p.Stale,
	}
}

// probeIndexLock observes .git/index.lock TWICE, a bounded settle window apart, and derives
// the three facts the reclaim decision needs: presence, age, and whether the lock is
// ADVANCING.
//
// The second sample is the point of this function. A single stat can only ever answer "how
// old does this mtime look", and age alone conflates the two states the commit lane must
// keep apart: an orphan left by a killed writer (mtime frozen forever) and a live writer
// that is simply slow (mtime still moving). Sampling twice separates them directly — if the
// mtime moved, or the file grew, some process is writing this lock right now and no reap may
// fire against it, however stale the first sample read. Everything downstream is then free to
// lower its age bar, because the "is anyone actually holding it" question has already been
// answered from the lock itself rather than inferred from an unrelated process inventory.
//
// The window is spent only when a lock is present, so the clear lane costs nothing, and it is
// spent through the injected Sleep so tests exercise this path without any wall-clock wait.
// It never removes a file.
func probeIndexLock(path string, opts Options, now time.Time) IndexLock {
	first := opts.Stat(path)
	out := IndexLock{Path: path, Present: first.Exists}
	if first.Err != "" {
		out.Detail = first.Err
		return out
	}
	if !first.Exists {
		return out
	}

	if opts.SettleWindow > 0 {
		opts.Sleep(opts.SettleWindow)
		out.SettleMillis = int64(opts.SettleWindow / time.Millisecond)
	}
	second := opts.Stat(path)
	switch {
	case second.Err != "":
		// The re-sample failed. Fall back to the first sample and say so: Advancing stays
		// false, which never widens a reap on its own — the age gates still have to pass.
		out.Detail = second.Err
		second = first
	case !second.Exists:
		// The lock cleared itself inside the settle window. There is nothing left to
		// reclaim, and reporting a vanished file as present would invite the actuator to
		// delete whatever a live writer creates next.
		out.Present = false
		return out
	default:
		out.Advancing = second.ModTime.After(first.ModTime) || second.Size != first.Size
	}

	if !second.ModTime.IsZero() {
		age := now.Sub(second.ModTime)
		if age < 0 {
			age = 0
		}
		out.ModTime = second.ModTime.UTC().Format(time.RFC3339)
		out.AgeSeconds = int64(age / time.Second)
		if !out.Advancing && opts.OwnerDeadIndexAge > 0 && age >= opts.OwnerDeadIndexAge {
			out.FrozenHint = true
		}
		if opts.StaleIndexAge > 0 && age >= opts.StaleIndexAge {
			out.StaleHint = true
			stale := "index.lock is older than " + opts.StaleIndexAge.String() + "; inspect live git processes before deleting"
			// A failed re-sample already wrote its error here; keep it rather than let the
			// age note bury the reason the settle witness could not run.
			if out.Detail == "" {
				out.Detail = stale
			} else {
				out.Detail += "; " + stale
			}
		}
	}
	return out
}

// NextIndexGlob is the filename pattern git leaves behind when an index writer dies
// between creating its temp index and renaming it over .git/index.
const NextIndexGlob = "next-index-*.lock"

// nextIndexPIDRe extracts the writing process's pid from a next-index temp filename.
var nextIndexPIDRe = regexp.MustCompile(`(?i)^next-index-(\d+)\.lock$`)

// probeNextIndexLocks enumerates the .git/next-index-<pid>.lock residue and ages each
// entry, exactly like probeIndexLock does for the single index.lock. It NEVER removes a
// file — the reap decision is DecideNextIndexReclaim's and the removal is the CLI
// actuator's. A file that vanishes between the glob and the stat is simply dropped: it
// is already gone, so there is nothing left to reclaim. Returns a non-empty string when
// the scan itself failed, so the caller can surface it as a report error rather than
// silently reporting "no residue" from a broken probe.
func probeNextIndexLocks(gitDir string, glob GlobFunc, stat FileStatFunc, alive PIDAliveFunc, now time.Time, staleAge time.Duration) ([]NextIndexLock, string) {
	if strings.TrimSpace(gitDir) == "" {
		return nil, ""
	}
	matches, err := glob(filepath.Join(gitDir, NextIndexGlob))
	if err != nil {
		return nil, err.Error()
	}
	sort.Strings(matches)
	out := make([]NextIndexLock, 0, len(matches))
	for _, path := range matches {
		f := stat(path)
		if !f.Exists {
			// Raced away (or unstattable) between the glob and the stat — nothing to reap.
			continue
		}
		row := NextIndexLock{Path: path}
		if m := nextIndexPIDRe.FindStringSubmatch(filepath.Base(path)); m != nil {
			if pid, cerr := strconv.Atoi(m[1]); cerr == nil && pid > 0 {
				row.PID = pid
				row.OwnerAlive = alive(pid)
			}
		}
		if !f.ModTime.IsZero() {
			age := now.Sub(f.ModTime)
			if age < 0 {
				age = 0
			}
			row.ModTime = f.ModTime.UTC().Format(time.RFC3339)
			row.AgeSeconds = int64(age / time.Second)
			row.StaleHint = staleAge > 0 && age >= staleAge
		}
		out = append(out, row)
	}
	if len(out) == 0 {
		return nil, ""
	}
	return out, ""
}

func finalize(rep *Report) {
	switch {
	case rep.CommitLock.Stale:
		rep.OK = false
		rep.Verdict = VerdictStale
		rep.Reason = fmt.Sprintf("fak commit lock is held by dead PID %d", rep.CommitLock.HolderPID)
		rep.NextAction = "run `fak tree-doctor --apply` or retry `fak commit`; both use the PID-guarded stale-lock reaper"
	case rep.IndexLock.StaleHint && !rep.IndexLock.Advancing && len(rep.LiveWriters) == 0:
		rep.OK = false
		rep.Verdict = VerdictBlocked
		rep.Reason = "git index.lock is present with no matching live writer found"
		rep.NextAction = "inspect live git/fak processes before removing .git/index.lock; never delete it while git is running"
	case rep.CommitLock.Present && rep.CommitLock.HolderPID == 0:
		rep.OK = false
		rep.Verdict = VerdictUnknown
		rep.Reason = "fak commit lock is present but has no parseable holder PID"
		rep.NextAction = "inspect the lock file and process inventory; do not remove it unless the owner is proven dead"
	case rep.CommitLock.Present || rep.IndexLock.Present || hasRelevantWriter(*rep):
		rep.OK = true
		rep.Verdict = VerdictBusy
		rep.Reason = busyReason(*rep)
		rep.NextAction = "wait for the live writer to finish; rerun status if the lock does not clear"
	default:
		rep.OK = true
		rep.Verdict = VerdictClear
		rep.Reason = "no fak commit lock, no git index lock, and no matching live writer found"
		rep.NextAction = "commit lane is clear; run `fak commit ...` when ready"
	}
}

func hasRelevantWriter(rep Report) bool {
	for _, w := range rep.LiveWriters {
		if w.Role == "owner" || w.RepoMatched {
			return true
		}
	}
	return false
}

func busyReason(rep Report) string {
	switch {
	case rep.CommitLock.HolderPID > 0 && rep.CommitLock.HolderAlive:
		return fmt.Sprintf("fak commit lock is held by live PID %d", rep.CommitLock.HolderPID)
	case rep.IndexLock.Present:
		return "git index.lock is present and a matching live writer was found"
	case len(rep.LiveWriters) > 0:
		return "matching live git/fak writer process found"
	default:
		return "commit lane is busy"
	}
}

var (
	fakCommitRe       = regexp.MustCompile(`(?i)(^|\s|[\\/])fak(?:\.exe)?["']?\s+commit(\s|$)`)
	fakCommitStatusRe = regexp.MustCompile(`(?i)(^|\s|[\\/])fak(?:\.exe)?["']?\s+commit\s+status(\s|$)`)
	gitWriterRe       = regexp.MustCompile(`(?i)(^|\s|[\\/])git(?:\.exe)?["']?\s+(add|commit|merge|rebase|checkout|reset)(\s|$)`)
)

func classifyProcesses(root string, holderPID int, procs []Process) (*ProcessFact, []ProcessFact, []ProcessFact) {
	var owner *ProcessFact
	var writers, queue []ProcessFact
	for _, p := range procs {
		f, ok := classifyProcess(root, p)
		if !ok {
			continue
		}
		if holderPID > 0 && p.PID == holderPID {
			f.Role = "owner"
			f.Confidence = "lock_pid"
			cp := f
			owner = &cp
		}
		writers = append(writers, f)
		if f.Match == "fak_commit" && p.PID != holderPID {
			q := f
			q.Role = "queued_candidate"
			queue = append(queue, q)
		}
	}
	sortProcessFacts(writers)
	sortProcessFacts(queue)
	return owner, writers, queue
}

func classifyProcess(root string, p Process) (ProcessFact, bool) {
	cmd := strings.TrimSpace(p.Command)
	if cmd == "" {
		cmd = strings.TrimSpace(p.Name)
	}
	if fakCommitStatusRe.MatchString(cmd) {
		return ProcessFact{}, false
	}
	match := ""
	switch {
	case fakCommitRe.MatchString(cmd):
		match = "fak_commit"
	case gitWriterRe.MatchString(cmd):
		match = "git_writer"
	default:
		return ProcessFact{}, false
	}
	repoMatched := commandMentionsRoot(cmd, root)
	conf := "global_process_match"
	if repoMatched {
		conf = "repo_command_match"
	}
	return ProcessFact{
		PID:         p.PID,
		ParentPID:   p.ParentPID,
		Name:        strings.TrimSpace(p.Name),
		Command:     boundCommand(cmd),
		Role:        "writer",
		Match:       match,
		RepoMatched: repoMatched,
		Confidence:  conf,
	}, true
}

func sortProcessFacts(rows []ProcessFact) {
	sort.Slice(rows, func(i, j int) bool { return rows[i].PID < rows[j].PID })
}

func commandMentionsRoot(command, root string) bool {
	root = strings.TrimSpace(root)
	if root == "" {
		return false
	}
	clean := filepath.Clean(root)
	cands := []string{
		strings.ToLower(clean),
		strings.ToLower(filepath.ToSlash(clean)),
		strings.ToLower(strings.ReplaceAll(clean, `\`, `/`)),
	}
	cmd := strings.ToLower(command)
	cmdSlash := strings.ReplaceAll(cmd, `\`, `/`)
	for _, c := range cands {
		if c != "" && (strings.Contains(cmd, c) || strings.Contains(cmdSlash, strings.ReplaceAll(c, `\`, `/`))) {
			return true
		}
	}
	return false
}

func boundCommand(s string) string {
	s = strings.TrimSpace(s)
	const max = 500
	if len(s) <= max {
		return s
	}
	return s[:max] + "..."
}

func cleanPath(path string) string {
	path = strings.TrimSpace(path)
	if path == "" {
		return ""
	}
	if abs, err := filepath.Abs(path); err == nil {
		path = abs
	}
	return filepath.Clean(path)
}

func DefaultProcessList(ctx context.Context) ([]Process, error) {
	if runtime.GOOS == "windows" {
		return windowsProcessList(ctx)
	}
	return unixProcessList(ctx)
}

func windowsProcessList(ctx context.Context) ([]Process, error) {
	const script = `$ErrorActionPreference = 'Stop'
$rows = @(Get-CimInstance Win32_Process | Where-Object {
  ($_.Name -match '^(fak(\.exe)?|git(\.exe)?)$') -and
  ($_.CommandLine -match '(?i)(\bfak(\.exe)?["'']?\s+commit\b|\bgit(\.exe)?["'']?\s+(add|commit|merge|rebase|checkout|reset)\b)')
} | ForEach-Object {
  [pscustomobject]@{
    pid = [int]$_.ProcessId
    parent_pid = [int]$_.ParentProcessId
    name = [string]$_.Name
    command = [string]$_.CommandLine
  }
})
ConvertTo-Json -Depth 3 -InputObject $rows`
	out, err := runProcessJSON(ctx, "powershell", "-NoProfile", "-NonInteractive", "-Command", script)
	if err != nil {
		out, err = runProcessJSON(ctx, "pwsh", "-NoProfile", "-NonInteractive", "-Command", script)
	}
	if err != nil {
		return nil, err
	}
	return decodeProcessJSON(out)
}

func unixProcessList(ctx context.Context) ([]Process, error) {
	cmd := exec.CommandContext(ctx, "ps", "-eo", "pid=,ppid=,comm=,args=")
	windowgate.ConfigureBackgroundCommand(cmd)
	out, err := cmd.Output()
	if err != nil {
		return nil, err
	}
	var procs []Process
	for _, line := range strings.Split(string(out), "\n") {
		fields := strings.Fields(line)
		if len(fields) < 4 {
			continue
		}
		pid, perr := strconv.Atoi(fields[0])
		ppid, pperr := strconv.Atoi(fields[1])
		if perr != nil || pperr != nil {
			continue
		}
		procs = append(procs, Process{
			PID:       pid,
			ParentPID: ppid,
			Name:      fields[2],
			Command:   strings.Join(fields[3:], " "),
		})
	}
	return procs, nil
}

func runProcessJSON(ctx context.Context, name string, args ...string) ([]byte, error) {
	cmd := exec.CommandContext(ctx, name, args...)
	windowgate.ConfigureBackgroundCommand(cmd)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		detail := strings.TrimSpace(stderr.String())
		if detail == "" {
			detail = err.Error()
		}
		return nil, fmt.Errorf("%s: %s", name, detail)
	}
	return out, nil
}

func decodeProcessJSON(data []byte) ([]Process, error) {
	data = bytes.TrimSpace(data)
	if len(data) == 0 || bytes.Equal(data, []byte("null")) {
		return nil, nil
	}
	var rows []Process
	if data[0] == '[' {
		if err := json.Unmarshal(data, &rows); err != nil {
			return nil, err
		}
		return rows, nil
	}
	var row Process
	if err := json.Unmarshal(data, &row); err != nil {
		return nil, err
	}
	return []Process{row}, nil
}
