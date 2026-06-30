// Package commitlane reports the read-only state of the shared fak commit lane.
//
// It observes the two local files that normally block a path-scoped commit:
// <gitdir>/fak-commit.lock, owned by fak's safecommit path, and <gitdir>/index.lock,
// owned by git itself. It never removes either file. Process inventory is best-effort
// and only used to make a lock owner or likely queue visible.
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

type Options struct {
	Dir           string
	Runner        Runner
	ProbeLock     ProbeLockFunc
	Stat          FileStatFunc
	ProcessList   ProcessListFunc
	Now           func() time.Time
	StaleIndexAge time.Duration
}

type Report struct {
	Schema       string        `json:"schema"`
	OK           bool          `json:"ok"`
	Verdict      string        `json:"verdict"`
	Reason       string        `json:"reason,omitempty"`
	NextAction   string        `json:"next_action,omitempty"`
	RepoRoot     string        `json:"repo_root,omitempty"`
	GitDir       string        `json:"git_dir,omitempty"`
	CommitLock   CommitLock    `json:"commit_lock"`
	IndexLock    IndexLock     `json:"index_lock"`
	Owner        *ProcessFact  `json:"owner,omitempty"`
	Queue        []ProcessFact `json:"queue,omitempty"`
	LiveWriters  []ProcessFact `json:"live_writers,omitempty"`
	ProcessProbe string        `json:"process_probe"`
	Errors       []string      `json:"errors,omitempty"`
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
	Detail     string `json:"detail,omitempty"`
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
	rep.IndexLock = probeIndexLock(filepath.Join(rep.GitDir, "index.lock"), opts.Stat, now, opts.StaleIndexAge)

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
	if opts.Now == nil {
		opts.Now = time.Now
	}
	if opts.StaleIndexAge == 0 {
		opts.StaleIndexAge = DefaultStaleIndexAge
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

func probeIndexLock(path string, stat FileStatFunc, now time.Time, staleAge time.Duration) IndexLock {
	f := stat(path)
	out := IndexLock{Path: path, Present: f.Exists}
	if f.Err != "" {
		out.Detail = f.Err
		return out
	}
	if !f.Exists {
		return out
	}
	if !f.ModTime.IsZero() {
		age := now.Sub(f.ModTime)
		if age < 0 {
			age = 0
		}
		out.ModTime = f.ModTime.UTC().Format(time.RFC3339)
		out.AgeSeconds = int64(age / time.Second)
		if staleAge > 0 && age >= staleAge {
			out.StaleHint = true
			out.Detail = "index.lock is older than " + staleAge.String() + "; inspect live git processes before deleting"
		}
	}
	return out
}

func finalize(rep *Report) {
	switch {
	case rep.CommitLock.Stale:
		rep.OK = false
		rep.Verdict = VerdictStale
		rep.Reason = fmt.Sprintf("fak commit lock is held by dead PID %d", rep.CommitLock.HolderPID)
		rep.NextAction = "run `fak tree-doctor --apply` or retry `fak commit`; both use the PID-guarded stale-lock reaper"
	case rep.IndexLock.StaleHint && len(rep.LiveWriters) == 0:
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
