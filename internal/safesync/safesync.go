package safesync

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

const (
	StateInSync      = "in-sync"
	StateBehind      = "behind"
	StateAhead       = "ahead"
	StateDiverged    = "diverged"
	StateNoRemoteRef = "no-remote-ref"
)

// Runner executes a git subcommand in repo. Err is non-nil only when git could
// not be started; a non-zero git exit is reported through Code and Stderr.
type Runner func(ctx context.Context, repo string, args ...string) RunResult

type RunResult struct {
	Stdout []byte
	Stderr []byte
	Code   int
	Err    error
}

type Options struct {
	Repo   string
	Remote string
	Branch string
	Fetch  bool
	Runner Runner `json:"-"`
}

type Entry struct {
	Status string `json:"status"`
	Path   string `json:"path"`
}

type Assessment struct {
	OK         bool    `json:"ok"`
	State      string  `json:"state"`
	Head       string  `json:"head,omitempty"`
	Target     string  `json:"target,omitempty"`
	TargetRef  string  `json:"target_ref,omitempty"`
	Branch     string  `json:"branch,omitempty"`
	WriteCount int     `json:"write_count,omitempty"`
	Identical  []Entry `json:"identical,omitempty"`
	Divergent  []Entry `json:"divergent,omitempty"`
	Reason     string  `json:"reason,omitempty"`
	Applied    bool    `json:"applied,omitempty"`
	NewHead    string  `json:"new_head,omitempty"`
}

type GitError struct {
	Args   []string
	Code   int
	Detail string
}

func (e *GitError) Error() string {
	if e == nil {
		return ""
	}
	detail := strings.TrimSpace(e.Detail)
	if detail == "" {
		detail = "git command failed"
	}
	return fmt.Sprintf("git %s -> %d: %s", strings.Join(e.Args, " "), e.Code, detail)
}

// Assess reports whether repo can safely fast-forward to remote/branch without
// clobbering dirty shared-worktree content. It is read-only except for the
// optional fetch.
func Assess(ctx context.Context, opts Options) (Assessment, error) {
	opts = normalizeOptions(opts)
	run := opts.Runner

	branch := strings.TrimSpace(opts.Branch)
	if branch == "" {
		var err error
		branch, err = currentBranch(ctx, run, opts.Repo)
		if err != nil {
			return Assessment{}, err
		}
	}
	targetRef := opts.Remote + "/" + branch

	if opts.Fetch {
		if _, err := checked(ctx, run, opts.Repo, "fetch", opts.Remote, branch); err != nil {
			return Assessment{}, err
		}
	}

	head, err := rev(ctx, run, opts.Repo, "HEAD")
	if err != nil {
		return Assessment{}, err
	}
	target, err := rev(ctx, run, opts.Repo, targetRef)
	if err != nil {
		var ge *GitError
		if errors.As(err, &ge) {
			return Assessment{
				OK:        false,
				State:     StateNoRemoteRef,
				TargetRef: targetRef,
				Branch:    branch,
				Reason:    "remote-tracking ref " + targetRef + " not found; fetch first",
			}, nil
		}
		return Assessment{}, err
	}

	base := Assessment{Head: head, Target: target, TargetRef: targetRef, Branch: branch}
	if head == target {
		base.OK = true
		base.State = StateInSync
		return base, nil
	}
	targetIsAncestor, err := isAncestor(ctx, run, opts.Repo, target, head)
	if err != nil {
		return Assessment{}, err
	}
	if targetIsAncestor {
		base.State = StateAhead
		base.Reason = "local branch is ahead of remote; nothing to fast-forward (push instead)"
		return base, nil
	}
	headIsAncestor, err := isAncestor(ctx, run, opts.Repo, head, target)
	if err != nil {
		return Assessment{}, err
	}
	if !headIsAncestor {
		base.State = StateDiverged
		base.Reason = "local and remote have diverged; not a fast-forward"
		return base, nil
	}

	entries, err := ffWriteSet(ctx, run, opts.Repo, head, target)
	if err != nil {
		return Assessment{}, err
	}
	identical, divergent := classify(opts.Repo, run, ctx, head, target, entries)
	base.State = StateBehind
	base.WriteCount = len(entries)
	base.Identical = identical
	base.Divergent = divergent
	base.OK = len(divergent) == 0
	if base.OK {
		base.Reason = "every fast-forward path is clean at HEAD or already byte-identical to the remote; safe to fast-forward"
	} else {
		base.Reason = fmt.Sprintf("%d path(s) diverge locally; refusing; sync at a quiescent moment or resolve by hand", len(divergent))
	}
	return base, nil
}

// Apply performs the same assessment and runs the fast-forward only when Assess
// says the behind state is safe. Refused states leave the tree untouched.
func Apply(ctx context.Context, opts Options) (Assessment, error) {
	opts = normalizeOptions(opts)
	run := opts.Runner
	info, err := Assess(ctx, opts)
	if err != nil {
		return info, err
	}
	if info.State == StateInSync {
		return info, nil
	}
	if info.State != StateBehind || !info.OK {
		info.Applied = false
		return info, nil
	}
	if err := applyFastForward(ctx, run, opts.Repo, info); err != nil {
		return info, err
	}
	newHead, err := rev(ctx, run, opts.Repo, "HEAD")
	if err != nil {
		return info, err
	}
	info.Applied = true
	info.NewHead = newHead
	return info, nil
}

func normalizeOptions(opts Options) Options {
	if strings.TrimSpace(opts.Repo) == "" {
		opts.Repo = "."
	}
	if strings.TrimSpace(opts.Remote) == "" {
		opts.Remote = "origin"
	}
	if opts.Runner == nil {
		opts.Runner = RealRunner
	}
	return opts
}

func RealRunner(ctx context.Context, repo string, args ...string) RunResult {
	cmdArgs := append([]string{"-C", repo}, args...)
	cmd := exec.CommandContext(ctx, "git", cmdArgs...)
	windowgate.ConfigureBackgroundCommand(cmd)
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
	return RunResult{Stdout: stdout.Bytes(), Stderr: stderr.Bytes(), Code: code, Err: err}
}

func currentBranch(ctx context.Context, run Runner, repo string) (string, error) {
	out, err := checked(ctx, run, repo, "rev-parse", "--abbrev-ref", "HEAD")
	if err != nil {
		return "", err
	}
	branch := strings.TrimSpace(string(out))
	if branch == "" || branch == "HEAD" {
		return "", fmt.Errorf("detached HEAD; no branch to sync")
	}
	return branch, nil
}

func rev(ctx context.Context, run Runner, repo, ref string) (string, error) {
	out, err := checked(ctx, run, repo, "rev-parse", "--verify", ref)
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(out)), nil
}

func checked(ctx context.Context, run Runner, repo string, args ...string) ([]byte, error) {
	res := run(ctx, repo, args...)
	if res.Err != nil {
		return nil, res.Err
	}
	if res.Code != 0 {
		detail := strings.TrimSpace(string(res.Stderr))
		if detail == "" {
			detail = strings.TrimSpace(string(res.Stdout))
		}
		return nil, &GitError{Args: append([]string(nil), args...), Code: res.Code, Detail: detail}
	}
	return res.Stdout, nil
}

func isAncestor(ctx context.Context, run Runner, repo, a, b string) (bool, error) {
	res := run(ctx, repo, "merge-base", "--is-ancestor", a, b)
	if res.Err != nil {
		return false, res.Err
	}
	switch res.Code {
	case 0:
		return true, nil
	case 1:
		return false, nil
	default:
		detail := strings.TrimSpace(string(res.Stderr))
		if detail == "" {
			detail = strings.TrimSpace(string(res.Stdout))
		}
		return false, &GitError{Args: []string{"merge-base", "--is-ancestor", a, b}, Code: res.Code, Detail: detail}
	}
}

func ffWriteSet(ctx context.Context, run Runner, repo, head, target string) ([]Entry, error) {
	out, err := checked(ctx, run, repo, "diff", "--name-status", "-z", head, target)
	if err != nil {
		return nil, err
	}
	return parseNameStatusZ(out), nil
}

func parseNameStatusZ(out []byte) []Entry {
	fields := strings.Split(string(out), "\x00")
	entries := make([]Entry, 0, len(fields)/2)
	for i := 0; i < len(fields); {
		status := fields[i]
		if status == "" {
			i++
			continue
		}
		code := status[:1]
		if code == "R" || code == "C" {
			if i+2 >= len(fields) {
				break
			}
			entries = append(entries, Entry{Status: status, Path: fields[i+1]})
			entries = append(entries, Entry{Status: status, Path: fields[i+2]})
			i += 3
			continue
		}
		if i+1 >= len(fields) {
			break
		}
		entries = append(entries, Entry{Status: code, Path: fields[i+1]})
		i += 2
	}
	return entries
}

func classify(repo string, run Runner, ctx context.Context, head, target string, entries []Entry) (identical, divergent []Entry) {
	for _, e := range entries {
		safe := false
		switch e.Status {
		case "M":
			wt, ok := worktreeBytes(repo, e.Path)
			tgt, exists := blobAt(ctx, run, repo, target, e.Path)
			base, baseExists := blobAt(ctx, run, repo, head, e.Path)
			safe = ok && ((exists && bytes.Equal(wt, tgt)) || (baseExists && bytes.Equal(wt, base)))
		case "A":
			wt, ok := worktreeBytes(repo, e.Path)
			tgt, exists := blobAt(ctx, run, repo, target, e.Path)
			safe = !ok || (exists && bytes.Equal(wt, tgt))
		case "D":
			wt, ok := worktreeBytes(repo, e.Path)
			base, exists := blobAt(ctx, run, repo, head, e.Path)
			safe = !ok || (exists && bytes.Equal(wt, base))
		default:
			safe = false
		}
		if safe {
			identical = append(identical, e)
		} else {
			divergent = append(divergent, e)
		}
	}
	return identical, divergent
}

func blobAt(ctx context.Context, run Runner, repo, ref, path string) ([]byte, bool) {
	res := run(ctx, repo, "show", ref+":"+path)
	if res.Err != nil || res.Code != 0 {
		return nil, false
	}
	return res.Stdout, true
}

func worktreeBytes(repo, path string) ([]byte, bool) {
	full, ok := safeWorktreePath(repo, path)
	if !ok {
		return nil, false
	}
	b, err := os.ReadFile(full)
	if err != nil {
		return nil, false
	}
	return b, true
}

func applyFastForward(ctx context.Context, run Runner, repo string, info Assessment) error {
	var modified []string
	for _, e := range info.Identical {
		if e.Status == "M" {
			modified = append(modified, e.Path)
		}
	}
	if len(modified) > 0 {
		args := append([]string{"checkout", "HEAD", "--"}, modified...)
		if _, err := checked(ctx, run, repo, args...); err != nil {
			return err
		}
	}
	for _, e := range info.Identical {
		if e.Status != "A" {
			continue
		}
		full, ok := safeWorktreePath(repo, e.Path)
		if !ok {
			return fmt.Errorf("unsafe repo path %q", e.Path)
		}
		if err := os.Remove(full); err != nil && !errors.Is(err, os.ErrNotExist) {
			return err
		}
	}
	if _, err := checked(ctx, run, repo, "merge", "--ff-only", info.TargetRef); err != nil {
		return err
	}
	return nil
}

func safeWorktreePath(repo, path string) (string, bool) {
	if filepath.IsAbs(path) || path == "" {
		return "", false
	}
	clean := filepath.Clean(filepath.FromSlash(path))
	if clean == "." || clean == ".." || strings.HasPrefix(clean, ".."+string(filepath.Separator)) {
		return "", false
	}
	return filepath.Join(repo, clean), true
}
