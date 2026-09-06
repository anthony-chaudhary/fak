// Package worktreewitness runs commands inside transient detached git
// worktrees pinned at trunk tip to isolate verdicts from uncommitted WIP.
package worktreewitness

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// DefaultRef is the remote-tracking trunk tip ref pinned by default.
const DefaultRef = "origin/main"

// Runner executes a command in dir and returns combined output, exit code, and start error.
type Runner func(dir string, name string, args ...string) (out string, code int, err error)

// Config parameterizes an isolated worktree witness run.
type Config struct {
	Repo          string
	Ref           string
	Fetch         bool
	Command       []string
	ArchiveDir    string
	ScratchParent string
}

// Result contains execution status, exit code, and revision metadata from a witness run.
type Result struct {
	Green    bool   `json:"green"`
	ExitCode int    `json:"exit_code"`
	Ref      string `json:"ref"`
	SHA      string `json:"sha"`
	ShortSHA string `json:"short_sha"`
	Command  string `json:"command"`
	Output   string `json:"output,omitempty"`
	Archived string `json:"archived,omitempty"`
	ReapNote string `json:"reap_note,omitempty"`
}

func classifyGreen(code int, ranOK bool) bool { return ranOK && code == 0 }

// ShortSHA truncates a commit SHA to 12 characters when longer.
func ShortSHA(sha string) string {
	if len(sha) > 12 {
		return sha[:12]
	}
	return sha
}

// Run executes cfg.Command inside an isolated, transient worktree pinned to cfg.Ref.
func Run(cfg Config, gitRun Runner, cmdRun Runner) (res Result, err error) {
	if len(cfg.Command) == 0 {
		return Result{}, fmt.Errorf("worktreewitness: no command to witness")
	}
	ref := cfg.Ref
	if ref == "" {
		ref = DefaultRef
	}
	res = Result{Ref: ref, Command: strings.Join(cfg.Command, " ")}

	if cfg.Fetch {
		remote, refspec := splitRef(ref)
		_, _, _ = gitRun(cfg.Repo, "git", "-C", cfg.Repo, "fetch", remote, refspec)
	}

	sha, err := resolveRef(cfg.Repo, ref, gitRun)
	if err != nil {
		return res, err
	}
	res.SHA, res.ShortSHA = sha, ShortSHA(sha)

	top, moduleRel, err := repoTopAndRel(cfg.Repo, gitRun)
	if err != nil {
		return res, err
	}

	parent, err := os.MkdirTemp(cfg.ScratchParent, "fak-witness-")
	if err != nil {
		return res, fmt.Errorf("worktreewitness: temp dir: %w", err)
	}
	defer os.RemoveAll(parent)
	wt := filepath.Join(parent, "wt")

	if out, code, err := gitRun(top, "git", "-C", top, "worktree", "add", "--detach", wt, sha); err != nil || code != 0 {
		return res, fmt.Errorf("worktreewitness: worktree add %s: code=%d err=%v: %s", ShortSHA(sha), code, err, out)
	}
	defer func() {
		archived, note := reap(top, wt, cfg.ArchiveDir, res.ShortSHA, gitRun)
		res.Archived, res.ReapNote = archived, note
	}()

	module := wt
	if moduleRel != "." && moduleRel != "" {
		module = filepath.Join(wt, filepath.FromSlash(moduleRel))
	}

	out, code, runErr := cmdRun(module, cfg.Command[0], cfg.Command[1:]...)
	res.Output = out
	res.ExitCode = code
	res.Green = classifyGreen(code, runErr == nil)
	if runErr != nil {
		return res, fmt.Errorf("worktreewitness: run %q: %w", res.Command, runErr)
	}
	return res, nil
}

func reap(top, wt, archiveDir, short string, gitRun Runner) (archived, note string) {
	dirty := worktreeDirty(wt, gitRun)
	if dirty && archiveDir != "" {
		if p, err := archiveWorktree(wt, archiveDir, short, gitRun); err == nil {
			archived = p
		} else {
			note = "archive failed: " + err.Error()
		}
	}
	if _, code, err := gitRun(top, "git", "-C", top, "worktree", "remove", "--force", wt); err != nil || code != 0 {
		_, _, _ = gitRun(top, "git", "-C", top, "worktree", "prune")
		if note == "" {
			note = "remove needed prune fallback"
		}
	}
	return archived, note
}

func worktreeDirty(wt string, gitRun Runner) bool {
	out, code, err := gitRun(wt, "git", "-C", wt, "status", "--porcelain")
	if err != nil || code != 0 {
		return true
	}
	return strings.TrimSpace(out) != ""
}

func archiveWorktree(wt, archiveDir, short string, gitRun Runner) (string, error) {
	dst := filepath.Join(archiveDir, "witness-"+short)
	if err := os.MkdirAll(dst, 0o755); err != nil {
		return "", err
	}
	diff, _, _ := gitRun(wt, "git", "-C", wt, "diff", "HEAD")
	if err := os.WriteFile(filepath.Join(dst, "diff.patch"), []byte(diff), 0o644); err != nil {
		return "", err
	}
	status, _, _ := gitRun(wt, "git", "-C", wt, "status", "--porcelain")
	if err := os.WriteFile(filepath.Join(dst, "untracked.txt"), []byte(status), 0o644); err != nil {
		return "", err
	}
	return dst, nil
}

func resolveRef(repo, ref string, gitRun Runner) (string, error) {
	out, code, err := gitRun(repo, "git", "-C", repo, "rev-parse", ref)
	if err != nil {
		return "", fmt.Errorf("worktreewitness: rev-parse %s: %w", ref, err)
	}
	if code != 0 {
		return "", fmt.Errorf("worktreewitness: rev-parse %s: exit %d: %s", ref, code, strings.TrimSpace(out))
	}
	sha := strings.TrimSpace(out)
	if sha == "" {
		return "", fmt.Errorf("worktreewitness: rev-parse %s resolved to empty", ref)
	}
	return sha, nil
}

func repoTopAndRel(repo string, gitRun Runner) (top, moduleRel string, err error) {
	out, code, terr := gitRun(repo, "git", "-C", repo, "rev-parse", "--show-toplevel")
	if terr != nil || code != 0 {
		return "", "", fmt.Errorf("worktreewitness: rev-parse --show-toplevel: code=%d err=%v", code, terr)
	}
	top = filepath.Clean(strings.TrimSpace(out))
	abs, aerr := filepath.Abs(repo)
	if aerr != nil {
		return "", "", aerr
	}
	rel, rerr := filepath.Rel(top, abs)
	if rerr != nil || rel == "" || rel == "." || strings.HasPrefix(rel, "..") {
		rel = "."
	}
	return top, filepath.ToSlash(rel), nil
}

func splitRef(ref string) (remote, refspec string) {
	if i := strings.IndexByte(ref, '/'); i > 0 {
		return ref[:i], ref[i+1:]
	}
	return "origin", ref
}
