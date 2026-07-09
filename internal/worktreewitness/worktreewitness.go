// Package worktreewitness runs a command inside a transient detached git
// worktree pinned at origin/main, so the verdict reflects the trunk tip and NOT
// the caller's dirty working tree.
//
// WHY THIS EXISTS
// ---------------
// This repo is a live multi-session fleet: the shared working tree is almost
// always dirty with some peer's uncommitted WIP. That makes the single most
// repeated trap here "a peer's uncommitted edit reds MY baseline" — e.g. another
// session's half-written change breaks the whole-package build, so `go test
// ./cmd/fak` fails on a tree the trunk tip would compile clean. The verdict is a
// false red: the trunk is green, but the dirty tree lies.
//
// A detached worktree at origin/main is the ONLY place to prove HEAD-is-green
// without a peer's dirty edits contaminating the verdict. AGENTS.md's safe-worktree
// envelope names this exact case as where a transient worktree ADDS velocity: it is
// the difference between shipping and thrashing on false reds. This package turns
// that hand-rolled recipe (`git worktree add --detach <tmp> origin/main` → run →
// `git worktree remove --force`) into one guarded, self-reaping operation.
//
// THE ENVELOPE THIS STAYS INSIDE
// ------------------------------
//   - Anchor: detached at the resolved origin/main SHA — never a named branch, so
//     nothing here can trip the OFF_TRUNK trunk guard. It commits nothing; it only
//     observes.
//   - Lifetime: transient — the worktree is created, used, and removed within one
//     Run call. A deferred reap always fires, even on a panic in the command.
//   - Writes: the worktree is a read-mostly staging area. If the command dirties it
//     (a build cache, a generated file), the reap ARCHIVES the diff + untracked
//     files first (best-effort) before force-removing — losing nothing — exactly as
//     tools/worktree_doctor.py --sweep-disposable does for dead scratch worktrees.
//
// Like internal/shipgate and internal/dojocal/worktree.go, this is a git/exec
// harness, not the dispatch hot path, so the os/exec-absence proof does not apply.
package worktreewitness

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// DefaultRef is the ref a witness pins by default: the fetched trunk tip. It is
// deliberately origin/main (the remote-tracking ref) rather than local main —
// local main on a hot shared tree lags or leads the real trunk by peer commits,
// and the whole point is to witness the TRUNK, not this checkout's idea of it.
const DefaultRef = "origin/main"

// Runner executes git (or the witnessed command) in dir, returning combined
// output and the process exit code. err is non-nil ONLY when the process could
// not be started at all (git missing, dir gone) — a non-zero exit is reported via
// code, never err, because a red command is the signal being measured, not a
// harness failure. A test injects a fake; production wires os/exec.
type Runner func(dir string, name string, args ...string) (out string, code int, err error)

// Config parameterizes a witness run.
type Config struct {
	// Repo is a path inside the working copy; git resolves the repo from there.
	Repo string
	// Ref is the ref to pin and detach at (default DefaultRef). Resolved to a SHA
	// ONCE so the witness is stable even if the trunk advances mid-run.
	Ref string
	// Fetch, when true, refreshes Ref from origin before resolving it, so the
	// witness reflects the freshest trunk tip rather than a stale local ref.
	Fetch bool
	// Command is the witnessed command + args, run inside the worktree's module
	// dir. Empty is an error — there is nothing to witness.
	Command []string
	// ArchiveDir is where a dirty worktree's diff + untracked files are saved
	// before reap. Empty disables archiving (the reap still force-removes).
	ArchiveDir string
	// ScratchParent is the parent dir for the ephemeral worktree ("" => os.TempDir).
	ScratchParent string
}

// Result is the witness verdict. Green is the headline bit; ExitCode carries the
// witnessed command's own exit status; SHA/ShortSHA name exactly what was witnessed
// so a caller can cite the trunk tip the verdict pins to.
type Result struct {
	Green    bool   `json:"green"`
	ExitCode int    `json:"exit_code"`
	Ref      string `json:"ref"`
	SHA      string `json:"sha"`
	ShortSHA string `json:"short_sha"`
	Command  string `json:"command"`
	// Output is the witnessed command's combined output (trimmed tail on very long
	// runs is the caller's concern; the core returns it whole).
	Output string `json:"output,omitempty"`
	// Archived is the path a dirty worktree's changes were saved to before reap,
	// or "" if the worktree was clean or archiving was disabled.
	Archived string `json:"archived,omitempty"`
	// ReapNote records a non-fatal cleanup wrinkle (e.g. a Windows-locked dir that
	// needed a prune fallback). Empty on a clean reap.
	ReapNote string `json:"reap_note,omitempty"`
}

// classifyGreen is the pure verdict rule: a witness is GREEN iff the command ran
// AND exited 0. It is separated from exec so the keep-bit is unit-testable and can
// never be set by anything but a real observed exit code.
func classifyGreen(code int, ranOK bool) bool { return ranOK && code == 0 }

// ShortSHA returns the first 12 chars of a SHA (or the whole string if shorter),
// matching the short form the release + dojo harnesses cite.
func ShortSHA(sha string) string {
	if len(sha) > 12 {
		return sha[:12]
	}
	return sha
}

// Run pins cfg.Ref to a SHA, creates a transient detached worktree there, runs
// cfg.Command inside it, and always reaps the worktree (archiving first if it was
// dirtied). It returns the witness Result. The trunk working tree is never touched.
//
// git is driven through gitRun; the witnessed command through cmdRun. Separating
// the two runners lets a test inject a fake git while running the real command, or
// vice versa, without either polluting the other's call log.
//
// The return values are NAMED (res, err) on purpose: the reap runs in a deferred
// closure that records the archive path + reap note, and a deferred write is only
// observed by the caller when it lands on the NAMED return value. With an unnamed
// return, `return res, nil` copies res before the defer fires and the reap's
// Archived/ReapNote writes are silently lost.
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
		// Best-effort refresh; a fetch failure (offline) is non-fatal — we witness
		// whatever the local ref resolves to and note nothing, matching the
		// conservative "staleness only makes the check more conservative" stance.
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
	// The reap ALWAYS runs — clean tree, red command, or a panic in cmdRun. It
	// archives a dirtied worktree first (losing nothing), then force-removes.
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
		// The command could not be STARTED (binary missing) — distinct from a red
		// exit. Surface it as an error so the caller does not read a non-start as a
		// clean red. The worktree still reaps via the deferred call.
		return res, fmt.Errorf("worktreewitness: run %q: %w", res.Command, runErr)
	}
	return res, nil
}

// reap archives a dirtied worktree's changes (if archiveDir is set and the tree is
// dirty), then force-removes the worktree, pruning on a remove failure so no stale
// .git/worktrees registration lingers (a Windows-locked binary can hold the dir).
// It returns (archivedPath, note): archivedPath is "" for a clean/undisabled
// archive; note records a non-fatal cleanup wrinkle. Best-effort throughout — the
// caller's temp parent is removed regardless.
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
		// Prune the registration so a stale entry does not linger; the temp parent
		// is removed by the caller's defer regardless.
		_, _, _ = gitRun(top, "git", "-C", top, "worktree", "prune")
		if note == "" {
			note = "remove needed prune fallback"
		}
	}
	return archived, note
}

// worktreeDirty reports whether the worktree has any tracked change OR untracked
// file — anything a reap would otherwise discard. A failed status reads as dirty
// (fail-safe: archive rather than risk silently dropping work).
func worktreeDirty(wt string, gitRun Runner) bool {
	out, code, err := gitRun(wt, "git", "-C", wt, "status", "--porcelain")
	if err != nil || code != 0 {
		return true
	}
	return strings.TrimSpace(out) != ""
}

// archiveWorktree saves the worktree's tracked diff + a list of untracked files
// into archiveDir before the reap force-removes it, so a dirtied witness loses
// nothing. It writes two files under a short-SHA-named subdir: diff.patch (tracked
// changes) and untracked.txt (the porcelain untracked list). Returns the subdir.
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

// resolveRef returns the full SHA a ref points at, via `git rev-parse`.
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

// repoTopAndRel resolves the git repo root and the module's path relative to it,
// so a worktree of the repo root can still find a module that lives in a subdir.
// For this repo (go.mod at the root) moduleRel is "." — but resolving it keeps the
// helper correct for a repo whose module is nested.
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
	// A cross-volume / unrelatable path (or an escaping "..") means we cannot place
	// the module under top — default to the worktree root (moduleRel "."), which is
	// correct for a module that lives AT the repo root (this repo) and safe for any
	// other: the command runs in the worktree top rather than a guessed subdir.
	if rerr != nil || rel == "" || rel == "." || strings.HasPrefix(rel, "..") {
		rel = "."
	}
	return top, filepath.ToSlash(rel), nil
}

// splitRef splits a remote-tracking ref like "origin/main" into (remote, refspec)
// for a fetch: ("origin", "main"). A ref with no slash fetches all of origin (the
// refspec is empty), and a local ref name still fetches origin/<name>.
func splitRef(ref string) (remote, refspec string) {
	if i := strings.IndexByte(ref, '/'); i > 0 {
		return ref[:i], ref[i+1:]
	}
	return "origin", ref
}
