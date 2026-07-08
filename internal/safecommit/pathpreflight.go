package safecommit

import (
	"context"
	"fmt"
	"strings"
)

// Path-preflight reasons — the closed vocabulary the pathspec classifier stamps into a
// PathPreflightReport. They extend the safecommit reason family (safecommit.go) and are the
// --json contract a worker consumes BEFORE a commit-by-path, so a raw
//
//	error: pathspec '<path>' did not match any file(s) known to git
//
// becomes a structured refusal that names the fix (issue #3228). A worker following the
// shared-trunk discipline of committing by explicit path (`git commit -- <path>`) repeatedly
// passes a path git cannot match — a not-yet-`git add`ed new file, a renamed/moved path, or a
// stale plan entry — and burns turns re-deriving what is actually tracked. This preflight is
// the mechanical guard behind that guidance (adjacent to the structured worker-prompt rules).
const (
	ReasonPathUntracked = "PATH_UNTRACKED" // in the worktree but not in the index — `git add` it
	ReasonPathUnmatched = "PATH_UNMATCHED" // matches no tracked or untracked file — a typo, rename, or stale plan path
)

// PathState is the per-pathspec classification. A pathspec is committable by explicit path
// (`git commit -- <path>`) ONLY when it is Tracked — git's pathspec matches the index, so no
// "did not match any file(s) known to git". Untracked and Unmatched both produce that raw
// error, but their fixes differ: an untracked worktree file needs a `git add`, while an
// unmatched pathspec is a wrong path (typo / renamed / stale plan) that no `git add` resolves.
type PathState string

const (
	PathTracked   PathState = "tracked"   // known to the index; committable by pathspec (OK)
	PathUntracked PathState = "untracked" // present in the worktree, not staged; needs `git add`
	PathUnmatched PathState = "unmatched" // matches no tracked or untracked file; typo/rename/stale
)

// PathClass is one classified pathspec plus the named fix for a non-OK state.
type PathClass struct {
	Path  string    `json:"path"`
	State PathState `json:"state"`
	Fix   string    `json:"fix,omitempty"`
}

// PathPreflightReport is the whole classification. OK is true only when EVERY requested
// pathspec is Tracked (committable by explicit path). When any is untracked or unmatched, OK
// is false and Reason names the FIRST blocking class so a caller has one closed token to
// branch on — an unmatched pathspec (a wrong path) outranks an untracked one (a real,
// un-added file), since it is the stronger "you passed the wrong thing" signal.
type PathPreflightReport struct {
	OK        bool        `json:"ok"`
	Dir       string      `json:"dir,omitempty"`
	Classes   []PathClass `json:"classes"`
	Untracked []string    `json:"untracked,omitempty"`
	Unmatched []string    `json:"unmatched,omitempty"`
	Reason    string      `json:"reason,omitempty"`
	Detail    string      `json:"detail,omitempty"`
}

// PreflightPaths classifies paths against the real git binary — the thin production wiring
// around ClassifyPaths, mirroring Commit's relationship to CommitWith.
func PreflightPaths(ctx context.Context, dir string, paths []string) (PathPreflightReport, error) {
	return ClassifyPaths(ctx, realRunner, dir, paths)
}

// ClassifyPaths is the testable core: each pathspec is probed against the index
// (`git ls-files --cached`) and then the untracked worktree
// (`git ls-files --others --exclude-standard`) through the injected runner, with no lock and
// no writes — a pure, read-only preflight. It shares gitgate's ONE repo-path rule
// (normalizePaths) with the executor so the preflight and the commit agree on what a path is.
//
// A pathspec staged purely as a DELETION (its blob is neither in the index nor the worktree)
// classifies as unmatched here; that is a rare shape the diagnostic does not special-case, and
// the real `fak commit` path — which stages and commits deletions by pathspec — is unaffected.
func ClassifyPaths(ctx context.Context, run Runner, dir string, paths []string) (PathPreflightReport, error) {
	norm, ok := normalizePaths(paths)
	rep := PathPreflightReport{Dir: dir}
	if !ok || len(norm) == 0 {
		rep.Reason = ReasonNoPath
		rep.Detail = "no valid repo-relative pathspec given"
		return rep, nil
	}

	// In a work tree? A read-only probe; anything else would classify against an absent index.
	if _, code, err := run(ctx, dir, "rev-parse", "--git-dir"); err != nil {
		return rep, fmt.Errorf("safecommit: git not executable: %w", err)
	} else if code != 0 {
		rep.Reason = ReasonNotARepo
		rep.Detail = "not inside a git work tree"
		return rep, nil
	}

	rep.OK = true
	for _, p := range norm {
		state, err := classifyOnePath(ctx, run, dir, p)
		if err != nil {
			return rep, err
		}
		cls := PathClass{Path: p, State: state}
		switch state {
		case PathUntracked:
			cls.Fix = fmt.Sprintf("git add -- %s   (present in the worktree but not staged)", p)
			rep.Untracked = append(rep.Untracked, p)
			rep.OK = false
		case PathUnmatched:
			cls.Fix = fmt.Sprintf("no file matches %q — check for a typo, a renamed/moved path, or a stale plan entry", p)
			rep.Unmatched = append(rep.Unmatched, p)
			rep.OK = false
		}
		rep.Classes = append(rep.Classes, cls)
	}

	if !rep.OK {
		if len(rep.Unmatched) > 0 {
			rep.Reason = ReasonPathUnmatched
		} else {
			rep.Reason = ReasonPathUntracked
		}
		rep.Detail = preflightDetail(rep)
	}
	return rep, nil
}

// classifyOnePath resolves a single pathspec: tracked (index) wins, else untracked (worktree),
// else it matches nothing git knows.
func classifyOnePath(ctx context.Context, run Runner, dir, p string) (PathState, error) {
	tracked, err := gitPathspecMatches(ctx, run, dir, "ls-files", "-z", "--cached", "--", p)
	if err != nil {
		return "", err
	}
	if tracked {
		return PathTracked, nil
	}
	untracked, err := gitPathspecMatches(ctx, run, dir, "ls-files", "-z", "--others", "--exclude-standard", "--", p)
	if err != nil {
		return "", err
	}
	if untracked {
		return PathUntracked, nil
	}
	return PathUnmatched, nil
}

// gitPathspecMatches reports whether a `git ls-files` variant listed at least one path for the
// pathspec. A NUL-delimited listing (-z) is split so a path with spaces is not miscounted; a
// non-zero exit in an already-verified repo means a malformed pathspec and is treated as "no
// match" rather than a hard error (the classifier's job is to name the mismatch, not crash).
func gitPathspecMatches(ctx context.Context, run Runner, dir string, args ...string) (bool, error) {
	out, code, err := run(ctx, dir, args...)
	if err != nil {
		return false, fmt.Errorf("safecommit: git not executable: %w", err)
	}
	if code != 0 {
		return false, nil
	}
	for _, seg := range strings.Split(out, "\x00") {
		if strings.TrimSpace(seg) != "" {
			return true, nil
		}
	}
	return false, nil
}

func preflightDetail(rep PathPreflightReport) string {
	var parts []string
	if len(rep.Unmatched) > 0 {
		parts = append(parts, fmt.Sprintf("%d pathspec(s) match nothing git knows: %s", len(rep.Unmatched), strings.Join(rep.Unmatched, ", ")))
	}
	if len(rep.Untracked) > 0 {
		parts = append(parts, fmt.Sprintf("%d untracked (need `git add`): %s", len(rep.Untracked), strings.Join(rep.Untracked, ", ")))
	}
	return strings.Join(parts, "; ")
}
