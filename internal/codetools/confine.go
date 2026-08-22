package codetools

import (
	"os"
	"path/filepath"
	"strings"
)

// confine.go — workspace confinement, the invariant every other file in this package
// depends on.
//
// THE ORDER MATTERS. Canonicalization runs BEFORE policy matching, never after. A policy
// that matched the model's raw spelling would be a policy over STRINGS: "src/../../etc/
// passwd", "src/./../..//etc/passwd", and "/etc/passwd" are three spellings of one file,
// and a protected-path rule written against the third would sail past the first two. So
// resolve() produces exactly one canonical (Abs, Rel) pair per file, and every policy
// question downstream is asked about that pair.
//
// TWO ESCAPES, TWO CHECKS. Lexical traversal ("..") is caught by filepath.Rel on the
// cleaned path. That is where internal/agent/readengine.go stops, on the argument that a
// read-only engine's worst case is reading a symlinked file already inside the tree. That
// argument does not hold, and this package does not inherit it: a symlink planted inside
// the workspace — by a prior tool call, by a checked-out repo, by a dependency — is a
// lexically-innocent path whose real target is anywhere on the host. Following one is how
// a READ tool reads ~/.aws/credentials while every ".." check passes. So evalWithin
// resolves symlinks on the longest EXISTING ancestor of the target and re-confines the
// result. Confinement is a property of where the bytes really are, not of how the path
// was spelled.
//
// WHY THE LONGEST EXISTING ANCESTOR rather than the target itself: EvalSymlinks fails
// outright on a path that does not exist, and a Read of a missing file must refuse with
// NOT_FOUND rather than with an unverifiable-path error. Resolving the deepest ancestor
// that DOES exist is both sound and complete for our purpose: the non-existent remainder
// is a suffix of already-cleaned plain names (resolve() rejected every ".." before this
// runs), so it cannot climb out of whatever the ancestor resolved to. It also keeps the
// check correct for the mutating tools that land on this same seam in #6704.

// Resolved is one canonical, confined filesystem path: Abs is the cleaned absolute path,
// Rel is its slash-separated path relative to the workspace root. Rel is what policy
// globs match, so policy is written in workspace terms and is portable across checkouts.
type Resolved struct {
	Abs string
	Rel string
}

type mutationTarget struct {
	Resolved
	Key string
}

// resolve canonicalizes a caller-supplied path argument and confines it to the toolset's
// workspace root, returning a Refusal for anything that escapes. A relative argument is
// taken against the root (the workspace IS the agent's cwd), an absolute one is accepted
// only if it already lands inside.
func (t *Toolset) resolve(arg string) (Resolved, *Refusal) {
	arg = strings.TrimSpace(arg)
	if arg == "" {
		return Resolved{}, refuse(CodeMalformed, "empty path")
	}
	if strings.ContainsRune(arg, 0) {
		// A NUL byte truncates the path at the syscall boundary while the Go-side
		// checks see the whole string — the classic poisoned-null-byte split between
		// what is validated and what is opened. Refuse rather than sanitize.
		return Resolved{}, refuse(CodeMalformed, "path contains NUL")
	}
	abs := arg
	if !filepath.IsAbs(abs) {
		abs = filepath.Join(t.root, abs)
	}
	abs = filepath.Clean(abs)
	rel, err := filepath.Rel(t.root, abs)
	if err != nil || rel == ".." || hasDotDotPrefix(rel) {
		return Resolved{}, refuse(CodePathEscape, "path escapes the workspace root: "+arg)
	}
	if err := t.evalWithin(abs); err != nil {
		return Resolved{}, err
	}
	return Resolved{Abs: abs, Rel: filepath.ToSlash(rel)}, nil
}

// resolveMutation returns the canonical path a mutation must actually publish to. A
// second call inside the target lock detects an alias or ancestor that moved after the
// caller first resolved it; final symlinks are refused instead of replaced or followed.
func (t *Toolset) resolveMutation(arg string) (mutationTarget, *Refusal) {
	res, r := t.resolve(arg)
	if r != nil {
		return mutationTarget{}, r
	}
	if info, err := os.Lstat(res.Abs); err == nil {
		if info.Mode()&os.ModeSymlink != 0 {
			return mutationTarget{}, refuse(CodeSymlinkEscape, "mutation refuses a symlink target")
		}
	} else if !os.IsNotExist(err) {
		return mutationTarget{}, refuse(CodeIO, err.Error())
	}

	ancestor := deepestExisting(res.Abs)
	if ancestor == "" {
		return mutationTarget{}, refuse(CodeSymlinkEscape, "cannot canonicalize mutation target")
	}
	canonical, err := filepath.EvalSymlinks(ancestor)
	if err != nil || !within(t.evalRoot, canonical) {
		return mutationTarget{}, refuse(CodeSymlinkEscape, "cannot canonicalize mutation target")
	}
	suffix, err := filepath.Rel(ancestor, res.Abs)
	if err != nil || suffix == ".." || hasDotDotPrefix(suffix) {
		return mutationTarget{}, refuse(CodePathEscape, "mutation target escapes its canonical ancestor")
	}
	abs := filepath.Clean(filepath.Join(canonical, suffix))
	if !within(t.evalRoot, abs) {
		return mutationTarget{}, refuse(CodeSymlinkEscape, "canonical mutation target escapes the workspace root")
	}
	return mutationTarget{Resolved: Resolved{Abs: abs, Rel: res.Rel}, Key: abs}, nil
}

// evalWithin denies a path whose REAL location — after resolving every symlink on the
// deepest existing ancestor — falls outside the workspace. It is the second half of
// confinement: resolve() has already established that the path is lexically inside, this
// establishes that the filesystem agrees.
//
// A path whose ancestor cannot be resolved at all (a permission fault mid-walk) is
// refused: an unverifiable path is treated as an escaping one, because the alternative is
// to admit it on the strength of a check that did not run.
func (t *Toolset) evalWithin(abs string) *Refusal {
	anc := deepestExisting(abs)
	if anc == "" {
		return nil // nothing on this path exists yet; the lexical check is the whole story
	}
	canon, err := filepath.EvalSymlinks(anc)
	if err != nil {
		return refuse(CodeSymlinkEscape, "cannot canonicalize path")
	}
	if !within(t.evalRoot, canon) {
		return refuse(CodeSymlinkEscape, "symlinked path escapes the workspace root")
	}
	return nil
}

// deepestExisting returns the deepest ancestor of abs (abs itself included) that exists
// on disk, or "" when nothing on the chain does. os.Lstat, not os.Stat: a DANGLING
// symlink still exists as a link and must be resolved and confined, and Stat would
// report it absent — the exact case where skipping the check would be wrong.
func deepestExisting(abs string) string {
	p := abs
	for {
		if _, err := os.Lstat(p); err == nil {
			return p
		}
		parent := filepath.Dir(p)
		if parent == p {
			return ""
		}
		p = parent
	}
}

// within reports whether p is base or lies under it, comparing CLEANED paths through
// filepath.Rel so the check is separator- and "."-segment-insensitive on every OS.
func within(base, p string) bool {
	rel, err := filepath.Rel(base, p)
	if err != nil {
		return false
	}
	return rel == "." || (rel != ".." && !hasDotDotPrefix(rel))
}

// hasDotDotPrefix reports whether a filepath.Rel result begins with a parent-dir segment
// ("../" or "..\"), i.e. the target escapes the base. A bare ".." is handled by callers.
func hasDotDotPrefix(rel string) bool {
	return len(rel) >= 3 && rel[0] == '.' && rel[1] == '.' && (rel[2] == '/' || rel[2] == '\\')
}

// A protected-subtree (.git, .dos) MUTATION floor deliberately does not live here: this
// slice ships no mutating tool, and a guard with nothing to guard reads as coverage it
// does not have. It lands with the Write/Edit engines in #6704. Searches are already kept
// out of those trees by walk()'s dot-directory skip (search.go), which is a context-window
// concern rather than a safety one.
