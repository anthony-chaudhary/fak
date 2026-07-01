// Rung C3 (issue #1872): the artifact-pointer resolver seam. A baton carries pointers,
// never bytes (baton.go), so the successor must be able to RE-READ each pointer against
// its durable store and learn whether it still resolves — the pointer-only carryover is
// only trustworthy if a dangling pointer is detectable. This file gives the closed
// verdict vocabulary, the per-store Resolver interface, and the first concrete store: a
// git-backed resolver for commit pointers. Other kinds (issue/memory/ledger/file) get
// their stores in later rungs; here they resolve to ResolveUnknown rather than a false
// positive.
package relay

import (
	"os/exec"
	"strings"
)

// ResolveVerdict is the closed outcome of resolving one artifact pointer against its
// durable store. It never states trust in the baton — it reports what the store says.
type ResolveVerdict string

const (
	// ResolveVerified means the store was reachable and the pointer resolves in it.
	ResolveVerified ResolveVerdict = "verified"
	// ResolveDangling means the store was reachable but the pointer does not resolve —
	// the durable object it names is gone or never existed. This is the failure the
	// pointer-only design exists to make detectable.
	ResolveDangling ResolveVerdict = "dangling"
	// ResolveUnknown means no verdict could be reached: the store was unreachable, or the
	// pointer's kind has no resolver yet. It is deliberately distinct from ResolveDangling
	// so an unreachable store is never mistaken for a missing object (fail closed, not a
	// false negative).
	ResolveUnknown ResolveVerdict = "unknown"
)

// Resolution is the typed result of resolving one Artifact: the pointer, the verdict, and
// a short human-readable detail (never consumed as progress).
type Resolution struct {
	Artifact Artifact       `json:"artifact"`
	Verdict  ResolveVerdict `json:"verdict"`
	Detail   string         `json:"detail"`
}

// Resolver re-reads the durable store an Artifact points at and reports whether the
// pointer still resolves. Implementations are per-store; a resolver returns
// ResolveUnknown for kinds it does not own rather than guessing.
type Resolver interface {
	Resolve(a Artifact) Resolution
}

// CommitResolver resolves ArtifactCommit pointers by asking git whether a ref names a
// real commit object. The store probe is injected (a function that reports whether a ref
// exists as a commit) so the resolver is unit-testable without a live repo; GitCommitExists
// provides the production wiring. Any non-commit kind resolves to ResolveUnknown.
type CommitResolver struct {
	// exists reports whether ref names a commit object in the store. A (false, nil) is a
	// clean "not found" (-> dangling); a non-nil error is an unreachable store (-> unknown).
	exists func(ref string) (bool, error)
}

// NewCommitResolver builds a CommitResolver over an injected commit-existence probe.
func NewCommitResolver(exists func(ref string) (bool, error)) CommitResolver {
	return CommitResolver{exists: exists}
}

// Resolve reports whether a's commit ref resolves in git. It classifies:
//   - a non-commit kind  -> ResolveUnknown (this resolver does not own that store);
//   - an empty ref       -> ResolveDangling (a commit pointer with no ref points nowhere);
//   - a probe error      -> ResolveUnknown (store unreachable, fail closed);
//   - ref not a commit   -> ResolveDangling;
//   - ref is a commit    -> ResolveVerified.
func (r CommitResolver) Resolve(a Artifact) Resolution {
	if a.Kind != string(ArtifactCommit) {
		return Resolution{Artifact: a, Verdict: ResolveUnknown, Detail: "commit resolver does not own kind " + a.Kind}
	}
	if a.Ref == "" {
		return Resolution{Artifact: a, Verdict: ResolveDangling, Detail: "commit pointer has an empty ref"}
	}
	ok, err := r.exists(a.Ref)
	if err != nil {
		return Resolution{Artifact: a, Verdict: ResolveUnknown, Detail: "git probe failed: " + err.Error()}
	}
	if !ok {
		return Resolution{Artifact: a, Verdict: ResolveDangling, Detail: "no commit object for ref " + a.Ref}
	}
	return Resolution{Artifact: a, Verdict: ResolveVerified, Detail: "commit object resolves for ref " + a.Ref}
}

// GitCommitExists returns a commit-existence probe backed by the git repo rooted at dir.
// It shells to `git -C <dir> cat-file -t <ref>` and reports whether the object type is
// exactly "commit". A non-zero git exit (an unknown ref) is a clean not-found — (false,
// nil), not an error — so a dangling pointer maps to ResolveDangling; only a failure to
// run git at all (e.g. git not installed) surfaces as a real error (-> ResolveUnknown).
func GitCommitExists(dir string) func(ref string) (bool, error) {
	return func(ref string) (bool, error) {
		out, err := exec.Command("git", "-C", dir, "cat-file", "-t", ref).Output()
		if err != nil {
			if _, ok := err.(*exec.ExitError); ok {
				return false, nil // git ran; the ref is not a known object
			}
			return false, err // git could not be run — store unreachable
		}
		return strings.TrimSpace(string(out)) == "commit", nil
	}
}
