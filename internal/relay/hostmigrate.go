// Issue #4144: the cross-host / cross-model WARM-resume seam for the baton's optional
// session-image pointer (ProgressCursor.SessionImage, baton.go).
//
// The gap it closes. A relay leg already hands its successor a pointer-only baton — a git
// anchor, a ledger ref, a wip_tree checkpoint — so the next leg re-derives progress instead
// of trusting a recap. But those anchors resume the successor COLD on a fresh host: it
// re-reads git and the ledger from zero. internal/sessionimage already makes a whole session
// a portable, integrity-checked VALUE (drive + recall core image with quarantine seals +
// trajectory) that survives an offload to another host, user, or model. This file is the
// missing seam between the two: it lets a baton NAME such an image, and lets the successor
// resolve that pointer to a warm resume — or fail closed to the existing cold path when the
// image is gone, truncated, or tampered.
//
// It rides the C3 resolver contract (resolve.go), not a parallel one. SessionImageResolver
// implements Resolver and returns the SAME closed ResolveVerdict (verified/dangling/unknown)
// the commit resolver does, with the SAME fail-closed distinction: an unreachable store is
// ResolveUnknown, never ResolveDangling, so a transient offload-store outage is never
// mistaken for a missing image. The store probe is injected exactly like CommitResolver's
// `exists`, so the resolver logic is unit-testable without a live bundle. The production
// wiring that loads and integrity-verifies a real bundle via sessionimage.LoadDir
// (sessionimage.RelayResumeProbe / sessionimage.RelayResolver) lives in internal/sessionimage,
// so relay — a foundation-tier package — imports no integrator and the tier DAG holds; a
// tier-4 integrator (sessionimage) may depend DOWN on this port, never the reverse. The
// load-bearing property the whole seam rests on: "verified" means the image loaded AND its
// sha256 part-index checked out — a corrupt image can never resolve verified, so it can never
// drive a warm resume.
//
// The warm/cold fork is minimal but REAL. ResolveResumeMode folds a baton's image pointer
// through a resolver into a two-state resume decision: ResumeWarm only on ResolveVerified,
// ResumeCold on absent / dangling / unknown. The production warm-branch load step
// (sessionimage.LoadWarmImage) returns the integrity-verified *sessionimage.Image the
// successor rehydrates. This file stays a pure fold over an injected resolver (no I/O in
// ResolveResumeMode); the one I/O touch lives in internal/sessionimage, the wiring layer.
package relay

import (
	"strings"
)

// SessionImageResolver resolves ArtifactImage pointers (a session-image bundle handle) by
// asking an injected probe whether the image loads and its integrity index checks out. The
// probe is injected — like CommitResolver.exists — so the resolver is hermetically testable
// without a real bundle on disk; SessionImageOnDisk provides the production probe. Any
// non-image kind resolves to ResolveUnknown (this resolver does not own that store).
type SessionImageResolver struct {
	// loads reports whether the image at handle loads cleanly with its sha256 part-index
	// verified. A (true, nil) is a whole, integrity-checked image (-> verified); a
	// (false, nil) is a reachable store whose image is gone or does not verify as a whole
	// (-> dangling); a non-nil error is an unreachable store (-> unknown, fail closed).
	loads func(handle string) (bool, error)
}

// NewSessionImageResolver builds a SessionImageResolver over an injected image-load probe.
func NewSessionImageResolver(loads func(handle string) (bool, error)) SessionImageResolver {
	return SessionImageResolver{loads: loads}
}

// Resolve reports whether a's session-image handle resolves to a whole, integrity-verified
// image. It classifies, mirroring CommitResolver.Resolve:
//   - a non-image kind        -> ResolveUnknown (this resolver does not own that store);
//   - an empty ref            -> ResolveDangling (an image pointer with no handle points nowhere);
//   - a probe error           -> ResolveUnknown (store unreachable, fail closed);
//   - image gone / unverified -> ResolveDangling;
//   - image loads + verifies  -> ResolveVerified.
func (r SessionImageResolver) Resolve(a Artifact) Resolution {
	if a.Kind != string(ArtifactImage) {
		return Resolution{Artifact: a, Verdict: ResolveUnknown, Detail: "session-image resolver does not own kind " + a.Kind}
	}
	if strings.TrimSpace(a.Ref) == "" {
		return Resolution{Artifact: a, Verdict: ResolveDangling, Detail: "session-image pointer has an empty handle"}
	}
	ok, err := r.loads(a.Ref)
	if err != nil {
		return Resolution{Artifact: a, Verdict: ResolveUnknown, Detail: "session-image store unreachable: " + err.Error()}
	}
	if !ok {
		return Resolution{Artifact: a, Verdict: ResolveDangling, Detail: "no whole, integrity-verified image at handle " + a.Ref}
	}
	return Resolution{Artifact: a, Verdict: ResolveVerified, Detail: "session image loads and integrity-verifies at handle " + a.Ref}
}

// ResolveSessionImage is the convenience a successor calls to resolve the cursor's optional
// SessionImage handle directly (rather than hand-building an Artifact). An empty handle is
// ResolveDangling ("the cursor pins no image") — the caller reads that as "no warm image,
// resume cold". It defers entirely to the resolver so the verified/dangling/unknown
// distinction (and its fail-closed edges) is the resolver's, not re-implemented here.
func ResolveSessionImage(cur ProgressCursor, r Resolver) Resolution {
	return r.Resolve(Artifact{Kind: string(ArtifactImage), Ref: cur.SessionImage})
}

// ResumeMode is the closed two-state resume decision the successor makes from its baton's
// session-image pointer: resume WARM from an integrity-verified image, or COLD from the
// git/ledger anchors alone. It is deliberately binary — there is no "maybe warm" — so a
// pointer that cannot be proven whole always takes the safe, existing cold path.
type ResumeMode string

const (
	// ResumeWarm means the baton's session-image pointer resolved verified: the successor
	// may LoadWarmImage and rehydrate the drive + recall core + trajectory instead of
	// rebuilding from cold anchors.
	ResumeWarm ResumeMode = "warm"
	// ResumeCold means resume from the git/ledger anchors alone — the baton carried no image
	// pointer, or the pointer did not resolve verified (dangling or unknown). It is the
	// pre-#4144 behavior and the fail-closed default: an image that is absent, gone,
	// truncated, tampered, or on an unreachable store never yields a warm resume.
	ResumeCold ResumeMode = "cold"
)

// ResolveResumeMode folds a baton's optional SessionImage pointer through a resolver into the
// warm/cold resume decision, and returns the underlying Resolution for the audit detail. A
// baton with no image pointer short-circuits to (ResumeCold, dangling-on-empty) WITHOUT
// consulting the resolver — an old baton resumes exactly as before. Otherwise the mode is
// ResumeWarm only on ResolveVerified; ResolveDangling and ResolveUnknown both fall to
// ResumeCold, so a warm resume is taken only when the image is proven whole. Pure over the
// injected resolver — no clock, no I/O; the I/O of actually loading the warm image is the
// separate LoadWarmImage step the warm branch runs.
func ResolveResumeMode(b Baton, r Resolver) (ResumeMode, Resolution) {
	if strings.TrimSpace(b.ProgressCursor.SessionImage) == "" {
		return ResumeCold, Resolution{
			Artifact: Artifact{Kind: string(ArtifactImage), Ref: ""},
			Verdict:  ResolveDangling,
			Detail:   "baton carries no session_image pointer; resuming cold from git/ledger anchors",
		}
	}
	res := ResolveSessionImage(b.ProgressCursor, r)
	if res.Verdict == ResolveVerified {
		return ResumeWarm, res
	}
	return ResumeCold, res
}
