// Package binstamp answers one question durably: "is the fak binary I am running built
// from the commit that is currently on the trunk, or is it stale?"
//
// It is the detection primitive behind keeping an always-on guard fleet converged on the
// latest verified fak. The Go toolchain embeds the VCS revision + dirty flag into a binary
// at build time (debug.ReadBuildInfo, the same data `fak version` prints); binstamp reads
// that stamp out of the RUNNING process and compares it to the repo HEAD. The comparison is
// deliberately conservative — it reports "stale" only when it can prove the running rev
// differs from a known HEAD; any ambiguity (no stamp, unreadable HEAD, dirty build) yields
// Unknown, never a false "stale" that could trigger an unwanted restart.
//
// It performs NO build and NO install: it is pure observation. The build/verify/swap path
// (which must be GATED on a green tree — never install a binary from a tree that does not
// compile or whose smoke test fails) lives elsewhere and consults this package to decide
// whether a swap is even warranted.
package binstamp

import (
	"runtime/debug"
	"strings"
)

// Stamp is the build provenance read out of a binary (or the running process).
type Stamp struct {
	Revision string // full VCS revision the binary was built from ("" if unstamped)
	Dirty    bool   // built from a tree with uncommitted changes
	HasVCS   bool   // a vcs.revision setting was present at all
}

// Freshness is the verdict of comparing a running stamp to a repo HEAD.
type Freshness int

const (
	// Unknown: cannot prove fresh OR stale — missing stamp, unreadable HEAD, or a dirty
	// build (whose rev is not a clean commit to compare). Callers must NOT restart on this.
	Unknown Freshness = iota
	// Fresh: the running rev equals HEAD — the binary is current.
	Fresh
	// Stale: the running rev is a clean commit that differs from HEAD — a newer fak exists.
	Stale
)

func (f Freshness) String() string {
	switch f {
	case Fresh:
		return "fresh"
	case Stale:
		return "stale"
	default:
		return "unknown"
	}
}

// Self reads the build stamp embedded in the currently-running process.
func Self() Stamp {
	bi, _ := debug.ReadBuildInfo()
	return stampFrom(bi)
}

// stampFrom extracts the stamp from a (possibly nil) BuildInfo. Split out so tests can
// drive the extraction with a synthetic BuildInfo.
func stampFrom(bi *debug.BuildInfo) Stamp {
	if bi == nil {
		return Stamp{}
	}
	var s Stamp
	for _, kv := range bi.Settings {
		switch kv.Key {
		case "vcs.revision":
			s.Revision = kv.Value
			s.HasVCS = true
		case "vcs.modified":
			s.Dirty = kv.Value == "true"
		}
	}
	return s
}

// Compare returns the freshness of a running stamp against a repo HEAD revision. headRev is
// the full SHA of the trunk tip (e.g. from `git rev-parse HEAD`). The rules are strict:
//   - no embedded revision, no HEAD, or a DIRTY build => Unknown (never restart on doubt);
//   - revisions equal (by prefix-safe match) => Fresh;
//   - both present, clean, and different => Stale.
func Compare(running Stamp, headRev string) Freshness {
	headRev = strings.TrimSpace(headRev)
	if !running.HasVCS || running.Revision == "" || headRev == "" {
		return Unknown
	}
	if running.Dirty {
		// A dirty binary's rev is its base commit, but its actual content is unknown — we
		// cannot honestly call it stale-vs-HEAD, and must never restart it out from under
		// a developer. Treat as Unknown.
		return Unknown
	}
	if revisionsMatch(running.Revision, headRev) {
		return Fresh
	}
	return Stale
}

// revisionsMatch compares two VCS revisions tolerant of differing lengths (one may be a
// short SHA): equal if either is a prefix of the other and the shorter is >= 7 chars.
func revisionsMatch(a, b string) bool {
	a, b = strings.ToLower(strings.TrimSpace(a)), strings.ToLower(strings.TrimSpace(b))
	if a == b {
		return true
	}
	short, long := a, b
	if len(short) > len(long) {
		short, long = long, short
	}
	if len(short) < 7 {
		return false
	}
	return strings.HasPrefix(long, short)
}
