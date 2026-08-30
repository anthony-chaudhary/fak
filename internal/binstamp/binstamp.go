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

	"github.com/anthony-chaudhary/fak/internal/appversion"
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
	stamp := stampFrom(bi)
	if !stamp.HasVCS {
		if rev := strings.TrimSpace(appversion.BuildCommit); rev != "" {
			stamp = Stamp{Revision: rev, HasVCS: true}
		}
	}
	return stamp
}

// stampFrom extracts the stamp from a (possibly nil) BuildInfo. Split out so tests can
// drive the extraction with a synthetic BuildInfo.
// FromBuildInfo converts Go build metadata read from any binary into a Stamp.
// It lets recovery tools inspect a stale target without executing it.
func FromBuildInfo(bi *debug.BuildInfo) Stamp { return stampFrom(bi) }

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
//
// It is the verdict-only face of Explain — callers that must decide "restart or not" want
// exactly the three-state Freshness and nothing else. Explain carries the same verdict plus
// the Cause, for human/diagnostic surfaces that need to say WHY.
func Compare(running Stamp, headRev string) Freshness {
	f, _ := Explain(running, headRev)
	return f
}

// Cause explains WHY Explain reached its Freshness — most importantly, which of the three
// distinct conditions that ALL collapse to Unknown in Compare actually applied. Compare's
// three-state verdict is right for a restart decision but wrong for a human: "Unknown" reads
// as a benign shrug, yet one of its causes (Unstamped) is a real defect — a binary that
// cannot attest its own commit can NEVER be checked for staleness. A diagnostic uses this to
// warn loudly on that case while staying quiet on the benign ones (a dev's Dirty build, or
// running outside a repo with NoHead).
type Cause int

const (
	// CauseMatched — Fresh: the running rev equals HEAD.
	CauseMatched Cause = iota
	// CauseDiverged — Stale: a clean, embedded rev that differs from HEAD.
	CauseDiverged
	// CauseUnstamped — Unknown: no VCS revision is embedded, so the binary cannot attest
	// which commit it was built from. Staleness is UNVERIFIABLE, not "fine" — this is the
	// load-bearing case a diagnostic must surface rather than swallow.
	CauseUnstamped
	// CauseDirty — Unknown: built from a tree with uncommitted changes, so the embedded rev
	// is a base commit that does not describe the binary's actual contents (a dev build).
	CauseDirty
	// CauseNoHead — Unknown: no HEAD to compare against (not a git repo, or HEAD unreadable).
	CauseNoHead
)

func (c Cause) String() string {
	switch c {
	case CauseMatched:
		return "matched"
	case CauseDiverged:
		return "diverged"
	case CauseUnstamped:
		return "unstamped"
	case CauseDirty:
		return "dirty"
	case CauseNoHead:
		return "no-head"
	default:
		return "unknown"
	}
}

// Explain returns the same Freshness as Compare, plus the Cause that produced it. The branch
// order encodes a priority: an unstamped binary dominates (its Unknown is a defect worth
// flagging even when there is also no HEAD to compare against), then a missing HEAD, then a
// dirty build, then the fresh/stale comparison. Freshness is byte-for-byte what Compare
// returns for every input — Compare is defined in terms of this function.
func Explain(running Stamp, headRev string) (Freshness, Cause) {
	headRev = strings.TrimSpace(headRev)
	if !running.HasVCS || running.Revision == "" {
		return Unknown, CauseUnstamped
	}
	if headRev == "" {
		return Unknown, CauseNoHead
	}
	if running.Dirty {
		// A dirty binary's rev is its base commit, but its actual content is unknown — we
		// cannot honestly call it stale-vs-HEAD, and must never restart it out from under
		// a developer. Treat as Unknown.
		return Unknown, CauseDirty
	}
	if revisionsMatch(running.Revision, headRev) {
		return Fresh, CauseMatched
	}
	return Stale, CauseDiverged
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
