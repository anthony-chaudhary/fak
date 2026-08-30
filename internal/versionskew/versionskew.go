// Package versionskew turns "I can't tell which fak is running" into a STRUCTURED,
// REFUSABLE version-skew verdict — the binary-provenance R2 witness (#3351, epic #2218 G2).
//
// internal/binstamp already answers the coarse restart question by EQUALITY: it compares the
// running binary's embedded VCS revision to a single HEAD and returns Fresh / Stale / Unknown.
// That three-state verdict is right for "restart or not", but it collapses the whole R2 drift
// condition — a running binary that is behind, ahead, dirty, or unstamped — into one word:
// Unknown. Unknown reads like a benign shrug, so NO gate can refuse a mixed-version wave on it.
// The exact condition R2 warns about (a running binary whose commit is dirty / ahead / absent)
// produces the LEAST-informative answer.
//
// versionskew reclassifies that condition using git ANCESTRY (`git merge-base --is-ancestor`)
// into a CLOSED set of tokens where the load-bearing cases are each their OWN verdict:
//
//   - a binary provably BEHIND the trunk tip is Skewed (a refusable token a gate can act on),
//     not a shrug;
//   - a binary that carries no VCS stamp is Unstamped — a DISTINCT refusable token, never a
//     silent Unknown that reads like success (the key fix);
//   - a dirty build is Dirty, an ahead build is Ahead, an off-trunk build is Diverged.
//
// Unknown survives only as the honest, NARROW residual: the running commit could not be located
// against the trunk at all (no trunk ref, or the commit is absent from the local repo). A gate
// must not refuse on a fact it could not establish.
//
// Classify is the PURE kernel (stamp + tip + a pre-computed ancestry Relation -> Verdict); it
// does NO I/O and is fully table-testable. Assess is the thin impure shell that reads the running
// stamp (binstamp.Self), resolves the trunk tip, and probes ancestry via git. The split mirrors
// binstamp/Compare and releasestale/Compute+Gather. It performs NO build and NO install: pure
// observation. The build/verify/swap path (`fak self-update`) consults a verdict like this to
// decide whether a swap is warranted; wiring the gate is the caller's job.
package versionskew

import (
	"context"
	"os/exec"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/binstamp"
)

// Verdict is the closed set of version-skew classifications. Every input maps to exactly one
// token; there is no open "other" and — crucially — the un-attestable and behind cases do NOT
// fall through to Unknown.
type Verdict int

const (
	// Unknown is the ONE residual: the running commit could not be located against the trunk
	// (no trunk ref supplied, or the commit is absent from the local repo, so ancestry is
	// uncomputable). It is deliberately NARROW — the R2 cases that used to land here
	// (behind / ahead / dirty / unstamped) each have their own token now. Not refusable: a gate
	// must not refuse on a fact it could not establish.
	Unknown Verdict = iota
	// Fresh: the running commit IS the trunk tip. Current; nothing to act on.
	Fresh
	// Skewed: the running commit is a STRICT ANCESTOR of the trunk tip — provably BEHIND. This is
	// the load-bearing refusable token: a gate can refuse admitting a behind binary into a wave
	// that must be converged on the tip.
	Skewed
	// Ahead: the trunk tip is a strict ancestor of the running commit — the binary is NEWER than
	// the tip (a fresh local / CI build not yet pushed). Distinct from Unknown so it never reads
	// as "can't tell", but NOT refusable-as-stale: ahead is not behind.
	Ahead
	// Diverged: neither commit is an ancestor of the other — the running binary is off the trunk
	// line entirely. Refusable: a fleet meant to converge on trunk should not admit it.
	Diverged
	// Unstamped: the running binary carries NO VCS revision, so it cannot attest which commit it
	// was built from. Staleness is UNVERIFIABLE — the key fix: this is its OWN refusable token,
	// NOT a silent Unknown that reads like success.
	Unstamped
	// Dirty: the binary was built from a tree with uncommitted changes, so its embedded rev is a
	// base commit that does not describe its actual contents. Un-attestable like Unstamped, and
	// likewise refusable rather than a benign shrug.
	Dirty
)

// String renders the verdict as its stable UPPERCASE token — the form a gate or a diagnostic
// surface prints and matches on.
func (v Verdict) String() string {
	switch v {
	case Fresh:
		return "FRESH"
	case Skewed:
		return "SKEWED"
	case Ahead:
		return "AHEAD"
	case Diverged:
		return "DIVERGED"
	case Unstamped:
		return "UNSTAMPED"
	case Dirty:
		return "DIRTY"
	default:
		return "UNKNOWN"
	}
}

// Refusable reports whether a gate should REFUSE work on this verdict — the property that makes
// the token actionable, and the whole point of not collapsing to Unknown. Skewed (behind),
// Diverged (off-trunk), Unstamped and Dirty (un-attestable) all block; Fresh and Ahead are
// acceptable, and Unknown is never a refusal (you cannot refuse on a fact you could not
// establish).
func (v Verdict) Refusable() bool {
	switch v {
	case Skewed, Diverged, Unstamped, Dirty:
		return true
	default:
		return false
	}
}

// Relation is the git-ancestry relationship between the running commit and the trunk tip, as
// established by two `git merge-base --is-ancestor` probes. It is the ONE impure input Classify
// needs; passing it in (rather than shelling out inside Classify) keeps the classifier a
// deterministic, table-testable pure function.
type Relation int

const (
	// RelUndetermined: ancestry could not be computed — a commit does not resolve in the local
	// repo (never fetched) or git was unavailable. Maps to Unknown.
	RelUndetermined Relation = iota
	// RelEqual: the running commit and the trunk tip are the same object.
	RelEqual
	// RelBehind: the running commit is a STRICT ancestor of the trunk tip (behind).
	RelBehind
	// RelAhead: the trunk tip is a STRICT ancestor of the running commit (ahead).
	RelAhead
	// RelDiverged: neither commit is an ancestor of the other.
	RelDiverged
)

// Classify is the pure kernel: it maps a running-binary stamp, the trunk tip revision, and a
// pre-computed ancestry Relation to exactly one Verdict. It performs NO I/O — the git probing
// that produces rel is the caller's job (Assess does it) — so it is fully deterministic and
// table-testable.
//
// Priority, highest first, chosen so the most defect-like condition wins and the un-attestable
// cases can never be masked by a coincidental ancestry answer:
//  1. Unstamped — no rev at all: nothing downstream is meaningful.
//  2. Dirty — a rev exists but does not describe the binary's contents.
//  3. no trunk tip, or undetermined ancestry — Unknown (the honest residual).
//  4. the ancestry relation — Fresh / Skewed / Ahead / Diverged.
func Classify(running binstamp.Stamp, trunkTip string, rel Relation) Verdict {
	if !running.HasVCS || strings.TrimSpace(running.Revision) == "" {
		return Unstamped
	}
	if running.Dirty {
		return Dirty
	}
	if strings.TrimSpace(trunkTip) == "" {
		return Unknown
	}
	switch rel {
	case RelEqual:
		return Fresh
	case RelBehind:
		return Skewed
	case RelAhead:
		return Ahead
	case RelDiverged:
		return Diverged
	default:
		return Unknown
	}
}

// Assessment bundles a Verdict with the evidence that produced it, for a diagnostic surface (a
// banner row, `self-update --check`, a gate log). The verdict alone tells a gate what to do; the
// evidence tells a human WHY.
type Assessment struct {
	Verdict  Verdict  // the closed-set classification
	Running  string   // the running binary's embedded rev ("" if unstamped)
	Dirty    bool     // the running binary was built from a dirty tree
	TrunkTip string   // the trunk tip rev compared against ("" if it could not be resolved)
	Relation Relation // the git-ancestry relationship that decided a stamped/clean verdict
}

// Runner runs a command in dir and returns (combined output, ok=exit-zero). It matches the shape
// of selfinstall.RealRunner so cmd/fak can pass that same runner in; a test can pass a fake.
type Runner func(ctx context.Context, dir, name string, args ...string) (string, bool)

// RealRunner is the default git-executing Runner: it runs name+args in dir and reports the
// combined output plus whether the process exited zero. Mirrors selfinstall.RealRunner so the
// two are interchangeable at the call site.
func RealRunner(ctx context.Context, dir, name string, args ...string) (string, bool) {
	cmd := exec.CommandContext(ctx, name, args...)
	if dir != "" {
		cmd.Dir = dir
	}
	out, err := cmd.CombinedOutput()
	return string(out), err == nil
}

// Assess reads the RUNNING process's stamp (binstamp.Self), resolves the trunk tip via the git
// runner + ref, probes ancestry, and returns the full Assessment. It is the one impure entry
// point; Classify is the pure kernel it delegates the verdict to.
func Assess(ctx context.Context, run Runner, dir, trunkRef string) Assessment {
	return AssessStamp(ctx, run, dir, trunkRef, binstamp.Self())
}

// AssessStamp is Assess with the running stamp injected, so a caller that already read a stamp —
// or a test driving a synthetic one against a temp repo — can exercise the full path without
// touching the live process. When the stamp is un-attestable (unstamped or dirty), it skips the
// git calls entirely: the verdict is decided by the stamp alone, and there is nothing a trunk tip
// could change.
func AssessStamp(ctx context.Context, run Runner, dir, trunkRef string, running binstamp.Stamp) Assessment {
	a := Assessment{Running: strings.TrimSpace(running.Revision), Dirty: running.Dirty, Relation: RelUndetermined}
	if running.HasVCS && strings.TrimSpace(running.Revision) != "" && !running.Dirty {
		a.TrunkTip = resolveRev(ctx, run, dir, trunkRef)
		if a.TrunkTip != "" {
			if strings.EqualFold(strings.TrimSpace(running.Revision), strings.TrimSpace(a.TrunkTip)) {
				a.Relation = RelEqual
			} else {
				a.Relation = ancestryOf(ctx, run, dir, running.Revision, a.TrunkTip)
			}
		}
	}
	a.Verdict = Classify(running, a.TrunkTip, a.Relation)
	return a
}

// resolveRev resolves a ref (a branch like origin/main, or a raw SHA) to a full commit SHA, or
// "" if it does not resolve. The ^{commit} peel + --verify --quiet means a tag or a missing ref
// fails cleanly instead of printing an error.
func resolveRev(ctx context.Context, run Runner, dir, ref string) string {
	ref = strings.TrimSpace(ref)
	if ref == "" {
		return ""
	}
	out, ok := run(ctx, dir, "git", "rev-parse", "--verify", "--quiet", ref+"^{commit}")
	if !ok {
		return ""
	}
	return firstLine(out)
}

// ancestryOf classifies the git-ancestry relationship between running and tip. Both commits must
// resolve locally first: an absent commit (never fetched) yields RelUndetermined -> Unknown,
// rather than being mistaken for a genuine Diverged. With both present, two --is-ancestor probes
// decide the relation; equality is the both-directions-ancestor case, so a short-vs-full SHA pair
// still classifies as RelEqual.
func ancestryOf(ctx context.Context, run Runner, dir, running, tip string) Relation {
	if !revExists(ctx, run, dir, running) || !revExists(ctx, run, dir, tip) {
		return RelUndetermined
	}
	behind := isAncestor(ctx, run, dir, running, tip) // running is an ancestor of (or equal to) tip
	ahead := isAncestor(ctx, run, dir, tip, running)  // tip is an ancestor of (or equal to) running
	switch {
	case behind && ahead:
		return RelEqual
	case behind:
		return RelBehind
	case ahead:
		return RelAhead
	default:
		return RelDiverged
	}
}

// revExists reports whether rev resolves to a commit object in the repo at dir.
func revExists(ctx context.Context, run Runner, dir, rev string) bool {
	_, ok := run(ctx, dir, "git", "rev-parse", "--verify", "--quiet", strings.TrimSpace(rev)+"^{commit}")
	return ok
}

// isAncestor reports whether a is an ancestor of (or equal to) b — `git merge-base --is-ancestor
// a b` exits zero. Both a and b are assumed to already resolve (revExists gated the caller), so a
// false here means "not an ancestor", not "unknown commit".
func isAncestor(ctx context.Context, run Runner, dir, a, b string) bool {
	_, ok := run(ctx, dir, "git", "merge-base", "--is-ancestor", strings.TrimSpace(a), strings.TrimSpace(b))
	return ok
}

// firstLine trims and returns the first non-empty line of git output (rev-parse prints the SHA on
// its own line, but a stray trailing newline or CRLF must not leak into the rev string).
func firstLine(s string) string {
	for _, line := range strings.Split(s, "\n") {
		if t := strings.TrimSpace(line); t != "" {
			return t
		}
	}
	return ""
}
