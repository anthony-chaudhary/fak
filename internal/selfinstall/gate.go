package selfinstall

// gate.go — the spawn gate's REFUSAL when the binary adjudicating a tick is not the one
// executing it (#6508, Done conditions 3 and 5).
//
// The dispatch preflight already MEASURED this and it changed nothing. Its `fak_bin` block
// reported `FAK_BIN_DISAGREEMENT` — the repo-root binary deciding admission was an
// `+uncommitted` working-tree compile on a different revision than the `tools/.bin` copy
// fronting every worker it admitted — and the tick still answered SPAWN_OK. A warning on an
// admission path that admits anyway is not a control; it is a log line. Worse, the property
// it fails to protect is the one everything else rests on: an unreviewed adjudicator decides
// who may run, so every downstream verdict inherits code no commit reviews.
//
// So this is the fold that turns that measurement into a verdict. It is pure — measured
// provenance in, (refuse, reason) out — and deliberately lives beside the role census rather
// than inside the preflight, because the SAME question ("do the deployed copies agree?") is
// asked by `self-update --check` over a live census and by the tick over the provenance the
// Python gate already collected. One rule, two callers, one test.
//
// Three states refuse and everything else fails OPEN:
//
//	dirty        the adjudicator is a working-tree compile: no commit reviews it
//	unpinned     the adjudicator ran but printed no VCS stamp: it cannot say what it is
//	disagreement the adjudicator and the worker guard are provably different builds
//
// Fail-open is not timidity, it is the same rule the resolvers already follow: a host with no
// fak built resolves nothing and launches the worker unwrapped, so an UNRESOLVED or
// unmeasured binary must not be able to wedge a fleet that never had the binary in the first
// place. Only a POSITIVE measurement of skew refuses.
//
// Allow is the operator escape hatch, and it exists for a concrete reason: in a shared
// maintainer checkout the repo-root gate binary is routinely a hand-build, and a refusal with
// no override would freeze the fleet exactly when someone is mid-test on it. An override is
// recorded in the reason it returns, so a tick admitted that way is never later mistaken for
// one that passed clean.

import "strings"

// RefuseBinSkew is the verdict a spawn preflight returns when the binary adjudicating the
// tick is unreviewable or disagrees with the binary the workers will run. It is a refusal
// like REFUSE_AT_CAP / REFUSE_GATE: the sweep stops on it.
const RefuseBinSkew = "REFUSE_BIN_SKEW"

// GateProvenance is the MEASURED identity of the two binaries that matter to an admission
// decision: the one adjudicating the gate and the one that will front the admitted worker.
//
// It carries measurement outcomes, not guesses. Resolved says a path was found at all;
// Attested says that path produced a usable VCS stamp. The two are separate because they fail
// open in opposite directions: an unresolved binary is a host that never had one (fail open),
// while a resolved binary that will not say which commit it is, is precisely the unreviewable
// adjudicator this refuses.
//
// The build ids compared are real VCS revisions — never a size/mtime "build key", which
// differs between two byte-identical copies and would make every host look skewed.
type GateProvenance struct {
	// Probed is false when no provenance was collected at all (an older payload, a
	// measurement that did not run). The zero value therefore never refuses.
	Probed bool

	GatePath     string // the adjudicating binary's path, for the reason line
	GateResolved bool   // a gate binary was found
	GateAttested bool   // it produced a usable VCS revision
	GateBuild    string // that revision
	GateDirty    bool   // it self-reports an `+uncommitted` working-tree build

	WorkerPath     string // the binary that will front the admitted worker
	WorkerResolved bool
	WorkerAttested bool
	WorkerBuild    string

	// Allow records an explicit operator override for THIS tick (normally an env knob read
	// by the impure shell). It converts the refusal into an admitted-but-annotated pass.
	Allow bool
}

// ProvenanceFromCensus reads the two admission-relevant roles out of a role census, so a
// caller holding a live census (`self-update --check`) asks the identical question the
// dispatch tick asks of the provenance its Python gate already measured.
func ProvenanceFromCensus(copies []HotCopy) GateProvenance {
	p := GateProvenance{Probed: true}
	for _, c := range copies {
		switch c.Role {
		case RoleGate:
			p.GatePath, p.GateResolved, p.GateAttested = c.Path, c.Present, c.Attested
			p.GateBuild, p.GateDirty = c.Build, c.Dirty
		case RoleWorker:
			p.WorkerPath, p.WorkerResolved, p.WorkerAttested = c.Path, c.Present, c.Attested
			p.WorkerBuild = c.Build
		}
	}
	return p
}

// GateSkewRefusal reports whether the spawn gate must refuse this tick, and why.
//
// refuse=true means the verdict becomes RefuseBinSkew. refuse=false with a non-empty reason
// is the OVERRIDDEN case: the fleet is admitted, but the skew is named so the admission
// carries its own caveat instead of looking clean.
//
// Precedence among the three refusing states is fixed (dirty, then unpinned, then
// disagreement) so the same host always produces the same reason — a replayable verdict, not
// one that depends on which check happened to run first.
func GateSkewRefusal(p GateProvenance) (bool, string) {
	why := gateSkew(p)
	if why == "" {
		return false, ""
	}
	if p.Allow {
		return false, "bin-skew ADMITTED BY OVERRIDE — " + why
	}
	return true, RefuseBinSkew + ": " + why
}

// gateSkew names the skew, or "" when there is none to name. Every early return here is a
// fail-open case: something was not measured, so nothing is claimed.
func gateSkew(p GateProvenance) string {
	if !p.Probed || !p.GateResolved {
		return "" // no measurement / no gate binary on this host: fail open, as resolution does
	}
	gate := orElse(p.GatePath, "the adjudicating binary")
	if p.GateDirty {
		return "the binary adjudicating this spawn gate (" + gate + " " + shortRev(p.GateBuild) +
			" +uncommitted) is a working-tree build that no commit reviews; refusing to let unreviewed code decide who may run"
	}
	if !p.GateAttested {
		return "the binary adjudicating this spawn gate (" + gate +
			") reports no VCS stamp, so it cannot say which commit it is; refusing on an unpinned adjudicator"
	}
	if !p.WorkerResolved || !p.WorkerAttested {
		return "" // nothing to disagree WITH: a host with no worker binary launches unwrapped
	}
	if !sameRev(p.GateBuild, p.WorkerBuild) {
		return "the binary adjudicating this spawn gate (" + gate + " " + shortRev(p.GateBuild) +
			") is a different build than the one fronting the workers it would admit (" +
			orElse(p.WorkerPath, "the worker guard") + " " + shortRev(p.WorkerBuild) +
			"); converge the hot copies (`fak self-update --check`) before admitting more load"
	}
	return ""
}

// ApplyGateSkew folds bin skew into an ALREADY-COMPUTED spawn verdict, and is the shape the
// dispatch tick consumes: (verdict, reason) in, (verdict, reason) out.
//
// okVerdict is the caller's admit token (dispatchtick.PreflightOKVerdict, "SPAWN_OK"), passed
// rather than restated here so this package cannot drift from the vocabulary the tick
// actually branches on. The fold is a no-op on any verdict that is NOT that token: a
// preflight which already refused for a higher-precedence reason (at-cap, no-seat, host) keeps
// its verdict and its reason, because the fleet is then already not growing and overwriting
// the operator's cause with this one would lose the binding term.
//
// An overridden skew leaves the admit verdict intact but ANNOTATES the reason, so the tick
// that was let through on an unreviewed adjudicator says so in its own payload.
func ApplyGateSkew(verdict, reason, okVerdict string, p GateProvenance) (string, string) {
	if verdict != okVerdict || okVerdict == "" {
		return verdict, reason
	}
	refuse, why := GateSkewRefusal(p)
	if why == "" {
		return verdict, reason
	}
	if refuse {
		return RefuseBinSkew, why
	}
	return verdict, strings.TrimSpace(strings.TrimSpace(reason) + " (" + why + ")")
}

// SkewSummary is the one-line operator rendering of a refusal, used where a payload wants the
// cause without the full audit. Empty when the gate is clean.
func SkewSummary(p GateProvenance) string {
	if why := gateSkew(p); why != "" {
		return strings.TrimSpace(why)
	}
	return ""
}
