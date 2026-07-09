package metrics

import (
	"sort"
	"strings"
)

// Release-train compatibility checks (issue #1658, gen/second-next).
//
// Generation-aware work still has to preserve release cadence and support
// windows: a long-horizon architecture bet rides the SAME release train as a
// gen/now bug fix, because every generation lives on one trunk. The question a
// release train must answer for each candidate change is therefore not "which
// horizon is this?" but "may these bytes ship on this train without breaking a
// shipped consumer or silently changing default behavior?"
//
// This file is the SHAPE of that gate: a pure, stdlib-only compatibility model
// over the four surfaces the issue names — experimental flags, migration notes,
// API stability, and release note placement. It is architectural exploration
// behind no default exposure; nothing here is wired into a live command, a
// release helper, or a runtime feature gate. A future agent promotes it by
// folding a real candidate (from the staged diff + the gen/* label + VERSION /
// docs/releases placement) into ReleaseTrainCandidate and mounting
// CheckReleaseTrain on the release-decide path (tools/release_decide.py's Go
// successor) so a laundered horizon or an in-place schema edit is refused
// before the tag is cut, rather than after.
//
// The four rules, each grounded in a contract this repo already keeps:
//
//   - EXPERIMENTAL FLAG — a candidate outside gen/now may not be default-exposed;
//     it names the explicit gate that keeps it off by default. ("Allowed risk =
//     architectural exploration, never default exposure.")
//   - MIGRATION NOTE — a candidate that supersedes a shipped surface (adds the
//     additive successor that replaces an older shipped one) carries a migration
//     note, so a support window has a documented path off the old surface.
//   - API STABILITY — a candidate never edits a shipped surface in place. The
//     frozen wire ABI (internal/abi) is additive-only and a versioned JSONL
//     schema never rewrites a shipped /N. See docs/generation-abi-compatibility-policy.md.
//   - RELEASE NOTE PLACEMENT — the note lands in the section its horizon folds
//     onto (see PublicSection). Filing a non-shipped horizon under "shipped" is
//     horizon laundering. See docs/generation-public-narrative.md.
//
// Generation stays orthogonal to three things, and this gate is built to keep
// them separate:
//
//   - PRIORITY — no rule reads a priority, and none penalizes a far horizon. A
//     properly gated, properly filed gen/future candidate PASSES every check,
//     exactly as a gen/now one does; gen/future is a horizon label, not a value
//     judgment. Only exposure and surface facts can block.
//   - SHARED TRUNK — there is one train and one trunk. A candidate is never
//     asked which branch or worktree it came from; the model has no such field.
//   - RUNTIME FEATURE GATES — DefaultExposed and ExperimentalFlag are INPUTS
//     that the caller reads off the runtime gate. The gate is never INFERRED
//     from the gen/* label: the label says WHEN an option is expected to mature,
//     the flag says WHETHER it is exposed. That is why a default-exposed
//     gen/second-next candidate blocks (a real exposure fact) while a gated
//     gen/future one passes (also a real exposure fact).
//
// Fail-closed: a candidate whose generation is outside the closed horizon
// vocabulary cannot be decided cheaply, so every check blocks rather than
// waving it onto the train — the same INDETERMINATE discipline the kernel's
// verification ladder uses.
//
// Promotion / demotion / invalidating-assumption for this artifact itself:
//   - Promotion evidence: CheckReleaseTrain runs on a real staged candidate in
//     the release-decide path and refuses at least one true positive (a laundered
//     release note, an in-place schema edit, or a default-exposed non-now change)
//     that would otherwise have shipped.
//   - Demotion/retirement evidence: retire this shape if the release train stops
//     being generation-aware (one horizon per train, so placement is trivial), or
//     if the additive-only ABI promise in docs/generation-abi-compatibility-policy.md
//     is dropped — at that point APIStability and MigrationNote have no contract
//     to enforce and the remaining two checks fold into the public-narrative lint.
//   - Invalidating assumption: that "supersedes" and "breaks in place" are
//     DECIDABLE from a candidate's diff without running its consumers. If a
//     shipped surface can be broken by an additive-looking change (a new enum
//     value a strict consumer rejects, a new JSONL field that shifts a fixed
//     column), then BreaksShippedSurface is not a boolean the differ can compute
//     and this gate reports a false pass. The first counterexample of that shape
//     invalidates the four-check decomposition, not just one rule.

// CompatCheck is the closed set of compatibility checks a release-train
// candidate must clear. A value outside this set is a bug, not a new check.
type CompatCheck string

const (
	// CheckExperimentalFlag verifies that a candidate outside gen/now is not
	// default-exposed and names the explicit gate holding it off by default.
	CheckExperimentalFlag CompatCheck = "experimental-flag"
	// CheckMigrationNote verifies that a candidate superseding a shipped surface
	// documents the path off it, so a support window is not silently closed.
	CheckMigrationNote CompatCheck = "migration-note"
	// CheckAPIStability verifies that no shipped surface is edited in place —
	// the additive-only promise that lets concurrent generations share one trunk.
	CheckAPIStability CompatCheck = "api-stability"
	// CheckReleaseNotePlacement verifies the note lands in the section the
	// candidate's horizon folds onto, refusing horizon laundering.
	CheckReleaseNotePlacement CompatCheck = "release-note-placement"
)

// CompatChecks is the ordered, closed check vocabulary. A new check is added
// here (and given a rule in CheckReleaseTrain), never inside the run loop.
var CompatChecks = []CompatCheck{
	CheckExperimentalFlag,
	CheckMigrationNote,
	CheckAPIStability,
	CheckReleaseNotePlacement,
}

// Label is the short human label for a check, for a render line or a Slack card.
func (c CompatCheck) Label() string {
	switch c {
	case CheckExperimentalFlag:
		return "Experimental flag"
	case CheckMigrationNote:
		return "Migration note"
	case CheckAPIStability:
		return "API stability"
	case CheckReleaseNotePlacement:
		return "Release note placement"
	default:
		return string(c)
	}
}

// CompatVerdict is the closed set of per-check outcomes.
type CompatVerdict string

const (
	// CompatPass means the check applied and the candidate satisfied it.
	CompatPass CompatVerdict = "pass"
	// CompatBlock means the candidate may not ride this train until it is fixed.
	CompatBlock CompatVerdict = "block"
	// CompatNotApplicable means the check does not bind this candidate (e.g. a
	// migration note for a change that supersedes nothing). It is not a pass —
	// it records that the rule had nothing to say.
	CompatNotApplicable CompatVerdict = "not-applicable"
)

// PublicSection folds a horizon onto the release-note section it may be filed
// under — the three public words (shipped / next / research) that the public
// narrative collapses the four internal streams into. An unknown horizon folds
// to "" so the caller fails closed rather than guessing a section.
func PublicSection(generation string) string {
	switch generation {
	case "now":
		return "shipped"
	case "next":
		return "next"
	case "second-next", "future":
		return "research"
	default:
		return ""
	}
}

// ReleaseTrainCandidate is one change proposed for the current release train.
// It is a plain data snapshot: a caller folds the staged diff, the gen/* label,
// and the runtime gate into it. This package reads no disk, no clock, and no
// git — and deliberately has no branch, worktree, or priority field, because a
// compatibility verdict may not depend on any of them.
type ReleaseTrainCandidate struct {
	// Change identifies the candidate for the render (an issue ref or subject).
	Change string `json:"change"`
	// Generation is the horizon key (one of RoadmapGenerations).
	Generation string `json:"generation"`
	// DefaultExposed reports whether the change is ON by default at runtime.
	// This is read off the feature gate, never inferred from Generation.
	DefaultExposed bool `json:"default_exposed"`
	// ExperimentalFlag names the explicit gate holding the change off by
	// default. Empty means no gate was named.
	ExperimentalFlag string `json:"experimental_flag"`
	// BreaksShippedSurface reports an IN-PLACE edit of a shipped wire ABI or a
	// shipped /N schema — the additive-only violation.
	BreaksShippedSurface bool `json:"breaks_shipped_surface"`
	// SupersedesShipped reports that the change lands an additive successor
	// intended to replace an older shipped surface, which starts a support
	// window and therefore requires a migration note.
	SupersedesShipped bool `json:"supersedes_shipped"`
	// MigrationNote is the path off the superseded surface. Empty means absent.
	MigrationNote string `json:"migration_note"`
	// ReleaseNoteSection is the section the note was filed under; compared
	// against PublicSection(Generation). Empty means unplaced.
	ReleaseNoteSection string `json:"release_note_section"`
}

// CompatFinding is one check's verdict on one candidate, with the reason a
// human (or the agent that must fix it) needs to act.
type CompatFinding struct {
	Check   CompatCheck   `json:"check"`
	Verdict CompatVerdict `json:"verdict"`
	Reason  string        `json:"reason"`
}

// CompatReport is the full gate result for one candidate: one finding per
// check, in CompatChecks order.
type CompatReport struct {
	Candidate ReleaseTrainCandidate `json:"candidate"`
	Findings  []CompatFinding       `json:"findings"`
}

// Blocked reports whether any check refused the candidate. A report with no
// blocks may ride the train; one with any block may not.
func (r CompatReport) Blocked() bool {
	for _, f := range r.Findings {
		if f.Verdict == CompatBlock {
			return true
		}
	}
	return false
}

// Blocks returns the refusing findings, in CompatChecks order, so a caller can
// report every reason at once instead of surfacing them one train at a time.
func (r CompatReport) Blocks() []CompatFinding {
	var out []CompatFinding
	for _, f := range r.Findings {
		if f.Verdict == CompatBlock {
			out = append(out, f)
		}
	}
	return out
}

// knownGeneration reports whether g is in the closed horizon vocabulary.
func knownGeneration(g string) bool {
	for _, s := range RoadmapGenerations {
		if s == g {
			return true
		}
	}
	return false
}

// CheckReleaseTrain runs every check in CompatChecks against the candidate and
// returns one finding per check, in order. It is pure and total: every check
// always yields a verdict, and an undecidable candidate blocks rather than
// passing.
func CheckReleaseTrain(c ReleaseTrainCandidate) CompatReport {
	report := CompatReport{Candidate: c, Findings: make([]CompatFinding, 0, len(CompatChecks))}

	// Fail closed: an unknown horizon makes every rule undecidable. Do not guess
	// a section, and do not assume the exposure default of a horizon we do not know.
	if !knownGeneration(c.Generation) {
		reason := "unknown generation " + quote(c.Generation) + "; expected one of " +
			strings.Join(RoadmapGenerations, ", ") + " — fail closed, candidate may not ride the train"
		for _, check := range CompatChecks {
			report.Findings = append(report.Findings, CompatFinding{Check: check, Verdict: CompatBlock, Reason: reason})
		}
		return report
	}

	for _, check := range CompatChecks {
		report.Findings = append(report.Findings, evaluate(check, c))
	}
	return report
}

// evaluate applies one check's rule. Split out so each rule reads as a rule.
func evaluate(check CompatCheck, c ReleaseTrainCandidate) CompatFinding {
	find := func(v CompatVerdict, reason string) CompatFinding {
		return CompatFinding{Check: check, Verdict: v, Reason: reason}
	}

	switch check {
	case CheckExperimentalFlag:
		// gen/now is the shipped horizon: default exposure is its normal state.
		if c.Generation == "now" {
			return find(CompatNotApplicable, "gen/now rides the train at default exposure; no experimental gate required")
		}
		if c.DefaultExposed {
			return find(CompatBlock, "gen/"+c.Generation+" is default-exposed; a horizon past now ships behind an explicit gate, never as default behavior")
		}
		if c.ExperimentalFlag == "" {
			return find(CompatBlock, "gen/"+c.Generation+" is not default-exposed but names no experimental flag; name the gate that holds it off")
		}
		return find(CompatPass, "held off by default behind "+quote(c.ExperimentalFlag))

	case CheckMigrationNote:
		if !c.SupersedesShipped {
			return find(CompatNotApplicable, "supersedes no shipped surface; no support window opens")
		}
		if c.MigrationNote == "" {
			return find(CompatBlock, "supersedes a shipped surface with no migration note; a support window may not close silently")
		}
		return find(CompatPass, "migration path documented: "+quote(c.MigrationNote))

	case CheckAPIStability:
		if c.BreaksShippedSurface {
			return find(CompatBlock, "edits a shipped surface in place; the wire ABI and shipped /N schemas are additive-only")
		}
		return find(CompatPass, "additive only; no shipped surface edited in place")

	case CheckReleaseNotePlacement:
		want := PublicSection(c.Generation)
		if c.ReleaseNoteSection == "" {
			return find(CompatBlock, "no release note placed; every train candidate files a note under "+quote(want))
		}
		if c.ReleaseNoteSection != want {
			return find(CompatBlock, "release note filed under "+quote(c.ReleaseNoteSection)+" but gen/"+c.Generation+" folds onto "+quote(want)+"; filing a non-shipped horizon as shipped is horizon laundering")
		}
		return find(CompatPass, "filed under "+quote(want)+", matching gen/"+c.Generation)

	default:
		// A check with no rule is a bug in this file, not a passing candidate.
		return find(CompatBlock, "no rule declared for check "+quote(string(check))+"; fail closed")
	}
}

// Render produces a deterministic text report: the orthogonality header, the
// candidate, then one line per check. Pure (no clock, no disk) so a test can
// assert its exact bytes and an operator surface can mount it.
func (r CompatReport) Render() string {
	var b strings.Builder
	b.WriteString("Release-train compatibility: ")
	if r.Candidate.Change == "" {
		b.WriteString("(unnamed candidate)")
	} else {
		b.WriteString(r.Candidate.Change)
	}
	b.WriteString(" [gen/")
	if r.Candidate.Generation == "" {
		b.WriteString("?")
	} else {
		b.WriteString(r.Candidate.Generation)
	}
	b.WriteString("]\n")
	b.WriteString(OrthogonalityNote)
	b.WriteString("\n\n")

	for _, f := range r.Findings {
		b.WriteString(pad(f.Check.Label(), roadmapLabelWidth+3))
		b.WriteString(" | ")
		b.WriteString(pad(string(f.Verdict), 15))
		b.WriteString(" | ")
		b.WriteString(f.Reason)
		b.WriteString("\n")
	}

	b.WriteString("\nverdict: ")
	if r.Blocked() {
		b.WriteString("BLOCKED (")
		names := make([]string, 0, len(r.Findings))
		for _, f := range r.Blocks() {
			names = append(names, string(f.Check))
		}
		sort.Strings(names)
		b.WriteString(strings.Join(names, ", "))
		b.WriteString(")\n")
	} else {
		b.WriteString("MAY RIDE\n")
	}
	return b.String()
}

// quote wraps s in double quotes for a reason string, rendering an empty value
// visibly rather than as a gap the reader has to notice.
func quote(s string) string {
	if s == "" {
		return `""`
	}
	return `"` + s + `"`
}
