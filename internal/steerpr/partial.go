package steerpr

// partial.go — issue #5027 (child of epic #5015): give a forming unit an EXPECTED
// SIZE, so an operator reading the overlay can tell "this intent is finished"
// from "this intent is 1 commit into a 12-commit fanout". The band says WHETHER
// to look; the partial state says whether it is still CHEAP TO ACT. Redirecting
// an intent after its fanout has burned is not steering, it is archaeology.
//
// The denominator is DECLARED, never predicted. M comes from the bound intent's
// own graph — the spine issue plus the fanout children carrying its
// `fanout-<leaf>-` marker key — or from an explicit cohort wave size. There is
// no commit-rate extrapolation, no historical average, no estimate. When the
// graph does not yield a denominator, M is UNKNOWN and says so.
//
// The load-bearing invariant of this file: an unknown denominator NEVER renders
// complete. Silently treating M = N would make every in-flight intent look
// finished, which inverts the whole point of the signal — an operator would read
// "done" exactly when the budget to redirect is still fully unspent. Complete is
// therefore a DERIVED bit that only NewPartial can set, never a caller-supplied
// field; see the constructor and TestUnknownExpectedNeverComplete.
//
// Like curve.go, this leaf stays PURE and stdlib-only (architest tier-1): it
// never reads the issue graph itself. The caller gathers issue rows through the
// existing gh seam and hands them in, so steerpr keeps no GitHub client of its
// own and stays unit-testable without a repo or a network.

import (
	"fmt"
	"strings"
)

// Denominator sources — how M was derived, carried so an operator (and a test)
// can tell a fanout-derived denominator from an operator-declared cohort wave.
const (
	// SourceFanout: M came from the bound spine issue plus its fanout children.
	SourceFanout = "fanout"
	// SourceCohort: M came from an explicit `issue cohort` wave size.
	SourceCohort = "cohort"
)

// Expectation is a DERIVED denominator with the provenance of its derivation.
// It is only ever produced by DeriveExpected or CohortExpectation — the two
// declared sources — so a fabricated M cannot enter through a struct literal
// that some future caller assembles by guessing.
type Expectation struct {
	Total  int    `json:"total"`
	Source string `json:"source"`
}

// IntentIssue is the subset of a `gh issue list --json number,body` row the
// denominator derivation reads — the SAME shape issuefanout's marker-key dedupe
// scan consumes, so both read the issue graph through one gathering seam.
type IntentIssue struct {
	Number int    `json:"number"`
	Body   string `json:"body"`
}

// Partial is a forming unit's membership completeness: N of M expected commits
// landed. It is ORTHOGONAL to both Band and Curve — Band says "was each claim
// confirmed", Curve says "is the objective progressing", Partial says "is the
// intent all here yet".
//
// Expected is a POINTER precisely so unknown is representable and explicit: a
// nil Expected marshals to `"expected": null` (the field is never omitted), which
// a machine consumer reads as unknown rather than as a missing key it might
// default to zero. A unit with no bound intent simply has no Partial at all.
type Partial struct {
	Landed   int    `json:"landed"`
	Expected *int   `json:"expected"` // nil = not derivable; NEVER fabricated, never silently N
	Complete bool   `json:"complete"`
	Source   string `json:"source,omitempty"`
}

// NewPartial folds a landed count and a derived expectation into the partial
// state. It is the ONLY constructor, and Complete is computed here rather than
// accepted from a caller — that is the structural reason an unknown denominator
// can never render complete.
//
// A non-ok expectation (or a non-positive total, which is not a real
// denominator) yields Expected nil and Complete false. Note the asymmetry: an
// unknown M forces Complete false REGARDLESS of how many commits landed. That is
// the #5027 acceptance gate in one line.
func NewPartial(landed int, exp Expectation, ok bool) Partial {
	p := Partial{Landed: landed}
	if !ok || exp.Total <= 0 {
		return p
	}
	total := exp.Total
	p.Expected = &total
	p.Source = exp.Source
	// Landed may EXCEED the declared M (a child closed outside the fanout, a
	// spine that took two commits). That is reported honestly as complete rather
	// than clamped: raising M to meet N would fabricate a denominator, which is
	// the one thing this file must never do.
	p.Complete = landed >= total
	return p
}

// Known reports whether a denominator was derivable at all. A nil Partial is not
// known — degrade cleanly, exactly like a nil Curve.
func (p *Partial) Known() bool {
	return p != nil && p.Expected != nil
}

// Forming reports whether the unit is a PARTIAL bundle: a known denominator with
// members still outstanding. This is the state that is cheap to steer, and the
// one the renderer must show distinctly from a finished unit.
//
// An unknown denominator is NOT forming — it is unknown. Collapsing the two
// would let "we could not read the graph" masquerade as a positive claim about
// how much work remains.
func (p *Partial) Forming() bool {
	return p.Known() && !p.Complete
}

// Remaining reports how many expected members have not landed, and whether that
// number means anything. An unknown denominator returns ok=false — there is no
// honest remaining count without an M.
func (p *Partial) Remaining() (int, bool) {
	if !p.Known() {
		return 0, false
	}
	if n := *p.Expected - p.Landed; n > 0 {
		return n, true
	}
	return 0, true
}

// Annotate renders the partial state as one operator-visible line, or "" when
// there is no partial state (no bound intent → no line → no warning).
//
// The three renderings are deliberately distinct strings, because the entire
// operator value is telling them apart at a glance:
//   - forming:  "⧗ forming: 3 of 12 expected commits landed (9 outstanding)"
//   - complete: "complete: 12 of 12 expected commits landed"
//   - unknown:  "expected: unknown — ... not rendered complete"
//
// The unknown line NAMES its own ignorance rather than going quiet, so an
// operator is never left to read silence as completeness.
func (p *Partial) Annotate() string {
	if p == nil {
		return ""
	}
	if !p.Known() {
		return fmt.Sprintf("expected: unknown — %d landed, no declared denominator (open fanout children or a cohort wave would give one); not rendered complete", p.Landed)
	}
	src := ""
	if p.Source != "" {
		src = " [" + p.Source + "]"
	}
	if p.Complete {
		return fmt.Sprintf("complete: %d of %d expected commits landed%s", p.Landed, *p.Expected, src)
	}
	remaining, _ := p.Remaining()
	return fmt.Sprintf("⧗ forming: %d of %d expected commits landed (%d outstanding)%s — still cheap to steer", p.Landed, *p.Expected, remaining, src)
}

// FanoutMarker is the marker-key prefix a fanout child carries in its body, and
// the ONE place this package spells that contract. issuefanout mints the key
// (`fanout-<leaf>-<slug>`) and stamps it into every filed body; steerpr can not
// import issuefanout to ask — the overlay fold is fenced to stdlib-only so it can
// never grow a gate (architest OVERLAY_WOULD_GATE) — so the format is necessarily
// re-spelled on this side.
//
// Two independent spellings of one contract drift silently, and the drift lands
// exactly on this issue's acceptance gate. If the minted key stops matching this
// prefix, DeriveExpected still finds the spine but counts ZERO children, returns
// a confident M = 1, and a one-commit unit renders COMPLETE — the M = N inversion
// #5027 exists to prevent, arriving with no error anywhere. Exporting the prefix
// makes the consumer's expectation an assertable value, which is what lets
// TestPartialDenominatorMatchesTheRealFanoutMarkerContract tie it to the key
// issuefanout actually mints.
func FanoutMarker(leaf string) string {
	leaf = strings.TrimSpace(leaf)
	if leaf == "" {
		return ""
	}
	return "fanout-" + leaf + "-"
}

// DeriveExpected counts the declared membership of a bound intent: the spine
// issue itself plus every fanout child carrying its `fanout-<leaf>-` marker key.
// That marker is the fanout contract's own dedupe key (issuefanout.LiveBody
// stamps it into every filed body), which is what makes the denominator
// COUNTABLE rather than guessed.
//
// Derivability requires actually LOCATING the spine in the gathered graph. That
// requirement is doing real work: without it, an empty issues slice (gh absent,
// scan failed, wrong repo) would derive M = 1 and render a 1-commit forming unit
// as complete — a fabricated denominator dressed as a real one. No spine found
// means the graph was not read, which is unknown, not "one".
//
// The caller is expected to gather with `--state all` (the same bounded scan
// issuefanout.ListExistingArgs composes). M is the intent's DECLARED membership,
// which must stay stable as children close; counting only currently-open
// children would shrink M every time work landed, so "3 of 12" would march
// toward "3 of 3" and report complete while nine children were still unfiled.
func DeriveExpected(leaf, spine string, issues []IntentIssue) (Expectation, bool) {
	leaf = strings.TrimSpace(leaf)
	spineNum := strings.TrimPrefix(strings.TrimSpace(spine), "#")
	if leaf == "" || spineNum == "" || len(issues) == 0 {
		return Expectation{}, false
	}
	marker := FanoutMarker(leaf)
	spineFound := false
	children := 0
	for _, issue := range issues {
		if fmt.Sprintf("%d", issue.Number) == spineNum {
			spineFound = true
			continue
		}
		// Plain substring match, matching the fanout dedupe scan's own rule: a
		// hand-filed issue that mentions the key in prose counts the same as a
		// marker-stamped one.
		if strings.Contains(issue.Body, marker) {
			children++
		}
	}
	if !spineFound {
		return Expectation{}, false
	}
	return Expectation{Total: 1 + children, Source: SourceFanout}, true
}

// CohortExpectation is the explicit denominator: an operator-declared `issue
// cohort` wave size. A non-positive wave is not a denominator — it returns
// unknown rather than zero, so a mis-set flag degrades to honest ignorance
// instead of rendering every unit complete against M = 0.
func CohortExpectation(wave int) (Expectation, bool) {
	if wave <= 0 {
		return Expectation{}, false
	}
	return Expectation{Total: wave, Source: SourceCohort}, true
}

// WaveExpectation resolves the cohort half of the denominator: a unit grouped by
// WAVE takes its M from the `fak issue cohort` plan's own declared membership —
// the members an operator planned into that wave — never from a leaf's fanout
// marker.
//
// Routing a wave unit through DeriveExpected instead is not merely imprecise, it
// is the M = N trap the acceptance gate names. A wave unit's key is "wave:2"
// (WaveKey's colon-prefixed form, which no leaf can spell), never a leaf, so the
// marker "fanout-wave:2-" matches no child and the
// derivation collapses to M = 1 (spine + zero children) — at which point a wave
// one commit into a twelve-member plan renders COMPLETE, which is precisely the
// inversion this signal exists to prevent. TestWaveUnitNeverRendersCompleteViaLeafDerivation
// reproduces that trap before pinning the fix.
//
// M is counted through WaveIndex rather than off the raw Issues slice so the
// denominator matches the set of refs that can ACTUALLY fold into this unit:
// duplicates within a wave collapse, and a ref an earlier wave already claimed
// stays with that earlier wave under the same first-claim-wins rule the fold
// uses. Counting the raw slice would over-count M against members that can never
// land here — a fabricated denominator by a subtler route.
//
// A non-wave key, an unplanned wave, or a wave whose members carry no issue
// numbers all return unknown rather than zero: a wave nobody can count is
// honestly uncountable, and CohortExpectation already refuses a non-positive
// wave for that reason.
func WaveExpectation(key string, bindings []WaveBinding) (Expectation, bool) {
	key = strings.TrimSpace(key)
	if !IsWaveKey(key) {
		return Expectation{}, false
	}
	members := 0
	for _, wave := range WaveIndex(bindings) {
		if wave == key {
			members++
		}
	}
	return CohortExpectation(members)
}

// WithPartial returns the unit with its partial state bound, computing Landed
// from the unit's own members so the numerator can never disagree with the
// commit list an operator is reading right below it. A value method returning a
// copy, keeping the pure fold free of caller state.
func (u Unit) WithPartial(exp Expectation, ok bool) Unit {
	p := NewPartial(len(u.Commits), exp, ok)
	u.Partial = &p
	return u
}

// AttachPartials binds each unit's partial state from the expectation returned
// by lookup, in place. A unit whose lookup returns ok=false still gets a Partial
// — an EXPLICITLY unknown one, not a nil.
//
// That is the deliberate difference from AttachCurves (where no objective means
// no curve): a unit with no derivable denominator is a real, reportable state
// the operator must see, whereas a unit with no bound objective is simply not
// participating in the curve axis. Dropping to nil here would make "unknown"
// indistinguishable from "not asked", and the issue requires unknown be explicit.
//
// steerpr stays free of the join key: lookup receives the whole unit so the
// caller binds by whatever it owns (a resolved issue, the leaf).
func AttachPartials(units []Unit, lookup func(Unit) (Expectation, bool)) {
	for i := range units {
		exp, ok := lookup(units[i])
		p := NewPartial(len(units[i].Commits), exp, ok)
		units[i].Partial = &p
	}
}

// PartialUnits returns the units that are forming — a known denominator with
// members outstanding. This is the set that is still cheap to redirect, and
// naming it lets the caller render it emphatically and lets a test assert a
// forming unit is never silently folded in with the finished ones.
func PartialUnits(units []Unit) []Unit {
	var out []Unit
	for _, u := range units {
		if u.Partial.Forming() {
			out = append(out, u)
		}
	}
	return out
}

// UnknownExpectedUnits returns the units whose denominator was not derivable.
// It is the honest counterpart to PartialUnits: an operator can see exactly how
// much of the overlay is NOT carrying a steering signal, rather than mistaking
// an unmeasured unit for a finished one.
func UnknownExpectedUnits(units []Unit) []Unit {
	var out []Unit
	for _, u := range units {
		if u.Partial != nil && !u.Partial.Known() {
			out = append(out, u)
		}
	}
	return out
}
