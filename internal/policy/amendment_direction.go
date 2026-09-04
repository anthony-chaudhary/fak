// amendment_direction.go closes the fail-OPEN hole DiffAmendment shipped with
// and then sharpens it (#5414, follow-on to epic #5170).
//
// DiffAmendment originally computed a direction for only EIGHT of
// adjudicator.Policy's exported fields. A reload delta that touched only the
// others came back Empty(), folded to AmendmentNone, and both admission gates
// (guard_core_lock.go, guard_self_tighten.go) admit AmendmentNone
// unconditionally — so a drop of RedactFields / EgressExtraDenyHosts /
// EgressBlockLists / LintWrites, or a widening of Complain / AdvisoryReasons /
// SecretPosture / Profile / EgressAllowHosts, rode through the STRONGEST posture
// as a declared no-op. This file supplies the missing analysis in two layers:
//
//  1. diffRemainingKnobs — a hand-written DIRECTIONAL rule per knob, so a
//     genuine tightening of an additive RATCHET knob is admitted as a tighten
//     instead of paying the widen toll, and a genuine loosening is gated.
//  2. residualAmendmentChanges — a reflection BACKSTOP over every exported field
//     that has no rule yet. It exists so a field added tomorrow cannot re-open
//     the hole while nobody is looking.
//
// THE ASYMMETRY THAT GOVERNS EVERY RULE BELOW. The two misclassifications are
// not interchangeable. Calling a WIDEN a TIGHTEN silently loosens a floor the
// operator asked to have gated — that is the dangerous direction, and it is
// unrecoverable because the gate never fires to surface it. Calling a TIGHTEN a
// WIDEN merely makes a safe change pay a confirm toll it did not owe — annoying,
// never dangerous. So every rule here classifies TIGHTEN only where the
// narrowing is PROVABLE from the knob's own semantics; any knob, or any
// individual transition, whose direction this package cannot prove falls back to
// WIDEN. Uncertainty resolves toward "gate it". Two live instances of that
// fallback ship today: a non-nil -> different-non-nil Profile edit (the elision
// bitmask is unexported, so the direction is unreadable from here) and a
// SecretPosture value outside the recognized set.
//
// A NOTE ON THE REGISTRY'S Direction FIELD, WHICH IS A DIFFERENT AXIS.
// PolicyKnob.Direction (amendment.go) says which way a knob may be moved BY ITS
// AUTHORIZED CHANNELS; the direction computed here says which way one CONCRETE
// change actually moved the floor. They can disagree without either being wrong:
// Posture is GATED_WIDEN / widen-only in the registry, yet DiffAmendment has
// always routed admit_and_log -> fail_closed into the Tighten bucket. Both
// EgressRestrict and SecretPosture have that same shape.
package policy

import (
	"fmt"
	"reflect"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
)

// analyzedAmendmentFields names every exported adjudicator.Policy field for
// which DiffAmendment computes a direction from a hand-written rule — the eight
// original rules in amendment_delta.go plus the twelve added here.
// residualAmendmentChanges sweeps everything NOT in this set into the gated
// bucket, so the set is also the honest inventory of what is precisely
// classified versus what is merely fail-closed.
var analyzedAmendmentFields = map[string]bool{
	// Hand-written in amendment_delta.go (the original eight).
	"Posture":          true,
	"Allow":            true,
	"AllowPrefix":      true,
	"Deny":             true,
	"SelfModifyGlobs":  true,
	"ArgPredicates":    true,
	"SecretPatterns":   true,
	"InlineEval":       true,
	"EgressBlockHosts": true,
	// Hand-written in diffRemainingKnobs below (#5414).
	"RedactFields":             true,
	"LintWrites":               true,
	"Profile":                  true,
	"Complain":                 true,
	"AdvisoryReasons":          true,
	"SecretPosture":            true,
	"EgressExtraDenyHosts":     true,
	"ResearchEgressAllowHosts": true,
	"EgressAllowHosts":         true,
	"EgressBlockLists":         true,
	"EgressRestrict":           true,
	"AutoRepairSidestep":       true,
	"TestLanes":                true,
	"ExemptLanes":              true,
	"DisableTestImmunity":      true,
	"Lane":                     true,
}

// diffRemainingKnobs applies the per-knob directional rules for the twelve
// fields the original DiffAmendment left unanalyzed. Each rule cites the knob
// semantics that make its tighten direction provable; where nothing proves it,
// the change is routed widen.
func diffRemainingKnobs(d *AmendmentDelta, old, next adjudicator.Policy) {
	// RedactFields — additive tighten. A key here makes TRANSFORM strip that
	// arg's value before dispatch, so adding one only ever removes bytes from
	// the call; removing one un-redacts an arg that was being scrubbed.
	d.additiveTightenSet("RedactFields", "redact_fields", old.RedactFields, next.RedactFields)

	// EgressExtraDenyHosts — additive tighten. Declared "it only ever TIGHTENS
	// the floor" at the field: the hardwired metadata class can never be
	// disabled, so these hosts are pure additions to the refusal set.
	d.additiveTightenSet("EgressExtraDenyHosts", "extra_deny_hosts",
		old.EgressExtraDenyHosts, next.EgressExtraDenyHosts)

	// EgressBlockLists — additive tighten. Subscribing to a bundled community
	// list compiles more hosts into the block set; unsubscribing releases them.
	d.additiveTightenSet("EgressBlockLists", "block_lists", old.EgressBlockLists, next.EgressBlockLists)

	// EgressAllowHosts — additive WIDEN, the mirror shape. These are adblock
	// `@@` exceptions that carve a host back open under a block rule (and under
	// EgressRestrict they are the total allowlist), so an added host reaches one
	// more destination and a removed one reaches strictly fewer.
	d.additiveWidenSet("EgressAllowHosts", "egress_allow_hosts", old.EgressAllowHosts, next.EgressAllowHosts)

	// LintWrites — off/on, TRUE is the stricter value: it arms the in-process
	// grammar rung that refuses an unparseable Go/JSON whole-file write with
	// MALFORMED before it lands.
	d.boolTightenOnTrue("LintWrites", "lint_writes", old.LintWrites, next.LintWrites)

	// EgressRestrict — off/on, TRUE is the stricter value: it inverts the egress
	// stance from "reachable unless listed" to "unreachable unless listed", and
	// an empty allowlist under restrict closes egress entirely.
	d.boolTightenOnTrue("EgressRestrict", "egress_restrict", old.EgressRestrict, next.EgressRestrict)

	// AutoRepairSidestep — off/on, TRUE is the LOOSER value: it replaces the
	// reversibility rung's preview-confirm HOLD with an in-flight TRANSFORM, so
	// turning it on removes a confirm the operator was getting.
	d.boolWidenOnTrue("AutoRepairSidestep", "auto_repair_sidestep", old.AutoRepairSidestep, next.AutoRepairSidestep)

	// Complain — a named tool has its DEFAULT_DENY downgraded to an
	// admit-and-log allow, so adding a name admits one more tool.
	for _, name := range sortedKeys(old.Complain, next.Complain, true) {
		d.route(true, AmendmentChange{Field: "Complain", Label: "added_complain", New: name})
	}
	for _, name := range sortedKeys(old.Complain, next.Complain, false) {
		d.route(false, AmendmentChange{Field: "Complain", Label: "removed_complain", Old: name})
	}

	// AdvisoryReasons — a named reason's refusal is downgraded to an
	// admit-and-log warn, so adding a reason stops a rung from denying.
	for _, code := range advisoryKeysOnlyIn(next.AdvisoryReasons, old.AdvisoryReasons) {
		d.route(true, AmendmentChange{Field: "AdvisoryReasons", Label: "added_advisory_reasons", New: abi.ReasonName(code)})
	}
	for _, code := range advisoryKeysOnlyIn(old.AdvisoryReasons, next.AdvisoryReasons) {
		d.route(false, AmendmentChange{Field: "AdvisoryReasons", Label: "removed_advisory_reasons", Old: abi.ReasonName(code)})
	}

	diffResearchAllowHosts(d, old, next)
	diffSecretVerdict(d, old, next)
	diffRungElision(d, old, next)

	d.additiveTightenSet("TestLanes", "test_lanes", old.TestLanes, next.TestLanes)
	d.additiveWidenSet("ExemptLanes", "exempt_lanes", old.ExemptLanes, next.ExemptLanes)
	d.boolWidenOnTrue("DisableTestImmunity", "disable_test_immunity", old.DisableTestImmunity, next.DisableTestImmunity)
	if old.Lane != next.Lane {
		if old.Lane == "" && next.Lane != "" {
			d.route(false, AmendmentChange{Field: "Lane", Label: "lane", New: next.Lane})
		} else if old.Lane != "" && next.Lane == "" {
			d.route(true, AmendmentChange{Field: "Lane", Label: "lane", Old: old.Lane})
		} else {
			d.route(true, AmendmentChange{Field: "Lane", Label: "lane", Old: old.Lane, New: next.Lane})
		}
	}
}

// diffResearchAllowHosts handles the one knob whose direction depends on
// whether the list is EMPTY, because emptiness is what arms it. An empty
// ResearchEgressAllowHosts means WebFetch is unconstrained by this rung; a
// non-empty one forces a strict allowlist and refuses every unlisted host. So
// switching the list on is a TIGHTEN even though it is composed entirely of
// additions, switching it off is a WIDEN even though it is composed entirely of
// removals, and while it is live on both sides it behaves like an ordinary
// allowlist (added host widens, removed host tightens).
func diffResearchAllowHosts(d *AmendmentDelta, old, next adjudicator.Policy) {
	const field, noun = "ResearchEgressAllowHosts", "research_allow_hosts"
	oldOn, nextOn := len(old.ResearchEgressAllowHosts) > 0, len(next.ResearchEgressAllowHosts) > 0
	switch {
	case !oldOn && nextOn:
		for _, h := range sortedStrings(next.ResearchEgressAllowHosts) {
			d.route(false, AmendmentChange{Field: field, Label: "research_allowlist_on", New: h})
		}
	case oldOn && !nextOn:
		for _, h := range sortedStrings(old.ResearchEgressAllowHosts) {
			d.route(true, AmendmentChange{Field: field, Label: "research_allowlist_off", Old: h})
		}
	default:
		d.additiveWidenSet(field, noun, old.ResearchEgressAllowHosts, next.ResearchEgressAllowHosts)
	}
}

// diffSecretVerdict ranks the on-discovery secret verdict by strictness and
// routes the transition by rank. An UNRECOGNIZED value on either side is not
// rankable, so the change is gated rather than guessed — the fallback this
// file's header promises.
func diffSecretVerdict(d *AmendmentDelta, old, next adjudicator.Policy) {
	if old.SecretPosture == next.SecretPosture {
		return
	}
	oldRank, oldOK := secretStrictness(old.SecretPosture)
	nextRank, nextOK := secretStrictness(next.SecretPosture)
	widened := !oldOK || !nextOK || nextRank < oldRank
	d.route(widened, AmendmentChange{
		Field: "SecretPosture",
		Label: "secret_verdict",
		Old:   secretVerdictWord(old.SecretPosture, oldOK),
		New:   secretVerdictWord(next.SecretPosture, nextOK),
	})
}

// secretStrictness ranks the recognized on-discovery secret verdicts, strictest
// first: fail_closed hard-denies, quarantine holds the bearing result out of
// the transcript and continues, admit_and_log lets a read-shaped result through
// with the would-deny recorded. The bool reports whether the value was
// recognized at all; an unrecognized value gets NO rank, because inventing one
// is exactly how a widen would get mislabelled a tighten.
func secretStrictness(p adjudicator.SecretPosture) (int, bool) {
	switch p {
	case adjudicator.SecretFailClosed:
		return 2, true
	case adjudicator.SecretQuarantine:
		return 1, true
	case adjudicator.SecretAdmitAndLog:
		return 0, true
	default:
		return 0, false
	}
}

// secretVerdictWord formats a verdict for the audit row. An unrecognized value
// prints as its raw number rather than through String(), which folds every
// unknown into "quarantine" and would make the refusal message assert a value
// the floor is not actually running.
func secretVerdictWord(p adjudicator.SecretPosture, known bool) string {
	if !known {
		return fmt.Sprintf("unrecognized(%d)", uint8(p))
	}
	return p.String()
}

// diffRungElision classifies a RungProfile change. A nil profile runs the full
// HEAD rung sequence; a non-nil one ELIDES sub-rungs per risk class, so arming
// a profile loosens and dropping it back to nil restores every rung.
//
// The third case is the honest one: two DIFFERENT non-nil profiles. The elision
// bitmask lives in an unexported RungProfile field, so this package cannot read
// which rungs moved and therefore cannot prove that the edit narrowed. It is
// routed WIDEN. That is deliberately the annoying-but-safe answer; making it
// precise needs an exported subset predicate on adjudicator.RungProfile.
func diffRungElision(d *AmendmentDelta, old, next adjudicator.Policy) {
	const field, label = "Profile", "rung_profile"
	switch {
	case old.Profile == nil && next.Profile == nil:
		return
	case old.Profile == nil:
		d.route(true, AmendmentChange{Field: field, Label: label, Old: "all-rungs", New: "elided"})
	case next.Profile == nil:
		d.route(false, AmendmentChange{Field: field, Label: label, Old: "elided", New: "all-rungs"})
	case !reflect.DeepEqual(*old.Profile, *next.Profile):
		d.route(true, AmendmentChange{Field: field, Label: label, Old: "elided", New: "elided-edited"})
	}
}

// residualAmendmentChanges is the fail-closed BACKSTOP. It reflects over every
// exported adjudicator.Policy field, skips the ones `analyzed` says already have
// a directional rule, and routes any remaining field that CHANGED into the
// widen bucket — gated, never admitted as a no-op.
//
// Widen is the only defensible default here: the sweep by construction knows
// nothing about the knob it just caught, and a knob whose direction is unknown
// must cost a confirm rather than pass silently. (A field that is not in
// PolicyKnobRegistry at all lands in Frozen instead — route() enforces that, and
// it is the stricter answer still.)
//
// `analyzed` is a parameter rather than a package-level read so a test can
// exercise the sweep on a field that does have a rule in production.
func residualAmendmentChanges(d *AmendmentDelta, old, next adjudicator.Policy, analyzed map[string]bool) {
	oldVal, nextVal := reflect.ValueOf(old), reflect.ValueOf(next)
	pt := oldVal.Type()
	for i := 0; i < pt.NumField(); i++ {
		f := pt.Field(i)
		if !f.IsExported() || analyzed[f.Name] {
			continue
		}
		a, b := oldVal.Field(i).Interface(), nextVal.Field(i).Interface()
		if reflect.DeepEqual(a, b) {
			continue
		}
		d.route(true, AmendmentChange{
			Field: f.Name,
			Label: "changed_" + snakeFieldName(f.Name),
			Old:   boundedFieldValue(a),
			New:   boundedFieldValue(b),
		})
	}
}

// ---- small shared helpers ----

// additiveTightenSet records a set-valued knob whose ADDED elements narrow what
// is admitted and whose REMOVED elements loosen it — the additive-RATCHET shape
// (deny hosts, block lists, redact keys). Removals are emitted before additions
// so the gated bucket reads first, matching the original rules' order.
func (d *AmendmentDelta) additiveTightenSet(field, noun string, old, next []string) {
	added, removed := diffStrings(old, next)
	for _, v := range removed {
		d.route(true, AmendmentChange{Field: field, Label: "removed_" + noun, Old: v})
	}
	for _, v := range added {
		d.route(false, AmendmentChange{Field: field, Label: "added_" + noun, New: v})
	}
}

// additiveWidenSet is the mirror: an ADDED element loosens (an allow exception,
// a live allowlist entry) and a REMOVED one narrows.
func (d *AmendmentDelta) additiveWidenSet(field, noun string, old, next []string) {
	added, removed := diffStrings(old, next)
	for _, v := range added {
		d.route(true, AmendmentChange{Field: field, Label: "added_" + noun, New: v})
	}
	for _, v := range removed {
		d.route(false, AmendmentChange{Field: field, Label: "removed_" + noun, Old: v})
	}
}

// boolTightenOnTrue records an off/on knob whose TRUE value is the stricter one.
func (d *AmendmentDelta) boolTightenOnTrue(field, label string, old, next bool) {
	if old == next {
		return
	}
	d.route(!next, AmendmentChange{Field: field, Label: label, Old: onOff(old), New: onOff(next)})
}

// boolWidenOnTrue records an off/on knob whose TRUE value is the LOOSER one.
func (d *AmendmentDelta) boolWidenOnTrue(field, label string, old, next bool) {
	if old == next {
		return
	}
	d.route(next, AmendmentChange{Field: field, Label: label, Old: onOff(old), New: onOff(next)})
}

func onOff(v bool) string {
	if v {
		return "on"
	}
	return "off"
}

// advisoryKeysOnlyIn returns the reason codes switched ON in `only` and not in
// `other`, sorted by code so the audit text is stable.
func advisoryKeysOnlyIn(only, other map[abi.ReasonCode]bool) []abi.ReasonCode {
	var out []abi.ReasonCode
	for code, on := range only {
		if on && !other[code] {
			out = append(out, code)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i] < out[j] })
	return out
}

// sortedStrings returns a sorted copy, leaving the caller's slice alone.
func sortedStrings(values []string) []string {
	out := append([]string(nil), values...)
	sort.Strings(out)
	return out
}

// snakeFieldName converts an exported Go field name into the snake_case label
// key the audit rows use ("EgressBlockHosts" -> "egress_block_hosts").
func snakeFieldName(name string) string {
	var b strings.Builder
	for i, r := range name {
		if i > 0 && r >= 'A' && r <= 'Z' {
			b.WriteByte('_')
		}
		if r >= 'A' && r <= 'Z' {
			r += 'a' - 'A'
		}
		b.WriteRune(r)
	}
	return b.String()
}

// boundedFieldValue formats a swept field's value for the audit row, capped so
// an unknown field of unknown size cannot blow up a refusal message.
func boundedFieldValue(v any) string {
	s := fmt.Sprintf("%v", v)
	const limit = 64
	if len(s) > limit {
		return s[:limit] + "..."
	}
	return s
}
