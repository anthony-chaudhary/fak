// amendment_delta.go classifies a policy reload's field-level delta by
// amendment class (#5177, epic #5170, Track C). The legacy reload gate saw the
// whole delta through one widening lens; DiffAmendment instead names every
// change and routes it through the PolicyKnobRegistry (amendment.go) into one
// of three buckets:
//
//	Tighten — a RATCHET-direction change (added deny / arg rule / secret
//	          pattern / block host, narrowed allow, stricter posture). Safe to
//	          auto-apply with no confirm friction: it can only narrow what is
//	          admitted.
//	Widen   — a loosening change (added allow, removed deny, loosened
//	          posture, removed arg rule / secret pattern / block host). The
//	          caller must hold its gated confirm channel open. It is ALSO where
//	          any change whose direction cannot be proven lands (#5414): a
//	          mislabelled widen silently loosens the floor, a mislabelled
//	          tighten only costs a confirm, so uncertainty resolves to Widen.
//	Frozen  — a change to a FROZEN knob, or to a field the registry does not
//	          classify at all (fail closed: an unclassified surface is treated
//	          as frozen, so a new Policy field cannot slip through the gate
//	          unlabeled). Refuse outright — no channel may confirm it.
package policy

import (
	"fmt"
	"regexp"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/adjudicator"
)

// Amendment delta classes — the closed overall verdict of DiffAmendment,
// derived from the buckets by severity (frozen > widen > tighten > none).
const (
	AmendmentNone            = "none"
	AmendmentTighten         = "tighten"
	AmendmentWiden           = "widen"
	AmendmentFrozenViolation = "frozen-violation"
)

// AmendmentChange is one classified field-level change. Field is the
// adjudicator.Policy field name (the PolicyKnobRegistry key), so a journal row
// can bind the change back to its declared amendment class. Label is the
// stable snake_case rendering key ("added_allow", "removed_deny", ...). Old is
// the removed element (empty for pure additions), New the added one (empty for
// pure removals); a transition (posture) carries both.
type AmendmentChange struct {
	Field string
	Label string
	Old   string
	New   string
}

func (c AmendmentChange) value() string {
	if c.Old != "" && c.New != "" {
		return c.Old + "->" + c.New
	}
	if c.New != "" {
		return c.New
	}
	return c.Old
}

// AmendmentDelta is the classified reload delta. The buckets are emitted in a
// deterministic field-scan order with sorted elements, so rendering is stable.
type AmendmentDelta struct {
	Tighten []AmendmentChange
	Widen   []AmendmentChange
	Frozen  []AmendmentChange
}

// Empty reports a no-op delta (byte-identical effective floor).
func (d AmendmentDelta) Empty() bool {
	return len(d.Tighten) == 0 && len(d.Widen) == 0 && len(d.Frozen) == 0
}

// Class folds the buckets into the closed overall verdict, most severe wins:
// any frozen violation makes the whole delta refusable regardless of what else
// it carries, and any widening makes it gated even when tightenings ride along.
func (d AmendmentDelta) Class() string {
	switch {
	case len(d.Frozen) > 0:
		return AmendmentFrozenViolation
	case len(d.Widen) > 0:
		return AmendmentWiden
	case len(d.Tighten) > 0:
		return AmendmentTighten
	default:
		return AmendmentNone
	}
}

// FormatAmendmentChanges renders one bucket as "label=v1,v2; label2=v3" in
// emission order — the audit-row / refusal-message form.
func FormatAmendmentChanges(changes []AmendmentChange) string {
	var labels []string
	byLabel := map[string][]string{}
	for _, c := range changes {
		if _, seen := byLabel[c.Label]; !seen {
			labels = append(labels, c.Label)
		}
		byLabel[c.Label] = append(byLabel[c.Label], c.value())
	}
	parts := make([]string, 0, len(labels))
	for _, label := range labels {
		parts = append(parts, label+"="+strings.Join(byLabel[label], ","))
	}
	return strings.Join(parts, "; ")
}

// route sends one change into the bucket its registry class dictates. A field
// the registry does not know, or one it declares FROZEN, lands in Frozen —
// fail closed, so the gate can never silently admit an unclassified knob.
func (d *AmendmentDelta) route(widened bool, c AmendmentChange) {
	knob, ok := KnobByField(c.Field)
	switch {
	case !ok || knob.Class == AmendFrozen:
		d.Frozen = append(d.Frozen, c)
	case widened:
		d.Widen = append(d.Widen, c)
	default:
		d.Tighten = append(d.Tighten, c)
	}
}

// DiffAmendment computes the classified amendment delta from the installed
// policy to the candidate one. Map/set fields are compared by element
// presence (a Deny reason relabel is not a delta); ArgPredicates compare by
// full rule identity, so an edited rule reads as removed+added and the
// removal keeps the delta gated.
func DiffAmendment(old, next adjudicator.Policy) AmendmentDelta {
	var d AmendmentDelta

	// Allow: added name widens, removed name tightens.
	for _, name := range sortedKeys(old.Allow, next.Allow, true) {
		d.route(true, AmendmentChange{Field: "Allow", Label: "added_allow", New: name})
	}
	for _, name := range sortedKeys(old.Allow, next.Allow, false) {
		d.route(false, AmendmentChange{Field: "Allow", Label: "removed_allow", Old: name})
	}

	// AllowPrefix: added prefix widens, removed prefix tightens.
	added, removed := diffStrings(old.AllowPrefix, next.AllowPrefix)
	for _, p := range added {
		d.route(true, AmendmentChange{Field: "AllowPrefix", Label: "added_allow_prefix", New: p})
	}
	for _, p := range removed {
		d.route(false, AmendmentChange{Field: "AllowPrefix", Label: "removed_allow_prefix", Old: p})
	}

	// Deny: removed entry widens, added entry tightens (the RATCHET direction).
	for _, name := range denyKeysOnlyIn(old.Deny, next.Deny) {
		d.route(true, AmendmentChange{Field: "Deny", Label: "removed_deny", Old: name})
	}
	for _, name := range denyKeysOnlyIn(next.Deny, old.Deny) {
		d.route(false, AmendmentChange{Field: "Deny", Label: "added_deny", New: name})
	}

	// SelfModifyGlobs: removed glob widens, added glob tightens.
	added, removed = diffStrings(old.SelfModifyGlobs, next.SelfModifyGlobs)
	for _, g := range removed {
		d.route(true, AmendmentChange{Field: "SelfModifyGlobs", Label: "removed_self_modify_globs", Old: g})
	}
	for _, g := range added {
		d.route(false, AmendmentChange{Field: "SelfModifyGlobs", Label: "added_self_modify_globs", New: g})
	}

	// Posture: diffing by strictness: fail_closed (2) > admit_and_log (1) > default_open (0).
	// Lowering strictness widens; raising strictness tightens.
	if old.Posture != next.Posture {
		oldRank, oldWord := postureStrictness(old.Posture)
		nextRank, nextWord := postureStrictness(next.Posture)
		widened := nextRank < oldRank
		d.route(widened, AmendmentChange{
			Field: "Posture",
			Label: "posture",
			Old:   oldWord,
			New:   nextWord,
		})
	}

	// ArgPredicates: an added rule can only turn an allow into a deny
	// (tighten); a removed rule un-constrains an argument (widen). Rules diff
	// by full identity — an edited rule reads as removed+added — but render as
	// the compact "tool:arg" the operator recognizes.
	added, removed = diffStrings(argPredicateKeys(old.ArgPredicates), argPredicateKeys(next.ArgPredicates))
	for _, r := range removed {
		d.route(true, AmendmentChange{Field: "ArgPredicates", Label: "removed_arg_rules", Old: argPredicateDisplay(r)})
	}
	for _, r := range added {
		d.route(false, AmendmentChange{Field: "ArgPredicates", Label: "added_arg_rules", New: argPredicateDisplay(r)})
	}

	// SecretPatterns: extend-only union over the canon floor — an added
	// pattern tightens, a removed one widens.
	added, removed = diffStrings(regexpStrings(old.SecretPatterns), regexpStrings(next.SecretPatterns))
	for _, p := range removed {
		d.route(true, AmendmentChange{Field: "SecretPatterns", Label: "removed_secret_patterns", Old: p})
	}
	for _, p := range added {
		d.route(false, AmendmentChange{Field: "SecretPatterns", Label: "added_secret_patterns", New: p})
	}

	// InlineEval: supplemental interpreter+flag specs only broaden write detection.
	added, removed = diffStrings(inlineEvalStrings(old.InlineEval), inlineEvalStrings(next.InlineEval))
	for _, spec := range removed {
		d.route(true, AmendmentChange{Field: "InlineEval", Label: "removed_inline_eval", Old: spec})
	}
	for _, spec := range added {
		d.route(false, AmendmentChange{Field: "InlineEval", Label: "added_inline_eval", New: spec})
	}

	// EgressBlockHosts: an added block host tightens, a removed one widens.
	added, removed = diffStrings(old.EgressBlockHosts, next.EgressBlockHosts)
	for _, h := range removed {
		d.route(true, AmendmentChange{Field: "EgressBlockHosts", Label: "removed_block_hosts", Old: h})
	}
	for _, h := range added {
		d.route(false, AmendmentChange{Field: "EgressBlockHosts", Label: "added_block_hosts", New: h})
	}

	// The eight rules above are the original analysis. #5414 supplies a
	// directional rule for every REMAINING exported field, then sweeps whatever
	// still has no rule into the gated bucket — see amendment_direction.go. The
	// order matters only for the audit text; the sweep is a backstop, so it runs last
	// and, because analyzedAmendmentFields covers everything the two rule layers
	// handle, it emits nothing until a new Policy field lands.
	diffRemainingKnobs(&d, old, next)
	residualAmendmentChanges(&d, old, next, analyzedAmendmentFields)

	return d
}

// sortedKeys returns the allow-map names present (true-valued) in exactly one
// side: next-only when addedSide, old-only otherwise. Sorted.
func sortedKeys(old, next map[string]bool, addedSide bool) []string {
	from, other := next, old
	if !addedSide {
		from, other = old, next
	}
	var out []string
	for name, on := range from {
		if on && !other[name] {
			out = append(out, name)
		}
	}
	sort.Strings(out)
	return out
}

// denyKeysOnlyIn returns the deny keys present in only (absent from other),
// sorted. Compared by key presence: relabeling a reason is not a delta.
func denyKeysOnlyIn[V any](only, other map[string]V) []string {
	var out []string
	for name := range only {
		if _, ok := other[name]; !ok {
			out = append(out, name)
		}
	}
	sort.Strings(out)
	return out
}

// diffStrings returns (added, removed) between two string sets, each sorted.
func diffStrings(old, next []string) (added, removed []string) {
	oldSet := stringMembers(old)
	nextSet := stringMembers(next)
	for _, v := range next {
		if _, ok := oldSet[v]; !ok {
			added = append(added, v)
		}
	}
	for _, v := range old {
		if _, ok := nextSet[v]; !ok {
			removed = append(removed, v)
		}
	}
	sort.Strings(added)
	sort.Strings(removed)
	return added, removed
}

func stringMembers(values []string) map[string]struct{} {
	set := make(map[string]struct{}, len(values))
	for _, v := range values {
		set[v] = struct{}{}
	}
	return set
}

// argPredicateKeys renders each rule's full identity, so an edited rule diffs
// as removed+added rather than disappearing into a same-slot mutation. The
// "tool:arg" prefix is what argPredicateDisplay recovers for rendering.
func argPredicateKeys(preds []adjudicator.ArgPredicate) []string {
	out := make([]string, 0, len(preds))
	for _, p := range preds {
		re := ""
		if p.Re != nil {
			re = p.Re.String()
		}
		out = append(out, fmt.Sprintf("%s:%s[kind=%d glob=%q re=%q n=%d reason=%v advisory=%t fix=%q]",
			p.Tool, p.Arg, p.Kind, p.Glob, re, p.N, p.Reason, p.Advisory, p.Fix))
	}
	return out
}

// argPredicateDisplay recovers the compact "tool:arg" prefix from a rule
// identity key.
func argPredicateDisplay(key string) string {
	if i := strings.Index(key, "["); i > 0 {
		return key[:i]
	}
	return key
}

func inlineEvalStrings(specs []adjudicator.InlineEvalSpec) []string {
	out := make([]string, 0, len(specs))
	for _, spec := range specs {
		flags := append([]string(nil), spec.Flags...)
		sort.Strings(flags)
		out = append(out, strings.ToLower(spec.Interp)+":"+strings.Join(flags, ","))
	}
	return out
}

func regexpStrings(patterns []*regexp.Regexp) []string {
	out := make([]string, 0, len(patterns))
	for _, p := range patterns {
		if p != nil {
			out = append(out, p.String())
		}
	}
	return out
}

func postureStrictness(p adjudicator.Posture) (int, string) {
	switch p {
	case adjudicator.PostureFailClosed:
		return 2, postureFailClosed
	case adjudicator.PostureAdmitAndLog:
		return 1, postureAdmitAndLog
	case adjudicator.PostureDefaultOpen:
		return 0, postureDefaultOpen
	default:
		return 0, fmt.Sprintf("unrecognized(%d)", uint8(p))
	}
}
