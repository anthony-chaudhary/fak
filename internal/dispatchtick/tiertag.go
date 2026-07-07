package dispatchtick

import (
	"regexp"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// ---------------------------------------------------------------------------
// C4 ISSUE TIER TAGGING (#3041) — turn an issue's namespaced tier labels into the
// typed IssueTier the C5 chooser consumes, so a real per-issue tier signal finally
// flows into RouteAccountForTier instead of every issue defaulting to the
// conservative frontier floor.
// ---------------------------------------------------------------------------
//
// The label grammar is issuecontract's: tier/T<N>-required|optimal, the
// "Priority/P1 is not model tier T1" disambiguation, and the model_tier_<role>_*
// flag vocabulary. This node MIRRORS that grammar BY NAME rather than importing
// issuecontract, keeping dispatchtick's modelroute+stdlib import surface (the same
// by-name mirror tierstatus.go uses for issuecost.Bucket). issuecontract stays the
// review-layer readout — including its body-field fallback; this is the typed
// dispatch bridge, and it reads LABELS only.
//
// SAFE DEGRADE is the load-bearing property. HasTier is set true ONLY when both
// roles resolve to a single valid tier AND optimal is at least as demanding as the
// required floor. Every other case — missing, invalid, conflicting, or
// contradictory tags — returns HasTier=false so resolve() applies the conservative
// frontier floor, together with the closed-vocab flags naming exactly why. An
// untagged or ambiguous issue is never silently handed to a weak worker.

// Closed-vocabulary tag flags. These MIRROR issuecontract's model_tier_<role>_*
// flag names by value (not import) so the two layers agree by name. A status
// surface renders them verbatim.
const (
	TagFlagRequiredMissing  = "model_tier_required_missing"
	TagFlagOptimalMissing   = "model_tier_optimal_missing"
	TagFlagRequiredInvalid  = "model_tier_required_invalid"
	TagFlagOptimalInvalid   = "model_tier_optimal_invalid"
	TagFlagRequiredConflict = "model_tier_required_conflict"
	TagFlagOptimalConflict  = "model_tier_optimal_conflict"
	// TagFlagContradiction: both tiers parsed, but optimal is WEAKER than the
	// required floor — the tags contradict, so neither is trusted.
	TagFlagContradiction = "model_tier_contradiction"
)

// tierLabelRE matches a namespaced model-tier label such as tier/T1-required or
// tier/T0-optimal (applied after lower-casing). Group 1 is the T<N> token, group 2
// the role. A priority label like priority/P1 does NOT match — the "Priority/P1 is
// not model tier T1" disambiguation, matching issuecontract's modelTierLabelRE.
var tierLabelRE = regexp.MustCompile(`^tier/(t[0-9]+)-(required|optimal)$`)

// IssueTierFromLabels builds an IssueTier from an issue's GitHub labels by parsing
// the namespaced tier/T<N>-required|optimal grammar. It returns the typed tier plus
// a closed-vocabulary flag list: an empty flag list with HasTier=true means the
// tags are present, valid, and self-consistent; any flag means the issue stays
// conservative (HasTier=false, resolved to the frontier floor by IssueTier.resolve).
func IssueTierFromLabels(labels []string) (IssueTier, []string) {
	req, reqOK, reqFlags := resolveRoleTier(labels, "required",
		TagFlagRequiredMissing, TagFlagRequiredInvalid, TagFlagRequiredConflict)
	opt, optOK, optFlags := resolveRoleTier(labels, "optimal",
		TagFlagOptimalMissing, TagFlagOptimalInvalid, TagFlagOptimalConflict)

	flags := append(append([]string(nil), reqFlags...), optFlags...)

	// Either role unresolved -> conservative, carrying the flags that name why.
	if !reqOK || !optOK {
		return IssueTier{}, flags
	}

	// Both parsed. Optimal must be at least as demanding as the required floor —
	// checked through the safe comparator (optimal <= required numerically), never a
	// raw `<`. A weaker optimal contradicts the floor and is not trusted.
	if !opt.MeetsRequirement(req) {
		return IssueTier{}, append(flags, TagFlagContradiction)
	}

	return IssueTier{Required: req, Optimal: opt, HasTier: true}, nil
}

// resolveRoleTier resolves the tier for one role (required|optimal) from the
// labels: 0 matching labels -> missing; >=2 DISTINCT tier tokens -> conflict;
// exactly one -> parsed via modelroute.ParseWorkTier, with an unparseable or
// out-of-range token -> invalid. A repeated identical tier is deduped, not a
// conflict, mirroring the issue-contract grammar.
func resolveRoleTier(labels []string, role, missingFlag, invalidFlag, conflictFlag string) (modelroute.WorkTier, bool, []string) {
	var tokens []string
	for _, label := range labels {
		m := tierLabelRE.FindStringSubmatch(strings.ToLower(strings.TrimSpace(label)))
		if m == nil || m[2] != role {
			continue
		}
		tokens = appendDistinctTierToken(tokens, m[1])
	}
	switch len(tokens) {
	case 0:
		return 0, false, []string{missingFlag}
	case 1:
		t, ok := modelroute.ParseWorkTier(tokens[0])
		if !ok {
			return 0, false, []string{invalidFlag}
		}
		return t, true, nil
	default:
		return 0, false, []string{conflictFlag}
	}
}

// appendDistinctTierToken appends tok to xs unless it is already present, so two
// labels naming the SAME tier collapse to one (not a false conflict).
func appendDistinctTierToken(xs []string, tok string) []string {
	for _, e := range xs {
		if e == tok {
			return xs
		}
	}
	return append(xs, tok)
}
