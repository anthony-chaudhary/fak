package issuepolicy

import (
	"regexp"
	"strconv"
	"strings"
)

// modelTierLabelRE matches a namespaced model-tier label such as
// tier/T1-required or tier/T0-optimal (case-insensitive). Group 1 is the tier
// number, group 2 is the role. A priority label like priority/P1 does NOT match
// — that is the "Priority/P1 is not model tier T1" disambiguation baked into the
// grammar itself.
var modelTierLabelRE = regexp.MustCompile(`^tier/t([0-9]+)-(required|optimal)$`)

// modelTierTokenRE finds a T<N> tier token inside a body field value like
// "`tier/T1-required`" or "T1". "tier"/"priority" prefixes never match because a
// digit must follow the t; "P1" never matches because the letter is not t.
var modelTierTokenRE = regexp.MustCompile(`t([0-9]+)`)

// ModelTier is the parsed required/optimal model-tier metadata for an issue,
// the admission point for tier-aware dispatch (#3041). required is the
// capability FLOOR the work needs; optimal is the recommended tier. Tiers use
// the repo's WorkTier numbering (internal/modelroute): T0 is the MOST demanding,
// so a HIGHER number is LESS capable — optimal must be at least as demanding as
// required or the tags contradict. Source is "label" (a namespaced
// tier/T?-required|optimal label) or "body" (a Required/Optimal model tier
// field), with the label preferred and the body the stated fallback. Flags name
// missing, invalid, or contradictory metadata from a closed vocabulary. This is
// advisory by default; Options.StrictModelTier turns a flagged issue triage-only.
type ModelTier struct {
	Required       string   `json:"required,omitempty"`
	Optimal        string   `json:"optimal,omitempty"`
	RequiredSource string   `json:"required_source,omitempty"`
	OptimalSource  string   `json:"optimal_source,omitempty"`
	Flags          []string `json:"flags,omitempty"`
}

// modelTier parses the required/optimal model-tier metadata for a candidate.
// Namespaced tier/T?-required|optimal labels are the primary source; the
// Required/Optimal model tier body fields are the documented fallback. The flags
// name every missing, invalid, conflicting, or contradictory condition so a
// producer can refuse over-/under-tiered work before automated dispatch. The
// readout is always computed; whether a flag holds dispatch is the caller's
// StrictModelTier choice.
func modelTier(c Candidate) ModelTier {
	reqTier, reqSource, reqFlags := resolveModelTier(c.Labels, "required", c.RequiredModelTier)
	optTier, optSource, optFlags := resolveModelTier(c.Labels, "optimal", c.OptimalModelTier)
	mt := ModelTier{
		Required:       reqTier,
		Optimal:        optTier,
		RequiredSource: reqSource,
		OptimalSource:  optSource,
	}
	flags := append([]string(nil), reqFlags...)
	flags = append(flags, optFlags...)
	// Contradiction: the recommended (optimal) tier fails to meet the required
	// floor. Because T0 is the MOST demanding tier (lowest number), "optimal is
	// less demanding than required" is a HIGHER optimal number — you would be
	// recommending a model that cannot even clear the stated requirement.
	if reqTier != "" && optTier != "" && modelTierLessDemanding(optTier, reqTier) {
		flags = append(flags, "model_tier_contradiction")
	}
	mt.Flags = compact(flags)
	return mt
}

// resolveModelTier picks the tier for one role (required|optimal), preferring a
// namespaced label over the body fallback field. A non-empty source is only
// reported alongside a resolved tier. Two distinct labels for the same role is a
// conflict (cannot decide); a present-but-unparseable field is invalid; an
// entirely absent tier is missing.
func resolveModelTier(labels []string, role, bodyField string) (tier, source string, flags []string) {
	labelTiers := modelTierLabelValues(labels, role)
	if len(labelTiers) == 1 {
		return labelTiers[0], "label", nil
	}
	if len(labelTiers) > 1 {
		return "", "", []string{"model_tier_" + role + "_conflict"}
	}
	field := strings.TrimSpace(strings.Trim(strings.TrimSpace(bodyField), "`"))
	if field == "" {
		return "", "", []string{"model_tier_" + role + "_missing"}
	}
	if t, ok := parseModelTierToken(field); ok {
		return t, "body", nil
	}
	return "", "", []string{"model_tier_" + role + "_invalid"}
}

// modelTierLabelValues returns the canonical tiers (e.g. "T1") named by the
// role's namespaced labels, deduped and sorted. A priority/P1 label never
// contributes here — the label grammar requires a tier/ prefix and a T<N> token,
// which keeps priority and model tier separate.
func modelTierLabelValues(labels []string, role string) []string {
	var out []string
	for _, label := range labels {
		m := modelTierLabelRE.FindStringSubmatch(strings.ToLower(strings.TrimSpace(label)))
		if m == nil || m[2] != role {
			continue
		}
		out = append(out, "T"+m[1])
	}
	return compact(out)
}

// parseModelTierToken extracts the canonical T<N> tier from a body-field value
// such as "`tier/T1-required`", "tier/T1-optimal", or a bare "T1". It returns
// false when no T<N> token is present — the case that makes a stray "P1" (a
// priority, not a tier) an invalid tier field rather than a silent T1.
func parseModelTierToken(s string) (string, bool) {
	s = strings.ToLower(strings.TrimSpace(strings.Trim(s, "`")))
	if s == "" {
		return "", false
	}
	m := modelTierTokenRE.FindStringSubmatch(s)
	if m == nil {
		return "", false
	}
	return "T" + m[1], true
}

// modelTierLessDemanding reports whether tier a is strictly LESS demanding than
// b — a higher tier number, since T0 is the most demanding. Unparseable tiers
// are treated as non-comparable (no contradiction asserted on garbage).
func modelTierLessDemanding(a, b string) bool {
	an, aok := modelTierNumber(a)
	bn, bok := modelTierNumber(b)
	if !aok || !bok {
		return false
	}
	return an > bn
}

func modelTierNumber(t string) (int, bool) {
	t = strings.TrimPrefix(strings.ToLower(strings.TrimSpace(t)), "t")
	n, err := strconv.Atoi(t)
	if err != nil || n < 0 {
		return 0, false
	}
	return n, true
}
