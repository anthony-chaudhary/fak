package fleetaccounts

import "strings"

// apextier.go is the single home for the RESTRICTED apex tier — Fable 5.
//
// An account's ModelTier is an integer where a LOWER number is a more capable (and
// more expensive) model. The operator-facing rungs are:
//
//	TierApex     (0)  Fable 5 only — the apex model. Reached ONLY by an explicit
//	                  operator request (task class "tier0"/"apex"); it is never
//	                  inferred from task text and is never an auto-escalation
//	                  fallback target. This is the durable encoding of "use Fable 5
//	                  only in the clear cases where it is needed".
//	TierFrontier (1)  the max-quality frontier set (Opus-class, GPT-6 Astra,
//	                  GPT-5.6 Sol/Terra, GPT-5.5, DeepSeek V4 Pro, Kimi K2.6) — the
//	                  default target for a hard task.
//	TierLight    (2)  the lightweight-work set (GPT-5.6 Luna, GLM-5.2, Gemini 3.5 Flash).
//	TierOther    (3)  everything else (local, unknown).
//
// The restriction lives in two seams that both consult this file: modelTierFromName
// (the taxonomy — only a Fable-5 model name classifies as apex) and the router
// (ClassifyTask targets apex only on an explicit request; RouteAccount never lets a
// non-apex target's fallback ladder reach a tier-0 account). Keeping the membership
// and the gate here means the taxonomy, the router, and any cost lens agree on what
// "apex" is instead of drifting apart.
const (
	TierApex     = 0
	TierFrontier = 1
	TierLight    = 2
	TierOther    = 3
)

// apexModelCompactTokens are the substrings — matched against the normalized,
// punctuation-stripped model name — that mark a model as the apex tier. Fable 5 is
// the only apex model today; this list is the ONE place that membership is declared.
var apexModelCompactTokens = []string{"claudefable5", "fable5"}

// IsApexModel reports whether a model id names the restricted apex tier (Fable 5),
// in any id shape it appears (claude-fable-5, fable_5, "Fable 5", …).
func IsApexModel(model string) bool {
	text := strings.ReplaceAll(strings.ToLower(model), "_", "-")
	text = strings.ReplaceAll(text, " ", "-")
	compact := nonAlnum.ReplaceAllString(text, "")
	for _, tok := range apexModelCompactTokens {
		if strings.Contains(compact, tok) {
			return true
		}
	}
	return false
}

// apexRequested reports whether an operator task-class token explicitly asks for the
// apex tier. Apex is EXPLICIT-ONLY: no task-text heuristic ever selects it, so this
// is the single gate that opens tier 0. Anything not in this closed set leaves apex
// unreachable, which is what keeps Fable 5 to the clear cases it is needed.
func apexRequested(taskClass string) bool {
	return in(strings.ToLower(strings.TrimSpace(taskClass)),
		"tier0", "t0", "0", "apex", "fable", "fable5", "fable-5")
}

// validModelTier reports whether t is a recognized rung of the taxonomy (apex through
// other). Used to accept an inferred apex (0) while a bare, unset profile tier of 0
// still routes through name inference — see cleanProfile.
func validModelTier(t int) bool {
	return t == TierApex || t == TierFrontier || t == TierLight || t == TierOther
}
