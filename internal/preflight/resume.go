package preflight

import "github.com/anthony-chaudhary/fak/internal/resume"

const (
	// ResumePreflightSchema is the content-free audit row schema for the pre-resume
	// advisory rung. It intentionally mirrors the gateway resume residual row's
	// plan/projection fields, adding phase so projected-at-preflight and
	// observed-at-bill rows can live in one ledger.
	ResumePreflightSchema = "fak.preflight.resume/1"

	ResumeBudgetColdPrefillExceeded = "COLD_PREFILL_BUDGET_EXCEEDED"
)

type ResumePath string

const (
	ResumePathWarmSplice ResumePath = "WARM-SPLICE"
	ResumePathWarm       ResumePath = "WARM"
	ResumePathCut        ResumePath = "CUT"
	ResumePathReset      ResumePath = "RESET"
)

// ResumeInput is the pure pre-resume rung input. The caller supplies resume.Plan's
// facts plus whether the host already knows the in-process warm-KV splice is
// available. BudgetTokens is advisory: exceeding it labels the row, never blocks.
type ResumeInput struct {
	Plan                resume.Input `json:"plan"`
	WarmSpliceAvailable bool         `json:"warm_splice_available"`
	BudgetTokens        int          `json:"budget_tokens,omitempty"`
}

func ResumeTokensFromBytes(n int) int {
	if n <= 0 {
		return 0
	}
	return (n + 3) / 4
}

type ResumePathCost struct {
	Path                   ResumePath      `json:"path"`
	Strategy               resume.Strategy `json:"strategy,omitempty"`
	Available              bool            `json:"available"`
	PrefillTokens          int             `json:"prefill_tokens"`
	ProjectedPromptCostUSD float64         `json:"projected_prompt_cost_usd"`
	Reason                 string          `json:"reason,omitempty"`
}

// ResumeVerdict is a deterministic, advisory preflight record. It does not admit
// or refuse a resume; it labels the cheapest safe path the caller should feed into
// the live resume/guard decision.
type ResumeVerdict struct {
	Schema string `json:"schema"`
	Phase  string `json:"phase"`

	Path        ResumePath      `json:"path"`
	Posture     resume.Posture  `json:"posture"`
	Recommended resume.Strategy `json:"recommended"`
	Reason      string          `json:"reason"`

	ResidentTokens          int     `json:"resident_tokens"`
	ProjectedPrefillTokens  int     `json:"projected_prefill_tokens"`
	ProjectedPromptCostUSD  float64 `json:"projected_prompt_cost_usd"`
	ProjectedColdWriteShare float64 `json:"projected_cold_write_share"`

	BudgetTokens   int    `json:"budget_tokens,omitempty"`
	BudgetReason   string `json:"budget_reason,omitempty"`
	BudgetExceeded bool   `json:"budget_exceeded,omitempty"`

	Advisory   bool             `json:"advisory"`
	Report     resume.Report    `json:"report"`
	Candidates []ResumePathCost `json:"candidates"`
}

// PlanResume runs the advisory pre-resume rung. It consumes resume.Plan for the
// strategy ranking and only adds preflight-specific labelling: WARM-SPLICE when
// the caller has a live in-process splice, and a fail-open budget reason.
func PlanResume(in ResumeInput) ResumeVerdict {
	rep := resume.Plan(in.Plan)
	candidates := resumeCandidates(rep, in.WarmSpliceAvailable)
	selected := selectResumeCandidate(rep, candidates, in.WarmSpliceAvailable)

	out := ResumeVerdict{
		Schema:                  ResumePreflightSchema,
		Phase:                   "projected_at_preflight",
		Path:                    selected.Path,
		Posture:                 rep.Posture,
		Recommended:             rep.Recommended,
		Reason:                  selected.Reason,
		ResidentTokens:          rep.ResidentTokens,
		ProjectedPrefillTokens:  selected.PrefillTokens,
		ProjectedPromptCostUSD:  selected.ProjectedPromptCostUSD,
		ProjectedColdWriteShare: resume.ColdWriteShare,
		BudgetTokens:            in.BudgetTokens,
		Advisory:                true,
		Report:                  rep,
		Candidates:              candidates,
	}
	if in.BudgetTokens > 0 && selected.PrefillTokens > in.BudgetTokens &&
		(rep.Posture == resume.PostureCold || rep.Posture == resume.PostureUnknown) {
		out.BudgetExceeded = true
		out.BudgetReason = ResumeBudgetColdPrefillExceeded
	}
	return out
}

func selectResumeCandidate(rep resume.Report, candidates []ResumePathCost, warmSplice bool) ResumePathCost {
	if warmSplice {
		return candidates[0]
	}
	want := pathForRecommendation(rep)
	for _, c := range candidates {
		if c.Path == want {
			if c.Strategy == rep.Recommended {
				c.Reason = rep.Reason
			}
			return c
		}
	}
	return ResumePathCost{Path: want, Strategy: rep.Recommended, Available: true, Reason: rep.Reason}
}

func resumeCandidates(rep resume.Report, warmSplice bool) []ResumePathCost {
	full := strategyCost(rep, resume.StrategyResumeFull)
	cut := strategyCost(rep, resume.StrategyCut)
	reset := strategyCost(rep, resume.StrategyReset)
	return []ResumePathCost{
		{
			Path:                   ResumePathWarmSplice,
			Available:              warmSplice,
			PrefillTokens:          0,
			ProjectedPromptCostUSD: 0,
			Reason:                 "warm_splice_available",
		},
		{
			Path:                   ResumePathWarm,
			Strategy:               resume.StrategyResumeFull,
			Available:              rep.Posture == resume.PostureWarm || rep.Posture == resume.PostureWarmHit,
			PrefillTokens:          full.PrefillTokens,
			ProjectedPromptCostUSD: promptCost(rep, full),
			Reason:                 resume.ReasonWarmPrefixIntact,
		},
		{
			Path:                   ResumePathCut,
			Strategy:               resume.StrategyCut,
			Available:              true,
			PrefillTokens:          cut.PrefillTokens,
			ProjectedPromptCostUSD: promptCost(rep, cut),
			Reason:                 rep.Reason,
		},
		{
			Path:                   ResumePathReset,
			Strategy:               resume.StrategyReset,
			Available:              true,
			PrefillTokens:          reset.PrefillTokens,
			ProjectedPromptCostUSD: promptCost(rep, reset),
			Reason:                 "reset_rebuild",
		},
	}
}

func pathForRecommendation(rep resume.Report) ResumePath {
	switch rep.Recommended {
	case resume.StrategyCut:
		return ResumePathCut
	case resume.StrategyReset:
		return ResumePathReset
	default:
		return ResumePathWarm
	}
}

func strategyCost(rep resume.Report, strategy resume.Strategy) resume.StrategyCost {
	for _, c := range rep.Strategies {
		if c.Strategy == strategy {
			return c
		}
	}
	return resume.StrategyCost{Strategy: strategy}
}

func promptCost(rep resume.Report, c resume.StrategyCost) float64 {
	if rep.Posture == resume.PostureWarm || rep.Posture == resume.PostureWarmHit {
		return float64(c.PrefillTokens) * rep.Pricing.InputPerMTokUSD / 1_000_000 * resume.CacheReadMultiplier
	}
	return c.ColdReprefillUSD
}
