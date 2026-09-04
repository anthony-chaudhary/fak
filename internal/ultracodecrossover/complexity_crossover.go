package ultracodecrossover

import (
	"errors"
	"fmt"
)

// ComplexityCrossoverSchema defines the canonical schema identifier for replayable crossover evaluations.
const ComplexityCrossoverSchema = "fak-ultracode-complexity-crossover/1"

// ComplexityCampaign is the replayable six-rung envelope for micro-context scoping.
type ComplexityCampaign struct {
	Schema                 string           `json:"schema"`
	CampaignVersion        string           `json:"campaign_version"`
	Model                  string           `json:"model"`
	Runtime                string           `json:"runtime"`
	Tokenizer              string           `json:"tokenizer"`
	CachePosture           string           `json:"cache_posture"`
	SourceReceipt          string           `json:"source_receipt"`
	PromotionEvidence      string           `json:"promotion_evidence"`
	DemotionEvidence       string           `json:"demotion_evidence"`
	InvalidatingAssumption string           `json:"invalidating_assumption"`
	Rungs                  []ComplexityRung `json:"rungs"`
}

// ComplexityRung defines an individual depth stage in the task-complexity ladder.
type ComplexityRung struct {
	Number          int              `json:"number"`
	Name            string           `json:"name"`
	DependencyDepth int              `json:"dependency_depth"`
	Task            string           `json:"task"`
	FrozenCheck     string           `json:"frozen_check"`
	Cells           []ComplexityCell `json:"cells"`
}

// ComplexityCell captures the execution metrics and acceptance status for a specific width and caching mode.
type ComplexityCell struct {
	Width         int    `json:"width"`
	Context       string `json:"context"`
	Cache         string `json:"cache"`
	Accepted      bool   `json:"accepted"`
	OutcomeDigest string `json:"outcome_digest"`
	InputTokens   int64  `json:"input_tokens"`
	CachedTokens  int64  `json:"cached_tokens"`
	SourceReceipt string `json:"source_receipt"`
}

// ComplexityReport synthesizes the crossover boundaries and token attribution across ladder rungs.
type ComplexityReport struct {
	Schema                 string                 `json:"schema"`
	CampaignVersion        string                 `json:"campaign_version"`
	Model                  string                 `json:"model"`
	Runtime                string                 `json:"runtime"`
	Tokenizer              string                 `json:"tokenizer"`
	CachePosture           string                 `json:"cache_posture"`
	Rungs                  []ComplexityRungResult `json:"rungs"`
	LastEqualOutcomeRung   int                    `json:"last_equal_outcome_rung"`
	FirstFailureRung       int                    `json:"first_failure_rung"`
	StoppedAfterRung       int                    `json:"stopped_after_rung"`
	FailureMode            string                 `json:"failure_mode"`
	ReplayCommand          string                 `json:"replay_command"`
	PromotionEvidence      string                 `json:"promotion_evidence"`
	DemotionEvidence       string                 `json:"demotion_evidence"`
	InvalidatingAssumption string                 `json:"invalidating_assumption"`
}

// ComplexityRungResult records the comparison outcome and token savings for a single complexity level.
type ComplexityRungResult struct {
	Number             int    `json:"number"`
	Name               string `json:"name"`
	Verdict            string `json:"verdict"`
	Reason             string `json:"reason,omitempty"`
	ScopeAvoidedTokens int64  `json:"scope_avoided_tokens"`
	PrefixReadTokens   int64  `json:"prefix_read_tokens"`
	ComparedCells      int    `json:"compared_cells"`
}

// EvaluateComplexityCampaign preserves each rung and abstention instead of averaging
// across unequal outcomes. It stops after two consecutive quality failures.
// Invariant: UltraCode crossover campaigns are fail-closed and bounded.
// Guard: Execution halts after two consecutive failures and rejects incomplete cell envelopes.
func EvaluateComplexityCampaign(c ComplexityCampaign) (ComplexityReport, error) {
	if c.Schema != ComplexityCrossoverSchema {
		return ComplexityReport{}, fmt.Errorf("schema must be %q", ComplexityCrossoverSchema)
	}
	if len(c.Rungs) != 6 {
		return ComplexityReport{}, fmt.Errorf("complexity ladder must contain exactly six rungs, got %d", len(c.Rungs))
	}
	if c.Model == "" || c.Runtime == "" || c.Tokenizer == "" || c.CampaignVersion == "" || c.SourceReceipt == "" {
		return ComplexityReport{}, errors.New("campaign must name version, model, runtime, tokenizer, and source receipt")
	}
	if c.PromotionEvidence == "" || c.DemotionEvidence == "" || c.InvalidatingAssumption == "" {
		return ComplexityReport{}, errors.New("generation evidence and invalidating assumption are required")
	}
	r := ComplexityReport{Schema: c.Schema, CampaignVersion: c.CampaignVersion, Model: c.Model, Runtime: c.Runtime, Tokenizer: c.Tokenizer, CachePosture: c.CachePosture, ReplayCommand: "go test ./internal/ultracodecrossover -run TestIssue8674ArtifactReplay -count=1", PromotionEvidence: c.PromotionEvidence, DemotionEvidence: c.DemotionEvidence, InvalidatingAssumption: c.InvalidatingAssumption}
	failures := 0
	for i, rung := range c.Rungs {
		if rung.Number != i+1 || rung.Name == "" || rung.Task == "" || rung.FrozenCheck == "" {
			return ComplexityReport{}, fmt.Errorf("rung %d identity and frozen check are required", i+1)
		}
		result, err := evaluateComplexityRung(rung)
		if err != nil {
			return ComplexityReport{}, fmt.Errorf("rung %d: %w", rung.Number, err)
		}
		r.Rungs = append(r.Rungs, result)
		if result.Verdict == "EQUAL_OUTCOME_GAIN" {
			failures = 0
			r.LastEqualOutcomeRung = rung.Number
		} else {
			if r.FirstFailureRung == 0 {
				r.FirstFailureRung = rung.Number
				r.FailureMode = result.Reason
			}
			failures++
			if failures == 2 {
				r.StoppedAfterRung = rung.Number
				break
			}
		}
	}
	if r.FirstFailureRung == 0 {
		r.FailureMode = "none observed in bounded envelope"
	}
	return r, nil
}

func evaluateComplexityRung(r ComplexityRung) (ComplexityRungResult, error) {
	out := ComplexityRungResult{Number: r.Number, Name: r.Name, Verdict: "EQUAL_OUTCOME_GAIN"}
	for _, width := range []int{1, 2, 4, 8} {
		for _, cache := range []string{"cold", "warm"} {
			var full, scoped *ComplexityCell
			for i := range r.Cells {
				c := &r.Cells[i]
				if c.Width == width && c.Cache == cache && c.Context == "full" {
					full = c
				}
				if c.Width == width && c.Cache == cache && c.Context == "scoped" {
					scoped = c
				}
			}
			if full == nil || scoped == nil {
				return out, fmt.Errorf("missing full/scoped %s cells at width %d", cache, width)
			}
			if full.SourceReceipt == "" || scoped.SourceReceipt == "" {
				return out, errors.New("cell source receipt is required")
			}
			out.ComparedCells += 2
			if !full.Accepted || !scoped.Accepted || full.OutcomeDigest != scoped.OutcomeDigest {
				out.Verdict = "ABSTAIN"
				out.Reason = "scoped outcome failed frozen check or diverged from full context (missing-context failure)"
				continue
			}
			if full.InputTokens > scoped.InputTokens {
				out.ScopeAvoidedTokens += full.InputTokens - scoped.InputTokens
			}
			if cache == "warm" {
				out.PrefixReadTokens += full.CachedTokens + scoped.CachedTokens
			}
		}
	}
	return out, nil
}
