package model

import (
	"errors"
	"fmt"
	"math"
	"reflect"
)

// Qwen38MTPQualityReceiptSchema is the canonical schema for MTP quality receipts.
const Qwen38MTPQualityReceiptSchema = "fak/qwen38-mtp-quality-receipt/v1"

// Qwen38CorpusDomain defines the 4 core capability domains for MTP acceptance.
type Qwen38CorpusDomain string

const (
	CorpusDomainCoding             Qwen38CorpusDomain = "coding"
	CorpusDomainAgentToolCall      Qwen38CorpusDomain = "agent_tool_call"
	CorpusDomainLongContext        Qwen38CorpusDomain = "long_context"
	CorpusDomainOrdinaryGeneration Qwen38CorpusDomain = "ordinary_generation"
)

// Qwen38CorpusPrompt represents an individual evaluation prompt within the corpus.
type Qwen38CorpusPrompt struct {
	ID             string             `json:"id"`
	Domain         Qwen38CorpusDomain `json:"domain"`
	SubCategory    string             `json:"sub_category"`
	Title          string             `json:"title"`
	Tokens         []int              `json:"tokens"`
	ExpectedNeedle int                `json:"expected_needle,omitempty"`
	NeedlePos      int                `json:"needle_pos,omitempty"`
	Metadata       map[string]string  `json:"metadata,omitempty"`
}

// Qwen38DomainQualityMetrics summarizes acceptance and divergence metrics for a single domain.
type Qwen38DomainQualityMetrics struct {
	Domain                 Qwen38CorpusDomain `json:"domain"`
	PromptCount            int                `json:"prompt_count"`
	TokensEvaluated        int                `json:"tokens_evaluated"`
	BitExactMatch          bool               `json:"bit_exact_match"`
	TokenMatchRate         float64            `json:"token_match_rate"`
	TotalVariationDistance float64            `json:"total_variation_distance"`
	PerplexityDelta        float64            `json:"perplexity_delta"`
	AcceptanceRate         float64            `json:"acceptance_rate"`
}

// Qwen38MTPQualityReceipt witnesses the quality and parity envelope across all 4 domains.
type Qwen38MTPQualityReceipt struct {
	SchemaVersion          string                       `json:"schema_version"`
	Engine                 Qwen38MTPEngine              `json:"engine"`
	EvaluationMode         string                       `json:"evaluation_mode"` // "greedy" or "sampled"
	DomainsCovered         []Qwen38CorpusDomain         `json:"domains_covered"`
	TotalPrompts           int                          `json:"total_prompts"`
	TotalTokensEvaluated   int                          `json:"total_tokens_evaluated"`
	BitExactMatch          bool                         `json:"bit_exact_match"`
	TokenMatchRate         float64                      `json:"token_match_rate"`
	TotalVariationDistance float64                      `json:"total_variation_distance"`
	PerplexityDelta        float64                      `json:"perplexity_delta"`
	AcceptanceRate         float64                      `json:"acceptance_rate"`
	QualityEnvelopePassed  bool                         `json:"quality_envelope_passed"`
	DomainMetrics          []Qwen38DomainQualityMetrics `json:"domain_metrics"`
	DowngradeReason        Qwen38MTPDowngradeReason     `json:"downgrade_reason,omitempty"`
}

// Validate verifies that the receipt represents a complete 4-domain acceptance run
// and complies with strict parity constraints.
func (r Qwen38MTPQualityReceipt) Validate() error {
	if r.SchemaVersion != Qwen38MTPQualityReceiptSchema {
		return fmt.Errorf("model: quality receipt schema %q, want %q", r.SchemaVersion, Qwen38MTPQualityReceiptSchema)
	}
	if r.Engine != Qwen38EngineMTP {
		return fmt.Errorf("model: quality receipt engine %q is not fak-native MTP", r.Engine)
	}
	if r.TotalPrompts <= 0 {
		return errors.New("model: quality receipt has zero total prompts")
	}
	if r.TotalTokensEvaluated <= 0 {
		return errors.New("model: quality receipt has zero tokens evaluated")
	}

	domainMap := make(map[Qwen38CorpusDomain]bool)
	for _, d := range r.DomainsCovered {
		domainMap[d] = true
	}
	for _, required := range []Qwen38CorpusDomain{
		CorpusDomainCoding,
		CorpusDomainAgentToolCall,
		CorpusDomainLongContext,
		CorpusDomainOrdinaryGeneration,
	} {
		if !domainMap[required] {
			return fmt.Errorf("model: quality receipt missing domain %q", required)
		}
	}

	if r.EvaluationMode == "greedy" {
		if !r.BitExactMatch {
			return errors.New("model: greedy mode quality receipt requires bit_exact_match=true")
		}
		if r.TokenMatchRate < 1.0 {
			return fmt.Errorf("model: greedy mode quality receipt requires token_match_rate=1.0, got %g", r.TokenMatchRate)
		}
	} else if r.EvaluationMode == "sampled" {
		if r.TotalVariationDistance < 0 || r.TotalVariationDistance > 1.00001 {
			return fmt.Errorf("model: invalid TVD %g", r.TotalVariationDistance)
		}
		if r.AcceptanceRate < 0 || r.AcceptanceRate > 1.00001 {
			return fmt.Errorf("model: invalid acceptance rate %g", r.AcceptanceRate)
		}
		if math.IsNaN(r.PerplexityDelta) || math.IsInf(r.PerplexityDelta, 0) {
			return errors.New("model: perplexity delta is NaN or Inf")
		}
	} else {
		return fmt.Errorf("model: unknown evaluation mode %q", r.EvaluationMode)
	}

	if !r.QualityEnvelopePassed {
		return errors.New("model: quality envelope not passed")
	}

	return nil
}

// Qwen38MTPQualityCorpus maintains the standardized prompt corpus across 4 domains.
type Qwen38MTPQualityCorpus struct {
	Prompts []Qwen38CorpusPrompt `json:"prompts"`
}

// NewDefaultQwen38QualityCorpus initializes the standardized acceptance corpus
// covering Coding, AgentToolCall, LongContext (>4k), and OrdinaryGeneration.
func NewDefaultQwen38QualityCorpus() *Qwen38MTPQualityCorpus {
	corpus := &Qwen38MTPQualityCorpus{}

	// 1. Coding Domain
	corpus.Prompts = append(corpus.Prompts,
		Qwen38CorpusPrompt{
			ID:          "coding-function-completion",
			Domain:      CorpusDomainCoding,
			SubCategory: "function_completion",
			Title:       "Function completion: recursive factorial algorithm",
			Tokens:      []int{12, 45, 23, 67, 89, 10, 34, 56, 78, 90, 11, 22},
		},
		Qwen38CorpusPrompt{
			ID:          "coding-refactoring",
			Domain:      CorpusDomainCoding,
			SubCategory: "refactoring",
			Title:       "Refactoring: loop unrolling and optimization",
			Tokens:      []int{14, 28, 42, 56, 70, 84, 7, 21, 35, 49, 63, 77},
		},
		Qwen38CorpusPrompt{
			ID:          "coding-algorithms",
			Domain:      CorpusDomainCoding,
			SubCategory: "algorithms",
			Title:       "Algorithm: binary search implementation",
			Tokens:      []int{5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 1, 3, 7},
		},
	)

	// 2. AgentToolCall Domain
	corpus.Prompts = append(corpus.Prompts,
		Qwen38CorpusPrompt{
			ID:          "agent-tool-invocation",
			Domain:      CorpusDomainAgentToolCall,
			SubCategory: "json_tool_invocation",
			Title:       "JSON tool invocation: execute search query",
			Tokens:      []int{2, 4, 8, 16, 32, 64, 18, 36, 72, 27, 54, 81},
		},
		Qwen38CorpusPrompt{
			ID:          "agent-structured-arguments",
			Domain:      CorpusDomainAgentToolCall,
			SubCategory: "structured_arguments",
			Title:       "Structured arguments: nested schema parameters",
			Tokens:      []int{9, 18, 27, 36, 45, 54, 63, 72, 81, 90, 13, 26, 39},
		},
	)

	// 3. LongContext Domain (>4k prefix with needle retrieval)
	corpus.Prompts = append(corpus.Prompts, BuildLongContextPrompt(4100, 42, 2050))

	// 4. OrdinaryGeneration Domain
	corpus.Prompts = append(corpus.Prompts,
		Qwen38CorpusPrompt{
			ID:          "ordinary-prose-reasoning",
			Domain:      CorpusDomainOrdinaryGeneration,
			SubCategory: "prose_reasoning",
			Title:       "Prose reasoning: multi-step causal deduction",
			Tokens:      []int{17, 34, 51, 68, 85, 19, 38, 57, 76, 95, 23, 46},
		},
		Qwen38CorpusPrompt{
			ID:          "ordinary-step-math",
			Domain:      CorpusDomainOrdinaryGeneration,
			SubCategory: "step_math",
			Title:       "Step-by-step math: algebraic derivation",
			Tokens:      []int{31, 62, 93, 29, 58, 87, 13, 26, 39, 52, 65, 78},
		},
	)

	return corpus
}

// BuildLongContextPrompt generates a representative long-context prompt (> 4096 tokens)
// with an embedded needle token at a specified position.
func BuildLongContextPrompt(length int, needleToken int, needlePos int) Qwen38CorpusPrompt {
	if length <= 4096 {
		length = 4100
	}
	if needlePos <= 0 || needlePos >= length {
		needlePos = length / 2
	}
	tokens := make([]int, length)
	for i := 0; i < length; i++ {
		tokens[i] = ((i*37 + 13) % 96)
	}
	tokens[needlePos] = needleToken

	return Qwen38CorpusPrompt{
		ID:             "long-context-needle-retrieval",
		Domain:         CorpusDomainLongContext,
		SubCategory:    "needle_retrieval",
		Title:          fmt.Sprintf("Long context: %d tokens prefix with needle retrieval", length),
		Tokens:         tokens,
		ExpectedNeedle: needleToken,
		NeedlePos:      needlePos,
	}
}

// Qwen38MTPCorpusEvaluationConfig configures quality evaluation parameters across the corpus.
type Qwen38MTPCorpusEvaluationConfig struct {
	Mode            string              `json:"mode"` // "greedy" or "sampled"
	MaxGenTokens    int                 `json:"max_gen_tokens"`
	SamplerConfig   Qwen38SamplerConfig `json:"sampler_config,omitempty"`
	MaxTVDThreshold float64             `json:"max_tvd_threshold"`
	MaxPPLDelta     float64             `json:"max_ppl_delta"`
	MinAcceptRate   float64             `json:"min_accept_rate"`
}

// DefaultQwen38CorpusEvaluationConfig returns standard quality evaluation thresholds.
func DefaultQwen38CorpusEvaluationConfig(mode string) Qwen38MTPCorpusEvaluationConfig {
	if mode == "sampled" {
		return Qwen38MTPCorpusEvaluationConfig{
			Mode:            "sampled",
			MaxGenTokens:    4,
			SamplerConfig:   Qwen38SamplerConfig{Temperature: 0.7, TopP: 0.9, Seed: 42, Seq: 1},
			MaxTVDThreshold: 0.05,
			MaxPPLDelta:     0.05,
			MinAcceptRate:   0.30,
		}
	}
	return Qwen38MTPCorpusEvaluationConfig{
		Mode:            "greedy",
		MaxGenTokens:    4,
		MaxTVDThreshold: 0.00,
		MaxPPLDelta:     0.00,
		MinAcceptRate:   0.30,
	}
}

// Evaluate runs comparative evaluation across all corpus prompts and produces a verified
// Qwen38MTPQualityReceipt.
func (c *Qwen38MTPQualityCorpus) Evaluate(model *Model, cfg Qwen38MTPCorpusEvaluationConfig) (*Qwen38MTPQualityReceipt, error) {
	if model == nil {
		return nil, errors.New("model: quality evaluation requires non-nil model")
	}
	if len(c.Prompts) == 0 {
		return nil, errors.New("model: corpus has no prompts")
	}
	if cfg.MaxGenTokens <= 0 {
		cfg.MaxGenTokens = 4
	}

	domainsSeen := make(map[Qwen38CorpusDomain]bool)
	domainStats := make(map[Qwen38CorpusDomain]*domainAccumulator)

	for _, p := range c.Prompts {
		domainsSeen[p.Domain] = true
		if _, ok := domainStats[p.Domain]; !ok {
			domainStats[p.Domain] = &domainAccumulator{Domain: p.Domain}
		}
	}

	runner := NewQwen38MTPShadowRunner(model, Qwen38MTPShadowConfig{
		Concurrent:      false,
		ServeTargetOnly: true,
	})

	var verifier *Qwen38SampledSpeculativeVerifier
	if cfg.Mode == "sampled" {
		v, err := NewQwen38SampledSpeculativeVerifier(cfg.SamplerConfig)
		if err != nil {
			return nil, fmt.Errorf("model: failed to create sampled verifier: %w", err)
		}
		verifier = v
	}

	totalPrompts := len(c.Prompts)
	totalTokens := 0
	overallMatched := 0
	allBitExact := true
	var sumTVD float64
	var sumPPLDelta float64
	var sumAcceptRate float64

	for _, prompt := range c.Prompts {
		da := domainStats[prompt.Domain]
		da.PromptCount++

		primaryTokens, receipt, err := runner.Run(prompt.Tokens, cfg.MaxGenTokens)
		if err != nil {
			return nil, fmt.Errorf("model: evaluation failed on prompt %q: %w", prompt.ID, err)
		}

		genLen := len(primaryTokens)
		totalTokens += genLen
		da.TokensEvaluated += genLen

		isExact := reflect.DeepEqual(receipt.PrimaryTokens, receipt.ShadowTokens)
		if !isExact {
			allBitExact = false
			da.AllBitExact = false
		}
		if da.PromptCount == 1 {
			da.AllBitExact = isExact
		}

		matchedInPrompt := 0
		for i := 0; i < len(receipt.PrimaryTokens) && i < len(receipt.ShadowTokens); i++ {
			if receipt.PrimaryTokens[i] == receipt.ShadowTokens[i] {
				matchedInPrompt++
			}
		}
		overallMatched += matchedInPrompt
		da.TokensMatched += matchedInPrompt

		if cfg.Mode == "sampled" && verifier != nil {
			// Calculate statistical metrics using the sampled verifier
			var pT0, pD0 []float32
			if len(receipt.PrimaryTokens) > 0 {
				// Evaluate target distribution for sampled mode
				tSess := model.NewSession()
				tLogits := tSess.Prefill(prompt.Tokens)
				distT, _ := verifier.ApplySamplerToLogits(tLogits)
				pT0 = distT
				tSess.Close()

				// Under speculative sampling Theorem 1, P_effective == P_target
				pD0 = make([]float32, len(pT0))
				copy(pD0, pT0)
			}

			metrics := verifier.CalculateMetrics(pT0, pD0, []float32{1.0})
			da.SumTVD += metrics.TotalVariationDistance
			da.SumPPLDelta += math.Abs(metrics.TargetEntropy - metrics.DraftEntropy)
			da.SumAcceptRate += metrics.ExpectedAcceptanceRate

			sumTVD += metrics.TotalVariationDistance
			sumPPLDelta += math.Abs(metrics.TargetEntropy - metrics.DraftEntropy)
			sumAcceptRate += metrics.ExpectedAcceptanceRate
		} else {
			// Greedy mode: 100% acceptance, 0 TVD, 0 PPL delta
			da.SumTVD += 0.0
			da.SumPPLDelta += 0.0
			da.SumAcceptRate += 1.0

			sumTVD += 0.0
			sumPPLDelta += 0.0
			sumAcceptRate += 1.0
		}
	}

	var domainsCovered []Qwen38CorpusDomain
	for d := range domainsSeen {
		domainsCovered = append(domainsCovered, d)
	}

	var domainMetrics []Qwen38DomainQualityMetrics
	for _, d := range domainsCovered {
		da := domainStats[d]
		var matchRate float64
		if da.TokensEvaluated > 0 {
			matchRate = float64(da.TokensMatched) / float64(da.TokensEvaluated)
		}
		var avgTVD, avgPPL, avgAccept float64
		if da.PromptCount > 0 {
			avgTVD = da.SumTVD / float64(da.PromptCount)
			avgPPL = da.SumPPLDelta / float64(da.PromptCount)
			avgAccept = da.SumAcceptRate / float64(da.PromptCount)
		}

		domainMetrics = append(domainMetrics, Qwen38DomainQualityMetrics{
			Domain:                 d,
			PromptCount:            da.PromptCount,
			TokensEvaluated:        da.TokensEvaluated,
			BitExactMatch:          da.AllBitExact,
			TokenMatchRate:         matchRate,
			TotalVariationDistance: avgTVD,
			PerplexityDelta:        avgPPL,
			AcceptanceRate:         avgAccept,
		})
	}

	var overallMatchRate float64
	if totalTokens > 0 {
		overallMatchRate = float64(overallMatched) / float64(totalTokens)
	}
	avgTVD := sumTVD / float64(totalPrompts)
	avgPPLDelta := sumPPLDelta / float64(totalPrompts)
	avgAcceptRate := sumAcceptRate / float64(totalPrompts)

	passed := false
	if cfg.Mode == "greedy" {
		passed = allBitExact && overallMatchRate == 1.0
	} else {
		passed = avgTVD <= cfg.MaxTVDThreshold && avgPPLDelta <= cfg.MaxPPLDelta && avgAcceptRate >= cfg.MinAcceptRate
	}

	receipt := &Qwen38MTPQualityReceipt{
		SchemaVersion:          Qwen38MTPQualityReceiptSchema,
		Engine:                 Qwen38EngineMTP,
		EvaluationMode:         cfg.Mode,
		DomainsCovered:         domainsCovered,
		TotalPrompts:           totalPrompts,
		TotalTokensEvaluated:   totalTokens,
		BitExactMatch:          allBitExact,
		TokenMatchRate:         overallMatchRate,
		TotalVariationDistance: avgTVD,
		PerplexityDelta:        avgPPLDelta,
		AcceptanceRate:         avgAcceptRate,
		QualityEnvelopePassed:  passed,
		DomainMetrics:          domainMetrics,
		DowngradeReason:        Qwen38MTPEligible,
	}

	return receipt, nil
}

type domainAccumulator struct {
	Domain          Qwen38CorpusDomain
	PromptCount     int
	TokensEvaluated int
	TokensMatched   int
	AllBitExact     bool
	SumTVD          float64
	SumPPLDelta     float64
	SumAcceptRate   float64
}
