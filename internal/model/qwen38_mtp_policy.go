package model

import (
	"errors"
	"fmt"
	"sync"
	"time"
)

// Qwen38DraftPolicyReceiptSchema is the canonical schema version for draft policy receipts.
const Qwen38DraftPolicyReceiptSchema = "fak/qwen38-draft-policy-receipt/v1"

// DraftSource identifies the closed vocabulary of supported draft mechanisms.
type DraftSource string

const (
	DraftSourceMTP          DraftSource = "mtp"
	DraftSourcePromptLookup DraftSource = "prompt_lookup"
	DraftSourceNGram        DraftSource = "ngram"
	DraftSourceNone         DraftSource = "none" // target-only decode
)

// ValidDraftSource checks whether the draft source belongs to the closed vocabulary.
func ValidDraftSource(s DraftSource) bool {
	switch s {
	case DraftSourceMTP, DraftSourcePromptLookup, DraftSourceNGram, DraftSourceNone:
		return true
	default:
		return false
	}
}

// DraftRejectionReason defines the closed vocabulary of reasons for rejecting candidate draft sources.
type DraftRejectionReason string

const (
	DraftRejectionNone                 DraftRejectionReason = ""
	DraftRejectionMutualExclusion      DraftRejectionReason = "mutual_exclusion"
	DraftRejectionLowAcceptance        DraftRejectionReason = "low_recent_acceptance"
	DraftRejectionSourceUnhealthy      DraftRejectionReason = "source_unhealthy"
	DraftRejectionPromptIneligible     DraftRejectionReason = "prompt_ineligible"
	DraftRejectionCacheInsufficient    DraftRejectionReason = "cache_state_insufficient"
	DraftRejectionPolicyDisabled       DraftRejectionReason = "disabled_by_policy"
	DraftRejectionLowerPriority        DraftRejectionReason = "lower_priority_fallback"
	DraftRejectionUnsupportedEnvelope  DraftRejectionReason = "unsupported_envelope"
	DraftRejectionIncompatibleStacking DraftRejectionReason = "incompatible_stacking_unauthorized"
)

// ValidDraftRejectionReason checks whether the rejection reason belongs to the closed vocabulary.
func ValidDraftRejectionReason(r DraftRejectionReason) bool {
	switch r {
	case DraftRejectionNone,
		DraftRejectionMutualExclusion,
		DraftRejectionLowAcceptance,
		DraftRejectionSourceUnhealthy,
		DraftRejectionPromptIneligible,
		DraftRejectionCacheInsufficient,
		DraftRejectionPolicyDisabled,
		DraftRejectionLowerPriority,
		DraftRejectionUnsupportedEnvelope,
		DraftRejectionIncompatibleStacking:
		return true
	default:
		return false
	}
}

// RejectedDraftSource records a rejected alternative alongside its typed rejection reason.
type RejectedDraftSource struct {
	Source DraftSource          `json:"source"`
	Reason DraftRejectionReason `json:"reason"`
	Detail string               `json:"detail,omitempty"`
}

// PromptCharacteristics describes the structural properties of an input prompt.
type PromptCharacteristics struct {
	TokenCount       int  `json:"token_count"`
	Repetitive       bool `json:"repetitive"`
	HasNGramMatch    bool `json:"has_ngram_match"`
	HasLookupMatch   bool `json:"has_lookup_match"`
	CodeOrStructured bool `json:"code_or_structured"`
}

// CacheCharacteristics captures prefix and KV cache state.
type CacheCharacteristics struct {
	PrefixCacheReady bool    `json:"prefix_cache_ready"`
	CacheHitRate     float64 `json:"cache_hit_rate"`
	ResidentTokens   int     `json:"resident_tokens"`
}

// DraftSourceHealth tracks operational availability and error counts.
type DraftSourceHealth struct {
	Healthy    bool   `json:"healthy"`
	ErrorCount int    `json:"error_count"`
	Detail     string `json:"detail,omitempty"`
}

// DraftSourcePerformance tracks empirical token proposal and acceptance statistics.
type DraftSourcePerformance struct {
	ProposedTokens int     `json:"proposed_tokens"`
	AcceptedTokens int     `json:"accepted_tokens"`
	AcceptanceRate float64 `json:"acceptance_rate"`
}

// DraftCandidateState groups eligibility, health, and performance for a draft source.
type DraftCandidateState struct {
	Source      DraftSource            `json:"source"`
	Eligible    bool                   `json:"eligible"`
	Health      DraftSourceHealth      `json:"health"`
	Performance DraftSourcePerformance `json:"performance"`
}

// Qwen38DraftPolicyInput contains contextual signals used to adjudicate draft source selection.
type Qwen38DraftPolicyInput struct {
	Prompt            PromptCharacteristics               `json:"prompt"`
	Cache             CacheCharacteristics                `json:"cache"`
	Candidates        map[DraftSource]DraftCandidateState `json:"candidates"`
	MTPEligibility    *Qwen38MTPEligibility               `json:"mtp_eligibility,omitempty"`
	AllowStacked      bool                                `json:"allow_stacked"`
	ExplicitStackAuth bool                                `json:"explicit_stack_auth"`
	MinAcceptanceRate float64                             `json:"min_acceptance_rate"`
	OperatorEnabled   bool                                `json:"operator_enabled"`
}

// Qwen38DraftPolicyReceipt witnesses the deterministic draft source decision.
type Qwen38DraftPolicyReceipt struct {
	SchemaVersion     string                   `json:"schema_version"`
	ChosenSource      DraftSource              `json:"chosen_source"`
	Engine            Qwen38MTPEngine          `json:"engine"`
	AllowStacked      bool                     `json:"allow_stacked"`
	StackedSources    []DraftSource            `json:"stacked_sources,omitempty"`
	Rejected          []RejectedDraftSource    `json:"rejected"`
	DowngradeReason   Qwen38MTPDowngradeReason `json:"downgrade_reason,omitempty"`
	DecisionLatencyNS uint64                   `json:"decision_latency_ns"`
}

// Validate verifies that the receipt adheres to all schema and mutual-exclusion invariants.
func (r Qwen38DraftPolicyReceipt) Validate() error {
	if r.SchemaVersion != Qwen38DraftPolicyReceiptSchema {
		return fmt.Errorf("model: draft policy receipt schema %q, want %q", r.SchemaVersion, Qwen38DraftPolicyReceiptSchema)
	}
	if !ValidDraftSource(r.ChosenSource) {
		return fmt.Errorf("model: draft policy receipt invalid chosen source %q", r.ChosenSource)
	}
	if !validQwen38MTPEngine(r.Engine) {
		return fmt.Errorf("model: draft policy receipt invalid or foreign engine %q", r.Engine)
	}
	if r.ChosenSource == DraftSourceMTP {
		if r.Engine != Qwen38EngineMTP {
			return fmt.Errorf("model: draft policy receipt chosen MTP requires engine %q, got %q", Qwen38EngineMTP, r.Engine)
		}
	} else {
		if r.Engine != Qwen38EngineTargetDecode {
			return fmt.Errorf("model: draft policy receipt chosen non-MTP source %q requires engine %q, got %q", r.ChosenSource, Qwen38EngineTargetDecode, r.Engine)
		}
	}
	if r.ChosenSource == DraftSourceNone {
		if r.DowngradeReason == Qwen38MTPEligible {
			return errors.New("model: draft policy receipt target-only selection requires a non-empty downgrade reason")
		}
	}
	if !r.AllowStacked && len(r.StackedSources) > 0 {
		return errors.New("model: draft policy receipt has stacked sources without allow_stacked=true")
	}

	seenRejected := make(map[DraftSource]bool, len(r.Rejected))
	for _, rej := range r.Rejected {
		if !ValidDraftSource(rej.Source) {
			return fmt.Errorf("model: draft policy receipt rejected source %q is invalid", rej.Source)
		}
		if !ValidDraftRejectionReason(rej.Reason) || rej.Reason == DraftRejectionNone {
			return fmt.Errorf("model: draft policy receipt rejected source %q has invalid reason %q", rej.Source, rej.Reason)
		}
		if seenRejected[rej.Source] {
			return fmt.Errorf("model: draft policy receipt duplicate rejected entry for source %q", rej.Source)
		}
		seenRejected[rej.Source] = true
		if rej.Source == r.ChosenSource {
			return fmt.Errorf("model: draft policy receipt chosen source %q cannot appear in rejected list", rej.Source)
		}
	}

	return nil
}

// Qwen38CompositeDraftPolicy manages deterministic selection and mutual exclusion
// across multiple native draft sources.
type Qwen38CompositeDraftPolicy struct {
	mu                sync.RWMutex
	MinAcceptanceRate float64
	AllowStacked      bool
}

// NewQwen38CompositeDraftPolicy constructs a default composite policy instance.
func NewQwen38CompositeDraftPolicy() *Qwen38CompositeDraftPolicy {
	return &Qwen38CompositeDraftPolicy{
		MinAcceptanceRate: 0.40,
		AllowStacked:      false,
	}
}

// SelectDraftSource deterministically adjudicates candidate draft sources and produces a witness receipt.
func (p *Qwen38CompositeDraftPolicy) SelectDraftSource(input Qwen38DraftPolicyInput) Qwen38DraftPolicyReceipt {
	start := time.Now()

	p.mu.RLock()
	defaultMinAcceptance := p.MinAcceptanceRate
	policyAllowStacked := p.AllowStacked
	p.mu.RUnlock()

	minAcceptance := defaultMinAcceptance
	if input.MinAcceptanceRate > 0 {
		minAcceptance = input.MinAcceptanceRate
	}

	effectiveAllowStacked := policyAllowStacked || input.AllowStacked

	allSources := []DraftSource{DraftSourceMTP, DraftSourcePromptLookup, DraftSourceNGram}

	if !input.OperatorEnabled {
		return qwen38PolicyDisabledReceipt(allSources, effectiveAllowStacked, start)
	}

	evals := make(map[DraftSource]qwen38DraftEvaluation, len(allSources))
	evals[DraftSourceMTP] = evaluateQwen38MTPCandidate(input, minAcceptance)
	evals[DraftSourcePromptLookup] = evaluateQwen38PromptLookupCandidate(input, minAcceptance)
	evals[DraftSourceNGram] = evaluateQwen38NGramCandidate(input, minAcceptance)

	sel := resolveQwen38DraftSelection(evals, allSources, effectiveAllowStacked, input.ExplicitStackAuth)

	lat := uint64(time.Since(start).Nanoseconds())
	if lat == 0 {
		lat = 1
	}

	return Qwen38DraftPolicyReceipt{
		SchemaVersion:     Qwen38DraftPolicyReceiptSchema,
		ChosenSource:      sel.chosenSource,
		Engine:            sel.engine,
		AllowStacked:      effectiveAllowStacked,
		StackedSources:    sel.stackedSources,
		Rejected:          sel.rejected,
		DowngradeReason:   sel.downgradeReason,
		DecisionLatencyNS: lat,
	}
}

type qwen38DraftEvaluation struct {
	eligible  bool
	rejection *RejectedDraftSource
}

func qwen38DraftRejected(src DraftSource, reason DraftRejectionReason, detail string) qwen38DraftEvaluation {
	return qwen38DraftEvaluation{
		eligible: false,
		rejection: &RejectedDraftSource{
			Source: src,
			Reason: reason,
			Detail: detail,
		},
	}
}

func qwen38LowAcceptanceDetail(rate, threshold float64) string {
	return fmt.Sprintf("acceptance rate %.2f < threshold %.2f", rate, threshold)
}

func qwen38PolicyDisabledReceipt(allSources []DraftSource, allowStacked bool, start time.Time) Qwen38DraftPolicyReceipt {
	rejected := make([]RejectedDraftSource, 0, len(allSources))
	for _, src := range allSources {
		rejected = append(rejected, RejectedDraftSource{
			Source: src,
			Reason: DraftRejectionPolicyDisabled,
			Detail: "operator policy disabled drafting",
		})
	}
	return Qwen38DraftPolicyReceipt{
		SchemaVersion:     Qwen38DraftPolicyReceiptSchema,
		ChosenSource:      DraftSourceNone,
		Engine:            Qwen38EngineTargetDecode,
		AllowStacked:      allowStacked,
		Rejected:          rejected,
		DowngradeReason:   Qwen38MTPDisabledByPolicy,
		DecisionLatencyNS: uint64(time.Since(start).Nanoseconds()),
	}
}

func evaluateQwen38MTPCandidate(input Qwen38DraftPolicyInput, minAcceptance float64) qwen38DraftEvaluation {
	mtpState, hasMTPState := input.Candidates[DraftSourceMTP]
	switch {
	case input.MTPEligibility != nil && !input.MTPEligibility.Eligible:
		return qwen38DraftRejected(DraftSourceMTP, DraftRejectionUnsupportedEnvelope, string(input.MTPEligibility.DowngradeReason))
	case hasMTPState && !mtpState.Eligible:
		return qwen38DraftRejected(DraftSourceMTP, DraftRejectionUnsupportedEnvelope, "mtp candidate marked ineligible")
	case hasMTPState && !mtpState.Health.Healthy:
		return qwen38DraftRejected(DraftSourceMTP, DraftRejectionSourceUnhealthy, mtpState.Health.Detail)
	case hasMTPState && mtpState.Performance.ProposedTokens > 0 && mtpState.Performance.AcceptanceRate < minAcceptance:
		return qwen38DraftRejected(DraftSourceMTP, DraftRejectionLowAcceptance, qwen38LowAcceptanceDetail(mtpState.Performance.AcceptanceRate, minAcceptance))
	case (input.MTPEligibility != nil && input.MTPEligibility.Eligible) || (hasMTPState && mtpState.Eligible):
		return qwen38DraftEvaluation{eligible: true}
	default:
		return qwen38DraftRejected(DraftSourceMTP, DraftRejectionUnsupportedEnvelope, "mtp eligibility not established")
	}
}

func evaluateQwen38PromptLookupCandidate(input Qwen38DraftPolicyInput, minAcceptance float64) qwen38DraftEvaluation {
	plState, hasPLState := input.Candidates[DraftSourcePromptLookup]
	switch {
	case !hasPLState || !plState.Eligible:
		return qwen38DraftRejected(DraftSourcePromptLookup, DraftRejectionUnsupportedEnvelope, "prompt lookup candidate not eligible")
	case !plState.Health.Healthy:
		return qwen38DraftRejected(DraftSourcePromptLookup, DraftRejectionSourceUnhealthy, plState.Health.Detail)
	case !input.Prompt.HasLookupMatch && !input.Prompt.Repetitive:
		return qwen38DraftRejected(DraftSourcePromptLookup, DraftRejectionPromptIneligible, "prompt lacks repeated tokens or matching lookup candidate spans")
	case !input.Cache.PrefixCacheReady && input.Prompt.TokenCount < 16:
		return qwen38DraftRejected(DraftSourcePromptLookup, DraftRejectionCacheInsufficient, "cold cache and short prompt insufficient for lookup retrieval")
	case plState.Performance.ProposedTokens > 0 && plState.Performance.AcceptanceRate < minAcceptance:
		return qwen38DraftRejected(DraftSourcePromptLookup, DraftRejectionLowAcceptance, qwen38LowAcceptanceDetail(plState.Performance.AcceptanceRate, minAcceptance))
	default:
		return qwen38DraftEvaluation{eligible: true}
	}
}

func evaluateQwen38NGramCandidate(input Qwen38DraftPolicyInput, minAcceptance float64) qwen38DraftEvaluation {
	ngState, hasNGState := input.Candidates[DraftSourceNGram]
	switch {
	case !hasNGState || !ngState.Eligible:
		return qwen38DraftRejected(DraftSourceNGram, DraftRejectionUnsupportedEnvelope, "ngram candidate not eligible")
	case !ngState.Health.Healthy:
		return qwen38DraftRejected(DraftSourceNGram, DraftRejectionSourceUnhealthy, ngState.Health.Detail)
	case !input.Prompt.HasNGramMatch && !input.Prompt.Repetitive:
		return qwen38DraftRejected(DraftSourceNGram, DraftRejectionPromptIneligible, "prompt lacks n-gram match patterns or repetitive structure")
	case ngState.Performance.ProposedTokens > 0 && ngState.Performance.AcceptanceRate < minAcceptance:
		return qwen38DraftRejected(DraftSourceNGram, DraftRejectionLowAcceptance, qwen38LowAcceptanceDetail(ngState.Performance.AcceptanceRate, minAcceptance))
	default:
		return qwen38DraftEvaluation{eligible: true}
	}
}

type qwen38DraftSelection struct {
	chosenSource    DraftSource
	engine          Qwen38MTPEngine
	downgradeReason Qwen38MTPDowngradeReason
	stackedSources  []DraftSource
	rejected        []RejectedDraftSource
}

func resolveQwen38DraftSelection(evals map[DraftSource]qwen38DraftEvaluation, allSources []DraftSource, allowStacked, explicitStackAuth bool) qwen38DraftSelection {
	sel := qwen38DraftSelection{chosenSource: DraftSourceNone, engine: Qwen38EngineTargetDecode}

	var eligibleSources []DraftSource
	for _, src := range allSources {
		if evals[src].eligible {
			eligibleSources = append(eligibleSources, src)
		}
	}

	if len(eligibleSources) == 0 {
		sel.downgradeReason = Qwen38MTPQualityOutsideEnvelope
		for _, src := range allSources {
			if evals[src].rejection != nil {
				sel.rejected = append(sel.rejected, *evals[src].rejection)
				if src == DraftSourceMTP && evals[src].rejection.Reason == DraftRejectionLowAcceptance {
					sel.downgradeReason = Qwen38MTPQualityOutsideEnvelope
				} else if src == DraftSourceMTP && evals[src].rejection.Reason == DraftRejectionUnsupportedEnvelope {
					sel.downgradeReason = Qwen38MTPModelUnsupported
				}
			}
		}
		return sel
	}

	canStack := allowStacked && explicitStackAuth
	if canStack && evals[DraftSourcePromptLookup].eligible && evals[DraftSourceNGram].eligible && !evals[DraftSourceMTP].eligible {
		// Compatible lexical stacking explicitly authorized.
		sel.chosenSource = DraftSourcePromptLookup
		sel.stackedSources = []DraftSource{DraftSourcePromptLookup, DraftSourceNGram}
		if evals[DraftSourceMTP].rejection != nil {
			sel.rejected = append(sel.rejected, *evals[DraftSourceMTP].rejection)
		}
		return sel
	}

	// Mutual exclusion priority ladder: MTP (neural model-native), then
	// PromptLookup (retrieval-assisted), then NGram (statistical lexical).
	if evals[DraftSourceMTP].eligible {
		sel.chosenSource = DraftSourceMTP
		sel.engine = Qwen38EngineMTP
	} else if evals[DraftSourcePromptLookup].eligible {
		sel.chosenSource = DraftSourcePromptLookup
	} else if evals[DraftSourceNGram].eligible {
		sel.chosenSource = DraftSourceNGram
	}

	for _, src := range allSources {
		if src == sel.chosenSource {
			continue
		}
		if evals[src].eligible {
			reason := DraftRejectionMutualExclusion
			detail := fmt.Sprintf("%s selected as primary; mutual exclusion forbids concurrent drafting", sel.chosenSource)
			if allowStacked && !explicitStackAuth {
				reason = DraftRejectionIncompatibleStacking
				detail = "stacking disallowed without explicit stacking authorization"
			}
			sel.rejected = append(sel.rejected, RejectedDraftSource{
				Source: src,
				Reason: reason,
				Detail: detail,
			})
		} else if evals[src].rejection != nil {
			sel.rejected = append(sel.rejected, *evals[src].rejection)
		}
	}
	return sel
}
