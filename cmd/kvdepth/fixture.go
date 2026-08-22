package main

import (
	"fmt"
	"math"
	"sort"
)

type SelfcheckAssertions struct {
	KnownDeepestReusablePrefix int  `json:"known_deepest_reusable_prefix_tokens"`
	KnownCliffUpperDepth       int  `json:"known_cliff_unreliable_at_tokens"`
	RecoveryAfterPressure      bool `json:"recovery_after_pressure"`
	MissingEvidenceUnknown     bool `json:"missing_evidence_boundary_unknown"`
	ObservedEnvelopeComplete   bool `json:"observed_envelope_complete"`
}

type SelfcheckResult struct {
	Schema        string              `json:"schema"`
	CampaignID    string              `json:"campaign_id"`
	Assertions    SelfcheckAssertions `json:"assertions"`
	KnownCliff    DepthReport         `json:"known_cliff"`
	MissingKVData DepthReport         `json:"missing_backend_evidence"`
}

type fixtureArm struct {
	ID          string
	Depth       int
	Suffix      string
	Turns       int
	Concurrency int
	Phase       string
}

func BuildSelfcheck(m Manifest) (SelfcheckResult, error) {
	knownObservations, err := SyntheticObservations(m, true)
	if err != nil {
		return SelfcheckResult{}, err
	}
	known, err := Analyze(m, knownObservations)
	if err != nil {
		return SelfcheckResult{}, fmt.Errorf("known cliff: %w", err)
	}
	missingObservations, err := SyntheticObservations(m, false)
	if err != nil {
		return SelfcheckResult{}, err
	}
	missing, err := Analyze(m, missingObservations)
	if err != nil {
		return SelfcheckResult{}, fmt.Errorf("missing evidence: %w", err)
	}
	if err := validateSelfcheckReports(known, missing); err != nil {
		return SelfcheckResult{}, err
	}
	return SelfcheckResult{
		Schema: selfcheckSchema, CampaignID: m.CampaignID,
		Assertions: SelfcheckAssertions{
			KnownDeepestReusablePrefix: 8192, KnownCliffUpperDepth: 12288,
			RecoveryAfterPressure: true, MissingEvidenceUnknown: true,
			ObservedEnvelopeComplete: true,
		},
		KnownCliff: known, MissingKVData: missing,
	}, nil
}

func validateSelfcheckReports(known, missing DepthReport) error {
	if known.Boundary.Status != "known" || known.Boundary.DeepestReliablePrefixTokens == nil || *known.Boundary.DeepestReliablePrefixTokens != 8192 || known.Boundary.Cliff == nil || known.Boundary.Cliff.UnreliableAtTokens != 12288 {
		return fmt.Errorf("known fixture did not recover the 8k/12k cliff: %+v; recovery: restore the synthetic reuse curve or analyzer threshold so the checked-in fixture yields the declared 8192..12288 boundary", known.Boundary)
	}
	if known.PressureRecovery.Status != "recovered" {
		return fmt.Errorf("known fixture did not recover after pressure: %+v; recovery: restore the pressure and recovery arms so reuse returns above the reliable threshold after pressure is removed", known.PressureRecovery)
	}
	if missing.Boundary.Status != "unknown" || missing.Boundary.DeepestReliablePrefixTokens != nil || missing.PressureRecovery.Status != "unknown" {
		return fmt.Errorf("missing metrics invented a boundary: boundary=%+v recovery=%+v; recovery: keep unsupported backend cache evidence absent and report the boundary and recovery as unknown", missing.Boundary, missing.PressureRecovery)
	}
	if !known.Envelope.Complete || !missing.Envelope.Complete {
		return fmt.Errorf("synthetic observed envelope incomplete: known=%+v missing=%+v; recovery: regenerate both synthetic fixtures across every declared campaign axis and counterbalanced order", known.Envelope, missing.Envelope)
	}
	return nil
}

// SyntheticObservations supplies two deterministic fixtures over the declared
// campaign envelope. includeKV=false removes the whole backend evidence object.
func SyntheticObservations(m Manifest, includeKV bool) ([]Observation, error) {
	if err := m.Validate(); err != nil {
		return nil, err
	}
	if !slicesContains(m.Axes.PrefixDepthTokens, 8192) || !slicesContains(m.Axes.PrefixDepthTokens, 12288) {
		return nil, fmt.Errorf("selfcheck manifest must declare 8192 and 12288 token depths; recovery: add both depths to axes.prefix_depth_tokens while preserving the six-depth ascending envelope")
	}
	reference := m.ReferenceArm
	arms := make([]fixtureArm, 0, len(m.Axes.PrefixDepthTokens)*len(m.Axes.SuffixPatterns)+4)
	for _, depth := range m.Axes.PrefixDepthTokens {
		for _, suffix := range m.Axes.SuffixPatterns {
			arms = append(arms, fixtureArm{
				ID: fmt.Sprintf("depth-%d-%s", depth, suffix.ID), Depth: depth, Suffix: suffix.ID,
				Turns: reference.TurnCount, Concurrency: reference.Concurrency, Phase: reference.PressurePhase,
			})
		}
	}
	primarySuffix := m.Axes.SuffixPatterns[0].ID
	arms = append(arms,
		fixtureArm{ID: "turn-variation", Depth: 8192, Suffix: primarySuffix, Turns: alternate(m.Axes.TurnCounts, reference.TurnCount), Concurrency: reference.Concurrency, Phase: "baseline"},
		fixtureArm{ID: "concurrency-variation", Depth: 8192, Suffix: primarySuffix, Turns: reference.TurnCount, Concurrency: alternate(m.Axes.Concurrency, reference.Concurrency), Phase: "baseline"},
		fixtureArm{ID: "capacity-pressure", Depth: 8192, Suffix: primarySuffix, Turns: reference.TurnCount, Concurrency: alternate(m.Axes.Concurrency, reference.Concurrency), Phase: "pressure"},
		fixtureArm{ID: "capacity-recovery", Depth: 8192, Suffix: primarySuffix, Turns: reference.TurnCount, Concurrency: alternate(m.Axes.Concurrency, reference.Concurrency), Phase: "recovery"},
	)

	observations := make([]Observation, 0, len(arms)*m.Axes.Repetitions*2)
	for _, arm := range arms {
		for repetition := 1; repetition <= m.Axes.Repetitions; repetition++ {
			cold, warm := syntheticPair(m, arm, repetition, includeKV)
			if repetition%2 == 1 {
				observations = append(observations, cold, warm)
			} else {
				observations = append(observations, warm, cold)
			}
		}
	}
	return observations, nil
}

func syntheticPair(m Manifest, arm fixtureArm, repetition int, includeKV bool) (Observation, Observation) {
	cached := syntheticReusedTokens(arm.Depth, arm.Phase, repetition)
	promptTokens := int64(arm.Depth + 2048 + arm.Turns*64)
	pressureTax := 0.0
	if arm.Phase == "pressure" {
		pressureTax = 80
	}
	coldTTFT := 520 + float64(arm.Depth)/18 + float64(arm.Concurrency*12+arm.Turns*3) + pressureTax + float64(repetition)
	warmTTFT := coldTTFT - float64(cached)/20
	if warmTTFT < 1 {
		warmTTFT = 1
	}
	semanticEqual := suffixChurn(m, arm.Suffix) == 0
	cold := baseSyntheticObservation(m, arm, repetition, "cold", promptTokens, coldTTFT, semanticEqual)
	warm := baseSyntheticObservation(m, arm, repetition, "warm", promptTokens, warmTTFT, semanticEqual)
	if repetition%2 == 1 {
		cold.OrderIndex, warm.OrderIndex = 1, 2
	} else {
		warm.OrderIndex, cold.OrderIndex = 1, 2
	}
	if includeKV {
		cold.KV = syntheticKV(arm, 0, false)
		warm.KV = syntheticKV(arm, cached, cached > 0)
	}
	return cold, warm
}

func baseSyntheticObservation(m Manifest, arm fixtureArm, repetition int, thermal string, promptTokens int64, ttft float64, semanticEqual bool) Observation {
	reset := m.Reset.BeforeWarmArm
	if thermal == "cold" {
		reset = m.Reset.BeforeColdArm
	}
	return Observation{
		Schema: observationSchema, CampaignID: m.CampaignID,
		RequestID: fmt.Sprintf("%s-r%d-%s", arm.ID, repetition, thermal), ArmID: arm.ID,
		PrefixDepthTokens: arm.Depth, SuffixPattern: arm.Suffix, TurnCount: arm.Turns,
		Concurrency: arm.Concurrency, PressurePhase: arm.Phase, Repetition: repetition,
		ThermalState: thermal, ResetProcedure: reset, Pins: m.Pins, Tokenization: m.Tokenization,
		PromptTokens: promptTokens, TTFTMillis: round6(ttft), UsefulWorkCompleted: true,
		SemanticPromptEqual: semanticEqual, TokenPrefixEqual: true,
	}
}

func syntheticKV(arm fixtureArm, cached int64, admitted bool) *KVSignals {
	resident := cached
	occupancy := math.Min(1, float64(arm.Depth)/16384)
	evictions, preemptions := int64(0), int64(0)
	if arm.Phase == "pressure" {
		occupancy, evictions, preemptions = .99, 3, 1
	}
	return &KVSignals{
		Admitted: &admitted, CachedInputTokens: &cached, ResidentTokens: &resident,
		OccupancyRatio: &occupancy, Evictions: &evictions, Preemptions: &preemptions,
	}
}

func syntheticReusedTokens(depth int, phase string, repetition int) int64 {
	jitter := int64((repetition - 2) * 8)
	switch phase {
	case "pressure":
		return 256 + jitter
	case "recovery":
		return 7782 + jitter
	}
	switch {
	case depth <= 8192:
		return int64(math.Round(float64(depth)*.95)) + jitter
	case depth == 12288:
		return 512 + jitter
	default:
		return max(0, jitter)
	}
}

func suffixChurn(m Manifest, id string) float64 {
	for _, pattern := range m.Axes.SuffixPatterns {
		if pattern.ID == id {
			return pattern.ChurnFraction
		}
	}
	return 1
}

func alternate(values []int, reference int) int {
	copyValues := append([]int(nil), values...)
	sort.Ints(copyValues)
	for _, value := range copyValues {
		if value != reference {
			return value
		}
	}
	return reference
}

func round6(value float64) float64 { return math.Round(value*1e6) / 1e6 }
