package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
)

type FloatStats struct {
	Samples  int     `json:"samples"`
	Mean     float64 `json:"mean"`
	CI95Low  float64 `json:"ci95_low"`
	CI95High float64 `json:"ci95_high"`
}

type DepthPoint struct {
	PrefixDepthTokens          int         `json:"prefix_depth_tokens"`
	Pairs                      int         `json:"pairs"`
	TTFTSavedMillis            FloatStats  `json:"ttft_saved_ms"`
	MarginalTTFTSavedMillis    *float64    `json:"marginal_ttft_saved_ms,omitempty"`
	PrefillSavedTokens         *FloatStats `json:"prefill_saved_tokens,omitempty"`
	MarginalPrefillSavedTokens *float64    `json:"marginal_prefill_saved_tokens,omitempty"`
	ReuseRatio                 *FloatStats `json:"reuse_ratio,omitempty"`
	OccupancyRatio             *FloatStats `json:"occupancy_ratio,omitempty"`
	KVEvidenceSamples          int         `json:"kv_signal_samples"`
	Evictions                  int64       `json:"evictions"`
	Preemptions                int64       `json:"preemptions"`
}

type CliffInterval struct {
	ReliableThroughTokens int `json:"reliable_through_tokens"`
	UnreliableAtTokens    int `json:"unreliable_at_tokens"`
}

type BoundaryFinding struct {
	Status                      string         `json:"status"`
	DeepestReliablePrefixTokens *int           `json:"deepest_reliable_prefix_tokens,omitempty"`
	Cliff                       *CliffInterval `json:"cliff,omitempty"`
	Reason                      string         `json:"reason,omitempty"`
}

type RecoveryFinding struct {
	Status             string      `json:"status"`
	PrefixDepthTokens  int         `json:"prefix_depth_tokens,omitempty"`
	PressureReuseRatio *FloatStats `json:"pressure_reuse_ratio,omitempty"`
	RecoveryReuseRatio *FloatStats `json:"recovery_reuse_ratio,omitempty"`
	Evictions          int64       `json:"evictions"`
	Preemptions        int64       `json:"preemptions"`
	Reason             string      `json:"reason,omitempty"`
}

type EnvelopeFinding struct {
	PrefixDepths             int  `json:"prefix_depths"`
	SuffixPatterns           int  `json:"suffix_patterns"`
	TurnCounts               int  `json:"turn_counts"`
	ConcurrencyValues        int  `json:"concurrency_values"`
	PressurePhases           int  `json:"pressure_phases"`
	MinimumRepetitionsPerArm int  `json:"minimum_repetitions_per_arm"`
	Counterbalanced          bool `json:"counterbalanced"`
	Complete                 bool `json:"complete"`
}

type EvidenceDimensions struct {
	WarmRequests            int `json:"warm_requests"`
	SemanticPromptEqual     int `json:"semantic_prompt_equal"`
	TokenPrefixEqual        int `json:"token_prefix_equal"`
	BackendStatusPresent    int `json:"backend_status_present"`
	Admitted                int `json:"backend_admitted"`
	ReuseMeasurementPresent int `json:"reuse_measurement_present"`
	ObservedReusePositive   int `json:"observed_reuse_positive"`
	UsefulWorkCompleted     int `json:"useful_work_completed"`
	UsefulWorkRequests      int `json:"useful_work_requests"`
}

type DepthReport struct {
	Schema           string             `json:"schema"`
	CampaignID       string             `json:"campaign_id"`
	ManifestSHA256   string             `json:"manifest_sha256"`
	Pins             Pins               `json:"pins"`
	Tokenization     Tokenization       `json:"tokenization"`
	Confidence       Confidence         `json:"confidence"`
	Envelope         EnvelopeFinding    `json:"observed_envelope"`
	Evidence         EvidenceDimensions `json:"evidence_dimensions"`
	DepthCurve       []DepthPoint       `json:"depth_curve"`
	Boundary         BoundaryFinding    `json:"reusable_prefix_boundary"`
	PressureRecovery RecoveryFinding    `json:"pressure_recovery"`
}

type pairedObservation struct {
	Cold Observation
	Warm Observation
}

func Analyze(m Manifest, observations []Observation) (DepthReport, error) {
	report := DepthReport{
		Schema: depthReportSchema, CampaignID: m.CampaignID, Pins: m.Pins,
		Tokenization: m.Tokenization, Confidence: m.Confidence,
	}
	if err := m.Validate(); err != nil {
		return report, err
	}
	report.ManifestSHA256 = manifestDigest(m)
	pairs, envelope, evidence, err := pairAndValidate(m, observations)
	if err != nil {
		return report, err
	}
	report.Envelope, report.Evidence = envelope, evidence
	report.DepthCurve = foldDepthCurve(m, pairs)
	report.Boundary = findBoundary(m, report.DepthCurve)
	report.PressureRecovery = findRecovery(m, pairs, report.Boundary)
	return report, nil
}

func pairAndValidate(m Manifest, observations []Observation) ([]pairedObservation, EnvelopeFinding, EvidenceDimensions, error) {
	if len(observations) == 0 {
		return nil, EnvelopeFinding{}, EvidenceDimensions{}, errors.New("at least one observation is required; recovery: supply request-level cold and warm observations for the declared campaign arms")
	}
	depths, patterns, turns, concurrencies, phases := map[int]bool{}, map[string]bool{}, map[int]bool{}, map[int]bool{}, map[string]bool{}
	byPair := map[string]map[string]Observation{}
	armRepetitions := map[string]map[int]bool{}
	evidence := EvidenceDimensions{}
	for i, observation := range observations {
		if err := validateObservation(m, observation); err != nil {
			return nil, EnvelopeFinding{}, EvidenceDimensions{}, fmt.Errorf("observation %d: %w", i+1, err)
		}
		depths[observation.PrefixDepthTokens], patterns[observation.SuffixPattern] = true, true
		turns[observation.TurnCount], concurrencies[observation.Concurrency], phases[observation.PressurePhase] = true, true, true
		armKey := observationArmKey(observation)
		if armRepetitions[armKey] == nil {
			armRepetitions[armKey] = map[int]bool{}
		}
		armRepetitions[armKey][observation.Repetition] = true
		pairKey := armKey + "/rep=" + strconv.Itoa(observation.Repetition)
		if byPair[pairKey] == nil {
			byPair[pairKey] = map[string]Observation{}
		}
		if _, exists := byPair[pairKey][observation.ThermalState]; exists {
			return nil, EnvelopeFinding{}, EvidenceDimensions{}, fmt.Errorf("duplicate %s request for pair %s; recovery: keep exactly one cold and one warm request per arm and repetition", observation.ThermalState, pairKey)
		}
		byPair[pairKey][observation.ThermalState] = observation
		evidence.UsefulWorkRequests++
		if observation.UsefulWorkCompleted {
			evidence.UsefulWorkCompleted++
		}
	}

	minimumRepetitions := m.Axes.Repetitions
	for _, repetitions := range armRepetitions {
		if len(repetitions) < minimumRepetitions {
			minimumRepetitions = len(repetitions)
		}
	}
	pairKeys := make([]string, 0, len(byPair))
	for key := range byPair {
		pairKeys = append(pairKeys, key)
	}
	sort.Strings(pairKeys)
	pairs := make([]pairedObservation, 0, len(pairKeys))
	counterbalanced := true
	for _, key := range pairKeys {
		states := byPair[key]
		cold, coldOK := states["cold"]
		warm, warmOK := states["warm"]
		if !coldOK || !warmOK {
			return nil, EnvelopeFinding{}, EvidenceDimensions{}, fmt.Errorf("pair %s requires one cold and one warm request; recovery: add the missing thermal request or remove the incomplete pair", key)
		}
		if cold.Repetition%2 == 1 {
			counterbalanced = counterbalanced && cold.OrderIndex == 1 && warm.OrderIndex == 2
		} else {
			counterbalanced = counterbalanced && warm.OrderIndex == 1 && cold.OrderIndex == 2
		}
		countEvidence(&evidence, warm)
		pairs = append(pairs, pairedObservation{Cold: cold, Warm: warm})
	}
	envelope := EnvelopeFinding{
		PrefixDepths: len(depths), SuffixPatterns: len(patterns), TurnCounts: len(turns),
		ConcurrencyValues: len(concurrencies), PressurePhases: len(phases),
		MinimumRepetitionsPerArm: minimumRepetitions, Counterbalanced: counterbalanced,
	}
	envelope.Complete = envelope.PrefixDepths >= len(m.Axes.PrefixDepthTokens) && envelope.SuffixPatterns >= len(m.Axes.SuffixPatterns) &&
		envelope.TurnCounts >= len(m.Axes.TurnCounts) && envelope.ConcurrencyValues >= len(m.Axes.Concurrency) &&
		envelope.PressurePhases >= 3 && minimumRepetitions >= m.Axes.Repetitions && counterbalanced
	if !envelope.Complete {
		return nil, envelope, evidence, fmt.Errorf("observed envelope incomplete: %+v; recovery: cover every declared depth, suffix, turn count, concurrency, pressure phase, repetition, and counterbalanced request order", envelope)
	}
	return pairs, envelope, evidence, nil
}

func validateObservation(m Manifest, observation Observation) error {
	if observation.Schema != observationSchema || observation.CampaignID != m.CampaignID || observation.RequestID == "" || observation.ArmID == "" {
		return errors.New("schema, campaign_id, request_id, and arm_id must match the campaign; recovery: use the observation schema, copy campaign_id from the manifest, and set non-empty request_id and arm_id values")
	}
	if observation.Pins != m.Pins || observation.Tokenization != m.Tokenization {
		return errors.New("model/runtime/fak/tokenization pins differ from the manifest; recovery: copy the exact model, runtime, fak, and tokenization pins from the campaign manifest")
	}
	if !slicesContains(m.Axes.PrefixDepthTokens, observation.PrefixDepthTokens) || !suffixDeclared(m, observation.SuffixPattern) ||
		!slicesContains(m.Axes.TurnCounts, observation.TurnCount) || !slicesContains(m.Axes.Concurrency, observation.Concurrency) || !phaseDeclared(m, observation.PressurePhase) {
		return errors.New("observation coordinates are outside the declared campaign axes; recovery: choose prefix depth, suffix pattern, turn count, concurrency, and pressure phase values declared by the manifest")
	}
	if observation.Repetition < 1 || observation.Repetition > m.Axes.Repetitions || !oneOf(observation.ThermalState, "cold", "warm") || observation.OrderIndex < 1 || observation.OrderIndex > 2 {
		return errors.New("invalid repetition, thermal_state, or order_index; recovery: use a positive repetition, thermal_state cold or warm, and order_index 1 or 2")
	}
	if observation.PromptTokens < int64(observation.PrefixDepthTokens) || observation.TTFTMillis <= 0 {
		return errors.New("prompt_tokens must cover the prefix and ttft_ms must be positive; recovery: record prompt_tokens at least as large as prefix_depth_tokens and a positive ttft_ms")
	}
	if !oneOf(observation.ResetProcedure, m.Reset.BeforeColdArm, m.Reset.BeforeWarmArm, m.Reset.AfterPressure) {
		return errors.New("reset_procedure is not declared by the manifest; recovery: use the manifest's before_cold_arm procedure for cold requests and before_warm_arm procedure for warm requests")
	}
	if observation.KV != nil {
		if invalidNonnegative(observation.KV.CachedInputTokens) || invalidNonnegative(observation.KV.ResidentTokens) || invalidNonnegative(observation.KV.Evictions) || invalidNonnegative(observation.KV.Preemptions) {
			return errors.New("backend cache token/counter fields must be non-negative when present; recovery: omit unsupported backend fields or record non-negative token and counter values")
		}
		if observation.KV.CachedInputTokens != nil && *observation.KV.CachedInputTokens > observation.PromptTokens {
			return errors.New("cached_input_tokens cannot exceed prompt_tokens; recovery: correct the backend reuse count so it does not exceed the request's prompt_tokens")
		}
		if observation.KV.OccupancyRatio != nil && (*observation.KV.OccupancyRatio < 0 || *observation.KV.OccupancyRatio > 1) {
			return errors.New("occupancy_ratio must be in [0,1]; recovery: record occupancy_ratio from 0 through 1 or omit it when unavailable")
		}
	}
	return nil
}

func foldDepthCurve(m Manifest, pairs []pairedObservation) []DepthPoint {
	byDepth := map[int][]pairedObservation{}
	for _, pair := range pairs {
		warm := pair.Warm
		if warm.TurnCount == m.ReferenceArm.TurnCount && warm.Concurrency == m.ReferenceArm.Concurrency && warm.PressurePhase == m.ReferenceArm.PressurePhase {
			byDepth[warm.PrefixDepthTokens] = append(byDepth[warm.PrefixDepthTokens], pair)
		}
	}
	points := make([]DepthPoint, 0, len(m.Axes.PrefixDepthTokens))
	var previousTTFT, previousPrefill *float64
	for _, depth := range m.Axes.PrefixDepthTokens {
		pairsAtDepth := byDepth[depth]
		ttftSaved, prefillSaved, ratios, occupancy := []float64{}, []float64{}, []float64{}, []float64{}
		point := DepthPoint{PrefixDepthTokens: depth, Pairs: len(pairsAtDepth)}
		for _, pair := range pairsAtDepth {
			ttftSaved = append(ttftSaved, pair.Cold.TTFTMillis-pair.Warm.TTFTMillis)
			if pair.Warm.KV == nil || pair.Warm.KV.CachedInputTokens == nil {
				continue
			}
			cached := float64(min(*pair.Warm.KV.CachedInputTokens, int64(depth)))
			prefillSaved = append(prefillSaved, cached)
			ratios = append(ratios, cached/float64(depth))
			point.KVEvidenceSamples++
			if pair.Warm.KV.OccupancyRatio != nil {
				occupancy = append(occupancy, *pair.Warm.KV.OccupancyRatio)
			}
			point.Evictions += valueOrZero(pair.Warm.KV.Evictions)
			point.Preemptions += valueOrZero(pair.Warm.KV.Preemptions)
		}
		point.TTFTSavedMillis = stats(ttftSaved)
		if previousTTFT != nil {
			marginal := point.TTFTSavedMillis.Mean - *previousTTFT
			point.MarginalTTFTSavedMillis = &marginal
		}
		previousTTFT = &point.TTFTSavedMillis.Mean
		if len(prefillSaved) > 0 {
			prefill, ratio := stats(prefillSaved), stats(ratios)
			point.PrefillSavedTokens, point.ReuseRatio = &prefill, &ratio
			if previousPrefill != nil {
				marginal := prefill.Mean - *previousPrefill
				point.MarginalPrefillSavedTokens = &marginal
			}
			previousPrefill = &prefill.Mean
		}
		if len(occupancy) > 0 {
			occupancyStats := stats(occupancy)
			point.OccupancyRatio = &occupancyStats
		}
		points = append(points, point)
	}
	return points
}

func findBoundary(m Manifest, curve []DepthPoint) BoundaryFinding {
	finding := BoundaryFinding{Status: "unknown", Reason: "backend cached-input evidence is unavailable"}
	for _, point := range curve {
		if point.ReuseRatio == nil || point.ReuseRatio.Samples < m.Confidence.MinimumSamples {
			return finding
		}
	}
	var deepest *int
	for i := range curve {
		point := curve[i]
		if point.ReuseRatio.CI95Low >= m.Confidence.ReliableReuseRatio {
			value := point.PrefixDepthTokens
			deepest = &value
			continue
		}
		if deepest != nil && point.ReuseRatio.CI95High < m.Confidence.ReliableReuseRatio {
			return BoundaryFinding{
				Status: "known", DeepestReliablePrefixTokens: deepest,
				Cliff: &CliffInterval{ReliableThroughTokens: *deepest, UnreliableAtTokens: point.PrefixDepthTokens},
			}
		}
	}
	if deepest != nil {
		return BoundaryFinding{Status: "no_cliff_observed", DeepestReliablePrefixTokens: deepest, Reason: "all measured depths clear the reuse threshold"}
	}
	return BoundaryFinding{Status: "known", Reason: "no measured depth clears the reuse threshold"}
}

func findRecovery(m Manifest, pairs []pairedObservation, boundary BoundaryFinding) RecoveryFinding {
	if boundary.DeepestReliablePrefixTokens == nil {
		return RecoveryFinding{Status: "unknown", Reason: "reusable prefix boundary is unknown"}
	}
	depth := *boundary.DeepestReliablePrefixTokens
	pressureRatios, recoveryRatios := []float64{}, []float64{}
	finding := RecoveryFinding{Status: "unknown", PrefixDepthTokens: depth}
	for _, pair := range pairs {
		warm := pair.Warm
		if warm.PrefixDepthTokens != depth || warm.KV == nil || warm.KV.CachedInputTokens == nil {
			continue
		}
		ratio := float64(min(*warm.KV.CachedInputTokens, int64(depth))) / float64(depth)
		switch warm.PressurePhase {
		case "pressure":
			pressureRatios = append(pressureRatios, ratio)
			finding.Evictions += valueOrZero(warm.KV.Evictions)
			finding.Preemptions += valueOrZero(warm.KV.Preemptions)
		case "recovery":
			recoveryRatios = append(recoveryRatios, ratio)
		}
	}
	if len(pressureRatios) < m.Confidence.MinimumSamples || len(recoveryRatios) < m.Confidence.MinimumSamples {
		finding.Reason = "pressure and recovery cache evidence require the declared minimum samples"
		return finding
	}
	pressure, recovery := stats(pressureRatios), stats(recoveryRatios)
	finding.PressureReuseRatio, finding.RecoveryReuseRatio = &pressure, &recovery
	if pressure.CI95High < m.Confidence.ReliableReuseRatio && recovery.CI95Low >= m.Confidence.ReliableReuseRatio {
		finding.Status = "recovered"
		return finding
	}
	finding.Status = "not_recovered"
	finding.Reason = "pressure did not create a witnessed cliff or recovery did not clear the reliability threshold"
	return finding
}

func stats(values []float64) FloatStats {
	result := FloatStats{Samples: len(values)}
	if len(values) == 0 {
		return result
	}
	for _, value := range values {
		result.Mean += value
	}
	result.Mean /= float64(len(values))
	if len(values) == 1 {
		result.CI95Low, result.CI95High = result.Mean, result.Mean
		return result
	}
	var squared float64
	for _, value := range values {
		delta := value - result.Mean
		squared += delta * delta
	}
	standardError := math.Sqrt(squared/float64(len(values)-1)) / math.Sqrt(float64(len(values)))
	margin := tCritical95(len(values)-1) * standardError
	result.CI95Low, result.CI95High = result.Mean-margin, result.Mean+margin
	return result
}

func tCritical95(degrees int) float64 {
	values := map[int]float64{1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042}
	if value, ok := values[degrees]; ok {
		return value
	}
	return 1.96
}

func manifestDigest(m Manifest) string {
	body, _ := json.Marshal(m)
	digest := sha256.Sum256(body)
	return "sha256:" + hex.EncodeToString(digest[:])
}

func observationArmKey(observation Observation) string {
	return strings.Join([]string{
		observation.ArmID,
		"depth=" + strconv.Itoa(observation.PrefixDepthTokens),
		"suffix=" + observation.SuffixPattern,
		"turns=" + strconv.Itoa(observation.TurnCount),
		"concurrency=" + strconv.Itoa(observation.Concurrency),
		"pressure=" + observation.PressurePhase,
	}, "/")
}

func countEvidence(result *EvidenceDimensions, warm Observation) {
	result.WarmRequests++
	if warm.SemanticPromptEqual {
		result.SemanticPromptEqual++
	}
	if warm.TokenPrefixEqual {
		result.TokenPrefixEqual++
	}
	if warm.KV == nil {
		return
	}
	if warm.KV.Admitted != nil {
		result.BackendStatusPresent++
		if *warm.KV.Admitted {
			result.Admitted++
		}
	}
	if warm.KV.CachedInputTokens != nil {
		result.ReuseMeasurementPresent++
		if *warm.KV.CachedInputTokens > 0 {
			result.ObservedReusePositive++
		}
	}
}

func suffixDeclared(m Manifest, id string) bool {
	for _, pattern := range m.Axes.SuffixPatterns {
		if pattern.ID == id {
			return true
		}
	}
	return false
}

func phaseDeclared(m Manifest, phase string) bool {
	for _, arm := range m.Axes.PressureArms {
		if arm.Phase == phase {
			return true
		}
	}
	return false
}

func slicesContains(values []int, value int) bool {
	for _, candidate := range values {
		if candidate == value {
			return true
		}
	}
	return false
}

func invalidNonnegative(value *int64) bool { return value != nil && *value < 0 }

func valueOrZero(value *int64) int64 {
	if value == nil {
		return 0
	}
	return *value
}
