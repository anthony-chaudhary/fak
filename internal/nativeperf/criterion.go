package nativeperf

import (
	cryptosha256 "crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
)

const ComparisonCriterionSchema = "fak-native-performance-comparison-criterion/v1"

// ComparisonIdentity binds a receipt to the preregistered criterion selected by
// its immutable run axes. The digest changes whenever the criterion changes.
type ComparisonIdentity struct {
	OutputClass      string `json:"output_class"`
	CriterionID      string `json:"criterion_id"`
	CriterionVersion int    `json:"criterion_version"`
	CriterionDigest  string `json:"criterion_digest"`
}

// ComparisonCriterion is the complete comparator policy selected by a run.
type ComparisonCriterion struct {
	Schema                 string  `json:"schema"`
	ID                     string  `json:"id"`
	Version                int     `json:"version"`
	OutputClass            string  `json:"output_class"`
	MinimumRepetitions     int     `json:"minimum_repetitions"`
	MaximumNoisePercent    float64 `json:"maximum_noise_percent"`
	InvestigateDropPercent float64 `json:"investigate_drop_percent"`
	RegressionDropPercent  float64 `json:"regression_drop_percent"`
	MinimumThroughput      float64 `json:"minimum_throughput_tokens_per_second"`
	QualityMetric          string  `json:"quality_metric"`
	MinimumQualityScore    float64 `json:"minimum_quality_score"`
	QualityHigherIsBetter  bool    `json:"quality_higher_is_better"`
}

// ResolveComparisonCriterion derives the preregistered comparator from receipt
// identity. Callers cannot supply thresholds through this path.
func ResolveComparisonCriterion(r ExperimentReceipt) (ComparisonCriterion, error) {
	criterion := ComparisonCriterion{
		Schema: ComparisonCriterionSchema, ID: "exact-match-native-throughput", Version: 1,
		OutputClass: "exact_match", MinimumRepetitions: 3, MaximumNoisePercent: 2,
		InvestigateDropPercent: 2, RegressionDropPercent: 5, MinimumThroughput: 90,
		QualityMetric: "exact_match", MinimumQualityScore: 1, QualityHigherIsBetter: true,
	}
	_, envelope, err := findLeverEnvelope(ActiveGraph(), r.ChangedLeverID)
	if err != nil || envelope.ID != r.EnvelopeID {
		return ComparisonCriterion{}, fmt.Errorf("comparison criterion is not registered for these run axes")
	}
	if r.ArtifactSHA256 != envelope.ArtifactSHA256 || r.Controls.PromptTokens != envelope.PromptTokens || r.Controls.DecodeTokens != envelope.DecodeTokens || r.Controls.Batch != 1 || r.Controls.ContextTokens != envelope.PromptTokens+envelope.DecodeTokens || r.Controls.Temperature != float64(envelope.Temperature) || r.Controls.Sampling != "greedy" || r.Controls.CacheState != "cold" || r.Controls.Warmups != 1 || r.Controls.Repetitions != envelope.Repetitions || r.Execution.Engine != envelope.Engine || r.Execution.ForwardPath != envelope.ForwardPath || r.Execution.FallbackCount != 0 || r.Quality.Name != criterion.OutputClass {
		return ComparisonCriterion{}, fmt.Errorf("comparison criterion is not registered: an immutable or undeclared control axis drifted")
	}
	if err := validateComparisonCriterion(criterion); err != nil {
		return ComparisonCriterion{}, err
	}
	return criterion, nil
}

func comparisonCriterionDigest(c ComparisonCriterion) (string, error) {
	if err := validateComparisonCriterion(c); err != nil {
		return "", err
	}
	data, err := json.Marshal(c)
	if err != nil {
		return "", fmt.Errorf("encode comparison criterion: %w", err)
	}
	sum := cryptosha256.Sum256(data)
	return hex.EncodeToString(sum[:]), nil
}

func comparisonIdentity(c ComparisonCriterion) (ComparisonIdentity, error) {
	digest, err := comparisonCriterionDigest(c)
	if err != nil {
		return ComparisonIdentity{}, err
	}
	return ComparisonIdentity{OutputClass: c.OutputClass, CriterionID: c.ID, CriterionVersion: c.Version, CriterionDigest: digest}, nil
}

func validateComparisonIdentity(r ExperimentReceipt) error {
	criterion, err := ResolveComparisonCriterion(r)
	if err != nil {
		return err
	}
	want, err := comparisonIdentity(criterion)
	if err != nil {
		return err
	}
	if r.Comparison != want {
		return fmt.Errorf("comparison identity does not match the criterion resolved from run identity")
	}
	return nil
}

func validateComparisonCriterion(c ComparisonCriterion) error {
	finite := func(v float64) bool { return !math.IsNaN(v) && !math.IsInf(v, 0) }
	if c.Schema != ComparisonCriterionSchema || c.ID == "" || c.Version <= 0 || c.OutputClass == "" || c.MinimumRepetitions < 2 || !finite(c.MaximumNoisePercent) || c.MaximumNoisePercent < 0 || !finite(c.InvestigateDropPercent) || c.InvestigateDropPercent < 0 || !finite(c.RegressionDropPercent) || c.RegressionDropPercent <= c.InvestigateDropPercent || !finite(c.MinimumThroughput) || c.MinimumThroughput <= 0 || c.QualityMetric == "" || !finite(c.MinimumQualityScore) {
		return fmt.Errorf("invalid comparison criterion")
	}
	return nil
}

func validatePolicyCriterion(p GatePolicy, c ComparisonCriterion) error {
	if p.MinimumRepetitions != c.MinimumRepetitions || p.MaximumNoisePercent != c.MaximumNoisePercent || p.InvestigateDropPercent != c.InvestigateDropPercent || p.RegressionDropPercent != c.RegressionDropPercent || p.MinimumThroughput != c.MinimumThroughput || p.QualityMetric != c.QualityMetric || p.MinimumQualityScore != c.MinimumQualityScore || p.QualityHigherIsBetter != c.QualityHigherIsBetter {
		return fmt.Errorf("gate policy bounds are absent, stale, or do not match the criterion resolved from run identity")
	}
	return nil
}
