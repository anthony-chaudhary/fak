package benchcli

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"math"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dojo"
)

// SHA256Digest returns the canonical digest form used by simulation evidence.
func SHA256Digest(data []byte) string {
	sum := sha256.Sum256(data)
	return "sha256:" + hex.EncodeToString(sum[:])
}

// SimulationEvidenceSchema versions the additive evidence contract embedded in
// BenchmarkArtifact. Large traces and calibration corpora remain external;
// their SHA-256 digests travel in this compact record.
const SimulationEvidenceSchema = "fak-simulation-evidence/1"

// EvidenceType says what produced a benchmark result. The closed vocabulary is
// deliberately more precise than a simulated boolean: each producer class has
// a different maximum claim it can honestly support.
type EvidenceType string

const (
	EvidenceStructuralCount      EvidenceType = "structural_count"
	EvidenceAnalyticalBound      EvidenceType = "analytical_bound"
	EvidenceTraceSimulation      EvidenceType = "trace_sim"
	EvidenceCycleSimulation      EvidenceType = "cycle_sim"
	EvidenceLearnedEstimate      EvidenceType = "learned_estimate"
	EvidenceCalibratedSimulation EvidenceType = "calibrated_sim"
	EvidenceHardwareMeasurement  EvidenceType = "hardware_measurement"
)

// ClaimCeiling is the strongest result language requested by an evidence
// record. It is a ceiling, so a stronger producer may deliberately request a
// weaker claim.
type ClaimCeiling string

const (
	ClaimCorrectnessOnly  ClaimCeiling = "correctness_only"
	ClaimBottleneckOnly   ClaimCeiling = "bottleneck_only"
	ClaimRelativeRank     ClaimCeiling = "relative_rank"
	ClaimAbsoluteEstimate ClaimCeiling = "absolute_estimate"
	ClaimMeasuredAbsolute ClaimCeiling = "measured_absolute"
)

// SimulationEvidence binds a deterministic or measured performance result to
// its maximum honest claim. The block is named for the issue that introduced
// it, but hardware_measurement is intentionally in the same vocabulary so a
// consumer can make one exhaustive, fail-closed decision.
type SimulationEvidence struct {
	Schema           string                 `json:"schema"`
	EvidenceType     EvidenceType           `json:"evidence_type"`
	ClaimCeiling     ClaimCeiling           `json:"claim_ceiling"`
	Engine           SimulationEngine       `json:"engine"`
	Workload         WorkloadProvenance     `json:"workload_provenance"`
	Trace            *TraceProvenance       `json:"trace_provenance,omitempty"`
	ValidityEnvelope ValidityEnvelope       `json:"validity_envelope"`
	ExcludedEffects  []string               `json:"excluded_effects"`
	Replay           ReplaySpec             `json:"replay"`
	LearnedModel     *LearnedModelInfo      `json:"learned_model,omitempty"`
	Calibration      *SimulationCalibration `json:"calibration,omitempty"`
	Cost             SimulationCost         `json:"simulator_cost"`
}

// SimulationEngine identifies the exact producer and configuration. Toolchain
// is mandatory for a calibrated profile because compiler/library drift can
// change code generation without changing the simulator's own revision.
type SimulationEngine struct {
	Name         string `json:"name"`
	Revision     string `json:"revision"`
	ConfigDigest string `json:"config_digest"`
	Toolchain    string `json:"toolchain,omitempty"`
}

type WorkloadProvenance struct {
	Name   string `json:"name"`
	Source string `json:"source"`
	Digest string `json:"digest"`
}

// TraceProvenance carries every input identity needed to decide whether a
// frozen trace remains admissible. InvalidatedBy is itself closed: trace
// ranking must be recaptured when a change can alter control flow, ISA,
// synchronization, allocation, or kernel selection.
type TraceProvenance struct {
	Artifact        string   `json:"artifact"`
	Digest          string   `json:"digest"`
	CaptureGPU      string   `json:"capture_gpu"`
	Compiler        string   `json:"compiler"`
	Libraries       []string `json:"libraries"`
	KernelSelection string   `json:"kernel_selection"`
	Input           string   `json:"input"`
	InputDigest     string   `json:"input_digest"`
	InvalidatedBy   []string `json:"invalidated_by"`
}

const (
	TraceInvalidationControlFlow     = "control_flow"
	TraceInvalidationISA             = "isa"
	TraceInvalidationSynchronization = "synchronization"
	TraceInvalidationAllocation      = "allocation"
	TraceInvalidationKernelChoice    = "kernel_choice"
)

type ValidityEnvelope struct {
	Description string            `json:"description"`
	Dimensions  map[string]string `json:"dimensions"`
}

// ReplaySpec separates deterministic replay from statistically independent
// replication. A fixed seed makes one execution reproducible; it does not make
// a stochastic or tail claim conclusive.
type ReplaySpec struct {
	Seed               *int64 `json:"seed"`
	Stream             string `json:"stream"`
	Repetitions        int    `json:"repetitions"`
	IndependentStreams int    `json:"independent_streams"`
	Stochastic         bool   `json:"stochastic"`
	Uncertainty        string `json:"uncertainty,omitempty"`
}

// LearnedModelInfo binds a learned selector to the data and compilation target
// that define its domain. AbstainOutsideEnvelope is load-bearing: guessing out
// of domain is not an estimate.
type LearnedModelInfo struct {
	TrainingSetDigest      string `json:"training_set_digest"`
	FeatureSchema          string `json:"feature_schema"`
	Compiler               string `json:"compiler"`
	Target                 string `json:"target"`
	AbstainOutsideEnvelope bool   `json:"abstain_outside_envelope"`
}

// SimulationCalibration is a paired simulation/real-hardware calibration row.
// GradeSimulationCalibration derives Residual, NormalizedError, Verdict, and
// Grade through the existing dojo prediction-calibration semantics.
type SimulationCalibration struct {
	Profile               string  `json:"profile"`
	Revision              string  `json:"revision"`
	Digest                string  `json:"digest"`
	Hardware              string  `json:"hardware"`
	HardwareProfile       string  `json:"hardware_profile"`
	HardwareProfileDigest string  `json:"hardware_profile_digest"`
	Date                  string  `json:"date"`
	IndependentHoldout    bool    `json:"independent_holdout"`
	MeasurementSource     string  `json:"measurement_source"`
	Metric                string  `json:"metric"`
	Unit                  string  `json:"unit"`
	Sample                int     `json:"sample"`
	Predicted             float64 `json:"predicted"`
	Measured              float64 `json:"measured"`
	LowerIsBetter         bool    `json:"lower_is_better,omitempty"`
	ErrorBand             float64 `json:"error_band"`
	Residual              float64 `json:"residual"`
	NormalizedError       float64 `json:"normalized_error"`
	Verdict               string  `json:"verdict"`
	Grade                 string  `json:"grade"`

	// Drift identity is copied into the versioned profile and must match the
	// enclosing evidence before an absolute estimate is admissible.
	EngineRevision     string `json:"engine_revision"`
	EngineConfigDigest string `json:"engine_config_digest"`
	Toolchain          string `json:"toolchain"`
	WorkloadDigest     string `json:"workload_digest"`
}

const (
	CalibrationCalibrated   = dojo.VerdictCalibrated
	CalibrationOverClaim    = dojo.VerdictOverClaim
	CalibrationUnderClaim   = dojo.VerdictUnderClaim
	CalibrationInsufficient = "INSUFFICIENT"
)

type SimulationCost struct {
	HostWallTimeMS float64 `json:"host_wall_time_ms"`
	HostCPUTimeMS  float64 `json:"host_cpu_time_ms"`
	Bytes          int64   `json:"bytes"`
}

var claimStrength = map[ClaimCeiling]int{
	ClaimCorrectnessOnly:  0,
	ClaimBottleneckOnly:   1,
	ClaimRelativeRank:     2,
	ClaimAbsoluteEstimate: 3,
	ClaimMeasuredAbsolute: 4,
}

var evidenceMaxClaim = map[EvidenceType]ClaimCeiling{
	EvidenceStructuralCount:      ClaimCorrectnessOnly,
	EvidenceAnalyticalBound:      ClaimBottleneckOnly,
	EvidenceTraceSimulation:      ClaimRelativeRank,
	EvidenceCycleSimulation:      ClaimRelativeRank,
	EvidenceLearnedEstimate:      ClaimRelativeRank,
	EvidenceCalibratedSimulation: ClaimAbsoluteEstimate,
	EvidenceHardwareMeasurement:  ClaimMeasuredAbsolute,
}

// GradeSimulationCalibration applies the shared dojo prediction-calibration
// decision to a held-out simulated-vs-real pair. An absent/non-independent
// measurement remains INSUFFICIENT rather than grading itself CALIBRATED.
func GradeSimulationCalibration(in SimulationCalibration) SimulationCalibration {
	out := in
	if !in.IndependentHoldout || in.Sample < 1 {
		out.Residual = 0
		out.NormalizedError = 0
		out.Verdict = CalibrationInsufficient
		out.Grade = "n/a"
		return out
	}
	band := dojo.DefaultCalibBand()
	if finitePositive(in.ErrorBand) {
		band.CalibratedMax = in.ErrorBand
	}
	episode := dojo.Score("simulation-independent-holdout", dojo.Prediction{
		Lever:         in.Profile,
		Metric:        in.Metric,
		Claimed:       in.Predicted,
		Unit:          in.Unit,
		Basis:         in.Revision,
		LowerIsBetter: in.LowerIsBetter,
	}, dojo.Outcome{
		Realized:   in.Measured,
		Provenance: dojo.Observed,
		Source:     in.MeasurementSource,
		Measured:   true,
		Sample:     in.Sample,
	}, band)
	out.Residual = episode.Residual
	out.NormalizedError = episode.CalibErr
	out.Verdict = episode.Verdict
	out.Grade = episode.Grade
	return out
}

// ValidateSimulationEvidence rejects unknown vocabulary, promotion beyond the
// producer's ceiling, incomplete provenance, and self-certified calibration.
// There is no permissive default: a new enum value must first be added to the
// exhaustive matrix above.
func ValidateSimulationEvidence(ev SimulationEvidence) error {
	var errs []error
	if ev.Schema != SimulationEvidenceSchema {
		errs = append(errs, fmt.Errorf("schema = %q, want %q", ev.Schema, SimulationEvidenceSchema))
	}
	maxClaim, knownEvidence := evidenceMaxClaim[ev.EvidenceType]
	strength, knownClaim := claimStrength[ev.ClaimCeiling]
	if !knownEvidence {
		errs = append(errs, fmt.Errorf("unknown evidence_type %q", ev.EvidenceType))
	}
	if !knownClaim {
		errs = append(errs, fmt.Errorf("unknown claim_ceiling %q", ev.ClaimCeiling))
	}
	if knownEvidence && knownClaim && strength > claimStrength[maxClaim] {
		errs = append(errs, fmt.Errorf("evidence_type %q cannot support claim_ceiling %q (maximum %q)", ev.EvidenceType, ev.ClaimCeiling, maxClaim))
	}

	if unknownIdentity(ev.Engine.Name) || unknownIdentity(ev.Engine.Revision) {
		errs = append(errs, errors.New("engine name and revision must be identifying values, not blank or unknown"))
	}
	if !validSHA256Digest(ev.Engine.ConfigDigest) {
		errs = append(errs, errors.New("engine config_digest must be a sha256 digest"))
	}
	if blank(ev.Workload.Name) || blank(ev.Workload.Source) || !validSHA256Digest(ev.Workload.Digest) {
		errs = append(errs, errors.New("workload provenance requires name, source, and sha256 digest"))
	}
	if blank(ev.ValidityEnvelope.Description) || len(ev.ValidityEnvelope.Dimensions) == 0 {
		errs = append(errs, errors.New("validity_envelope requires a description and dimensions"))
	} else {
		for key, value := range ev.ValidityEnvelope.Dimensions {
			if blank(key) || blank(value) {
				errs = append(errs, errors.New("validity_envelope dimensions cannot contain blank keys or values"))
				break
			}
		}
	}
	if len(ev.ExcludedEffects) == 0 {
		errs = append(errs, errors.New("excluded_effects must explicitly name omissions or contain \"none\""))
	} else if hasBlank(ev.ExcludedEffects) {
		errs = append(errs, errors.New("excluded_effects cannot contain blank entries"))
	}

	if ev.EvidenceType != EvidenceHardwareMeasurement {
		if ev.Replay.Seed == nil || blank(ev.Replay.Stream) || ev.Replay.Repetitions < 1 || ev.Replay.IndependentStreams < 1 {
			errs = append(errs, errors.New("simulated evidence requires seed, stream, repetitions >= 1, and independent_streams >= 1"))
		}
	}
	if ev.Replay.Stochastic {
		errs = append(errs, validateIndependentReplication(ev.Replay)...)
	}
	if !finiteNonnegative(ev.Cost.HostWallTimeMS) || !finiteNonnegative(ev.Cost.HostCPUTimeMS) || ev.Cost.Bytes < 0 {
		errs = append(errs, errors.New("simulator_cost values must be finite and non-negative"))
	}

	if ev.Trace != nil {
		errs = append(errs, validateTrace(*ev.Trace)...)
	} else if ev.EvidenceType == EvidenceTraceSimulation {
		errs = append(errs, errors.New("trace_sim requires trace_provenance"))
	}
	if ev.LearnedModel != nil {
		errs = append(errs, validateLearnedModel(*ev.LearnedModel)...)
	} else if ev.EvidenceType == EvidenceLearnedEstimate {
		errs = append(errs, errors.New("learned_estimate requires learned_model identity"))
	}
	if ev.Calibration != nil {
		errs = append(errs, validateCalibration(ev, *ev.Calibration)...)
	}
	if ev.EvidenceType == EvidenceCalibratedSimulation ||
		(ev.ClaimCeiling == ClaimAbsoluteEstimate && ev.EvidenceType != EvidenceHardwareMeasurement) {
		if ev.Calibration == nil {
			errs = append(errs, errors.New("calibrated_sim and absolute_estimate require a calibration profile"))
		} else if !ev.Calibration.IndependentHoldout || ev.Calibration.Sample < 1 {
			errs = append(errs, errors.New("absolute estimates require an independent held-out hardware measurement"))
		}
	}
	return errors.Join(errs...)
}

// ValidateBenchmarkArtifact validates the optional evidence block and the claim
// shapes only visible in the surrounding result. Legacy artifacts without the
// additive block remain decodable; a consumer that requires the new contract
// decides separately whether absence is admissible.
func ValidateBenchmarkArtifact(art BenchmarkArtifact) error {
	if art.SimulationEvidence == nil {
		return nil
	}
	var errs []error
	if err := ValidateSimulationEvidence(*art.SimulationEvidence); err != nil {
		errs = append(errs, err)
	}
	if artifactHasTailMetric(art) {
		errs = append(errs, validateIndependentReplication(art.SimulationEvidence.Replay)...)
	}
	ev := art.SimulationEvidence
	if ev.EvidenceType == EvidenceStructuralCount {
		if names := matchingMetricNames(art.Results.Metrics, isPerformanceMetric); len(names) > 0 {
			errs = append(errs, fmt.Errorf("structural_count cannot support elapsed-time, latency, throughput, energy, or cost metrics: %s", strings.Join(names, ", ")))
		}
	}
	if ev.EvidenceType != EvidenceHardwareMeasurement {
		if artifactMakesCompetitiveClaim(art) {
			errs = append(errs, errors.New("simulated evidence cannot support a competitive or native-faster claim"))
		}
		if artifactClaimsMeasuredOrAchieved(art) {
			errs = append(errs, errors.New("non-hardware evidence cannot mark a result measured or achieved"))
		}
		performance := matchingMetricNames(art.Results.Metrics, isPerformanceMetric)
		switch ev.EvidenceType {
		case EvidenceAnalyticalBound:
			var unmarked []string
			artifactProjected := artifactIsProjectionMarked(art)
			for _, name := range performance {
				if ev.ClaimCeiling != ClaimBottleneckOnly || (!artifactProjected && !metricIsProjectionMarked(name)) {
					unmarked = append(unmarked, name)
				}
			}
			if len(unmarked) > 0 {
				errs = append(errs, fmt.Errorf("analytical absolute performance bounds require bottleneck_only plus an explicit projected/theoretical marker: %s", strings.Join(unmarked, ", ")))
			}
		case EvidenceCalibratedSimulation:
			if len(performance) > 0 && ev.ClaimCeiling != ClaimAbsoluteEstimate {
				errs = append(errs, fmt.Errorf("calibrated_sim absolute performance metrics require absolute_estimate with valid calibration: %s", strings.Join(performance, ", ")))
			}
		case EvidenceStructuralCount:
			// The structural-specific error above already names the violated
			// mechanism-only ceiling.
		default:
			if len(performance) > 0 {
				errs = append(errs, fmt.Errorf("evidence_type %q cannot emit absolute latency, throughput, energy, or cost metrics, even when marked projected: %s", ev.EvidenceType, strings.Join(performance, ", ")))
			}
		}
	}
	return errors.Join(errs...)
}

func validateTrace(trace TraceProvenance) []error {
	var errs []error
	if blank(trace.Artifact) || !validSHA256Digest(trace.Digest) || blank(trace.CaptureGPU) ||
		blank(trace.Compiler) || len(trace.Libraries) == 0 || hasBlank(trace.Libraries) ||
		blank(trace.KernelSelection) || blank(trace.Input) || !validSHA256Digest(trace.InputDigest) {
		errs = append(errs, errors.New("trace_provenance requires artifact/digest, capture GPU, compiler, libraries, kernel selection, and input/digest"))
	}
	want := map[string]bool{
		TraceInvalidationControlFlow: false, TraceInvalidationISA: false,
		TraceInvalidationSynchronization: false, TraceInvalidationAllocation: false,
		TraceInvalidationKernelChoice: false,
	}
	for _, trigger := range trace.InvalidatedBy {
		if _, ok := want[trigger]; !ok {
			errs = append(errs, fmt.Errorf("unknown trace invalidation trigger %q", trigger))
			continue
		}
		want[trigger] = true
	}
	var missing []string
	for trigger, found := range want {
		if !found {
			missing = append(missing, trigger)
		}
	}
	if len(missing) > 0 {
		sort.Strings(missing)
		errs = append(errs, fmt.Errorf("trace invalidation triggers missing %s", strings.Join(missing, ", ")))
	}
	return errs
}

func validateLearnedModel(model LearnedModelInfo) []error {
	var errs []error
	if !validSHA256Digest(model.TrainingSetDigest) || blank(model.FeatureSchema) || blank(model.Compiler) || blank(model.Target) {
		errs = append(errs, errors.New("learned_model requires training digest, feature schema, compiler, and target"))
	}
	if !model.AbstainOutsideEnvelope {
		errs = append(errs, errors.New("learned_model must abstain outside the validity envelope"))
	}
	return errs
}

func validateCalibration(ev SimulationEvidence, cal SimulationCalibration) []error {
	var errs []error
	if blank(cal.Profile) || blank(cal.Revision) || !validSHA256Digest(cal.Digest) ||
		blank(cal.Hardware) || blank(cal.HardwareProfile) || !validSHA256Digest(cal.HardwareProfileDigest) ||
		!validCalibrationDate(cal.Date) || blank(cal.MeasurementSource) || blank(cal.Metric) || blank(cal.Unit) ||
		!finite(cal.Predicted) || !finite(cal.Measured) || !finitePositive(cal.ErrorBand) {
		errs = append(errs, errors.New("calibration requires versioned profile/digest, dated hardware profile/digest, paired sim-real metric, source, and positive error band"))
	}
	if cal.EngineRevision != ev.Engine.Revision || cal.EngineConfigDigest != ev.Engine.ConfigDigest ||
		cal.Toolchain != ev.Engine.Toolchain || cal.WorkloadDigest != ev.Workload.Digest {
		errs = append(errs, errors.New("calibration drift identity does not match engine revision/config/toolchain and workload digest"))
	}
	if blank(cal.Toolchain) {
		errs = append(errs, errors.New("calibration toolchain identity is required"))
	}
	want := GradeSimulationCalibration(cal)
	if cal.Verdict != want.Verdict || cal.Grade != want.Grade || !nearlyEqual(cal.Residual, want.Residual) || !nearlyEqual(cal.NormalizedError, want.NormalizedError) {
		errs = append(errs, fmt.Errorf("calibration grade mismatch: got %s/%s residual %.9g error %.9g, want %s/%s residual %.9g error %.9g",
			cal.Verdict, cal.Grade, cal.Residual, cal.NormalizedError,
			want.Verdict, want.Grade, want.Residual, want.NormalizedError))
	}
	return errs
}

func validateIndependentReplication(replay ReplaySpec) []error {
	var errs []error
	if replay.Repetitions < 2 || replay.IndependentStreams < 2 || blank(replay.Uncertainty) {
		errs = append(errs, errors.New("stochastic/tail claims require at least two repetitions, two independent streams, and uncertainty"))
	}
	return errs
}

func artifactHasTailMetric(art BenchmarkArtifact) bool {
	return anyKeyContains(art.Results.Metrics, []string{"p90", "p95", "p99", "p999", "percentile", "tail"})
}

func artifactMakesCompetitiveClaim(art BenchmarkArtifact) bool {
	needles := []string{"llama.cpp", "llama_cpp", "fak-vs-", "fak_vs_", "native faster", "native_faster", "competitive"}
	if anyKeyOrStringContains(art.Results.Metrics, needles) || anyKeyOrStringContains(art.Results.Baseline, needles) {
		return true
	}
	for _, claim := range art.Results.ClaimLanguage {
		if containsAny(strings.ToLower(claim), needles) {
			return true
		}
	}
	return false
}

func artifactClaimsMeasuredOrAchieved(art BenchmarkArtifact) bool {
	if anyKeyOrStringMatches(art.Results.Metrics, positiveMeasurementMarker) {
		return true
	}
	for _, claim := range art.Results.ClaimLanguage {
		if positiveMeasurementMarker(claim) {
			return true
		}
	}
	return false
}

func anyKeyOrStringMatches(v any, match func(string) bool) bool {
	switch x := v.(type) {
	case map[string]any:
		for key, item := range x {
			if match(key) || anyKeyOrStringMatches(item, match) {
				return true
			}
		}
	case []any:
		for _, item := range x {
			if anyKeyOrStringMatches(item, match) {
				return true
			}
		}
	case string:
		return match(x)
	}
	return false
}

func artifactIsProjectionMarked(art BenchmarkArtifact) bool {
	for _, claim := range art.Results.ClaimLanguage {
		if metricIsProjectionMarked(claim) {
			return true
		}
	}
	return false
}

func matchingMetricNames(metrics map[string]any, match func(string) bool) []string {
	var names []string
	for name := range metrics {
		if match(name) {
			names = append(names, name)
		}
	}
	sort.Strings(names)
	return names
}

func isPerformanceMetric(name string) bool {
	name = strings.ToLower(name)
	return containsAny(name, []string{
		"elapsed", "latency", "duration", "_time", "time_", "seconds", "_ms", "_ns",
		"throughput", "tok_per_sec", "tokens_per_second", "tokens_s", "tok_s", "bandwidth", "qps",
		"energy", "joule", "watt", "cost", "usd",
	})
}

func metricIsProjectionMarked(name string) bool {
	return containsAny(strings.ToLower(name), []string{
		"projected", "projection", "estimate", "estimated", "theoretical", "simulated", "bound", "roofline",
	})
}

func positiveMeasurementMarker(value string) bool {
	value = strings.ToLower(value)
	for _, negative := range []string{
		"not measured or achieved", "neither measured nor achieved",
		"not measured", "not-measured", "not_measured", "unmeasured",
		"not achieved", "not-achieved", "not_achieved",
	} {
		value = strings.ReplaceAll(value, negative, "")
	}
	return containsWord(value, "measured") || containsWord(value, "achieved")
}

func containsWord(value, word string) bool {
	for _, field := range strings.FieldsFunc(value, func(r rune) bool {
		return (r < 'a' || r > 'z') && (r < '0' || r > '9')
	}) {
		if field == word {
			return true
		}
	}
	return false
}

func anyKeyContains(v any, needles []string) bool {
	switch x := v.(type) {
	case map[string]any:
		for key, item := range x {
			if containsAny(strings.ToLower(key), needles) || anyKeyContains(item, needles) {
				return true
			}
		}
	case []any:
		for _, item := range x {
			if anyKeyContains(item, needles) {
				return true
			}
		}
	}
	return false
}

func anyKeyOrStringContains(v any, needles []string) bool {
	switch x := v.(type) {
	case map[string]any:
		for key, item := range x {
			if containsAny(strings.ToLower(key), needles) || anyKeyOrStringContains(item, needles) {
				return true
			}
		}
	case []any:
		for _, item := range x {
			if anyKeyOrStringContains(item, needles) {
				return true
			}
		}
	case string:
		return containsAny(strings.ToLower(x), needles)
	}
	return false
}

func containsAny(value string, needles []string) bool {
	for _, needle := range needles {
		if strings.Contains(value, needle) {
			return true
		}
	}
	return false
}

func blank(value string) bool { return strings.TrimSpace(value) == "" }

func unknownIdentity(value string) bool {
	value = strings.TrimSpace(value)
	return value == "" || strings.EqualFold(value, lineageUnknown)
}

func hasBlank(values []string) bool {
	for _, value := range values {
		if blank(value) {
			return true
		}
	}
	return false
}

func validSHA256Digest(value string) bool {
	const prefix = "sha256:"
	if !strings.HasPrefix(value, prefix) {
		return false
	}
	raw, err := hex.DecodeString(strings.TrimPrefix(value, prefix))
	return err == nil && len(raw) == 32
}

func validCalibrationDate(value string) bool {
	if _, err := time.Parse(time.RFC3339, value); err == nil {
		return true
	}
	_, err := time.Parse(time.DateOnly, value)
	return err == nil
}

func finite(value float64) bool            { return !math.IsInf(value, 0) && !math.IsNaN(value) }
func finitePositive(value float64) bool    { return finite(value) && value > 0 }
func finiteNonnegative(value float64) bool { return finite(value) && value >= 0 }

func nearlyEqual(left, right float64) bool {
	return math.Abs(left-right) <= 1e-12*math.Max(1, math.Max(math.Abs(left), math.Abs(right)))
}
