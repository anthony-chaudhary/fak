// Structural adaptation of trusted-engine flip envelope calibration from
// humanityslastsystem/hls (Apache-2.0, src/hls/verify/calibration.py @ 39cd872cd1d3).
package logitoracle

import (
	"errors"
	"fmt"
	"math"
)

const (
	// DefaultMinServingsPerEngine is the minimum number of clean servings required per trusted engine.
	DefaultMinServingsPerEngine = 8
	// DefaultGarwoodAlpha is the one-sided Garwood confidence level (0.025 -> 97.5% upper bound).
	DefaultGarwoodAlpha = 0.025
)

var (
	ErrInvalidCalibration              = errors.New("logitoracle: invalid flip calibration inputs")
	ErrCalibrationMismatch             = errors.New("logitoracle: calibration workload hash or GPU kind mismatch")
	ErrInsufficientExposure            = errors.New("logitoracle: candidate token exposure below calibrated target")
	ErrCalibrationSelfValidationFailed = errors.New("logitoracle: trusted engine serving failed self-validation")
)

// CalibrationSpec configures trusted-engine flip-envelope calibration.
type CalibrationSpec struct {
	WorkloadHash         string    `json:"workload_hash"`
	GPUKind              string    `json:"gpu_kind"`
	Rungs                []float64 `json:"rungs,omitempty"`
	GarwoodAlpha         float64   `json:"garwood_alpha,omitempty"`
	ExposureAlpha        float64   `json:"exposure_alpha,omitempty"`
	ExposureMu           float64   `json:"exposure_mu"`
	MinServingsPerEngine int       `json:"min_servings_per_engine,omitempty"`
	GateDelta            float64   `json:"gate_delta,omitempty"`
	GateMultiplier       float64   `json:"gate_multiplier,omitempty"`
}

func (s CalibrationSpec) withDefaults() (CalibrationSpec, error) {
	if s.WorkloadHash == "" || s.GPUKind == "" {
		return s, fmt.Errorf("%w: workload_hash and gpu_kind are required", ErrInvalidCalibration)
	}
	if len(s.Rungs) == 0 {
		s.Rungs = append([]float64(nil), DefaultFlipGapRungs...)
	}
	for i, r := range s.Rungs {
		if math.IsNaN(r) || math.IsInf(r, 0) || r <= 0 || (i > 0 && r <= s.Rungs[i-1]) {
			return s, fmt.Errorf("%w: invalid rung[%d]=%v", ErrInvalidCalibration, i, r)
		}
	}
	if s.GarwoodAlpha == 0 {
		s.GarwoodAlpha = DefaultGarwoodAlpha
	}
	if math.IsNaN(s.GarwoodAlpha) || s.GarwoodAlpha <= 0 || s.GarwoodAlpha >= 1 {
		return s, fmt.Errorf("%w: garwood_alpha=%v must be in (0, 1)", ErrInvalidCalibration, s.GarwoodAlpha)
	}
	if s.ExposureAlpha == 0 {
		s.ExposureAlpha = DefaultGarwoodAlpha
	}
	if math.IsNaN(s.ExposureAlpha) || s.ExposureAlpha <= 0 || s.ExposureAlpha >= 1 {
		return s, fmt.Errorf("%w: exposure_alpha=%v must be in (0, 1)", ErrInvalidCalibration, s.ExposureAlpha)
	}
	if math.IsNaN(s.ExposureMu) || math.IsInf(s.ExposureMu, 0) || s.ExposureMu <= 0 {
		return s, fmt.Errorf("%w: exposure_mu=%v must be finite and > 0", ErrInvalidCalibration, s.ExposureMu)
	}
	if s.MinServingsPerEngine == 0 {
		s.MinServingsPerEngine = DefaultMinServingsPerEngine
	}
	if s.MinServingsPerEngine < DefaultMinServingsPerEngine {
		return s, fmt.Errorf("%w: min_servings_per_engine=%d must be >= %d", ErrInvalidCalibration, s.MinServingsPerEngine, DefaultMinServingsPerEngine)
	}
	if s.GateDelta == 0 {
		s.GateDelta = 0.05
	}
	if math.IsNaN(s.GateDelta) || s.GateDelta <= 0 || s.GateDelta >= 1 {
		return s, fmt.Errorf("%w: gate_delta=%v must be in (0, 1)", ErrInvalidCalibration, s.GateDelta)
	}
	if s.GateMultiplier == 0 {
		s.GateMultiplier = 1.0
	}
	if math.IsNaN(s.GateMultiplier) || math.IsInf(s.GateMultiplier, 0) || s.GateMultiplier <= 0 {
		return s, fmt.Errorf("%w: gate_multiplier=%v must be > 0", ErrInvalidCalibration, s.GateMultiplier)
	}
	return s, nil
}

// TrustedEngineServings holds clean teacher-forced servings from one trusted engine.
type TrustedEngineServings struct {
	Engine   string           `json:"engine"`
	Servings [][]PositionFlip `json:"servings"`
}

// EngineCalibrationSummary records per-engine calibrated flip statistics.
type EngineCalibrationSummary struct {
	Engine       string    `json:"engine"`
	Servings     int       `json:"servings"`
	TotalTokens  int       `json:"total_tokens"`
	TotalFlips   int       `json:"total_flips"`
	MeanFlipRate float64   `json:"mean_flip_rate"`
	LambdaUp     []float64 `json:"lambda_up"`
}

// CalibratedEnvelope is the locked, workload- and GPU-keyed flip-gap gate.
type CalibratedEnvelope struct {
	WorkloadHash   string                     `json:"workload_hash"`
	GPUKind        string                     `json:"gpu_kind"`
	ExposureTarget int                        `json:"exposure_target"`
	GateConfig     FlipGateConfig             `json:"gate_config"`
	Engines        []EngineCalibrationSummary `json:"engines"`
}

// GarwoodUpperCount returns m such that P(Poisson(m) <= count) = alpha,
// equivalent to 0.5 * chi2_inv(1 - alpha, 2*(count + 1)).
func GarwoodUpperCount(count int, alpha float64) float64 {
	if count < 0 {
		count = 0
	}
	if count == 0 {
		return -math.Log(alpha)
	}
	lo := float64(count)
	hi := float64(count) + 10.0
	for 1.0-PoissonUpperTail(hi, count) > alpha {
		lo = hi
		hi *= 2.0
		if hi > 1e12 {
			return hi
		}
	}
	for i := 0; i < 64; i++ {
		mid := 0.5 * (lo + hi)
		cdf := 1.0 - PoissonUpperTail(mid, count)
		if cdf > alpha {
			lo = mid
		} else {
			hi = mid
		}
	}
	return 0.5 * (lo + hi)
}

// CalibrateFlipEnvelope derives a CalibratedEnvelope from trusted-engine servings.
// Each engine must supply >= 8 servings; per-rung LambdaUp takes the most-permissive
// Garwood upper rate across engines, QMax locks to the worst engine's mean flip rate,
// and all trusted servings are self-validated against the resulting gate.
func CalibrateFlipEnvelope(spec CalibrationSpec, engines []TrustedEngineServings) (CalibratedEnvelope, error) {
	cfgSpec, err := spec.withDefaults()
	if err != nil {
		return CalibratedEnvelope{}, err
	}
	if len(engines) == 0 {
		return CalibratedEnvelope{}, fmt.Errorf("%w: at least one trusted engine is required", ErrInvalidCalibration)
	}

	numRungs := len(cfgSpec.Rungs)
	maxLambdaUp := make([]float64, numRungs)
	qMax := 0.0
	summaries := make([]EngineCalibrationSummary, 0, len(engines))

	for _, eng := range engines {
		if eng.Engine == "" {
			return CalibratedEnvelope{}, fmt.Errorf("%w: engine name is required", ErrInvalidCalibration)
		}
		if len(eng.Servings) < cfgSpec.MinServingsPerEngine {
			return CalibratedEnvelope{}, fmt.Errorf("%w: engine %q has %d servings, need >= %d",
				ErrInvalidCalibration, eng.Engine, len(eng.Servings), cfgSpec.MinServingsPerEngine)
		}

		totalTokens := 0
		totalFlips := 0
		nestedCounts := make([]int, numRungs)

		for sIdx, serving := range eng.Servings {
			if len(serving) == 0 {
				return CalibratedEnvelope{}, fmt.Errorf("%w: engine %q serving[%d] is empty",
					ErrInvalidCalibration, eng.Engine, sIdx)
			}
			for pIdx, p := range serving {
				gap, isFlip, finite := p.FlipGap()
				if !finite {
					return CalibratedEnvelope{}, fmt.Errorf("%w: non-finite gap in engine %q serving[%d] pos[%d]",
						ErrInvalidCalibration, eng.Engine, sIdx, pIdx)
				}
				totalTokens++
				if !isFlip {
					continue
				}
				totalFlips++
				for rIdx, rung := range cfgSpec.Rungs {
					if gap >= rung {
						nestedCounts[rIdx]++
					}
				}
			}
		}

		meanRate := float64(totalFlips) / float64(totalTokens)
		if meanRate > qMax {
			qMax = meanRate
		}
		engLambdaUp := make([]float64, numRungs)
		for rIdx := range cfgSpec.Rungs {
			upperCount := GarwoodUpperCount(nestedCounts[rIdx], cfgSpec.GarwoodAlpha)
			rate := upperCount / float64(totalTokens)
			engLambdaUp[rIdx] = rate
			if rate > maxLambdaUp[rIdx] {
				maxLambdaUp[rIdx] = rate
			}
		}

		summaries = append(summaries, EngineCalibrationSummary{
			Engine:       eng.Engine,
			Servings:     len(eng.Servings),
			TotalTokens:  totalTokens,
			TotalFlips:   totalFlips,
			MeanFlipRate: meanRate,
			LambdaUp:     engLambdaUp,
		})
	}

	gateCfg := FlipGateConfig{
		Rungs:      cfgSpec.Rungs,
		LambdaUp:   maxLambdaUp,
		QMax:       qMax,
		Multiplier: cfgSpec.GateMultiplier,
		Delta:      cfgSpec.GateDelta,
	}

	// Self-validate every trusted engine's aggregate serving against the calibrated gate.
	for _, eng := range engines {
		var combined []PositionFlip
		for _, s := range eng.Servings {
			combined = append(combined, s...)
		}
		rep, err := EvaluateFlipCurve(combined, gateCfg)
		if err != nil {
			return CalibratedEnvelope{}, fmt.Errorf("%w: engine %q: %v", ErrCalibrationSelfValidationFailed, eng.Engine, err)
		}
		if !rep.Pass {
			return CalibratedEnvelope{}, fmt.Errorf("%w: engine %q failed self-validation: %s",
				ErrCalibrationSelfValidationFailed, eng.Engine, rep.FailureReason)
		}
	}

	exposureTarget := int(math.Ceil(math.Log(1.0/cfgSpec.ExposureAlpha) / cfgSpec.ExposureMu))

	return CalibratedEnvelope{
		WorkloadHash:   cfgSpec.WorkloadHash,
		GPUKind:        cfgSpec.GPUKind,
		ExposureTarget: exposureTarget,
		GateConfig:     gateCfg,
		Engines:        summaries,
	}, nil
}

// ValidateFor verifies that the calibrated envelope matches the candidate's
// workload hash and GPU kind, refusing stale or mismatched calibrations.
func (e CalibratedEnvelope) ValidateFor(workloadHash, gpuKind string) error {
	if workloadHash == "" || gpuKind == "" {
		return fmt.Errorf("%w: empty workload_hash or gpu_kind", ErrCalibrationMismatch)
	}
	if e.WorkloadHash != workloadHash || e.GPUKind != gpuKind {
		return fmt.Errorf("%w: calibrated=(%q, %q) != candidate=(%q, %q)",
			ErrCalibrationMismatch, e.WorkloadHash, e.GPUKind, workloadHash, gpuKind)
	}
	return nil
}

// Evaluate validates workload/GPU identity and minimum token exposure, then
// grades the candidate's served positions against the calibrated flip-gap gate.
func (e CalibratedEnvelope) Evaluate(workloadHash, gpuKind string, flips []PositionFlip) (FlipCurveReport, error) {
	if err := e.ValidateFor(workloadHash, gpuKind); err != nil {
		return FlipCurveReport{}, err
	}
	if len(flips) < e.ExposureTarget {
		return FlipCurveReport{}, fmt.Errorf("%w: got %d tokens, need >= %d",
			ErrInsufficientExposure, len(flips), e.ExposureTarget)
	}
	return EvaluateFlipCurve(flips, e.GateConfig)
}
