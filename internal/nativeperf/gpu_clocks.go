package nativeperf

import (
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strings"
	"time"
)

// GPUClockObservationSchema is the versioned schema for GPU post-pass clock observations.
const GPUClockObservationSchema = "fak-native-gpu-clock-observation/v1"

// ClockReadStatus represents the status of a GPU clock read after a pass.
type ClockReadStatus string

const (
	// ClockStatusObserved indicates the SM and memory clocks were read successfully.
	ClockStatusObserved ClockReadStatus = "observed"
	// ClockStatusFailed indicates reading the clocks produced a driver or hardware error.
	ClockStatusFailed ClockReadStatus = "failed"
	// ClockStatusUnreadable indicates the platform/device does not expose clock counters or access is denied.
	ClockStatusUnreadable ClockReadStatus = "unreadable"
)

// PostPassClockObservation captures SM and memory clock observations for a single pass.
type PostPassClockObservation struct {
	PassID         string          `json:"pass_id"`
	PassIndex      int             `json:"pass_index"`
	DeviceID       string          `json:"device_id,omitempty"`
	SMClockMHz     float64         `json:"sm_clock_mhz"`
	MemoryClockMHz float64         `json:"memory_clock_mhz"`
	Status         ClockReadStatus `json:"status"`
	Error          string          `json:"error,omitempty"`
	ObservedAtUnix int64           `json:"observed_at_unix,omitempty"`
}

func isFinitePositive(v float64) bool {
	return v > 0 && !math.IsNaN(v) && !math.IsInf(v, 0)
}

// IsObserved reports whether this observation captured valid, positive clock rates.
func (o PostPassClockObservation) IsObserved() bool {
	return o.Status == ClockStatusObserved && isFinitePositive(o.SMClockMHz) && isFinitePositive(o.MemoryClockMHz) && o.Error == ""
}

// IsFailed reports whether this observation failed with an error.
func (o PostPassClockObservation) IsFailed() bool {
	if o.Status == ClockStatusFailed {
		return true
	}
	if o.Status == ClockStatusUnreadable {
		return false
	}
	if o.Error != "" {
		return true
	}
	if o.Status == ClockStatusObserved && (!isFinitePositive(o.SMClockMHz) || !isFinitePositive(o.MemoryClockMHz)) {
		return true
	}
	return false
}

// IsUnreadable reports whether clocks were unreadable on this platform or device.
func (o PostPassClockObservation) IsUnreadable() bool {
	return o.Status == ClockStatusUnreadable
}

// ClockCoverageBuckets categorizes pass clock readings into mutually exclusive buckets.
type ClockCoverageBuckets struct {
	Observed   int `json:"observed"`
	Failed     int `json:"failed"`
	Unreadable int `json:"unreadable"`
	Total      int `json:"total"`
}

// CoverageRatio returns the proportion of passes that produced observed clocks.
func (b ClockCoverageBuckets) CoverageRatio() float64 {
	if b.Total <= 0 {
		return 0
	}
	return float64(b.Observed) / float64(b.Total)
}

// ObservedRatio is an alias for CoverageRatio.
func (b ClockCoverageBuckets) ObservedRatio() float64 {
	return b.CoverageRatio()
}

// FailedRatio returns the proportion of passes that failed clock acquisition.
func (b ClockCoverageBuckets) FailedRatio() float64 {
	if b.Total <= 0 {
		return 0
	}
	return float64(b.Failed) / float64(b.Total)
}

// UnreadableRatio returns the proportion of passes where clocks were unreadable.
func (b ClockCoverageBuckets) UnreadableRatio() float64 {
	if b.Total <= 0 {
		return 0
	}
	return float64(b.Unreadable) / float64(b.Total)
}

// IsComplete reports whether every pass produced a successfully observed clock.
func (b ClockCoverageBuckets) IsComplete() bool {
	return b.Total > 0 && b.Observed == b.Total
}

// TabulateClockCoverage aggregates a sequence of post-pass clock observations into coverage buckets.
func TabulateClockCoverage(observations []PostPassClockObservation) ClockCoverageBuckets {
	var buckets ClockCoverageBuckets
	buckets.Total = len(observations)

	for _, obs := range observations {
		switch {
		case obs.IsUnreadable():
			buckets.Unreadable++
		case obs.IsObserved():
			buckets.Observed++
		case obs.IsFailed():
			buckets.Failed++
		default:
			// Fail-closed fallback: any unrecognized or inconsistent state is categorized as failed
			buckets.Failed++
		}
	}
	return buckets
}

// ClockTolerancePolicy specifies the allowable clock drift and coverage bounds.
type ClockTolerancePolicy struct {
	// MinCoverageRatio is the minimum fraction of passes requiring observed clocks (e.g., 1.0 for 100%).
	MinCoverageRatio float64 `json:"min_coverage_ratio"`
	// AllowUnreadable permits runs when clock monitoring is unsupported by the host or platform.
	AllowUnreadable bool `json:"allow_unreadable,omitempty"`
	// MaxSMClockDriftPercent is the maximum relative drift percentage allowed for SM clocks.
	MaxSMClockDriftPercent float64 `json:"max_sm_clock_drift_percent"`
	// MaxMemoryClockDriftPercent is the maximum relative drift percentage allowed for Memory clocks.
	MaxMemoryClockDriftPercent float64 `json:"max_memory_clock_drift_percent"`
	// TargetSMClockMHz is an optional pinned target SM clock. When positive, drift is evaluated against target.
	TargetSMClockMHz float64 `json:"target_sm_clock_mhz,omitempty"`
	// TargetMemoryClockMHz is an optional pinned target Memory clock. When positive, drift is evaluated against target.
	TargetMemoryClockMHz float64 `json:"target_memory_clock_mhz,omitempty"`
}

// ClockValidationResult summarizes clock coverage and tolerance compliance.
type ClockValidationResult struct {
	Valid              bool                 `json:"valid"`
	Coverage           ClockCoverageBuckets `json:"coverage"`
	CoverageRatio      float64              `json:"coverage_ratio"`
	ObservedPasses     int                  `json:"observed_passes"`
	MinSMClockMHz      float64              `json:"min_sm_clock_mhz"`
	MaxSMClockMHz      float64              `json:"max_sm_clock_mhz"`
	MeanSMClockMHz     float64              `json:"mean_sm_clock_mhz"`
	SMDriftPercent     float64              `json:"sm_drift_percent"`
	MinMemoryClockMHz  float64              `json:"min_memory_clock_mhz"`
	MaxMemoryClockMHz  float64              `json:"max_memory_clock_mhz"`
	MeanMemoryClockMHz float64              `json:"mean_memory_clock_mhz"`
	MemoryDriftPercent float64              `json:"memory_drift_percent"`
	Violations         []string             `json:"violations,omitempty"`
}

// ClockValidationError is returned when clock observations violate coverage or tolerance policy.
type ClockValidationError struct {
	Result ClockValidationResult
}

func (e *ClockValidationError) Error() string {
	if len(e.Result.Violations) == 0 {
		return "nativeperf: clock validation failed"
	}
	return fmt.Sprintf("nativeperf: clock validation failed: %s", strings.Join(e.Result.Violations, "; "))
}

// ValidateClockObservations inspects a sequence of post-pass clock observations against policy.
func ValidateClockObservations(observations []PostPassClockObservation, policy ClockTolerancePolicy) (*ClockValidationResult, error) {
	buckets := TabulateClockCoverage(observations)
	res := &ClockValidationResult{
		Coverage:      buckets,
		CoverageRatio: buckets.CoverageRatio(),
	}

	if buckets.Total == 0 {
		if policy.MinCoverageRatio > 0 {
			res.Violations = append(res.Violations, "zero clock observations provided")
		}
		res.Valid = len(res.Violations) == 0
		if !res.Valid {
			return res, &ClockValidationError{Result: *res}
		}
		return res, nil
	}

	// 1. Coverage validation
	if res.CoverageRatio < policy.MinCoverageRatio {
		res.Violations = append(res.Violations, fmt.Sprintf(
			"coverage ratio %.2f below minimum required %.2f (%d/%d observed)",
			res.CoverageRatio, policy.MinCoverageRatio, buckets.Observed, buckets.Total,
		))
	}

	if !policy.AllowUnreadable && buckets.Unreadable > 0 {
		res.Violations = append(res.Violations, fmt.Sprintf(
			"%d unreadable clock observations not permitted by policy", buckets.Unreadable,
		))
	}

	if buckets.Failed > 0 {
		res.Violations = append(res.Violations, fmt.Sprintf(
			"%d clock observations reported errors", buckets.Failed,
		))
	}

	// 2. Collect valid observed clocks
	var smClocks []float64
	var memClocks []float64
	for _, o := range observations {
		if o.IsObserved() {
			smClocks = append(smClocks, o.SMClockMHz)
			memClocks = append(memClocks, o.MemoryClockMHz)
		}
	}
	res.ObservedPasses = len(smClocks)

	// If no passes were observed, check if policy permitted it
	if res.ObservedPasses == 0 {
		if buckets.Unreadable == buckets.Total && policy.AllowUnreadable && policy.MinCoverageRatio <= 0 {
			res.Valid = true
			return res, nil
		}
		if len(res.Violations) == 0 {
			res.Violations = append(res.Violations, "no observed clock measurements available for tolerance evaluation")
		}
		res.Valid = false
		return res, &ClockValidationError{Result: *res}
	}

	// 3. Compute SM clock statistics and drift
	minSM, maxSM, meanSM := computeClockStats(smClocks)
	res.MinSMClockMHz = minSM
	res.MaxSMClockMHz = maxSM
	res.MeanSMClockMHz = meanSM

	var smDrift float64
	if isFinitePositive(policy.TargetSMClockMHz) {
		dev := math.Max(math.Abs(maxSM-policy.TargetSMClockMHz), math.Abs(minSM-policy.TargetSMClockMHz))
		smDrift = (dev / policy.TargetSMClockMHz) * 100
	} else if meanSM > 0 {
		dev := math.Max(maxSM-meanSM, meanSM-minSM)
		smDrift = (dev / meanSM) * 100
	}
	res.SMDriftPercent = smDrift

	if isFinitePositive(policy.MaxSMClockDriftPercent) && smDrift > policy.MaxSMClockDriftPercent {
		res.Violations = append(res.Violations, fmt.Sprintf(
			"SM clock drift %.2f%% exceeds tolerance %.2f%% (min=%.1f, max=%.1f, mean=%.1f MHz)",
			smDrift, policy.MaxSMClockDriftPercent, minSM, maxSM, meanSM,
		))
	}

	// 4. Compute Memory clock statistics and drift
	minMem, maxMem, meanMem := computeClockStats(memClocks)
	res.MinMemoryClockMHz = minMem
	res.MaxMemoryClockMHz = maxMem
	res.MeanMemoryClockMHz = meanMem

	var memDrift float64
	if isFinitePositive(policy.TargetMemoryClockMHz) {
		dev := math.Max(math.Abs(maxMem-policy.TargetMemoryClockMHz), math.Abs(minMem-policy.TargetMemoryClockMHz))
		memDrift = (dev / policy.TargetMemoryClockMHz) * 100
	} else if meanMem > 0 {
		dev := math.Max(maxMem-meanMem, meanMem-minMem)
		memDrift = (dev / meanMem) * 100
	}
	res.MemoryDriftPercent = memDrift

	if isFinitePositive(policy.MaxMemoryClockDriftPercent) && memDrift > policy.MaxMemoryClockDriftPercent {
		res.Violations = append(res.Violations, fmt.Sprintf(
			"memory clock drift %.2f%% exceeds tolerance %.2f%% (min=%.1f, max=%.1f, mean=%.1f MHz)",
			memDrift, policy.MaxMemoryClockDriftPercent, minMem, maxMem, meanMem,
		))
	}

	res.Valid = len(res.Violations) == 0
	if !res.Valid {
		return res, &ClockValidationError{Result: *res}
	}
	return res, nil
}

func computeClockStats(values []float64) (minVal, maxVal, meanVal float64) {
	if len(values) == 0 {
		return 0, 0, 0
	}
	minVal = values[0]
	maxVal = values[0]
	var sum float64
	for _, v := range values {
		if v < minVal {
			minVal = v
		}
		if v > maxVal {
			maxVal = v
		}
		sum += v
	}
	meanVal = sum / float64(len(values))
	return minVal, maxVal, meanVal
}

// BindPostPassClockObservation produces an observed post-pass clock observation.
func BindPostPassClockObservation(passID string, passIndex int, deviceID string, smMHz, memMHz float64) PostPassClockObservation {
	return PostPassClockObservation{
		PassID:         passID,
		PassIndex:      passIndex,
		DeviceID:       deviceID,
		SMClockMHz:     smMHz,
		MemoryClockMHz: memMHz,
		Status:         ClockStatusObserved,
		ObservedAtUnix: time.Now().UnixNano(),
	}
}

// BindFailedClockObservation produces a failed post-pass clock observation.
func BindFailedClockObservation(passID string, passIndex int, deviceID string, err error) PostPassClockObservation {
	errStr := "unknown clock query failure"
	if err != nil {
		errStr = err.Error()
	}
	return PostPassClockObservation{
		PassID:         passID,
		PassIndex:      passIndex,
		DeviceID:       deviceID,
		Status:         ClockStatusFailed,
		Error:          errStr,
		ObservedAtUnix: time.Now().UnixNano(),
	}
}

// BindUnreadableClockObservation produces an unreadable post-pass clock observation.
func BindUnreadableClockObservation(passID string, passIndex int, deviceID string, reason string) PostPassClockObservation {
	return PostPassClockObservation{
		PassID:         passID,
		PassIndex:      passIndex,
		DeviceID:       deviceID,
		Status:         ClockStatusUnreadable,
		Error:          reason,
		ObservedAtUnix: time.Now().UnixNano(),
	}
}

// GPUClockEvidence bundles versioned post-pass clock observations for an execution run.
type GPUClockEvidence struct {
	Schema       string                     `json:"schema"`
	DeviceID     string                     `json:"device_id,omitempty"`
	Observations []PostPassClockObservation `json:"observations"`
}

// NewGPUClockEvidence creates a GPUClockEvidence container tagged with GPUClockObservationSchema.
func NewGPUClockEvidence(deviceID string, observations []PostPassClockObservation) GPUClockEvidence {
	return GPUClockEvidence{
		Schema:       GPUClockObservationSchema,
		DeviceID:     deviceID,
		Observations: observations,
	}
}

// MarshalClockObservations returns the serialized JSON representation of clock observations.
func MarshalClockObservations(observations []PostPassClockObservation) ([]byte, error) {
	return json.Marshal(observations)
}

// UnmarshalClockObservations parses JSON bytes into clock observations.
func UnmarshalClockObservations(data []byte) ([]PostPassClockObservation, error) {
	var obs []PostPassClockObservation
	if err := json.Unmarshal(data, &obs); err != nil {
		return nil, err
	}
	sort.Slice(obs, func(i, j int) bool {
		return obs[i].PassIndex < obs[j].PassIndex
	})
	return obs, nil
}
