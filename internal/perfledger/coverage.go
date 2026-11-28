// Adapted from ai-dynamo/aiperf
// modified from the original): phase-duration coverage ratios reporting how far
// TTFT and ITL signals extend into a profiled phase.
package perfledger

import (
	"errors"
	"fmt"
	"math"
	"sort"
)

// Default metric and phase column tags for TTFT/ITL coverage checks.
const (
	DefaultTTFTMetricTag  = "time_to_first_token"
	DefaultITLMetricTag   = "inter_token_latency"
	DefaultPhaseColumnTag = "benchmark_phase"
	PhaseWarmup           = "warmup"
	PhaseProfiling        = "profiling"
)

// MetricCoverage summarizes how far a metric's finite observations extend
// across a phase's duration.
type MetricCoverage struct {
	Tag                 string  `json:"tag"`
	ValidCount          int     `json:"valid_count"`
	TotalCount          int     `json:"total_count"`
	PhaseStartNS        float64 `json:"phase_start_ns"`
	PhaseEndNS          float64 `json:"phase_end_ns"`
	SignalStartNS       float64 `json:"signal_start_ns,omitempty"`
	SignalEndNS         float64 `json:"signal_end_ns,omitempty"`
	CoverageRatio       float64 `json:"coverage_ratio"`
	ActiveCoverageRatio float64 `json:"active_coverage_ratio"`
	SampleCoverageRatio float64 `json:"sample_coverage_ratio"`
}

// MeetsFloor reports whether CoverageRatio is at least minRatio.
func (c MetricCoverage) MeetsFloor(minRatio float64) bool {
	if math.IsNaN(minRatio) || math.IsInf(minRatio, 0) {
		return false
	}
	return c.CoverageRatio >= minRatio
}

// PhaseCoverageReport holds TTFT and ITL phase-duration coverage ratios for a
// phase (or entire run).
type PhaseCoverageReport struct {
	Phase        string         `json:"phase,omitempty"`
	PhaseStartNS float64        `json:"phase_start_ns"`
	PhaseEndNS   float64        `json:"phase_end_ns"`
	TotalRecords int            `json:"total_records"`
	TTFT         MetricCoverage `json:"ttft"`
	ITL          MetricCoverage `json:"itl"`
}

type coverageInterval struct {
	start float64
	end   float64
}

// ComputeMetricCoverage calculates how far finite, non-negative observations of
// values extend into the phase window spanned by masked records' [startNS, endNS].
func ComputeMetricCoverage(tag string, startNS, endNS, values []float64, mask []bool) (MetricCoverage, error) {
	if len(startNS) != len(endNS) || len(startNS) != len(values) {
		return MetricCoverage{}, fmt.Errorf("perfledger: ComputeMetricCoverage: mismatched lengths start=%d end=%d values=%d", len(startNS), len(endNS), len(values))
	}
	if mask != nil && len(mask) != len(startNS) {
		return MetricCoverage{}, fmt.Errorf("perfledger: ComputeMetricCoverage: mask length %d != %d", len(mask), len(startNS))
	}

	cov := MetricCoverage{Tag: tag}
	phaseStart := math.Inf(1)
	phaseEnd := math.Inf(-1)
	sigStart := math.Inf(1)
	sigEnd := math.Inf(-1)
	var active []coverageInterval

	for i := range startNS {
		if mask != nil && !mask[i] {
			continue
		}
		st, en := startNS[i], endNS[i]
		if !isFiniteFloat(st) || !isFiniteFloat(en) {
			continue
		}
		if en < st {
			en = st
		}
		cov.TotalCount++
		if st < phaseStart {
			phaseStart = st
		}
		if en > phaseEnd {
			phaseEnd = en
		}
		v := values[i]
		if !isFiniteFloat(v) || v < 0 {
			continue
		}
		cov.ValidCount++
		if st < sigStart {
			sigStart = st
		}
		if en > sigEnd {
			sigEnd = en
		}
		active = append(active, coverageInterval{start: st, end: en})
	}

	if cov.TotalCount == 0 {
		return cov, nil
	}
	cov.PhaseStartNS = phaseStart
	cov.PhaseEndNS = phaseEnd
	cov.SampleCoverageRatio = float64(cov.ValidCount) / float64(cov.TotalCount)

	if cov.ValidCount == 0 {
		return cov, nil
	}
	cov.SignalStartNS = sigStart
	cov.SignalEndNS = sigEnd

	phaseDur := phaseEnd - phaseStart
	if phaseDur <= 0 {
		return cov, nil
	}
	cov.CoverageRatio = clampUnit((sigEnd - sigStart) / phaseDur)
	cov.ActiveCoverageRatio = clampUnit(mergedIntervalDuration(active) / phaseDur)
	return cov, nil
}

// ComputeTTFTITLCoverage computes TTFT and ITL phase-duration coverage ratios
// over parallel record slices and an optional phase mask.
func ComputeTTFTITLCoverage(startNS, endNS, ttft, itl []float64, mask []bool) (PhaseCoverageReport, error) {
	ttftCov, err := ComputeMetricCoverage(DefaultTTFTMetricTag, startNS, endNS, ttft, mask)
	if err != nil {
		return PhaseCoverageReport{}, err
	}
	itlCov, err := ComputeMetricCoverage(DefaultITLMetricTag, startNS, endNS, itl, mask)
	if err != nil {
		return PhaseCoverageReport{}, err
	}
	return PhaseCoverageReport{
		PhaseStartNS: ttftCov.PhaseStartNS,
		PhaseEndNS:   ttftCov.PhaseEndNS,
		TotalRecords: ttftCov.TotalCount,
		TTFT:         ttftCov,
		ITL:          itlCov,
	}, nil
}

// PhaseTTFTITLCoverage computes TTFT and ITL coverage ratios for phase (or all
// records when phase is empty) directly from a ColumnStore.
func (s *ColumnStore) PhaseTTFTITLCoverage(phase, ttftTag, itlTag string) (PhaseCoverageReport, error) {
	if s == nil {
		return PhaseCoverageReport{}, errors.New("perfledger: PhaseTTFTITLCoverage: nil ColumnStore")
	}
	if ttftTag == "" {
		ttftTag = DefaultTTFTMetricTag
	}
	if itlTag == "" {
		itlTag = DefaultITLMetricTag
	}
	var mask []bool
	if phase != "" {
		var err error
		mask, err = PhaseMask(s, DefaultPhaseColumnTag, phase)
		if err != nil {
			return PhaseCoverageReport{}, err
		}
	}
	starts := s.startNS[:s.count]
	ends := s.endNS[:s.count]
	ttftVals := s.Numeric(ttftTag)
	itlVals := s.Numeric(itlTag)

	ttftCov, err := ComputeMetricCoverage(ttftTag, starts, ends, ttftVals, mask)
	if err != nil {
		return PhaseCoverageReport{}, err
	}
	itlCov, err := ComputeMetricCoverage(itlTag, starts, ends, itlVals, mask)
	if err != nil {
		return PhaseCoverageReport{}, err
	}
	return PhaseCoverageReport{
		Phase:        phase,
		PhaseStartNS: ttftCov.PhaseStartNS,
		PhaseEndNS:   ttftCov.PhaseEndNS,
		TotalRecords: ttftCov.TotalCount,
		TTFT:         ttftCov,
		ITL:          itlCov,
	}, nil
}

func mergedIntervalDuration(intervals []coverageInterval) float64 {
	if len(intervals) == 0 {
		return 0
	}
	sort.Slice(intervals, func(i, j int) bool {
		if intervals[i].start != intervals[j].start {
			return intervals[i].start < intervals[j].start
		}
		return intervals[i].end < intervals[j].end
	})
	total := 0.0
	curStart, curEnd := intervals[0].start, intervals[0].end
	for _, iv := range intervals[1:] {
		if iv.start <= curEnd {
			if iv.end > curEnd {
				curEnd = iv.end
			}
			continue
		}
		total += curEnd - curStart
		curStart, curEnd = iv.start, iv.end
	}
	total += curEnd - curStart
	return total
}

func clampUnit(v float64) float64 {
	if math.IsNaN(v) || v <= 0 {
		return 0
	}
	if v >= 1 {
		return 1
	}
	return v
}

func isFiniteFloat(v float64) bool {
	return !math.IsNaN(v) && !math.IsInf(v, 0)
}
