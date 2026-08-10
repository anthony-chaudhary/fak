package bench

import (
	"errors"
	"fmt"
	"math"
	"sort"
	"time"
)

const HeldScheduleReportSchema = "fak-held-schedule-report/1"

// HeldScheduleJob is one hardware-measured request kept out of calibration.
// ServiceMS is the observed wall service time; token classes are known before admission.
type HeldScheduleJob struct {
	ID                  string  `json:"id"`
	UncachedInputTokens int64   `json:"uncached_input_tokens"`
	CachedInputTokens   int64   `json:"cached_input_tokens"`
	DecodeTokens        int64   `json:"decode_tokens"`
	ServiceMS           float64 `json:"service_ms"`
}

type HeldSchedulePolicy struct {
	Name            string  `json:"name"`
	FixedMS         float64 `json:"fixed_ms"`
	PrefillRate     float64 `json:"prefill_rate_ms_per_token"`
	CacheReadRate   float64 `json:"cache_read_rate_ms_per_token"`
	DecodeRate      float64 `json:"decode_rate_ms_per_token"`
	DecisionMS      float64 `json:"decision_ms_per_request"`
	DecisionSamples int     `json:"decision_samples"`
}

type HeldScheduleOutcome struct {
	Policy                HeldSchedulePolicy `json:"policy"`
	Order                 []string           `json:"order"`
	MeanCompletionMS      float64            `json:"mean_completion_ms"`
	P95CompletionMS       float64            `json:"p95_completion_ms"`
	MakespanMS            float64            `json:"makespan_ms"`
	TotalDecisionOverhead float64            `json:"total_decision_overhead_ms"`
}

type HeldScheduleReport struct {
	Schema                       string              `json:"schema"`
	Jobs                         int                 `json:"jobs"`
	Calibrated                   HeldScheduleOutcome `json:"calibrated"`
	ScalarTotal                  HeldScheduleOutcome `json:"scalar_total"`
	MeanCompletionReducedMS      float64             `json:"mean_completion_reduced_ms"`
	AddedDecisionOverheadMS      float64             `json:"added_decision_overhead_ms"`
	NetMeanCompletionValueMS     float64             `json:"net_mean_completion_value_ms"`
	NetMeanCompletionValuePct    float64             `json:"net_mean_completion_value_pct"`
	CalibratedBeatsScalarTotal   bool                `json:"calibrated_beats_scalar_total"`
	HardwareMeasuredServiceTimes bool                `json:"hardware_measured_service_times"`
}

// EvaluateHeldSchedule compares non-preemptive shortest-predicted-job-first queues.
// All jobs arrive together. The actual completion clock advances by measured ServiceMS;
// only ordering uses each policy's prediction, so held service truth never leaks into admission.
func EvaluateHeldSchedule(jobs []HeldScheduleJob, calibrated, scalar HeldSchedulePolicy) (HeldScheduleReport, error) {
	if len(jobs) < 3 {
		return HeldScheduleReport{}, errors.New("held scheduling requires at least 3 jobs")
	}
	seen := map[string]bool{}
	for i, j := range jobs {
		if j.ID == "" || seen[j.ID] {
			return HeldScheduleReport{}, fmt.Errorf("held job %d has empty or duplicate id", i)
		}
		seen[j.ID] = true
		if j.UncachedInputTokens < 0 || j.CachedInputTokens < 0 || j.DecodeTokens < 0 || j.ServiceMS <= 0 || math.IsNaN(j.ServiceMS) || math.IsInf(j.ServiceMS, 0) {
			return HeldScheduleReport{}, fmt.Errorf("held job %q has invalid tokens or service time", j.ID)
		}
	}
	for _, p := range []HeldSchedulePolicy{calibrated, scalar} {
		if p.Name == "" || p.DecisionMS < 0 || math.IsNaN(p.DecisionMS) || math.IsInf(p.DecisionMS, 0) {
			return HeldScheduleReport{}, errors.New("schedule policy needs a name and finite non-negative measured overhead")
		}
	}
	co := scheduleOutcome(jobs, calibrated)
	so := scheduleOutcome(jobs, scalar)
	reduced := so.MeanCompletionMS - co.MeanCompletionMS
	added := co.TotalDecisionOverhead - so.TotalDecisionOverhead
	net := reduced - added
	rep := HeldScheduleReport{
		Schema: HeldScheduleReportSchema, Jobs: len(jobs), Calibrated: co, ScalarTotal: so,
		MeanCompletionReducedMS: reduced, AddedDecisionOverheadMS: added,
		NetMeanCompletionValueMS: net, CalibratedBeatsScalarTotal: net > 0,
		HardwareMeasuredServiceTimes: true,
	}
	if so.MeanCompletionMS > 0 {
		rep.NetMeanCompletionValuePct = 100 * net / so.MeanCompletionMS
	}
	return rep, nil
}

func scheduleOutcome(jobs []HeldScheduleJob, p HeldSchedulePolicy) HeldScheduleOutcome {
	ordered := append([]HeldScheduleJob(nil), jobs...)
	sort.SliceStable(ordered, func(i, j int) bool {
		a, b := predictHeld(ordered[i], p), predictHeld(ordered[j], p)
		if a == b {
			return ordered[i].ID < ordered[j].ID
		}
		return a < b
	})
	out := HeldScheduleOutcome{Policy: p, Order: make([]string, len(ordered)), TotalDecisionOverhead: p.DecisionMS * float64(len(ordered))}
	completions := make([]float64, len(ordered))
	for i, j := range ordered {
		out.Order[i] = j.ID
		out.MakespanMS += j.ServiceMS
		completions[i] = out.MakespanMS
		out.MeanCompletionMS += out.MakespanMS
	}
	out.MeanCompletionMS /= float64(len(ordered))
	sorted := append([]float64(nil), completions...)
	sort.Float64s(sorted)
	out.P95CompletionMS = sorted[int(math.Ceil(.95*float64(len(sorted))))-1]
	return out
}

func predictHeld(j HeldScheduleJob, p HeldSchedulePolicy) float64 {
	return p.FixedMS + float64(j.UncachedInputTokens)*p.PrefillRate + float64(j.CachedInputTokens)*p.CacheReadRate + float64(j.DecodeTokens)*p.DecodeRate
}

// MeasureHeldDecisionOverhead measures only the admission calculation and stable ordering,
// reporting milliseconds per admitted request. It runs both policies over identical jobs.
func MeasureHeldDecisionOverhead(jobs []HeldScheduleJob, calibrated, scalar HeldSchedulePolicy, iterations int) (HeldSchedulePolicy, HeldSchedulePolicy, error) {
	if iterations < 1 || len(jobs) < 3 {
		return calibrated, scalar, errors.New("overhead measurement needs jobs and positive iterations")
	}
	measure := func(p HeldSchedulePolicy) float64 {
		start := time.Now()
		for n := 0; n < iterations; n++ {
			ordered := append([]HeldScheduleJob(nil), jobs...)
			sort.SliceStable(ordered, func(i, j int) bool { return predictHeld(ordered[i], p) < predictHeld(ordered[j], p) })
		}
		return float64(time.Since(start).Nanoseconds()) / 1e6 / float64(iterations*len(jobs))
	}
	calibrated.DecisionMS, scalar.DecisionMS = measure(calibrated), measure(scalar)
	calibrated.DecisionSamples, scalar.DecisionSamples = iterations*len(jobs), iterations*len(jobs)
	return calibrated, scalar, nil
}
