package modelperfobs

import (
	"encoding/json"
	"errors"
	"math"
	"reflect"
	"testing"
)

// sloFixtureLevel is one scripted estimator response.
type sloFixtureLevel struct {
	ttftMS     float64
	itlMS      float64
	e2eSeconds float64
	throughput float64
	memoryOut  bool
	err        error
}

// sloDefaultLadder is the shared scenario: throughput rises with concurrency
// until level 8, where resident memory runs out.
var sloDefaultLadder = map[int]sloFixtureLevel{
	1: {ttftMS: 100, itlMS: 20, e2eSeconds: 2.0, throughput: 10},
	2: {ttftMS: 150, itlMS: 30, e2eSeconds: 3.0, throughput: 18},
	4: {ttftMS: 260, itlMS: 55, e2eSeconds: 5.5, throughput: 30},
	8: {memoryOut: true},
}

func sloEstimator(levels map[int]sloFixtureLevel, queried *[]int) func(int) (PerformanceEstimate, error) {
	return func(concurrency int) (PerformanceEstimate, error) {
		*queried = append(*queried, concurrency)
		f := levels[concurrency]
		if f.err != nil {
			return PerformanceEstimate{}, f.err
		}
		return PerformanceEstimate{
			Concurrency:         concurrency,
			TTFTMS:              f.ttftMS,
			ITLMS:               f.itlMS,
			E2ELatencySeconds:   f.e2eSeconds,
			OutputThroughputTPS: f.throughput,
			MemoryInfeasible:    f.memoryOut,
		}, nil
	}
}

func TestSLOFeasibilitySweep(t *testing.T) {
	t.Run("selects best-throughput feasible point and prunes at memory limit", func(t *testing.T) {
		var queried []int
		res, err := SweepSLOFeasibility(SLOFeasibilitySweepInput{
			Levels:      []int{1, 2, 4, 8, 16},
			Estimate:    sloEstimator(sloDefaultLadder, &queried),
			Constraints: []SLOConstraint{{Metric: SLOMetricTTFT, StatType: SLOStatMean, Operator: SLOOpLessThanOrEqual, Value: 300, Unit: SLOUnitMilliseconds}},
		})
		if err != nil {
			t.Fatalf("sweep: %v", err)
		}
		if !res.Feasible || res.Best == nil || res.Best.Concurrency != 4 {
			t.Fatalf("best = %+v, want feasible concurrency 4", res.Best)
		}
		if res.Best.OutputThroughputTPS != 30 {
			t.Errorf("best throughput = %g, want 30", res.Best.OutputThroughputTPS)
		}
		if res.PrunedAtConcurrency != 8 {
			t.Errorf("pruned at %d, want 8", res.PrunedAtConcurrency)
		}
		// The prune must stop the walk: level 16 is never estimated.
		if !reflect.DeepEqual(queried, []int{1, 2, 4, 8}) {
			t.Errorf("estimated levels = %v, want [1 2 4 8]", queried)
		}
		statuses := make([]SLOLevelStatus, len(res.Levels))
		for i, l := range res.Levels {
			statuses[i] = l.Status
		}
		want := []SLOLevelStatus{SLOLevelEvaluated, SLOLevelEvaluated, SLOLevelEvaluated, SLOLevelMemoryInfeasible}
		if !reflect.DeepEqual(statuses, want) {
			t.Errorf("statuses = %v, want %v", statuses, want)
		}
		if res.Schema != SLOFeasibilitySweepSchema {
			t.Errorf("schema = %q, want %q", res.Schema, SLOFeasibilitySweepSchema)
		}
	})

	t.Run("stat type adjustment tightens percentile SLOs", func(t *testing.T) {
		bound := SLOConstraint{Metric: SLOMetricTTFT, StatType: SLOStatMean, Operator: SLOOpLessThanOrEqual, Value: 190, Unit: SLOUnitMilliseconds}
		var meanQueried, p95Queried []int
		meanRes, err := SweepSLOFeasibility(SLOFeasibilitySweepInput{
			Levels: []int{1, 2, 4}, Estimate: sloEstimator(sloDefaultLadder, &meanQueried), Constraints: []SLOConstraint{bound},
		})
		if err != nil {
			t.Fatalf("mean sweep: %v", err)
		}
		// Level 4 raw TTFT 260 exceeds 190; level 2 at 150 holds.
		if meanRes.Best == nil || meanRes.Best.Concurrency != 2 {
			t.Fatalf("mean best = %+v, want concurrency 2", meanRes.Best)
		}

		p95 := bound
		p95.StatType = SLOStatP95
		p95Res, err := SweepSLOFeasibility(SLOFeasibilitySweepInput{
			Levels: []int{1, 2, 4}, Estimate: sloEstimator(sloDefaultLadder, &p95Queried), Constraints: []SLOConstraint{p95},
		})
		if err != nil {
			t.Fatalf("p95 sweep: %v", err)
		}
		// Level 2 scaled by the p95 factor: 150 * 1.3 = 195 > 190, so only 1 holds.
		if p95Res.Best == nil || p95Res.Best.Concurrency != 1 {
			t.Fatalf("p95 best = %+v, want concurrency 1", p95Res.Best)
		}
	})

	t.Run("unit conversion seconds vs milliseconds", func(t *testing.T) {
		seconds := SLOConstraint{Metric: SLOMetricE2ELatency, StatType: SLOStatMean, Operator: SLOOpLessThan, Value: 4, Unit: SLOUnitSeconds}
		millis := SLOConstraint{Metric: SLOMetricE2ELatency, StatType: SLOStatMean, Operator: SLOOpLessThan, Value: 4000, Unit: SLOUnitMilliseconds}
		for _, c := range []SLOConstraint{seconds, millis} {
			var queried []int
			res, err := SweepSLOFeasibility(SLOFeasibilitySweepInput{
				Levels: []int{1, 2, 4}, Estimate: sloEstimator(sloDefaultLadder, &queried), Constraints: []SLOConstraint{c},
			})
			if err != nil {
				t.Fatalf("e2e sweep (%s): %v", c.Unit, err)
			}
			// 2.0s and 3.0s hold under 4s; 5.5s does not.
			if res.Best == nil || res.Best.Concurrency != 2 {
				t.Fatalf("unit %s: best = %+v, want concurrency 2", c.Unit, res.Best)
			}
		}
		greater := SLOConstraint{Metric: SLOMetricE2ELatency, StatType: SLOStatMean, Operator: SLOOpGreaterThan, Value: 5, Unit: SLOUnitSeconds}
		var greaterQueried []int
		greaterRes, err := SweepSLOFeasibility(SLOFeasibilitySweepInput{
			Levels: []int{1, 2, 4}, Estimate: sloEstimator(sloDefaultLadder, &greaterQueried), Constraints: []SLOConstraint{greater},
		})
		if err != nil {
			t.Fatalf("e2e greater sweep: %v", err)
		}
		if greaterRes.Best == nil || greaterRes.Best.Concurrency != 4 {
			t.Fatalf("greater best = %+v, want concurrency 4", greaterRes.Best)
		}
	})

	t.Run("skips estimator errors and continues", func(t *testing.T) {
		levels := map[int]sloFixtureLevel{
			1: {ttftMS: 100, itlMS: 20, e2eSeconds: 2.0, throughput: 10},
			2: {err: errors.New("probe failed")},
			4: {ttftMS: 260, itlMS: 55, e2eSeconds: 5.5, throughput: 30},
			8: {memoryOut: true},
		}
		var queried []int
		res, err := SweepSLOFeasibility(SLOFeasibilitySweepInput{
			Levels:   []int{1, 2, 4, 8},
			Estimate: sloEstimator(levels, &queried),
		})
		if err != nil {
			t.Fatalf("sweep: %v", err)
		}
		if !res.Feasible || res.Best == nil || res.Best.Concurrency != 4 {
			t.Fatalf("best = %+v, want feasible concurrency 4", res.Best)
		}
		if res.Levels[1].Status != SLOLevelEstimatorError || res.Levels[1].Detail != "probe failed" {
			t.Errorf("level 2 outcome = %+v, want estimator-error with detail", res.Levels[1])
		}
		if !reflect.DeepEqual(queried, []int{1, 2, 4, 8}) {
			t.Errorf("estimated levels = %v, want [1 2 4 8]", queried)
		}
	})

	t.Run("reports infeasible when no level holds", func(t *testing.T) {
		strict := SLOConstraint{Metric: SLOMetricTTFT, StatType: SLOStatMean, Operator: SLOOpLessThan, Value: 10, Unit: SLOUnitMilliseconds}
		var openQueried, prunedQueried []int
		openRes, err := SweepSLOFeasibility(SLOFeasibilitySweepInput{
			Levels: []int{1, 2, 4}, Estimate: sloEstimator(sloDefaultLadder, &openQueried), Constraints: []SLOConstraint{strict},
		})
		if err != nil {
			t.Fatalf("sweep: %v", err)
		}
		if openRes.Feasible || openRes.Best != nil {
			t.Fatalf("result = %+v, want infeasible with no best", openRes)
		}
		if openRes.Reason != "no concurrency level met the SLO constraints" {
			t.Errorf("reason = %q", openRes.Reason)
		}

		prunedRes, err := SweepSLOFeasibility(SLOFeasibilitySweepInput{
			Levels: []int{1, 2, 4}, Estimate: sloEstimator(map[int]sloFixtureLevel{
				1: {ttftMS: 100, itlMS: 20, e2eSeconds: 2.0, throughput: 10},
				2: {ttftMS: 150, itlMS: 30, e2eSeconds: 3.0, throughput: 18},
				4: {memoryOut: true},
			}, &prunedQueried),
			Constraints: []SLOConstraint{strict},
		})
		if err != nil {
			t.Fatalf("sweep: %v", err)
		}
		if prunedRes.Feasible || prunedRes.PrunedAtConcurrency != 4 {
			t.Fatalf("result = %+v, want pruned at 4 and infeasible", prunedRes)
		}
		if prunedRes.Reason != "no feasible level before memory pruning at concurrency 4" {
			t.Errorf("reason = %q", prunedRes.Reason)
		}
	})

	t.Run("ties keep the lower concurrency", func(t *testing.T) {
		levels := map[int]sloFixtureLevel{
			2: {ttftMS: 150, itlMS: 30, e2eSeconds: 3.0, throughput: 18},
			4: {ttftMS: 260, itlMS: 55, e2eSeconds: 5.5, throughput: 18},
		}
		var queried []int
		res, err := SweepSLOFeasibility(SLOFeasibilitySweepInput{
			Levels: []int{2, 4}, Estimate: sloEstimator(levels, &queried),
		})
		if err != nil {
			t.Fatalf("sweep: %v", err)
		}
		if res.Best == nil || res.Best.Concurrency != 2 {
			t.Fatalf("best = %+v, want concurrency 2 on a throughput tie", res.Best)
		}
	})

	t.Run("sentinel errors are closed", func(t *testing.T) {
		est := PerformanceEstimate{Concurrency: 1, TTFTMS: 100, ITLMS: 20, E2ELatencySeconds: 2, OutputThroughputTPS: 10}
		cases := []struct {
			name string
			make func() error
			want error
		}{
			{"unknown metric", func() error { _, err := SatisfiesSLO(est, []SLOConstraint{{Metric: "tps"}}); return err }, ErrSLOConstraintInvalid},
			{"unknown stat", func() error {
				_, err := SatisfiesSLO(est, []SLOConstraint{{Metric: SLOMetricTTFT, StatType: "p50"}})
				return err
			}, ErrSLOConstraintInvalid},
			{"unknown operator", func() error {
				_, err := SatisfiesSLO(est, []SLOConstraint{{Metric: SLOMetricTTFT, StatType: SLOStatMean, Operator: "~", Value: 1, Unit: SLOUnitMilliseconds}})
				return err
			}, ErrSLOConstraintInvalid},
			{"nan value", func() error {
				_, err := SatisfiesSLO(est, []SLOConstraint{{Metric: SLOMetricTTFT, StatType: SLOStatMean, Operator: SLOOpLessThan, Value: math.NaN(), Unit: SLOUnitMilliseconds}})
				return err
			}, ErrSLOConstraintInvalid},
			{"negative value", func() error {
				_, err := SatisfiesSLO(est, []SLOConstraint{{Metric: SLOMetricTTFT, StatType: SLOStatMean, Operator: SLOOpLessThan, Value: -1, Unit: SLOUnitMilliseconds}})
				return err
			}, ErrSLOConstraintInvalid},
			{"unknown unit", func() error {
				_, err := SatisfiesSLO(est, []SLOConstraint{{Metric: SLOMetricTTFT, StatType: SLOStatMean, Operator: SLOOpLessThan, Value: 1, Unit: "us"}})
				return err
			}, ErrSLOConstraintInvalid},
			{"negative estimate", func() error { bad := est; bad.ITLMS = -1; _, err := SatisfiesSLO(bad, nil); return err }, ErrSLOEstimateInvalid},
			{"nan estimate", func() error { bad := est; bad.TTFTMS = math.Inf(1); _, err := SatisfiesSLO(bad, nil); return err }, ErrSLOEstimateInvalid},
		}
		for _, tc := range cases {
			if err := tc.make(); !errors.Is(err, tc.want) {
				t.Errorf("%s: err = %v, want errors.Is %v", tc.name, err, tc.want)
			}
		}

		var queried []int
		ladderCases := []struct {
			name string
			in   SLOFeasibilitySweepInput
		}{
			{"nil estimator", SLOFeasibilitySweepInput{Levels: []int{1}}},
			{"empty levels", SLOFeasibilitySweepInput{Estimate: sloEstimator(sloDefaultLadder, &queried)}},
			{"descending levels", SLOFeasibilitySweepInput{Levels: []int{4, 2}, Estimate: sloEstimator(sloDefaultLadder, &queried)}},
			{"zero level", SLOFeasibilitySweepInput{Levels: []int{0, 2}, Estimate: sloEstimator(sloDefaultLadder, &queried)}},
		}
		for _, tc := range ladderCases {
			if _, err := SweepSLOFeasibility(tc.in); !errors.Is(err, ErrSLOLadderInvalid) {
				t.Errorf("%s: err = %v, want errors.Is %v", tc.name, err, ErrSLOLadderInvalid)
			}
		}
		if _, err := PowerOfTwoConcurrencyLevels(0); !errors.Is(err, ErrSLOLadderInvalid) {
			t.Errorf("PowerOfTwoConcurrencyLevels(0): err = %v, want ErrSLOLadderInvalid", err)
		}
	})

	t.Run("power of two ladder", func(t *testing.T) {
		got, err := PowerOfTwoConcurrencyLevels(100)
		if err != nil {
			t.Fatalf("ladder: %v", err)
		}
		want := []int{1, 2, 4, 8, 16, 32, 64}
		if !reflect.DeepEqual(got, want) {
			t.Errorf("ladder(100) = %v, want %v", got, want)
		}
		exact, err := PowerOfTwoConcurrencyLevels(64)
		if err != nil {
			t.Fatalf("ladder: %v", err)
		}
		if !reflect.DeepEqual(exact, want) {
			t.Errorf("ladder(64) = %v, want %v", exact, want)
		}
		one, err := PowerOfTwoConcurrencyLevels(1)
		if err != nil {
			t.Fatalf("ladder: %v", err)
		}
		if !reflect.DeepEqual(one, []int{1}) {
			t.Errorf("ladder(1) = %v, want [1]", one)
		}
	})

	t.Run("closed contract schema round trip", func(t *testing.T) {
		var queried []int
		res, err := SweepSLOFeasibility(SLOFeasibilitySweepInput{
			Levels:      []int{1, 2, 4, 8},
			Estimate:    sloEstimator(sloDefaultLadder, &queried),
			Constraints: []SLOConstraint{{Metric: SLOMetricTTFT, StatType: SLOStatMean, Operator: SLOOpLessThanOrEqual, Value: 300, Unit: SLOUnitMilliseconds}},
		})
		if err != nil {
			t.Fatalf("sweep: %v", err)
		}
		raw, err := json.Marshal(res)
		if err != nil {
			t.Fatalf("marshal: %v", err)
		}
		var back SLOFeasibilitySweepResult
		if err := json.Unmarshal(raw, &back); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}
		if back.Schema != SLOFeasibilitySweepSchema || back.Feasible != res.Feasible || back.PrunedAtConcurrency != res.PrunedAtConcurrency {
			t.Fatalf("round trip drifted: %+v", back)
		}
		if back.Best == nil || back.Best.Concurrency != 4 || back.Best.OutputThroughputTPS != 30 {
			t.Fatalf("round trip best = %+v, want concurrency 4 at 30 tps", back.Best)
		}
	})
}
