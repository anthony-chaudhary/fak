package modelperfobs

import (
	"encoding/json"
	"errors"
	"math"
	"reflect"
	"testing"
)

// sweepRecordRow builds one well-formed row: a config identity plus measured
// throughputs, with latency statistics folded from samples so the schema test
// also covers the sample-to-statistics seam.
func sweepRecordRow(t *testing.T, serverArg, clientArg string, ttftSamples []float64, inTP, outTP float64) SweepTestResult {
	t.Helper()
	stats, err := SweepLatencyStatsFromSamples(ttftSamples)
	if err != nil {
		t.Fatalf("stats from samples %v: %v", ttftSamples, err)
	}
	return SweepTestResult{
		Config: SweepConfig{
			ServerArgs: SweepArgs{"tp_size": serverArg},
			ClientArgs: SweepArgs{"max_concurrency": clientArg},
		},
		Results: SweepMetrics{
			Completed:            4,
			RequestThroughputRPS: 2,
			InputThroughputTPS:   inTP,
			OutputThroughputTPS:  outTP,
			TTFT:                 stats,
			ITL:                  SweepLatencyStats{MeanMS: 8, MedianMS: 7, P95MS: 12, P99MS: 15},
			TPOT:                 SweepLatencyStats{MeanMS: 6, MedianMS: 6, P95MS: 9, P99MS: 11},
			E2ELatency:           SweepLatencyStats{MeanMS: 900, MedianMS: 880, P95MS: 1200, P99MS: 1400},
		},
	}
}

// fak-test:justify why=contract when=changed:internal/modelperfobs/**
// fak-test:runtime fast est=10ms
func TestSweepRecordSchema(t *testing.T) {
	t.Run("statistics are the four closed percentiles of the samples", func(t *testing.T) {
		stats, err := SweepLatencyStatsFromSamples([]float64{10, 20, 30, 40})
		if err != nil {
			t.Fatalf("stats: %v", err)
		}
		want := SweepLatencyStats{MeanMS: 25, MedianMS: 30, P95MS: 40, P99MS: 40}
		if stats != want {
			t.Errorf("stats = %+v, want %+v", stats, want)
		}
	})

	t.Run("config identity is order-independent content addressing", func(t *testing.T) {
		a := SweepConfig{
			ServerArgs: SweepArgs{"tp_size": "2", "attention_backend": "flashinfer"},
			ClientArgs: SweepArgs{"max_concurrency": "64"},
		}
		b := SweepConfig{
			ServerArgs: SweepArgs{"attention_backend": "flashinfer", "tp_size": "2"},
			ClientArgs: SweepArgs{"max_concurrency": "64"},
		}
		if a.ID() != b.ID() {
			t.Errorf("arg map order changed the config id: %q vs %q", a.ID(), b.ID())
		}
		changed := a
		changed.ClientArgs = SweepArgs{"max_concurrency": "128"}
		if changed.ID() == a.ID() {
			t.Errorf("a changed client arg kept the config id %q", a.ID())
		}
	})

	t.Run("best configurations select the throughput winners", func(t *testing.T) {
		slow := sweepRecordRow(t, "1", "32", []float64{100, 120, 140, 160}, 500, 40)
		fast := sweepRecordRow(t, "2", "64", []float64{90, 110, 130, 150}, 900, 70)
		record, err := BuildSweepRecord([]SweepTestResult{slow, fast}, SweepMetadata{GPUType: "H100", GPUCount: 1, ModelTag: "qwen3.8"}, nil)
		if err != nil {
			t.Fatalf("build: %v", err)
		}
		if record.Schema != SweepRecordSchema {
			t.Errorf("schema = %q, want %q", record.Schema, SweepRecordSchema)
		}
		if record.Metadata.TotalTests != 2 {
			t.Errorf("total_tests = %d, want 2", record.Metadata.TotalTests)
		}
		if got := record.BestConfigurations.BestInputThroughput; got == nil || got.Config.ServerArgs["tp_size"] != "2" {
			t.Errorf("best input throughput = %+v, want the tp_size=2 config", got)
		}
		if got := record.BestConfigurations.BestOutputThroughput; got == nil || got.ThroughputTPS != 70 {
			t.Errorf("best output throughput = %+v, want 70", got)
		}
		// The sweep declared no constraints, so the constrained slots stay unset.
		if record.BestConfigurations.BestInputThroughputConstrained != nil ||
			record.BestConfigurations.BestOutputThroughputConstrained != nil {
			t.Errorf("unconstrained sweep filled constrained bests: %+v", record.BestConfigurations)
		}
	})

	t.Run("constrained bests read the measured statistic the constraint names", func(t *testing.T) {
		// Row one wins raw output throughput but misses the p99 TTFT bound; row
		// two is slower and holds every constraint.
		overTail := sweepRecordRow(t, "1", "128", []float64{1000, 1100, 1200, 1300}, 900, 70)
		within := sweepRecordRow(t, "2", "64", []float64{100, 120, 140, 160}, 500, 40)
		constraints := []SLOConstraint{{Metric: SLOMetricTTFT, StatType: SLOStatP99, Operator: SLOOpLessThanOrEqual, Value: 500, Unit: SLOUnitMilliseconds}}
		record, err := BuildSweepRecord([]SweepTestResult{overTail, within}, SweepMetadata{}, constraints)
		if err != nil {
			t.Fatalf("build: %v", err)
		}
		if got := record.BestConfigurations.BestOutputThroughput; got == nil || got.ThroughputTPS != 70 {
			t.Fatalf("unconstrained best output = %+v, want the 70 tps row", got)
		}
		constrained := record.BestConfigurations.BestOutputThroughputConstrained
		if constrained == nil || constrained.ThroughputTPS != 40 {
			t.Fatalf("constrained best output = %+v, want the 40 tps row that holds p99<=500ms", constrained)
		}
		if constrained.Config.ServerArgs["tp_size"] != "2" {
			t.Errorf("constrained winner = %+v, want the tp_size=2 config", constrained)
		}
		// A median constraint reads the median, not a tail: the 160ms-tail row
		// still has a 140ms median and holds a 150ms median bound.
		medianBound := []SLOConstraint{{Metric: SLOMetricTTFT, StatType: SLOStatMedian, Operator: SLOOpLessThanOrEqual, Value: 150, Unit: SLOUnitMilliseconds}}
		medianRecord, err := BuildSweepRecord([]SweepTestResult{within}, SweepMetadata{}, medianBound)
		if err != nil {
			t.Fatalf("median build: %v", err)
		}
		if got := medianRecord.BestConfigurations.BestOutputThroughputConstrained; got == nil {
			t.Errorf("median constraint 150ms rejected a row with median 140ms")
		}
	})

	t.Run("best-configuration ties resolve by config identity", func(t *testing.T) {
		first := sweepRecordRow(t, "2", "64", []float64{100}, 800, 60)
		second := sweepRecordRow(t, "1", "64", []float64{100}, 800, 60)
		forward, err := BuildSweepRecord([]SweepTestResult{first, second}, SweepMetadata{}, nil)
		if err != nil {
			t.Fatalf("forward build: %v", err)
		}
		reverse, err := BuildSweepRecord([]SweepTestResult{second, first}, SweepMetadata{}, nil)
		if err != nil {
			t.Fatalf("reverse build: %v", err)
		}
		if forward.BestConfigurations.BestOutputThroughput == nil || reverse.BestConfigurations.BestOutputThroughput == nil {
			t.Fatal("tie produced no best output throughput")
		}
		if forward.BestConfigurations.BestOutputThroughput.Config.ID() !=
			reverse.BestConfigurations.BestOutputThroughput.Config.ID() {
			t.Errorf("row order changed the tied winner: %+v vs %+v",
				forward.BestConfigurations.BestOutputThroughput, reverse.BestConfigurations.BestOutputThroughput)
		}
	})

	t.Run("invalid rows are refused with the sentinel", func(t *testing.T) {
		good := sweepRecordRow(t, "1", "32", []float64{100, 120}, 500, 40)
		cases := []struct {
			name    string
			results []SweepTestResult
			meta    SweepMetadata
			cons    []SLOConstraint
		}{
			{
				name:    "config without an identity",
				results: []SweepTestResult{{Config: SweepConfig{}, Results: good.Results}},
			},
			{
				name:    "nan measured statistic",
				results: []SweepTestResult{{Config: good.Config, Results: func() SweepMetrics { m := good.Results; m.TTFT.P95MS = math.NaN(); return m }()}},
			},
			{
				name:    "negative throughput",
				results: []SweepTestResult{{Config: good.Config, Results: func() SweepMetrics { m := good.Results; m.OutputThroughputTPS = -1; return m }()}},
			},
			{
				name:    "total_tests disagreement",
				results: []SweepTestResult{good},
				meta:    SweepMetadata{TotalTests: 5},
			},
			{
				name:    "constraint outside the sweep axes",
				results: []SweepTestResult{good},
				cons:    []SLOConstraint{{Metric: SLOMetricTTFT, StatType: "p50", Operator: SLOOpLessThan, Value: 1, Unit: SLOUnitMilliseconds}},
			},
		}
		for _, tc := range cases {
			if _, err := BuildSweepRecord(tc.results, tc.meta, tc.cons); !errors.Is(err, ErrSweepRecordInvalid) && !errors.Is(err, ErrSLOConstraintInvalid) {
				t.Errorf("%s: err = %v, want errors.Is ErrSweepRecordInvalid or ErrSLOConstraintInvalid", tc.name, err)
			}
		}
		for _, bad := range [][]float64{nil, {}, {10, math.Inf(1)}, {10, -1}} {
			if _, err := SweepLatencyStatsFromSamples(bad); !errors.Is(err, ErrSweepRecordInvalid) {
				t.Errorf("samples %v: err = %v, want errors.Is ErrSweepRecordInvalid", bad, err)
			}
		}
	})

	t.Run("closed schema round trip", func(t *testing.T) {
		row := sweepRecordRow(t, "2", "64", []float64{90, 110, 130, 150}, 900, 70)
		constraints := []SLOConstraint{{Metric: SLOMetricE2ELatency, StatType: SLOStatP99, Operator: SLOOpLessThanOrEqual, Value: 2, Unit: SLOUnitSeconds}}
		record, err := BuildSweepRecord([]SweepTestResult{row}, SweepMetadata{ModelTag: "qwen3.8", TotalTests: 1}, constraints)
		if err != nil {
			t.Fatalf("build: %v", err)
		}
		raw, err := json.Marshal(record)
		if err != nil {
			t.Fatalf("marshal: %v", err)
		}
		var back SweepRecord
		if err := json.Unmarshal(raw, &back); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}
		if !reflect.DeepEqual(back, record) {
			t.Errorf("round trip drifted:\n got %+v\nwant %+v", back, record)
		}
		if back.BestConfigurations.BestOutputThroughputConstrained == nil {
			t.Errorf("round trip lost the constrained best: %s", raw)
		}
	})
}
