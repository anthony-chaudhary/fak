package benchscore

import (
	"errors"
	"testing"
)

// TestSLOConstraintGrammar pins the SLO constraint mini-grammar
// (metric[:stat]<op><value>[unit]) adapted from bentoml/llm-optimizer
// llm-optimizer-1: parser shape, sentinel errors, and evaluator semantics.
func TestSLOConstraintGrammar(t *testing.T) {
	t.Run("Parse", func(t *testing.T) {
		tests := []struct {
			in      string
			want    []SLOConstraint
			wantErr error
		}{
			{
				in:   "ttft<300ms",
				want: []SLOConstraint{{Metric: "ttft", Stat: "mean", Op: "<", Value: 300, Unit: "ms"}},
			},
			{
				in:   "ttft:median<300ms",
				want: []SLOConstraint{{Metric: "ttft", Stat: "median", Op: "<", Value: 300, Unit: "ms"}},
			},
			{
				in: "itl:p95<8.5ms;ttft:p99<0.5s",
				want: []SLOConstraint{
					{Metric: "itl", Stat: "p95", Op: "<", Value: 8.5, Unit: "ms"},
					{Metric: "ttft", Stat: "p99", Op: "<", Value: 0.5, Unit: "s"},
				},
			},
			{
				// Case-insensitive stat, seconds unit, two-character operator.
				in:   "TTFT:P95<=1s",
				want: []SLOConstraint{{Metric: "TTFT", Stat: "p95", Op: "<=", Value: 1, Unit: "s"}},
			},
			{
				// Bare number defaults to milliseconds; ">" and ">=" operators.
				in: "itl>8;e2e_latency>=2",
				want: []SLOConstraint{
					{Metric: "itl", Stat: "mean", Op: ">", Value: 8, Unit: "ms"},
					{Metric: "e2e_latency", Stat: "mean", Op: ">=", Value: 2, Unit: "ms"},
				},
			},
			{
				// Whitespace around parts and empty parts are tolerated.
				in: " ttft < 300ms ; ; itl:p99 != 4 ms ",
				want: []SLOConstraint{
					{Metric: "ttft", Stat: "mean", Op: "<", Value: 300, Unit: "ms"},
					{Metric: "itl", Stat: "p99", Op: "!=", Value: 4, Unit: "ms"},
				},
			},
			{in: "", want: nil},
			{in: "   ;  ", want: nil},
			{
				in:      "ttft300ms",
				wantErr: ErrSLONoOperator,
			},
			{
				in:      ":mean<300ms",
				wantErr: ErrSLOEmptyMetric,
			},
			{
				in:      "ttft:p50<300ms",
				wantErr: ErrSLOUnknownStat,
			},
			{
				in:      "ttft<abc",
				wantErr: ErrSLOInvalidValue,
			},
			{
				in:      "ttft<NaN",
				wantErr: ErrSLOInvalidValue,
			},
		}
		for _, tc := range tests {
			got, err := ParseSLOConstraints(tc.in)
			if tc.wantErr != nil {
				if !errors.Is(err, tc.wantErr) {
					t.Fatalf("ParseSLOConstraints(%q) error = %v, want errors.Is %v", tc.in, err, tc.wantErr)
				}
				continue
			}
			if err != nil {
				t.Fatalf("ParseSLOConstraints(%q) unexpected error: %v", tc.in, err)
			}
			if len(got) != len(tc.want) {
				t.Fatalf("ParseSLOConstraints(%q) = %+v, want %+v", tc.in, got, tc.want)
			}
			for i := range got {
				if got[i] != tc.want[i] {
					t.Fatalf("ParseSLOConstraints(%q)[%d] = %+v, want %+v", tc.in, i, got[i], tc.want[i])
				}
			}
		}
	})

	t.Run("Evaluate", func(t *testing.T) {
		samples := map[string][]float64{
			// 1..100 ms: mean 50.5, median 50.5, p95 95, p99 99 (nearest-rank).
			"ttft": seqMS(1, 100),
			// Odd count 9: mean 11, median is the middle (5th) value 11, p95
			// is the ceil(0.95*9)=9th smallest value, 19.
			"itl": {3, 5, 7, 9, 11, 13, 15, 17, 19},
		}

		tests := []struct {
			name     string
			cons     string
			wantPass bool
			wantObs  float64
		}{
			{"mean under bound", "ttft<51ms", true, 50.5},
			{"mean over bound", "ttft<50ms", false, 50.5},
			{"median exact boundary", "ttft:median<=50.5ms", true, 50.5},
			{"p95 nearest rank", "ttft:p95<96ms", true, 95},
			{"p99 nearest rank", "ttft:p99<99ms", false, 99},
			{"seconds unit converts", "ttft:p95<0.1s", true, 0.095},
			{"odd median", "itl:median<11.5ms", true, 11},
			{"p95 small sample clamps to max", "itl:p95<20ms", true, 19},
			{"greater operator on mean", "itl>5ms", true, 11},
			{"equality exact", "itl:median==11ms", true, 11},
			{"inequality", "itl:median!=11ms", false, 11},
		}
		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				cons, err := ParseSLOConstraints(tc.cons)
				if err != nil {
					t.Fatalf("ParseSLOConstraints(%q): %v", tc.cons, err)
				}
				verdicts, err := EvaluateSLO(cons, samples)
				if err != nil {
					t.Fatalf("EvaluateSLO: %v", err)
				}
				if len(verdicts) != 1 {
					t.Fatalf("verdicts = %+v, want exactly one", verdicts)
				}
				v := verdicts[0]
				if v.Pass != tc.wantPass {
					t.Fatalf("%s: pass = %v, want %v (observed %v)", tc.cons, v.Pass, tc.wantPass, v.Observed)
				}
				if v.Observed != tc.wantObs {
					t.Fatalf("%s: observed = %v, want %v", tc.cons, v.Observed, tc.wantObs)
				}
			})
		}
	})

	t.Run("FeasibleAndMissingSamples", func(t *testing.T) {
		samples := map[string][]float64{"ttft": {100, 120}} // mean 110 ms
		ok, err := SLOFeasible([]SLOConstraint{{Metric: "ttft", Stat: "mean", Op: "<", Value: 111, Unit: "ms"}}, samples)
		if err != nil || !ok {
			t.Fatalf("SLOFeasible = (%v, %v), want (true, nil)", ok, err)
		}
		ok, err = SLOFeasible([]SLOConstraint{{Metric: "ttft", Stat: "mean", Op: "<", Value: 100, Unit: "ms"}}, samples)
		if err != nil || ok {
			t.Fatalf("SLOFeasible = (%v, %v), want (false, nil)", ok, err)
		}
		_, err = EvaluateSLO([]SLOConstraint{{Metric: "itl", Stat: "mean", Op: "<", Value: 10, Unit: "ms"}}, samples)
		if !errors.Is(err, ErrSLOMissingSamples) {
			t.Fatalf("EvaluateSLO missing metric error = %v, want errors.Is %v", err, ErrSLOMissingSamples)
		}
	})
}

func seqMS(lo, hi int) []float64 {
	out := make([]float64, 0, hi-lo+1)
	for i := lo; i <= hi; i++ {
		out = append(out, float64(i))
	}
	return out
}
