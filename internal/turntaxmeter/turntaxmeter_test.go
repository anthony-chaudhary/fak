package turntaxmeter

import (
	"bytes"
	"fmt"
	"strings"
	"testing"
	"time"
)

var (
	benchSinkBool     bool
	benchSinkReason   string
	benchSinkBudget   Budget
	benchSinkDispatch []DispatchLatency
	benchSinkObs      []HookObservation
	benchSinkSkipped  int
	benchSinkRollup   HookLatencyRollup
	benchSinkVerdict  HookLatencyVerdict
	benchSinkStat     HookLatencyStats
	benchSinkOK       bool
)

func BenchmarkSamplerShouldSample(b *testing.B) {
	rates := []int{1, 8, 64}
	for _, rate := range rates {
		b.Run(fmt.Sprintf("rate_%d", rate), func(b *testing.B) {
			s := NewSampler(rate)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				benchSinkBool = s.ShouldSample()
			}
		})
	}
}

func BenchmarkSamplerShouldSampleParallel(b *testing.B) {
	rates := []int{1, 8, 64}
	for _, rate := range rates {
		b.Run(fmt.Sprintf("rate_%d", rate), func(b *testing.B) {
			s := NewSampler(rate)
			b.ReportAllocs()
			b.ResetTimer()
			b.RunParallel(func(pb *testing.PB) {
				localBool := false
				for pb.Next() {
					localBool = s.ShouldSample()
				}
				benchSinkBool = localBool
			})
		})
	}
}

func BenchmarkCheckSpan(b *testing.B) {
	cases := []struct {
		name string
		span Span
	}{
		{
			name: "fastpath_vdso_hit",
			span: Span{Rung: "vdso", Method: "serve", ElapsedNS: 3400, TokenDelta: 0},
		},
		{
			name: "adjudicator_decide_hit",
			span: Span{Rung: "adjudicator", Method: "decide", ElapsedNS: 4500, TokenDelta: 0},
		},
		{
			name: "adjudicator_decide_breach",
			span: Span{Rung: "adjudicator", Method: "decide", ElapsedNS: 10000, TokenDelta: 0},
		},
		{
			name: "normgate_token_add",
			span: Span{Rung: "normgate", Method: "normalize", ElapsedNS: 8000, TokenDelta: 128},
		},
		{
			name: "undeclared_rung_failopen",
			span: Span{Rung: "custom_extension", Method: "evaluate", ElapsedNS: 5000, TokenDelta: 0},
		},
	}

	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			span := tc.span
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				benchSinkBool, benchSinkReason = CheckSpan(span)
			}
		})
	}
}

func BenchmarkDefaultBudget(b *testing.B) {
	cases := []struct {
		name   string
		rung   string
		method string
	}{
		{name: "declared_vdso", rung: "vdso", method: "serve"},
		{name: "declared_adjudicator", rung: "adjudicator", method: "decide"},
		{name: "declared_witness", rung: "witness", method: "confirm"},
		{name: "undeclared", rung: "nonexistent", method: "missing"},
	}

	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			r, m := tc.rung, tc.method
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				benchSinkBudget, benchSinkOK = DefaultBudget(r, m)
			}
		})
	}
}

func BenchmarkFoldDispatchLatency(b *testing.B) {
	sizes := []int{10, 100, 1000}
	phases := []string{"preflight", "gitgate", "adjudicator", "normgate", "total"}

	for _, n := range sizes {
		b.Run(fmt.Sprintf("rows_%d", n), func(b *testing.B) {
			rows := make([]map[string]int64, n)
			for i := 0; i < n; i++ {
				row := make(map[string]int64, len(phases))
				for pIdx, p := range phases {
					row[p] = int64((i + 1) * (pIdx + 1))
				}
				rows[i] = row
			}

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				benchSinkDispatch = FoldDispatchLatency(rows)
			}
		})
	}
}

func BenchmarkParseHookObservations(b *testing.B) {
	counts := []int{10, 100, 1000}
	baseTime := time.Date(2026, 7, 1, 15, 0, 0, 0, time.UTC)

	for _, count := range counts {
		b.Run(fmt.Sprintf("lines_%d", count), func(b *testing.B) {
			var sb strings.Builder
			for i := 0; i < count; i++ {
				ts := baseTime.Add(time.Duration(i) * time.Second).Format(time.RFC3339)
				switch i % 5 {
				case 0:
					sb.WriteString(fmt.Sprintf(`{"schema":{"family":"hook-observation","version":1},"verb":"posttool","outcome":"passthrough","latency_ms":%0.2f,"ts":%q}`+"\n", 50.0+float64(i%20), ts))
				case 1:
					sb.WriteString(fmt.Sprintf(`{"schema":{"family":"hook-observation","version":1},"verb":"pretool","rung":"admission","outcome":"allow","latency_ms":%0.2f,"ts":%q}`+"\n", 10.0+float64(i%30), ts))
				case 2:
					sb.WriteString(fmt.Sprintf(`{"schema":{"family":"hook-observation","version":1},"verb":"pretool","rung":"provenance","outcome":"passthrough","latency_ms":%0.2f,"ts":%q}`+"\n", 70.0+float64(i%50), ts))
				case 3:
					sb.WriteString(`{"schema":{"family":"lease-heartbeat","version":2},"latency_ms":999.0}` + "\n")
				case 4:
					sb.WriteString("malformed raw stream line\n")
				}
			}
			data := []byte(sb.String())
			b.SetBytes(int64(len(data)))
			b.ReportAllocs()
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				r := bytes.NewReader(data)
				benchSinkObs, benchSinkSkipped, _ = ParseHookObservations(r)
			}
		})
	}
}

func BenchmarkFoldHookLatency(b *testing.B) {
	counts := []int{10, 100, 1000}
	baseTime := time.Date(2026, 7, 1, 15, 0, 0, 0, time.UTC)

	for _, count := range counts {
		b.Run(fmt.Sprintf("obs_%d", count), func(b *testing.B) {
			obs := make([]HookObservation, count)
			for i := 0; i < count; i++ {
				t := baseTime.Add(time.Duration(i) * time.Second)
				switch i % 3 {
				case 0:
					obs[i] = HookObservation{Verb: "posttool", Outcome: "passthrough", LatencyMS: 60.0 + float64(i%40), At: t}
				case 1:
					obs[i] = HookObservation{Verb: "pretool", Rung: "admission", Outcome: "allow", LatencyMS: 15.0 + float64(i%25), At: t}
				case 2:
					obs[i] = HookObservation{Verb: "pretool", Rung: "provenance", Outcome: "passthrough", LatencyMS: 85.0 + float64(i%60), At: t}
				}
			}

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				benchSinkRollup = FoldHookLatency(obs)
			}
		})
	}
}

func BenchmarkJudgeHookLatency(b *testing.B) {
	cases := []struct {
		name   string
		total  HookLatencyStats
		budget float64
	}{
		{
			name:   "thin_sample",
			total:  HookLatencyStats{Count: 5, P99MS: 300.0},
			budget: 250.0,
		},
		{
			name:   "within_budget",
			total:  HookLatencyStats{Count: 100, P99MS: 180.0},
			budget: 250.0,
		},
		{
			name:   "breach_regression",
			total:  HookLatencyStats{Count: 100, P99MS: 320.0},
			budget: 250.0,
		},
		{
			name:   "failopen_no_budget",
			total:  HookLatencyStats{Count: 100, P99MS: 999.0},
			budget: 0.0,
		},
	}

	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			total := tc.total
			budget := tc.budget
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				benchSinkVerdict = JudgeHookLatency(total, budget)
			}
		})
	}
}

func BenchmarkFilterHookObservationsSince(b *testing.B) {
	const count = 1000
	baseTime := time.Date(2026, 7, 1, 15, 0, 0, 0, time.UTC)
	obs := make([]HookObservation, count)
	for i := 0; i < count; i++ {
		obs[i] = HookObservation{
			Verb:      "pretool",
			Rung:      "admission",
			LatencyMS: 20.0 + float64(i%10),
			At:        baseTime.Add(time.Duration(i) * time.Minute),
		}
	}
	cutoff := baseTime.Add(500 * time.Minute)

	b.Run("all_time_zero_cutoff", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkObs = FilterHookObservationsSince(obs, time.Time{})
		}
	})

	b.Run("window_half_retained", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkObs = FilterHookObservationsSince(obs, cutoff)
		}
	})
}

func BenchmarkTailRung(b *testing.B) {
	obs := make([]HookObservation, 100)
	baseTime := time.Date(2026, 7, 1, 15, 0, 0, 0, time.UTC)
	for i := 0; i < 100; i++ {
		t := baseTime.Add(time.Duration(i) * time.Second)
		r := "admission"
		if i%2 == 0 {
			r = "provenance"
		}
		obs[i] = HookObservation{Verb: "pretool", Rung: r, LatencyMS: float64(i), At: t}
	}
	rollup := FoldHookLatency(obs)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkStat, benchSinkOK = rollup.TailRung()
	}
}
