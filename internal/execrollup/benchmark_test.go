package execrollup

import (
	"testing"
)

var (
	sinkRollup Rollup
	sinkString string
	sinkBytes  []byte
)

func sampleCleanInputs() Inputs {
	return Inputs{
		Workspace:   "/work/fak",
		Commit:      "c0ffee1",
		GeneratedAt: "2026-09-06T12:00:00Z",
		Dispatch: PlaneInput{
			Payload: map[string]any{
				"closure": map[string]any{
					"na":           false,
					"closure_rate": 0.95,
					"counts": map[string]any{
						"TRUE_RESOLVED":  950,
						"CLAIMED_CLOSED": 50,
					},
					"open_witnessed_closable": 0,
				},
				"throughput": map[string]any{
					"na":                      false,
					"verdict":                 "ON_TARGET",
					"completed_rate_per_hour": 15.0,
					"target_per_hour":         10.0,
					"primary_window_hours":    6,
				},
				"backend_health": map[string]any{"dead_count": 0},
				"workers":        map[string]any{"silent_count": 0},
				"backlog":        map[string]any{"open_issues": 30},
			},
		},
		Loops: PlaneInput{
			Payload: map[string]any{
				"rollup": map[string]any{
					"loops": 8,
					"live":  8,
					"dark":  0,
				},
			},
		},
		Cadence: PlaneInput{
			Payload: map[string]any{
				"scores": map[string]any{
					"debt":            0,
					"trend_direction": "flat",
				},
				"work": map[string]any{
					"commits":     120,
					"ships":       110,
					"window_days": 7,
				},
			},
		},
		Fleet: PlaneInput{
			Payload: map[string]any{
				"total":     4,
				"reachable": 4,
				"attention": []any{
					map[string]any{"level": "ok", "title": "box-1 healthy"},
				},
			},
		},
	}
}

func sampleActiveInputs() Inputs {
	return Inputs{
		Workspace:   "/work/fak",
		Commit:      "deadbeef",
		GeneratedAt: "2026-09-06T12:00:00Z",
		Dispatch: PlaneInput{
			Payload: map[string]any{
				"closure": map[string]any{
					"na":           false,
					"closure_rate": 0.72,
					"counts": map[string]any{
						"TRUE_RESOLVED":  180,
						"CLAIMED_CLOSED": 70,
					},
					"open_witnessed_closable": 5,
				},
				"throughput": map[string]any{
					"na":                      false,
					"verdict":                 "BELOW_TARGET",
					"completed_rate_per_hour": 3.5,
					"target_per_hour":         8.0,
					"primary_window_hours":    12,
				},
				"backend_health": map[string]any{"dead_count": 1},
				"workers":        map[string]any{"silent_count": 2},
				"backlog":        map[string]any{"open_issues": 85},
			},
		},
		Loops: PlaneInput{
			Payload: map[string]any{
				"rollup": map[string]any{
					"loops": 10,
					"live":  7,
					"dark":  3,
				},
			},
		},
		Cadence: PlaneInput{
			Payload: map[string]any{
				"scores": map[string]any{
					"debt":            14,
					"trend_direction": "regressed",
					"trend_summary":   "test coverage -4%",
				},
				"work": map[string]any{
					"commits":     80,
					"ships":       50,
					"window_days": 14,
				},
				"maturity": map[string]any{
					"debt":                   3,
					"route_lane":             "advmodel",
					"route_item":             "dogfood capability",
					"route_key":              "maturity/advmodel/dogfood",
					"route_skipped_private": 2,
				},
			},
		},
		Fleet: PlaneInput{
			Payload: map[string]any{
				"total":     8,
				"reachable": 6,
				"attention": []any{
					map[string]any{
						"level":  "crit",
						"title":  "gpu-node-3 offline",
						"detail": "unreachable for 45m",
					},
					map[string]any{
						"level":  "warn",
						"title":  "gpu-node-5 throttled",
						"detail": "thermal limit reached",
					},
					map[string]any{
						"level": "ok",
						"title": "gpu-node-1 running",
					},
				},
			},
		},
	}
}

func sampleDegradedInputs() Inputs {
	return Inputs{
		Workspace:   "/work/fak",
		Commit:      "feedface",
		GeneratedAt: "2026-09-06T12:00:00Z",
		Dispatch: PlaneInput{
			Err: "dispatch collector timeout after 30s",
		},
		Loops: PlaneInput{
			Payload: map[string]any{
				"rollup": map[string]any{
					"loops": 4,
					"live":  3,
					"dark":  1,
				},
			},
		},
		Cadence: PlaneInput{
			Err: "cadence git read failed",
		},
		Fleet: PlaneInput{
			Payload: map[string]any{
				"total":     2,
				"reachable": 2,
				"attention": []any{},
			},
		},
	}
}

// BenchmarkFoldClean measures Fold performance on a clean, fully-measured fleet.
func BenchmarkFoldClean(b *testing.B) {
	in := sampleCleanInputs()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkRollup = Fold(in)
	}
}

// BenchmarkFoldRollup measures Fold performance across clean, active attention, and degraded scenarios.
func BenchmarkFoldRollup(b *testing.B) {
	cases := []struct {
		name string
		in   Inputs
	}{
		{"clean", sampleCleanInputs()},
		{"active_attention", sampleActiveInputs()},
		{"degraded_unmeasured", sampleDegradedInputs()},
	}
	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			in := tc.in
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				sinkRollup = Fold(in)
			}
		})
	}
}

// BenchmarkFoldDegraded measures Fold performance specifically under partial collector failures.
func BenchmarkFoldDegraded(b *testing.B) {
	in := sampleDegradedInputs()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkRollup = Fold(in)
	}
}

// BenchmarkRender measures markdown rendering of clean, active, and degraded Rollup envelopes.
func BenchmarkRender(b *testing.B) {
	cleanRollup := Fold(sampleCleanInputs())
	activeRollup := Fold(sampleActiveInputs())
	degradedRollup := Fold(sampleDegradedInputs())

	cases := []struct {
		name string
		r    Rollup
	}{
		{"clean", cleanRollup},
		{"active_attention", activeRollup},
		{"degraded_unmeasured", degradedRollup},
	}
	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			r := tc.r
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				sinkString = Render(r)
			}
		})
	}
}

// BenchmarkJSON measures JSON serialization of the executive roll-up envelope.
func BenchmarkJSON(b *testing.B) {
	r := Fold(sampleActiveInputs())
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkBytes, _ = JSON(r)
	}
}
