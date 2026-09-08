package doomloop

import (
	"bytes"
	"fmt"
	"strings"
	"testing"
)

var (
	benchResultSink    Result
	benchNudgeSink     PredictiveNudge
	benchStringSink    string
	benchReportSink    CalibrationReport
	benchDecisionsSink []Decision
	benchDistSink      Distribution
)

// BenchmarkDoomLoop measures baseline classification latency for a confirmed
// doom-loop sample stream.
func BenchmarkDoomLoop(b *testing.B) {
	samples := []Sample{
		{UnixMillis: 0, Effort: 10, Progress: 4, Alive: true},
		{UnixMillis: 60000, Effort: 20, Progress: 4, Alive: true},
		{UnixMillis: 120000, Effort: 30, Progress: 4, Alive: true},
		{UnixMillis: 180000, Effort: 40, Progress: 4, Alive: true},
		{UnixMillis: 240000, Effort: 50, Progress: 4, Alive: true},
		{UnixMillis: 300000, Effort: 60, Progress: 4, Alive: true},
	}
	cfg := DefaultConfig()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res := Classify(samples, cfg)
		if res.Verdict != VerdictDoomLoop {
			b.Fatalf("expected DOOM_LOOP, got %v", res.Verdict)
		}
		benchResultSink = res
	}
}

// BenchmarkClassify benchmarks Classify across distinct execution branches:
// confirmed doom loop, advancing progress, quiet idle worker, frozen wedged worker,
// and long sample history streams.
func BenchmarkClassify(b *testing.B) {
	cfg := DefaultConfig()

	doomSamples := []Sample{
		{UnixMillis: 0, Effort: 10, Progress: 1, Alive: true},
		{UnixMillis: 1000, Effort: 20, Progress: 1, Alive: true},
		{UnixMillis: 2000, Effort: 30, Progress: 1, Alive: true},
		{UnixMillis: 3000, Effort: 40, Progress: 1, Alive: true},
		{UnixMillis: 4000, Effort: 50, Progress: 1, Alive: true},
	}

	healthySamples := []Sample{
		{UnixMillis: 0, Effort: 10, Progress: 1, Alive: true},
		{UnixMillis: 1000, Effort: 20, Progress: 2, Alive: true},
		{UnixMillis: 2000, Effort: 30, Progress: 3, Alive: true},
		{UnixMillis: 3000, Effort: 40, Progress: 4, Alive: true},
	}

	idleSamples := []Sample{
		{UnixMillis: 0, Effort: 10, Progress: 1, Alive: true},
		{UnixMillis: 1000, Effort: 10, Progress: 1, Alive: true},
		{UnixMillis: 2000, Effort: 10, Progress: 1, Alive: true},
	}

	wedgedSamples := []Sample{
		{UnixMillis: 0, Effort: 10, Progress: 1, Alive: false},
		{UnixMillis: 1000, Effort: 10, Progress: 1, Alive: false},
		{UnixMillis: 2000, Effort: 10, Progress: 1, Alive: false},
	}

	longSamples := make([]Sample, 128)
	for i := range longSamples {
		longSamples[i] = Sample{
			UnixMillis: int64(i * 1000),
			Effort:     int64(i * 10),
			Progress:   int64(i / 10),
			Alive:      true,
		}
	}

	b.Run("DoomLoop", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchResultSink = Classify(doomSamples, cfg)
		}
	})

	b.Run("HealthyProgress", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchResultSink = Classify(healthySamples, cfg)
		}
	})

	b.Run("Idle", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchResultSink = Classify(idleSamples, cfg)
		}
	})

	b.Run("Wedged", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchResultSink = Classify(wedgedSamples, cfg)
		}
	})

	b.Run("DeepHistory128", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchResultSink = Classify(longSamples, cfg)
		}
	})
}

// BenchmarkResultInterpretation measures human-readable interpretation formatting.
func BenchmarkResultInterpretation(b *testing.B) {
	results := []Result{
		{Verdict: VerdictDoomLoop, Correction: CorrectNudge},
		{Verdict: VerdictDoomLoop, Correction: CorrectEscalate},
		{Verdict: VerdictHealthy, Correction: CorrectObserve},
		{Verdict: VerdictHealthy, Correction: CorrectNone},
		{Verdict: VerdictIdle, Correction: CorrectNone},
		{Verdict: VerdictWedged, Correction: CorrectNone},
		{Verdict: VerdictUnknown, Correction: CorrectNone},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchStringSink = results[i%len(results)].Interpretation()
	}
}

// BenchmarkPredictiveNudge exercises turn-by-turn predictive nudge evaluation.
func BenchmarkPredictiveNudge(b *testing.B) {
	tracker := NewPredictiveNudgeTracker("bench-objective", 6)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		pn := tracker.RecordTurn(i%4 == 0, "simulated compile error")
		benchNudgeSink = pn
	}
}

// BenchmarkEvaluatePredictiveNudge benchmarks the pure nudge decision evaluation.
func BenchmarkEvaluatePredictiveNudge(b *testing.B) {
	b.Run("BelowThreshold", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchNudgeSink = EvaluatePredictiveNudge("bench-task", 1, 6, "err")
		}
	})

	b.Run("TriggeredNudge", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchNudgeSink = EvaluatePredictiveNudge("bench-task", 3, 6, "err")
		}
	})

	b.Run("FullHaltBoundary", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchNudgeSink = EvaluatePredictiveNudge("bench-task", 6, 6, "err")
		}
	})
}

// BenchmarkPredictiveNudgeTracker benchmarks tracker operations under state mutation.
func BenchmarkPredictiveNudgeTracker(b *testing.B) {
	b.Run("RecordTurnMixed", func(b *testing.B) {
		t := NewPredictiveNudgeTracker("tracker-bench", 6)
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchNudgeSink = t.RecordTurn(i%3 == 0, "syntax error")
		}
	})

	b.Run("RecordFlatTurn", func(b *testing.B) {
		t := NewPredictiveNudgeTracker("tracker-bench", 6)
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchNudgeSink = t.RecordFlatTurn("compile error")
		}
	})

	b.Run("EvaluateReadOnly", func(b *testing.B) {
		t := NewPredictiveNudgeTracker("tracker-bench", 6)
		t.RecordFlatTurn("test fail")
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchNudgeSink = t.Evaluate()
		}
	})

	b.Run("SetObjectiveReset", func(b *testing.B) {
		t := NewPredictiveNudgeTracker("obj-a", 6)
		objectives := []string{"obj-a", "obj-b", "obj-c"}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			t.SetObjective(objectives[i%len(objectives)])
		}
	})
}

// BenchmarkCalibrate exercises offline threshold calibration over decision streams.
func BenchmarkCalibrate(b *testing.B) {
	cfg := DefaultCalibrateConfig()

	smallDecisions := []Decision{
		{Session: "w1", Streak: 1, Correction: "OBSERVE", UnixMillis: 1000},
		{Session: "w1", Streak: 2, Correction: "OBSERVE", UnixMillis: 2000},
		{Session: "w1", Streak: 3, Correction: "NUDGE", UnixMillis: 3000},
		{Session: "w1", Streak: 0, Correction: "NONE", UnixMillis: 4000},

		{Session: "w2", Streak: 1, Correction: "OBSERVE", UnixMillis: 1100},
		{Session: "w2", Streak: 2, Correction: "OBSERVE", UnixMillis: 2100},
		{Session: "w2", Streak: 3, Correction: "NUDGE", UnixMillis: 3100},
		{Session: "w2", Streak: 4, Correction: "NUDGE", UnixMillis: 4100},
		{Session: "w2", Streak: 0, Correction: "NONE", UnixMillis: 5100},

		{Session: "w3", Streak: 1, Correction: "OBSERVE", UnixMillis: 1200},
		{Session: "w3", Streak: 2, Correction: "OBSERVE", UnixMillis: 2200},
		{Session: "w3", Streak: 3, Correction: "NUDGE", UnixMillis: 3200},
		{Session: "w3", Streak: 4, Correction: "NUDGE", UnixMillis: 4200},
		{Session: "w3", Streak: 5, Correction: "NUDGE", UnixMillis: 5200},
		{Session: "w3", Streak: 6, Correction: "ESCALATE", UnixMillis: 6200},
		{Session: "w3", Streak: 0, Correction: "NONE", UnixMillis: 7200},
	}

	largeDecisions := make([]Decision, 180)
	for i := range largeDecisions {
		sess := fmt.Sprintf("worker-%d", i%10)
		streak := (i % 6)
		correction := "OBSERVE"
		if streak >= 3 {
			correction = "NUDGE"
		}
		if streak == 0 {
			correction = "NONE"
		}
		largeDecisions[i] = Decision{
			Session:    sess,
			Streak:     streak,
			Correction: correction,
			UnixMillis: int64(i * 1000),
		}
	}

	b.Run("SmallCohort", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchReportSink = Calibrate(smallDecisions, cfg)
		}
	})

	b.Run("LargeCohort180", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchReportSink = Calibrate(largeDecisions, cfg)
		}
	})
}

// BenchmarkParseDecisions benchmarks JSONL decision stream deserialization.
func BenchmarkParseDecisions(b *testing.B) {
	raw := `{"session":"sess-1","unix_millis":1000,"verdict":"HEALTHY","correction":"OBSERVE","reason":"watching","burning_flat_streak":1,"effort_delta":10,"progress_delta":0,"samples":3}
{"session":"sess-1","unix_millis":2000,"verdict":"DOOM_LOOP","correction":"NUDGE","reason":"confirmed","burning_flat_streak":3,"effort_delta":10,"progress_delta":0,"samples":4}
{"session":"sess-1","unix_millis":3000,"verdict":"HEALTHY","correction":"NONE","reason":"progress","burning_flat_streak":0,"effort_delta":10,"progress_delta":1,"samples":5}
{"session":"sess-2","unix_millis":1500,"verdict":"HEALTHY","correction":"OBSERVE","reason":"watching","burning_flat_streak":2,"effort_delta":15,"progress_delta":0,"samples":3}
{"session":"sess-2","unix_millis":2500,"verdict":"DOOM_LOOP","correction":"ESCALATE","reason":"persistent","burning_flat_streak":6,"effort_delta":15,"progress_delta":0,"samples":7}
`
	rawBytes := []byte(strings.Repeat(raw, 5))

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ds, err := ParseDecisions(bytes.NewReader(rawBytes))
		if err != nil {
			b.Fatalf("ParseDecisions failed: %v", err)
		}
		benchDecisionsSink = ds
	}
}

// BenchmarkDistributionQuantiles benchmarks distribution and percentile calculation.
func BenchmarkDistributionQuantiles(b *testing.B) {
	streaks := []int{1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 7, 8, 9, 10}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchDistSink = distOf(streaks)
	}
}
