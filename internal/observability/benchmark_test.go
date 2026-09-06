package observability

import (
	"testing"
)

func BenchmarkCheckPromptTokenAlarm(b *testing.B) {
	const turnTokens = 12000
	const baselineTokens = 6000

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		alarm := CheckPromptTokenAlarm(turnTokens, baselineTokens)
		if alarm.Triggered {
			b.Fatalf("unexpected alarm trigger: %+v", alarm)
		}
	}
}

func BenchmarkCheckLatencySpikeAlarm(b *testing.B) {
	const current = 2.4
	const median = 2.0
	const spikes = 0

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		alarm := CheckLatencySpikeAlarm(current, median, spikes)
		if alarm.Triggered {
			b.Fatalf("unexpected alarm trigger: %+v", alarm)
		}
	}
}

func BenchmarkEvaluateHealth(b *testing.B) {
	latencies := []float64{1.2, 1.4, 1.1, 1.3, 1.5, 1.2, 1.4, 1.3, 1.2, 1.4}
	const promptTokens = 5000
	const baselinePrompt = 4800

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		report := EvaluateHealth(promptTokens, baselinePrompt, latencies, "")
		if !report.OK {
			b.Fatalf("unexpected health report failure: %+v", report)
		}
	}
}
