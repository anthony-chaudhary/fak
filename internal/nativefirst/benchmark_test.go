package nativefirst

import (
	"testing"
)

var benchSink *Finding

// BenchmarkNativeFirst measures the throughput of ScanLine across a balanced
// corpus of policy violations, whitelisted references, and unrelated statements.
func BenchmarkNativeFirst(b *testing.B) {
	corpus := []string{
		"Qwen3.8 native performance defaults to " + "llama" + ".cpp.",
		"Native falls back to " + "llama" + "-server.",
		"The native backend auto-selects " + "llama" + "cpp.",
		"Benchmark fak-native against " + "llama" + ".cpp.",
		"Use " + "llama" + ".cpp explicitly for parity diagnosis.",
		"Study and borrow a " + "llama" + ".cpp kernel.",
		"The interop adapter delegates explicitly to " + "llama" + "-server.",
		"The default policy is fail closed.",
		"Execute native kernel pipeline on accelerator device.",
		"Comparing prefill latency against published baseline.",
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, line := range corpus {
			benchSink = ScanLine(line)
		}
	}
}

// BenchmarkScanLineViolation measures detection of prohibited runtime substitutions.
func BenchmarkScanLineViolation(b *testing.B) {
	var line = "Qwen3.8 native performance defaults to " + "llama" + ".cpp."
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSink = ScanLine(line)
	}
}

// BenchmarkScanLineWhitelisted measures filtering of permitted reference usage.
func BenchmarkScanLineWhitelisted(b *testing.B) {
	var line = "Benchmark fak-native against " + "llama" + ".cpp reference."
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSink = ScanLine(line)
	}
}

// BenchmarkScanLineClean measures rejection speed for clean text.
func BenchmarkScanLineClean(b *testing.B) {
	const line = "Execute native kernel pipeline on accelerator device."
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSink = ScanLine(line)
	}
}

// TestBenchmarkNativeFirstSanity executes the benchmarks once to ensure correctness.
func TestBenchmarkNativeFirstSanity(t *testing.T) {
	res := testing.Benchmark(BenchmarkNativeFirst)
	if res.N <= 0 {
		t.Fatalf("expected benchmark iterations > 0, got %d", res.N)
	}
}
