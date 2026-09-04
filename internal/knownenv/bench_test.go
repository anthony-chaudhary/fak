package knownenv

import (
	"testing"
)

// TestBenchmarkKnownEnvSanity verifies that the benchmarked workload functions correctly.
func TestBenchmarkKnownEnvSanity(t *testing.T) {
	registry := DefaultRegistry()
	output := "rsync: [sender] write error: Broken pipe (32)\n" +
		"rsync error: error in socket IO (code 23) at io.c(line 820)\n"
	got := Annotate(output, 23, registry)
	if len(got) == 0 {
		t.Fatal("expected at least one annotation for rsync-23 failure output")
	}
	if got[0].ID != "rsync-23" {
		t.Fatalf("unexpected annotation ID: %s", got[0].ID)
	}
}

// BenchmarkKnownEnv exercises known environment matching and annotation across
// positive and negative tool output patterns in a loop.
func BenchmarkKnownEnv(b *testing.B) {
	registry := DefaultRegistry()
	output := "rsync: [sender] write error: Broken pipe (32)\n" +
		"rsync error: error in socket IO (code 23) at io.c(line 820)\n"
	cleanOutput := "=== RUN TestSomething\n--- PASS: TestSomething (0.00s)\nPASS\n"

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = Annotate(output, 23, registry)
		_ = Annotate(cleanOutput, 0, registry)
	}
}
