package completiondist

import (
	"testing"
)

func BenchmarkCompletionDist(b *testing.B) {
	data := []byte(`{"issue":1,"duration_sec":300}
{"issue":2,"duration_sec":600}
{"issue":3,"duration_sec":720}
{"issue":4,"duration_sec":1200}
{"issue":5,"duration_sec":1800}
{"issue":6,"duration_sec":2400}
{"issue":7,"duration_sec":3000}
{"issue":8,"duration_sec":5400}
{"issue":9,"duration_sec":9000}
{"issue":10,"duration_sec":20000}`)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		samples, err := ParseSamples(data)
		if err != nil {
			b.Fatalf("ParseSamples: %v", err)
		}
		dist := Build(samples)
		if dist.Count != 10 {
			b.Fatalf("Build count = %d, want 10", dist.Count)
		}
	}
}

func TestBenchmarkCompletionDist(t *testing.T) {
	data := []byte(`{"issue":1,"duration_sec":300}
{"issue":2,"duration_sec":600}`)
	samples, err := ParseSamples(data)
	if err != nil {
		t.Fatalf("ParseSamples: %v", err)
	}
	dist := Build(samples)
	if dist.Count != 2 {
		t.Fatalf("dist.Count = %d, want 2", dist.Count)
	}
}
