package configguide

import (
	"testing"
)

func BenchmarkConfigGuide(b *testing.B) {
	opts := []Options{
		{Posture: "default"},
		{Posture: "long-session", Budget: 200000},
		{Posture: "team-gateway", KeyEnv: "BENCH_KEY", Bind: "127.0.0.1:0"},
		{Posture: "hardened", PolicyPath: "bench-policy.json", KeyEnv: "HARDENED_KEY", Bind: "127.0.0.1:0"},
	}
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		opt := opts[i%len(opts)]
		res, err := Guide(opt)
		if err != nil {
			b.Fatalf("Guide(%s) failed: %v", opt.Posture, err)
		}
		if res.Schema != Schema {
			b.Fatalf("unexpected schema: %s", res.Schema)
		}
	}
}

func TestBenchmarkConfigGuide(t *testing.T) {
	opts := []Options{
		{Posture: "default"},
		{Posture: "long-session", Budget: 200000},
		{Posture: "team-gateway", KeyEnv: "BENCH_KEY", Bind: "127.0.0.1:0"},
		{Posture: "hardened", PolicyPath: "bench-policy.json", KeyEnv: "HARDENED_KEY", Bind: "127.0.0.1:0"},
	}
	for _, opt := range opts {
		res, err := Guide(opt)
		if err != nil {
			t.Fatalf("Guide(%s) failed: %v", opt.Posture, err)
		}
		if res.Schema != Schema {
			t.Fatalf("unexpected schema: %s", res.Schema)
		}
	}
}
