package ratelimit

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

func TestCompareLocalKeepsDistributedAlternativesExplicit(t *testing.T) {
	got := CompareLocal()
	want := map[string]struct {
		kind      string
		available bool
	}{
		"fak native rate limiter": {"native", true},
		"no limiter":              {"baseline", true},
		"Envoy local rate limit":  {"external", false},
		"Kong rate limiting":      {"external", false},
		"Redis-cell":              {"external", false},
	}
	if len(got.Arms) != len(want) {
		t.Fatalf("arms=%d want %d: %#v", len(got.Arms), len(want), got.Arms)
	}
	for _, arm := range got.Arms {
		expected, ok := want[arm.Name]
		if !ok {
			t.Fatalf("unexpected arm %q", arm.Name)
		}
		if arm.Kind != expected.kind || arm.Available != expected.available {
			t.Errorf("arm %q = %q available=%v; want %q/%v", arm.Name, arm.Kind, arm.Available, expected.kind, expected.available)
		}
		if !arm.Available && (arm.Correct || arm.Latency != 0 || arm.Admitted != 0 || arm.Denied != 0 || arm.Bytes != 0 || arm.CostUSD != 0) {
			t.Errorf("unavailable arm %q claims a result: %#v", arm.Name, arm)
		}
	}
	if !got.Arms[0].Correct || got.Arms[0].Admitted != 750 || got.Arms[0].Denied != 250 {
		t.Fatalf("native result=%#v", got.Arms[0])
	}
	if got.Arms[1].Correct {
		t.Fatalf("baseline unexpectedly correct: %#v", got.Arms[1])
	}
}

func BenchmarkNativeRateLimitCall(b *testing.B) {
	limiter := New()
	limiter.SetLimit(Limit{MaxCalls: b.N + 1}, KeyPerTrace)
	call := &abi.ToolCall{TraceID: "benchmark-trace", Tool: "search"}
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if verdict := limiter.Adjudicate(ctx, call); verdict.Kind != abi.VerdictDefer {
			b.Fatalf("native fixture failed: %+v", verdict)
		}
	}
}
