package enginecache

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestCompareLocalKeepsEngineAlternativesExplicit(t *testing.T) {
	got := CompareLocal()
	want := map[string]struct {
		kind        string
		integration bool
		available   bool
	}{
		"fak native invalidation planner + adapter": {"native", false, true},
		"no invalidation": {"baseline", false, true},
		"vLLM":            {"external", false, false},
		"SGLang":          {"external", false, false},
		"LMCache":         {"external", false, false},
		"fak + vLLM":      {"integration", true, false},
		"fak + SGLang":    {"integration", true, false},
	}
	if len(got.Arms) != len(want) {
		t.Fatalf("arms=%d want %d: %#v", len(got.Arms), len(want), got.Arms)
	}
	for _, arm := range got.Arms {
		expected, ok := want[arm.Name]
		if !ok {
			t.Fatalf("unexpected arm %q", arm.Name)
		}
		if arm.Kind != expected.kind || arm.Integration != expected.integration || arm.Available != expected.available {
			t.Errorf("arm %q = kind %q integration=%v available=%v; want %q/%v/%v", arm.Name, arm.Kind, arm.Integration, arm.Available, expected.kind, expected.integration, expected.available)
		}
		if !arm.Available && (arm.Correct || arm.Latency != 0 || arm.Requests != 0 || arm.Bytes != 0 || arm.CostUSD != 0) {
			t.Errorf("unavailable arm %q claims a result: %#v", arm.Name, arm)
		}
	}
	if !got.Arms[0].Correct {
		t.Fatalf("native loopback witness failed: %#v", got.Arms[0])
	}
}

func BenchmarkCompareNativeInvalidation(b *testing.B) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusOK) }))
	defer server.Close()
	client := Client{Engine: EngineVLLM, BaseURL: server.URL}
	dirs := comparisonDirectives()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		result, err := client.Invalidate(context.Background(), dirs)
		if err != nil || result.StatusCode != http.StatusOK {
			b.Fatalf("native witness failed: result=%+v err=%v", result, err)
		}
	}
}
