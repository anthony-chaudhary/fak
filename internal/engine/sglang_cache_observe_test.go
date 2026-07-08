package engine

// sglang_cache_observe_test.go — the focused witness for the SGLang radix/prefix-cache
// observation adapter (item 34 / #1552, External-engines lane, epic #1490). It drives the
// adapter against a FAKE SGLang /metrics transport on both required paths:
//
//   - signal PRESENT: /metrics exposes sglang:{cache,prefix_cache}_hit_rate → the adapter
//     reports a CachePassiveObserve capability with the observed radix reuse ratio (never
//     an active class), and
//   - signal ABSENT: /metrics is disabled, or exposes no radix hit-rate gauge → the
//     adapter reports an explicit CacheUnknown "unavailable" label, never a fabricated
//     reuse number.
//
// Every path asserts ProvenanceProvider (observed provider gauge, not a kernel witness)
// and ColdPathCorrect (observation changes no serving path). Mirrors
// vllm_cache_observe_test.go so the External-engine adapters share one witnessed contract.

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// fakeSGLangMetrics stands up a fake SGLang /metrics endpoint returning the given status
// and body, and returns an observer pointed at it. SGLang's HTTP root is not an OpenAI /v1
// frontend, so the base URL carries NO /v1 suffix and the observer derives /metrics from
// the bare root (matching sglang.go's metricsURL derivation).
func fakeSGLangMetrics(t *testing.T, status int, body string) (*SGLangCacheObserver, func()) {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/metrics" {
			http.NotFound(w, r)
			return
		}
		w.WriteHeader(status)
		_, _ = w.Write([]byte(body))
	}))
	return NewSGLangCacheObserver(SGLangConfig{BaseURL: srv.URL}), srv.Close
}

// TestSGLangObserveSignalPresent: a /metrics scrape carrying the radix hit-rate gauge is
// mapped to a passive-observe capability with provider provenance and the observed ratio.
func TestSGLangObserveSignalPresent(t *testing.T) {
	// A live worker exposing the RadixAttention hit-rate gauge as a 0..1 ratio.
	body := strings.Join([]string{
		"# HELP sglang:cache_hit_rate Radix cache hit rate.",
		"# TYPE sglang:cache_hit_rate gauge",
		"sglang:cache_hit_rate 0.75",
		"sglang:num_running_reqs 2",
	}, "\n") + "\n"

	obs, closeSrv := fakeSGLangMetrics(t, http.StatusOK, body)
	defer closeSrv()

	sig, err := obs.Observe(context.Background())
	if err != nil {
		t.Fatalf("Observe returned error: %v", err)
	}
	if !sig.Present {
		t.Fatalf("expected a present radix-cache signal, got %+v", sig)
	}
	if sig.HitRatio != 0.75 {
		t.Errorf("HitRatio = %v, want 0.75", sig.HitRatio)
	}

	cap := obs.CacheCapability()
	if cap.Engine != SGLangEngineID {
		t.Errorf("Engine = %q, want %q", cap.Engine, SGLangEngineID)
	}
	if cap.Verdict != CachePassiveObserve {
		t.Errorf("Verdict = %q, want %q (SGLang radix cache is passive, never active)", cap.Verdict, CachePassiveObserve)
	}
	if cap.Verdict.Active() {
		t.Errorf("a passive-observe verdict must not report Active(): %+v", cap)
	}
	if cap.Provenance != ProvenanceProvider {
		t.Errorf("Provenance = %q, want %q (observed provider gauge, not a kernel witness)", cap.Provenance, ProvenanceProvider)
	}
	if !cap.ColdPathCorrect {
		t.Errorf("ColdPathCorrect must be true: observation changes no serving path")
	}
	if !cap.Valid() {
		t.Errorf("capability must be well-formed: %+v", cap)
	}
	if !strings.Contains(cap.Evidence, "observed radix reuse") {
		t.Errorf("Evidence should name the observed radix reuse: %q", cap.Evidence)
	}
}

// TestSGLangObserveSignalPresentZero: the gauge present but zero (a live radix cache with
// no reuse yet) is still a PRESENT passive-observe signal, distinct from unavailable — the
// adapter must not collapse "0 reuse observed" into "no signal".
func TestSGLangObserveSignalPresentZero(t *testing.T) {
	body := "sglang:prefix_cache_hit_rate 0\nsglang:num_running_reqs 1\n"
	obs, closeSrv := fakeSGLangMetrics(t, http.StatusOK, body)
	defer closeSrv()

	sig, err := obs.Observe(context.Background())
	if err != nil {
		t.Fatalf("Observe returned error: %v", err)
	}
	if !sig.Present {
		t.Fatalf("present-but-zero gauge must read as a PRESENT signal, got %+v", sig)
	}
	if cap := obs.CacheCapability(); cap.Verdict != CachePassiveObserve {
		t.Errorf("Verdict = %q, want %q (a live cache reporting zero reuse is still observed)", cap.Verdict, CachePassiveObserve)
	}
}

// TestSGLangObserveSignalAbsent: every no-evidence shape (metrics endpoint disabled, or a
// scrape without the radix hit-rate gauge) maps to the explicit CacheUnknown "unavailable"
// label — never a fabricated reuse number.
func TestSGLangObserveSignalAbsent(t *testing.T) {
	cases := []struct {
		name   string
		status int
		body   string
	}{
		// SGLang can be run with metrics disabled; a non-200 is a legitimate "no surface".
		{"metrics-disabled", http.StatusNotFound, "404 page not found\n"},
		// A metrics surface with no radix hit-rate gauge (older build / prefix caching off)
		// exposes no reuse signal.
		{"no-radix-gauge", http.StatusOK, "sglang:num_running_reqs 1\nsglang:token_usage 0.5\n"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			obs, closeSrv := fakeSGLangMetrics(t, tc.status, tc.body)
			defer closeSrv()

			sig, err := obs.Observe(context.Background())
			if err != nil {
				t.Fatalf("Observe returned error: %v", err)
			}
			if sig.Present {
				t.Fatalf("expected an absent radix-cache signal, got %+v", sig)
			}

			cap := obs.CacheCapability()
			if cap.Verdict != CacheUnknown {
				t.Errorf("Verdict = %q, want %q (no signal → unavailable)", cap.Verdict, CacheUnknown)
			}
			if cap.Verdict.Active() {
				t.Errorf("an unavailable verdict must not report Active(): %+v", cap)
			}
			if cap.Provenance != ProvenanceProvider {
				t.Errorf("Provenance = %q, want %q", cap.Provenance, ProvenanceProvider)
			}
			if !cap.ColdPathCorrect {
				t.Errorf("ColdPathCorrect must stay true on the cold/absent path")
			}
			if !cap.Valid() {
				t.Errorf("capability must be well-formed: %+v", cap)
			}
			if !strings.Contains(cap.Evidence, "unavailable") {
				t.Errorf("Evidence must carry the explicit unavailable label: %q", cap.Evidence)
			}
		})
	}
}

// TestSGLangPureCapabilityMapping pins the pure signal→capability map without any IO, so
// the mapping contract is witnessed independently of the transport.
func TestSGLangPureCapabilityMapping(t *testing.T) {
	present := SGLangRadixCacheSignal{Present: true, HitRatio: 0.6}.Capability()
	if present.Verdict != CachePassiveObserve || !present.Valid() {
		t.Errorf("present signal → %+v, want a valid passive-observe capability", present)
	}
	absent := SGLangRadixCacheSignal{}.Capability()
	if absent.Verdict != CacheUnknown || !absent.Valid() {
		t.Errorf("absent signal → %+v, want a valid unknown capability", absent)
	}
	if !present.ColdPathCorrect || !absent.ColdPathCorrect {
		t.Errorf("both mappings must keep ColdPathCorrect true (observation only)")
	}
}

// TestSGLangPureSignalDecode pins the metrics-text→signal decode, including the
// 0..100 percentage → 0..1 ratio normalization SGLang's gauge may need.
func TestSGLangPureSignalDecode(t *testing.T) {
	// A build that reports the hit rate as a percentage (42%) must normalize to 0.42.
	sig := ObserveSGLangRadixCache("sglang:cache_hit_rate 42\n")
	if !sig.Present {
		t.Fatalf("gauge present → signal must be Present, got %+v", sig)
	}
	if sig.HitRatio != 0.42 {
		t.Errorf("normalized HitRatio = %v, want 0.42 (42%% → 0.42)", sig.HitRatio)
	}
	// A ratio already in 0..1 is left as-is.
	if r := ObserveSGLangRadixCache("sglang:prefix_cache_hit_rate 0.9\n").HitRatio; r != 0.9 {
		t.Errorf("ratio HitRatio = %v, want 0.9 (already 0..1, not normalized)", r)
	}
}

// TestSGLangCapabilityFailsClosedBeforeObserve: an observer that has not read yet must
// report the unavailable label, never infer a positive.
func TestSGLangCapabilityFailsClosedBeforeObserve(t *testing.T) {
	obs := NewSGLangCacheObserver(SGLangConfig{BaseURL: "http://example.invalid"})
	cap := obs.CacheCapability()
	if cap.Verdict != CacheUnknown {
		t.Errorf("un-observed observer Verdict = %q, want %q (fail closed)", cap.Verdict, CacheUnknown)
	}
	if cap.Verdict.Active() {
		t.Errorf("fail-closed default must not be Active")
	}
	if !cap.Valid() {
		t.Errorf("fail-closed capability must be well-formed: %+v", cap)
	}
}

// TestSGLangObserveRequiresURL: with neither BaseURL nor MetricsURL, Observe refuses rather
// than reading a fabricated signal.
func TestSGLangObserveRequiresURL(t *testing.T) {
	obs := NewSGLangCacheObserver(SGLangConfig{})
	if _, err := obs.Observe(context.Background()); err == nil {
		t.Fatalf("expected an error when no SGLang metrics URL is configured")
	}
}

// Compile-time proof the observer satisfies the item-32 producer seam.
var _ CacheCapabilityProducer = (*SGLangCacheObserver)(nil)
