package engine

// vllm_cache_observe_test.go — the focused witness for the vLLM prefix-cache observation
// adapter (item 33 / #1551, External-engines lane, epic #1490). It drives the adapter
// against a FAKE vLLM /metrics transport on both required paths:
//
//   - signal PRESENT: /metrics exposes vllm:prefix_cache_{queries,hits} → the adapter
//     reports a CachePassiveObserve capability with the observed reuse counters (never an
//     active class), and
//   - signal ABSENT: /metrics is disabled, or exposes no prefix-cache counters → the
//     adapter reports an explicit CacheUnknown "unavailable" label, never a fabricated
//     reuse number.
//
// Every path asserts ProvenanceProvider (observed provider counter, not a kernel witness)
// and ColdPathCorrect (observation changes no serving path). Mirrors llama_test.go so the
// three External-engine adapters share one witnessed contract.

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// fakeVLLMMetrics stands up a fake vLLM /metrics endpoint returning the given status and
// body, and returns an observer pointed at it. The observer derives /metrics from the
// worker's OpenAI base URL (with the trailing /v1 stripped), so the base URL carries a
// /v1 suffix to exercise that derivation.
func fakeVLLMMetrics(t *testing.T, status int, body string) (*VLLMCacheObserver, func()) {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/metrics" {
			http.NotFound(w, r)
			return
		}
		w.WriteHeader(status)
		_, _ = w.Write([]byte(body))
	}))
	return NewVLLMCacheObserver(VLLMConfig{BaseURL: srv.URL + "/v1"}), srv.Close
}

// TestVLLMObserveSignalPresent: a /metrics scrape carrying the prefix-cache counters is
// mapped to a passive-observe capability with provider provenance and the observed reuse
// counts.
func TestVLLMObserveSignalPresent(t *testing.T) {
	// A live worker exposing both counters — some queries, fewer hits (real reuse rate).
	body := strings.Join([]string{
		"# HELP vllm:prefix_cache_queries Prefix cache queries.",
		"# TYPE vllm:prefix_cache_queries counter",
		"vllm:prefix_cache_queries 40",
		"vllm:prefix_cache_hits 30",
		"vllm:num_requests_running 2",
	}, "\n") + "\n"

	obs, closeSrv := fakeVLLMMetrics(t, http.StatusOK, body)
	defer closeSrv()

	sig, err := obs.Observe(context.Background())
	if err != nil {
		t.Fatalf("Observe returned error: %v", err)
	}
	if !sig.Present {
		t.Fatalf("expected a present prefix-cache signal, got %+v", sig)
	}
	if sig.Queries != 40 || sig.Hits != 30 {
		t.Errorf("Queries/Hits = %v/%v, want 40/30", sig.Queries, sig.Hits)
	}

	cap := obs.CacheCapability()
	if cap.Engine != VLLMEngineID {
		t.Errorf("Engine = %q, want %q", cap.Engine, VLLMEngineID)
	}
	if cap.Verdict != CachePassiveObserve {
		t.Errorf("Verdict = %q, want %q (vLLM prefix cache is passive, never active)", cap.Verdict, CachePassiveObserve)
	}
	if cap.Verdict.Active() {
		t.Errorf("a passive-observe verdict must not report Active(): %+v", cap)
	}
	if cap.Provenance != ProvenanceProvider {
		t.Errorf("Provenance = %q, want %q (observed provider counter, not a kernel witness)", cap.Provenance, ProvenanceProvider)
	}
	if !cap.ColdPathCorrect {
		t.Errorf("ColdPathCorrect must be true: observation changes no serving path")
	}
	if !cap.Valid() {
		t.Errorf("capability must be well-formed: %+v", cap)
	}
	if !strings.Contains(cap.Evidence, "observed prefix reuse") {
		t.Errorf("Evidence should name the observed prefix reuse: %q", cap.Evidence)
	}
}

// TestVLLMObserveSignalPresentZero: the counters present but zero (a live cache with no
// reuse yet) is still a PRESENT passive-observe signal, distinct from unavailable — the
// adapter must not collapse "0 reuse observed" into "no signal".
func TestVLLMObserveSignalPresentZero(t *testing.T) {
	body := "vllm:prefix_cache_queries 0\nvllm:prefix_cache_hits 0\n"
	obs, closeSrv := fakeVLLMMetrics(t, http.StatusOK, body)
	defer closeSrv()

	sig, err := obs.Observe(context.Background())
	if err != nil {
		t.Fatalf("Observe returned error: %v", err)
	}
	if !sig.Present {
		t.Fatalf("present-but-zero counters must read as a PRESENT signal, got %+v", sig)
	}
	if cap := obs.CacheCapability(); cap.Verdict != CachePassiveObserve {
		t.Errorf("Verdict = %q, want %q (a live cache reporting zero reuse is still observed)", cap.Verdict, CachePassiveObserve)
	}
}

// TestVLLMObserveSignalAbsent: every no-evidence shape (metrics endpoint disabled, or a
// scrape without the prefix-cache counters) maps to the explicit CacheUnknown
// "unavailable" label — never a fabricated reuse number.
func TestVLLMObserveSignalAbsent(t *testing.T) {
	cases := []struct {
		name   string
		status int
		body   string
	}{
		// vLLM can be run with metrics disabled; a non-200 is a legitimate "no surface".
		{"metrics-disabled", http.StatusNotFound, "404 page not found\n"},
		// A metrics surface with no prefix-cache counters (older build / disabled prefix
		// caching) exposes no reuse signal.
		{"no-prefix-counters", http.StatusOK, "vllm:num_requests_running 1\nvllm:kv_cache_usage_perc 0.5\n"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			obs, closeSrv := fakeVLLMMetrics(t, tc.status, tc.body)
			defer closeSrv()

			sig, err := obs.Observe(context.Background())
			if err != nil {
				t.Fatalf("Observe returned error: %v", err)
			}
			if sig.Present {
				t.Fatalf("expected an absent prefix-cache signal, got %+v", sig)
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

// TestVLLMPureCapabilityMapping pins the pure signal→capability map without any IO, so
// the mapping contract is witnessed independently of the transport.
func TestVLLMPureCapabilityMapping(t *testing.T) {
	present := VLLMPrefixCacheSignal{Present: true, Queries: 10, Hits: 7}.Capability()
	if present.Verdict != CachePassiveObserve || !present.Valid() {
		t.Errorf("present signal → %+v, want a valid passive-observe capability", present)
	}
	absent := VLLMPrefixCacheSignal{}.Capability()
	if absent.Verdict != CacheUnknown || !absent.Valid() {
		t.Errorf("absent signal → %+v, want a valid unknown capability", absent)
	}
	if !present.ColdPathCorrect || !absent.ColdPathCorrect {
		t.Errorf("both mappings must keep ColdPathCorrect true (observation only)")
	}
}

// TestVLLMPureSignalDecode pins the metrics-text→signal decode, including summing a
// counter across data-parallel-rank label sets.
func TestVLLMPureSignalDecode(t *testing.T) {
	text := strings.Join([]string{
		`vllm:prefix_cache_queries{dp_rank="0"} 10`,
		`vllm:prefix_cache_queries{dp_rank="1"} 15`,
		`vllm:prefix_cache_hits{dp_rank="0"} 6`,
		`vllm:prefix_cache_hits{dp_rank="1"} 9`,
	}, "\n") + "\n"
	sig := ObserveVLLMPrefixCache(text)
	if !sig.Present {
		t.Fatalf("counters present → signal must be Present, got %+v", sig)
	}
	if sig.Queries != 25 || sig.Hits != 15 {
		t.Errorf("summed Queries/Hits = %v/%v, want 25/15", sig.Queries, sig.Hits)
	}
}

// TestVLLMCapabilityFailsClosedBeforeObserve: an observer that has not read yet must
// report the unavailable label, never infer a positive.
func TestVLLMCapabilityFailsClosedBeforeObserve(t *testing.T) {
	obs := NewVLLMCacheObserver(VLLMConfig{BaseURL: "http://example.invalid/v1"})
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

// TestVLLMObserveRequiresURL: with neither BaseURL nor MetricsURL, Observe refuses rather
// than reading a fabricated signal.
func TestVLLMObserveRequiresURL(t *testing.T) {
	obs := NewVLLMCacheObserver(VLLMConfig{})
	if _, err := obs.Observe(context.Background()); err == nil {
		t.Fatalf("expected an error when no vLLM metrics URL is configured")
	}
}

// Compile-time proof the observer satisfies the item-32 producer seam.
var _ CacheCapabilityProducer = (*VLLMCacheObserver)(nil)
