package engine

// llama_test.go — the focused witness for the llama.cpp/llama-server session-cache
// observation adapter (item 35 / #1553). It drives the adapter against a FAKE
// llama-server transport on both required paths:
//
//   - signal PRESENT: /slots exposes per-slot prompt-cache accounting → the adapter
//     reports a CachePassiveObserve capability (never an active class), and
//   - signal ABSENT: /slots is disabled / empty / field-less → the adapter reports an
//     explicit CacheUnknown passive/no-evidence label, never a fabricated number.
//
// Every path asserts ProvenanceProvider (observed provider counter, not a kernel
// witness) and ColdPathCorrect (observation changes no serving path).

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// fakeLlamaSlots stands up a fake llama-server /slots endpoint returning the given
// status and body, and returns an observer pointed at it.
func fakeLlamaSlots(t *testing.T, status int, body string) (*LlamaCacheObserver, func()) {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/slots" {
			http.NotFound(w, r)
			return
		}
		w.WriteHeader(status)
		_, _ = w.Write([]byte(body))
	}))
	return NewLlamaCacheObserver(LlamaConfig{BaseURL: srv.URL}), srv.Close
}

// TestLlamaObserveSignalPresent: a /slots response carrying prompt-cache accounting is
// mapped to a passive-observe capability with provider provenance.
func TestLlamaObserveSignalPresent(t *testing.T) {
	// Two slots, one idle (accounting present but zero reuse), one processing with a
	// non-zero n_past/cache_tokens. The symptom SURFACE existing is what matters.
	body := `[
		{"id":0,"state":0,"n_past":0,"cache_tokens":0,"n_prompt_tokens_processed":0},
		{"id":1,"state":1,"n_past":128,"cache_tokens":96,"n_prompt_tokens_processed":32}
	]`
	obs, closeSrv := fakeLlamaSlots(t, http.StatusOK, body)
	defer closeSrv()

	sig, err := obs.Observe(context.Background())
	if err != nil {
		t.Fatalf("Observe returned error: %v", err)
	}
	if !sig.Present {
		t.Fatalf("expected a present session-cache signal, got %+v", sig)
	}
	if sig.Slots != 2 {
		t.Errorf("Slots = %d, want 2", sig.Slots)
	}
	// All three accounting fields were exposed, sorted and de-duplicated.
	if got := strings.Join(sig.Fields, ","); got != "cache_tokens,n_past,n_prompt_tokens_processed" {
		t.Errorf("Fields = %q, want the three accounting fields sorted", got)
	}

	cap := obs.CacheCapability()
	if cap.Engine != LlamaEngineID {
		t.Errorf("Engine = %q, want %q", cap.Engine, LlamaEngineID)
	}
	if cap.Verdict != CachePassiveObserve {
		t.Errorf("Verdict = %q, want %q (llama session cache is passive, never active)", cap.Verdict, CachePassiveObserve)
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
	if !strings.Contains(cap.Evidence, "passive-observe") {
		t.Errorf("Evidence should name the passive-observe boundary: %q", cap.Evidence)
	}
}

// TestLlamaObserveSignalAbsent: every no-evidence shape (endpoint disabled, empty
// array, slots without an accounting field) maps to the explicit CacheUnknown
// passive/no-evidence label — never a fabricated cache-state number.
func TestLlamaObserveSignalAbsent(t *testing.T) {
	cases := []struct {
		name   string
		status int
		body   string
	}{
		// llama-server answers 501 when --slots is disabled.
		{"slots-disabled", http.StatusNotImplemented, `{"error":{"code":501,"message":"slots endpoint is disabled"}}`},
		{"no-slots", http.StatusOK, `[]`},
		{"no-accounting-field", http.StatusOK, `[{"id":0,"state":0,"prompt":"hi"}]`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			obs, closeSrv := fakeLlamaSlots(t, tc.status, tc.body)
			defer closeSrv()

			sig, err := obs.Observe(context.Background())
			if err != nil {
				t.Fatalf("Observe returned error: %v", err)
			}
			if sig.Present {
				t.Fatalf("expected an absent session-cache signal, got %+v", sig)
			}

			cap := obs.CacheCapability()
			if cap.Verdict != CacheUnknown {
				t.Errorf("Verdict = %q, want %q (no signal → no-evidence)", cap.Verdict, CacheUnknown)
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
			if !strings.Contains(cap.Evidence, "no-evidence") {
				t.Errorf("Evidence must carry the explicit no-evidence label: %q", cap.Evidence)
			}
		})
	}
}

// TestLlamaCapabilityFailsClosedBeforeObserve: an observer that has not read yet must
// report the unknown/no-evidence label, never infer a positive.
func TestLlamaCapabilityFailsClosedBeforeObserve(t *testing.T) {
	obs := NewLlamaCacheObserver(LlamaConfig{BaseURL: "http://example.invalid"})
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

// TestLlamaPureCapabilityMapping pins the pure signal→capability map without any IO, so
// the mapping contract is witnessed independently of the transport.
func TestLlamaPureCapabilityMapping(t *testing.T) {
	present := LlamaSessionCache{Present: true, Slots: 1, Fields: []string{"cache_tokens"}}.Capability()
	if present.Verdict != CachePassiveObserve || !present.Valid() {
		t.Errorf("present signal → %+v, want a valid passive-observe capability", present)
	}
	absent := LlamaSessionCache{}.Capability()
	if absent.Verdict != CacheUnknown || !absent.Valid() {
		t.Errorf("absent signal → %+v, want a valid unknown capability", absent)
	}
	if !present.ColdPathCorrect || !absent.ColdPathCorrect {
		t.Errorf("both mappings must keep ColdPathCorrect true (observation only)")
	}
}

// TestLlamaObserveRequiresURL: with neither BaseURL nor SlotsURL, Observe refuses
// rather than reading a fabricated signal.
func TestLlamaObserveRequiresURL(t *testing.T) {
	obs := NewLlamaCacheObserver(LlamaConfig{})
	if _, err := obs.Observe(context.Background()); err == nil {
		t.Fatalf("expected an error when no llama-server URL is configured")
	}
}

// Compile-time proof the observer satisfies the item-32 producer seam.
var _ CacheCapabilityProducer = (*LlamaCacheObserver)(nil)
