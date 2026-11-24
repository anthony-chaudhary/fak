package routing

import (
	"net/http"
	"strings"
	"testing"
)

func zenFreeTierFallback() RungOutcome {
	return RungOutcome{Backend: "opencode-zen", Failure: Classify(403, witnessedZenFreeTierRefusal)}
}

func leasedHalos() []RungOutcome {
	return []RungOutcome{
		{Backend: "strix1", Skipped: true, SkipReason: "not_ready:leased(bench-worker)"},
		{Backend: "strix3", Skipped: true, SkipReason: "not_ready:leased(bench-worker)"},
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestClassifyFreeTierCarriesClientOnlyRefusal(t *testing.T) {
	if got := Classify(403, witnessedZenFreeTierRefusal).Refusal; got != RefusalFreeTierClientOnly {
		t.Fatalf("Refusal = %q, want %q", got, RefusalFreeTierClientOnly)
	}
	if got := Classify(404, `{"error":"model not found"}`).Refusal; got != "" {
		t.Fatalf("plain model wall Refusal = %q, want empty", got)
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestExhaustedWalkVerdictMaskedPrimary(t *testing.T) {
	fallback := RungOutcome{Backend: "opencode-go", Failure: Classify(400, `{"error":"model not found"}`)}
	v, ok := ExhaustedWalkVerdict(leasedHalos(), fallback)
	if !ok {
		t.Fatal("ok = false, want masked-primary verdict")
	}
	if v.Refusal != RefusalFallbackMaskedPrimary || v.Status != http.StatusServiceUnavailable || !v.Retryable {
		t.Fatalf("verdict = (%s, %d, retryable=%v), want (%s, 503, true)", v.Refusal, v.Status, v.Retryable, RefusalFallbackMaskedPrimary)
	}
	if v.PrimaryBackend != "strix1" || v.PrimaryCause != "not_ready:leased" || v.Holder != "bench-worker" {
		t.Fatalf("primary = (%s, %s, %s), want (strix1, not_ready:leased, bench-worker)", v.PrimaryBackend, v.PrimaryCause, v.Holder)
	}
	if v.FallbackClass != FailureModelMissing {
		t.Fatalf("FallbackClass = %s, want %s", v.FallbackClass, FailureModelMissing)
	}
	if tok := v.FailoverToken(); !strings.HasPrefix(tok, string(RefusalFallbackMaskedPrimary)+" ") {
		t.Fatalf("FailoverToken does not lead with the refusal token")
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestExhaustedWalkVerdictMaskedPrimaryOutranksFreeTier(t *testing.T) {
	v, ok := ExhaustedWalkVerdict(leasedHalos(), zenFreeTierFallback())
	if !ok || v.Refusal != RefusalFallbackMaskedPrimary {
		t.Fatalf("verdict = (%s, ok=%v), want %s", v.Refusal, ok, RefusalFallbackMaskedPrimary)
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestExhaustedWalkVerdictDialedPrimaryFailure(t *testing.T) {
	primaries := []RungOutcome{{Backend: "strix1", Failure: Classify(503, "")}}
	fallback := RungOutcome{Backend: "opencode-go", Failure: Classify(404, "")}
	v, ok := ExhaustedWalkVerdict(primaries, fallback)
	if !ok || v.Refusal != RefusalFallbackMaskedPrimary || v.PrimaryCause != string(FailureUnavailable) || v.Holder != "" {
		t.Fatalf("verdict = (%s, cause=%s, holder=%s, ok=%v)", v.Refusal, v.PrimaryCause, v.Holder, ok)
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestExhaustedWalkVerdictFreeTierFailsFast(t *testing.T) {
	v, ok := ExhaustedWalkVerdict(nil, zenFreeTierFallback())
	if !ok {
		t.Fatal("ok = false, want free-tier verdict")
	}
	if v.Refusal != RefusalFreeTierClientOnly || v.Status != http.StatusForbidden || v.Retryable {
		t.Fatalf("verdict = (%s, %d, retryable=%v), want (%s, 403, false)", v.Refusal, v.Status, v.Retryable, RefusalFreeTierClientOnly)
	}
	if v.Status >= 500 {
		t.Fatalf("free-tier status %d is a 5xx a client would retry", v.Status)
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestExhaustedWalkVerdictRelaysHonestFallback(t *testing.T) {
	cases := map[string]struct {
		primaries []RungOutcome
		fallback  RungOutcome
	}{
		"no primaries, plain model wall": {nil, RungOutcome{Backend: "opencode-go", Failure: Classify(404, "")}},
		"fallback not a model wall":      {leasedHalos(), RungOutcome{Backend: "opencode-go", Failure: Classify(429, "")}},
		"primary not missed":             {[]RungOutcome{{Backend: "strix1"}}, RungOutcome{Backend: "opencode-go", Failure: Classify(404, "")}},
		"fallback skipped":               {leasedHalos(), RungOutcome{Backend: "opencode-go", Skipped: true}},
	}
	for name, c := range cases {
		if v, ok := ExhaustedWalkVerdict(c.primaries, c.fallback); ok {
			t.Errorf("%s: ok = true (%s), want relay", name, v.Refusal)
		}
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestClientRefusalVocabularyClosed(t *testing.T) {
	for _, r := range ClientRefusals() {
		if !r.IsKnown() {
			t.Fatalf("%s not known", r)
		}
	}
	if ClientRefusal("ROUTING_SOMETHING_ELSE").IsKnown() {
		t.Fatal("unknown refusal reads as known")
	}
}
