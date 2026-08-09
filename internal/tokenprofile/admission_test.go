package tokenprofile

import "testing"

func TestRouteEqualTotalProfilesByIndependentEnvelope(t *testing.T) {
	targets := []Capacity{
		{Name: "prefill", Envelopes: &Envelopes{UncachedPrefill: 90, CachedInput: 10, ReservedDecode: 20}},
		{Name: "decode", Envelopes: &Envelopes{UncachedPrefill: 20, CachedInput: 10, ReservedDecode: 90}},
	}
	prefillHeavy := Forecast{InputTokens: 80, MaxOutputTokens: 20}
	decodeHeavy := Forecast{InputTokens: 20, MaxOutputTokens: 80}

	if got, refusal := Admit(prefillHeavy, targets); refusal != nil || got != "prefill" {
		t.Fatalf("prefill-heavy route = %q, %v; want prefill", got, refusal)
	}
	if got, refusal := Admit(decodeHeavy, targets); refusal != nil || got != "decode" {
		t.Fatalf("decode-heavy route = %q, %v; want decode", got, refusal)
	}
}

func TestDecodeExhaustionDoesNotHeadOfLineBlockPrefill(t *testing.T) {
	targets := []Capacity{{
		Name:      "mixed",
		Envelopes: &Envelopes{UncachedPrefill: 100, CachedInput: 100, ReservedDecode: 10},
		Used:      Envelopes{ReservedDecode: 10},
	}}
	decodeHeavy := Forecast{InputTokens: 5, MaxOutputTokens: 20}
	if _, refusal := Admit(decodeHeavy, targets); refusal == nil || refusal.Class != EnvelopeReservedDecode || refusal.Recovery == "" {
		t.Fatalf("refusal = %#v; want typed decode refusal with recovery", refusal)
	}
	prefillOnly := Forecast{InputTokens: 80}
	if got, refusal := Admit(prefillOnly, targets); refusal != nil || got != "mixed" {
		t.Fatalf("prefill route after decode refusal = %q, %v; want mixed", got, refusal)
	}
}

func TestAbsentClassPolicyPreservesScalarAdmission(t *testing.T) {
	targets := []Capacity{{Name: "legacy", TotalTokens: 100, UsedTotal: 20}}
	request := Forecast{InputTokens: 30, CachedInputTokens: 20, MaxOutputTokens: 30}
	if got, refusal := Admit(request, targets); refusal != nil || got != "legacy" {
		t.Fatalf("legacy route = %q, %v; want admit at scalar total", got, refusal)
	}
	if _, refusal := Admit(Forecast{MaxOutputTokens: 21}, targets); refusal == nil || refusal.Class != EnvelopeTotalTokens {
		t.Fatalf("legacy overflow = %#v; want total-token refusal", refusal)
	}
}

func TestCachedInputRefusalNamesClassAndRecovery(t *testing.T) {
	targets := []Capacity{{Name: "small-kv", Envelopes: &Envelopes{UncachedPrefill: 100, CachedInput: 4, ReservedDecode: 100}}}
	_, refusal := Admit(Forecast{CachedInputTokens: 5}, targets)
	if refusal == nil || refusal.Class != EnvelopeCachedInput || refusal.Required != 5 || refusal.Available != 4 || refusal.Recovery == "" {
		t.Fatalf("refusal = %#v; want cached-input class, amounts, and recovery", refusal)
	}
}
