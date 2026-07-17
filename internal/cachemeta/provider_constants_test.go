package cachemeta

import "testing"

// allProviderConstants gathers every ProviderConstant this package exposes so the
// provenance invariants below can be asserted uniformly. A named key lets a failure
// point at the offending constant.
func allProviderConstants() map[string]ProviderConstant {
	all := map[string]ProviderConstant{
		"min_prefix":    ProviderMinPrefixConstant,
		"read_discount": ProviderReadDiscountConstant,
	}
	for k, v := range ProviderTTLConstants {
		all["ttl."+k] = v
	}
	return all
}

// TestProviderConstantsCarryFreshnessProvenance is the #1542 default witness: every
// provider constant (TTL, min-prefix, read-discount) is a MEASURED-or-HYPOTHESIS
// record with a non-empty date and source — never a bare literal.
func TestProviderConstantsCarryFreshnessProvenance(t *testing.T) {
	for name, c := range allProviderConstants() {
		switch c.Status {
		case FreshnessMeasured, FreshnessHypothesis:
			// closed vocabulary honored
		default:
			t.Errorf("constant %q has Status %q; want one of {MEASURED, HYPOTHESIS}", name, c.Status)
		}
		if c.Date == "" {
			t.Errorf("constant %q has empty Date; every constant must record when its status was affirmed", name)
		}
		if c.Source == "" {
			t.Errorf("constant %q has empty Source; every constant must cite its provenance", name)
		}
		if c.Unit == "" {
			t.Errorf("constant %q has empty Unit; a value without a unit is not interpretable", name)
		}
	}
}

// TestProviderTTLMillisUnchanged guards against behavior drift: routing the TTL
// through the records must return byte-identical values to the pre-#1542 literal
// switch for every known retention hint (and 0 for anything else).
func TestProviderTTLMillisUnchanged(t *testing.T) {
	cases := map[string]int64{
		"5m":      5 * 60 * 1000,
		"5min":    5 * 60 * 1000,
		"1h":      60 * 60 * 1000,
		"60m":     60 * 60 * 1000,
		"":        0,
		"7d":      0,
		"unknown": 0,
	}
	for retention, want := range cases {
		if got := providerTTLMillis(retention); got != want {
			t.Errorf("providerTTLMillis(%q) = %d; want %d (behavior must not drift from the literal defaults)", retention, got, want)
		}
	}
}
