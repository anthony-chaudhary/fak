package main

import (
	"testing"
	"time"
)

// TestTenantAdmissionResolveFlags pins the --tenant-* flag to gate-policy mapping
// (#14453): all-off stays fully inert (the historical byte-for-byte path), each armed
// dimension lands in its gate's policy, and a negative value fails loud instead of
// silently disabling a fairness gate.
func TestTenantAdmissionResolveFlags(t *testing.T) {
	newSF := func(rps, burst, inflight, tokens int, window time.Duration) *serveFlags {
		return &serveFlags{
			tenantRateRPS:           &rps,
			tenantRateBurst:         &burst,
			tenantMaxInflight:       &inflight,
			tenantTokenWindowTokens: &tokens,
			tenantTokenWindow:       &window,
		}
	}

	t.Run("all_off_is_inert", func(t *testing.T) {
		rate, ig, tok, err := resolveTenantAdmission(newSF(0, 0, 0, 0, time.Minute))
		if err != nil {
			t.Fatalf("resolveTenantAdmission(all-off) err = %v, want nil", err)
		}
		if rate.RequestsPerSecond != 0 || rate.RequestBurst != 0 {
			t.Fatalf("rate policy = %+v, want disabled", rate)
		}
		if ig.PerPrincipal != 0 {
			t.Fatalf("inflight policy = %+v, want disabled", ig)
		}
		if tok.Caps.MaxTotalTokens != 0 {
			t.Fatalf("token policy = %+v, want disabled", tok)
		}
	})

	t.Run("armed_dimensions_map_to_policies", func(t *testing.T) {
		rate, ig, tok, err := resolveTenantAdmission(newSF(4, 8, 2, 1000, 30*time.Second))
		if err != nil {
			t.Fatalf("resolveTenantAdmission(armed) err = %v, want nil", err)
		}
		if rate.RequestsPerSecond != 4 || rate.RequestBurst != 8 {
			t.Fatalf("rate policy = %+v, want 4 req/s burst 8", rate)
		}
		if ig.PerPrincipal != 2 {
			t.Fatalf("inflight policy = %+v, want PerPrincipal 2", ig)
		}
		if tok.Caps.MaxTotalTokens != 1000 || tok.Window != 30*time.Second {
			t.Fatalf("token policy = %+v, want 1000 tokens per 30s", tok)
		}
	})

	for name, sf := range map[string]*serveFlags{
		"negative_rate":      newSF(-1, 0, 0, 0, time.Minute),
		"negative_burst":     newSF(1, -1, 0, 0, time.Minute),
		"negative_inflight":  newSF(0, 0, -1, 0, time.Minute),
		"negative_tokens":    newSF(0, 0, 0, -1, time.Minute),
		"nonpositive_window": newSF(0, 0, 0, 0, 0),
	} {
		t.Run(name, func(t *testing.T) {
			if _, _, _, err := resolveTenantAdmission(sf); err == nil {
				t.Fatalf("resolveTenantAdmission(%s) err = nil, want a loud refusal", name)
			}
		})
	}
}
