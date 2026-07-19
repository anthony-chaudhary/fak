package dispatchtick

import "testing"

// The contract for #3575's cross-provider seat failover: the pure decision that the
// preflight shell wires the live tick's launch-target onto. Each case is one leg of the
// acceptance -- unset knob is byte-identical, a servable primary is never overridden, a
// single no-seat blip does not switch, and only a debounced wall over a servable fallback
// fails over.
func TestDecideFallbackProduct(t *testing.T) {
	const primary, fallback = "claude", "codex"
	cases := []struct {
		name          string
		in            FallbackProductInput
		wantEngaged   bool
		wantArmed     bool
		wantEnabled   bool
		wantLaunch    string
		wantRefusedBy string
	}{
		{
			name:        "disabled knob launches primary and stays disabled (byte-identical)",
			in:          FallbackProductInput{Enabled: false, PrimaryProduct: primary, FallbackProduct: "", PrimaryVerdict: PreflightRefuseNoAccount, ConsecutiveRefusals: 9},
			wantEnabled: false,
			wantLaunch:  primary,
		},
		{
			name:        "fallback equal to primary is treated as disabled",
			in:          FallbackProductInput{Enabled: true, PrimaryProduct: primary, FallbackProduct: primary, PrimaryVerdict: PreflightRefuseNoAccount, ConsecutiveRefusals: 9, FallbackServable: true},
			wantEnabled: false,
			wantLaunch:  primary,
		},
		{
			name:        "servable primary (SPAWN_OK) never fails over",
			in:          FallbackProductInput{Enabled: true, PrimaryProduct: primary, FallbackProduct: fallback, PrimaryVerdict: PreflightOKVerdict, ConsecutiveRefusals: 0, FallbackServable: true},
			wantEnabled: true,
			wantLaunch:  primary,
		},
		{
			name:        "non-seat refusal (at-cap) holds on primary, does not fail over",
			in:          FallbackProductInput{Enabled: true, PrimaryProduct: primary, FallbackProduct: fallback, PrimaryVerdict: PreflightRefuseAtCap, ConsecutiveRefusals: 9, FallbackServable: true},
			wantEnabled: true,
			wantLaunch:  primary,
		},
		{
			name:          "single no-seat tick is below the debounce -- no switch",
			in:            FallbackProductInput{Enabled: true, PrimaryProduct: primary, FallbackProduct: fallback, PrimaryVerdict: PreflightRefuseNoAccount, ConsecutiveRefusals: 1, DebounceThreshold: 3, FallbackServable: true},
			wantEnabled:   true,
			wantArmed:     false,
			wantLaunch:    primary,
			wantRefusedBy: primary,
		},
		{
			name:          "N consecutive no-seat ticks over a servable fallback fails over",
			in:            FallbackProductInput{Enabled: true, PrimaryProduct: primary, FallbackProduct: fallback, PrimaryVerdict: PreflightRefuseNoAccount, ConsecutiveRefusals: 3, DebounceThreshold: 3, FallbackServable: true},
			wantEnabled:   true,
			wantEngaged:   true,
			wantArmed:     true,
			wantLaunch:    fallback,
			wantRefusedBy: primary,
		},
		{
			name:          "past the debounce but fallback also walled -- armed, holds on primary",
			in:            FallbackProductInput{Enabled: true, PrimaryProduct: primary, FallbackProduct: fallback, PrimaryVerdict: PreflightRefuseNoAccount, ConsecutiveRefusals: 5, DebounceThreshold: 3, FallbackServable: false},
			wantEnabled:   true,
			wantEngaged:   false,
			wantArmed:     true,
			wantLaunch:    primary,
			wantRefusedBy: primary,
		},
		{
			name:          "default debounce applies when threshold unset (2 ticks is below default 3)",
			in:            FallbackProductInput{Enabled: true, PrimaryProduct: primary, FallbackProduct: fallback, PrimaryVerdict: PreflightRefuseNoAccount, ConsecutiveRefusals: 2, FallbackServable: true},
			wantEnabled:   true,
			wantEngaged:   false,
			wantLaunch:    primary,
			wantRefusedBy: primary,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := DecideFallbackProduct(tc.in)
			if got.Engaged != tc.wantEngaged {
				t.Errorf("Engaged = %v, want %v (reason: %s)", got.Engaged, tc.wantEngaged, got.Reason)
			}
			if got.ShouldFailover() != tc.wantEngaged {
				t.Errorf("ShouldFailover = %v, want %v", got.ShouldFailover(), tc.wantEngaged)
			}
			if got.Armed != tc.wantArmed {
				t.Errorf("Armed = %v, want %v (reason: %s)", got.Armed, tc.wantArmed, got.Reason)
			}
			if got.Enabled != tc.wantEnabled {
				t.Errorf("Enabled = %v, want %v", got.Enabled, tc.wantEnabled)
			}
			if got.LaunchProduct != tc.wantLaunch {
				t.Errorf("LaunchProduct = %q, want %q", got.LaunchProduct, tc.wantLaunch)
			}
			if got.RefusedProduct != tc.wantRefusedBy {
				t.Errorf("RefusedProduct = %q, want %q", got.RefusedProduct, tc.wantRefusedBy)
			}
			if got.Reason == "" {
				t.Error("Reason must never be empty -- the readout has to be legible")
			}
		})
	}
}

// The debounce counter: leading REFUSE_NO_ACCOUNT verdicts in a newest-first slice, stopping
// at the first non-wall verdict, so a fresh wall counts from zero.
func TestCountTrailingNoAccountRefusals(t *testing.T) {
	cases := []struct {
		name    string
		reasons []string
		want    int
	}{
		{"empty ledger", nil, 0},
		{"first verdict admitted", []string{PreflightOKVerdict, PreflightRefuseNoAccount}, 0},
		{"two walls then an admit", []string{PreflightRefuseNoAccount, PreflightRefuseNoAccount, PreflightOKVerdict, PreflightRefuseNoAccount}, 2},
		{"unbroken wall", []string{PreflightRefuseNoAccount, PreflightRefuseNoAccount, PreflightRefuseNoAccount}, 3},
		{"a different refusal ends the wall", []string{PreflightRefuseNoAccount, PreflightRefuseAtCap, PreflightRefuseNoAccount}, 1},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := CountTrailingNoAccountRefusals(tc.reasons); got != tc.want {
				t.Errorf("CountTrailingNoAccountRefusals(%v) = %d, want %d", tc.reasons, got, tc.want)
			}
		})
	}
}

// The readout is the operator-legible half of the acceptance: which pool refused, which
// product would take the work, every key present so a loop or a human can route on it.
func TestFallbackProductChoiceMap(t *testing.T) {
	m := DecideFallbackProduct(FallbackProductInput{
		Enabled:             true,
		PrimaryProduct:      "claude",
		FallbackProduct:     "codex",
		PrimaryVerdict:      PreflightRefuseNoAccount,
		ConsecutiveRefusals: 3,
		DebounceThreshold:   3,
		FallbackServable:    true,
	}).Map()
	for _, k := range []string{"schema", "enabled", "engaged", "armed", "primary_product", "refused_product", "fallback_product", "fallback_servable", "launch_product", "launch_backend", "consecutive_refusals", "debounce_threshold", "reason"} {
		if _, ok := m[k]; !ok {
			t.Errorf("readout missing key %q", k)
		}
	}
	if m["schema"] != FallbackSchema {
		t.Errorf("schema = %v, want %q", m["schema"], FallbackSchema)
	}
	if m["engaged"] != true || m["launch_product"] != "codex" || m["refused_product"] != "claude" {
		t.Errorf("engaged readout wrong: %+v", m)
	}
}
