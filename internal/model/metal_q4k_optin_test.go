//go:build darwin && arm64 && cgo

package model

import "testing"

// TestQ4KM5OptInDefaultsOn pins the default-on contract for the widened-panel wide-tile candidate
// (fak#13041 / fak#13124). The sanctioned on-silicon M3 Pro receipt pinned a crossover row, so the
// model-layer config-surface opt-in now defaults ON; only an explicit SetQ4KM5OptIn(false) forces
// the scalar kernel. The real safety is the encode-time device/version-pinned crossover gate in
// metalgemm (q4kGEMMModeForPrompt), which keeps default-on inert on any device without a pinned row.
// The knob is a config-surface seam rather than an environment read (internal/envconfiglint
// CONFIG_NOT_ENV); the former FAK_Q4K_M5 env read was relocated here.
func TestQ4KM5OptInDefaultsOn(t *testing.T) {
	// Undeclared default: ON. Restore the ambient declaration after the test.
	defer SetQ4KM5OptIn(true)
	if got := q4kM5OptIn(); got != true {
		t.Errorf("q4kM5OptIn with the seam undeclared = %t, want true (default ON)", got)
	}
	for _, tc := range []struct {
		on   bool
		want bool
	}{
		{true, true},   // explicit enable
		{false, false}, // explicit opt-out
	} {
		SetQ4KM5OptIn(tc.on)
		if got := q4kM5OptIn(); got != tc.want {
			t.Errorf("q4kM5OptIn with SetQ4KM5OptIn(%t) = %t, want %t", tc.on, got, tc.want)
		}
	}
	// Only the seam declaration flips the knob; a bare default stays ON.
	SetQ4KM5OptIn(true)
	if got := q4kM5OptIn(); got != true {
		t.Errorf("q4kM5OptIn after re-enable = %t, want true", got)
	}
}

// TestQ4KMMOptInDefaultsOff pins the sibling MM32 opt-in as still default-OFF: unlike the wide-tile
// candidate it has not yet earned a sanctioned default-on receipt, so only an explicit
// FAK_Q4K_MM=1 enables it. This guards against accidentally widening the MM32 envelope.
func TestQ4KMMOptInDefaultsOff(t *testing.T) {
	for _, tc := range []struct {
		env  string
		want bool
	}{
		{"", false},     // unset: default OFF
		{"1", true},     // explicit enable
		{"0", false},    // explicit disable
		{"true", false}, // only the exact token "1" enables it
	} {
		t.Setenv("FAK_Q4K_MM", tc.env)
		if got := q4kMMOptIn(); got != tc.want {
			t.Errorf("q4kMMOptIn with FAK_Q4K_MM=%q = %t, want %t", tc.env, got, tc.want)
		}
	}
}
