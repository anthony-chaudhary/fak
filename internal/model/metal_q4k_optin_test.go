//go:build darwin && arm64 && cgo

package model

import "testing"

// TestQ4KM5OptInDefaultsOn pins the default-on contract for the widened-panel wide-tile candidate
// (fak#13041 / fak#13124). The sanctioned on-silicon M3 Pro receipt pinned a crossover row, so the
// model-layer process opt-in now defaults ON; only an explicit FAK_Q4K_M5=0 forces the scalar
// kernel. The real safety is the encode-time device/version-pinned crossover gate in metalgemm
// (q4kGEMMModeForPrompt), which keeps default-on inert on any device without a pinned row.
func TestQ4KM5OptInDefaultsOn(t *testing.T) {
	for _, tc := range []struct {
		env  string
		want bool
	}{
		{"", true},      // unset: default ON
		{"1", true},     // explicit enable
		{"0", false},    // explicit opt-out
		{"true", true},  // any non-"0" value stays on (opt-out is the only escape)
		{"false", true}, // "false" is not the documented opt-out token
		{"00", true},    // only the exact token "0" opts out
	} {
		t.Setenv("FAK_Q4K_M5", tc.env)
		if got := q4kM5OptIn(); got != tc.want {
			t.Errorf("q4kM5OptIn with FAK_Q4K_M5=%q = %t, want %t", tc.env, got, tc.want)
		}
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
