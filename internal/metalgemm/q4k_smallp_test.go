package metalgemm

import "testing"

// TestSmallPSelectorEligibility pins the platform-neutral small-P policy (fak#13694): the band is
// exactly 2<=P<=20, it is on by default, and the opt-out (SetGEMMUseSmallP(false) or
// FAK_Q4K_SMALLP=0|off|false) removes every prompt length from it.
// fak-test:runtime fast est=1ms lane=default
func TestSmallPSelectorEligibility(t *testing.T) {
	prior := q4kUseSmallP.Load()
	defer q4kUseSmallP.Store(prior)

	want := map[int]bool{0: false, 1: false, 2: true, 6: true, 9: true, 16: true, 20: true, 21: false, 32: false, 64: false}
	SetGEMMUseSmallP(true)
	if !GEMMUseSmallP() {
		t.Fatal("SetGEMMUseSmallP(true) did not enable the route")
	}
	for P, ok := range want {
		if got := q4kSmallPEligible(P); got != ok {
			t.Errorf("default-on P=%d eligible=%t, want %t", P, got, ok)
		}
		if got := q4kSmallPPromptInBand(P); got != ok {
			t.Errorf("P=%d inBand=%t, want %t", P, got, ok)
		}
	}
	SetGEMMUseSmallP(false)
	for P := range want {
		if q4kSmallPEligible(P) {
			t.Errorf("opt-out P=%d still eligible", P)
		}
	}
	for v, on := range map[string]bool{"": true, "1": true, "on": true, "0": false, "off": false, " FALSE ": false} {
		if got := q4kSmallPEnvEnabled(v); got != on {
			t.Errorf("FAK_Q4K_SMALLP=%q enabled=%t, want %t", v, got, on)
		}
	}
}
