package amdgpuclock

import "testing"

// fak-test:runtime fast est=10ms lane=default
func TestCanonicalDPMShaderAcceptance(t *testing.T) {
	t.Run("ParseDPMSCLK", func(t *testing.T) {
		raw := `0: 600Mhz *
1: 1100Mhz 
2: 2200Mhz 
`
		states, activeIdx, activeFreq, err := ParseDPMSCLK(raw)
		if err != nil {
			t.Fatalf("ParseDPMSCLK failed: %v", err)
		}
		if len(states) != 3 {
			t.Fatalf("expected 3 states, got %d", len(states))
		}
		if activeIdx != 0 {
			t.Errorf("expected activeIdx 0, got %d", activeIdx)
		}
		if activeFreq != 600 {
			t.Errorf("expected activeFreq 600, got %d", activeFreq)
		}

		// Test active state 2
		raw2 := `0: 600Mhz
1: 1100Mhz
2: 2200Mhz *
`
		_, activeIdx2, activeFreq2, err := ParseDPMSCLK(raw2)
		if err != nil {
			t.Fatalf("ParseDPMSCLK failed on active state 2: %v", err)
		}
		if activeIdx2 != 2 || activeFreq2 != 2200 {
			t.Errorf("expected activeIdx 2 and 2200 MHz, got %d and %d", activeIdx2, activeFreq2)
		}

		// Test empty/invalid
		_, _, _, err = ParseDPMSCLK("")
		if err == nil {
			t.Error("expected error on empty input")
		}
		_, _, _, err = ParseDPMSCLK("none")
		if err == nil {
			t.Error("expected error on 'none' input")
		}
	})

	t.Run("FindStateForFrequency", func(t *testing.T) {
		states := []DPMSCLKState{
			{Index: 0, FreqMHz: 600},
			{Index: 1, FreqMHz: 1100},
			{Index: 2, FreqMHz: 2200},
		}

		idx, err := FindStateForFrequency(states, 2200)
		if err != nil || idx != 2 {
			t.Errorf("expected state 2 for 2200 MHz, got %d (err: %v)", idx, err)
		}

		// Closest <= 2000 is 1100 (state 1)
		idx, err = FindStateForFrequency(states, 2000)
		if err != nil || idx != 1 {
			t.Errorf("expected state 1 for 2000 MHz, got %d (err: %v)", idx, err)
		}

		// Higher than max returns max state
		idx, err = FindStateForFrequency(states, 2900)
		if err != nil || idx != 2 {
			t.Errorf("expected state 2 for 2900 MHz, got %d (err: %v)", idx, err)
		}
	})

}

// fak-test:runtime fast est=10ms lane=default
func TestCanonicalDPMMemoryUnits(t *testing.T) {
	states, idx, mhz, mts, err := ParseDPMMCLK("0: 2000MT/s *\n1: 6400MT/s\n2: 8000MT/s\n")
	if err != nil || len(states) != 3 || idx != 0 || mhz != 400 || mts != 2000 {
		t.Fatalf("memory units: %+v %d %d %d %v", states, idx, mhz, mts, err)
	}
	peak, err := FindPeakMCLKState(states, 8000)
	if err != nil || peak != 2 {
		t.Fatalf("peak=%d %v", peak, err)
	}
	if FrequencyToMTs(1000) != 8000 {
		t.Fatal("LPDDR conversion changed")
	}
	for _, raw := range []string{"", "none"} {
		if _, _, _, _, err := ParseDPMMCLK(raw); err == nil {
			t.Fatalf("invalid input %q accepted", raw)
		}
	}
}
