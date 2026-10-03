package model

import (
	"math"
	"reflect"
	"testing"
)

// fak-test:runtime fast est=0.1s
// [SW-VERIFIED] This oracle consumes verification observations, with no model,
// Metal device, timing claim, or physical acceptance-rate claim.
func TestMetalMTPAcceptanceWindowContainsObservedTrials(t *testing.T) {
	for _, size := range []int{8, 32} {
		c := &MetalMTPCoordinator{cfg: DefaultMetalMTPConfig()}
		c.cfg.WindowSize = size
		c.cfg.FallbackToSerial = false
		var history []bool
		proposedTotal, acceptedTotal := 0, 0
		for _, accepted := range []int{0, 4, 1, 0, 3, 2, 4, 4, 0, 2, 1, 3, 4, 0, 0, 4, 2, 3, 1} {
			// An accepted prefix is observed true; rejection terminates target
			// verification. Later drafted tokens have no acceptance observation.
			for n := 0; n < accepted; n++ {
				history = append(history, true)
			}
			if accepted < 4 {
				history = append(history, false)
			}
			if len(history) > size {
				history = history[len(history)-size:]
			}
			proposedTotal += 4
			acceptedTotal += accepted
			c.mu.Lock()
			c.recordAcceptanceLocked(4, accepted)
			got := append([]bool(nil), c.windowOutcomes...)
			if len(got) == size {
				got = append(append([]bool(nil), got[c.windowHead:]...), got[:c.windowHead]...)
			}
			c.mu.Unlock()
			if !reflect.DeepEqual(got, history) {
				t.Fatalf("window=%d accepted=%d observed=%v want=%v", size, accepted, got, history)
			}
			stats := c.Stats()
			if stats.TotalProposed != proposedTotal || stats.TotalAccepted != acceptedTotal || stats.TotalRollbacks != proposedTotal-acceptedTotal {
				t.Fatalf("lifetime accounting changed: %+v want proposed=%d accepted=%d", stats, proposedTotal, acceptedTotal)
			}
		}
	}
}

// fak-test:runtime fast est=0.1s
func TestMetalMTPCensoredSuffixDoesNotTriggerStickyFallback(t *testing.T) {
	for _, tc := range []struct {
		name         string
		accepted     []int
		wantFallback bool
		observedRate float64
	}{
		// 20 accepted, seven first rejections per cycle: alpha=20/27.
		// The early 4,0,0 prefix exposes the erroneous suffix penalty before
		// the true observations have filled the minimum evaluation window.
		{"healthy", []int{4, 0, 0, 4, 0, 3, 4, 0, 2, 3}, false, 20.0 / 27},
		{"low", []int{0, 0, 0, 0, 0, 0, 3}, true, 0.30},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c := &MetalMTPCoordinator{cfg: DefaultMetalMTPConfig()}
			for cycle := 0; cycle < 8; cycle++ {
				for _, accepted := range tc.accepted {
					c.mu.Lock()
					c.recordAcceptanceLocked(4, accepted)
					c.mu.Unlock()
					if !tc.wantFallback && c.InFallback() {
						t.Fatalf("healthy observed alpha=%.6f falsely entered sticky fallback: %+v", tc.observedRate, c.Stats())
					}
				}
			}
			if c.InFallback() != tc.wantFallback {
				t.Fatalf("alpha=%.6f fallback=%v want=%v", tc.observedRate, c.InFallback(), tc.wantFallback)
			}
			if tc.wantFallback {
				c.mu.Lock()
				c.recordAcceptanceLocked(4, 4)
				c.mu.Unlock()
				if !c.InFallback() {
					t.Fatal("fallback lost stickiness after one good round")
				}
			}
			stats := c.Stats()
			if math.IsNaN(stats.RollingRate) || stats.WindowProposed != 32 {
				t.Fatalf("bounded rolling stats=%+v", stats)
			}
		})
	}
}

// fak-test:runtime fast est=0.1s
// The target's verification outcome and the output admitted to the caller are
// separate observations. A quota/EOS limit does not turn verified true tokens
// into rejections; a matched tree leaf does not imply a further mismatch.
func TestMetalMTPAdmissionAndTreeExhaustionPreserveVerificationEvidence(t *testing.T) {
	for _, tc := range []struct {
		name                         string
		proposed, admitted, verified int
		rejected                     bool
		want                         []bool
	}{
		{"quota", 4, 2, 4, false, []bool{true, true, true, true}},
		{"eos", 4, 1, 4, false, []bool{true, true, true, true}},
		{"matched_tree_leaf", 9, 3, 3, false, []bool{true, true, true}},
		{"actual_mismatch", 9, 2, 2, true, []bool{true, true, false}},
		{"unverified_abort", 4, 0, 0, false, nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := DefaultMetalMTPConfig()
			cfg.FallbackToSerial = false
			c := &MetalMTPCoordinator{cfg: cfg}
			c.mu.Lock()
			recordMTPObservedTestOutcome(c, tc.proposed, tc.admitted, tc.verified, tc.rejected)
			got := append([]bool(nil), c.windowOutcomes...)
			c.mu.Unlock()
			if !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("verification evidence=%v want=%v (proposed=%d admitted=%d verified=%d rejected=%v)", got, tc.want, tc.proposed, tc.admitted, tc.verified, tc.rejected)
			}
			stats := c.Stats()
			if stats.TotalProposed != tc.proposed || stats.TotalAccepted != tc.admitted || stats.TotalRollbacks != tc.proposed-tc.admitted {
				t.Fatalf("output admission lifetime metrics changed: %+v", stats)
			}
		})
	}
}

// Keep the regression buildable on its parent. The old entry point exposes
// the runtime symptom when the richer verification-observation seam is absent.
func recordMTPObservedTestOutcome(c *MetalMTPCoordinator, proposed, admitted, verified int, rejected bool) {
	type observedRecorder interface {
		recordObservedAcceptanceLocked(int, int, int, bool)
	}
	if recorder, ok := any(c).(observedRecorder); ok {
		recorder.recordObservedAcceptanceLocked(proposed, admitted, verified, rejected)
		return
	}
	c.recordAcceptanceLocked(proposed, admitted)
}
