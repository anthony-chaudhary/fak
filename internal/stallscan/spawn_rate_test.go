package stallscan

import "testing"

// spawn_rate_test.go — the spawn axis pinned to a MEASURED distribution.
//
// These cases exist because the axis was very nearly shipped mis-calibrated. The
// SpawnBurst field is documented as a GROSS birth count, but nothing in the tree
// ever assigned it, so it sat at zero and the axis never fired. Wiring the real
// gross count in — without changing the threshold — would have flipped the defect
// rather than fixed it: live capture on the reference box (2026-08-05, 101
// one-second ticks under ordinary fleet load: python dispatchers shelling
// git/bash/grep, a peer `go test` run, 22 resident fak workers) measured
//
//	median 22 gross births/sec, p95 63, max 83
//
// every one of which clears the count threshold of 8. A gate that refuses on 95%
// of ticks of a healthy box is not a working gate.
//
// The numbers below are that capture. They are the negative class — a loaded but
// WORKING host — and their job is to fail if anyone re-tightens the threshold to
// where ordinary fleet load trips it.

// measuredWorkingBox is the 2026-08-05 gross-birth capture: the busiest ordinary
// state this box was observed in without a freeze.
var measuredWorkingBox = []struct {
	name   string
	perSec float64
}{
	{"median", 22},
	{"p95", 63},
	{"observed max", 83},
}

func calmSample(birthsPerSec float64) Sample {
	// A sample that is otherwise unremarkable, so only the spawn axis can decide.
	return Sample{
		TotalFaultsPerSec:     1000,
		HardFaultsPerSec:      10,
		ContextSwitchesPerSec: 1000,
		SystemCallsPerSec:     1000,
		ProcessCount:          400,
		AvailableMB:           200000,
		DiskQueueLen:          0,
		SpawnBurst:            int(birthsPerSec),
		SpawnWindowSeconds:    1,
	}
}

// TestSpawnRateDoesNotFireOnMeasuredWorkingBox is the anti-false-positive pin: no
// point in the measured working-box distribution may produce a spawn_storm.
func TestSpawnRateDoesNotFireOnMeasuredWorkingBox(t *testing.T) {
	th := DefaultThresholds()
	for _, tc := range measuredWorkingBox {
		v := Classify(calmSample(tc.perSec), th)
		if v.Cause == CauseSpawnStorm {
			t.Errorf("%s of the measured WORKING-box distribution (%.0f births/sec) classified as %s; "+
				"the box was not frozen at that rate, so this threshold refuses healthy hosts",
				tc.name, tc.perSec, v.Cause)
		}
		if v.Level == LevelStall {
			t.Errorf("%s (%.0f births/sec) classified as stall: %v", tc.name, tc.perSec, v.Reasons)
		}
	}
}

// TestSpawnRateCountThresholdWouldHaveFiredOnAllOfThem records WHY the rate axis
// exists. If someone deletes the window and reverts to the bare count, this is the
// damage. Not a behavioural assertion on shipped code — a pinned witness of the
// defect the rate path prevents.
func TestSpawnRateCountThresholdWouldHaveFiredOnAllOfThem(t *testing.T) {
	th := DefaultThresholds()
	for _, tc := range measuredWorkingBox {
		s := calmSample(tc.perSec)
		s.SpawnWindowSeconds = 0 // drop the window -> legacy bare-count comparison
		v := Classify(s, th)
		if v.Cause != CauseSpawnStorm {
			t.Fatalf("expected the WINDOWLESS count path to (wrongly) fire at %.0f births "+
				"against SpawnBurstStall=%d — if it no longer does, the count threshold moved and "+
				"this witness needs re-deriving", tc.perSec, th.SpawnBurstStall)
		}
	}
}

// TestSpawnRateFiresAboveCalibratedRate keeps the axis from being disarmed
// outright: well above the measured working-box maximum it must still fire.
func TestSpawnRateFiresAboveCalibratedRate(t *testing.T) {
	th := DefaultThresholds()
	s := calmSample(th.SpawnBurstRateStall + 50)
	v := Classify(s, th)
	if v.Cause != CauseSpawnStorm {
		t.Fatalf("cause = %s, want %s at %.0f births/sec (threshold %.0f)",
			v.Cause, CauseSpawnStorm, th.SpawnBurstRateStall+50, th.SpawnBurstRateStall)
	}
	if v.Level != LevelStall {
		t.Fatalf("level = %s, want %s", v.Level, LevelStall)
	}
}

// TestSpawnRateThresholdClearsMeasuredMax is the calibration invariant itself: the
// default must sit above the highest rate observed on a working host, or the axis
// is guaranteed to produce false refusals in normal operation.
func TestSpawnRateThresholdClearsMeasuredMax(t *testing.T) {
	const measuredMax = 83.0 // 2026-08-05 capture, 101 ticks
	th := DefaultThresholds()
	if th.SpawnBurstRateStall <= measuredMax {
		t.Fatalf("SpawnBurstRateStall = %.0f/sec is at or below the measured working-box max of %.0f/sec; "+
			"ordinary fleet load would trip the spawn axis", th.SpawnBurstRateStall, measuredMax)
	}
}

// TestSpawnRateWindowRequiredForRatePath pins the guard against the conflation the
// whole change is about: a count with no window is not a rate, and must not be
// divided by an assumed one.
func TestSpawnRateWindowRequiredForRatePath(t *testing.T) {
	s := Sample{SpawnBurst: 40, SpawnWindowSeconds: 0}
	if _, ok := s.spawnRate(); ok {
		t.Fatal("spawnRate reported ok with no window — a bare count was treated as a rate")
	}
	s.SpawnWindowSeconds = 2
	r, ok := s.spawnRate()
	if !ok || r != 20 {
		t.Fatalf("spawnRate = (%v, %v), want (20, true)", r, ok)
	}
}

// TestSpawnRateProcessDeltaKeepsCountPath guards the fallback. ProcessDelta is a
// NET delta — a different quantity from a gross birth count — and SpawnBurstStall
// was calibrated against exactly it. Applying the gross RATE threshold to a net
// delta would silently disarm the axis for every caller with no gross count.
func TestSpawnRateProcessDeltaKeepsCountPath(t *testing.T) {
	th := DefaultThresholds()
	s := calmSample(0)
	s.SpawnBurst = 0
	s.SpawnWindowSeconds = 1 // window known, but there is no gross count to divide
	s.ProcessDelta = th.SpawnBurstStall + 1
	v := Classify(s, th)
	if v.Cause != CauseSpawnStorm {
		t.Fatalf("cause = %s, want %s: a net ProcessDelta of %d must still fire on the count "+
			"path even when a window is present", v.Cause, CauseSpawnStorm, s.ProcessDelta)
	}
}
