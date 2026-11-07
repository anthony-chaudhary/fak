package agent

import (
	"testing"
	"time"
)

// batchHealthWithFloor builds a planner carrying the given per-token floor
// (0 = unknown) and a freshly constructed health governor around it.
func batchHealthWithFloor(t *testing.T, floor time.Duration) (*InKernelPlanner, *batchStepHealth) {
	t.Helper()
	p := &InKernelPlanner{}
	if floor > 0 {
		p.batchSerialPerLaneNS.Store(int64(floor))
	}
	return p, newBatchStepHealth(p)
}

func assertFloor(t *testing.T, p *InKernelPlanner, want time.Duration) {
	t.Helper()
	if got := p.batchSerialPerLaneNS.Load(); got != int64(want) {
		t.Fatalf("floor=%d ns, want %d ns", got, int64(want))
	}
}

// TestBatchStepHealth pins the cohort batched->serial degradation policy: a
// ratcheting-MIN per-token floor fed by BOTH serial steps and healthy batched
// steps (wall/lanes each), a per-token violation budget of floor x margin,
// degradation after two consecutive violations, the serial probe cadence, and
// the margin env knob. Measured context: rtp-test-h200 q2kxl rank-1 collapse
// (aggregate decode 25.3 -> 0.9 tok/s from C=1 to C=16) and the diag2 finding
// that serial probes queue behind the shared stream's prefill backlog, so the
// floor must co-feed from healthy low-lane batched steps.
//
// fak-test:runtime fast est=50ms lane=default
func TestBatchStepHealth(t *testing.T) {
	t.Run("min ratchet over both sources", func(t *testing.T) {
		p, h := batchHealthWithFloor(t, 40*time.Millisecond) // budget at margin 1250: 50ms/token
		// A larger serial sample never raises the floor.
		h.observeSerial(160*time.Millisecond, 2) // per = 80ms > 40ms
		assertFloor(t, p, 40*time.Millisecond)
		// An equal serial sample leaves it.
		h.observeSerial(80*time.Millisecond, 2) // per = 40ms == floor
		assertFloor(t, p, 40*time.Millisecond)
		// A healthy batched observation lowers the floor exactly like a serial one.
		h.observeBatched(70*time.Millisecond, 2) // per = 35ms <= 50ms budget
		assertFloor(t, p, 35*time.Millisecond)
		// Later larger samples from either source never raise it again.
		h.observeSerial(200*time.Millisecond, 2)  // per = 100ms
		h.observeBatched(120*time.Millisecond, 2) // per = 60ms > 43.75ms budget: violation, no feed
		assertFloor(t, p, 35*time.Millisecond)
	})

	t.Run("healthy batched step feeds the floor", func(t *testing.T) {
		// The measured healthy C=2/4 regime runs ~24-26ms/token; a healthy
		// batched step at 24ms/token must lower a 26ms floor.
		p, h := batchHealthWithFloor(t, 26*time.Millisecond)
		h.observeBatched(48*time.Millisecond, 2) // per = 24ms <= 32.5ms budget
		assertFloor(t, p, 24*time.Millisecond)
		// A violating batched step does NOT feed the floor.
		h.observeBatched(400*time.Millisecond, 2) // per = 200ms >> 30ms budget
		assertFloor(t, p, 24*time.Millisecond)
	})

	t.Run("batched step seeds an unknown floor", func(t *testing.T) {
		p, h := batchHealthWithFloor(t, 0)
		h.observeBatched(50*time.Millisecond, 2) // per = 25ms seeds
		assertFloor(t, p, 25*time.Millisecond)
		if !h.calibrated() {
			t.Fatal("seeded floor did not calibrate the governor")
		}
	})

	t.Run("calibration window folds as MAX then ratchets MIN", func(t *testing.T) {
		// Measured failure this pins: the serve warmup's micro-forward
		// (337us for a 1-lane serial step on the q2kxl H200 route) seeded a
		// garbage-low floor that the MIN ratchet could never raise, forcing
		// every cell to serial. The first batchStepSeedSamples samples fold as
		// MAX; afterwards the MIN ratchet takes over.
		p, h := batchHealthWithFloor(t, 0)
		h.observeSerial(337*time.Microsecond, 1) // warmup-class micro-forward
		assertFloor(t, p, 337*time.Microsecond)
		h.observeSerial(60*time.Millisecond, 2) // real 2-lane serial step: per = 30ms
		assertFloor(t, p, 30*time.Millisecond)  // calibration MAX raised it
		h.observeBatched(50*time.Millisecond, 2)
		assertFloor(t, p, 30*time.Millisecond) // per = 25ms < 30ms inside the window: MAX keeps 30ms
		// Window closed: the MIN ratchet takes over.
		h.observeSerial(40*time.Millisecond, 2) // per = 20ms lowers
		assertFloor(t, p, 20*time.Millisecond)
		h.observeSerial(80*time.Millisecond, 2) // per = 40ms: MIN keeps 20ms
		assertFloor(t, p, 20*time.Millisecond)
		if got := p.batchSeedSamples.Load(); got != batchStepSeedSamples {
			t.Fatalf("seed samples=%d, want exactly %d (the window closed)", got, batchStepSeedSamples)
		}
		if !p.batchSeedCalibrating.Load() {
			t.Fatal("calibration window never opened for an unknown floor")
		}
	})

	t.Run("preset floor skips the calibration window", func(t *testing.T) {
		p, h := batchHealthWithFloor(t, 40*time.Millisecond)
		h.observeSerial(160*time.Millisecond, 2) // per = 80ms > preset 40ms
		assertFloor(t, p, 40*time.Millisecond)   // MIN ratchet: never raised
		if p.batchSeedCalibrating.Load() {
			t.Fatal("preset floor opened the calibration window")
		}
	})

	t.Run("per-token violation boundary", func(t *testing.T) {
		// floor 40ms, margin 1250 -> budget 50ms per token.
		p, h := batchHealthWithFloor(t, 40*time.Millisecond)
		h.observeBatched(100*time.Millisecond, 2) // per = 50ms == budget: not a violation
		if h.violations != 0 || h.degraded {
			t.Fatalf("boundary-equal step counted as violation: violations=%d degraded=%v", h.violations, h.degraded)
		}
		h.observeBatched(101*time.Millisecond, 2) // per = 50.5ms > 50ms: violation
		if h.violations != 1 || h.degraded {
			t.Fatalf("violating step: violations=%d degraded=%v, want 1/false", h.violations, h.degraded)
		}
		_ = p
	})

	t.Run("two consecutive violations degrade", func(t *testing.T) {
		_, h := batchHealthWithFloor(t, 40*time.Millisecond)
		h.observeBatched(101*time.Millisecond, 2)
		if h.useSerial() {
			t.Fatal("single violation degraded the episode")
		}
		h.observeBatched(101*time.Millisecond, 2)
		if !h.useSerial() {
			t.Fatal("two consecutive violations did not degrade the episode")
		}
		// Once degraded, further batched observations are inert.
		h.observeBatched(101*time.Millisecond, 2)
		if h.violations != 2 {
			t.Fatalf("degraded episode kept counting violations: %d", h.violations)
		}
	})

	t.Run("clean step resets the violation streak", func(t *testing.T) {
		_, h := batchHealthWithFloor(t, 40*time.Millisecond)
		h.observeBatched(101*time.Millisecond, 2) // violation 1
		h.observeBatched(90*time.Millisecond, 2)  // per = 45ms <= 50ms: clean
		if h.violations != 0 {
			t.Fatalf("clean step left violations=%d", h.violations)
		}
		h.observeBatched(101*time.Millisecond, 2) // violation again, but only #1
		if h.useSerial() {
			t.Fatal("reset streak re-degraded on a single violation")
		}
	})

	t.Run("probe due on the 8 cadence", func(t *testing.T) {
		_, h := batchHealthWithFloor(t, 40*time.Millisecond)
		for i := 0; i < batchStepProbeEvery-1; i++ {
			h.observeBatched(90*time.Millisecond, 2) // healthy steps
			if h.probeDue() {
				t.Fatalf("probe due after %d batched steps, want %d", i+1, batchStepProbeEvery)
			}
		}
		h.observeBatched(90*time.Millisecond, 2)
		if !h.probeDue() {
			t.Fatalf("probe not due after %d batched steps", batchStepProbeEvery)
		}
	})

	t.Run("probe never when uncalibrated or degraded", func(t *testing.T) {
		_, h := batchHealthWithFloor(t, 0)
		h.sinceProbe = batchStepProbeEvery * 2
		if h.probeDue() {
			t.Fatal("probe due without a calibrated floor")
		}
		p, h := batchHealthWithFloor(t, 40*time.Millisecond)
		h.degraded = true
		h.sinceProbe = batchStepProbeEvery * 2
		if h.probeDue() {
			t.Fatal("probe due in a degraded episode")
		}
		_ = p
	})

	t.Run("margin env parsing", func(t *testing.T) {
		cases := []struct {
			raw  string
			want int
		}{
			{"", 1250},     // unset: default margin
			{"0", 0},       // explicit disable
			{"50", 1250},   // below 100: ignored
			{"99", 1250},   // just below: ignored
			{"100", 100},   // minimum accepted
			{"2000", 2000}, // custom margin
			{"junk", 1250}, // unparseable: ignored
			{"-5", 1250},   // negative: ignored
		}
		for _, tc := range cases {
			t.Setenv(batchStepHealthEnv, tc.raw)
			h := newBatchStepHealth(&InKernelPlanner{})
			if h.marginNum != tc.want {
				t.Errorf("FAK_INKERNEL_BATCH_HEALTH_MARGIN=%q gave margin %d, want %d", tc.raw, h.marginNum, tc.want)
			}
		}
	})

	t.Run("explicit zero disables the policy", func(t *testing.T) {
		t.Setenv(batchStepHealthEnv, "0")
		p := &InKernelPlanner{}
		p.batchSerialPerLaneNS.Store(int64(40 * time.Millisecond))
		h := newBatchStepHealth(p)
		h.observeBatched(400*time.Millisecond, 2) // would violate if armed
		if h.degraded || h.violations != 0 {
			t.Fatalf("disabled policy reacted: violations=%d degraded=%v", h.violations, h.degraded)
		}
		assertFloor(t, p, 40*time.Millisecond) // disabled policy neither feeds nor probes
		if h.calibrated() {
			t.Fatal("disabled policy reports calibrated")
		}
	})

	t.Run("debt caps at the 2s window", func(t *testing.T) {
		share := newInKernelDecodeShare(0.5, 0) // p=0.5: a prompt step of t owes t
		share.stepped(10_000_000_000, true)     // 10s prompt fill clamps to the cap
		if share.debtNs != inKernelDecodeShareDebtCapNs {
			t.Fatalf("debt=%d, want capped %d", share.debtNs, inKernelDecodeShareDebtCapNs)
		}
		share.stepped(10_000_000_000, true) // stays capped
		if share.debtNs != inKernelDecodeShareDebtCapNs {
			t.Fatalf("debt=%d, want capped %d", share.debtNs, inKernelDecodeShareDebtCapNs)
		}
		share.rounded(1_000_000_000)
		if share.debtNs != inKernelDecodeShareDebtCapNs-1_000_000_000 {
			t.Fatalf("rounded debt=%d, want %d", share.debtNs, inKernelDecodeShareDebtCapNs-1_000_000_000)
		}
		// Sub-cap charges accumulate normally up to the cap, then clamp.
		fresh := newInKernelDecodeShare(0.5, 0)
		fresh.stepped(1_000_000_000, true)
		fresh.stepped(1_000_000_000, true)
		if fresh.debtNs != inKernelDecodeShareDebtCapNs {
			t.Fatalf("accumulated debt=%d, want %d", fresh.debtNs, inKernelDecodeShareDebtCapNs)
		}
		fresh.stepped(1_000_000_000, true)
		if fresh.debtNs != inKernelDecodeShareDebtCapNs {
			t.Fatalf("over-cap debt=%d, want clamped %d", fresh.debtNs, inKernelDecodeShareDebtCapNs)
		}
	})

	t.Run("decode share defaults to 0.5 when unset", func(t *testing.T) {
		t.Setenv(inKernelDecodeShareEnv, "")
		t.Setenv(inKernelPromptRowBudgetEnv, "")
		if got := inKernelDecodeShareFromEnv(); got.promptShare != 0.5 {
			t.Fatalf("unset share gave prompt share %v, want 0.5", got.promptShare)
		}
		t.Setenv(inKernelDecodeShareEnv, "0")
		if got := inKernelDecodeShareFromEnv(); got.promptShare != 1 {
			t.Fatalf("explicit 0 gave prompt share %v, want 1 (historical whole-prompt order)", got.promptShare)
		}
		t.Setenv(inKernelDecodeShareEnv, "0.25")
		if got := inKernelDecodeShareFromEnv(); got.promptShare != 0.75 {
			t.Fatalf("0.25 share gave prompt share %v, want 0.75", got.promptShare)
		}
	})
}
