package modelengine

import (
	"context"
	"testing"
	"time"
)

// fak-test:runtime medium est=4s lane=default
func TestNativeSchedulerSpecAdmissionPad(t *testing.T) {
	t.Run("newcomer_pads_into_resident_mtp_panel", func(t *testing.T) {
		engine, scheduler := newSpecAdmissionPadFixture(t, 0, 3)
		runContext, cancelRun := context.WithTimeout(context.Background(), 15*time.Second)
		defer cancelRun()

		resident := admitCapacityRequest(t, engine, runContext, "specpad-resident")
		nativeSchedulerDriveIteration(t, scheduler) // promote the resident, decode its first token
		awaitCapacityToken(t, resident)

		newcomer := admitCapacityRequest(t, engine, runContext, "specpad-newcomer")
		nativeSchedulerDriveIteration(t, scheduler) // promote the newcomer into the spec-carrying step

		residentLane, ok := resident.(*schedLane)
		if !ok {
			t.Fatalf("resident request type = %T, want *schedLane", resident)
		}
		newcomerLane, ok := newcomer.(*schedLane)
		if !ok {
			t.Fatalf("newcomer request type = %T, want *schedLane", newcomer)
		}

		// Admission reserved K placeholder draft slots for the newcomer: one
		// padded promotion reserving 3 slots means its first decode step is
		// scheduled for 1 target + 3 placeholders = 4 tokens.
		receipt := scheduler.SpecAdmissionPadReceipt()
		if receipt.Pads != 1 || receipt.Slots != 3 {
			t.Fatalf("admission pad receipt = %+v, want one padded promotion reserving 3 placeholder draft slots", receipt)
		}
		scheduledTokens := 1 + int(receipt.Slots/receipt.Pads)
		if scheduledTokens != 4 {
			t.Fatalf("newcomer scheduled tokens = %d, want 4 (1 target + 3 placeholders)", scheduledTokens)
		}

		scheduler.mu.Lock()
		panel, panelOK := scheduler.specStepPanelLocked()
		newcomerRunning, residentRunning := false, false
		newcomerPad, residentPad := -1, -1
		for _, ln := range scheduler.lanes {
			switch ln {
			case newcomerLane:
				newcomerRunning = true
				newcomerPad = ln.specAdmissionPad
			case residentLane:
				residentRunning = true
				residentPad = ln.specAdmissionPad
			}
		}
		scheduler.mu.Unlock()
		if !newcomerRunning || !residentRunning {
			t.Fatalf("running set after promotion = newcomer:%t resident:%t, want both", newcomerRunning, residentRunning)
		}
		// The first decode forward already consumed the reservation: rejected
		// placeholders resolve to -1 and publish nothing.
		if newcomerPad != 0 {
			t.Fatalf("newcomer admission pad after first decode step = %d, want 0 (placeholders resolved)", newcomerPad)
		}
		if residentPad != 0 {
			t.Fatalf("resident admission pad = %d, want 0 (resident drafting unchanged)", residentPad)
		}
		if !panelOK || panel != 4 {
			t.Fatalf("step panel = (%d, %t), want resident bucket (4, true)", panel, panelOK)
		}
		if panel != scheduledTokens {
			t.Fatalf("step shape %d differs from newcomer scheduled tokens %d", panel, scheduledTokens)
		}
	})

	t.Run("exhausted_budget_holds_newcomer_in_waiting", func(t *testing.T) {
		engine, scheduler := newSpecAdmissionPadFixture(t, 2, 3)
		runContext, cancelRun := context.WithTimeout(context.Background(), 15*time.Second)
		defer cancelRun()

		resident := admitCapacityRequest(t, engine, runContext, "specpad-resident")
		nativeSchedulerDriveIteration(t, scheduler)
		awaitCapacityToken(t, resident)

		// Remaining budget is 2 - 1 (resident) = 1; the padded cost 1+3 = 4
		// cannot fit, so the step prefers not to schedule the newcomer.
		newcomer := admitCapacityRequest(t, engine, runContext, "specpad-newcomer")
		nativeSchedulerDriveIteration(t, scheduler)

		assertCapacityNoToken(t, newcomer)
		newcomerLane, ok := newcomer.(*schedLane)
		if !ok {
			t.Fatalf("newcomer request type = %T, want *schedLane", newcomer)
		}
		scheduler.mu.Lock()
		waitingHeld := false
		for _, ln := range scheduler.waiting {
			if ln == newcomerLane {
				waitingHeld = true
				break
			}
		}
		scheduler.mu.Unlock()
		if !waitingHeld {
			t.Fatal("budget-exhausted step admitted the padded newcomer, want it held in waiting")
		}
		if receipt := scheduler.SpecAdmissionPadReceipt(); receipt.Pads != 0 || receipt.Slots != 0 {
			t.Fatalf("admission pad receipt = %+v, want no reservation when the pad is refused", receipt)
		}

		// Skip-not-break: the refusal holds only this lane; the resident keeps
		// decoding through the same step loop.
		select {
		case _, open := <-resident.Tokens():
			if !open {
				t.Fatal("resident token stream closed before the post-refusal step")
			}
		case <-time.After(5 * time.Second):
			t.Fatal("timed out waiting for the resident's post-refusal token")
		}
		nativeSchedulerDriveIteration(t, scheduler)
		assertCapacityNoToken(t, newcomer)
	})

	t.Run("non_spec_admission_unchanged", func(t *testing.T) {
		engine, scheduler := newSpecAdmissionPadFixture(t, 0, 0)
		runContext, cancelRun := context.WithTimeout(context.Background(), 15*time.Second)
		defer cancelRun()

		resident := admitCapacityRequest(t, engine, runContext, "specpad-resident")
		nativeSchedulerDriveIteration(t, scheduler)
		awaitCapacityToken(t, resident)

		newcomer := admitCapacityRequest(t, engine, runContext, "specpad-newcomer")
		nativeSchedulerDriveIteration(t, scheduler)

		newcomerLane, ok := newcomer.(*schedLane)
		if !ok {
			t.Fatalf("newcomer request type = %T, want *schedLane", newcomer)
		}
		receipt := scheduler.SpecAdmissionPadReceipt()
		if receipt.Pads != 0 || receipt.Slots != 0 {
			t.Fatalf("admission pad receipt = %+v, want zero without the spec knob", receipt)
		}
		scheduler.mu.Lock()
		panel, panelOK := scheduler.specStepPanelLocked()
		newcomerRunning := false
		newcomerPad := -1
		for _, ln := range scheduler.lanes {
			if ln == newcomerLane {
				newcomerRunning = true
				newcomerPad = ln.specAdmissionPad
			}
		}
		scheduler.mu.Unlock()
		if !newcomerRunning {
			t.Fatal("newcomer was not promoted into the running set")
		}
		if newcomerPad != 0 {
			t.Fatalf("newcomer admission pad = %d, want 0 (non-spec batches admit with 1 token unchanged)", newcomerPad)
		}
		if panelOK {
			t.Fatalf("step panel = (%d, true), want no spec panel without the knob", panel)
		}
	})

	t.Run("serve_env_knob_arms_the_pad", func(t *testing.T) {
		t.Setenv(envNativeSpecAdmissionPad, "3")
		scheduler := &NativeScheduler{}
		applyNativeServeKnobs(scheduler, 128)
		scheduler.mu.Lock()
		got := scheduler.specPadK
		scheduler.mu.Unlock()
		if got != 3 {
			t.Fatalf("specPadK after applyNativeServeKnobs = %d, want 3", got)
		}
	})
}

// newSpecAdmissionPadFixture builds a two-lane engine over the resident-Q4_K
// prefill fixture with deterministic 8-token prompts, parked in manual-drain
// mode so the scheduler advances only when the test drives an iteration.
func newSpecAdmissionPadFixture(t *testing.T, maxTokensPerStep, specPad int) (*Engine, *NativeScheduler) {
	t.Helper()
	prefillModel := nativeSchedulerPrefillModel(t)
	engine := New()
	engine.PreloadQ4K(prefillModel)
	scheduler := engine.nativeScheduler()
	scheduler.prepare = nativeSchedulerPrefillPrepare(map[string][]int{
		"specpad-resident": nativeSchedulerQwenPrompt(8),
		"specpad-newcomer": nativeSchedulerQwenPrompt(8),
	})
	scheduler.SetMaxRunning(2)
	if maxTokensPerStep > 0 {
		scheduler.SetMaxTokensPerStep(maxTokensPerStep)
	}
	if specPad > 0 {
		scheduler.SetSpecAdmissionPad(specPad)
	}
	nativeSchedulerBeginManualDrain(t, scheduler)
	t.Cleanup(func() {
		nativeSchedulerEndManualDrain(scheduler)
	})
	return engine, scheduler
}
