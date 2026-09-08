package model

import (
	"errors"
	"math"
	"sync"
	"testing"
)

func TestSpeculativeControl_InitializationAndClamping(t *testing.T) {
	kMax := 8
	ctrl := NewSpeculativeControl(kMax, 4, 0.6)

	if ctrl.KMax() != kMax {
		t.Fatalf("expected KMax=%d, got %d", kMax, ctrl.KMax())
	}
	if ctrl.GetDepth() != 4 {
		t.Fatalf("expected initial depth=4, got %d", ctrl.GetDepth())
	}
	if math.Abs(ctrl.GetAcceptanceThreshold()-0.6) > 1e-6 {
		t.Fatalf("expected initial threshold=0.6, got %f", ctrl.GetAcceptanceThreshold())
	}
	if !ctrl.IsActive() {
		t.Fatalf("expected IsActive=true for depth=4")
	}

	// Initialization clamping for defaultK
	ctrlUnder := NewSpeculativeControl(kMax, -5, 0.5)
	if ctrlUnder.GetDepth() != 0 {
		t.Fatalf("expected defaultK=-5 clamped to 0, got %d", ctrlUnder.GetDepth())
	}
	if ctrlUnder.IsActive() {
		t.Fatalf("expected IsActive=false for depth=0")
	}

	ctrlOver := NewSpeculativeControl(kMax, 100, 0.5)
	if ctrlOver.GetDepth() != kMax {
		t.Fatalf("expected defaultK=100 clamped to %d, got %d", kMax, ctrlOver.GetDepth())
	}

	// Initialization clamping for defaultThreshold
	ctrlLowThresh := NewSpeculativeControl(kMax, 2, -0.5)
	if ctrlLowThresh.GetAcceptanceThreshold() != 0.0 {
		t.Fatalf("expected defaultThreshold=-0.5 clamped to 0.0, got %f", ctrlLowThresh.GetAcceptanceThreshold())
	}

	ctrlHighThresh := NewSpeculativeControl(kMax, 2, 2.5)
	if ctrlHighThresh.GetAcceptanceThreshold() != 1.0 {
		t.Fatalf("expected defaultThreshold=2.5 clamped to 1.0, got %f", ctrlHighThresh.GetAcceptanceThreshold())
	}

	// Clamping via SetDepth
	if err := ctrl.SetDepth(-10); err != nil {
		t.Fatalf("unexpected error on SetDepth(-10): %v", err)
	}
	if ctrl.GetDepth() != 0 {
		t.Fatalf("expected depth clamped to 0, got %d", ctrl.GetDepth())
	}

	if err := ctrl.SetDepth(20); err != nil {
		t.Fatalf("unexpected error on SetDepth(20): %v", err)
	}
	if ctrl.GetDepth() != kMax {
		t.Fatalf("expected depth clamped to %d, got %d", kMax, ctrl.GetDepth())
	}

	if err := ctrl.SetDepth(5); err != nil {
		t.Fatalf("unexpected error on SetDepth(5): %v", err)
	}
	if ctrl.GetDepth() != 5 {
		t.Fatalf("expected depth 5, got %d", ctrl.GetDepth())
	}

	// Clamp method
	if clamped := ctrl.Clamp(-1); clamped != 0 {
		t.Fatalf("Clamp(-1) = %d, want 0", clamped)
	}
	if clamped := ctrl.Clamp(kMax + 10); clamped != kMax {
		t.Fatalf("Clamp(%d) = %d, want %d", kMax+10, clamped, kMax)
	}
	if clamped := ctrl.Clamp(3); clamped != 3 {
		t.Fatalf("Clamp(3) = %d, want 3", clamped)
	}

	// Threshold hot-swap and range validation
	if err := ctrl.SetAcceptanceThreshold(-0.01); !errors.Is(err, ErrInvalidAcceptanceThreshold) {
		t.Fatalf("expected ErrInvalidAcceptanceThreshold for -0.01, got %v", err)
	}
	if err := ctrl.SetAcceptanceThreshold(1.0001); !errors.Is(err, ErrInvalidAcceptanceThreshold) {
		t.Fatalf("expected ErrInvalidAcceptanceThreshold for 1.0001, got %v", err)
	}
	if err := ctrl.SetAcceptanceThreshold(math.NaN()); !errors.Is(err, ErrInvalidAcceptanceThreshold) {
		t.Fatalf("expected ErrInvalidAcceptanceThreshold for NaN, got %v", err)
	}

	if err := ctrl.SetAcceptanceThreshold(0.0); err != nil {
		t.Fatalf("unexpected error for 0.0: %v", err)
	}
	if ctrl.GetAcceptanceThreshold() != 0.0 {
		t.Fatalf("expected threshold=0.0, got %f", ctrl.GetAcceptanceThreshold())
	}

	if err := ctrl.SetAcceptanceThreshold(1.0); err != nil {
		t.Fatalf("unexpected error for 1.0: %v", err)
	}
	if ctrl.GetAcceptanceThreshold() != 1.0 {
		t.Fatalf("expected threshold=1.0, got %f", ctrl.GetAcceptanceThreshold())
	}

	if err := ctrl.SetAcceptanceThreshold(0.85); err != nil {
		t.Fatalf("unexpected error for 0.85: %v", err)
	}
	if math.Abs(ctrl.GetAcceptanceThreshold()-0.85) > 1e-6 {
		t.Fatalf("expected threshold=0.85, got %f", ctrl.GetAcceptanceThreshold())
	}

	// Snapshot
	snap := ctrl.Snapshot()
	if snap.KMax != kMax || snap.Depth != 5 || math.Abs(snap.AcceptanceThreshold-0.85) > 1e-6 || !snap.IsActive {
		t.Fatalf("snapshot mismatch: %+v", snap)
	}

	// Nil receiver safety
	var nilCtrl *SpeculativeControl
	if nilCtrl.GetDepth() != 0 || nilCtrl.KMax() != 0 || nilCtrl.IsActive() || nilCtrl.GetAcceptanceThreshold() != 0.0 {
		t.Fatalf("nilCtrl should return safe zero values")
	}
	if nilCtrl.Clamp(5) != 0 {
		t.Fatalf("nilCtrl.Clamp should return 0")
	}
	if err := nilCtrl.SetDepth(3); err == nil {
		t.Fatalf("nilCtrl.SetDepth should return error")
	}
	if err := nilCtrl.SetAcceptanceThreshold(0.5); err == nil {
		t.Fatalf("nilCtrl.SetAcceptanceThreshold should return error")
	}
	nilSnap := nilCtrl.Snapshot()
	if nilSnap.KMax != 0 || nilSnap.Depth != 0 || nilSnap.IsActive {
		t.Fatalf("nilSnap should be empty zero-value")
	}
}

func TestSpeculativeControl_ConcurrentHotSwap(t *testing.T) {
	kMax := 10
	ctrl := NewSpeculativeControl(kMax, 5, 0.5)

	numWorkers := 8
	iterations := 2000
	var wg sync.WaitGroup
	wg.Add(numWorkers * 3)

	// Depth writers: concurrent updates with out-of-bound and in-bound values
	for w := 0; w < numWorkers; w++ {
		go func(workerID int) {
			defer wg.Done()
			for i := 0; i < iterations; i++ {
				targetK := (i + workerID) % (kMax + 5) - 2 // includes -2, -1, ..., kMax+2
				_ = ctrl.SetDepth(targetK)
			}
		}(w)
	}

	// Threshold writers: concurrent updates with valid thresholds
	thresholds := []float64{0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0}
	for w := 0; w < numWorkers; w++ {
		go func(workerID int) {
			defer wg.Done()
			for i := 0; i < iterations; i++ {
				tVal := thresholds[(i+workerID)%len(thresholds)]
				_ = ctrl.SetAcceptanceThreshold(tVal)
			}
		}(w)
	}

	// Readers: concurrently validating invariants
	for w := 0; w < numWorkers; w++ {
		go func() {
			defer wg.Done()
			for i := 0; i < iterations; i++ {
				d := ctrl.GetDepth()
				if d < 0 || d > kMax {
					t.Errorf("concurrent depth invariant broken: %d not in [0, %d]", d, kMax)
				}

				th := ctrl.GetAcceptanceThreshold()
				if math.IsNaN(th) || th < 0.0 || th > 1.0 {
					t.Errorf("concurrent threshold invariant broken: %f not in [0.0, 1.0]", th)
				}

				_ = ctrl.IsActive()

				snap := ctrl.Snapshot()
				if snap.KMax != kMax || snap.Depth < 0 || snap.Depth > kMax || snap.AcceptanceThreshold < 0.0 || snap.AcceptanceThreshold > 1.0 {
					t.Errorf("invalid snapshot during concurrent execution: %+v", snap)
				}
				if snap.IsActive != (snap.Depth > 0) {
					t.Errorf("snapshot IsActive inconsistent with Depth: %+v", snap)
				}

				slice := ctrl.ActiveDraftSlice()
				if len(slice) > kMax || cap(slice) != kMax {
					t.Errorf("active draft slice broken: len=%d cap=%d", len(slice), cap(slice))
				}
			}
		}()
	}

	wg.Wait()
}

func TestSpeculativeControl_KZeroBypass(t *testing.T) {
	kMax := 8
	ctrl := NewSpeculativeControl(kMax, 0, 0.7)

	if ctrl.IsActive() {
		t.Fatalf("expected IsActive=false when K=0")
	}
	if !ctrl.ShouldBypass() {
		t.Fatalf("expected ShouldBypass=true when K=0")
	}
	if !ctrl.Bypass() {
		t.Fatalf("expected Bypass=true when K=0")
	}

	// Test StepOrBypass with K=0: fallback should run, speculative must NOT run
	speculativeCalled := false
	fallbackCalled := false

	err := ctrl.StepOrBypass(
		func(k int, threshold float64) error {
			speculativeCalled = true
			return nil
		},
		func() error {
			fallbackCalled = true
			return nil
		},
	)
	if err != nil {
		t.Fatalf("unexpected error in StepOrBypass: %v", err)
	}
	if speculativeCalled {
		t.Fatalf("speculative function was called despite K=0 bypass")
	}
	if !fallbackCalled {
		t.Fatalf("fallback target function was not called for K=0 bypass")
	}

	// Test ExecuteWithBypass with K=0
	specTokensCalled := false
	fallbackTokensCalled := false
	res, err := ctrl.ExecuteWithBypass(
		func(k int, threshold float64) ([]int, error) {
			specTokensCalled = true
			return []int{10, 20}, nil
		},
		func() ([]int, error) {
			fallbackTokensCalled = true
			return []int{99}, nil
		},
	)
	if err != nil {
		t.Fatalf("unexpected error in ExecuteWithBypass: %v", err)
	}
	if specTokensCalled {
		t.Fatalf("speculative function called in ExecuteWithBypass despite K=0")
	}
	if !fallbackTokensCalled {
		t.Fatalf("fallback function not called in ExecuteWithBypass")
	}
	if len(res) != 1 || res[0] != 99 {
		t.Fatalf("unexpected fallback tokens: %v", res)
	}

	// Now switch K to 4: speculative path should run, fallback must NOT run
	if err := ctrl.SetDepth(4); err != nil {
		t.Fatalf("failed to set depth: %v", err)
	}
	if !ctrl.IsActive() {
		t.Fatalf("expected IsActive=true when K=4")
	}
	if ctrl.ShouldBypass() {
		t.Fatalf("expected ShouldBypass=false when K=4")
	}

	speculativeCalled = false
	fallbackCalled = false
	var observedK int
	var observedThresh float64

	err = ctrl.StepOrBypass(
		func(k int, threshold float64) error {
			speculativeCalled = true
			observedK = k
			observedThresh = threshold
			return nil
		},
		func() error {
			fallbackCalled = true
			return nil
		},
	)
	if err != nil {
		t.Fatalf("unexpected error in StepOrBypass: %v", err)
	}
	if !speculativeCalled {
		t.Fatalf("speculative function should have been called for K=4")
	}
	if fallbackCalled {
		t.Fatalf("fallback function should NOT have been called for K=4")
	}
	if observedK != 4 {
		t.Fatalf("expected observedK=4, got %d", observedK)
	}
	if math.Abs(observedThresh-0.7) > 1e-6 {
		t.Fatalf("expected observedThresh=0.7, got %f", observedThresh)
	}

	// ExecuteWithBypass when K=4
	specTokensCalled = false
	fallbackTokensCalled = false
	res, err = ctrl.ExecuteWithBypass(
		func(k int, threshold float64) ([]int, error) {
			specTokensCalled = true
			return []int{10, 20, 30, 40}, nil
		},
		func() ([]int, error) {
			fallbackTokensCalled = true
			return []int{99}, nil
		},
	)
	if err != nil {
		t.Fatalf("unexpected error in ExecuteWithBypass: %v", err)
	}
	if !specTokensCalled {
		t.Fatalf("speculative function not called when K=4")
	}
	if fallbackTokensCalled {
		t.Fatalf("fallback function called when K=4")
	}
	if len(res) != 4 || res[0] != 10 || res[3] != 40 {
		t.Fatalf("unexpected speculative tokens: %v", res)
	}
}

func TestSpeculativeControl_ZeroAllocation(t *testing.T) {
	kMax := 8
	ctrl := NewSpeculativeControl(kMax, 4, 0.75)

	// Test GetDepth zero-allocation
	allocs := testing.AllocsPerRun(1000, func() {
		_ = ctrl.GetDepth()
	})
	if allocs > 0 {
		t.Fatalf("GetDepth allocated %f times, expected 0", allocs)
	}

	// Test Clamp zero-allocation
	allocs = testing.AllocsPerRun(1000, func() {
		_ = ctrl.Clamp(5)
	})
	if allocs > 0 {
		t.Fatalf("Clamp allocated %f times, expected 0", allocs)
	}

	// Test GetAcceptanceThreshold zero-allocation
	allocs = testing.AllocsPerRun(1000, func() {
		_ = ctrl.GetAcceptanceThreshold()
	})
	if allocs > 0 {
		t.Fatalf("GetAcceptanceThreshold allocated %f times, expected 0", allocs)
	}

	// Test IsActive zero-allocation
	allocs = testing.AllocsPerRun(1000, func() {
		_ = ctrl.IsActive()
	})
	if allocs > 0 {
		t.Fatalf("IsActive allocated %f times, expected 0", allocs)
	}

	// Test KMax zero-allocation
	allocs = testing.AllocsPerRun(1000, func() {
		_ = ctrl.KMax()
	})
	if allocs > 0 {
		t.Fatalf("KMax allocated %f times, expected 0", allocs)
	}

	// Test ActiveDraftSlice zero-allocation
	allocs = testing.AllocsPerRun(1000, func() {
		_ = ctrl.ActiveDraftSlice()
	})
	if allocs > 0 {
		t.Fatalf("ActiveDraftSlice allocated %f times, expected 0", allocs)
	}

	// Test ActiveDraftSlice32 zero-allocation
	allocs = testing.AllocsPerRun(1000, func() {
		_ = ctrl.ActiveDraftSlice32()
	})
	if allocs > 0 {
		t.Fatalf("ActiveDraftSlice32 allocated %f times, expected 0", allocs)
	}
}

func TestSpeculativeControl_PreallocatedBuffers(t *testing.T) {
	kMax := 8
	ctrl := NewSpeculativeControl(kMax, 3, 0.5)

	if len(ctrl.DraftTokensBuffer()) != kMax || cap(ctrl.DraftTokensBuffer()) != kMax {
		t.Fatalf("DraftTokensBuffer dimension mismatch")
	}
	if len(ctrl.AcceptedTokensBuffer()) != kMax+1 || cap(ctrl.AcceptedTokensBuffer()) != kMax+1 {
		t.Fatalf("AcceptedTokensBuffer dimension mismatch")
	}
	if len(ctrl.DraftTokensIntBuffer()) != kMax || cap(ctrl.DraftTokensIntBuffer()) != kMax {
		t.Fatalf("DraftTokensIntBuffer dimension mismatch")
	}
	if len(ctrl.AcceptedTokensIntBuffer()) != kMax+1 || cap(ctrl.AcceptedTokensIntBuffer()) != kMax+1 {
		t.Fatalf("AcceptedTokensIntBuffer dimension mismatch")
	}
	if len(ctrl.AcceptanceMaskBuffer()) != kMax || cap(ctrl.AcceptanceMaskBuffer()) != kMax {
		t.Fatalf("AcceptanceMaskBuffer dimension mismatch")
	}

	slice := ctrl.ActiveDraftSlice()
	if len(slice) != 3 || cap(slice) != kMax {
		t.Fatalf("ActiveDraftSlice length=%d cap=%d, want len=3 cap=%d", len(slice), cap(slice), kMax)
	}

	slice32 := ctrl.ActiveDraftSlice32()
	if len(slice32) != 3 || cap(slice32) != kMax {
		t.Fatalf("ActiveDraftSlice32 length=%d cap=%d, want len=3 cap=%d", len(slice32), cap(slice32), kMax)
	}

	// Update depth to 7
	_ = ctrl.SetDepth(7)
	slice = ctrl.ActiveDraftSlice()
	if len(slice) != 7 || cap(slice) != kMax {
		t.Fatalf("ActiveDraftSlice length=%d cap=%d, want len=7 cap=%d", len(slice), cap(slice), kMax)
	}

	// Slices point to same underlying array (no reallocation)
	slice[0] = 42
	if ctrl.DraftTokensIntBuffer()[0] != 42 {
		t.Fatalf("ActiveDraftSlice should share backing array with DraftTokensIntBuffer")
	}
}
