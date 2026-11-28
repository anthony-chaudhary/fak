package kvbudget

import (
	"errors"
	"testing"
)

// fak-test:runtime fast est=5ms lane=default
func TestLoadPreflightShrinkRetry(t *testing.T) {
	// (a) Predicted over 80% ceiling is refused before allocation.
	spec := LoadPreflightSpec{
		FreeBytes:        1000,
		WeightsBytes:     500,
		Ctx:              512,
		MinCtx:           128,
		NumParallel:      1,
		BytesPerToken:    1,
		BatchSize:        64,
		MinBatchSize:     16,
		BytesPerBatchTok: 1,
	}
	// Predicted = 500 + 512 + 64 = 1076 > 800.
	if err := CheckLoadPreflight(spec); !errors.Is(err, ErrPreflightOverCeiling) {
		t.Fatalf("expected ErrPreflightOverCeiling, got %v", err)
	}

	// (b) Shrink ladder reduces ctx, then batch, then evicts, then errors.
	step1, err := NextLoadShrinkStep(spec, false)
	if err != nil || step1.Action != ShrinkCtx || step1.Spec.Ctx != 256 {
		t.Fatalf("step1 = %+v err=%v", step1, err)
	}
	step2, err := NextLoadShrinkStep(step1.Spec, false)
	if err != nil || step2.Action != ShrinkCtx || step2.Spec.Ctx != 128 {
		t.Fatalf("step2 = %+v err=%v", step2, err)
	}
	// Now predicted = 500 + 128 + 64 = 692 <= 800, fits!
	if err := CheckLoadPreflight(step2.Spec); err != nil {
		t.Fatalf("expected shrunk spec to fit, got %v", err)
	}
	// Continue ladder down to batch -> evict -> error.
	step3, err := NextLoadShrinkStep(step2.Spec, false)
	if err != nil || step3.Action != ShrinkBatch || step3.Spec.BatchSize != 32 {
		t.Fatalf("step3 = %+v err=%v", step3, err)
	}
	step4, err := NextLoadShrinkStep(step3.Spec, false)
	if err != nil || step4.Action != ShrinkBatch || step4.Spec.BatchSize != 16 {
		t.Fatalf("step4 = %+v err=%v", step4, err)
	}
	step5, err := NextLoadShrinkStep(step4.Spec, true)
	if err != nil || step5.Action != ShrinkEvictCached {
		t.Fatalf("step5 = %+v err=%v", step5, err)
	}
	if _, err := NextLoadShrinkStep(step4.Spec, false); !errors.Is(err, ErrShrinkExhausted) {
		t.Fatalf("expected ErrShrinkExhausted, got %v", err)
	}

	// (c) Recovery wait rejects stale free bytes (< 75% of expected usage returned).
	if _, err := WaitForVRAMRecovery(100, 1000, []uint64{200, 500, 849}); !errors.Is(err, ErrVRAMRecoveryStale) {
		t.Fatalf("expected ErrVRAMRecoveryStale, got %v", err)
	}
	if got, err := WaitForVRAMRecovery(100, 1000, []uint64{200, 850}); err != nil || got != 850 {
		t.Fatalf("expected recovery at 850, got %d err=%v", got, err)
	}
}
