package modelengine

// fak-test:runtime fast est=1s

import (
	"bytes"
	"errors"
	"testing"
)

func TestGraphCaptureBoundSizesAllocationAndCopiesFromStaticBound(t *testing.T) {
	hostReads := 0
	plan, err := PlanGraphCaptureDispatch(GraphCaptureDispatchRequest{
		Capturing:        true,
		MaxTokensPerRank: 64,
		RowBytes:         8,
		TokenBound:       16,
		// Even if a host read would return 0 (the pre-fix zero-row capture bug),
		// capture must never consult it and must size from TokenBound.
		HostTokenCountReader: func() int {
			hostReads++
			return 0
		},
	})
	if err != nil {
		t.Fatalf("PlanGraphCaptureDispatch: %v", err)
	}
	if hostReads != 0 || plan.CaptureHostReads != 0 {
		t.Fatalf("capture performed host reads: callback=%d plan=%d, want 0", hostReads, plan.CaptureHostReads)
	}
	if !plan.Captured || plan.StaticRowBound != 16 || plan.AllocatedRows != 16 || plan.CopyRows != 16 {
		t.Fatalf("unexpected static plan sizing: %+v", plan)
	}
	if plan.AllocatedBytes != 128 || plan.CopyBytes != 128 {
		t.Fatalf("bytes = alloc:%d copy:%d, want 128/128", plan.AllocatedBytes, plan.CopyBytes)
	}

	// Replay 3 live tokens, then 16 live tokens, using the same captured static buffer.
	src := make([][]byte, 16)
	for i := range src {
		src[i] = bytes.Repeat([]byte{byte(i + 1)}, 8)
	}
	got3, err := plan.Replay(3, src)
	if err != nil {
		t.Fatalf("Replay(3): %v", err)
	}
	if len(got3) != 3 {
		t.Fatalf("Replay(3) len = %d, want 3", len(got3))
	}
	for i := 0; i < 3; i++ {
		if !bytes.Equal(got3[i], src[i]) {
			t.Fatalf("Replay(3) row %d mismatch", i)
		}
	}
	// Verify unused tail slots in the static capture buffer were cleared.
	for i := 3; i < 16; i++ {
		for _, b := range plan.stagedCopyBuffer[i] {
			if b != 0 {
				t.Fatalf("Replay(3) left non-zero tail byte in captured row %d", i)
			}
		}
	}

	got16, err := plan.Replay(16, src)
	if err != nil {
		t.Fatalf("Replay(16): %v", err)
	}
	if len(got16) != 16 {
		t.Fatalf("Replay(16) len = %d, want 16", len(got16))
	}
	if _, err := plan.Replay(17, make([][]byte, 17)); !errors.Is(err, ErrGraphCaptureBoundExceedsCapacity) {
		t.Fatalf("Replay(17) err = %v, want ErrGraphCaptureBoundExceedsCapacity", err)
	}
}

func TestGraphCaptureBoundRefusesMissingCount(t *testing.T) {
	for _, tc := range []struct {
		name  string
		bound int
	}{
		{name: "unset_zero", bound: 0},
		{name: "negative_sentinel", bound: -1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			hostReads := 0
			_, err := PlanGraphCaptureDispatch(GraphCaptureDispatchRequest{
				Capturing:        true,
				MaxTokensPerRank: 32,
				RowBytes:         8,
				TokenBound:       tc.bound,
				HostTokenCountReader: func() int {
					hostReads++
					return 4
				},
			})
			if !errors.Is(err, ErrGraphCaptureMissingTokenBound) {
				t.Fatalf("err = %v, want ErrGraphCaptureMissingTokenBound", err)
			}
			if hostReads != 0 {
				t.Fatalf("refused capture invoked host read %d times, want 0", hostReads)
			}
		})
	}
}

func TestGraphCaptureBoundRefusesOverCapacity(t *testing.T) {
	_, err := PlanGraphCaptureDispatch(GraphCaptureDispatchRequest{
		Capturing:        true,
		MaxTokensPerRank: 16,
		RowBytes:         8,
		TokenBound:       17,
	})
	if !errors.Is(err, ErrGraphCaptureBoundExceedsCapacity) {
		t.Fatalf("err = %v, want ErrGraphCaptureBoundExceedsCapacity", err)
	}
}

func TestGraphCaptureBoundEagerModeAllowsHostRead(t *testing.T) {
	hostReads := 0
	plan, err := PlanGraphCaptureDispatch(GraphCaptureDispatchRequest{
		Capturing:        false,
		MaxTokensPerRank: 32,
		RowBytes:         4,
		TokenBound:       0,
		HostTokenCountReader: func() int {
			hostReads++
			return 5
		},
	})
	if err != nil {
		t.Fatalf("PlanGraphCaptureDispatch eager: %v", err)
	}
	if hostReads != 1 || plan.CaptureHostReads != 1 || plan.AllocatedRows != 5 {
		t.Fatalf("eager plan = %+v hostReads=%d, want 5 rows and 1 host read", plan, hostReads)
	}
}
