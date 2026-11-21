package model

// qwen35_cohort_unlock_test.go — witness that qwen35CohortMutation releases every
// HAL state lock it acquired when the forward callback exits abnormally. A panic
// in fn used to skip the post-closure unlock, stranding q.mu so the scheduler's
// cleanup path (Session.Close re-enters q.mu) deadlocked.

import (
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

func TestQwen35CohortMutationReleasesLocksOnPanic(t *testing.T) {
	sessions := []*Session{
		{qwen35HAL: &qwen35HALState{}},
		{qwen35HAL: &qwen35HALState{}},
	}
	lanes := make([]compute.Qwen35SequencePrefillRequest, len(sessions))
	func() {
		defer func() {
			if recover() == nil {
				t.Fatal("forward panic did not propagate")
			}
		}()
		_ = qwen35CohortMutation(nil, sessions, lanes, func() error { panic("forward exploded") })
	}()
	for i, s := range sessions {
		if !s.qwen35HAL.mu.TryLock() {
			t.Fatalf("session %d: HAL state lock stranded after a panicking forward", i)
		}
		s.qwen35HAL.mu.Unlock()
	}

	// Error exit: the callback's error is returned and the locks are released.
	want := errors.New("forward failed")
	if err := qwen35CohortMutation(nil, sessions, lanes, func() error { return want }); !errors.Is(err, want) {
		t.Fatalf("err = %v, want %v", err, want)
	}
	for i, s := range sessions {
		if !s.qwen35HAL.mu.TryLock() {
			t.Fatalf("session %d: HAL state lock stranded after an erroring forward", i)
		}
		s.qwen35HAL.mu.Unlock()
	}
}
