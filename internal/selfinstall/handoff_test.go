package selfinstall

import (
	"context"
	"errors"
	"testing"
	"time"
)

func TestHandoffDrainsActiveCallAndRefusesNewAdmissions(t *testing.T) {
	var h Handoff
	release, err := h.Admit()
	if err != nil {
		t.Fatal(err)
	}
	done := make(chan HandoffSnapshot, 1)
	go func() {
		done <- h.Drain(context.Background(), "session-7", "rev-new", func(_ context.Context, session, revision string) error {
			if session != "session-7" || revision != "rev-new" {
				t.Fatalf("successor identity = %q/%q", session, revision)
			}
			return nil
		})
	}()
	for h.Snapshot().State != HandoffDraining {
		time.Sleep(time.Millisecond)
	}
	if _, err := h.Admit(); !errors.Is(err, ErrHandoffDraining) {
		t.Fatalf("admission error = %v", err)
	}
	select {
	case got := <-done:
		t.Fatalf("handoff completed before active call: %+v", got)
	default:
	}
	release()
	got := <-done
	if got.State != HandoffHandedOff || got.SessionID != "session-7" || got.Revision != "rev-new" {
		t.Fatalf("snapshot = %+v", got)
	}
}

func TestHandoffTimeoutRefusesWithoutLaunching(t *testing.T) {
	var h Handoff
	release, _ := h.Admit()
	defer release()
	ctx, cancel := context.WithTimeout(context.Background(), time.Millisecond)
	defer cancel()
	got := h.Drain(ctx, "s", "r", func(context.Context, string, string) error { t.Fatal("launched after timeout"); return nil })
	if got.State != HandoffRefused || !errors.Is(got.Err, context.DeadlineExceeded) {
		t.Fatalf("snapshot = %+v", got)
	}
}

func TestHandoffCanceledSuccessorRefuses(t *testing.T) {
	var h Handoff
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	got := h.Drain(ctx, "s", "r", func(context.Context, string, string) error { return nil })
	if got.State != HandoffRefused || !errors.Is(got.Err, context.Canceled) {
		t.Fatalf("snapshot = %+v", got)
	}
}
