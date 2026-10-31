package gateway

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"
)

func awaitNativeCollector(t *testing.T, ch <-chan struct{}) {
	t.Helper()
	select {
	case <-ch:
	case <-time.After(3 * time.Second):
		t.Fatal("native collector did not reach the expected ownership boundary")
	}
}

func TestCoalescingSenderNativeBarriers(t *testing.T) {
	entered, release := make(chan struct{}), make(chan struct{})
	var releaseOnce sync.Once
	var delivered []DeltaChunk
	s := NewCoalescingSender(func(c DeltaChunk) error {
		if len(delivered) == 0 {
			close(entered)
			<-release
		}
		delivered = append(delivered, c)
		return nil
	})
	t.Cleanup(func() { releaseOnce.Do(func() { close(release) }); _ = s.Close() })
	if err := s.Send("first", 1); err != nil {
		t.Fatal(err)
	}
	awaitNativeCollector(t, entered)
	if err := s.barrier("progress-1"); err != nil {
		t.Fatal(err)
	}
	if err := s.Send("second-", 2); err != nil {
		t.Fatal(err)
	}
	if err := s.Send("third", 3); err != nil {
		t.Fatal(err)
	}
	if err := s.barrier("progress-2"); err != nil {
		t.Fatal(err)
	}
	releaseOnce.Do(func() { close(release) })
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	if len(delivered) != 4 {
		t.Fatalf("delivered %d chunks, want 4: %+v", len(delivered), delivered)
	}
	want := []string{"first", "progress-1", "second-third", "progress-2"}
	for i, c := range delivered {
		if c.Text != want[i] || c.barrier != (i == 1 || i == 3) {
			t.Fatalf("chunk %d = %+v, want %q barrier=%v", i, c, want[i], i == 1 || i == 3)
		}
	}
	if delivered[2].Tokens != 5 {
		t.Fatalf("merged token count = %d", delivered[2].Tokens)
	}
}

func TestCoalescingSenderNativeBarrierBudgets(t *testing.T) {
	for _, budget := range []string{"bytes", "events"} {
		t.Run(budget, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			entered := make(chan struct{})
			s := newCoalescingSender(ctx, false, func(DeltaChunk) error {
				close(entered)
				<-ctx.Done()
				return ctx.Err()
			}, cancel)
			t.Cleanup(func() { cancel(); _ = s.Close() })
			if err := s.Send("x", 1); err != nil {
				t.Fatal(err)
			}
			awaitNativeCollector(t, entered)
			if budget == "bytes" {
				if err := s.barrier(strings.Repeat("b", streamPendingBytes-1)); err != nil {
					t.Fatal(err)
				}
			} else {
				for i := 1; i < streamPendingEvents; i++ {
					if err := s.barrier("b"); err != nil {
						t.Fatalf("event %d: %v", i, err)
					}
				}
			}
			if err := s.barrier("overflow"); !errors.Is(err, errStreamBackpressure) {
				t.Fatalf("overflow error = %v", err)
			}
			if err := s.Close(); !errors.Is(err, errStreamBackpressure) {
				t.Fatalf("Close lost first budget failure: %v", err)
			}
		})
	}
}

func TestCoalescingSenderCanceledNativeBarrier(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	s := newCoalescingSender(ctx, false, func(DeltaChunk) error {
		t.Error("canceled barrier reached delivery")
		return nil
	}, nil)
	if err := s.barrier("not delivered"); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled enqueue = %v", err)
	}
	if err := s.Close(); !errors.Is(err, context.Canceled) {
		t.Fatalf("Close lost cancellation = %v", err)
	}
}
