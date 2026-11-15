package ratelimit

import (
	"testing"
	"time"
)

// TestWindowCaps exercises the host-side sliding-window + token-bucket cap
// model (window.go) with an injected clock: admission within the caps,
// cap-refusal when a window is full or a bucket is empty, bucket refill over
// time, and fail-closed refusal on an invalid cap.
func TestWindowCaps(t *testing.T) {
	t.Run("SlidingWindowAdmitsWithinCap", func(t *testing.T) {
		var now time.Time
		s := NewStoreWithClock(func() time.Time { return now })
		caps := RouteCaps{Window: WindowCap{Window: 10 * time.Second, Limit: 3}}

		for i := 0; i < 3; i++ {
			if a := s.AdmitAt("route-a", caps, 0, now); !a.Admit {
				t.Fatalf("call %d refused inside cap: %+v", i+1, a.Refusal)
			}
			now = now.Add(time.Second)
		}
		if events, _ := s.State("route-a", caps, now); events != 3 {
			t.Fatalf("window events = %d, want 3", events)
		}
	})

	t.Run("SlidingWindowRefusesWhenFull", func(t *testing.T) {
		var now time.Time
		s := NewStoreWithClock(func() time.Time { return now })
		caps := RouteCaps{Window: WindowCap{Window: 10 * time.Second, Limit: 2}}

		s.AdmitAt("route-b", caps, 0, now)
		now = now.Add(time.Second)
		s.AdmitAt("route-b", caps, 0, now)

		a := s.AdmitAt("route-b", caps, 0, now)
		if a.Admit {
			t.Fatalf("third call inside the window admitted, want refusal")
		}
		if a.Refusal.Cap != RefusalWindowFull {
			t.Fatalf("refusal cap = %q, want %q", a.Refusal.Cap, RefusalWindowFull)
		}
		if a.Refusal.Limit != 2 || a.Refusal.Used != 2 || a.Refusal.Requested != 1 {
			t.Fatalf("refusal = %+v, want limit=2 used=2 requested=1", a.Refusal)
		}
		// The oldest event (t=0) rolls out of the window at t=10; from now=t=1
		// that is 9s away.
		if a.Refusal.RetryAfter != 9*time.Second {
			t.Fatalf("retry_after = %v, want 9s", a.Refusal.RetryAfter)
		}
		// A refused call records no event: the window still holds exactly 2.
		if events, _ := s.State("route-b", caps, now); events != 2 {
			t.Fatalf("window events after refusal = %d, want 2", events)
		}
	})

	t.Run("SlidingWindowReleasesAsEventsAge", func(t *testing.T) {
		var now time.Time
		s := NewStoreWithClock(func() time.Time { return now })
		caps := RouteCaps{Window: WindowCap{Window: 10 * time.Second, Limit: 2}}

		s.AdmitAt("route-c", caps, 0, now)
		now = now.Add(time.Second)
		s.AdmitAt("route-c", caps, 0, now)

		// Just before the first event ages out (t=0 rolls out at t=10): still full.
		now = now.Add(8 * time.Second)
		if a := s.AdmitAt("route-c", caps, 0, now); a.Admit {
			t.Fatalf("call admitted while both events are still in the window")
		}
		// The first event (t=0) is now outside (now-10s, now]: one slot frees.
		now = now.Add(time.Second)
		if a := s.AdmitAt("route-c", caps, 0, now); !a.Admit {
			t.Fatalf("call refused after the oldest event aged out: %+v", a.Refusal)
		}
		if events, _ := s.State("route-c", caps, now); events != 2 {
			t.Fatalf("window events = %d, want 2 (one aged out, one admitted)", events)
		}
	})

	t.Run("TokenBucketRefillsOverTime", func(t *testing.T) {
		var now time.Time
		s := NewStoreWithClock(func() time.Time { return now })
		caps := RouteCaps{Bucket: BucketCap{Capacity: 100, Rate: 10}}

		// Drain the bucket: 100 tokens in, then the next 100 is refused.
		if a := s.AdmitAt("route-d", caps, 100, now); !a.Admit {
			t.Fatalf("full-cost call refused on a fresh bucket: %+v", a.Refusal)
		}
		a := s.AdmitAt("route-d", caps, 100, now)
		if a.Admit || a.Refusal.Cap != RefusalBucketEmpty {
			t.Fatalf("empty-bucket call = %+v, want bucket_empty refusal", a)
		}
		if a.Refusal.Limit != 100 || a.Refusal.Used != 100 || a.Refusal.Requested != 100 {
			t.Fatalf("refusal = %+v, want limit=100 used=100 requested=100", a.Refusal)
		}
		// 100 tokens short at 10/s: the advisory back-off is 10s.
		if a.Refusal.RetryAfter != 10*time.Second {
			t.Fatalf("retry_after = %v, want 10s", a.Refusal.RetryAfter)
		}

		// Refill: 30s later the bucket is back at capacity.
		now = now.Add(30 * time.Second)
		if _, tokens := s.State("route-d", caps, now); tokens != 100 {
			t.Fatalf("bucket tokens after 30s = %d, want 100 (refilled to capacity)", tokens)
		}
		if a := s.AdmitAt("route-d", caps, 100, now); !a.Admit {
			t.Fatalf("call refused after full refill: %+v", a.Refusal)
		}
		if _, tokens := s.State("route-d", caps, now); tokens != 0 {
			t.Fatalf("bucket tokens after re-drain = %d, want 0", tokens)
		}

		// Partial refill: 5s of 10/s tops up 50 tokens — enough for 50, not 100.
		now = now.Add(5 * time.Second)
		if a := s.AdmitAt("route-d", caps, 50, now); !a.Admit {
			t.Fatalf("50-token call refused after partial refill: %+v", a.Refusal)
		}
		if a := s.AdmitAt("route-d", caps, 1, now); a.Admit {
			t.Fatalf("call admitted on an empty bucket after the partial refill")
		}
	})

	t.Run("InvalidCapFailsClosed", func(t *testing.T) {
		var now time.Time
		s := NewStoreWithClock(func() time.Time { return now })

		// A window cap with no duration is malformed: refuse, do not admit.
		badWindow := RouteCaps{Window: WindowCap{Limit: 5}}
		if a := s.AdmitAt("route-e", badWindow, 0, now); a.Admit {
			t.Fatalf("window cap without a duration admitted, want fail-closed refusal")
		}
		if a := s.AdmitAt("route-e", badWindow, 0, now); a.Refusal.Cap != RefusalInvalidCap {
			t.Fatalf("refusal cap = %q, want %q", a.Refusal.Cap, RefusalInvalidCap)
		}
		// A bucket cap with no refill rate is malformed too.
		badBucket := RouteCaps{Bucket: BucketCap{Capacity: 10}}
		if a := s.AdmitAt("route-e", badBucket, 5, now); a.Admit {
			t.Fatalf("bucket cap without a rate admitted, want fail-closed refusal")
		}
		// A negative request cost is malformed.
		if a := s.AdmitAt("route-e", RouteCaps{}, -1, now); a.Admit {
			t.Fatalf("negative cost admitted, want fail-closed refusal")
		}
		// A refused call must not have created route state.
		if events, tokens := s.State("route-e", RouteCaps{}, now); events != 0 || tokens != 0 {
			t.Fatalf("route state after invalid-cap refusals = events %d tokens %d, want 0/0", events, tokens)
		}
	})

	t.Run("WindowAndBucketCompose", func(t *testing.T) {
		var now time.Time
		s := NewStoreWithClock(func() time.Time { return now })
		caps := RouteCaps{
			Window: WindowCap{Window: 10 * time.Second, Limit: 2},
			Bucket: BucketCap{Capacity: 100, Rate: 10},
		}

		// The bucket fires first: 60+60 > 100, while the window still has room.
		if a := s.AdmitAt("route-f", caps, 60, now); !a.Admit {
			t.Fatalf("first 60-token call refused: %+v", a.Refusal)
		}
		a := s.AdmitAt("route-f", caps, 60, now)
		if a.Admit || a.Refusal.Cap != RefusalBucketEmpty {
			t.Fatalf("second 60-token call = %+v, want bucket_empty (window still has room)", a)
		}
		// The window fires next: the admitted 60-token call already holds one
		// slot, so one 1-token call fills the window and the next is refused.
		if a := s.AdmitAt("route-f", caps, 1, now); !a.Admit {
			t.Fatalf("1-token call refused while the window has room: %+v", a.Refusal)
		}
		a = s.AdmitAt("route-f", caps, 1, now)
		if a.Admit || a.Refusal.Cap != RefusalWindowFull {
			t.Fatalf("second 1-token call = %+v, want window_full", a)
		}
		if events, tokens := s.State("route-f", caps, now); events != 2 || tokens != 39 {
			t.Fatalf("state = events %d tokens %d, want events 2 tokens 39", events, tokens)
		}
	})
}
