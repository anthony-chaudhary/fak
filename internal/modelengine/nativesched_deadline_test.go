package modelengine

import (
	"context"
	"fmt"
	"slices"
	"testing"
	"time"
)

const deadlineWitnessLanes = 10_000

func TestWaitingDeadlineHeapExpiresStaggeredLanesAndRestoresAfterCancellation(t *testing.T) {
	base := time.Date(2026, time.August, 29, 12, 0, 0, 0, time.UTC)
	now := base
	h := newWaitingDeadlineHeap(func() time.Time { return now })
	lanes := make([]*schedLane, deadlineWitnessLanes)
	wantByTick := make([][]*schedLane, 257)

	for i := range lanes {
		lane := &schedLane{seqNo: int64(i)}
		lanes[i] = lane
		tick := 1 + (i*73)%256
		h.schedule(lane, base.Add(time.Duration(tick)*time.Millisecond))
		if i%7 != 0 {
			wantByTick[tick] = append(wantByTick[tick], lane)
		}
	}
	for i, lane := range lanes {
		if i%7 == 0 && !h.cancel(lane) {
			t.Fatalf("cancel lane %d: missing from heap", i)
		}
	}

	var got []*schedLane
	for tick := 1; tick <= 256; tick++ {
		now = base.Add(time.Duration(tick) * time.Millisecond)
		start := len(got)
		got = h.expireReady(got)
		want := wantByTick[tick]
		if !slices.Equal(got[start:], want) {
			t.Fatalf("tick %d expiry mismatch: got=%s want=%s", tick, laneSeqs(got[start:]), laneSeqs(want))
		}
	}
	if h.len() != 0 {
		t.Fatalf("heap retained %d lanes after final expiry", h.len())
	}
	if len(got) != deadlineWitnessLanes-deadlineWitnessLanes/7-1 {
		t.Fatalf("expired %d lanes, want %d", len(got), deadlineWitnessLanes-deadlineWitnessLanes/7-1)
	}
}

func TestWaitingDeadlineHeapRescheduleAndNextDeadline(t *testing.T) {
	base := time.Unix(1_800_000_000, 0)
	now := base
	h := newWaitingDeadlineHeap(func() time.Time { return now })
	first := &schedLane{seqNo: 1}
	second := &schedLane{seqNo: 2}
	h.schedule(first, base.Add(2*time.Second))
	h.schedule(second, base.Add(time.Second))
	h.schedule(first, base.Add(500*time.Millisecond))

	got, ok := h.nextDeadline()
	if !ok || !got.Equal(base.Add(500*time.Millisecond)) {
		t.Fatalf("next deadline = %v, %v", got, ok)
	}
	now = base.Add(500 * time.Millisecond)
	if expired := h.expireReady(nil); !slices.Equal(expired, []*schedLane{first}) {
		t.Fatalf("rescheduled expiry = %s, want [1]", laneSeqs(expired))
	}
	if h.cancel(first) {
		t.Fatal("expired lane remained cancelable")
	}
}

func BenchmarkNativeWaitingDeadlineWake(b *testing.B) {
	for _, lanes := range []int{1_000, deadlineWitnessLanes} {
		b.Run(fmt.Sprintf("context_scan/%d", lanes), func(b *testing.B) {
			benchmarkDeadlineScan(b, lanes)
		})
		b.Run(fmt.Sprintf("shared_heap/%d", lanes), func(b *testing.B) {
			benchmarkDeadlineHeap(b, lanes)
		})
	}
}

func benchmarkDeadlineHeap(b *testing.B, laneCount int) {
	base := time.Unix(1_800_000_000, 0)
	lanes := make([]schedLane, laneCount)
	b.ReportAllocs()
	b.SetBytes(int64(laneCount))
	b.ResetTimer()
	for range b.N {
		now := base
		h := newWaitingDeadlineHeap(func() time.Time { return now })
		for i := range lanes {
			h.schedule(&lanes[i], base.Add(time.Duration(1+(i*73)%256)*time.Millisecond))
		}
		for i := 0; i < laneCount; i += 7 {
			h.cancel(&lanes[i])
		}
		expired := 0
		for tick := 1; tick <= 256; tick++ {
			now = base.Add(time.Duration(tick) * time.Millisecond)
			expired += len(h.expireReady(nil))
		}
		if expired == 0 {
			b.Fatal("shared heap expired no lanes")
		}
	}
}

func benchmarkDeadlineScan(b *testing.B, laneCount int) {
	base := time.Unix(1_800_000_000, 0)
	deadlines := make([]time.Time, laneCount)
	cancelled := make([]bool, laneCount)
	b.ReportAllocs()
	b.SetBytes(int64(laneCount))
	b.ResetTimer()
	for range b.N {
		for i := range deadlines {
			deadlines[i] = base.Add(time.Duration(1+(i*73)%256) * time.Millisecond)
			cancelled[i] = i%7 == 0
		}
		expired := 0
		for tick := 1; tick <= 256; tick++ {
			now := base.Add(time.Duration(tick) * time.Millisecond)
			for i := range deadlines {
				if !cancelled[i] && !deadlines[i].After(now) {
					cancelled[i] = true
					expired++
				}
			}
		}
		if expired == 0 {
			b.Fatal("deadline scan expired no lanes")
		}
	}
}

func laneSeqs(lanes []*schedLane) string {
	seqs := make([]int64, len(lanes))
	for i, lane := range lanes {
		seqs[i] = lane.seqNo
	}
	return fmt.Sprint(seqs)
}

func TestNativeSchedulerPromotesEarliestDeadlineFirst(t *testing.T) {
	base := time.Unix(1_800_000_000, 0)
	lane := func(seq int64, deadline time.Time) *schedLane {
		return &schedLane{ctx: context.Background(), state: schedLaneDecode, seqNo: seq, deadline: deadline}
	}
	promoteOrder := func(waiting []*schedLane) []int64 {
		s := newNativeScheduler(nil, nil)
		s.mu.Lock()
		defer s.mu.Unlock()
		s.waiting = waiting
		var got []int64
		for len(s.waiting) > 0 {
			s.orderWaitingByDeadlineLocked()
			s.promoteWaitingLocked(1)
			if len(s.lanes) != 1 {
				t.Fatalf("promoted %d lanes with maxRun=1", len(s.lanes))
			}
			got = append(got, s.lanes[0].seqNo)
			s.lanes = s.lanes[:0]
		}
		return got
	}

	// Deadlines t+30, t+10, none, t+20 promote as 10, 20, 30, none.
	got := promoteOrder([]*schedLane{
		lane(1, base.Add(30*time.Second)),
		lane(2, base.Add(10*time.Second)),
		lane(3, time.Time{}),
		lane(4, base.Add(20*time.Second)),
	})
	if want := []int64{2, 4, 1, 3}; !slices.Equal(got, want) {
		t.Fatalf("EDF promotion order = %v, want %v", got, want)
	}

	// Equal deadlines stay FIFO, and no-deadline lanes keep FIFO among themselves.
	got = promoteOrder([]*schedLane{
		lane(1, time.Time{}),
		lane(2, base.Add(5*time.Second)),
		lane(3, time.Time{}),
		lane(4, base.Add(5*time.Second)),
	})
	if want := []int64{2, 4, 1, 3}; !slices.Equal(got, want) {
		t.Fatalf("tie promotion order = %v, want %v", got, want)
	}

	// No deadlines anywhere: default FIFO promotion is unchanged.
	got = promoteOrder([]*schedLane{lane(3, time.Time{}), lane(1, time.Time{}), lane(2, time.Time{})})
	if want := []int64{3, 1, 2}; !slices.Equal(got, want) {
		t.Fatalf("no-deadline promotion order = %v, want FIFO %v", got, want)
	}
}

func TestNativeLaneDeadlineFromMeta(t *testing.T) {
	for _, tc := range []struct {
		meta map[string]string
		want time.Time
	}{
		{nil, time.Time{}},
		{map[string]string{"deadline_unix_ms": "1800000000123"}, time.UnixMilli(1_800_000_000_123)},
		{map[string]string{"sched.deadline_unix_ms": " 42 "}, time.UnixMilli(42)},
		{map[string]string{"deadline_unix_ms": "soon"}, time.Time{}},
		{map[string]string{"deadline_unix_ms": "-5"}, time.Time{}},
	} {
		if got := nativeLaneDeadlineFromMeta(tc.meta); !got.Equal(tc.want) {
			t.Fatalf("nativeLaneDeadlineFromMeta(%v) = %v, want %v", tc.meta, got, tc.want)
		}
	}
}
