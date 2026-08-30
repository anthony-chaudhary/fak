package modelengine

import (
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
