package modelengine

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/model"
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

// fak-test:runtime medium est=1s lane=default
// TestNativeSchedulerPromotesEarliestDeadlineFirst enters through the registered
// Engine lifecycle, holds execution until admission completes, then observes real
// decode order and terminal ownership. Priority metadata is not an EDF class.
func TestNativeSchedulerPromotesEarliestDeadlineFirst(t *testing.T) {
	type request struct {
		name     string
		deadline int
		priority int
		cancel   bool
	}
	for _, tc := range []struct {
		name         string
		requests     []request
		want         []string
		picker       bool
		preemptOne   bool
		capacityGate bool
		capacityMode NativePreemptionMode
	}{
		{
			name: "deadline_arrival_priority_and_cancel",
			requests: []request{
				{name: "thirty", deadline: 30, priority: -100},
				{name: "ten-first", deadline: 10, priority: 20},
				{name: "none-first", priority: -999},
				{name: "twenty", deadline: 20},
				{name: "ten-second", deadline: 10, priority: -200},
				{name: "none-second", priority: 99},
				{name: "canceled", deadline: 5, priority: -1000, cancel: true},
			},
			want: []string{"ten-first", "ten-second", "twenty", "thirty", "none-first", "none-second"},
		},
		{
			name: "deadline_overrides_picker",
			requests: []request{
				{name: "thirty", deadline: 30}, {name: "ten", deadline: 10}, {name: "twenty", deadline: 20},
			},
			want:   []string{"ten", "twenty", "thirty"},
			picker: true,
		},
		{
			name: "no_deadline_fifo",
			requests: []request{
				{name: "first", priority: 50}, {name: "second", priority: -50}, {name: "third"},
			},
			want: []string{"first", "second", "third"},
		},
		{
			name: "no_deadline_explicit_picker",
			requests: []request{
				{name: "first"}, {name: "second"}, {name: "third"},
			},
			want:   []string{"third", "second", "first"},
			picker: true,
		},
		{
			name: "recompute_resident_reservation_drains_before_deferred_deadline",
			requests: []request{
				{name: "resident", deadline: 30}, {name: "deferred-twenty", deadline: 20}, {name: "deferred-ten", deadline: 10},
			},
			want:         []string{"resident", "deferred-ten", "deferred-twenty"},
			capacityGate: true,
			capacityMode: NativePreemptRecompute,
		},
		{
			name: "swap_resident_reservation_drains_before_deferred_deadline",
			requests: []request{
				{name: "resident", deadline: 30}, {name: "deferred-twenty", deadline: 20}, {name: "deferred-ten", deadline: 10},
			},
			want:         []string{"resident", "deferred-ten", "deferred-twenty"},
			capacityGate: true,
			capacityMode: NativePreemptSwap,
		},
		{
			name: "preempted_before_deadlines",
			requests: []request{
				{name: "readmitted"}, {name: "twenty", deadline: 20}, {name: "ten", deadline: 10},
			},
			want:       []string{"readmitted", "ten", "twenty"},
			preemptOne: true,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			e := New()
			e.Preload(model.NewSynthetic(SyntheticConfig()))
			s := e.nativeScheduler()
			s.SetMaxRunning(1)
			s.SetKVPreemptionPolicy(NativePreemptionPolicy{})
			s.SetInBatchPrefixDedup(false)
			if tc.capacityGate {
				s.SetKVPreemptionPolicy(NativePreemptionPolicy{MaxBlocks: 4, BlockTokens: 16, Mode: tc.capacityMode})
			}
			if tc.preemptOne {
				s.SetKVPreemptionPolicy(NativePreemptionPolicy{MaxBlocks: 100_000, BlockTokens: 16, Mode: NativePreemptSwap})
			}
			if tc.picker {
				s.SetPromotionPicker(func(c []PromotionCandidate, slots int, _ map[string]int) []int {
					if slots <= 0 {
						return nil
					}
					return []int{len(c) - 1}
				})
			}
			s.executor.Lock()
			var unlocked sync.Once
			release := func() { unlocked.Do(s.executor.Unlock) }
			// A cancellation watchdog bounds a broken fixture without introducing
			// an inherited deadline into the no-deadline test cases.
			parent, cancelParent := context.WithCancel(context.Background())
			watchdog := time.AfterFunc(5*time.Second, cancelParent)
			t.Cleanup(func() {
				watchdog.Stop()
				cancelParent()
				s.Close()
				release()
				ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
				defer cancel()
				if err := s.CloseAndWait(ctx); err != nil {
					t.Errorf("CloseAndWait: %v", err)
				}
			})
			events := make(chan *schedLane, len(tc.requests)*genTokens)
			s.observeNativeEvent = func(event nativeSchedulerEvent) {
				if event.Kind == nativeSchedulerEventDecode {
					events <- event.Lane
				}
			}
			base := time.Now().Add(time.Minute)
			lanes := make([]*schedLane, 0, len(tc.requests))
			for i, r := range tc.requests {
				ctx := parent
				if r.deadline != 0 {
					var cancel context.CancelFunc
					ctx, cancel = context.WithDeadline(parent, base.Add(time.Duration(r.deadline)*time.Second))
					t.Cleanup(cancel)
				}
				call := inlineCall(r.name, `{}`)
				call.Meta = map[string]string{"priority": fmt.Sprint(r.priority)}
				req, err := e.Admit(ctx, call)
				if err != nil {
					t.Fatalf("Admit %s: %v", r.name, err)
				}
				ln, ok := req.(*schedLane)
				if !ok {
					t.Fatalf("Engine.Admit returned %T", req)
				}
				lanes = append(lanes, ln)
				if got, has := ln.ctx.Deadline(); has != (r.deadline != 0) || (has && !got.Equal(base.Add(time.Duration(r.deadline)*time.Second))) {
					t.Fatalf("Admit %s lost its exact caller deadline", r.name)
				}
				if ln.priority != r.priority {
					t.Fatalf("Admit %s priority=%d, want %d", r.name, ln.priority, r.priority)
				}
				if r.cancel {
					req.Cancel()
				}
				if tc.preemptOne && i == 0 {
					// Seed a real swap/restore round before later deadlines arrive;
					// victim selection is covered separately, not replaced here.
					s.mu.Lock()
					s.promoteWaitingLocked(1)
					promoted := len(s.lanes) == 1 && s.lanes[0] == ln
					if promoted {
						err = s.preemptLaneLocked(ln)
						if err == nil {
							s.lanes = s.lanes[:0]
						}
					}
					s.mu.Unlock()
					if !promoted || err != nil {
						t.Fatalf("seed preemption: promoted=%v err=%v", promoted, err)
					}
				}
			}
			if tc.capacityGate {
				// Actual admission owns the first resident reservation. Later,
				// earlier-deadline requests are genuinely deferred, not fabricated.
				s.mu.Lock()
				resident := lanes[0].sess != nil
				deferred := lanes[1].sess == nil && lanes[2].sess == nil
				waiting := len(s.waiting) == len(lanes) && len(s.lanes) == 0
				s.mu.Unlock()
				if !resident || !deferred || !waiting {
					t.Fatalf("reservation setup: resident=%v deferred=%v waiting=%v", resident, deferred, waiting)
				}
			}
			type outcome struct {
				count  int
				result *abi.Result
				err    error
			}
			outcomes := make([]outcome, len(lanes))
			var drained sync.WaitGroup
			for i, ln := range lanes {
				drained.Add(1)
				go func() {
					defer drained.Done()
					for range ln.Tokens() {
						outcomes[i].count++
					}
					outcomes[i].result, outcomes[i].err = ln.Result()
				}()
			}
			release()
			drained.Wait()
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			if err := s.CloseAndWait(ctx); err != nil {
				t.Fatalf("CloseAndWait: %v", err)
			}
			close(events)
			for i, got := range outcomes {
				if tc.requests[i].cancel {
					if got.count != 0 || got.result != nil || !errors.Is(got.err, context.Canceled) || !lanes[i].Reclaimed() {
						t.Fatalf("canceled lane %s: %+v", tc.requests[i].name, got)
					}
				} else if got.err != nil || got.result == nil || got.result.Status != abi.StatusOK || got.count != genTokens || !lanes[i].Reclaimed() {
					t.Fatalf("completed lane %s: %+v", tc.requests[i].name, got)
				}
			}
			seen := make(map[*schedLane]bool)
			var got []string
			for ln := range events {
				if !seen[ln] {
					seen[ln] = true
					got = append(got, ln.tool)
				}
			}
			if !slices.Equal(got, tc.want) {
				t.Fatalf("native decode order=%v, want %v", got, tc.want)
			}
			if tc.capacityGate && (s.maxObservedRunning > 1 || s.maxObservedKVBlocks > s.preemption.MaxBlocks) {
				t.Fatalf("capacity breach: running=%d KV blocks=%d limit=%d", s.maxObservedRunning, s.maxObservedKVBlocks, s.preemption.MaxBlocks)
			}
			if tc.preemptOne && s.preemptStats.Readmitted != 1 {
				t.Fatalf("readmitted=%d, want 1", s.preemptStats.Readmitted)
			}
		})
	}
}
