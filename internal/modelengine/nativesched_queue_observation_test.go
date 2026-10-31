package modelengine

// These tests drive the real Go scheduler and recorder with synthetic CPU model
// state and a deterministic clock. They do not validate GPU execution, measured
// overhead, or end-to-end request latency. Runtime estimates are not measurements.

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"math"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/enginestep"
	"github.com/anthony-chaudhary/fak/internal/model"
)

func nativeQueueObservationFixture(t *testing.T) (*NativeScheduler, *enginestep.Recorder, *time.Time) {
	t.Helper()
	s := NewNativeScheduler(model.NewSynthetic(SyntheticConfig()))
	// The test owns the executor; no background scheduler or wall-clock sleeps.
	s.started.Do(func() {})
	now := time.Unix(100, 0)
	s.now = func() time.Time { return now }
	rec := enginestep.New(32)
	rec.SetClock(func() time.Time {
		// Recorder output must run outside scheduler.mu, including the deferred
		// materialization return. The clock runs under recorder.mu.
		if !s.mu.TryLock() {
			t.Error("recorder called while scheduler mutex was held")
		} else {
			s.mu.Unlock()
		}
		return now
	})
	s.recorder = rec
	t.Cleanup(func() {
		s.Close()
		s.executor.Lock()
		s.mu.Lock()
		for _, group := range [][]*schedLane{s.lanes, s.waiting, s.preempted} {
			for _, lane := range group {
				lane.finish(nil, context.Canceled)
			}
		}
		s.lanes, s.waiting, s.preempted = nil, nil, nil
		s.mu.Unlock()
		s.executor.Unlock()
	})
	return s, rec, &now
}

func admitQueueObservationLane(t *testing.T, s *NativeScheduler, name string) *schedLane {
	t.Helper()
	req, err := s.AdmitTokenIDs(context.Background(), name, []int{1, 2})
	if err != nil {
		t.Fatalf("AdmitTokenIDs: %v", err)
	}
	lane, ok := req.(*schedLane)
	if !ok {
		t.Fatalf("request type = %T", req)
	}
	return lane
}

func runQueueObservationIteration(s *NativeScheduler) {
	s.executor.Lock()
	s.runIteration(true)
	s.executor.Unlock()
	// Nonblocking drain prevents the token channel from obscuring queue timing.
	for _, lane := range s.lanes {
		select {
		case <-lane.tokens:
		default:
		}
	}
}

func assertQueueObservation(t *testing.T, rec *enginestep.Recorder, count uint64, sum time.Duration) {
	t.Helper()
	phase := rec.Snapshot(0, "").Phases[string(enginestep.PhaseSchedulerWait)]
	if phase.Count != count || math.Abs(phase.SumSeconds-sum.Seconds()) > 1e-12 {
		t.Fatalf("scheduler_wait = %+v, want count=%d sum=%s", phase, count, sum)
	}
}

// fak-test:runtime fast est=200ms lane=default
func TestNativeSchedulerQueueResidenceProductionConsumers(t *testing.T) {
	s, rec, now := nativeQueueObservationFixture(t)
	s.SetSessionProfilerFactory(func(NativeSessionLifecycle) *model.PhaseProfiler {
		*now = now.Add(5 * time.Second)
		return nil
	})
	lane := admitQueueObservationLane(t, s, "before-enqueue-prefill")
	if !lane.queueWaitPending || !lane.queueEnteredAt.Equal(*now) {
		t.Fatal("ordinary admission did not stamp its actual enqueue after prefill")
	}
	*now = now.Add(7 * time.Millisecond)
	runQueueObservationIteration(s)
	assertQueueObservation(t, rec, 1, 7*time.Millisecond)
	if lane.queueWaitPending {
		t.Fatal("first promotion left a second observation armed")
	}
	*now = now.Add(time.Second)
	runQueueObservationIteration(s)
	assertQueueObservation(t, rec, 1, 7*time.Millisecond)

	var prom bytes.Buffer
	rec.WritePrometheus(&prom)
	for _, want := range []string{
		`fak_engine_phase_seconds_count{phase="scheduler_wait"} 1` + "\n",
		`fak_engine_phase_seconds_sum{phase="scheduler_wait"} 0.007` + "\n",
		`fak_engine_phase_seconds_count{phase="admission_wait"} 0` + "\n",
	} {
		if !strings.Contains(prom.String(), want) {
			t.Fatalf("production Prometheus consumer missing %q", want)
		}
	}
	body, err := json.Marshal(rec.Snapshot(32, enginestep.KindPhase))
	if err != nil {
		t.Fatal(err)
	}
	var decoded enginestep.Snapshot
	if err := json.Unmarshal(body, &decoded); err != nil {
		t.Fatal(err)
	}
	if decoded.Phases["scheduler_wait"].Count != 1 {
		t.Fatalf("JSON consumer lost queue phase: %s", body)
	}
	var queueRecords int
	for _, record := range decoded.Recent {
		if record.Phase == "scheduler_wait" {
			queueRecords++
			if record.DurationNS != int64(7*time.Millisecond) {
				t.Fatalf("queue ring duration = %d", record.DurationNS)
			}
		}
	}
	if queueRecords != 1 {
		t.Fatalf("queue ring records = %d, want 1", queueRecords)
	}
}

// fak-test:runtime fast est=200ms lane=default
func TestNativeSchedulerQueueResidenceCapacityEnqueues(t *testing.T) {
	s, rec, now := nativeQueueObservationFixture(t)
	s.SetMaxRunning(1)
	s.SetKVPreemptionPolicy(NativePreemptionPolicy{Mode: NativePreemptRecompute, MaxBlocks: 1, BlockTokens: 16})
	s.SetSessionProfilerFactory(func(NativeSessionLifecycle) *model.PhaseProfiler {
		*now = now.Add(3 * time.Millisecond)
		return nil
	})
	entered := *now
	first := admitQueueObservationLane(t, s, "reserved-initialization")
	if !first.queueEnteredAt.Equal(entered) || first.sess == nil {
		t.Fatal("capacity reservation did not stamp before its initialization")
	}
	second := admitQueueObservationLane(t, s, "deferred-no-session")
	if !second.queueWaitPending || second.sess != nil || !second.queueEnteredAt.Equal(*now) {
		t.Fatal("deferred admission lost its timestamp or materialized early")
	}
	*now = now.Add(7 * time.Millisecond)
	runQueueObservationIteration(s)
	assertQueueObservation(t, rec, 1, 10*time.Millisecond)
	if !second.queueWaitPending {
		t.Fatal("capacity-blocked lane was counted as promoted")
	}
	// Retire the first running owner through the real cancellation boundary.
	first.Cancel()
	runQueueObservationIteration(s)
	*now = now.Add(5 * time.Millisecond)
	runQueueObservationIteration(s)
	// The second first-promotion observation survives runIteration's early
	// materialization return and excludes that later 3ms of initialization.
	assertQueueObservation(t, rec, 2, 22*time.Millisecond)
	if second.sess == nil || second.queueWaitPending {
		t.Fatal("deferred lane did not materialize after promotion")
	}
	if _, err := first.Result(); !errors.Is(err, context.Canceled) {
		t.Fatalf("first result = %v, want cancellation", err)
	}
}

// fak-test:runtime fast est=200ms lane=default
func TestNativeSchedulerQueueResidenceNeverPromoted(t *testing.T) {
	s, rec, now := nativeQueueObservationFixture(t)
	lane := admitQueueObservationLane(t, s, "cancel-before-promotion")
	lane.Cancel()
	*now = now.Add(time.Second)
	runQueueObservationIteration(s)
	assertQueueObservation(t, rec, 0, 0)
	if !lane.Reclaimed() {
		t.Fatal("canceled waiting lane did not release its original owner")
	}
	s.SetMaxRunning(1)
	s.SetKVPreemptionPolicy(NativePreemptionPolicy{Mode: NativePreemptRecompute, MaxBlocks: 1, BlockTokens: 1})
	if req, err := s.AdmitTokenIDs(context.Background(), "refused", []int{1, 2}); req != nil || !errors.Is(err, ErrNativePromptExceedsKVCapacity) {
		t.Fatalf("oversized admission = (%v, %v)", req, err)
	}
	assertQueueObservation(t, rec, 0, 0)
}

// fak-test:runtime fast est=200ms lane=default
func TestNativeSchedulerQueueResidenceOnlyFirstPromotion(t *testing.T) {
	s, rec, now := nativeQueueObservationFixture(t)
	lane := admitQueueObservationLane(t, s, "first-promotion-only")
	*now = now.Add(4 * time.Millisecond)
	runQueueObservationIteration(s)
	assertQueueObservation(t, rec, 1, 4*time.Millisecond)
	// A synthetic requeue of the same owner isolates the exactly-once contract;
	// this is not a swap/restore, allocator, or physical preemption witness.
	s.mu.Lock()
	s.lanes = nil
	s.waiting = append(s.waiting, lane)
	s.mu.Unlock()
	*now = now.Add(time.Second)
	runQueueObservationIteration(s)
	assertQueueObservation(t, rec, 1, 4*time.Millisecond)
}

// fak-test:runtime fast est=10ms lane=default
func TestNativeSchedulerQueueResidenceInvalidClockAndDisabled(t *testing.T) {
	for _, name := range []string{"backward", "saturated", "nil-recorder", "unobserved-owner"} {
		t.Run(name, func(t *testing.T) {
			s := NewNativeScheduler(nil)
			now := time.Unix(100, 0)
			s.now = func() time.Time { return now }
			lane := &schedLane{ctx: context.Background()}
			if name == "nil-recorder" {
				s.recorder = nil
				s.now = func() time.Time { t.Fatal("nil recorder used queue clock"); return time.Time{} }
			}
			s.mu.Lock()
			if name == "unobserved-owner" {
				s.waiting = append(s.waiting, lane)
			} else {
				s.enqueueNewWaitingLocked(lane)
			}
			switch name {
			case "backward":
				now = now.Add(-time.Nanosecond)
			case "saturated":
				now = now.AddDate(400, 0, 0)
			}
			waits := s.promoteWaitingLocked(0)
			s.mu.Unlock()
			if len(waits) != 0 || lane.queueWaitPending || len(s.lanes) != 1 {
				t.Fatalf("waits=%v pending=%v running=%d", waits, lane.queueWaitPending, len(s.lanes))
			}
		})
	}
}
