package modelengine

// The witness drives actual ordinary scheduler iterations and dense host swap
// with synthetic CPU weights. It is not hardware, served-path qualification, or
// a measured latency improvement. The benchmark isolates idle boundary overhead.

import (
	"bytes"
	"encoding/json"
	"math"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/enginestep"
)

var nativeStateObservationPhases = [...]enginestep.Phase{
	"scheduler_state_wait", "scheduler_readmit", "scheduler_preempt", "scheduler_state_other",
}

func nativeStateObservationSum(rec *enginestep.Recorder) float64 {
	var sum float64
	for _, phase := range nativeStateObservationPhases {
		sum += rec.Snapshot(0, "").Phases[string(phase)].SumSeconds
	}
	return sum
}

func nativeStateObservationStep(t *testing.T, s *NativeScheduler, rec *enginestep.Recorder) {
	t.Helper()
	before := nativeStateObservationSum(rec)
	started := time.Now()
	nativeSchedulerDriveIteration(t, s)
	elapsed := time.Since(started).Seconds()
	// The four non-overlapping intervals must fit inside the actual iteration;
	// model/stream work and recorder callbacks are excluded from their sum.
	if delta := nativeStateObservationSum(rec) - before; delta < 0 || delta > elapsed+1e-9 {
		t.Fatalf("state partition=%g seconds, whole iteration=%g", delta, elapsed)
	}
}

func nativeStateObservationCallbacks(t *testing.T, s *NativeScheduler) {
	t.Helper()
	enginestep.SetPhaseObserver(func(phase enginestep.Phase, d time.Duration) {
		for _, expected := range nativeStateObservationPhases {
			if phase != expected {
				continue
			}
			if d < 0 {
				t.Errorf("negative state interval %q: %s", phase, d)
			}
			if !s.mu.TryLock() {
				t.Errorf("state observer %q called under scheduler mutex", phase)
			} else {
				s.mu.Unlock()
			}
		}
	})
	t.Cleanup(func() { enginestep.SetPhaseObserver(nil) })
}

// fak-test:runtime medium est=2s lane=default
func TestNativeSchedulerStatePhaseObservations(t *testing.T) {
	t.Run("dense-swap-and-readmit", func(t *testing.T) {
		drive := func(t *testing.T, pressure bool) ([][]int, NativePreemptionStats, *enginestep.Recorder, int) {
			t.Helper()
			s, rec, _ := nativeQueueObservationFixture(t)
			s.coupler = nil
			nativeStateObservationCallbacks(t, s)
			if pressure {
				s.SetKVPreemptionPolicy(NativePreemptionPolicy{Mode: NativePreemptSwap, MaxBlocks: 1, BlockTokens: 128})
			}
			lanes := []*schedLane{
				admitQueueObservationLane(t, s, "first"),
				admitQueueObservationLane(t, s, "second"),
			}
			out := make([][]int, len(lanes))
			iterations := 0
			for iterations < 2*genTokens+8 {
				nativeStateObservationStep(t, s, rec)
				iterations++
				finished := true
				for i, lane := range lanes {
					out[i] = append(out[i], nativeSchedulerDrainAvailable(lane)...)
					finished = finished && lane.terminal
				}
				if finished {
					for _, lane := range lanes {
						if _, err := lane.Result(); err != nil {
							t.Fatal(err)
						}
					}
					return out, s.KVPreemptionStats(), rec, iterations
				}
			}
			t.Fatal("bounded scheduler drive did not retire both requests")
			return nil, NativePreemptionStats{}, rec, iterations
		}
		want, _, _, _ := drive(t, false)
		got, stats, rec, iterations := drive(t, true)
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("timed pressure changed token streams: got %v want %v", got, want)
		}
		if stats.SwapPreemptions != 1 || stats.Readmitted != 1 || stats.SwapBytes == 0 || stats.SwapRestoredBytes != stats.SwapBytes {
			t.Fatalf("no actual dense round trip: %+v", stats)
		}
		snapshot := rec.Snapshot(32, enginestep.KindPhase)
		for _, phase := range nativeStateObservationPhases {
			stat := snapshot.Phases[string(phase)]
			if stat.Count != uint64(iterations) || stat.SumSeconds < 0 {
				t.Fatalf("%s=%+v, want %d visited passes/boundaries", phase, stat, iterations)
			}
		}
		var prom bytes.Buffer
		rec.WritePrometheus(&prom)
		body, err := json.Marshal(snapshot)
		if err != nil {
			t.Fatal(err)
		}
		var decoded enginestep.Snapshot
		if err := json.Unmarshal(body, &decoded); err != nil {
			t.Fatal(err)
		}
		for _, phase := range nativeStateObservationPhases {
			if !strings.Contains(prom.String(), "fak_engine_phase_seconds_count{phase=\""+string(phase)+"\"}") ||
				decoded.Phases[string(phase)].Count != uint64(iterations) {
				t.Fatalf("existing metric/JSON consumer lost %q", phase)
			}
			for _, record := range snapshot.Recent {
				if record.Phase == string(phase) {
					t.Fatalf("per-iteration timing flushed decode ring: %+v", record)
				}
			}
		}
	})

	t.Run("restore-refusal-includes-cleanup", func(t *testing.T) {
		s, rec, _ := nativeQueueObservationFixture(t)
		nativeStateObservationCallbacks(t, s)
		s.SetKVPreemptionPolicy(NativePreemptionPolicy{Mode: NativePreemptSwap, MaxBlocks: 1, BlockTokens: 128})
		lane := admitQueueObservationLane(t, s, "bad-swap")
		s.mu.Lock()
		s.waiting = nil
		s.closeLaneSession(lane.sess)
		lane.sess = nil
		lane.preemptMode = NativePreemptSwap
		lane.hostKV = []byte("invalid-host-snapshot")
		s.preempted = []*schedLane{lane}
		s.mu.Unlock()
		nativeStateObservationStep(t, s, rec)
		if !lane.terminal || lane.err == nil || !lane.Reclaimed() || lane.hostKV != nil || len(s.preempted) != 0 {
			t.Fatal("refused restore did not retire the actual owned lane")
		}
		snapshot := rec.Snapshot(0, "")
		for _, phase := range nativeStateObservationPhases {
			if snapshot.Phases[string(phase)].Count != 1 {
				t.Fatalf("refused restore lost %s", phase)
			}
		}
		if s.KVPreemptionStats().Readmitted != 0 {
			t.Fatal("failed restore became a successful readmission")
		}
	})

	t.Run("deferred-materialization-first-unlock", func(t *testing.T) {
		s, rec, _ := nativeQueueObservationFixture(t)
		nativeStateObservationCallbacks(t, s)
		s.SetMaxRunning(1)
		s.SetKVPreemptionPolicy(NativePreemptionPolicy{Mode: NativePreemptRecompute, MaxBlocks: 1, BlockTokens: 16})
		first := admitQueueObservationLane(t, s, "occupied")
		second := admitQueueObservationLane(t, s, "deferred")
		if second.sess != nil {
			t.Fatal("fixture did not defer initialization")
		}
		first.Cancel()
		nativeStateObservationStep(t, s, rec)
		if second.sess == nil {
			t.Fatal("real early-return materialization did not publish its owner")
		}
		phases := rec.Snapshot(0, "").Phases
		if phases["scheduler_state_wait"].Count != 1 || phases["scheduler_state_other"].Count != 1 ||
			phases["scheduler_preempt"].Count != 0 || phases["scheduler_readmit"].Count != 0 {
			t.Fatalf("early-return/skip denominators wrong: %+v", phases)
		}
	})

	t.Run("initial-mutex-wait", func(t *testing.T) {
		s, rec, _ := nativeQueueObservationFixture(t)
		nativeStateObservationCallbacks(t, s)
		s.mu.Lock()
		entered, done := make(chan struct{}), make(chan struct{})
		go func() {
			s.executor.Lock()
			close(entered)
			s.runIteration(false)
			s.executor.Unlock()
			close(done)
		}()
		<-entered
		// Controlled contention, not a timing threshold or hardware benchmark.
		time.Sleep(time.Millisecond)
		select {
		case <-done:
			s.mu.Unlock()
			t.Fatal("iteration passed the held initial mutex")
		default:
		}
		s.mu.Unlock()
		select {
		case <-done:
		case <-time.After(time.Second):
			t.Fatal("iteration failed to finish after mutex release")
		}
		stat := rec.Snapshot(0, "").Phases["scheduler_state_wait"]
		if stat.Count != 1 || stat.SumSeconds <= 0 || math.IsNaN(stat.SumSeconds) {
			t.Fatalf("no completed initial lock wait: %+v", stat)
		}
	})

	t.Run("nil-recorder", func(t *testing.T) {
		s, rec, _ := nativeQueueObservationFixture(t)
		s.recorder = nil
		s.now = func() time.Time { t.Fatal("idle disabled recording read request clock"); return time.Time{} }
		nativeSchedulerDriveIteration(t, s)
		if len(rec.Snapshot(0, "").Phases) != 0 {
			t.Fatal("nil recorder emitted a phase")
		}
	})
}

// Same source-bound idle workload is runnable with this fixture on the parent.
// Report both on/off ns/op and B/op; neither measures model execution or TTFT.
func BenchmarkNativeSchedulerStatePhaseOverhead(b *testing.B) {
	for _, enabled := range []bool{false, true} {
		name := "recorder-off"
		if enabled {
			name = "recorder-on"
		}
		b.Run(name, func(b *testing.B) {
			s := NewNativeScheduler(nil)
			s.started.Do(func() {})
			s.recorder = nil
			if enabled {
				s.recorder = enginestep.New(1)
			}
			b.ReportAllocs()
			b.ResetTimer()
			s.executor.Lock()
			for i := 0; i < b.N; i++ {
				s.runIteration(false)
			}
			s.executor.Unlock()
			b.StopTimer()
			s.Close()
		})
	}
}
