package modelengine

// Deterministic clock/channel witnesses for the actual deferred-token boundary.
// They do not measure real backpressure, observer overhead or device execution.

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/enginestep"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// fak-test:runtime fast est=3s lane=default
func TestNativeSchedulerTokenPublicationObservation(t *testing.T) {
	for _, tc := range []struct {
		name       string
		full       bool
		cancelWait bool
		preCancel  bool
		noPending  bool
		nilRecord  bool
		elapsed    time.Duration
		wantCount  uint64
	}{
		{name: "ready_send", elapsed: 7 * time.Millisecond, wantCount: 1},
		{name: "zero_duration_is_observed", wantCount: 1},
		{name: "full_channel_released", full: true, elapsed: 11 * time.Millisecond, wantCount: 1},
		{name: "cancel_wins_full_channel", full: true, cancelWait: true, elapsed: 13 * time.Millisecond, wantCount: 1},
		{name: "canceled_before_select", preCancel: true},
		{name: "no_pending_token", noPending: true},
		{name: "nil_recorder_no_clock", nilRecord: true},
		{name: "negative_interval_omitted", elapsed: -time.Nanosecond},
		{name: "saturated_interval_omitted", elapsed: time.Duration(1<<63 - 1)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s, ln := deferredLifecycleFixture(t, []int{1, 2, 3})
			// Match production's bounded output capacity before any publication.
			ln.tokens = make(chan abi.EngineToken, 1)
			rec := enginestep.New(8)
			rec.ObservePhase(enginestep.PhaseRequest, time.Millisecond)
			s.recorder = rec
			if tc.nilRecord {
				s.recorder = nil
			}
			var closes atomic.Int64
			s.closeSession = func(session *model.Session) {
				closes.Add(1)
				session.Close()
			}
			s.executor.Lock()
			staged := ln.stageDeferredToken()
			s.executor.Unlock()
			if !staged || ln.deferredToken == nil || ln.deferredToken.done {
				t.Fatal("fixture did not stage a nonterminal token from real prefill logits")
			}
			wantID := ln.deferredToken.id
			if tc.noPending {
				ln.deferredToken = nil
			}
			const bufferedID = -17 // an older channel value, not part of this attempt
			if tc.full {
				ln.tokens <- abi.EngineToken{ID: bufferedID}
			}
			if tc.preCancel {
				ln.Cancel()
			}
			entered := make(chan struct{})
			base := time.Unix(100, 0)
			var clockCalls atomic.Int64
			priorClock := s.now
			s.now = func() time.Time {
				n := clockCalls.Add(1)
				if !s.mu.TryLock() {
					t.Error("publication clock ran under scheduler mutex")
				} else {
					s.mu.Unlock()
				}
				if n == 1 {
					close(entered)
					return base
				}
				return base.Add(tc.elapsed)
			}
			// Restore the clock before the existing fixture's cancellation drain.
			t.Cleanup(func() { s.now = priorClock })
			finished := make(chan bool, 1)
			joined := false
			t.Cleanup(func() {
				if joined {
					return
				}
				ln.Cancel()
				select {
				case <-finished:
				case <-time.After(2 * time.Second):
					t.Error("publication worker did not retire after cancellation")
				}
			})
			go func() {
				s.executor.Lock()
				continued := ln.retireDeferredToken(1)
				s.executor.Unlock()
				finished <- continued
			}()
			if !tc.preCancel && !tc.noPending && !tc.nilRecord {
				select {
				case <-entered:
				case <-time.After(2 * time.Second):
					t.Fatal("publication attempt did not enter its clock boundary")
				}
			}
			if tc.full {
				select {
				case <-finished:
					joined = true
					t.Fatal("full channel completed before release or cancellation")
				default:
				}
				if tc.cancelWait {
					ln.Cancel()
				} else if old := <-ln.tokens; old.ID != bufferedID {
					t.Fatalf("buffered output changed before release: %+v", old)
				}
			}
			var continued bool
			select {
			case continued = <-finished:
				joined = true
			case <-time.After(2 * time.Second):
				t.Fatal("publication worker did not complete")
			}
			wantClocks := int64(2)
			if tc.preCancel || tc.noPending || tc.nilRecord {
				wantClocks = 0
			}
			if got := clockCalls.Load(); got != wantClocks {
				t.Fatalf("clock calls=%d, want %d", got, wantClocks)
			}
			if tc.preCancel || tc.cancelWait {
				if continued || !ln.terminal || closes.Load() != 1 || !errors.Is(ln.err, context.Canceled) {
					t.Fatalf("cancel continuation=%t terminal=%t closes=%d err=%v", continued, ln.terminal, closes.Load(), ln.err)
				}
				if ln.emitted != 0 || len(ln.gen) != 0 {
					t.Fatal("canceled send was counted as emitted output")
				}
				var remaining []int
				for token := range ln.tokens {
					remaining = append(remaining, token.ID)
				}
				if tc.full {
					if len(remaining) != 1 || remaining[0] != bufferedID {
						t.Fatalf("cancel lost earlier buffered output or published pending token: %v", remaining)
					}
				} else if len(remaining) != 0 {
					t.Fatalf("pre-select cancellation published output: %v", remaining)
				}
			} else if tc.noPending {
				if continued || ln.terminal || ln.emitted != 0 || len(ln.gen) != 0 || closes.Load() != 0 {
					t.Fatal("missing pending token mutated the lane")
				}
			} else {
				if !continued || ln.terminal || ln.emitted != 1 || len(ln.gen) != 1 || ln.gen[0] != wantID || closes.Load() != 0 {
					t.Fatal("successful publication changed output/accounting/lifetime")
				}
				select {
				case token := <-ln.tokens:
					if token.ID != wantID {
						t.Fatalf("token=%d, want %d", token.ID, wantID)
					}
				default:
					t.Fatal("successful publication produced no token")
				}
			}
			if ln.deferredToken != nil && !tc.preCancel {
				t.Fatal("completed attempt retained its pending token")
			}
			snap := rec.Snapshot(8, "")
			phase := snap.Phases[string(enginestep.PhaseTokenPublish)]
			wantSum := 0.0
			if tc.wantCount != 0 {
				wantSum = tc.elapsed.Seconds()
			}
			if phase.Count != tc.wantCount || math.Abs(phase.SumSeconds-wantSum) > 1e-12 {
				t.Fatalf("token_publish=%+v, want count=%d sum=%g", phase, tc.wantCount, wantSum)
			}
			if len(snap.Recent) != 1 || snap.Recent[0].Phase != string(enginestep.PhaseRequest) || len(snap.Decode) != 0 || snap.Phases[string(enginestep.PhasePrefill)].Count != 0 {
				t.Fatal("per-token phase displaced recent decode evidence or changed model-phase counters")
			}
			var prom bytes.Buffer
			rec.WritePrometheus(&prom)
			wantLine := fmt.Sprintf("fak_engine_phase_seconds_count{phase=\"token_publish\"} %d\n", tc.wantCount)
			if !bytes.Contains(prom.Bytes(), []byte(wantLine)) {
				t.Fatalf("Prometheus consumer missing %q", wantLine)
			}
			encoded, err := json.Marshal(snap)
			if err != nil {
				t.Fatal(err)
			}
			var decoded enginestep.Snapshot
			if err := json.Unmarshal(encoded, &decoded); err != nil {
				t.Fatal(err)
			}
			if got := decoded.Phases["token_publish"]; got.Count != tc.wantCount || math.Abs(got.SumSeconds-wantSum) > 1e-12 {
				t.Fatalf("JSON consumer lost phase: %+v", got)
			}
		})
	}

	t.Run("actual_iteration_keeps_decode_envelope", func(t *testing.T) {
		s, ln := deferredLifecycleFixture(t, []int{4, 5, 6})
		rec := enginestep.New(16)
		s.recorder = rec
		rec.SetClock(func() time.Time {
			if !s.mu.TryLock() {
				t.Error("iteration recorder ran under scheduler mutex")
			} else {
				s.mu.Unlock()
			}
			return time.Unix(200, 0)
		})
		s.executor.Lock()
		s.runIteration(true)
		s.executor.Unlock()
		snap := rec.Snapshot(16, "")
		if snap.Phases[string(enginestep.PhaseTokenPublish)].Count != 1 || ln.emitted != 1 {
			t.Fatal("real scheduler iteration did not record exactly one publication")
		}
		var prom bytes.Buffer
		rec.WritePrometheus(&prom)
		if got := promCount(t, prom.String(), enginestep.MetricDecodeStepSeconds); got != 1 {
			t.Fatalf("decode envelope count=%g, want 1", got)
		}
		for _, record := range snap.Recent {
			if record.Phase == string(enginestep.PhaseTokenPublish) {
				t.Fatal("per-token phase flooded the recent-step ring")
			}
		}
	})
}
