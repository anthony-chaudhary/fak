package modelengine

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/enginestep"
	"github.com/anthony-chaudhary/fak/internal/radixkv"
)

// Actual scheduler, host Qwen Session and recorder with synthetic weights and a
// deterministic clock. This is not GPU completion, serving throughput or measured
// observer overhead. Runtime is an estimate, not a returned measurement.
// fak-test:runtime medium est=4s lane=default
func TestNativeSchedulerComputedPrefillObservations(t *testing.T) {
	cfg := nativeSchedulerPrefillConfig()
	cfg.EOSTokenID, cfg.EOSTokenIDs = -1, nil
	m := nativeSchedulerPrefillModel(t, cfg)
	prompt := nativeSchedulerQwenPrompt(35) // Includes a real three-token tail.
	control := m.NewSession()
	control.Quant, control.Q4K = false, true
	wantLogits := copyF32(control.Prefill(prompt))
	control.Close()

	for _, tc := range []struct {
		name                              string
		cached                            int
		cancelAt                          string
		invalidClock                      string
		wantTokens, wantChunks, wantCalls int
	}{
		{name: "cold", wantTokens: 35, wantChunks: 3, wantCalls: 3},
		{name: "cached prefix is not computed", cached: 16, wantTokens: 19, wantChunks: 2, wantCalls: 2},
		{name: "full hit", cached: 35},
		{name: "canceled before execution", cancelAt: "before"},
		{name: "canceled before publication", cancelAt: "model", wantCalls: 1},
		{name: "canceled after publication", cancelAt: "published", wantTokens: 16, wantChunks: 1, wantCalls: 1},
		{name: "regressing clock omitted", invalidClock: "regress", wantCalls: 3},
		{name: "saturated clock omitted", invalidClock: "saturate", wantCalls: 3},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := newNativeScheduler(m, nativeSchedulerPrefillPrepare(map[string][]int{"target": prompt}))
			s.coupler = nil         // Isolate this witness from host-dependent worker budgets.
			s.started.Do(func() {}) // The test is the only executor; no background loop.
			if err := s.SetQwenPrefillMaxTokensPerIteration(nativeQwenPrefillMinChunkTokens); err != nil {
				t.Fatal(err)
			}
			if tc.cached > 0 {
				seed := m.NewSession()
				seed.Quant, seed.Q4K = false, true
				logits := seed.Prefill(prompt[:tc.cached])
				tree := radixkv.New(0)
				boundary, matched := tree.Lookup(prompt[:tc.cached])
				tree.Done(tree.InsertWithLogits(boundary, prompt[matched:tc.cached], seed.Cache, logits))
				seed.Close()
				s.SetRadixKV(tree)
			}
			now := time.Unix(100, 0)
			var modelStart time.Time
			modelClockReads := 0
			s.now = func() time.Time {
				now = now.Add(time.Millisecond)
				switch modelClockReads {
				case 2:
					modelClockReads--
					modelStart = now
				case 1:
					modelClockReads--
					switch tc.invalidClock {
					case "regress":
						return modelStart.Add(-time.Millisecond)
					case "saturate":
						return modelStart.Add(time.Duration(1<<63 - 1))
					}
				}
				return now
			}
			rec := enginestep.New(64)
			rec.SetClock(func() time.Time {
				if !s.mu.TryLock() {
					t.Error("recorder callback ran under scheduler mutex")
				} else {
					s.mu.Unlock()
				}
				return now
			})
			s.recorder = rec
			nativeSchedulerBeginManualDrain(t, s)
			defer nativeSchedulerEndManualDrain(s)
			lane := nativeSchedulerAdmitLane(t, s, "target")
			defer func() {
				lane.Cancel()
				nativeSchedulerDriveIteration(t, s)
				nativeSchedulerDrainAvailable(lane)
			}()
			calls, published := 0, 0
			var finalLogits []float32
			s.beforeModelExecute = func(kind nativeSchedulerEventKind, ln *schedLane) {
				if kind == nativeSchedulerEventPrefill {
					calls++
					modelClockReads = 2
					if tc.cancelAt == "model" {
						ln.Cancel()
					}
				}
			}
			s.observeNativeEvent = func(ev nativeSchedulerEvent) {
				if ev.Kind == nativeSchedulerEventPrefill {
					published++
					if tc.cancelAt == "published" {
						ev.Lane.Cancel()
					}
				}
				if ev.Kind == nativeSchedulerEventTransition {
					finalLogits = copyF32(ev.Lane.logits)
				}
			}
			if tc.cancelAt == "before" {
				lane.Cancel()
			}
			var output []int
			for i := 0; i < 5; i++ {
				nativeSchedulerDriveIteration(t, s)
				output = append(output, nativeSchedulerDrainAvailable(lane)...)
				if lane.terminal || len(output) > 0 {
					break
				}
			}
			if calls != tc.wantCalls {
				t.Fatalf("actual model calls=%d, want %d", calls, tc.wantCalls)
			}
			if tc.cancelAt == "" {
				if len(output) != 1 || output[0] != argmax(wantLogits) {
					t.Fatalf("first continuation=%v, want [%d]", output, argmax(wantLogits))
				}
				nativeSchedulerAssertLogitParity(t, finalLogits, wantLogits)
			} else if !lane.terminal || len(output) != 0 {
				t.Fatalf("cancellation changed execution outcome: terminal=%v output=%v", lane.terminal, output)
			}
			if tc.invalidClock == "" && published != tc.wantChunks {
				t.Fatalf("successful chunk publications=%d, want %d", published, tc.wantChunks)
			}
			snap := rec.Snapshot(64, enginestep.KindPrefillChunk)
			if snap.PrefillTokens != uint64(tc.wantTokens) || snap.PrefillChunks != uint64(tc.wantChunks) || len(snap.Recent) != tc.wantChunks {
				t.Fatalf("computed telemetry=%+v, want tokens=%d chunks=%d", snap, tc.wantTokens, tc.wantChunks)
			}
			// The existing aggregate phase is still recorded ONCE per attempt,
			// independently of success, cancellation and computed token records.
			if got := snap.Phases[string(enginestep.PhasePrefill)].Count; got != uint64(tc.wantCalls) {
				t.Fatalf("aggregate prefill count=%d, want %d (no double record)", got, tc.wantCalls)
			}
			tokens := 0
			for i, record := range snap.Recent {
				want := nativeQwenPrefillMinChunkTokens
				if left := tc.wantTokens - tokens; left < want {
					want = left
				}
				if record.Kind != enginestep.KindPrefillChunk || record.Tokens != want || record.DurationNS != int64(time.Millisecond) {
					t.Fatalf("computed chunk[%d]=%+v, want tokens=%d/model-call=1ms", i, record, want)
				}
				tokens += record.Tokens
			}
			var prom bytes.Buffer
			rec.WritePrometheus(&prom)
			for _, want := range []string{
				fmt.Sprintf("fak_engine_prefill_tokens_total %d\n", tc.wantTokens),
				fmt.Sprintf("fak_engine_prefill_chunks_total %d\n", tc.wantChunks),
			} {
				if !strings.Contains(prom.String(), want) {
					t.Fatalf("real Prometheus consumer missing %q", want)
				}
			}
			body, err := json.Marshal(snap)
			if err != nil {
				t.Fatal(err)
			}
			var decoded enginestep.Snapshot
			if err := json.Unmarshal(body, &decoded); err != nil {
				t.Fatal(err)
			}
			if decoded.PrefillTokens != snap.PrefillTokens || decoded.PrefillChunks != snap.PrefillChunks || len(decoded.Recent) != len(snap.Recent) {
				t.Fatalf("JSON consumer lost computed observations: %s", body)
			}
		})
	}
}
