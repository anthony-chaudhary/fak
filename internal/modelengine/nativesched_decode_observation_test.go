package modelengine

import (
	"context"
	"slices"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/enginestep"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// fak-test:runtime fast est=1s lane=default
// This exercises the real lifecycle scheduler and model, not GPU performance.
func TestNativeSchedulerDecodeObservationMatchesExecution(t *testing.T) {
	for _, tc := range []struct {
		name        string
		unsupported bool
		solo        bool
		terminal    int
		serial      uint64
		batched     uint64
		tokens      uint64
	}{
		{name: "serial_fallback", unsupported: true, serial: 2, tokens: 2},
		{name: "shared_batch", batched: 1, tokens: 2},
		{name: "one_lane_retires", terminal: 1, serial: 1, tokens: 1},
		{name: "all_lanes_retire", terminal: 2},
		{name: "solo_multiple_forwards", solo: true, serial: 3, tokens: 3},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := SyntheticConfig()
			if tc.unsupported {
				// The scalar forward supports attention softcap, while the shared
				// panel deliberately declines this configuration before execution.
				cfg.AttnSoftcap = 2
			}
			m := model.NewSynthetic(cfg)
			s := NewNativeScheduler(m)
			s.started.Do(func() {}) // The test owns the executor; no background loop.
			rec := enginestep.New(32)
			s.recorder = rec
			t.Cleanup(func() {
				s.Close()
				s.executor.Lock()
				s.mu.Lock()
				for _, ln := range s.lanes {
					ln.finish(nil, context.Canceled)
				}
				s.mu.Unlock()
				s.executor.Unlock()
			})

			prompts := [][]int{{1, 2}, {3, 4, 5}}
			emits := 1
			if tc.solo {
				prompts, emits = prompts[:1], 4
			}
			var lanes []*schedLane
			var wantLogits [][]float32
			var wantTokens [][]int
			for i, prompt := range prompts {
				ref := m.NewSession()
				logits := ref.Prefill(prompt)
				var generated []int
				for n := 0; n < emits; n++ {
					token := argmax(logits)
					generated = append(generated, token)
					if i >= tc.terminal && (!tc.solo || n < emits-1) {
						logits = ref.Step(token)
					}
				}
				wantTokens = append(wantTokens, generated)
				wantLogits = append(wantLogits, append([]float32(nil), logits...))
				ref.Close()
				s.mu.Lock()
				ln := decodeUIDOrderLane(t, s, prompt)
				if i < tc.terminal {
					ln.emitted = genTokens - 1
				} else if tc.solo {
					ln.emitted = genTokens - emits
				}
				ln.tokens = make(chan abi.EngineToken, emits)
				s.lanes = append(s.lanes, ln)
				s.mu.Unlock()
				lanes = append(lanes, ln)
			}

			s.executor.Lock()
			s.runIteration(false)
			s.executor.Unlock()
			for i, ln := range lanes {
				if !slices.Equal(ln.gen, wantTokens[i]) {
					t.Errorf("lane %d emitted %v, want serial tokens %v", i, ln.gen, wantTokens[i])
				}
				if i >= tc.terminal && !slices.Equal(ln.logits, wantLogits[i]) {
					t.Errorf("lane %d next logits differ from serial reference", i)
				}
			}
			snap := rec.Snapshot(32, enginestep.KindDecodeStep)
			serial, batched := snap.Decode[enginestep.PathSerial], snap.Decode[enginestep.PathBatched]
			if serial.Steps != tc.serial || batched.Steps != tc.batched || serial.Tokens+batched.Tokens != tc.tokens {
				t.Fatalf("execution observations: serial=%+v batched=%+v; want steps %d/%d and %d forwarded tokens", serial, batched, tc.serial, tc.batched, tc.tokens)
			}
			if tc.unsupported {
				for _, step := range snap.Recent {
					if step.Path != enginestep.PathSerial || step.Reason != "unsupported_config" || step.Lanes != 1 {
						t.Errorf("scalar fallback lacks actual per-lane reason: %+v", step)
					}
				}
			}
		})
	}
}
