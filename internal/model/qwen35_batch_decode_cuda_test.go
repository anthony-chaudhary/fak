//go:build cuda

package model

import (
	"fmt"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// TestCUDAQwen35IndependentDecodeBatchMatchesSerial holds the CUDA lane-panel decode to the
// serial per-lane device decode: for B in {2,4,8} independent lanes with ragged prefixes, two
// consecutive StepBatch calls (the second reads the GDN/KV state the first wrote) must match
// each lane's own serial Session.Step logits and greedy token, with zero serial fallbacks and
// the shared-panel receipt the batch dispatch promises.
func TestCUDAQwen35IndependentDecodeBatchMatchesSerial(t *testing.T) {
	be, _ := requiredCUDAQwen35SequenceBackend(t)
	witness, ok := compute.Backend(be).(interface{ Qwen35GDNLanePanelLaunches() uint64 })
	if !ok {
		t.Fatalf("CUDA backend %T lacks the lane-panel GDN launch witness", be)
	}
	batch, ok := compute.Backend(be).(compute.Qwen35IndependentDecodeBatchBackend)
	if !ok || batch.Qwen35IndependentDecodeBatchPath() != compute.Qwen35IndependentDecodeBatchPath {
		t.Fatalf("CUDA backend %T lacks the independent decode batch path", be)
	}
	for _, lanes := range []int{2, 4, 8} {
		t.Run(fmt.Sprintf("B%d", lanes), func(t *testing.T) {
			// The CUDA prompt-attention panel admits only the Qwen3.8 source-spine head geometry.
			cfg := qwen35HybridTestCfg()
			cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim = 24, 4, 256
			cfg.PartialRotaryFactor = .25
			m := NewSynthetic(cfg)
			serial := make([]*Session, lanes)
			batched := make([]*Session, lanes)
			for lane := 0; lane < lanes; lane++ {
				var err error
				if serial[lane], err = m.NewBackendSessionChecked(be); err != nil {
					t.Fatal(err)
				}
				t.Cleanup(serial[lane].Close)
				if batched[lane], err = m.NewBackendSessionChecked(be); err != nil {
					t.Fatal(err)
				}
				t.Cleanup(batched[lane].Close)
				prefix := make([]int, lane%3+2)
				for i := range prefix {
					prefix[i] = (17*lane + 11*i + 3) % m.Cfg.VocabSize
				}
				for _, s := range []*Session{serial[lane], batched[lane]} {
					if _, used, err := s.tryQwen35SequencePrefill(prefix, true); err != nil || !used {
						t.Fatalf("lane %d prefill used=%t err=%v", lane, used, err)
					}
				}
			}
			be.Recycle()
			for step := 0; step < 2; step++ {
				ids := make([]int, lanes)
				want := make([][]float32, lanes)
				beforeKV := make([]int, lanes)
				for lane := range ids {
					ids[lane] = (41 + 7*lane + 13*step) % m.Cfg.VocabSize
					beforeKV[lane] = batched[lane].halKV.Len()
					want[lane] = append([]float32(nil), serial[lane].Step(ids[lane])...)
					be.Recycle()
				}
				fallbacks := 0
				bs := &BatchSession{M: m, Seqs: batched, ObserveSerialStep: func(string, time.Duration) { fallbacks++ }}
				launchesBefore := witness.Qwen35GDNLanePanelLaunches()
				got := bs.StepBatch(ids)
				be.Recycle()
				wantPanels, wantMACs := compute.Qwen35IndependentDecodeBatchProjectionWork(batched[0].qwen35SequencePrefillRequest([]int{ids[0]}, true), lanes)
				if fallbacks != 0 || bs.LastStepSharedPanels() != wantPanels || bs.LastStepMACs() != wantMACs {
					t.Fatalf("step %d fallbacks=%d shared_panels=%d/%d macs=%d/%d", step, fallbacks, bs.LastStepSharedPanels(), wantPanels, bs.LastStepMACs(), wantMACs)
				}
				linear := 0
				for l := 0; l < m.Cfg.NumLayers; l++ {
					if m.Cfg.isLinearAttnLayer(l) {
						linear++
					}
				}
				if launched := witness.Qwen35GDNLanePanelLaunches() - launchesBefore; launched != uint64(linear) {
					t.Fatalf("step %d lane-panel GDN launches=%d, want one per linear layer (%d)", step, launched, linear)
				}
				for lane := range got {
					stats := compareQwen35GDNVector(t, fmt.Sprintf("B%d step %d lane %d", lanes, step, lane), want[lane], got[lane])
					if argmaxF32(got[lane]) != argmaxF32(want[lane]) {
						t.Fatalf("step %d lane %d greedy token mismatch", step, lane)
					}
					if gotKV := batched[lane].halKV.Len(); gotKV != beforeKV[lane]+1 {
						t.Fatalf("step %d lane %d KV length=%d want=%d", step, lane, gotKV, beforeKV[lane]+1)
					}
					t.Logf("B%d step=%d lane=%d cosine=%.9f max_abs=%.3e", lanes, step, lane, stats.cosine, stats.maxAbs)
				}
			}
		})
	}
}
