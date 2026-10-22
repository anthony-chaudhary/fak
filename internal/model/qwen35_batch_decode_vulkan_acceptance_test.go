//go:build vulkan && (windows || linux) && cgo

package model

import (
	"errors"
	"fmt"
	"reflect"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

func newVulkanQwen35AcceptanceSessions(t *testing.T, be compute.Backend, m *Model, lanes int) ([]*Session, [][]int) {
	t.Helper()
	sessions := make([]*Session, lanes)
	prefixes := make([][]int, lanes)
	for lane := range sessions {
		var err error
		sessions[lane], err = m.NewBackendSessionChecked(be)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(sessions[lane].Close)
		prefixes[lane] = make([]int, lane%3+2)
		for i := range prefixes[lane] {
			prefixes[lane][i] = (17*lane + 11*i + 3) % m.Cfg.VocabSize
		}
		if _, used, err := sessions[lane].tryQwen35SequencePrefill(prefixes[lane], true); err != nil || !used {
			t.Fatalf("lane %d prefill used=%t err=%v", lane, used, err)
		}
	}
	return sessions, prefixes
}

func assertVulkanQwen35ContinuationEqual(t *testing.T, be compute.Backend, label string, got, want *Session) {
	t.Helper()
	if got.halKV.Len() != want.halKV.Len() || !reflect.DeepEqual(got.halKV.Pos(), want.halKV.Pos()) {
		t.Fatalf("%s KV position mismatch: len=%d/%d pos=%v/%v", label, got.halKV.Len(), want.halKV.Len(), got.halKV.Pos(), want.halKV.Pos())
	}
	if !reflect.DeepEqual(got.halLineage, want.halLineage) {
		t.Fatalf("%s token lineage mismatch: %+v/%+v", label, got.halLineage, want.halLineage)
	}
	attention := 0
	req := got.qwen35SequencePrefillRequest([]int{0}, true)
	for layer, desc := range req.Layers {
		if desc.Linear {
			assertFloat32BitsEqual(t, fmt.Sprintf("%s layer %d convolution", label, layer), be.Read(got.qwen35HAL.layers[layer].conv), be.Read(want.qwen35HAL.layers[layer].conv))
			assertFloat32BitsEqual(t, fmt.Sprintf("%s layer %d recurrent", label, layer), be.Read(got.qwen35HAL.layers[layer].recurrent), be.Read(want.qwen35HAL.layers[layer].recurrent))
			continue
		}
		assertFloat32BitsEqual(t, fmt.Sprintf("%s attention %d keys", label, attention), be.Read(got.halKV.KeysView(attention)), be.Read(want.halKV.KeysView(attention)))
		assertFloat32BitsEqual(t, fmt.Sprintf("%s attention %d values", label, attention), be.Read(got.halKV.ValuesView(attention)), be.Read(want.halKV.ValuesView(attention)))
		attention++
	}
}

// fak-test:runtime slow est=30s lane=optin
func TestVulkanQwen35IndependentDecodeBatchRaggedStateParity(t *testing.T) {
	be, _ := requiredVulkanSequenceBackend(t)
	cfg := qwen35HybridTestCfg()
	cfg.PartialRotaryFactor = .5
	m := NewSynthetic(cfg)
	const lanes = 5
	batched, _ := newVulkanQwen35AcceptanceSessions(t, be, m, lanes)
	reference, _ := newVulkanQwen35AcceptanceSessions(t, be, m, lanes)
	ids := []int{41, 48, 55, 62, 69}
	active := []bool{true, false, true, false, true}
	if recycler, ok := be.(interface{ Recycle() }); ok {
		recycler.Recycle()
	}
	fallbacks := 0
	bs := &BatchSession{M: m, Seqs: batched, ObserveSerialStep: func(string, time.Duration) { fallbacks++ }}
	got := bs.StepBatchActive(ids, active)
	firstRequest := batched[0].qwen35SequencePrefillRequest([]int{ids[0]}, true)
	wantPanels, wantMACs := qwen35VulkanBatchProjectionWork(firstRequest, 3)
	if fallbacks != 0 || bs.LastStepSharedPanels() != wantPanels || bs.LastStepMACs() != wantMACs {
		t.Fatalf("fallbacks=%d panels=%d/%d macs=%d/%d", fallbacks, bs.LastStepSharedPanels(), wantPanels, bs.LastStepMACs(), wantMACs)
	}
	for lane := range batched {
		if active[lane] {
			wantLogits := reference[lane].Step(ids[lane])
			compareVulkanSequenceVector(t, fmt.Sprintf("ragged_lane_%d", lane), got[lane], wantLogits)
			if argmaxF32(got[lane]) != argmaxF32(wantLogits) {
				t.Fatalf("lane %d greedy token mismatch", lane)
			}
		} else if got[lane] != nil {
			t.Fatalf("inactive lane %d produced logits", lane)
		}
		assertVulkanQwen35ContinuationEqual(t, be, fmt.Sprintf("lane %d", lane), batched[lane], reference[lane])
	}
}

func callVulkanQwen35BatchForTest(sessions []*Session, ids []int, edit func(*compute.Qwen35IndependentDecodeBatchRequest)) (result compute.Qwen35IndependentDecodeBatchResult, err error) {
	be := sessions[0].Backend
	batch, ok := qwen35IndependentDecodeBatchFor(be)
	if !ok {
		return result, errors.New("missing independent decode backend")
	}
	req := compute.Qwen35IndependentDecodeBatchRequest{Path: compute.Qwen35IndependentDecodeBatchPath, Lanes: make([]compute.Qwen35SequencePrefillRequest, len(sessions))}
	for i := range sessions {
		req.Lanes[i] = sessions[i].qwen35SequencePrefillRequest([]int{ids[i]}, true)
	}
	var bind func(int) error
	bind = func(i int) error {
		if i == len(sessions) {
			edit(&req)
			result, err = batch.Qwen35IndependentDecodeBatch(req)
			return err
		}
		_, bindErr := sessions[i].qwen35HAL.mutateSequence(be, func(states []compute.Qwen35SequenceState) (compute.Qwen35SequencePrefillResult, error) {
			req.Lanes[i].States = states
			return compute.Qwen35SequencePrefillResult{}, bind(i + 1)
		})
		return bindErr
	}
	err = bind(0)
	return result, err
}

// fak-test:runtime slow est=20s lane=optin
func TestVulkanQwen35IndependentDecodeBatchNegativePreflight(t *testing.T) {
	be, _ := requiredVulkanSequenceBackend(t)
	m := NewSynthetic(qwen35HybridTestCfg())
	for _, tc := range []struct {
		name string
		edit func(*compute.Qwen35IndependentDecodeBatchRequest)
	}{
		{"missing_weights", func(req *compute.Qwen35IndependentDecodeBatchRequest) { req.Lanes[1].Layers[0].Gate = compute.Tensor{} }},
		{"aliased_recurrent_state", func(req *compute.Qwen35IndependentDecodeBatchRequest) {
			for layer := range req.Lanes[0].Layers {
				if req.Lanes[0].Layers[layer].Linear {
					req.Lanes[1].States[layer] = req.Lanes[0].States[layer]
					return
				}
			}
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			sessions, _ := newVulkanQwen35AcceptanceSessions(t, be, m, 2)
			before := []int{sessions[0].halKV.Len(), sessions[1].halKV.Len()}
			_, err := callVulkanQwen35BatchForTest(sessions, []int{31, 37}, tc.edit)
			var batchErr *compute.Qwen35IndependentDecodeBatchError
			if !errors.As(err, &batchErr) || batchErr.Mutated || batchErr.Stage != "preflight" {
				t.Fatalf("error=%v, want preflight non-mutating refusal", err)
			}
			for i := range sessions {
				if sessions[i].halKV.Len() != before[i] {
					t.Fatalf("lane %d KV mutated: %d -> %d", i, before[i], sessions[i].halKV.Len())
				}
			}
		})
	}

	sessions, _ := newVulkanQwen35AcceptanceSessions(t, be, m, 2)
	bs := &BatchSession{M: NewSynthetic(qwen35HybridTestCfg()), Seqs: sessions}
	before := []int{sessions[0].halKV.Len(), sessions[1].halKV.Len()}
	if _, accepted := stepBatchQwen35HybridQ4KVulkan(bs, []int{43, 47}); accepted {
		t.Fatal("mixed model ownership admitted")
	}
	for i := range sessions {
		if sessions[i].halKV.Len() != before[i] {
			t.Fatalf("mixed ownership mutated lane %d", i)
		}
	}
}
