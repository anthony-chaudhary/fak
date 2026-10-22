//go:build vulkan && (windows || linux) && cgo

package model

import (
	"errors"
	"fmt"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type wrongQwen35IndependentDecodeBatchMarker struct{}

type qwen35BatchLifecycleBackend struct {
	*sequencePrefillBackend
	calls   int
	mutated bool
	cause   error
}

func (b *qwen35BatchLifecycleBackend) Qwen35IndependentDecodeBatchPath() string {
	return compute.Qwen35IndependentDecodeBatchPath
}

func (b *qwen35BatchLifecycleBackend) Qwen35IndependentDecodeBatch(compute.Qwen35IndependentDecodeBatchRequest) (compute.Qwen35IndependentDecodeBatchResult, error) {
	b.calls++
	return compute.Qwen35IndependentDecodeBatchResult{}, &compute.Qwen35IndependentDecodeBatchError{Stage: "injected batch", Mutated: b.mutated, Cause: b.cause}
}

func (wrongQwen35IndependentDecodeBatchMarker) Qwen35IndependentDecodeBatchPath() string {
	return "wrong"
}

func (wrongQwen35IndependentDecodeBatchMarker) Qwen35IndependentDecodeBatch(compute.Qwen35IndependentDecodeBatchRequest) (compute.Qwen35IndependentDecodeBatchResult, error) {
	panic("wrong-marker backend must be rejected before execution")
}

func TestVulkanQwen35IndependentDecodeBatchRejectsWrongCapabilityMarker(t *testing.T) {
	if _, ok := qwen35IndependentDecodeBatchFor(wrongQwen35IndependentDecodeBatchMarker{}); ok {
		t.Fatal("wrong capability marker admitted")
	}
}

func TestVulkanQwen35IndependentDecodeBatchMatchesCPU(t *testing.T) {
	be, _ := requiredVulkanSequenceBackend(t)
	batch, ok := be.(compute.Qwen35IndependentDecodeBatchBackend)
	if !ok || batch.Qwen35IndependentDecodeBatchPath() != compute.Qwen35IndependentDecodeBatchPath {
		t.Fatalf("physical Vulkan backend %T lacks independent decode batch path", be)
	}

	for _, lanes := range []int{2, 4, 8} {
		t.Run(fmt.Sprintf("B%d", lanes), func(t *testing.T) {
			cfg := qwen35HybridTestCfg()
			cfg.PartialRotaryFactor = .5
			m := NewSynthetic(cfg)
			serial := make([]*Session, lanes)
			device := make([]*Session, lanes)
			ids := make([]int, lanes)
			beforeKV := make([]int, lanes)
			want := make([][]float32, lanes)
			for lane := 0; lane < lanes; lane++ {
				serial[lane] = m.NewSession()
				var err error
				device[lane], err = m.NewBackendSessionChecked(be)
				if err != nil {
					t.Fatal(err)
				}
				t.Cleanup(device[lane].Close)
				prefix := make([]int, lane%3+2)
				for i := range prefix {
					prefix[i] = (17*lane + 11*i + 3) % cfg.VocabSize
					serial[lane].Step(prefix[i])
				}
				if _, used, err := device[lane].tryQwen35SequencePrefill(prefix, true); err != nil || !used {
					t.Fatalf("lane %d prefill used=%t err=%v", lane, used, err)
				}
				beforeKV[lane] = device[lane].halKV.Len()
				ids[lane] = (41 + 7*lane) % cfg.VocabSize
				want[lane] = append([]float32(nil), serial[lane].Step(ids[lane])...)
			}
			if recycler, ok := be.(interface{ Recycle() }); ok {
				recycler.Recycle()
			}
			for lane := 1; lane < lanes; lane++ {
				for layer := range device[0].qwen35HAL.layers {
					first, other := device[0].qwen35HAL.layers[layer], device[lane].qwen35HAL.layers[layer]
					if first.conv.Buf() != nil && (first.conv.Buf() == other.conv.Buf() || first.recurrent.Buf() == other.recurrent.Buf()) {
						t.Fatalf("lane %d layer %d aliases recurrent state", lane, layer)
					}
				}
			}

			fallbacks := 0
			bs := &BatchSession{M: m, Seqs: device, ObserveSerialStep: func(string, time.Duration) { fallbacks++ }}
			got := bs.StepBatch(ids)
			firstRequest := device[0].qwen35SequencePrefillRequest([]int{ids[0]}, true)
			wantPanels, wantMACs := qwen35VulkanBatchProjectionWork(firstRequest, lanes)
			if fallbacks != 0 || bs.LastStepSharedPanels() != wantPanels || bs.LastStepMACs() != wantMACs {
				t.Fatalf("fallbacks=%d shared_panels=%d macs=%d", fallbacks, bs.LastStepSharedPanels(), bs.LastStepMACs())
			}
			for lane := range got {
				compareVulkanSequenceVector(t, fmt.Sprintf("B%d_lane_%d", lanes, lane), got[lane], want[lane])
				if argmaxF32(got[lane]) != argmaxF32(want[lane]) {
					t.Fatalf("lane %d greedy token mismatch", lane)
				}
				if gotKV := device[lane].halKV.Len(); gotKV != beforeKV[lane]+1 {
					t.Fatalf("lane %d KV length=%d want=%d", lane, gotKV, beforeKV[lane]+1)
				}
			}
		})
	}
}

func qwen35VulkanBatchProjectionWork(req compute.Qwen35SequencePrefillRequest, lanes int) (int, int64) {
	panels := 1
	macs := int64(req.Output.Numel())
	for _, layer := range req.Layers {
		weights := []compute.Tensor{layer.Gate, layer.Up, layer.Down}
		if layer.Linear {
			weights = append(weights, layer.GDNInQKV, layer.GDNInZ, layer.GDNInB, layer.GDNInA, layer.GDNOut)
		} else {
			weights = append(weights, layer.Q, layer.K, layer.V, layer.O)
		}
		panels += len(weights)
		for _, weight := range weights {
			macs += int64(weight.Numel())
		}
	}
	return panels, int64(lanes) * macs
}

func TestQwen35HybridBatchPostMutationFailureDoesNotReplay(t *testing.T) {
	newBatch := func(t *testing.T, mutated bool) (*BatchSession, *qwen35BatchLifecycleBackend) {
		t.Helper()
		m := NewSynthetic(qwen35HybridTestCfg())
		backend := &qwen35BatchLifecycleBackend{sequencePrefillBackend: newSequencePrefillBackend(m), mutated: mutated, cause: errors.New("injected batch failure")}
		seqs := make([]*Session, 2)
		for i := range seqs {
			var err error
			seqs[i], err = m.NewBackendSessionChecked(backend)
			if err != nil {
				t.Fatalf("lane %d: %v", i, err)
			}
			t.Cleanup(seqs[i].Close)
		}
		return &BatchSession{M: m, Seqs: seqs}, backend
	}

	t.Run("unmutated remains eligible for fallback", func(t *testing.T) {
		bs, backend := newBatch(t, false)
		if _, accepted := stepBatchQwen35HybridQ4KVulkan(bs, []int{3, 7}); accepted {
			t.Fatal("unmutated failure was accepted")
		}
		if backend.calls != 1 {
			t.Fatalf("batch calls=%d, want 1", backend.calls)
		}
		for lane, s := range bs.Seqs {
			if s.BackendSessionClosed() || s.halFailure != nil {
				t.Fatalf("lane %d closed on unmutated refusal: closed=%t failure=%v", lane, s.BackendSessionClosed(), s.halFailure)
			}
		}
	})

	t.Run("mixed backend declines before capability call", func(t *testing.T) {
		m := NewSynthetic(qwen35HybridTestCfg())
		backends := []*qwen35BatchLifecycleBackend{
			{sequencePrefillBackend: newSequencePrefillBackend(m), cause: errors.New("first backend must not run")},
			{sequencePrefillBackend: newSequencePrefillBackend(m), cause: errors.New("second backend must not run")},
		}
		seqs := make([]*Session, len(backends))
		for i := range backends {
			var err error
			seqs[i], err = m.NewBackendSessionChecked(backends[i])
			if err != nil {
				t.Fatalf("lane %d: %v", i, err)
			}
			t.Cleanup(seqs[i].Close)
		}
		if _, accepted := stepBatchQwen35HybridQ4KVulkan(&BatchSession{M: m, Seqs: seqs}, []int{3, 7}); accepted {
			t.Fatal("mixed backend cohort admitted")
		}
		for lane := range seqs {
			if backends[lane].calls != 0 || seqs[lane].BackendSessionClosed() {
				t.Fatalf("lane %d calls=%d closed=%t", lane, backends[lane].calls, seqs[lane].BackendSessionClosed())
			}
		}
	})

	t.Run("mutated closes every lane and never replays", func(t *testing.T) {
		bs, backend := newBatch(t, true)
		fallbacks := 0
		bs.ObserveSerialStep = func(string, time.Duration) { fallbacks++ }
		var recovered any
		func() {
			defer func() { recovered = recover() }()
			bs.StepBatch([]int{3, 7})
		}()
		err, ok := recovered.(error)
		if !ok || !compute.Qwen35IndependentDecodeBatchMutated(err) || !errors.Is(err, backend.cause) || backend.calls != 1 || fallbacks != 0 {
			t.Fatalf("recovered=%v calls=%d fallbacks=%d", recovered, backend.calls, fallbacks)
		}
		for lane, s := range bs.Seqs {
			if !s.BackendSessionClosed() || s.halFailure != err || s.qwen35HAL != nil {
				t.Fatalf("lane %d lifecycle: closed=%t failure=%v state=%#v", lane, s.BackendSessionClosed(), s.halFailure, s.qwen35HAL)
			}
			var retry any
			func() {
				defer func() { retry = recover() }()
				s.Step(11 + lane)
			}()
			if retry != s.halFailure || backend.calls != 1 {
				t.Fatalf("lane %d retry=%v failure=%v calls=%d", lane, retry, s.halFailure, backend.calls)
			}
		}
	})
}
