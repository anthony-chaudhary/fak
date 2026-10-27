//go:build cuda && cgo

package compute

import (
	"fmt"
	"math"
	"os"
	"testing"
)

// One resident witness compares the candidate to the exact retained seven-launch
// CUDA composition, including prefix/suffix sentinels and publication metadata.
// FAK_CUDA_GDN_REQUIRED=1 makes absent hardware fail rather than skip.
// fak-test:runtime integration est=5s lane=optin
func TestCUDAQwen35AttentionPostprocessMatchesComposition(t *testing.T) {
	be := cudaGDNBackend(t)
	if !cudaQwen35AttentionPostprocessEnabled {
		if os.Getenv(cudaGDNRequiredEnv) == "1" {
			t.Fatal("set FAK_CUDA_QWEN35_ATTN_POSTPROCESS=1 to witness the selected candidate")
		}
		t.Skip("experimental attention postprocess disabled")
	}
	t.Cleanup(be.Recycle)
	for _, tc := range []struct{ tokens, start, rotary int }{{1, 0, 2}, {7, 5, 64}, {128, 0, 64}, {129, 5, 256}} {
		t.Run(fmt.Sprintf("P%d-start%d-rope%d", tc.tokens, tc.start, tc.rotary), func(t *testing.T) {
			const hd, qw, kw = 256, 24 * 256, 4 * 256
			const eps, theta = float32(1e-6), float64(1e7)
			rng := qwen35PanelLCG(0x0616c0da)
			upload := func(shape []int, values []float32) Tensor {
				x := be.UploadClass(NewF32(Default(), shape, values), F32, MemoryScratchpad, "qwen35-postprocess-witness")
				t.Cleanup(func() { be.Free(x) })
				return x
			}
			q := upload([]int{tc.tokens, qw}, rng.values(tc.tokens*qw, 0.5))
			k := upload([]int{tc.tokens, kw}, rng.values(tc.tokens*kw, 0.5))
			v := upload([]int{tc.tokens, kw}, rng.values(tc.tokens*kw, 0.5))
			qn := upload([]int{hd}, rng.values(hd, 1))
			kn := upload([]int{hd}, rng.values(hd, 1))
			capacity := (tc.start + tc.tokens + 3) * kw
			sentinel := make([]float32, capacity)
			for i := range sentinel {
				sentinel[i] = -12345.75
			}
			makeKV := func() (*cudaKV, []Tensor) {
				buffers := []Tensor{upload([]int{capacity}, sentinel), upload([]int{capacity}, sentinel), upload([]int{capacity}, sentinel)}
				kv := &cudaKV{be: be, cfg: KVConfig{NumLayers: 1, NumKVHeads: 4, HeadDim: hd}, Kraw: make([]dslice, 1), K: make([]dslice, 1), V: make([]dslice, 1)}
				for i, dst := range []*dslice{&kv.Kraw[0], &kv.K[0], &kv.V[0]} {
					*dst = dslice{ptr: buffers[i].buf.(*cudaBuf).ptr, len: tc.start * kw, cap: capacity}
				}
				for pos := 0; pos < tc.start; pos++ {
					kv.pos = append(kv.pos, pos)
				}
				return kv, buffers // uploads, not kv.Free, own these test allocations
			}
			baseline, wantCache := makeKV()
			candidate, gotCache := makeKV()
			var want, got Tensor
			var launchDelta uint64
			be.ResetHostXfer()
			be.ResetH2DXfer()
			err := func() error {
				cudaMu.Lock()
				defer cudaMu.Unlock()
				qNorm, _, err := be.qwen35SequenceRMSNormLocked(q, qn, tc.tokens*24, eps, "postprocess-oracle-q")
				if err != nil {
					return err
				}
				kNorm, _, err := be.qwen35SequenceRMSNormLocked(k, kn, tc.tokens*4, eps, "postprocess-oracle-k")
				if err != nil {
					return err
				}
				var kr Tensor
				want, _, kr, _, err = be.qwen35SequenceRoPELocked(qNorm, kNorm, tc.tokens, tc.start, 24, 4, hd, tc.rotary, theta)
				if err != nil {
					return err
				}
				if err = be.qwen35SequenceAppendKVLocked(baseline, 0, kNorm, kr, v, tc.start, tc.tokens); err != nil {
					return err
				}
				// An output/output alias must refuse before any enqueue or metadata write.
				original := candidate.Kraw[0]
				candidate.Kraw[0] = candidate.K[0]
				beforeRefusal := cudaQwen35AttentionPostprocessLaunchesLocked()
				_, _, selected, aliasErr := be.qwen35SequenceAttentionPostprocessLocked(q, k, v, qn, kn, candidate, 0, tc.tokens, tc.start, tc.rotary, eps, theta)
				candidate.Kraw[0] = original
				if !selected || aliasErr == nil || cudaQwen35AttentionPostprocessLaunchesLocked() != beforeRefusal || candidate.Len() != tc.start {
					return fmt.Errorf("alias refusal failed: %v", aliasErr)
				}
				before := cudaQwen35AttentionPostprocessLaunchesLocked()
				var used bool
				got, _, used, err = be.qwen35SequenceAttentionPostprocessLocked(q, k, v, qn, kn, candidate, 0, tc.tokens, tc.start, tc.rotary, eps, theta)
				if err != nil {
					return err
				}
				if !used {
					return fmt.Errorf("candidate declined on acceptance device")
				}
				launchDelta = cudaQwen35AttentionPostprocessLaunchesLocked() - before
				return be.qwen35SequenceSyncLocked()
			}()
			if err != nil {
				t.Fatal(err)
			}
			if launchDelta != 1 {
				t.Fatalf("candidate enqueues=%d, want 1", launchDelta)
			}
			if be.HostXferBytes() != 0 || be.H2DXferBytes() != 0 {
				t.Fatal("postprocess staged through the host")
			}
			for i, pair := range [][2]Tensor{{want, got}, {wantCache[0], gotCache[0]}, {wantCache[1], gotCache[1]}, {wantCache[2], gotCache[2]}} {
				a, b := be.Read(pair[0]), be.Read(pair[1])
				if len(a) != len(b) {
					t.Fatalf("buffer %d length mismatch", i)
				}
				for j := range a {
					if math.IsNaN(float64(b[j])) || math.IsInf(float64(b[j]), 0) || math.Float32bits(a[j]) != math.Float32bits(b[j]) {
						t.Fatalf("buffer %d element %d: baseline %08x candidate %08x", i, j, math.Float32bits(a[j]), math.Float32bits(b[j]))
					}
				}
			}
			if candidate.Len() != tc.start+tc.tokens || candidate.Kraw[0].len != baseline.Kraw[0].len || candidate.K[0].len != baseline.K[0].len || candidate.V[0].len != baseline.V[0].len {
				t.Fatal("KV publication differs from composition")
			}
		})
	}
}
