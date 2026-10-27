//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"fmt"
	"math"
	"os"
	"testing"
)

// fak-test:runtime integration est=2s lane=optin
// UNRUN. Exact native composition is the oracle, not a different CPU reduction.
// This witnesses the primitive and its shared-cache ownership, not model quality.
func TestVulkanQwen35AttentionQGMatchesSeparate(t *testing.T) {
	if os.Getenv("FAK_VULKAN_QWEN35_ATTN_POSTPROCESS") != "1" || os.Getenv("FAK_VULKAN_DISPATCH_PROFILE") != "1" {
		t.Skip("requires candidate and dispatch profiling enabled at initialization")
	}
	v := vk(t)
	if !v.haveQwenAttentionQG {
		t.Fatal("requested optional module unavailable; a skip is not qualification")
	}
	if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") == "1" {
		if kind := v.v41PhysicalDeviceType(); kind != 1 && kind != 2 {
			t.Fatalf("required physical GPU not established: device type %d", kind)
		}
	}
	for _, shape := range [][4]int{{8, 0, 2, 0}, {128, 64, 17, 3}, {320, 320, 3, 5}, {1024, 1024, 2, 1}} {
		for _, scalar := range []string{"0", "1"} {
			for _, batch := range []bool{false, true} {
				t.Run(fmt.Sprintf("hd%d_rot%d_P%d_prefix%d_scalar%s_batch%v", shape[0], shape[1], shape[2], shape[3], scalar, batch), func(t *testing.T) {
					t.Setenv("FAK_VULKAN_QWEN35_PARTIAL_ROPE_SCALAR_REFERENCE", scalar)
					hd, rotary, tokens, prefix := shape[0], shape[1], shape[2], shape[3]
					const qHeads, kHeads = 4, 2
					const theta, eps = 10000.0, float32(1e-6)
					qw, kw := qHeads*hd, kHeads*hd
					data := func(n, salt int) []float32 {
						x := make([]float32, n)
						for i := range x {
							x[i] = float32((i*17+salt*11)%71-35) / 23
						}
						return x
					}
					upload := func(shape []int, values []float32) Tensor {
						x := v.Upload(NewF32(Default(), shape, values), F32)
						t.Cleanup(func() { v.Free(x) })
						return x
					}
					qData := data(tokens*qw, 1)
					for i := 0; i < hd; i++ {
						qData[i] = 0 // exact zero row, with subsequent groups reusing scratch
					}
					gateData := data(tokens*qw, 7)
					gateData[0] = math.Float32frombits(1 << 31) // copied sign bit is observable
					qgData := make([]float32, 2*tokens*qw)
					for row := 0; row < tokens*qHeads; row++ {
						copy(qgData[2*row*hd:(2*row+1)*hd], qData[row*hd:(row+1)*hd])
						copy(qgData[(2*row+1)*hd:(2*row+2)*hd], gateData[row*hd:(row+1)*hd])
					}
					qg := upload([]int{tokens, 2 * qw}, qgData)
					q, wantGate := v.SplitQwen35QueryGate(qg, qHeads, hd)
					defer v.Free(q)
					defer v.Free(wantGate)
					k := upload([]int{tokens, kw}, data(tokens*kw, 2))
					values := upload([]int{tokens, kw}, data(tokens*kw, 3))
					qn := upload([]int{hd}, data(hd, 4))
					kn := upload([]int{hd}, data(hd, 5))
					nq, nk := v.RMSNorm(q, qn, eps), v.RMSNorm(k, kn, eps)
					defer v.Free(nq)
					defer v.Free(nk)
					wantQ, wantK := v.PartialRoPEQK(nq, nk, prefix, qHeads, kHeads, hd, rotary, theta)
					defer v.Free(wantQ)
					defer v.Free(wantK)
					prefixData := data(prefix*kw, 6)
					var prefixTensor Tensor
					if prefix > 0 {
						prefixTensor = upload([]int{prefix, kw}, prefixData)
					}
					seed := v.NewKV(KVConfig{NumLayers: 1, NumKVHeads: kHeads, HeadDim: hd}).(*vulkanKV)
					defer seed.Free()
					func() {
						vulkanMu.Lock()
						defer vulkanMu.Unlock()
						for _, cache := range []*vslice{&seed.Kraw[0], &seed.K[0], &seed.V[0]} {
							v.makeVSliceWritableCapacity(cache, (prefix+tokens)*kw, (prefix+2*tokens)*kw, true, "attention QG witness")
							if prefix > 0 {
								v.growAppend(cache, v.vp(prefixTensor), prefix*kw, "attention QG witness prefix")
							}
						}
					}()
					first, second := seed.Clone().(*vulkanKV), seed.Clone().(*vulkanKV)
					defer first.Free()
					defer second.Free()
					call := func(cache *vulkanKV, inputQG Tensor) (Tensor, Tensor, bool, error) {
						vulkanMu.Lock()
						defer vulkanMu.Unlock()
						return v.tryQwen35AttentionQGLocked(inputQG, k, values, qn, kn, cache, 0, tokens, prefix, qHeads, kHeads, hd, rotary, theta, eps)
					}
					before := v.VulkanDebugDispatchProfileSnapshot()
					if batch {
						v.BeginBatch()
					}
					gotQ, gotGate, used, err := call(first, qg)
					if batch {
						v.FlushBatch()
					}
					defer v.Free(gotQ)
					defer v.Free(gotGate)
					if !used || err != nil {
						t.Fatalf("candidate not selected: %v %v", used, err)
					}
					after := v.VulkanDebugDispatchProfileSnapshot()
					if after.ComputeDispatches-before.ComputeDispatches != 1 {
						t.Fatal("QG postprocess must record exactly one compute dispatch")
					}
					if first.K[0].ptr != seed.K[0].ptr || second.K[0].ptr != seed.K[0].ptr {
						t.Fatal("first shared-frontier append should retain reserved backing")
					}
					// The second fork still points at the old checkpoint: it must detach
					// even though capacity is sufficient and both inputs happen to match.
					gotSecond, secondGate, used, err := call(second, qg)
					defer v.Free(gotSecond)
					defer v.Free(secondGate)
					if !used || err != nil || first.K[0].ptr == second.K[0].ptr || first.Kraw[0].ptr == second.Kraw[0].ptr || first.V[0].ptr == second.V[0].ptr {
						t.Fatalf("divergent fork did not detach safely: used=%v err=%v", used, err)
					}
					assertBits := func(label string, got, want []float32) {
						t.Helper()
						if len(got) != len(want) {
							t.Fatalf("%s length=%d want=%d", label, len(got), len(want))
						}
						for i := range got {
							if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
								t.Fatalf("%s[%d]=%08x want=%08x", label, i, math.Float32bits(got[i]), math.Float32bits(want[i]))
							}
						}
					}
					assertBits("Q", v.Read(gotQ), v.Read(wantQ))
					assertBits("second Q", v.Read(gotSecond), v.Read(wantQ))
					assertBits("gate", v.Read(gotGate), v.Read(wantGate))
					assertBits("second gate", v.Read(secondGate), v.Read(wantGate))
					if gotGate.buf == qg.buf || gotGate.buf == gotQ.buf || gotGate.buf == secondGate.buf {
						t.Fatal("each gate must own a distinct output allocation")
					}
					wantTail := [][]float32{v.Read(nk), v.Read(wantK), v.Read(values)}
					for _, cache := range []*vulkanKV{first, second} {
						for i, slice := range []*vslice{&cache.Kraw[0], &cache.K[0], &cache.V[0]} {
							got := func() []float32 { vulkanMu.Lock(); defer vulkanMu.Unlock(); return cache.readVS(slice) }()
							want := append(append([]float32(nil), prefixData...), wantTail[i]...)
							assertBits(fmt.Sprintf("cache%d", i), got, want)
							if slice.len != (prefix+tokens)*kw || slice.backing.highWater != slice.len {
								t.Fatal("successful append metadata mismatch")
							}
						}
					}
					for _, slice := range []*vslice{&seed.Kraw[0], &seed.K[0], &seed.V[0]} {
						got := func() []float32 { vulkanMu.Lock(); defer vulkanMu.Unlock(); return seed.readVS(slice) }()
						assertBits("seed prefix", got, prefixData)
					}
					// Forged logical metadata must fail native live-extent validation,
					// without publishing lengths or retaining the fresh Q and gate allocations.
					shortQG := upload([]int{2 * qw}, qgData[:2*qw])
					shortQG.Shape = []int{tokens, 2 * qw}
					failureCache := seed.Clone().(*vulkanKV)
					defer failureCache.Free()
					before = v.VulkanDebugDispatchProfileSnapshot()
					startTransient := len(v.transient)
					_, _, used, err = call(failureCache, shortQG)
					if used || err == nil || len(v.transient) != startTransient {
						t.Fatal("invalid native extent must fail closed and release fresh output")
					}
					for _, slice := range []*vslice{&failureCache.Kraw[0], &failureCache.K[0], &failureCache.V[0]} {
						if slice.len != prefix*kw {
							t.Fatal("failed append published a new length")
						}
					}
					wrongType := qg
					wrongType.Dtype = F16
					oldPtr := seed.K[0].ptr
					_, _, used, err = call(seed, wrongType)
					if used || err != nil || seed.K[0].ptr != oldPtr || len(v.transient) != startTransient {
						t.Fatal("unsupported type must decline mutation-free")
					}
					after = v.VulkanDebugDispatchProfileSnapshot()
					if after.ComputeDispatches != before.ComputeDispatches {
						t.Fatal("invalid/refused inputs must not dispatch compute")
					}
				})
			}
		}
	}
}
