package compute

import (
	"flag"
	"fmt"
	"math"
	"reflect"
	"runtime"
	"testing"
)

// Untagged: a required invocation must fail, not silently run zero CUDA tests.
// Runtime selection remains the recorded compile-time flag, never this test flag.
var cudaKV16PrefillGQA4Required = flag.String("fak-cuda-kv16-prefill-gqa4-required", "", "require qualification archive (1) or separately rebuilt default archive (default)")

type cudaKV16PrefillGQA4Witness interface {
	Backend
	PromptPanelAttention(Tensor, KVStore, int, int, int, int, float32) Tensor
	Recycle()
}

// Actual Upload -> KV append -> ordinary panel attention -> checked Read.
// The reference uses rounded stored KV, not pre-narrowing F32. This witnesses
// eager kernel selection/correctness, NOT capture, model quality or latency.
// fak-test:justify why=qualification when=changed:internal/compute/cuda/kv_precision.cuh
// fak-test:runtime slow est=30s lane=optin
func TestCUDAKV16PrefillGQA4(t *testing.T) {
	mode := *cudaKV16PrefillGQA4Required
	if mode == "" {
		t.Skip("explicit device qualification not requested")
	}
	if mode != "1" && mode != "default" {
		t.Fatal("required mode must be 1 or default")
	}
	raw, ok := Lookup("cuda")
	if !ok {
		t.Fatal("required CUDA backend absent; no CPU or zero-test fallback")
	}
	be, ok := raw.(cudaKV16PrefillGQA4Witness)
	if !ok {
		t.Fatalf("required native panel consumer absent: %T", raw)
	}
	if !KernelProfileReaderRegistered() {
		t.Fatal("start with existing FAK_CUDA_TIMESTAMP_PROFILE=1; selected archive needs real trace evidence")
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	t.Cleanup(be.Recycle)
	// No new environment selector: clear and test the existing historical tile4
	// override explicitly. It must retain priority over the new build profile.
	t.Setenv("FAK_CUDA_KV16_PREFILL", "")
	shapes := []struct{ p, prefix, hd, kv, gqa int }{
		{4, 0, 128, 2, 4}, {5, 3, 65, 2, 5},
		{7, 9, 256, 4, 6}, // actual Qwen head geometry, 4+2 sibling tails
		{64, 17, 128, 1, 8}, {512, 5, 65, 1, 4},
		{4, 1, 128, 2, 1}, {5, 3, 128, 2, 2}, {7, 9, 128, 2, 3},
		{1, 5, 256, 4, 6}, {3, 2, 256, 4, 6}, {4, 1, 257, 2, 6},
	}
	for _, precision := range []KVPrecision{KVPrecisionF16, KVPrecisionBF16, KVPrecisionF32} {
		for _, shape := range shapes {
			if precision == KVPrecisionF32 && !(shape.p == 7 && shape.kv == 4) {
				continue
			}
			t.Run(fmt.Sprintf("%s-P%d-KV%d-G%d-D%d-prefix%d", precision, shape.p, shape.kv, shape.gqa, shape.hd, shape.prefix), func(t *testing.T) {
				cfg := KVConfig{NumLayers: 1, NumKVHeads: shape.kv, HeadDim: shape.hd, Precision: precision}
				kv := be.NewKV(cfg)
				defer kv.Free()
				width, heads := shape.kv*shape.hd, shape.kv*shape.gqa
				for pos := 0; pos < shape.prefix+shape.p; pos++ {
					k, v := make([]float32, width), make([]float32, width)
					for i := range k {
						// Distinct KV heads/positions catch cross-head or tail spills.
						k[i] = float32(math.Sin(float64(pos*width+i)*0.13)) * 0.4
						v[i] = float32(math.Cos(float64(pos*width+i)*0.07)) * 0.7
					}
					kd := be.Upload(NewF32(be, []int{width}, k), F32)
					vd := be.Upload(NewF32(be, []int{width}, v), F32)
					kv.AppendKV(0, kd, kd, vd, pos)
					be.Free(kd)
					be.Free(vd)
				}
				snapshot, err := SnapshotKVToHost(kv)
				if err != nil {
					t.Fatal(err)
				}
				q := make([]float32, shape.p*heads*shape.hd)
				for i := range q {
					q[i] = float32(math.Sin(float64(i)*0.19)) * 0.3
				}
				qd := be.Upload(NewF32(be, []int{shape.p, heads * shape.hd}, q), F32)
				defer be.Free(qd)
				scale := float32(1 / math.Sqrt(float64(shape.hd)))
				want := cudaKV16PrefillGQA4Reference(q, snapshot.K[0], snapshot.V[0], shape.p, shape.prefix, heads, shape.kv, shape.hd, scale)
				invoke := func(legacy bool) []float32 {
					t.Helper()
					if legacy {
						t.Setenv("FAK_CUDA_KV16_PREFILL", "tile4")
					} else {
						t.Setenv("FAK_CUDA_KV16_PREFILL", "")
					}
					before := KernelProfileSnapshot()
					y := be.PromptPanelAttention(qd, kv, 0, shape.p, shape.prefix, shape.gqa, scale)
					got := be.Read(y) // native checked completion, not enqueue proof
					be.Free(y)
					delta := KernelProfileDelta(before, KernelProfileSnapshot())
					if delta == nil || delta.Backend != "cuda" || delta.Truncated != 0 || delta.UntimedCompute != 0 {
						t.Fatalf("unusable complete native trace: %+v", delta)
					}
					selected := precision != KVPrecisionF32 && mode == "1" && !legacy && shape.p >= 4 && shape.gqa >= 4 && shape.hd <= 256
					oldTile := precision != KVPrecisionF32 && legacy && shape.p >= 4 && shape.hd <= 256
					kernel := "k_kv_attention_typed"
					gx, gy := uint32(shape.p*heads), uint32(1)
					if oldTile {
						kernel = "k_kv_attention_prompt4_typed"
						gx = uint32((shape.p + 3) / 4)
						gy = uint32(heads)
					}
					if selected {
						kernel = "k_kv_attention_prompt4_gqa4_typed"
						gx = uint32((shape.p + 3) / 4)
						gy = uint32(shape.kv * ((shape.gqa + 3) / 4))
					}
					if precision == KVPrecisionF32 {
						kernel = ""
					} // existing F32/MMA route, not this typed-KV family
					var found uint64
					for _, row := range delta.Rows {
						if kernel == "" {
							if row.Kernel == "k_kv_attention_prompt4_gqa4_typed" || row.Kernel == "k_kv_attention_prompt4_typed" || row.Kernel == "k_kv_attention_typed" {
								t.Fatalf("F32 entered typed-KV route: %+v", row)
							}
							if KernelOpClass(row.Kernel) == KernelOpAttention {
								found += row.Calls
							}
							continue
						}
						if row.Kernel == kernel {
							if row.GX != gx || row.GY != gy {
								t.Fatalf("wrong attention grid %+v want %dx%d", row, gx, gy)
							}
							found += row.Calls
						} else if row.Kernel == "k_kv_attention_prompt4_gqa4_typed" || row.Kernel == "k_kv_attention_prompt4_typed" || row.Kernel == "k_kv_attention_typed" {
							t.Fatalf("unexpected competing attention row %+v", row)
						}
					}
					if (kernel != "" && found != 1) || (kernel == "" && found == 0) {
						t.Fatalf("required kernel=%s calls=%d, archive/mode mismatch", kernel, found)
					}
					cudaKV16PrefillGQA4Check(t, got, want)
					return got
				}
				got := invoke(false)
				legacy := invoke(true)
				for i := range got {
					if math.Abs(float64(got[i])-float64(legacy[i])) > 3e-5 {
						t.Fatalf("old tile4 disagreement %d: %g / %g", i, got[i], legacy[i])
					}
				}
				qback := be.Read(qd)
				if len(qback) != len(q) {
					t.Fatal("query extent changed")
				}
				for i := range q {
					if math.Float32bits(q[i]) != math.Float32bits(qback[i]) {
						t.Fatal("query mutated")
					}
				}
				after, err := SnapshotKVToHost(kv)
				if err != nil || !reflect.DeepEqual(snapshot, after) {
					t.Fatalf("attention mutated owned KV: %v", err)
				}
			})
		}
	}
}

// Binary64 causal reference over the exact physically rounded KV snapshot.
// It deliberately does not reproduce the kernel's warp reduction/online order.
func cudaKV16PrefillGQA4Reference(q, k, v []float32, p, prefix, heads, kv, hd int, scale float32) []float64 {
	result := make([]float64, p*heads*hd)
	scores := make([]float64, prefix+p)
	for token := 0; token < p; token++ {
		for h := 0; h < heads; h++ {
			n := prefix + token + 1
			kvh := h / (heads / kv)
			maximum := math.Inf(-1)
			for pos := 0; pos < n; pos++ {
				dot := 0.0
				for d := 0; d < hd; d++ {
					dot += float64(q[(token*heads+h)*hd+d]) * float64(k[(pos*kv+kvh)*hd+d])
				}
				scores[pos] = dot * float64(scale)
				maximum = math.Max(maximum, scores[pos])
			}
			total := 0.0
			for pos := 0; pos < n; pos++ {
				scores[pos] = math.Exp(scores[pos] - maximum)
				total += scores[pos]
			}
			for pos := 0; pos < n; pos++ {
				weight := scores[pos] / total
				for d := 0; d < hd; d++ {
					result[(token*heads+h)*hd+d] += weight * float64(v[(pos*kv+kvh)*hd+d])
				}
			}
		}
	}
	return result
}

func cudaKV16PrefillGQA4Check(t *testing.T, got []float32, want []float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("output extent %d want %d", len(got), len(want))
	}
	for i, x := range got {
		y := want[i]
		if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) || math.IsNaN(y) || math.IsInf(y, 0) || math.Abs(float64(x)-y) > 3e-5 {
			t.Fatalf("rounded-KV causal reference mismatch %d: %g want %g", i, x, y)
		}
	}
}
