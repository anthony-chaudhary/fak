//go:build cuda && cgo

package compute

import (
	"fmt"
	"math"
	"os"
	"reflect"
	"testing"
)

// This bounds F32 reduction-order drift, not a BF16/FP16 rounding budget.
// Whole-model greedy/logit acceptance remains separate and unchanged.
const qwen35VLLMFusedQKAbsRelTolerance = 2e-6

// fak-test:runtime integration est=5s lane=optin
func TestCUDAQwen35VLLMFusedQKMatchesComposition(t *testing.T) {
	be := cudaGDNBackend(t)
	if !cudaQwen35VLLMQKEnabled {
		if os.Getenv(cudaGDNRequiredEnv) == "1" {
			t.Fatal("required witness needs FAK_CUDA_QWEN35_VLLM_QK=1")
		}
		t.Skip("vLLM-derived QK candidate disabled")
	}
	for _, tc := range []struct{ hd, rotary, heads, kvheads int }{
		{64, 0, 6, 2}, {128, 8, 6, 2}, {192, 128, 6, 2},
		{192, 192, 6, 2}, {256, 64, 24, 4}, {256, 256, 24, 4},
	} {
		t.Run(fmt.Sprintf("head%d-rope%d", tc.hd, tc.rotary), func(t *testing.T) {
			t.Cleanup(be.Recycle)
			const row, position, capacityRows = 3, 19, 6
			const eps, theta = float32(3e-5), float64(1e7)
			width := tc.kvheads * tc.hd
			upload := func(data []float32) Tensor {
				out := be.UploadClass(NewF32(Default(), []int{len(data)}, data), F32, MemoryScratchpad, "vllm-qk-witness")
				t.Cleanup(func() { be.Free(out) })
				return out
			}
			values := func(n int, phase float64) []float32 {
				out := make([]float32, n)
				for i := range out {
					out[i] = float32(0.7*math.Sin(float64(i)*0.17+phase) + 0.1)
				}
				return out
			}
			qHost, kHost, vHost := values(tc.heads*tc.hd, 0), values(width, 1), values(width, 2)
			q, key, value := upload(qHost), upload(kHost), upload(vHost)
			qw, kw := upload(values(tc.hd, 3)), upload(values(tc.hd, 4))
			qNorm, kNorm := be.RMSNorm(q, qw, eps), be.RMSNorm(key, kw, eps)
			wantQ, wantK := be.PartialRoPEQK(qNorm, kNorm, position, tc.heads, tc.kvheads, tc.hd, tc.rotary, theta)
			if tc.rotary == tc.hd {
				wantQ = be.RoPE(qNorm, position, tc.heads, tc.hd, theta)
				wantK = be.RoPE(kNorm, position, tc.kvheads, tc.hd, theta)
			}
			sentinel := make([]float32, capacityRows*width)
			for i := range sentinel {
				sentinel[i] = -12345.75
			}
			buffers := []Tensor{upload(sentinel), upload(sentinel), upload(sentinel)}
			kv := &cudaKV{be: be, cfg: KVConfig{NumLayers: 1, NumKVHeads: tc.kvheads, HeadDim: tc.hd},
				Kraw: make([]dslice, 1), K: make([]dslice, 1), V: make([]dslice, 1), pos: []int{2, 4, 6}}
			for i, plane := range []*dslice{&kv.Kraw[0], &kv.K[0], &kv.V[0]} {
				*plane = dslice{ptr: buffers[i].buf.(*cudaBuf).ptr, len: row * width, cap: len(sentinel), dtype: F32}
			}
			before := cudaQwen35VLLMQKLaunches()
			original := kv.Kraw[0]
			kv.Kraw[0] = kv.K[0]
			_, selected, err := kv.AppendQKNormRoPE(q, key, value, qw, kw, 0, position, tc.heads, tc.rotary, eps, theta)
			kv.Kraw[0] = original
			if !selected || err == nil || cudaQwen35VLLMQKLaunches() != before || kv.Len() != row {
				t.Fatalf("output alias did not refuse before enqueue/publication: selected=%v err=%v", selected, err)
			}
			capacity := kv.K[0].cap
			kv.K[0].cap = kv.K[0].len
			_, selected, err = kv.AppendQKNormRoPE(q, key, value, qw, kw, 0, position, tc.heads, tc.rotary, eps, theta)
			kv.K[0].cap = capacity
			if selected || err != nil || cudaQwen35VLLMQKLaunches() != before || kv.Len() != row {
				t.Fatalf("growth boundary was not an effect-free decline: selected=%v err=%v", selected, err)
			}
			be.ResetHostXfer()
			be.ResetH2DXfer()
			gotQ, selected, err := kv.AppendQKNormRoPE(q, key, value, qw, kw, 0, position, tc.heads, tc.rotary, eps, theta)
			if err != nil || !selected {
				t.Fatalf("candidate not selected on witness device: selected=%v err=%v", selected, err)
			}
			if cudaQwen35VLLMQKLaunches()-before != 1 || be.HostXferBytes() != 0 || be.H2DXferBytes() != 0 {
				t.Fatal("candidate did not use exactly one enqueue and zero activation transfers")
			}
			compare := func(label string, got, want []float32) {
				t.Helper()
				if len(got) != len(want) {
					t.Fatalf("%s lengths %d/%d", label, len(got), len(want))
				}
				for i, value := range got {
					delta := math.Abs(float64(value) - float64(want[i]))
					limit := qwen35VLLMFusedQKAbsRelTolerance * (1 + math.Abs(float64(want[i])))
					if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) || delta > limit {
						t.Fatalf("%s[%d]=%g want=%g delta=%g limit=%g", label, i, value, want[i], delta, limit)
					}
				}
			}
			compare("query", be.Read(gotQ), be.Read(wantQ)) // Checked ordinary read owns the completion fence.
			wants := [][]float32{be.Read(kNorm), be.Read(wantK), vHost}
			for plane, tensor := range buffers {
				got := be.Read(tensor)
				compare(fmt.Sprintf("cache%d", plane), got[row*width:(row+1)*width], wants[plane])
				for i, x := range got {
					if (i < row*width || i >= (row+1)*width) && math.Float32bits(x) != math.Float32bits(sentinel[i]) {
						t.Fatalf("cache%d sentinel %d changed", plane, i)
					}
				}
				if plane == 2 && !reflect.DeepEqual(got[row*width:(row+1)*width], vHost) {
					t.Fatal("value copy changed bits")
				}
			}
			for i, tensor := range []Tensor{q, key, value} {
				if !reflect.DeepEqual(be.Read(tensor), [][]float32{qHost, kHost, vHost}[i]) {
					t.Fatalf("input %d mutated", i)
				}
			}
			if !reflect.DeepEqual(kv.Pos(), []int{2, 4, 6, position}) ||
				kv.Kraw[0].len != (row+1)*width || kv.K[0].len != (row+1)*width || kv.V[0].len != (row+1)*width {
				t.Fatal("absolute position or exact layer frontier publication changed")
			}
		})
	}
}
