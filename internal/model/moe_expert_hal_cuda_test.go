//go:build cuda

package model

import (
	"math"
	"os"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

const cudaExpertRequiredEnv = "FAK_CUDA_EXPERT_REQUIRED"

type cudaExpertRecordingBackend struct {
	compute.Backend
	matmuls int
	swiglu  int
	reads   int
	uploads map[compute.Dtype]int
}

func (b *cudaExpertRecordingBackend) SupportsRoutedExpertKQuant() bool { return true }
func (b *cudaExpertRecordingBackend) Upload(t compute.Tensor, as compute.Dtype) compute.Tensor {
	b.uploads[as]++
	return b.Backend.Upload(t, as)
}
func (b *cudaExpertRecordingBackend) MatMul(w, x compute.Tensor) compute.Tensor {
	b.matmuls++
	return b.Backend.MatMul(w, x)
}
func (b *cudaExpertRecordingBackend) SwiGLU(g, u compute.Tensor) compute.Tensor {
	b.swiglu++
	return b.Backend.SwiGLU(g, u)
}
func (b *cudaExpertRecordingBackend) Read(t compute.Tensor) []float32 {
	b.reads++
	return b.Backend.Read(t)
}

func cudaExpertBackend(t *testing.T) compute.Backend {
	t.Helper()
	be, ok := compute.Lookup("cuda")
	if !ok {
		if os.Getenv(cudaExpertRequiredEnv) == "1" {
			t.Fatal("cuda backend not registered while FAK_CUDA_EXPERT_REQUIRED=1")
		}
		t.Skip("cuda backend not registered (set FAK_CUDA_EXPERT_REQUIRED=1 on acceptance node to prohibit skips)")
	}
	return be
}

func TestCUDAExpertSwiGLUKQuantResidentWarmParity(t *testing.T) {
	const H = 256
	for _, tc := range []struct {
		name string
		put  func(*Model, string, int)
	}{
		{name: "Q4_K", put: func(m *Model, name string, seed int) {
			m.q4kw[name] = &q4kTensor{out: H, in: H, raw: buildRawQ4K(t, H, H, seed), nblk: 1}
		}},
		{name: "Q5_K_raw", put: func(m *Model, name string, seed int) {
			m.kqw[name] = expertHALQ5KTensor(H, H, uint64(seed))
		}},
		{name: "Q6_K_raw", put: func(m *Model, name string, seed int) {
			m.kqw[name] = expertHALQ6KTensor(H, H, int64(seed))
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := NewSyntheticMoE(expertHALTestConfig(H))
			m.q4kw = map[string]*q4kTensor{}
			m.kqw = map[string]*kQuantTensor{}
			for i, suffix := range []string{"gate_proj.weight", "up_proj.weight", "down_proj.weight"} {
				tc.put(m, expertName(0, 0, suffix), 501+i)
			}
			x := make([]float32, H)
			for i := range x {
				x[i] = float32((i%17)-8) / 128
			}
			ref := &Session{M: m, Backend: compute.Default(), Q4K: true, halW: map[string]compute.Tensor{}}
			want := expertSwiGLU(m, 0, 0, x, sessionQ4KKernel{s: ref})

			be := &cudaExpertRecordingBackend{Backend: cudaExpertBackend(t), uploads: map[compute.Dtype]int{}}
			s := &Session{M: m, Backend: be, Q4K: true, halW: map[string]compute.Tensor{}}
			got := expertSwiGLU(m, 0, 0, x, sessionQ4KKernel{s: s})
			firstQ4, firstQ5, firstQ6 := be.uploads[compute.Q4_K], be.uploads[compute.Q5_K], be.uploads[compute.Q6_K]
			if be.matmuls != 3 || be.swiglu != 1 || be.reads != 1 {
				t.Fatalf("cold device ops matmul=%d swiglu=%d reads=%d, want 3/1/1", be.matmuls, be.swiglu, be.reads)
			}
			if firstQ4+firstQ5+firstQ6 != 3 || be.uploads[compute.F16] != 0 {
				t.Fatalf("cold resident raw uploads q4=%d q5=%d q6=%d f16=%d, want raw total 3 and f16 0", firstQ4, firstQ5, firstQ6, be.uploads[compute.F16])
			}

			warm := expertSwiGLU(m, 0, 0, x, sessionQ4KKernel{s: s})
			if be.uploads[compute.Q4_K] != firstQ4 || be.uploads[compute.Q5_K] != firstQ5 || be.uploads[compute.Q6_K] != firstQ6 {
				t.Fatalf("warm token reuploaded expert weights: q4 %d->%d q5 %d->%d q6 %d->%d", firstQ4, be.uploads[compute.Q4_K], firstQ5, be.uploads[compute.Q5_K], firstQ6, be.uploads[compute.Q6_K])
			}
			if be.matmuls != 6 || be.swiglu != 2 || be.reads != 2 {
				t.Fatalf("two-token device ops matmul=%d swiglu=%d reads=%d, want 6/2/2", be.matmuls, be.swiglu, be.reads)
			}
			for label, out := range map[string][]float32{"cold": got, "warm": warm} {
				var dot, ng, nw float64
				for i := range want {
					dot += float64(out[i]) * float64(want[i])
					ng += float64(out[i]) * float64(out[i])
					nw += float64(want[i]) * float64(want[i])
				}
				cos := dot / (math.Sqrt(ng)*math.Sqrt(nw) + 1e-30)
				const floor = 0.997
				if math.IsNaN(cos) || math.IsInf(cos, 0) || cos < floor {
					t.Fatalf("%s %s CUDA expert cosine %.8f < %.3f", tc.name, label, cos, floor)
				}
				t.Logf("%s %s: cosine=%.8f resident_uploads(q4=%d q5=%d q6=%d) ops(matmul=%d swiglu=%d) final_d2h=%d", tc.name, label, cos, firstQ4, firstQ5, firstQ6, be.matmuls, be.swiglu, be.reads)
			}
			s.Close()
		})
	}
}
