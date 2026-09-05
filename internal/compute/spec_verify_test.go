package compute

import (
	"math"
	"os"
	"strings"
	"testing"
)

// groundTruthCausalAttention is an independent reference implementation of causal multi-query attention.
// For query token qi (0 <= qi < qLen), with prefix = kvLen - qLen,
// attendable keys are 0 <= j <= prefix + qi.
func groundTruthCausalAttention(q, k, v []float32, qLen, kvLen, nH, nHkv, d int) []float32 {
	grp := nH / nHkv
	w := nHkv * d
	prefix := kvLen - qLen
	scale := float32(1.0 / math.Sqrt(float64(d)))
	out := make([]float32, qLen*nH*d)

	for qi := 0; qi < qLen; qi++ {
		attendLen := prefix + qi + 1
		for h := 0; h < nH; h++ {
			kvh := h / grp
			qh := q[(qi*nH+h)*d : (qi*nH+h+1)*d]

			scores := make([]float32, attendLen)
			for j := 0; j < attendLen; j++ {
				kj := k[j*w+kvh*d : j*w+(kvh+1)*d]
				var s float32
				for dim := 0; dim < d; dim++ {
					s += qh[dim] * kj[dim]
				}
				scores[j] = s * scale
			}

			// Softmax
			maxScore := float32(-1e30)
			for _, sc := range scores {
				if sc > maxScore {
					maxScore = sc
				}
			}
			var sumExp float32
			expScores := make([]float32, attendLen)
			for j, sc := range scores {
				expScores[j] = float32(math.Exp(float64(sc - maxScore)))
				sumExp += expScores[j]
			}
			invSum := float32(1.0) / sumExp
			for j := range expScores {
				expScores[j] *= invSum
			}

			// Weighted sum over V
			oh := out[(qi*nH+h)*d : (qi*nH+h+1)*d]
			for j := 0; j < attendLen; j++ {
				vj := v[j*w+kvh*d : j*w+(kvh+1)*d]
				weight := expScores[j]
				for dim := 0; dim < d; dim++ {
					oh[dim] += weight * vj[dim]
				}
			}
		}
	}
	return out
}

func maxAbsDelta(a, b []float32) float32 {
	var maxD float32
	for i := range a {
		d := float32(math.Abs(float64(a[i] - b[i])))
		if d > maxD {
			maxD = d
		}
	}
	return maxD
}

// TestSpecVerifyAttentionCPURef verifies cosine similarity >= 0.999 vs ground truth
// across query lengths qLen in {4, 8, 16}.
func TestSpecVerifyAttentionCPURef(t *testing.T) {
	ref := Default() // cpu-ref

	qLens := []int{4, 8, 16}
	configs := []struct {
		name     string
		nH, nHkv int
		d, kvLen int
	}{
		{"mha_d64", 8, 8, 64, 32},
		{"gqa_d64", 8, 2, 64, 64},
		{"mqa_d64", 8, 1, 64, 48},
		{"gqa_d128", 16, 4, 128, 128},
	}

	for _, qLen := range qLens {
		for _, cfg := range configs {
			t.Run(cfg.name, func(t *testing.T) {
				var s lcg = lcg(11100 + qLen*100 + cfg.nH)
				qData := randVec(&s, qLen*cfg.nH*cfg.d)
				kData := randVec(&s, cfg.kvLen*cfg.nHkv*cfg.d)
				vData := randVec(&s, cfg.kvLen*cfg.nHkv*cfg.d)

				qTen := NewF32(ref, []int{qLen, cfg.nH, cfg.d}, qData)
				kTen := NewF32(ref, []int{cfg.kvLen, cfg.nHkv, cfg.d}, kData)
				vTen := NewF32(ref, []int{cfg.kvLen, cfg.nHkv, cfg.d}, vData)
				var outTen Tensor

				err := SpecVerifyAttention(&qTen, &kTen, &vTen, &outTen, qLen, cfg.kvLen, cfg.nH, cfg.nHkv, cfg.d)
				if err != nil {
					t.Fatalf("SpecVerifyAttention failed: %v", err)
				}

				got := ref.Read(outTen)
				want := groundTruthCausalAttention(qData, kData, vData, qLen, cfg.kvLen, cfg.nH, cfg.nHkv, cfg.d)

				if len(got) != len(want) {
					t.Fatalf("len mismatch: got %d want %d", len(got), len(want))
				}

				cos := cosine(got, want)
				maxDelta := maxAbsDelta(got, want)

				if cos < 0.999 {
					t.Fatalf("qLen=%d %s: cosine %.6f < 0.999 (maxDelta=%.2e)", qLen, cfg.name, cos, maxDelta)
				}
				t.Logf("qLen=%2d %-10s: cosine=%.8f maxDelta=%.2e", qLen, cfg.name, cos, maxDelta)
			})
		}
	}
}

// mockSpecVerifyBackend wraps a Backend and counts SpecVerifyAttention invocations.
type mockSpecVerifyBackend struct {
	Backend
	calls int
}

func (m *mockSpecVerifyBackend) SpecVerifyAttention(q, k, v, out *Tensor, qLen, kvLen, nH, nHkv, d int) error {
	m.calls++
	return m.Backend.(SpecVerifyBackend).SpecVerifyAttention(q, k, v, out, qLen, kvLen, nH, nHkv, d)
}

// TestSpecVerifyAttentionMockBackend verifies that a mock backend running speculative verify attention
// executes cleanly on all platforms (including darwin/arm64 without CUDA device) and achieves cosine >= 0.999.
func TestSpecVerifyAttentionMockBackend(t *testing.T) {
	ref := Default()
	mock := &mockSpecVerifyBackend{Backend: ref}

	for _, qLen := range []int{4, 8, 16} {
		var s lcg = lcg(42 + qLen)
		nH, nHkv, d, kvLen := 8, 2, 64, 32
		qData := randVec(&s, qLen*nH*d)
		kData := randVec(&s, kvLen*nHkv*d)
		vData := randVec(&s, kvLen*nHkv*d)

		qTen := NewF32(mock, []int{qLen, nH, d}, qData)
		kTen := NewF32(mock, []int{kvLen, nHkv, d}, kData)
		vTen := NewF32(mock, []int{kvLen, nHkv, d}, vData)
		var outTen Tensor

		err := mock.SpecVerifyAttention(&qTen, &kTen, &vTen, &outTen, qLen, kvLen, nH, nHkv, d)
		if err != nil {
			t.Fatalf("mock SpecVerifyAttention failed: %v", err)
		}

		got := ref.Read(outTen)
		want := groundTruthCausalAttention(qData, kData, vData, qLen, kvLen, nH, nHkv, d)
		cos := cosine(got, want)
		if cos < 0.999 {
			t.Fatalf("mock qLen=%d: cosine %.6f < 0.999", qLen, cos)
		}
	}

	if mock.calls != 3 {
		t.Fatalf("mock calls = %d, want 3", mock.calls)
	}
}

// TestSpecVerifyAttentionCausalInvariance verifies that future KV tokens have zero influence
// on earlier speculative draft tokens.
func TestSpecVerifyAttentionCausalInvariance(t *testing.T) {
	ref := Default()
	qLen, kvLen, nH, nHkv, d := 4, 32, 4, 2, 32
	prefix := kvLen - qLen

	var s lcg = 777
	qData := randVec(&s, qLen*nH*d)
	kData := randVec(&s, kvLen*nHkv*d)
	vData := randVec(&s, kvLen*nHkv*d)

	qTen := NewF32(ref, []int{qLen, nH, d}, qData)
	kTen := NewF32(ref, []int{kvLen, nHkv, d}, kData)
	vTen := NewF32(ref, []int{kvLen, nHkv, d}, vData)
	var out1 Tensor
	if err := SpecVerifyAttention(&qTen, &kTen, &vTen, &out1, qLen, kvLen, nH, nHkv, d); err != nil {
		t.Fatalf("initial SpecVerifyAttention failed: %v", err)
	}
	res1 := ref.Read(out1)

	// Corrupt future KV tokens for token 0 (indices > prefix)
	kCorrupt := append([]float32(nil), kData...)
	vCorrupt := append([]float32(nil), vData...)
	w := nHkv * d
	for j := prefix + 1; j < kvLen; j++ {
		for i := 0; i < w; i++ {
			kCorrupt[j*w+i] += 100.0
			vCorrupt[j*w+i] += 100.0
		}
	}

	kTen2 := NewF32(ref, []int{kvLen, nHkv, d}, kCorrupt)
	vTen2 := NewF32(ref, []int{kvLen, nHkv, d}, vCorrupt)
	var out2 Tensor
	if err := SpecVerifyAttention(&qTen, &kTen2, &vTen2, &out2, qLen, kvLen, nH, nHkv, d); err != nil {
		t.Fatalf("corrupted SpecVerifyAttention failed: %v", err)
	}
	res2 := ref.Read(out2)

	// Token 0 output must be strictly bit-identical
	tok0Len := nH * d
	for i := 0; i < tok0Len; i++ {
		if res1[i] != res2[i] {
			t.Fatalf("token 0 changed when future keys were corrupted: res1[%d]=%v res2[%d]=%v", i, res1[i], i, res2[i])
		}
	}
}

// TestSpecVerifyAttentionValidation verifies input bounds and error checks.
func TestSpecVerifyAttentionValidation(t *testing.T) {
	ref := Default()
	valid := NewF32(ref, []int{4, 4, 32}, make([]float32, 4*4*32))
	var out Tensor

	// Nil arguments
	if err := SpecVerifyAttention(nil, &valid, &valid, &out, 4, 4, 4, 4, 32); err == nil {
		t.Error("expected error on nil q")
	}
	if err := SpecVerifyAttention(&valid, nil, &valid, &out, 4, 4, 4, 4, 32); err == nil {
		t.Error("expected error on nil k")
	}
	if err := SpecVerifyAttention(&valid, &valid, nil, &out, 4, 4, 4, 4, 32); err == nil {
		t.Error("expected error on nil v")
	}
	if err := SpecVerifyAttention(&valid, &valid, &valid, nil, 4, 4, 4, 4, 32); err == nil {
		t.Error("expected error on nil out")
	}

	// Invalid lengths
	if err := SpecVerifyAttention(&valid, &valid, &valid, &out, 0, 4, 4, 4, 32); err == nil {
		t.Error("expected error on qLen <= 0")
	}
	if err := SpecVerifyAttention(&valid, &valid, &valid, &out, 8, 4, 4, 4, 32); err == nil {
		t.Error("expected error on kvLen < qLen")
	}

	// Invalid heads
	if err := SpecVerifyAttention(&valid, &valid, &valid, &out, 4, 4, 5, 2, 32); err == nil {
		t.Error("expected error on nH % nHkv != 0")
	}

	// Invalid head dim
	if err := SpecVerifyAttention(&valid, &valid, &valid, &out, 4, 4, 4, 4, 0); err == nil {
		t.Error("expected error on d <= 0")
	}
}

// TestSpecVerifyAttentionCUDASourceContract inspects the CUDA kernel source and header files
// to ensure exact symbol alignment without needing a CUDA compiler.
func TestSpecVerifyAttentionCUDASourceContract(t *testing.T) {
	cuBytes, err := os.ReadFile("cuda_kernels.cu")
	if err != nil {
		t.Fatalf("failed to read cuda_kernels.cu: %v", err)
	}
	cuSrc := string(cuBytes)

	for _, symbol := range []string{
		"k_spec_verify_attention",
		"k_spec_verify_combine",
		"#define SPEC_VERIFY_BLOCK_M",
		"#define SPEC_VERIFY_NUM_SEGMENTS",
		"#define BLOCK_M SPEC_VERIFY_BLOCK_M",
		"#define NUM_SEGMENTS SPEC_VERIFY_NUM_SEGMENTS",
		"fcuda_spec_verify_attention_f32",
	} {
		if !strings.Contains(cuSrc, symbol) {
			t.Errorf("cuda_kernels.cu missing required symbol %q", symbol)
		}
	}

	hBytes, err := os.ReadFile("cuda_backend.h")
	if err != nil {
		t.Fatalf("failed to read cuda_backend.h: %v", err)
	}
	hSrc := string(hBytes)
	if !strings.Contains(hSrc, "fcuda_spec_verify_attention_f32") {
		t.Error("cuda_backend.h missing fcuda_spec_verify_attention_f32 declaration")
	}

	goBytes, err := os.ReadFile("cuda.go")
	if err != nil {
		t.Fatalf("failed to read cuda.go: %v", err)
	}
	goSrc := string(goBytes)
	if !strings.Contains(goSrc, "func (c *cudaBackend) SpecVerifyAttention") {
		t.Error("cuda.go missing SpecVerifyAttention method")
	}
}
