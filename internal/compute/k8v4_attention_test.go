package compute

import (
	"errors"
	"math"
	"math/rand"
	"os"
	"strings"
	"testing"
)

func k8v4Cosine(a, b []float32) float64 {
	var dot, na, nb float64
	for i := range a {
		da, db := float64(a[i]), float64(b[i])
		dot += da * db
		na += da * da
		nb += db * db
	}
	den := math.Sqrt(na) * math.Sqrt(nb)
	if den == 0 {
		return 0
	}
	return dot / den
}

// fak-test:runtime fast est=1s lane=default
func TestK8V4Group32RoundTripAndSymmetricAffineKeys(t *testing.T) {
	t.Parallel()
	const kvHeads = 4
	const headDim = 128
	const tokens = 3

	cache, err := NewK8V4Cache(8, kvHeads, headDim)
	if err != nil {
		t.Fatalf("NewK8V4Cache: %v", err)
	}

	rng := rand.New(rand.NewSource(20261127))
	rowElems := tokens * kvHeads * headDim
	kSrc := make([]float32, rowElems)
	vSrc := make([]float32, rowElems)
	for i := range kSrc {
		kSrc[i] = float32(rng.NormFloat64() * 1.5)
		vSrc[i] = float32(rng.NormFloat64()*1.2 + 0.75) // shifted asymmetric V distribution
	}
	// Force row 0, kvHead 0, group 0 to exact symmetric range [-1.27, +1.27] -> zero == 0.
	for i := 0; i < K8V4GroupSize; i++ {
		if i%2 == 0 {
			kSrc[i] = 1.27
		} else {
			kSrc[i] = -1.27
		}
	}
	// Force row 0, kvHead 0, group 1 to a constant V group to verify constant preservation.
	for i := K8V4GroupSize; i < 2*K8V4GroupSize; i++ {
		vSrc[i] = 2.5
	}

	if err := QuantizeAppendK8V4(cache, kSrc, vSrc, tokens); err != nil {
		t.Fatalf("QuantizeAppendK8V4: %v", err)
	}
	if cache.Positions != tokens {
		t.Fatalf("Positions=%d, want %d", cache.Positions, tokens)
	}
	if gotZero := fp16ToFloat32(cache.KMeta[1]); gotZero != 0 {
		t.Fatalf("symmetric K group expected zero=0, got %g", gotZero)
	}

	for pos := 0; pos < tokens; pos++ {
		for kh := 0; kh < kvHeads; kh++ {
			kOut, vOut, err := DequantizeK8V4Row(cache, pos, kh)
			if err != nil {
				t.Fatalf("DequantizeK8V4Row(%d,%d): %v", pos, kh, err)
			}
			base := (pos*kvHeads + kh) * headDim
			kRef := kSrc[base : base+headDim]
			vRef := vSrc[base : base+headDim]

			if kCos := k8v4Cosine(kRef, kOut); kCos < 0.9999 {
				t.Fatalf("pos=%d kh=%d K cosine=%.6f < 0.9999", pos, kh, kCos)
			}
			if vCos := k8v4Cosine(vRef, vOut); vCos < 0.99 {
				t.Fatalf("pos=%d kh=%d V cosine=%.6f < 0.99", pos, kh, vCos)
			}

			// Verify elementwise error stays within the stored group scale bound (+ f16 rounding slack).
			rowMetaBase := (pos*kvHeads + kh) * cache.GroupsPerHead * 2
			for g := 0; g < cache.GroupsPerHead; g++ {
				kScale := float64(fp16ToFloat32(cache.KMeta[rowMetaBase+g*2]))
				vScale := float64(fp16ToFloat32(cache.VMeta[rowMetaBase+g*2]))
				for i := 0; i < K8V4GroupSize; i++ {
					idx := g*K8V4GroupSize + i
					kErr := math.Abs(float64(kRef[idx]) - float64(kOut[idx]))
					vErr := math.Abs(float64(vRef[idx]) - float64(vOut[idx]))
					if kErr > 0.5*kScale+1e-3 {
						t.Fatalf("pos=%d kh=%d g=%d i=%d kErr=%g > half-step=%g", pos, kh, g, i, kErr, 0.5*kScale)
					}
					if vErr > 0.5*vScale+1e-3 {
						t.Fatalf("pos=%d kh=%d g=%d i=%d vErr=%g > half-step=%g", pos, kh, g, i, vErr, 0.5*vScale)
					}
				}
			}
		}
	}
}

// fak-test:runtime fast est=1s lane=default
func TestK8V4DecodeAttentionHostPrefixAndVerifyParity(t *testing.T) {
	t.Parallel()
	const (
		kvHeads = 4
		heads   = 12 // GQA ratio 3
		headDim = 128
		prefix  = 14
		newToks = 3
		total   = prefix + newToks
	)
	rng := rand.New(rand.NewSource(998877))
	kAll := make([]float32, total*kvHeads*headDim)
	vAll := make([]float32, total*kvHeads*headDim)
	for i := range kAll {
		kAll[i] = float32(rng.NormFloat64() * 0.6)
		vAll[i] = float32(rng.NormFloat64() * 0.6)
	}
	q := make([]float32, newToks*heads*headDim)
	for i := range q {
		q[i] = float32(rng.NormFloat64() * 0.6)
	}

	cache, err := NewK8V4Cache(32, kvHeads, headDim)
	if err != nil {
		t.Fatalf("NewK8V4Cache: %v", err)
	}
	// Append prefix tokens first, then append newToks tokens incrementally.
	if err := QuantizeAppendK8V4(cache, kAll[:prefix*kvHeads*headDim], vAll[:prefix*kvHeads*headDim], prefix); err != nil {
		t.Fatalf("append prefix: %v", err)
	}
	if err := QuantizeAppendK8V4(cache, kAll[prefix*kvHeads*headDim:], vAll[prefix*kvHeads*headDim:], newToks); err != nil {
		t.Fatalf("append newToks: %v", err)
	}

	attnScale := float32(1.0 / math.Sqrt(float64(headDim)))
	got, err := K8V4DecodeAttentionHost(cache, q, newToks, prefix, heads, attnScale)
	if err != nil {
		t.Fatalf("K8V4DecodeAttentionHost: %v", err)
	}
	want, err := ExactAttentionF32Host(q, kAll, vAll, newToks, prefix, heads, kvHeads, headDim, attnScale)
	if err != nil {
		t.Fatalf("ExactAttentionF32Host: %v", err)
	}

	cos := k8v4Cosine(want, got)
	var maxDiff float64
	for i := range want {
		if d := math.Abs(float64(want[i]) - float64(got[i])); d > maxDiff {
			maxDiff = d
		}
	}
	if cos < 0.997 {
		t.Fatalf("K8V4 online-softmax attention cosine=%.6f < 0.997 vs exact F32", cos)
	}
	if maxDiff > 0.12 {
		t.Fatalf("K8V4 attention max abs diff=%.6f > 0.12", maxDiff)
	}
	t.Logf("[SW-VERIFIED] K8V4 online-softmax decode attention parity: cosine=%.6f maxDiff=%.6f", cos, maxDiff)
}

// fak-test:runtime fast est=1s lane=default
func TestK8V4FailClosedGeometryAndNonFiniteRefusals(t *testing.T) {
	t.Parallel()
	if _, err := NewK8V4Cache(0, 4, 128); !errors.Is(err, ErrK8V4InvalidShape) {
		t.Fatalf("zero capacity: got %v", err)
	}
	if _, err := NewK8V4Cache(8, 0, 128); !errors.Is(err, ErrK8V4InvalidShape) {
		t.Fatalf("zero kvHeads: got %v", err)
	}
	if _, err := NewK8V4Cache(8, 4, 48); !errors.Is(err, ErrK8V4InvalidShape) {
		t.Fatalf("non-32-aligned headDim: got %v", err)
	}

	cache, err := NewK8V4Cache(2, 2, 64)
	if err != nil {
		t.Fatalf("NewK8V4Cache: %v", err)
	}
	good := make([]float32, 2*64)
	if err := QuantizeAppendK8V4(cache, good, good, 1); err != nil {
		t.Fatalf("seed append: %v", err)
	}

	// Non-finite append must fail closed without mutating cache.Positions.
	badNan := append([]float32(nil), good...)
	badNan[17] = float32(math.NaN())
	if err := QuantizeAppendK8V4(cache, badNan, good, 1); !errors.Is(err, ErrK8V4NonFinite) {
		t.Fatalf("nan key append: got %v", err)
	}
	if cache.Positions != 1 {
		t.Fatalf("Positions mutated after failed append: %d", cache.Positions)
	}
	// Overflowing f16 scale must fail closed without mutating cache.Positions.
	badScale := append([]float32(nil), good...)
	badScale[0] = 1e12
	if err := QuantizeAppendK8V4(cache, badScale, good, 1); !errors.Is(err, ErrK8V4InvalidScale) {
		t.Fatalf("f16 overflow append: got %v", err)
	}
	if cache.Positions != 1 {
		t.Fatalf("Positions mutated after f16 scale overflow: %d", cache.Positions)
	}
	// Capacity overflow and mismatched slices.
	if err := QuantizeAppendK8V4(cache, make([]float32, 4*64), make([]float32, 4*64), 2); !errors.Is(err, ErrK8V4Capacity) {
		t.Fatalf("capacity overflow: got %v", err)
	}
	if err := QuantizeAppendK8V4(cache, nil, nil, 0); !errors.Is(err, ErrK8V4InvalidShape) {
		t.Fatalf("empty append: got %v", err)
	}
	if err := QuantizeAppendK8V4(cache, good[:10], good, 1); !errors.Is(err, ErrK8V4InvalidShape) {
		t.Fatalf("mismatched append: got %v", err)
	}

	// Attention fail-closed cases.
	qGood := make([]float32, 4*64)
	if _, err := K8V4DecodeAttentionHost(cache, qGood, 1, 0, 3, 0.125); !errors.Is(err, ErrK8V4InvalidShape) {
		t.Fatalf("non-divisible GQA heads: got %v", err)
	}
	if _, err := K8V4DecodeAttentionHost(cache, qGood, 1, 1, 4, 0.125); !errors.Is(err, ErrK8V4InvalidShape) {
		t.Fatalf("prefix+tokens exceeds cached positions: got %v", err)
	}
	if _, err := K8V4DecodeAttentionHost(cache, qGood, 1, 0, 4, 0); !errors.Is(err, ErrK8V4InvalidShape) {
		t.Fatalf("zero scale: got %v", err)
	}
	if _, err := K8V4DecodeAttentionHost(cache, nil, 1, 0, 4, 0.125); !errors.Is(err, ErrK8V4InvalidShape) {
		t.Fatalf("empty q: got %v", err)
	}
	qNan := append([]float32(nil), qGood...)
	qNan[5] = float32(math.Inf(1))
	if _, err := K8V4DecodeAttentionHost(cache, qNan, 1, 0, 4, 0.125); !errors.Is(err, ErrK8V4NonFinite) {
		t.Fatalf("inf q: got %v", err)
	}
	// Corrupted negative scale in cache metadata must fail closed.
	saved := cache.KMeta[0]
	cache.KMeta[0] = 0xBC00 // -1.0 in f16
	if _, err := K8V4DecodeAttentionHost(cache, qGood, 1, 0, 4, 0.125); !errors.Is(err, ErrK8V4InvalidScale) {
		t.Fatalf("negative cached f16 scale: got %v", err)
	}
	cache.KMeta[0] = saved
}

// fak-test:runtime fast est=1s lane=default
func TestK8V4LongContextTrafficReductionVsFP8(t *testing.T) {
	t.Parallel()
	rep, err := CompareK8V4VsFP8Traffic(131072, 8, 128)
	if err != nil {
		t.Fatalf("CompareK8V4VsFP8Traffic: %v", err)
	}
	// K8V4 per 32 elems: 36 B (K) + 20 B (V) = 56 B vs group-32 FP8+f32-scale: 72 B -> 16/72 = 22.22% reduction (~25%),
	// raw payload 12 bits vs 16 bits = 25.00% reduction, and strictly smaller than per-head FP8+f32-scale too.
	if rep.K8V4BytesPerToken != 8*(128+4*4+64+4*4) {
		t.Fatalf("K8V4BytesPerToken=%d, want %d", rep.K8V4BytesPerToken, 8*(128+16+64+16))
	}
	if math.Abs(rep.ReductionVsFP8Group32-(16.0/72.0)) > 1e-9 {
		t.Fatalf("ReductionVsFP8Group32=%.6f, want %.6f", rep.ReductionVsFP8Group32, 16.0/72.0)
	}
	if math.Abs(rep.PayloadReductionVsRawFP8-0.25) > 1e-9 {
		t.Fatalf("PayloadReductionVsRawFP8=%.6f, want 0.25", rep.PayloadReductionVsRawFP8)
	}
	if rep.K8V4TotalBytes >= rep.FP8PerHeadTotalBytes || rep.K8V4TotalBytes >= rep.FP8Group32TotalBytes {
		t.Fatalf("K8V4 total=%d not smaller than FP8 baselines (group32=%d perHead=%d)",
			rep.K8V4TotalBytes, rep.FP8Group32TotalBytes, rep.FP8PerHeadTotalBytes)
	}
	if _, err := K8V4KVBytesPerToken(8, 90); !errors.Is(err, ErrK8V4InvalidShape) {
		t.Fatalf("invalid group-32 headDim should fail closed, got %v", err)
	}
	t.Logf("[SW-VERIFIED] 128K ctx KV traffic (8x128): K8V4=%d B/tok vs FP8-group32=%d B/tok (%.2f%% reduction, 25.0%% raw payload reduction), FP8-per-head=%d B/tok (%.2f%% reduction)",
		rep.K8V4BytesPerToken, rep.FP8Group32BytesPerToken, rep.ReductionVsFP8Group32*100,
		rep.FP8PerHeadBytesPerToken, rep.ReductionVsFP8PerHead*100)
}

// fak-test:runtime fast est=1s lane=default
func TestK8V4CUDASourceContractUnwired(t *testing.T) {
	t.Parallel()
	b, err := os.ReadFile("cuda/kv_fp8_attention_owner.cuh")
	if err != nil {
		t.Fatal(err)
	}
	src := string(b)
	for _, want := range []string{
		"magnitudedev/magnitude",
		"attention_decode_k8v4.cu:46@8d7dbcc1ba",
		"attention_append_k8v4.cu:11@8d7dbcc1ba",
		"#ifdef __CUDACC__",
		"fcuda_kv_fp8_attention_owned",
		"k_kv_k8v4_append",
		"k_kv_k8v4_decode_attention",
		"fcuda_kv_k8v4_append_async",
		"fcuda_kv_k8v4_decode_attention_async",
		"kGroupSize = 32",
		"__float2half_rn",
	} {
		if !strings.Contains(src, want) {
			t.Fatalf("kv_fp8_attention_owner.cuh missing contract fragment %q", want)
		}
	}
	idxGuard := strings.Index(src, "#ifdef __CUDACC__")
	idxGlobal := strings.Index(src, "__global__")
	idxEnd := strings.LastIndex(src, "#endif // __CUDACC__")
	if idxGuard < 0 || idxGlobal < idxGuard || idxEnd < idxGlobal {
		t.Fatalf("k8v4 CUDA kernels must remain wrapped inside #ifdef __CUDACC__ ... #endif")
	}
	for _, fname := range []string{"cuda_kernels.cu", "cuda_backend.h"} {
		body, err := os.ReadFile(fname)
		if err != nil {
			t.Fatal(err)
		}
		text := string(body)
		for _, sym := range []string{"kv_fp8_attention_owner.cuh", "fcuda_kv_k8v4_append_async", "fcuda_kv_k8v4_decode_attention_async"} {
			if strings.Contains(text, sym) {
				t.Fatalf("%s must not wire unwired k8v4/owner symbol %q yet", fname, sym)
			}
		}
	}
}
