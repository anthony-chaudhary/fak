package compute

import (
	"math"
	"os"
	"strings"
	"testing"
)

// fak-test:runtime fast est=2ms lane=default
func TestV41IndexerScoreContract(t *testing.T) {
	t.Parallel()
	q := Tensor{Dtype: F32, Layout: RowMajor, Shape: []int{2, 3}}
	k := Tensor{Dtype: F32, Layout: RowMajor, Shape: []int{4, 3}}
	w := Tensor{Dtype: F32, Layout: RowMajor, Shape: []int{2}}
	if qb, kb, wb, ob, err := validateV41IndexerScore(q, k, w, 4, 2, 3); err != nil || qb != 24 || kb != 48 || wb != 8 || ob != 16 {
		t.Fatalf("valid geometry bytes=%d/%d/%d/%d err=%v", qb, kb, wb, ob, err)
	}
	k.Shape = []int{0, 3}
	if _, kb, _, ob, err := validateV41IndexerScore(q, k, w, 0, 2, 3); err != nil || kb != 0 || ob != 0 {
		t.Fatalf("empty geometry bytes=%d/%d err=%v", kb, ob, err)
	}
	for _, dims := range [][3]int{{-1, 2, 3}, {4, 0, 3}, {4, 2, 0}, {1<<31 - 1, 2, 3}, {4, 1<<31 - 1, 3}} {
		if _, _, _, _, err := validateV41IndexerScore(q, k, w, dims[0], dims[1], dims[2]); err == nil {
			t.Errorf("accepted malformed/overflow geometry %v", dims)
		}
	}
	k.Shape = []int{4, 3}
	for _, mutate := range []func(*Tensor){
		func(x *Tensor) { x.Shape = []int{6} },
		func(x *Tensor) { x.Dtype = BF16 },
		func(x *Tensor) { x.Layout = ColMajor },
		func(x *Tensor) { x.Quant = &QuantSpec{} },
	} {
		bad := q
		mutate(&bad)
		if _, _, _, _, err := validateV41IndexerScore(bad, k, w, 4, 2, 3); err == nil {
			t.Errorf("accepted malformed Q %+v", bad)
		}
	}
	for _, tc := range v41IndexerScoreFixtures() {
		got := v41IndexerScoreIndependentOracle(tc.q, tc.keys, tc.weights, tc.heads, tc.dim)
		v41IndexerScoreCompareBits(t, got, tc.want)
	}
	// This control would produce a positive residual under contraction, while
	// explicit binary32 rounding produces zero before ReLU. Cosine cannot prove it.
	a, c := math.Float32frombits(0x3f800001), math.Float32frombits(0x3f7ffffe)
	if fused := float32(1 - float64(a)*float64(c)); fused <= 0 {
		t.Fatalf("bad FMA-distinguishing control %g", fused)
	}
}

type v41IndexerScoreFixture struct {
	name                   string
	heads, dim             int
	q, keys, weights, want []float32
}

func v41IndexerScoreFixtures() []v41IndexerScoreFixture {
	a, c := math.Float32frombits(0x3f800001), math.Float32frombits(0x3f7ffffe)
	inf, nan := float32(math.Inf(1)), float32(math.NaN())
	return []v41IndexerScoreFixture{
		{"signed-heads", 2, 1, []float32{1, -1}, []float32{2, -2}, []float32{1, -2}, []float32{2, -4}},
		{"relu-before-weight", 2, 1, []float32{1, -1}, []float32{2}, []float32{-1, 1}, []float32{-2}},
		{"dimension-order", 1, 3, []float32{1 << 24, 1, -(1 << 24)}, []float32{1, 1, 1}, []float32{1}, []float32{0}},
		{"round-nearest-even", 1, 3, []float32{1 << 24, 3, -(1 << 24)}, []float32{1, 1, 1}, []float32{1}, []float32{4}},
		{"head-order", 3, 1, []float32{1 << 24, 1, 1 << 24}, []float32{1}, []float32{1, 1, -1}, []float32{0}},
		{"separate-products", 1, 2, []float32{1, -a}, []float32{1, c}, []float32{1}, []float32{0}},
		{"separate-head-products", 2, 1, []float32{1, a}, []float32{1}, []float32{-1, c}, []float32{0}},
		{"smallest-subnormal", 1, 1, []float32{math.SmallestNonzeroFloat32}, []float32{1}, []float32{1}, []float32{math.SmallestNonzeroFloat32}},
		{"subnormal-product", 1, 1, []float32{math.Float32frombits(0x00800000)}, []float32{0.5}, []float32{1}, []float32{math.Float32frombits(0x00400000)}},
		{"positive-overflow", 1, 1, []float32{math.MaxFloat32}, []float32{2}, []float32{1}, []float32{inf}},
		{"negative-weight-overflow", 1, 1, []float32{math.MaxFloat32}, []float32{2}, []float32{-1}, []float32{-inf}},
		{"negative-dot-overflow", 1, 1, []float32{-math.MaxFloat32}, []float32{2}, []float32{1}, []float32{0}},
		{"overflow-times-zero", 1, 1, []float32{math.MaxFloat32}, []float32{2}, []float32{0}, []float32{nan}},
		{"opposing-infinities", 2, 1, []float32{math.MaxFloat32, math.MaxFloat32}, []float32{2}, []float32{1, -1}, []float32{nan}},
		{"signed-zero-input", 1, 1, []float32{math.Float32frombits(0x80000000)}, []float32{1}, []float32{-1}, []float32{0}},
	}
}

// Test-only oracle: products of binary32 operands fit exactly in binary64;
// explicit round trips pin each binary32 product and each binary32 sum. It does
// not call the production model scorer, which still needs separate FMA review.
func v41IndexerScoreIndependentOracle(q, keys, weights []float32, heads, dim int) []float32 {
	out := make([]float32, len(keys)/dim)
	for row := range out {
		var score float32
		for h := 0; h < heads; h++ {
			var dot float32
			for d := 0; d < dim; d++ {
				product := float32(float64(q[h*dim+d]) * float64(keys[row*dim+d]))
				dot = float32(float64(dot) + float64(product))
			}
			if dot < 0 {
				dot = 0
			}
			weighted := float32(float64(dot) * float64(weights[h]))
			score = float32(float64(score) + float64(weighted))
		}
		out[row] = score
	}
	return out
}

func v41IndexerScoreCompareBits(t *testing.T, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("score shape %d, want %d", len(got), len(want))
	}
	for i, expected := range want {
		if math.IsNaN(float64(expected)) {
			if !math.IsNaN(float64(got[i])) {
				t.Fatalf("score[%d]=%g, want arithmetic NaN", i, got[i])
			}
			continue // the float-controls specification does not preserve NaN payload bits
		}
		if math.Float32bits(got[i]) != math.Float32bits(expected) {
			t.Fatalf("score[%d]=%08x want=%08x", i, math.Float32bits(got[i]), math.Float32bits(expected))
		}
	}
}

// Deferred test SOURCE only. This guard is not compiler, module or GPU evidence.
// fak-test:runtime fast est=5ms lane=default
func TestV41IndexerScoreSourceContract(t *testing.T) {
	t.Parallel()
	read := func(path string) string {
		b, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		return string(b)
	}
	shader := read("shaders/v41_indexer_score.comp")
	for _, token := range []string{
		"#extension GL_EXT_spirv_intrinsics : require", "capabilities = [4464], 4459, 32", "capabilities = [4466], 4461, 32", "capabilities = [4467], 4462, 32",
		"layout(local_size_x = 1) in;", "for (int h = 0; h < pc.heads; ++h)", "for (int d = 0; d < pc.headDim; ++d)",
		"precise float product =", "precise float nextDot = dot + product;", "if (dot < 0.0) dot = 0.0;", "precise float weighted = dot * Weights[h];", "precise float nextScore = score + weighted;",
	} {
		if !strings.Contains(shader, token) {
			t.Errorf("missing shader contract %q", token)
		}
	}
	for _, token := range []string{"fma(", "dot(", "float16_t", "double", "isinf(", "isnan(", "subgroup", "max("} {
		if strings.Contains(shader, token) {
			t.Errorf("forbidden shader behavior %q", token)
		}
	}
	shim := read("vulkan_shim.cpp")
	for _, token := range []string{
		"VkPhysicalDeviceFloatControlsProperties floatControls", "floatControls.shaderDenormPreserveFloat32 != VK_TRUE", "floatControls.shaderSignedZeroInfNanPreserveFloat32 != VK_TRUE", "floatControls.shaderRoundingModeRTEFloat32 != VK_TRUE",
		"!v41IndexerScoreABI(code)", "modes != 7 || capabilities != 15", "floatResults.size() != 4", "!decorated(result.second, noContraction, 0)", "modeEntry != entry",
		"if (rows == 0) return 0;", "if (aliases(bufs[3], bufs[i])) return 2;", "g_kern[K_V41_INDEXER_SCORE], bufs", "counts[i] * sizeof(float) != bufs[i]->bytes",
	} {
		if !strings.Contains(shim, token) {
			t.Errorf("missing native contract %q", token)
		}
	}
	adapter := read("vulkan_v41_indexer_score.go")
	for _, token := range []string{"C.fvk_batch_flush_status()", "outBuf.v41CheckedRead = true", "C.fvk_free(outBuf.ptr)", "ClassifyVulkanPanic", "C.fvk_v41_indexer_score_unavailable_reason()"} {
		if !strings.Contains(adapter, token) {
			t.Errorf("missing adapter contract %q", token)
		}
	}
}
