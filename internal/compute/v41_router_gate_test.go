package compute

import (
	"math"
	"os"
	"regexp"
	"strings"
	"testing"
)

// fak-test:runtime fast est=1ms lane=default
func TestV41RouterGateTopKContract(t *testing.T) {
	t.Parallel()
	weight := Tensor{Dtype: F32, Layout: RowMajor, Shape: []int{384, 5120}}
	x := Tensor{Dtype: F32, Layout: RowMajor, Shape: []int{5120}}
	bias := Tensor{Dtype: F32, Layout: RowMajor, Shape: []int{384}}
	dummy := Tensor{Dtype: F32, Layout: RowMajor, Shape: []int{1}}
	wb, xb, bb, lb, pb, err := validateV41RouterGateTopK(weight, x, bias, 384, 5120, 6, 1.5, true)
	if err != nil || wb != 384*5120*4 || xb != 5120*4 || bb != 384*4 || lb != 384*4 || pb != 6*4 {
		t.Fatalf("V4.1 geometry: bytes=%d/%d/%d/%d/%d err=%v", wb, xb, bb, lb, pb, err)
	}
	if _, _, bb, _, _, err := validateV41RouterGateTopK(weight, x, dummy, 384, 5120, 6, 1.5, false); err != nil || bb != 4 {
		t.Fatalf("bias-free route must take a [1] dummy: bytes=%d err=%v", bb, err)
	}
	for _, dims := range [][3]int{
		{0, 5120, 6}, {384, 0, 6}, {384, 5120, 0}, {4, 5120, 6}, {384, 5120, V41RouterTopKMaxK + 1},
		{V41RouterGateTopKMaxExperts + 1, 5120, 6}, {384, V41RouterGateTopKMaxHidden + 1, 6},
	} {
		w := weight
		w.Shape = []int{dims[0], dims[1]}
		in := x
		in.Shape = []int{dims[1]}
		b := bias
		b.Shape = []int{dims[0]}
		if _, _, _, _, _, err := validateV41RouterGateTopK(w, in, b, dims[0], dims[1], dims[2], 1.5, true); err == nil {
			t.Errorf("accepted invalid geometry %v", dims)
		}
	}
	for _, scale := range []float32{0, -1, float32(math.Inf(1)), float32(math.NaN())} {
		if _, _, _, _, _, err := validateV41RouterGateTopK(weight, x, bias, 384, 5120, 6, scale, true); err == nil {
			t.Errorf("accepted route scale %g", scale)
		}
	}
	for name, mutate := range map[string]func(w, x, b *Tensor){
		"weight-transposed": func(w, _, _ *Tensor) { w.Shape = []int{5120, 384} },
		"weight-bf16":       func(w, _, _ *Tensor) { w.Dtype = BF16 },
		"weight-quantized":  func(w, _, _ *Tensor) { w.Quant = &QuantSpec{} },
		"x-panel":           func(_, x, _ *Tensor) { x.Shape = []int{1, 5120} },
		"x-colmajor":        func(_, x, _ *Tensor) { x.Layout = ColMajor },
		"bias-dummy":        func(_, _, b *Tensor) { b.Shape = []int{1} },
	} {
		w, in, b := weight, x, bias
		mutate(&w, &in, &b)
		if _, _, _, _, _, err := validateV41RouterGateTopK(w, in, b, 384, 5120, 6, 1.5, true); err == nil {
			t.Errorf("%s: accepted malformed operand", name)
		}
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestV41RouterGateGroups(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct{ experts, hidden, want int }{
		{384, 5120, 32}, // five chunks per logit, twelve logits per group
		{72, 3*1024 + 40, 5},
		{1024, 1024, 16},
		{16, 64 * 1024, 16},
		{1, 7, 1},
	} {
		if got := v41RouterGateGroups(tc.experts, tc.hidden); got != tc.want {
			t.Errorf("groups(%d,%d)=%d, want %d", tc.experts, tc.hidden, got, tc.want)
		}
	}
	if v41RouterGateStatusSeed != [4]uint32{^uint32(0), 0, 0, ^uint32(0)} || V41RouterTopKStage(v41RouterGateStatusSeed[0]) <= V41RouterTopKStageWeight {
		t.Fatal("status seed must be an invalid stage with a wrapping arrival counter")
	}
}

// The tie fixture's leading picks follow the host router's lower-expert-first
// rule before any device sees it.
// fak-test:runtime fast est=50ms lane=default
func TestV41RouterGateProbeFixtureTies(t *testing.T) {
	t.Parallel()
	const experts, hidden = 72, 3*1024 + 40
	w, x, bias := v41RouterGateProbeFixture(experts, hidden)
	logits := make([]float64, experts)
	best := math.Inf(-1)
	for o := range logits {
		for i := 0; i < hidden; i++ {
			logits[o] += float64(w[o*hidden+i]) * float64(x[i])
		}
		if o%32 != 5 {
			best = math.Max(best, logits[o]+float64(bias[o]))
		}
	}
	if logits[5] != logits[37] || logits[37] != logits[69] || bias[5] != bias[69] || logits[5] <= best+1 {
		t.Fatalf("tie rows must lead by a margin: tie=%g next=%g", logits[5], best)
	}
}

// orderKeyMirror is the shader's orderKey on the host.
func orderKeyMirror(value float32) uint32 {
	bits := math.Float32bits(value)
	if value == 0 {
		bits = 0
	}
	if bits&0x80000000 != 0 {
		return ^bits
	}
	return bits | 0x80000000
}

// The round's subgroup max over order keys, then min expert among equal keys,
// picks exactly the router's ranksBefore maximum: the key is strictly
// monotonic over finite selection values and joins only the two zeros.
// fak-test:runtime fast est=20ms lane=default
func TestV41RouterGateOrderKey(t *testing.T) {
	t.Parallel()
	values := []float32{float32(math.Inf(-1)), -math.MaxFloat32, -2, -1, -math.SmallestNonzeroFloat32, float32(math.Copysign(0, -1)), 0,
		math.SmallestNonzeroFloat32, 1e-38, 0.5, 1, 1.0000001, 3, math.MaxFloat32}
	for i := 0; i < 4096; i++ {
		values = append(values, decodeSplitValueHost(i)*float32(math.Pow(2, float64(i%60-30))))
	}
	for _, a := range values {
		for _, b := range values {
			ka, kb := orderKeyMirror(a), orderKeyMirror(b)
			if (a > b) != (ka > kb) || (a == b) != (ka == kb) {
				t.Fatalf("order key disagrees: %g (%#08x) vs %g (%#08x)", a, ka, b, kb)
			}
		}
	}
	if orderKeyMirror(-math.MaxFloat32) == 0 {
		t.Fatal("finite keys must stay above the empty-lane key 0")
	}
}

// decodeSplitValueHost mirrors the device witnesses' mixed-sign value.
func decodeSplitValueHost(i int) float32 {
	h := uint32(i)*2654435761 + uint32(104729)*40503
	h ^= h >> 15
	h *= 2246822519
	h ^= h >> 13
	return float32(int32(h>>8)-(1<<23)) / float32(1<<23)
}

func readShaderSource(t *testing.T, name string) string {
	t.Helper()
	raw, err := os.ReadFile("shaders/" + name)
	if err != nil {
		t.Fatal(err)
	}
	return strings.ReplaceAll(string(raw), "\r\n", "\n")
}

// glslFunction returns one function's source by its signature line.
func glslFunction(t *testing.T, src, signature string) string {
	t.Helper()
	start := strings.Index(src, signature)
	if start < 0 {
		t.Fatalf("missing %q", signature)
	}
	end := strings.Index(src[start:], "\n}\n")
	if end < 0 {
		t.Fatalf("unterminated %q", signature)
	}
	return src[start : start+end+3]
}

// The fused module keeps the decode chunk sum of matmul.comp and the scoring,
// tie and normalization arithmetic of v41_router_topk.comp character for
// character, which is what keeps it bit-identical on any one compiler.
// Source witness only: the device witness compares the bits.
// fak-test:runtime fast est=5ms lane=default
func TestV41RouterGateSourceContract(t *testing.T) {
	t.Parallel()
	fused := readShaderSource(t, "v41_router_gate_topk.comp")
	matmul := readShaderSource(t, "matmul.comp")
	router := readShaderSource(t, "v41_router_topk.comp")
	decode := glslFunction(t, matmul, "void decodeChunks() {")
	loopStart := strings.Index(decode, "        uint c0 = c * SHARED_CAP;")
	loopEnd := strings.Index(decode, "        Part[tid] = acc;")
	if loopStart < 0 || loopEnd < loopStart || !strings.Contains(fused, decode[loopStart:loopEnd]) {
		t.Fatal("fused projection lost matmul.comp's decode chunk sum")
	}
	if !strings.Contains(fused, "        float y = 0.0;\n        for (uint k = 0u; k < chunks; ++k) y += Part[tid + k];\n") {
		t.Fatal("fused projection lost the chunk-order partial sum onto 0.0")
	}
	for _, signature := range []string{"float log1pUnit(float e) {", "float routerScore(float z) {", "bool finiteValue(float x) {"} {
		if glslFunction(t, fused, signature) != glslFunction(t, router, signature) {
			t.Errorf("%s drifted from v41_router_topk.comp", signature)
		}
	}
	scoring := regexp.MustCompile(`(?s)        float z = Logits\[[^\n]*\n(.*?)\n        \}\n`).FindStringSubmatch(router)
	if scoring == nil || !strings.Contains(fused, strings.Replace(scoring[1], "float b = Bias[e];", "float b = sScoreOf[e];", 1)) {
		t.Fatal("fused scoring and fault order drifted from the router")
	}
	weights := strings.NewReplacer("rowScores[slot]", "sPickScore[slot]", "rowPicked[slot]", "sPick[slot]", "fault(row, ", "fault(",
		"Experts[pickBase + uint(slot)]", "Experts[uint(slot)]", "Weights[pickBase + uint(slot)]", "Weights[uint(slot)]")
	tail := router[strings.Index(router, "    precise float sum = 0.0;"):]
	tail = tail[:strings.Index(tail, "\n}\n")]
	if !strings.Contains(fused, weights.Replace(tail)) {
		t.Fatal("fused normalization and weights drifted from the router")
	}
	if !strings.Contains(router, "return value > otherValue || (value == otherValue && expert < otherExpert);") {
		t.Fatal("the router's selection order changed; the fused order key encodes the old one")
	}
	for _, want := range []string{
		"layout(local_size_x = 64) in;",
		"layout(std430, binding = 3) coherent writeonly buffer LogitBuf { float Logits[]; };",
		"layout(std430, binding = 3) coherent readonly buffer LogitInBuf { float LogitsIn[]; };",
		"zv[j] = LogitsIn[e];", "float z = sSelect[e];", "coherent buffer StatusBuf",
		"uint prev = atomicAdd(Status[3], 1u);",
		"sLast = prev + 1u == gl_NumWorkGroups.x - 1u ? 1u : 0u;",
		"if (e < pc.experts && (taken & (1u << j)) == 0u && (!has || cv[j] > best)) { has = true; best = cv[j]; bestExpert = e; }",
		"return (bits & 0x80000000u) != 0u ? ~bits : bits | 0x80000000u;",
		"uint bits = value == 0.0 ? 0u : floatBitsToUint(value);",
		"int winner = subgroupMin(has && key == topKey ? bestExpert : NO_FAULT);",
		"if (sKey[g] > topKey || (sKey[g] == topKey && sExpert[g] < winner)) {",
		"if (faultExpert == firstFault) fault(faultStage, faultExpert, faultValue);",
	} {
		if !strings.Contains(fused, want) {
			t.Errorf("fused shader lost contract fragment %q", want)
		}
	}
	shim, err := os.ReadFile("vulkan_shim.cpp")
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		`buildKernel(g_kern[K_V41_ROUTER_GATE_TOPK], v41RouterGateModule, 7, 24)`,
		`g_v41RouterGateSubgroups = haveSubgroupBasic && haveSubgroupArithmetic;`,
		`const uint64_t perGroup = 64u / chunks;`,
		`experts > 1024 || hidden <= 0 || hidden > 65536`,
	} {
		if !strings.Contains(string(shim), want) {
			t.Errorf("shim lost router gate ABI fragment %q", want)
		}
	}
	if V41RouterGateTopKMaxExperts != 1024 || V41RouterGateTopKMaxHidden != 65536 {
		t.Fatal("Go bounds drifted from the shim and shader ABI")
	}
}
