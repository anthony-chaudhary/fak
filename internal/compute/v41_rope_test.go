package compute

import (
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// fak-test:runtime fast est=1ms lane=default
func TestV41TailRoPEContract(t *testing.T) {
	t.Parallel()
	q := Tensor{Dtype: F32, Layout: RowMajor, Shape: []int{2, 7}}
	kv := Tensor{Dtype: F32, Layout: RowMajor, Shape: []int{7}}
	table := Tensor{Dtype: F32, Layout: RowMajor, Shape: []int{2, 2}}
	if qb, kb, tb, err := validateV41TailRoPE(q, kv, table, 2, 7, 4); err != nil || qb != 56 || kb != 28 || tb != 16 {
		t.Fatalf("valid odd-prefix geometry: bytes=%d/%d/%d err=%v", qb, kb, tb, err)
	}
	for _, dims := range [][3]int{{0, 7, 4}, {2, 7, 3}, {2, 7, 8}, {2, 7, 0}, {1<<31 - 1, 7, 4}} {
		if _, _, _, err := validateV41TailRoPE(q, kv, table, dims[0], dims[1], dims[2]); err == nil {
			t.Errorf("accepted invalid geometry %v", dims)
		}
	}
	for _, mutate := range []func(*Tensor){
		func(x *Tensor) { x.Shape = []int{14} },
		func(x *Tensor) { x.Dtype = BF16 },
		func(x *Tensor) { x.Layout = ColMajor },
		func(x *Tensor) { x.Quant = &QuantSpec{} },
	} {
		bad := q
		mutate(&bad)
		if _, _, _, err := validateV41TailRoPE(bad, kv, table, 2, 7, 4); err == nil {
			t.Errorf("accepted malformed Q: %+v", bad)
		}
	}
	badTable := table
	badTable.Shape = []int{4}
	if _, _, _, err := validateV41TailRoPE(q, kv, badTable, 2, 7, 4); err == nil {
		t.Fatal("accepted a flattened or wrong table layout")
	}

	// Exact-operation control: contraction changes a cancellation residual, so
	// cosine alone cannot establish that the shader obeyed this operation order.
	a, c := math.Float32frombits(0x3f800001), math.Float32frombits(0x3f7ffffe)
	got := v41TailRoPEReference([]float32{a, 1}, []float32{1, c}, 2, 2)
	fused := float32(float64(a)*float64(c) - 1)
	if got[0] != 0 || fused == got[0] {
		t.Fatalf("separate-product cancellation control failed: separate=%g fused=%g", got[0], fused)
	}
}

// v41TailRoPEReference is an independent test-only scalar oracle. It consumes
// supplied table bytes directly and cannot be selected by a production backend.
func v41TailRoPEReference(input, table []float32, headDim, rotaryDim int) []float32 {
	out := append([]float32(nil), input...)
	for head := 0; head < len(input)/headDim; head++ {
		for j := 0; j < rotaryDim/2; j++ {
			i := head*headDim + headDim - rotaryDim + 2*j
			a, b := input[i], input[i+1]
			s, c := table[2*j], table[2*j+1]
			out[i] = float32(a*c) - float32(b*s)
			out[i+1] = float32(b*c) + float32(a*s)
		}
	}
	return out
}

// fak-test:runtime fast est=5ms lane=default
// Source witness only: no compiler invocation, SPIR-V fabrication, or GPU proof.
func TestV41TailRoPESourceContract(t *testing.T) {
	t.Parallel()
	read := func(path string) string {
		t.Helper()
		raw, err := os.ReadFile(filepath.FromSlash(path))
		if err != nil {
			t.Fatal(err)
		}
		return string(raw)
	}
	shader := read("shaders/v41_tail_rope_qk.comp")
	for _, token := range []string{
		"layout(local_size_x = 256) in;", "uint Q[]", "uint KV[]", "uint QOut[]", "uint KVOut[]",
		"float SinCos[]", "uint tail = uint(pc.headDim - pc.rotaryDim);", "if ((lane & 1u) != 0u) return;",
		"KVOut[index] = KV[index];", "QOut[index] = Q[index];", "float s = SinCos[lane];", "float c = SinCos[lane + 1u];",
		"precise float ac = a * c;", "precise float bs = b * s;", "precise float bc = b * c;", "precise float as = a * s;",
		"precise float x = ac - bs;", "precise float y = bc + as;",
	} {
		if !strings.Contains(shader, token) {
			t.Errorf("missing shader contract %q", token)
		}
	}
	for _, forbidden := range []string{"sin(", "cos(", "pow(", "fma(", "float16_t", "halfRotary"} {
		if strings.Contains(shader, forbidden) {
			t.Errorf("forbidden shader behavior %q", forbidden)
		}
	}
	shim := read("vulkan_shim.cpp")
	for _, token := range []string{
		`ok &= buildKernel(g_kern[K_V41_TAIL_ROPE_QK], P("v41_tail_rope_qk.spv"), 5, 3 * sizeof(int));`,
		"case K_ROPE: case K_QWEN35_PARTIAL_ROPE_PANEL: case K_V41_TAIL_ROPE_QK:",
		"nbuf != 5 || pcsize != 12 || !v41TailRoPEABI(code)", "buffers.size() != 5", "seenBindings == 31",
		"!decorated(id, noContraction, 0)", "!atOffset(pushStruct, m, m * 4)",
		"if (i != out && aliases(bufs[out], bufs[i])) return 2;", "groups > g_maxComputeWorkGroupCountX",
		"if (!dispatch(g_kern[K_V41_TAIL_ROPE_QK]", "g_have_v41_tail_rope_qk = 0;",
		"g_have_v41_tail_rope_qk = 1;", "v41BeginCmdChecked()", "v41EndSubmitWaitChecked(cmd, true)",
		"if (g_v41SubmissionPendingFailure) return;", "if (g_batching || g_v41SubmissionPendingFailure)",
	} {
		if !strings.Contains(shim, token) {
			t.Errorf("missing native contract %q", token)
		}
	}
	begin := strings.Index(shim, "VkCommandBuffer v41BeginCmdChecked()")
	if begin < 0 {
		t.Fatal("missing V4.1 checked command helpers")
	}
	end := strings.Index(shim[begin:], "VkResult restoreBeginCommand()")
	if end < 0 || strings.Contains(shim[begin:begin+end], "VKCHECK(") {
		t.Fatal("V4.1 checked command helpers must return errors instead of aborting")
	}
	goBackend := read("vulkan.go")
	for _, site := range []string{"RetireRequestResources", "dallocForClass", "dallocHostVisFor", "dallocWeightFor", "dallocTransient", "recycleTransientLocked", "trimTransientLocked"} {
		if !strings.Contains(goBackend, `requireVulkanSubmissionHealthyLocked("`+site+`")`) {
			t.Errorf("missing sticky-status ownership guard at %s", site)
		}
	}
	if !strings.Contains(goBackend, "C.fvk_retire_request()\n\trequireVulkanSubmissionHealthyLocked(\"RetireRequestResources\")") {
		t.Fatal("request retirement must check the completion result before publishing pooled owners")
	}
}
