package compute

import (
	"os"
	"strings"
	"testing"
)

// These source guards are not CUDA compilation, execution, or proof of numeric
// equivalence. They keep the actual producer/consumer/retirement seams visible
// to the ordinary host test lane while required-device witnesses stay explicit.
// fak-test:runtime fast est=10ms lane=default
func TestCUDAQ56KMMQSourceContract(t *testing.T) {
	read := func(path string) string {
		t.Helper()
		b, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		return string(b)
	}
	header := read("cuda/q56k_mmq.cuh")
	for _, marker := range []string{
		"0e1d9185c5fe82e905d1f5ae6b2e5dcd607a8dfd",
		"#ifdef FAK_CUDA_Q56K_MMQ_QUALIFICATION",
		"sizeof(block_q5_K) == 176", "sizeof(block_q6_K) == 210",
		"info.cc < 80 || info.compiled_arch < 80",
		"p < 128 || in % QK_K",
		"cudaPointerGetAttributes", "attr.type == cudaMemoryTypeDevice",
		"g_capture_open || g_q4k_qkv_async_pending != 0",
		"cudaMemsetAsync(activation, 0, plan.activation_bytes, g_stream)",
		"if constexpr (kind == 5) y[ib].ds4", "else y[ib].d4",
		"FAK_ALLOC_SITE(\"q56k-mmq-q8_1\")", "FAK_ALLOC_SITE(\"q56k-mmq-stream-k\")",
		"sequence_split_note_failed_fence(drained)",
		"g_cuda_cleanup_status = (int)drained",
		"return first ? first : g_cuda_cleanup_status",
	} {
		if !strings.Contains(header, marker) {
			t.Fatalf("native contract missing %q", marker)
		}
	}
	// The error observation stays INSIDE each profiling body and before the
	// next launch; a separate CGo query could observe another host thread.
	if strings.Count(header, "rc = (int)cudaGetLastError();") != 3 {
		t.Fatal("quantize/main/fixup must each capture immediate launch status")
	}
	if strings.Count(header, "cudaStreamSynchronize(g_stream)") != 1 {
		t.Fatal("the family must retain exactly the single failure-only drain")
	}
	quant := read("cuda/quant_matmul.cuh")
	for _, kind := range []string{"5", "6"} {
		checked := "extern \"C\" int fcuda_q" + kind + "k_matmul_checked_f32"
		start := strings.Index(quant, checked)
		if start < 0 {
			t.Fatal("old checked ABI missing")
		}
		end := strings.Index(quant[start:], "\n}")
		if end < 0 {
			t.Fatal("checked ABI body missing")
		}
		body := quant[start : start+end]
		if strings.Contains(body, "FAK_CUDA_Q56K_MMQ_QUALIFICATION") ||
			strings.Contains(body, "fak_q56k_mmq") || !strings.Contains(body, "q56k_mmvq_run") {
			t.Fatal("legacy checked ABI must not widen the separate packed-GDN contract")
		}
	}
	tu := read("cuda_kernels.cu")
	i := strings.Index(tu, "#include \"cuda/q56k_mmq.cuh\"")
	if i < 0 || i > strings.Index(tu, "#include \"cuda/quant_matmul.cuh\"") {
		t.Fatal("new checked family must be included before the actual dispatcher")
	}
	abi := read("cuda_backend.h")
	for _, sym := range []string{"fcuda_q56k_mmq_enabled", "fcuda_q56k_mmq_launch_count", "fcuda_q56k_mmq_preflight", "fcuda_q56k_mmq_matmul_checked_f32", "fcuda_q56k_mmq_retirement_failed"} {
		if !strings.Contains(abi, sym) {
			t.Fatal("native interface enrollment missing", sym)
		}
	}
	goSide := read("cuda_q56k_mmq.go")
	if !strings.Contains(goSide, "C.fcuda_q56k_mmq_enabled() != 0") ||
		!strings.Contains(goSide, "p >= 128") || strings.Contains(goSide, "os.Getenv") ||
		!strings.Contains(goSide, "C.fcuda_qwen35_iq2_sequence_fence()") ||
		!strings.Contains(goSide, "C.fcuda_q56k_mmq_matmul_checked_f32") ||
		!strings.Contains(header, "if (kind == 6) return fak_q56k_mmq::run<6>") {
		t.Fatal("typed immutable selection/final shared fence contract changed")
	}
	ordinary := read("cuda.go")
	if !strings.Contains(ordinary, "c.validateQ56KProjectionOperands(w, X, P)") ||
		!strings.Contains(ordinary, "c.q56KProjectionStatus(w, wb, X, y, P)") ||
		!strings.Contains(ordinary, "cudaQ56KMMQRetirementFailedLocked(status)") ||
		!strings.Contains(header, "sequence_split_note_failed_fence((cudaError_t)first)") ||
		!strings.Contains(goSide, "return c.validateQ56KMMQWeight(w, p)") ||
		!strings.Contains(goSide, "return c.validateQ56KMMQOperands(w, x, p)") {
		t.Fatal("ordinary whole-operation and activation admission no longer compose")
	}
	sequence := read("cuda_qwen35_sequence.go")
	if !strings.Contains(sequence, "launch.Op == cudaQ56KMMQOp") ||
		!strings.Contains(sequence, "qwen35SequenceIQ2Qualification || cudaQ56KMMQQualification") {
		t.Fatal("mixed sequence failure/retirement no longer preserves shared terminal ownership")
	}
	kernels := read("cuda_kernels.go")
	if !strings.Contains(kernels, "status = c.qwen35Q56KMMQFenceLocked()") ||
		!strings.Contains(kernels, "c.q56KProjectionStatus(w, wb, x, y, tokens)") {
		t.Fatal("ordinary projection/final fence bypasses selected terminal bridge")
	}
	if !strings.Contains(sequence, "c.validateQ56KProjectionWeight(w, panelRows)") ||
		!strings.Contains(sequence, "c.validateQ56KMMVQWeight(w, panelRows)") {
		t.Fatal("whole-operation admission must preserve separate packed-GDN math")
	}
}
