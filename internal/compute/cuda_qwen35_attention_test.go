package compute

import (
	"os"
	"strings"
	"testing"
)

func qwen38PromptAttentionSource(t *testing.T, name string) string {
	t.Helper()
	source, err := os.ReadFile(name)
	if err != nil {
		t.Fatalf("read %s: %v", name, err)
	}
	return string(source)
}

func qwen38SourceFunction(t *testing.T, source, marker string) string {
	t.Helper()
	start := strings.Index(source, marker)
	if start < 0 {
		t.Fatalf("source marker %q not found", marker)
	}
	open := strings.Index(source[start:], "{")
	if open < 0 {
		t.Fatalf("source marker %q has no body", marker)
	}
	open += start
	depth := 0
	for index := open; index < len(source); index++ {
		switch source[index] {
		case '{':
			depth++
		case '}':
			depth--
			if depth == 0 {
				return source[start : index+1]
			}
		}
	}
	t.Fatalf("source marker %q has an unterminated body", marker)
	return ""
}

// Follow the dispatch seam through its helper rather than accepting a helper
// name alone as evidence of a launch. Both routing guards use this same spine.
func qwen38PromptAttentionSIMTSource(t *testing.T, source string) string {
	t.Helper()
	body := qwen38SourceFunction(t, source, "static int qwen38_causal_attention_panel_launch_simt(")
	shared := strings.Index(body, "((size_t)QWEN38_PROMPT_HEAD_DIM + FLASH_THREADS) * sizeof(float)")
	launch := strings.Index(body, "k_qwen38_causal_attention_panel_hd256<<<tokens * nH, FLASH_THREADS, shmem, g_stream>>>")
	status := strings.Index(body, "cudaError_t launch = cudaGetLastError();")
	result := strings.Index(body, "return launch == cudaSuccess ? 0 : 50000 + (int)launch;")
	if shared < 0 || launch < 0 || status < 0 || result < 0 || !(shared < launch && launch < status && status < result) {
		t.Fatalf("SIMT helper must provision hd256 shared memory, launch, then report CUDA status: shared=%d launch=%d status=%d result=%d", shared, launch, status, result)
	}
	if strings.Contains(body, "return 0;") || strings.Contains(body, "k_qwen35_causal_attention_panel<<<") {
		t.Fatal("SIMT helper bypasses the exact hd256 status-returning launch")
	}
	return body
}

func TestQwen38PromptAttentionHD256SourceSpine(t *testing.T) {
	source := cudaSourceClosure(t, "cuda_kernels.cu")
	body := qwen38SourceFunction(t, source, "__global__ void k_qwen38_causal_attention_panel_hd256")
	for _, want := range []string{
		"float acc[QWEN38_PROMPT_HEAD_DIM / FLASH_THREADS]",
		"float *red = smem + QWEN38_PROMPT_HEAD_DIM",
		"float correction = expf(m - nextM)",
		"probability = expf(score - nextM)",
		"Out[(size_t)token * nH * QWEN38_PROMPT_HEAD_DIM",
	} {
		if !strings.Contains(body, want) {
			t.Errorf("hd256 source spine missing %q", want)
		}
	}
	if !strings.Contains(source, "#define QWEN38_PROMPT_HEAD_DIM 256") {
		t.Fatal("hd256 source path does not bind the Qwen3.8 head dimension")
	}
	if strings.Contains(body, "hd > 32") {
		t.Fatal("hd256 source path retains the legacy successful no-op guard")
	}
}

func TestQwen38PromptAttentionABIDispatchIsFailClosed(t *testing.T) {
	cu := cudaSourceClosure(t, "cuda_kernels.cu")
	goSource := qwen38PromptAttentionSource(t, "cuda_kernels.go")
	const abi = "int fak_qwen35_causal_attention_panel_f32("
	if !strings.Contains(cu, "extern \"C\" "+abi) || !strings.Contains(goSource, abi) {
		t.Fatal("Go/CUDA prompt-attention ABI declarations diverged")
	}
	body := qwen38SourceFunction(t, cu, "extern \"C\" "+abi)
	preflight := strings.Index(body, "tokens <= 0 || prefix < 0 || nH != 24 || nKV != 4")
	headDim := strings.Index(body, "hd != QWEN38_PROMPT_HEAD_DIM")
	clearError := strings.Index(body, "cudaGetLastError()")
	launch := strings.Index(body, "return qwen38_causal_attention_panel_launch_simt(")
	if preflight < 0 || headDim < 0 || clearError < 0 || launch < 0 || !(preflight < headDim && headDim < clearError && clearError < launch) {
		t.Fatalf("ABI must refuse every non-exact dimension before CUDA effects, then launch hd256: tuple=%d head_dim=%d clear=%d launch=%d", preflight, headDim, clearError, launch)
	}
	for _, overflow := range []string{"tokens > INT_MAX / nH", "prefix > INT_MAX - tokens", "positions > INT_MAX / kvWidth", "tokens > INT_MAX / qWidth", "tokens > INT_MAX / nH / nsplits"} {
		if guard := strings.Index(body, overflow); guard < 0 || guard > clearError {
			t.Fatalf("ABI must keep int-overflow refusal %q before CUDA effects: at=%d clear=%d", overflow, guard, clearError)
		}
	}
	if strings.Contains(body, "tokens != 2") || strings.Contains(body, "prefix != 1") {
		t.Fatal("prompt-attention ABI still pins the single (tokens=2, prefix=1) panel the Go validator no longer requires")
	}
	qwen38PromptAttentionSIMTSource(t, cu)
	if !strings.Contains(body, "((size_t)QWEN38_PROMPT_HEAD_DIM + FLASH_THREADS) * sizeof(float)") {
		t.Fatal("split-KV dispatcher does not provision query-plus-reduction shared memory")
	}
	if strings.Contains(body, "return 0;") {
		t.Fatal("prompt-attention ABI contains an unconditional successful return")
	}
	if strings.Contains(body, "k_qwen35_causal_attention_panel<<<") {
		t.Fatal("prompt-attention ABI retains a generic success launch outside the exact hd256 spine")
	}
}

func TestQwen38PromptAttentionGoLauncherPreflightsBeforeAllocation(t *testing.T) {
	source := qwen38PromptAttentionSource(t, "cuda_kernels.go")
	body := qwen38SourceFunction(t, source, "func (c *cudaBackend) qwen35SequenceAttentionLocked")
	validate := strings.Index(body, "validateQwen35CausalAttentionPanelGeometry")
	scale := strings.Index(body, "validateQwen35CausalAttentionPanelScale")
	allocate := strings.Index(body, "qwen35SequenceAllocLocked")
	launch := strings.Index(body, "C.fak_qwen35_causal_attention_panel_f32")
	if validate < 0 || scale < 0 || allocate < 0 || launch < 0 || !(validate < scale && scale < allocate && allocate < launch) {
		t.Fatalf("Go launcher effect order invalid: geometry=%d scale=%d allocation=%d launch=%d", validate, scale, allocate, launch)
	}
	sequence := qwen38PromptAttentionSource(t, "cuda_qwen35_sequence.go")
	geometry := qwen38SourceFunction(t, sequence, "func validateQwen35SequenceGeometry")
	if !strings.Contains(geometry, "validateQwen35CausalAttentionPanelGeometry") {
		t.Fatal("full sequence request does not apply launcher geometry during effect-free preflight")
	}
}

func TestQwen38PromptAttentionHardwareWitnessRemainsExplicit(t *testing.T) {
	source := qwen38PromptAttentionSource(t, "cuda_qwen35_sequence_test.go")
	for _, want := range []string{
		`{"qwen3.8-24x4-hd256", 2, 1, 24, 4, qwen38CausalAttentionPanelHeadDim}`,
		"similarity < cudaFlashAttnCosineMin",
		"resident prompt-panel kernel transferred host bytes before proof Read",
		"retained caller sentinel",
	} {
		if !strings.Contains(source, want) {
			t.Errorf("sanctioned CUDA witness is missing %q", want)
		}
	}
}

func TestQwen38PromptAttentionMMADispatchIsOptInAndFallsBack(t *testing.T) {
	cu := cudaSourceClosure(t, "cuda_kernels.cu")
	body := qwen38SourceFunction(t, cu, "extern \"C\" int fak_qwen35_causal_attention_panel_f32(")
	clearError := strings.Index(body, "cudaGetLastError()")
	selectMMA := strings.Index(body, "qwen38_prefill_attention_wants_mma()")
	launchMMA := strings.Index(body, "fak_qwen38_fattn_mma_hd256_launch(")
	fallback := strings.Index(body, "if (mma != -2) return mma;")
	launchSIMT := strings.Index(body, "return qwen38_causal_attention_panel_launch_simt(")
	if clearError < 0 || selectMMA < 0 || launchMMA < 0 || fallback < 0 || launchSIMT < 0 ||
		!(clearError < selectMMA && selectMMA < launchMMA && launchMMA < fallback && fallback < launchSIMT) {
		t.Fatalf("mma dispatch must follow every refusal and fall back to the SIMT launch: clear=%d select=%d mma=%d fallback=%d simt=%d", clearError, selectMMA, launchMMA, fallback, launchSIMT)
	}
	qwen38PromptAttentionSIMTSource(t, cu)
	const simtReturn = "return qwen38_causal_attention_panel_launch_simt(dQ, dK, dV, dOut, tokens, prefix, nH, nKV, scale);"
	if strings.Count(body, simtReturn) != 2 {
		t.Fatal("single-pass and unavailable split-KV scratch must both return the SIMT helper's status")
	}
	for _, marker := range []string{"if (nsplits <= 1)", "if (!splitReady)"} {
		branch := qwen38SourceFunction(t, body, marker)
		if !strings.Contains(branch, simtReturn) {
			t.Fatalf("%s no longer returns the actual SIMT fallback", marker)
		}
	}
	mode := qwen38SourceFunction(t, cu, "static bool qwen38_prefill_attention_wants_mma(void)")
	for _, want := range []string{
		`getenv("FAK_CUDA_QWEN38_PREFILL_ATTN")`,
		`strcmp(mode, "simt") == 0) return false;`,
		`strcmp(mode, "mma") == 0) return true;`,
	} {
		if !strings.Contains(mode, want) {
			t.Errorf("prompt-attention kernel selector missing %q", want)
		}
	}
	// SIMT stays the default until the P-05 hardware A/B records a keep verdict.
	if !strings.Contains(cu, "#define QWEN38_PREFILL_ATTN_MMA_DEFAULT 0") {
		t.Fatal("mma prompt attention became the default without a hardware keep receipt")
	}
	launcher := qwen38SourceFunction(t, cu, "static int fak_qwen38_fattn_mma_hd256_launch(")
	capacity := strings.Index(launcher, "if (major < 8) return -2;")
	capture := strings.Index(launcher, "if (g_capture_open) return -2;")
	stage := strings.Index(launcher, "k_qwen38_fattn_stage_kv_f16<<<")
	if capacity < 0 || capture < 0 || stage < 0 || !(capacity < stage && capture < stage) {
		t.Fatalf("mma launcher must decline (-2) before any effect: sm80=%d capture=%d stage=%d", capacity, capture, stage)
	}
}

func TestQwen38PromptAttentionMMAHeaderCompilesAsCXX14Port(t *testing.T) {
	header := qwen38PromptAttentionSource(t, "cuda/fattn_mma_hd256.cuh")
	for _, want := range []string{
		"0e1d9185c5fe82e905d1f5ae6b2e5dcd607a8dfd (MIT)",
		"mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32",
		"ldmatrix.sync.aligned.m8n8.x4.trans.b16",
		"cp.async.cg.shared.global.L2::64B",
		"if (i > causal_row0 + j) {",
		"__launch_bounds__(nthreads, occupancy)",
	} {
		if !strings.Contains(header, want) {
			t.Errorf("hd256 mma port missing %q", want)
		}
	}
	for _, banned := range []string{"if constexpr", "std::is_same_v", "__device__ (auto"} {
		if strings.Contains(header, banned) {
			t.Errorf("hd256 mma port retains C++17-only %q; the port was written C++14-clean", banned)
		}
	}
}
