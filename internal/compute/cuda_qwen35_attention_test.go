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

func TestQwen38PromptAttentionHD256SourceSpine(t *testing.T) {
	source := qwen38PromptAttentionSource(t, "cuda_kernels.cu")
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
	cu := qwen38PromptAttentionSource(t, "cuda_kernels.cu")
	goSource := qwen38PromptAttentionSource(t, "cuda_kernels.go")
	const abi = "int fak_qwen35_causal_attention_panel_f32("
	if !strings.Contains(cu, "extern \"C\" "+abi) || !strings.Contains(goSource, abi) {
		t.Fatal("Go/CUDA prompt-attention ABI declarations diverged")
	}
	body := qwen38SourceFunction(t, cu, "extern \"C\" "+abi)
	preflight := strings.Index(body, "tokens <= 0 || prefix < 0 || nH != 24 || nKV != 4")
	headDim := strings.Index(body, "hd != QWEN38_PROMPT_HEAD_DIM")
	clearError := strings.Index(body, "cudaGetLastError()")
	launch := strings.Index(body, "k_qwen38_causal_attention_panel_hd256<<<")
	if preflight < 0 || headDim < 0 || clearError < 0 || launch < 0 || !(preflight < headDim && headDim < clearError && clearError < launch) {
		t.Fatalf("ABI must refuse every non-exact dimension before CUDA effects, then launch hd256: tuple=%d head_dim=%d clear=%d launch=%d", preflight, headDim, clearError, launch)
	}
	if !strings.Contains(body, "((size_t)QWEN38_PROMPT_HEAD_DIM + FLASH_THREADS) * sizeof(float)") {
		t.Fatal("hd256 dispatcher does not provision query-plus-reduction shared memory")
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
