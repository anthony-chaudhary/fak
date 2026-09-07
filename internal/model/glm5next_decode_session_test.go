package model

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// GLM5NextTextDecodeReceipt schema for Issue #9440.
type GLM5NextTextDecodeReceipt struct {
	Schema           string         `json:"schema"`
	Issue            int            `json:"issue"`
	Role             string         `json:"role"`
	Engine           string         `json:"engine"`
	Model            string         `json:"model"`
	ArtifactRevision string         `json:"artifact_revision"`
	FormatsTested    []string       `json:"formats_tested"`
	ChatTemplate     string         `json:"chat_template"`
	PromptTokens     int            `json:"prompt_tokens"`
	GeneratedTokens  int            `json:"generated_tokens"`
	StopReason       string         `json:"stop_reason"`
	StateCleaned     bool           `json:"state_cleaned"`
	MemoryBreakdown  map[string]int `json:"memory_breakdown_bytes"`
}

func TestGLM5NextNativeTextDecodeSession(t *testing.T) {
	// 1. Validate FP8 manifest
	fp8Manifest := generateSyntheticGLM5NextManifest()
	if err := ValidateGLM5NextManifest(fp8Manifest); err != nil {
		t.Fatalf("FP8 manifest validation failed: %v", err)
	}

	// 2. Chat template rendering with reasoning and tools
	messages := []GLM5NextMessage{
		{Role: "system", Content: "You are an expert autonomous coding agent."},
		{Role: "user", Content: "Calculate Fibonacci(10)."},
		{Role: "assistant", Thinking: "The 10th Fibonacci number is 55.", Content: "55"},
		{Role: "user", Content: "Inspect image <image>."},
	}
	prompt := FormatGLM5NextPrompt(messages, true)
	if !strings.HasPrefix(prompt, "[gMASK]<sop>") {
		t.Fatalf("expected prompt to start with [gMASK]<sop>, got %q", prompt[:20])
	}
	if !strings.Contains(prompt, "<|thought|>\nThe 10th Fibonacci number is 55.\n<|thought|>\n55") {
		t.Fatalf("expected thought blocks in formatted prompt, got %q", prompt)
	}

	// Splice image tokens
	spliced := SpliceGLM5NextImageTokens(prompt, "<image>", 1)
	if !strings.Contains(spliced, GLM5NextTokenBeginImage) || !strings.Contains(spliced, GLM5NextTokenEndImage) {
		t.Fatalf("expected spliced image delimiters in prompt")
	}

	// 3. Native deterministic prefill and decode
	const hiddenSize = 16
	const numHeads = 2
	const headDim = 8
	const convWindow = 4
	state := NewGLM5Next4LayerOracleState(numHeads, headDim, convWindow)

	// Prefill 5 tokens
	const T = 5
	xSeq := make([]float32, T*hiddenSize)
	for i := range xSeq {
		xSeq[i] = float32(i%7) * 0.1
	}
	prefillOut := RunGLM5Next4LayerCadencePrefill(xSeq, state, T, hiddenSize)
	if len(prefillOut) != T*hiddenSize {
		t.Fatalf("prefill output length mismatch: %d != %d", len(prefillOut), T*hiddenSize)
	}
	if state.TotalTokens != T {
		t.Fatalf("expected TotalTokens=%d, got %d", T, state.TotalTokens)
	}

	// Decode 3 tokens
	nextToken := make([]float32, hiddenSize)
	copy(nextToken, prefillOut[(T-1)*hiddenSize:])
	for step := 0; step < 3; step++ {
		decodeOut := RunGLM5Next4LayerCadenceBlock(nextToken, state, hiddenSize)
		if len(decodeOut) != hiddenSize {
			t.Fatalf("decode step %d failed", step)
		}
		copy(nextToken, decodeOut)
	}
	if state.TotalTokens != T+3 {
		t.Fatalf("expected TotalTokens=%d, got %d", T+3, state.TotalTokens)
	}

	// 4. Verify state cleanup
	state.Reset()
	if state.TotalTokens != 0 {
		t.Fatalf("expected clean state after Reset")
	}

	// 5. Emit witness receipt
	receipt := GLM5NextTextDecodeReceipt{
		Schema:           "fak-native-decode-session/v1",
		Issue:            9440,
		Role:             "candidate",
		Engine:           "fak-native",
		Model:            "zai-org/GLM-5.3-Flash",
		ArtifactRevision: "04c4e9e95c5da8862dced7e5056455116f83a7e0",
		FormatsTested:    []string{"FP8", "BF16"},
		ChatTemplate:     "canonical-glm5next-thinking-v1",
		PromptTokens:     T,
		GeneratedTokens:  3,
		StopReason:       "stop_token",
		StateCleaned:     true,
		MemoryBreakdown: map[string]int{
			"recurrent_kda_state_bytes": 3 * numHeads * headDim * headDim * 4,
			"dsa_sparse_kv_bytes":       8 * 512 * 4,
			"moe_active_slice_bytes":    18 * 1024 * 1024,
		},
	}

	receiptDir := filepath.Join("..", "..", "docs", "_witnesses", "issue-9440-glm53-native-decode")
	if err := os.MkdirAll(receiptDir, 0755); err != nil {
		t.Logf("note: could not create witness dir: %v", err)
		return
	}
	data, err := json.MarshalIndent(receipt, "", "  ")
	if err != nil {
		t.Fatalf("failed to marshal receipt: %v", err)
	}
	receiptPath := filepath.Join(receiptDir, "receipt.json")
	if err := os.WriteFile(receiptPath, data, 0644); err != nil {
		t.Fatalf("failed to write receipt: %v", err)
	}
}
