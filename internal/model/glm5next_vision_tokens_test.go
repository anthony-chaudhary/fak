package model

import (
	"strings"
	"testing"
)

func TestEstimateGLM5NextImageTokenBudget(t *testing.T) {
	// Standard single 448x448 tile: 256 pads + 2 boundary tokens = 258
	if got := EstimateGLM5NextImageTokenBudget(448, 448); got != 258 {
		t.Fatalf("EstimateGLM5NextImageTokenBudget(448, 448) = %d, want 258", got)
	}

	// 2 tiles horizontal (896x448): 2 * 256 + 2 = 514
	if got := EstimateGLM5NextImageTokenBudget(896, 448); got != 514 {
		t.Fatalf("EstimateGLM5NextImageTokenBudget(896, 448) = %d, want 514", got)
	}

	// 4 tiles 2x2 (896x896): 4 * 256 + 2 = 1026
	if got := EstimateGLM5NextImageTokenBudget(896, 896); got != 1026 {
		t.Fatalf("EstimateGLM5NextImageTokenBudget(896, 896) = %d, want 1026", got)
	}

	// Non-positive dimensions: 0
	if got := EstimateGLM5NextImageTokenBudget(0, 448); got != 0 {
		t.Fatalf("EstimateGLM5NextImageTokenBudget(0, 448) = %d, want 0", got)
	}
}

func TestSpliceGLM5NextImageTokens(t *testing.T) {
	prompt := "User: what is in this image: <image>?"
	spliced := SpliceGLM5NextImageTokens(prompt, "<image>", 1)

	if !strings.Contains(spliced, GLM5NextTokenBeginImage) || !strings.Contains(spliced, GLM5NextTokenEndImage) {
		t.Fatalf("spliced prompt missing boundary tokens: %s", spliced)
	}

	// Count number of <|image_pad|> occurrences
	padCount := strings.Count(spliced, GLM5NextTokenImagePad)
	if padCount != 256 {
		t.Fatalf("expected 256 image pad tokens, got %d", padCount)
	}

	// Verify prefix and suffix preserved
	if !strings.HasPrefix(spliced, "User: what is in this image: ") {
		t.Fatalf("prompt prefix corrupted: %s", spliced)
	}
	if !strings.HasSuffix(spliced, "?") {
		t.Fatalf("prompt suffix corrupted: %s", spliced)
	}
}
