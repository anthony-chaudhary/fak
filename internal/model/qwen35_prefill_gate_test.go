package model

import "testing"

func TestQwen35HybridPrefillTokenLoopEscapeHatch(t *testing.T) {
	cfg := qwen35HybridTestCfg()
	if !q8Qwen35HybridPrefillOK(cfg, qwen35HybridQBatchMinPrompt) ||
		!q4kQwen35HybridPrefillOK(cfg, qwen35HybridQBatchMinPrompt) {
		t.Fatal("baseline Q8/Q4_K hybrid prefill gates unexpectedly declined")
	}
	t.Setenv("FAK_QWEN35_PREFILL_TOKEN_LOOP", "1")
	if q8Qwen35HybridPrefillOK(cfg, qwen35HybridQBatchMinPrompt) ||
		q4kQwen35HybridPrefillOK(cfg, qwen35HybridQBatchMinPrompt) {
		t.Fatal("FAK_QWEN35_PREFILL_TOKEN_LOOP must force both Q8 and Q4_K token-loop paths")
	}
}
