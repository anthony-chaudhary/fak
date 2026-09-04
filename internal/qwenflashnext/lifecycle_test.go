package qwenflashnext

import (
	"testing"
)

// Invariant: Qwen chat parsing must extract thinking analysis and final responses separated by canonical stop tokens.
// Guard: ParseResponse extracts reasoning blocks without leaking thought tags into final outputs.

func TestQwenFlashNextLifecycle(t *testing.T) {
	t.Parallel()

	resp := "<think>\nreasoning block\n</think>\n\nFinal answer.<|im_end|>"
	parsed, err := ParseResponse(resp)
	if err != nil {
		t.Fatalf("ParseResponse failed: %v", err)
	}
	if parsed.Analysis != "reasoning block" || parsed.Final != "Final answer." || !parsed.Stopped {
		t.Fatalf("unexpected parsed response: %+v", parsed)
	}
}
