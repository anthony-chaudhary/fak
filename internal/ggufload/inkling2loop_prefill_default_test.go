package ggufload

import (
	"os"
	"testing"
)

// TestInkling2PrefillBatchedDefault pins the batched prefill default: on when the
// flag is unset, off only for an explicit FAK_INKLING2_BATCHED_PREFILL=0.
func TestInkling2PrefillBatchedDefault(t *testing.T) {
	t.Setenv("FAK_INKLING2_BATCHED_PREFILL", "")
	os.Unsetenv("FAK_INKLING2_BATCHED_PREFILL")
	if !inkling2BatchedPrefill() {
		t.Fatal("batched prefill is off with the flag unset, want the default on")
	}
	t.Setenv("FAK_INKLING2_BATCHED_PREFILL", "0")
	if inkling2BatchedPrefill() {
		t.Fatal("batched prefill is on with FAK_INKLING2_BATCHED_PREFILL=0, want the per-token step")
	}
	t.Setenv("FAK_INKLING2_BATCHED_PREFILL", "1")
	if !inkling2BatchedPrefill() {
		t.Fatal("batched prefill is off with FAK_INKLING2_BATCHED_PREFILL=1")
	}
}
