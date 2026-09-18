package main

import (
	"strings"
	"testing"
)

// guard_metal_flag_test.go — the `fak guard --gguf --metal` session-forward selector.
// Metal here is the Apple-Silicon CPU-session seam (the same one `fak serve --metal`
// uses), NOT a registered compute HAL backend. These tests pin the guard-facing
// resolver contract: verb-corrected errors and the mutual-exclusion refusal, without
// requiring a Metal device (the shared resolver's non-Metal stub answers deterministically).

// TestResolveGuardMetalMutualExclusion pins that an explicit --backend plus --metal is
// refused with a guard-verb error (never silently picks one), mirroring `fak run`.
func TestResolveGuardMetalMutualExclusion(t *testing.T) {
	t.Setenv("FAK_BACKEND", "")
	t.Setenv("FAK_METAL", "")

	_, err := resolveGuardMetal(true, false, "cpu")
	if err == nil {
		t.Fatalf("expected mutual-exclusion error for --metal + --backend cpu")
	}
	if !strings.Contains(err.Error(), "fak guard:") {
		t.Errorf("expected guard-verb error prefix, got: %v", err)
	}
	if strings.Contains(err.Error(), "fak serve:") {
		t.Errorf("verb was not rewritten from serve to guard: %v", err)
	}
	if !strings.Contains(err.Error(), "mutually exclusive") {
		t.Errorf("expected mutual-exclusion wording, got: %v", err)
	}
}

// TestResolveGuardMetalAutoWithoutDevice pins the auto-select posture: with no flag, no
// env, and no explicit backend, the resolver never errors — it either selects the session
// seam (device present) or falls back to the compute path (no device / non-Metal build).
func TestResolveGuardMetalAutoWithoutDevice(t *testing.T) {
	t.Setenv("FAK_BACKEND", "")
	t.Setenv("FAK_METAL", "")

	if _, err := resolveGuardMetal(false, false, ""); err != nil {
		t.Fatalf("auto resolve must not error without --backend/--metal, got: %v", err)
	}
}
