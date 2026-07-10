package main

import (
	"strings"
	"testing"
)

// A plain --base-url is passthrough: neither the hosted-provider default nor an explicit
// /v1 root is rewritten, so existing `fak deepseekbench --base-url ...` usage is unchanged
// and only the @lab/<model> alias engages the readiness gate.
func TestResolveBenchBaseURL_Passthrough(t *testing.T) {
	for _, raw := range []string{
		"https://api.deepseek.com", // hosted DeepSeek default
		"http://host:8000/v1",      // explicit self-hosted root
		"http://10.0.0.5:8000",     // bare host:port, left as the operator typed it
		"",                         // empty stays empty; the flag default owns this
	} {
		got, err := resolveBenchBaseURL(raw)
		if err != nil {
			t.Fatalf("resolveBenchBaseURL(%q) unexpected error: %v", raw, err)
		}
		if got != raw {
			t.Errorf("resolveBenchBaseURL(%q) = %q, want unchanged passthrough", raw, got)
		}
	}
}

// An @lab/<model> alias NEVER silently passes through. A model id that is in no local
// target config (and, on a box with no lab configured, no readiness file either) must fail
// CLOSED — empty base, non-nil LAB_* structured refusal — so a benchmark cannot start
// against a box the readiness gate has not cleared. Every resolver refusal is LAB_*-prefixed
// (LAB_READINESS_NOT_READY / LAB_TARGET_NOT_FOUND / LAB_TARGET_CONFIG_MISSING / ...), so the
// assertion is deterministic regardless of the ambient readiness/targets files.
func TestResolveBenchBaseURL_AliasFailsClosed(t *testing.T) {
	got, err := resolveBenchBaseURL("@lab/nonexistent-model-for-test-xyz")
	if err == nil {
		t.Fatalf("expected a fail-closed error for an unresolvable @lab alias, got base %q", got)
	}
	if got != "" {
		t.Errorf("fail-closed must return an empty base, got %q", got)
	}
	if !strings.HasPrefix(err.Error(), "LAB_") {
		t.Errorf("expected a LAB_* structured refusal, got %v", err)
	}
}
