package main

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// TestResolveKVPrecisionValue covers the serve --kv-precision resolver: empty/env
// fallback resolves to f32, q8 tokens resolve to q8_0, and a typo is reported as
// invalid so serve can refuse rather than silently serving the wrong tier.
func TestResolveKVPrecisionValue(t *testing.T) {
	t.Setenv("FAK_UP_KV_PRECISION", "")
	if got, ok := resolveKVPrecisionValue(""); !ok || got != "" {
		t.Fatalf("empty = (%q,%v), want the zero value true (planner reads '' as f32)", got, ok)
	}
	if got, ok := resolveKVPrecisionValue("q8_0"); !ok || got != model.KVPrecisionQ8_0 {
		t.Fatalf("q8_0 = (%s,%v), want (q8_0,true)", got, ok)
	}
	t.Setenv("FAK_UP_KV_PRECISION", "q8")
	if got, ok := resolveKVPrecisionValue(""); !ok || got != model.KVPrecisionQ8_0 {
		t.Fatalf("env q8 = (%s,%v), want (q8_0,true)", got, ok)
	}
	if _, ok := resolveKVPrecisionValue("bogus"); ok {
		t.Fatal("a typo must be reported invalid")
	}
}

// TestResolveUpKVPrecisionPinsEnv proves the up resolver pins the resolved tier into
// FAK_UP_KV_PRECISION so the loader dep path and the per-request planner agree.
func TestResolveUpKVPrecisionPinsEnv(t *testing.T) {
	t.Setenv("FAK_UP_KV_PRECISION", "")
	prec, err := resolveUpKVPrecision("")
	if err != nil || prec != model.KVPrecisionFP32 {
		t.Fatalf("default = (%s,%v), want (f32,nil)", prec, err)
	}
	prec, err = resolveUpKVPrecision("q8_0")
	if err != nil || prec != model.KVPrecisionQ8_0 {
		t.Fatalf("q8_0 = (%s,%v), want (q8_0,nil)", prec, err)
	}
	// The planner reads this env when its own InKernelPlannerConfig.KVPrecision is unset.
	if got, ok := resolveKVPrecisionValue(""); !ok || got != model.KVPrecisionQ8_0 {
		t.Fatalf("env not pinned: got (%s,%v)", got, ok)
	}
	if _, err := resolveUpKVPrecision("nope"); err == nil {
		t.Fatal("unknown token must error")
	}
}

// TestValidateServeKVPrecisionRefusesTypo pins startup refusal on a bad token.
func TestValidateServeKVPrecisionRefusesTypo(t *testing.T) {
	t.Setenv("FAK_UP_KV_PRECISION", "")
	if err := validateServeKVPrecision(""); err != nil {
		t.Fatalf("empty must be valid: %v", err)
	}
	if err := validateServeKVPrecision("q8_0"); err != nil {
		t.Fatalf("q8_0 must be valid: %v", err)
	}
	if err := validateServeKVPrecision("q9"); err == nil {
		t.Fatal("a typo must be refused at startup")
	}
}
