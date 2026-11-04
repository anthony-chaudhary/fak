package main

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/modelreg"
)

// installTransferSpy swaps the byte-transfer seam for a counting spy and
// restores the production transfer on cleanup. It returns the spy's call log.
func installTransferSpy(t *testing.T) *[]string {
	t.Helper()
	var calls []string
	prev := modelTransfer
	modelTransfer = func(ctx context.Context, ref string, progress io.Writer) (string, error) {
		calls = append(calls, ref)
		return "", errors.New("spy: no transfer in unit tests")
	}
	t.Cleanup(func() { modelTransfer = prev })
	return &calls
}

// TestModelUnknownTypeRejectedBeforeTransfer pins the pre-transfer capability
// gate: an alias-declared model family absent from the compiled capability
// table is refused with ErrUnknownModelType BEFORE any download (the fetch
// spy observes zero transfers), the refusal exits 1 and names no registry
// contents, while every registered family resolves and reaches the transfer
// seam exactly as before.
func TestModelUnknownTypeRejectedBeforeTransfer(t *testing.T) {
	calls := installTransferSpy(t)
	modelsDir := t.TempDir()
	t.Setenv("FAK_MODELS_DIR", modelsDir)
	// A user-registry alias declaring a family outside the compiled table:
	// the only surface where an unknown type is DECLARED before transfer.
	reg := map[string]string{"exotic:7b": "hf://owner/exotic-repo/exotic-7b-Q4_K_M.gguf"}
	raw, err := json.Marshal(reg)
	if err != nil {
		t.Fatalf("marshal registry: %v", err)
	}
	if err := os.WriteFile(filepath.Join(modelsDir, "registry.json"), raw, 0o644); err != nil {
		t.Fatalf("write registry: %v", err)
	}

	// Unknown type refuses before transfer: sentinel, exit 1, zero fetches.
	if err := modelTypeGate("exotic:7b", true); !errors.Is(err, ErrUnknownModelType) {
		t.Fatalf("modelTypeGate(exotic:7b) = %v, want ErrUnknownModelType", err)
	}
	plan := planModelTransfer("exotic:7b", "pull", 1)
	if plan.kind != modelTransferBlocked || plan.exit != 1 {
		t.Fatalf("plan = {kind:%v exit:%d}, want blocked/exit 1", plan.kind, plan.exit)
	}
	if len(*calls) != 0 {
		t.Fatalf("fetch spy observed %d transfers after refusal, want 0", len(*calls))
	}
	// Rejection probe: the refusal states the type is unsupported and
	// enumerates no known family (no registry contents in the message).
	if !strings.Contains(plan.message, `"exotic"`) || !strings.Contains(plan.message, "not supported") {
		t.Fatalf("refusal %q must state the requested type is unsupported", plan.message)
	}
	for _, c := range modelCapabilities {
		if strings.Contains(plan.message, c.Family) {
			t.Fatalf("refusal %q names registry family %q", plan.message, c.Family)
		}
	}

	// The historical not-found path is unchanged for a bare unknown name that
	// declares no alias: it still refuses pre-transfer with notFoundExit.
	bare := planModelTransfer("totally-bare-name", "pull", 2)
	if bare.kind != modelTransferNotFound || bare.exit != 2 {
		t.Fatalf("bare unknown name = {kind:%v exit:%d}, want notFound/exit 2", bare.kind, bare.exit)
	}

	// Non-expanded refs declare no type and pass through the gate.
	if err := modelTypeGate("hf://owner/exotic-repo/file.gguf", false); err != nil {
		t.Fatalf("bare URI must pass the gate, got %v", err)
	}
	if err := modelTypeGate("exotic:7b", false); err != nil {
		t.Fatalf("non-expanded ref must pass the gate, got %v", err)
	}

	// Registered families proceed: each embedded-catalog alias resolves to its
	// catalog target and reaches the transfer seam, exactly as today.
	registered := map[string]string{
		"smollm2":            "smollm2",
		"qwen2.5:0.5b":       "qwen2.5:0.5b",
		"qwen2.5-coder:3b":   "qwen2.5-coder:3b",
		"qwen38:27b":         "qwen38:27b",
		"qwen3.8-70b-q4_k_m": "qwen3.8-70b-q4_k_m",
		"ornith":             "ornith",
		"glm-5.3-flash":      "glm-5.3-flash",
		"glm53":              "glm53",
	}
	for alias, catalogKey := range registered {
		wantRef := embeddedCatalogTarget(t, catalogKey)
		got := planModelTransfer(alias, "pull", 1)
		if got.kind != modelTransferFetch {
			t.Fatalf("%s: plan kind = %v, want fetch", alias, got.kind)
		}
		if got.ref != wantRef {
			t.Fatalf("%s: ref = %q, want %q", alias, got.ref, wantRef)
		}
	}
	if len(*calls) != 0 {
		t.Fatalf("planning transferred %d refs, want 0 (transfer is the caller's step)", len(*calls))
	}
}

// embeddedCatalogTarget reads the embedded catalog target without going
// through modelreg.Resolve, so the test asserts the exact ref the gate lets
// through instead of re-deriving it through the same code under test.
func embeddedCatalogTarget(t *testing.T, key string) string {
	t.Helper()
	target, ok := modelreg.Catalog[key]
	if !ok {
		t.Fatalf("embedded catalog has no key %q", key)
	}
	return target
}

// TestServeCapabilityReportCompiledFamilies pins the serve-fit capability
// report: exact deterministic lines for the compiled table, and the shared
// table property — every listed family is admitted by the pre-transfer gate
// while an unlisted one is refused with the leaf-1 sentinel.
func TestServeCapabilityReportCompiledFamilies(t *testing.T) {
	const want = "fak capabilities: version=test-v1 chip=linux/amd64 backends=cpu-ref\n" +
		"  family=glm-5.3-flash formats=bf16,fp8 flags=- env=-\n" +
		"  family=glm53 formats=bf16,fp8 flags=- env=-\n" +
		"  family=ornith formats=fp8,q4_k_m flags=--gguf env=FAK_Q4K\n" +
		"  family=qwen2.5 formats=q4_k_m,q8_0 flags=--gguf env=FAK_Q4K\n" +
		"  family=qwen2.5-coder formats=q4_k_m flags=--gguf env=FAK_Q4K\n" +
		"  family=qwen3.8 formats=fp8,q4_k_m,ud_q2_k_xl flags=--gguf env=FAK_Q4K,FAK_STREAM_Q4K\n" +
		"  family=qwen38 formats=fp8,q4_k_m,ud_q2_k_xl flags=--gguf env=FAK_Q4K,FAK_STREAM_Q4K\n" +
		"  family=smollm2 formats=q8_0 flags=--gguf env=-\n"
	got := serveCapabilityReport("test-v1", "linux/amd64", []string{"cpu-ref"})
	if got != want {
		t.Fatalf("report mismatch:\ngot:\n%s\nwant:\n%s", got, want)
	}
	if again := serveCapabilityReport("test-v1", "linux/amd64", []string{"cpu-ref"}); again != got {
		t.Fatalf("report is not deterministic: %q vs %q", again, got)
	}

	// One table, two gates: listed families are admitted, unlisted refused.
	for _, c := range modelCapabilities {
		if err := modelTypeGate(c.Family+":probe", true); err != nil {
			t.Fatalf("listed family %s refused by the gate: %v", c.Family, err)
		}
	}
	if err := modelTypeGate("unlisted:probe", true); !errors.Is(err, ErrUnknownModelType) {
		t.Fatalf("unlisted family: got %v, want ErrUnknownModelType", err)
	}

	// Context-token validation stays byte-for-byte: 0=auto, positive ok, negative errors.
	if err := validateServeNativeContextTokens(0); err != nil {
		t.Fatalf("0 tokens must be auto, got %v", err)
	}
	if err := validateServeNativeContextTokens(4096); err != nil {
		t.Fatalf("positive tokens must pass, got %v", err)
	}
	if err := validateServeNativeContextTokens(-1); err == nil ||
		!strings.Contains(err.Error(), "must be 0 (auto) or positive") {
		t.Fatalf("negative tokens: got %v, want the 0-or-positive refusal", err)
	}
}
