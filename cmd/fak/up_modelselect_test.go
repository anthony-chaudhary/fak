package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/macfit"
	"github.com/anthony-chaudhary/fak/internal/modelreg"
)

// TestTurnkeyTierAliasesAreUnshadowed pins every canonical turnkey alias to the
// embedded catalog and asserts it is quant-qualified, so a user registry.json
// overlay cannot silently re-bind a tier's default artifact to another quant.
func TestTurnkeyTierAliasesAreUnshadowed(t *testing.T) {
	for _, tier := range macfit.StandardTiers {
		tier := tier
		t.Run(tier.Name, func(t *testing.T) {
			alias := resolveTurnkeyModelRef(tier.Name)
			target, ok := modelreg.Catalog[alias]
			if !ok {
				t.Fatalf("tier %s resolved to %q, not an embedded catalog alias", tier.Name, alias)
			}
			if !strings.HasPrefix(target, "hf://") {
				t.Fatalf("tier %s alias %q target = %q, want hf:// URI", tier.Name, alias, target)
			}
			if got := inferQuantFromFilename(target); !strings.EqualFold(got, tier.QuantTier) {
				t.Fatalf("tier %s alias %q names quant %q, want %q (target %q)", tier.Name, alias, got, tier.QuantTier, target)
			}
		})
	}
}

// (a) resolveTurnkeyModelRef("27b") returns the unshadowed quant-qualified alias.
func TestResolveTurnkeyModelRef27BUsesQ4KM(t *testing.T) {
	got := resolveTurnkeyModelRef("27b")
	if got != "qwen38:27b-q4_k_m" {
		t.Fatalf("resolveTurnkeyModelRef(27b) = %q, want qwen38:27b-q4_k_m", got)
	}
	if _, ok := modelreg.Catalog[got]; !ok {
		t.Fatalf("resolved alias %q is not in the embedded catalog", got)
	}
}

// (b) A simulated user overlay shadowing "qwen38:27b" cannot change the turnkey
// 27B resolution: the resolver returns the quant-qualified alias and modelreg
// resolves it to the embedded Q4_K_M target, not the overlay's 2-bit artifact.
func TestResolveTurnkey27BIgnoresUserShadow(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("FAK_MODELS_DIR", dir)
	shadow := `{"qwen38:27b":"hf://unsloth/Qwen3.8-27B-GGUF@deadbeef/Qwen3.8-27B-UD-Q2_K_XL.gguf"}`
	if err := os.WriteFile(filepath.Join(dir, "registry.json"), []byte(shadow), 0o644); err != nil {
		t.Fatal(err)
	}

	alias := resolveTurnkeyModelRef("27b")
	if alias == "qwen38:27b" {
		t.Fatalf("turnkey 27B resolved to the user-shadowable bare alias %q", alias)
	}
	resolved, expanded := modelreg.Resolve(alias)
	if !expanded {
		t.Fatalf("Resolve(%q) did not expand; got %q", alias, resolved)
	}
	if resolved != modelreg.Catalog["qwen38:27b-q4_k_m"] {
		t.Fatalf("Resolve(%q) = %q, want embedded Q4_K_M target %q", alias, resolved, modelreg.Catalog["qwen38:27b-q4_k_m"])
	}
	if got := inferQuantFromFilename(resolved); got != "Q4_K_M" {
		t.Fatalf("resolved artifact quant = %q, want Q4_K_M (resolved %q)", got, resolved)
	}
}

// (c) A tier/artifact quant or size mismatch fails loud; the error names both the
// tier expectation and the resolved artifact.
func TestValidateTurnkeyArtifactMismatch(t *testing.T) {
	tier27 := macfit.ModelTier{Name: "27B", QuantTier: "Q4_K_M", WeightBytes: 16 * macfit.GiB}

	if err := validateTurnkeyArtifact(tier27, artifactDescriptor{
		Ref:      "qwen38:27b",
		Filename: "Qwen3.8-27B-UD-Q2_K_XL.gguf",
		Quant:    "Q2_K",
	}); err == nil {
		t.Fatal("quant mismatch did not fail; want loud error")
	} else if !strings.Contains(err.Error(), "Q2_K") || !strings.Contains(err.Error(), "Q4_K_M") {
		t.Fatalf("quant mismatch error = %q, want it to name both Q2_K and Q4_K_M", err)
	}

	// A 9.3 GiB artifact under a 16 GiB tier budget is out of tolerance even when
	// the filename yields no quant signal.
	if err := validateTurnkeyArtifact(tier27, artifactDescriptor{
		Ref:       "qwen38:27b",
		Filename:  "mystery.gguf",
		SizeBytes: 10 * 1000 * 1000 * 1000,
	}); err == nil {
		t.Fatal("size mismatch did not fail; want loud error")
	} else if !strings.Contains(err.Error(), "27B") || !strings.Contains(err.Error(), "GiB") {
		t.Fatalf("size mismatch error = %q, want it to name the tier and both sizes", err)
	}

	// A consistent descriptor passes.
	if err := validateTurnkeyArtifact(tier27, artifactDescriptor{
		Ref:       "qwen38:27b-q4_k_m",
		Filename:  "Qwen3.8-27B-Q4_K_M.gguf",
		Quant:     "Q4_K_M",
		SizeBytes: 16 * macfit.GiB,
	}); err != nil {
		t.Fatalf("consistent artifact rejected: %v", err)
	}
}

// (d) An explicit operator-supplied path or hf:// URI is an opinionated override
// and passes through, even when it would fail the alias consistency gate.
func TestValidateTurnkeyArtifactExplicitPassesThrough(t *testing.T) {
	tier27 := macfit.ModelTier{Name: "27B", QuantTier: "Q4_K_M", WeightBytes: 16 * macfit.GiB}
	art := artifactDescriptor{
		Ref:      "hf://unsloth/Qwen3.8-27B-GGUF@deadbeef/Qwen3.8-27B-UD-Q2_K_XL.gguf",
		Filename: "Qwen3.8-27B-UD-Q2_K_XL.gguf",
		Quant:    "Q2_K",
		Explicit: true,
	}
	if err := validateTurnkeyArtifact(tier27, art); err != nil {
		t.Fatalf("explicit hf:// URI rejected: %v", err)
	}

	dir := t.TempDir()
	local := filepath.Join(dir, "custom.gguf")
	if err := os.WriteFile(local, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if !turnkeyRefIsExplicit(local) {
		t.Fatalf("turnkeyRefIsExplicit(%q) = false, want true for an existing file", local)
	}
	if resolved := resolveTurnkeyModelRef(local); resolved != local {
		t.Fatalf("resolveTurnkeyModelRef(explicit path) = %q, want passthrough %q", resolved, local)
	}
	if !turnkeyRefIsExplicit(art.Ref) {
		t.Fatalf("turnkeyRefIsExplicit(hf:// URI) = false, want true")
	}
	if turnkeyRefIsExplicit("27B") {
		t.Fatal("turnkeyRefIsExplicit(tier alias) = true, want false")
	}
}

// The consistency gate must evaluate the RESOLVED target, not the alias string:
// an overlay shadowing the canonical quant-qualified alias with a 2-bit artifact
// must fail loud, because the alias itself still reads as Q4_K_M.
func TestValidateTurnkeyResolvedTargetCatchesShadow(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("FAK_MODELS_DIR", dir)
	shadow := `{"qwen38:27b-q4_k_m":"hf://unsloth/Qwen3.8-27B-GGUF@deadbeef/Qwen3.8-27B-UD-Q2_K_XL.gguf"}`
	if err := os.WriteFile(filepath.Join(dir, "registry.json"), []byte(shadow), 0o644); err != nil {
		t.Fatal(err)
	}
	tier27 := macfit.ModelTier{Name: "27B", QuantTier: "Q4_K_M", WeightBytes: 16 * macfit.GiB}

	alias := resolveTurnkeyModelRef("27b")
	resolved, expanded := modelreg.Resolve(alias)
	if !expanded || !strings.Contains(resolved, "UD-Q2_K_XL") {
		t.Fatalf("test setup: Resolve(%q) = (%q, %v); want the shadowed Q2 target", alias, resolved, expanded)
	}
	if err := validateTurnkeyArtifact(tier27, describeTurnkeyArtifact(resolved, false)); err == nil {
		t.Fatal("shadowed resolved target passed the gate; want loud quant mismatch")
	}
}
