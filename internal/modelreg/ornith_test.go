package modelreg

import (
	"strings"
	"testing"
)

// TestResolveOrnithAliases pins the Ornith 1.0 (Qwen3.5-family) run-by-name aliases
// added for epic #1026 child #1029: each seeded name must expand to the exact embedded
// target so `fak run ornith:9b` / `fak serve --gguf ornith:35b-gguf` resolve the way
// qwen2.5/smollm2 already do. Targets were verified fetchable (HTTP 200 per repo, each
// GGUF filename confirmed present) when seeded; this test guards the mapping, not the
// network.
func TestResolveOrnithAliases(t *testing.T) {
	withCacheRoot(t)
	want := map[string]string{
		"ornith":          "hf://deepreinforce-ai/Ornith-1.0-9B-GGUF/ornith-1.0-9b-Q4_K_M.gguf",
		"ornith:9b-gguf":  "hf://deepreinforce-ai/Ornith-1.0-9B-GGUF/ornith-1.0-9b-Q4_K_M.gguf",
		"ornith:9b":       "hf://deepreinforce-ai/Ornith-1.0-9B",
		"ornith:35b":      "hf://deepreinforce-ai/Ornith-1.0-35B",
		"ornith:35b-gguf": "hf://deepreinforce-ai/Ornith-1.0-35B-GGUF/ornith-1.0-35b-Q4_K_M.gguf",
		"ornith:35b-fp8":  "hf://deepreinforce-ai/Ornith-1.0-35B-FP8",
		"ornith:397b":     "hf://deepreinforce-ai/Ornith-1.0-397B",
		"ornith:397b-fp8": "hf://deepreinforce-ai/Ornith-1.0-397B-FP8",
	}
	for name, target := range want {
		got, expanded := Resolve(name)
		if !expanded {
			t.Errorf("Resolve(%q) did not expand; got %q", name, got)
			continue
		}
		if got != target {
			t.Errorf("Resolve(%q) = %q; want %q", name, got, target)
		}
	}
}

// TestOrnithBareAndGGUFRunnable: the bare family name and the explicit -gguf alias both
// seed the same laptop-runnable single-file GGUF (the Catalog's "just works" convention,
// like smollm2), so `fak run ornith` downloads one concrete .gguf with no extra steps.
func TestOrnithBareAndGGUFRunnable(t *testing.T) {
	withCacheRoot(t)
	bare, _ := Resolve("ornith")
	gguf, _ := Resolve("ornith:9b-gguf")
	if bare != gguf {
		t.Fatalf("bare ornith (%q) and ornith:9b-gguf (%q) must seed the same runnable GGUF", bare, gguf)
	}
	if !strings.HasSuffix(bare, ".gguf") {
		t.Fatalf("bare ornith must resolve to a single-file GGUF, got %q", bare)
	}
}
