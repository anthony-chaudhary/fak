package main

import (
	"bytes"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/macfit"
	"github.com/anthony-chaudhary/fak/internal/modelreg"
)

func TestTurnkeyCatalogResolution(t *testing.T) {
	// Keep model resolution independent of user-local registry and cache state.
	t.Setenv("FAK_MODELS_DIR", t.TempDir())
	t.Setenv("FAK_MODEL_DIR", t.TempDir())
	t.Setenv("FAK_UP_STATIC_ONLY", "1")

	want70B, registered := modelreg.Catalog["qwen38:70b"]
	if !registered || !strings.HasPrefix(want70B, "hf://") {
		t.Fatalf("qwen38:70b catalog entry = %q, present=%v; want registered hf:// URI", want70B, registered)
	}

	t.Run("64 GiB dry run", func(t *testing.T) {
		var stdout, stderr bytes.Buffer
		runTurnkeyUp(nil, &stdout, &stderr, []string{"--dry-run", "--memory-gib", "64"})
		if stderr.Len() != 0 {
			t.Fatalf("fak up dry-run stderr = %q, want empty", stderr.String())
		}
		for _, want := range []string{"Model Tier     : 70B", "Resolved URI   : " + want70B} {
			if !strings.Contains(stdout.String(), want) {
				t.Fatalf("fak up dry-run output missing %q:\n%s", want, stdout.String())
			}
		}
	})

	for _, tier := range macfit.StandardTiers {
		tier := tier
		t.Run("tier "+tier.Name, func(t *testing.T) {
			aliases := make(map[string]string, 2)
			for label, input := range map[string]string{"name": tier.Name, "model ID": tier.ModelID} {
				alias := resolveTurnkeyModelRef(input)
				aliases[label] = alias
				resolved, expanded := modelreg.Resolve(alias)
				if !expanded || !strings.HasPrefix(resolved, "hf://") {
					t.Errorf("%s %q resolved via %q to (%q, %v); want registered hf:// URI", label, input, alias, resolved, expanded)
				}
			}
			if aliases["name"] != aliases["model ID"] {
				t.Errorf("name %q and model ID %q resolve to different aliases: %q != %q", tier.Name, tier.ModelID, aliases["name"], aliases["model ID"])
			}
			if tier.Name == "70B" && aliases["name"] != "qwen38:70b" {
				t.Errorf("70B name resolves to %q, want qwen38:70b", aliases["name"])
			}
		})
	}
}
