package modelreg

import "testing"

func TestInkling2ShortAliasesResolveToMiniPin(t *testing.T) {
	withCacheRoot(t)
	want, expanded := Resolve("inkling2-mini")
	if !expanded || want != "hf://unsloth/Inkling-2-Mini-GGUF@8078d149286eec7193795413b9e0643da6d11fa1/Mainline_Support/Inkling-2-Mini-UD-Q3_K_XL.gguf_file" {
		t.Fatalf("Resolve(inkling2-mini) = %q, %v", want, expanded)
	}
	for _, alias := range []string{"inkling", "inkling-2-mini"} {
		if got, ok := Resolve(alias); !ok || got != want {
			t.Fatalf("Resolve(%q) = %q, %v; want %q", alias, got, ok, want)
		}
	}
}
