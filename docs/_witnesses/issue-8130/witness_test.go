package issue8130

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

func TestQwen38UpstreamSupportMapContract(t *testing.T) {
	path := filepath.Join("..", "..", "notes", "qwen38-upstream-support-map-2026-08-26.md")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	doc := string(raw)
	required := []string{"#8130", "Stale after:", "GDN/full-attention", "NextN/MTP", "Multimodal projector", "Tool-call chat template", "Long context", "FP8/GGUF", "Prefix/KV cache", "PRESENT", "PARTIAL", "ABSENT", "DEFAULT", "OPTIONAL-MODULE", "RECIPE", "WATCH", "EXCLUDE"}
	for _, want := range required {
		if !strings.Contains(doc, want) {
			t.Errorf("map missing %q", want)
		}
	}
	runtimes := []string{"Transformers", "vLLM", "llama.cpp", "SGLang", "MLX-LM"}
	for _, runtime := range runtimes {
		if !strings.Contains(doc, "| "+runtime+" |") {
			t.Errorf("runtime row missing %s", runtime)
		}
	}
	pins := regexp.MustCompile(`@[0-9a-f]{40}`).FindAllString(doc, -1)
	if len(pins) < 6 {
		t.Errorf("pinned primary-source groups=%d, want >=6", len(pins))
	}
	decisions := regexp.MustCompile(`(?m)^\| [0-9]+ \| (DEFAULT|OPTIONAL-MODULE|RECIPE|WATCH|EXCLUDE) \|`).FindAllString(doc, -1)
	if len(decisions) < 8 {
		t.Errorf("typed decisions=%d, want >=8", len(decisions))
	}
	if !strings.Contains(doc, "vendor-reported comparisons, not matched fak evidence") {
		t.Error("benchmark disclosure caveat missing")
	}
	if !strings.Contains(doc, "runtime version, flags, accelerator, quality result, and engine receipt") {
		t.Error("reproducibility fields missing")
	}
}
