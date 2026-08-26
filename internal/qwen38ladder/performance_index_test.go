package qwen38ladder

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestQwenPerformanceIndexRoutesCurrentResults keeps the Qwen result front door
// and its durable worker-publishing contract from silently disappearing again.
func TestQwenPerformanceIndexRoutesCurrentResults(t *testing.T) {
	root := filepath.Join("..", "..")
	indexPath := filepath.Join(root, "docs", "benchmarks", "QWEN-PERFORMANCE-INDEX.md")
	body, err := os.ReadFile(indexPath)
	if err != nil {
		t.Fatal(err)
	}
	page := string(body)
	required := []string{
		"This is the one current index for Qwen performance updates",
		"docs/_witnesses/",
		"Qwen3.8-27B BF16",
		"Qwen3.8-27B Q4_K_M, A100-class CUDA, fak-native",
		"Qwen3.8-27B Q4_K_M, Apple M3 Pro Metal, fak-native",
		"Qwen3.6-27B Q4_K_M, Apple M3 Pro",
		"AMD/Vulkan or CPU-only",
		"Update this index in the same landing",
		"Native/performance rows must name the fak-native engine",
	}
	for _, want := range required {
		if !strings.Contains(page, want) {
			t.Errorf("Qwen performance index missing %q", want)
		}
	}

	for _, frontDoor := range []string{"README.md", "llms.txt", "BENCHMARK-AUTHORITY.md"} {
		content, err := os.ReadFile(filepath.Join(root, frontDoor))
		if err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(string(content), "docs/benchmarks/QWEN-PERFORMANCE-INDEX.md") {
			t.Errorf("%s does not route readers to the Qwen performance index", frontDoor)
		}
	}
}
