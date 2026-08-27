package qwen38ladder

import (
	"os"
	"path/filepath"
	"regexp"
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
		"<!-- qwen38-frontdoor:begin -->",
		"ACCEPTED",
		"APPROXIMATE",
		"DIAGNOSTIC",
		"3.3 vs 6.966061",
		"~47%",
		"P31/T64",
		"P32/T64",
		"0/5 exact",
		"This is the one current index for Qwen performance updates",
		"rows are envelopes, not a timeline",
		"docs/_witnesses/",
		"q38-bf16-tp2-arithmetic-ttfc",
		"q38-q4km-native-cuda-a100-cold-decode",
		"q38-q4km-native-metal-m3pro-fullrun",
		"q36-q4km-metal-m3pro-parity-bar",
		"AMD/Vulkan or CPU-only",
		"Newer code awaiting comparable remeasurement",
		"Replace atomically",
		"Native/performance rows must name the fak-native engine",
	}
	for _, want := range required {
		if !strings.Contains(page, want) {
			t.Errorf("Qwen performance index missing %q", want)
		}
	}

	for _, frontDoor := range []string{"README.md", "llms.txt", "llms-full.txt", "INDEX.md", "BENCHMARK-AUTHORITY.md"} {
		content, err := os.ReadFile(filepath.Join(root, frontDoor))
		if err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(string(content), "docs/benchmarks/QWEN-PERFORMANCE-INDEX.md") {
			t.Errorf("%s does not route readers to the Qwen performance index", frontDoor)
		}
	}
	directoryIndex, err := os.ReadFile(filepath.Join(root, "docs", "benchmarks", "README.md"))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(directoryIndex), "QWEN-PERFORMANCE-INDEX.md") {
		t.Error("docs/benchmarks/README.md does not route readers to the Qwen performance index")
	}

	for _, frontDoor := range []string{"docs/benchmarks/README.md", "INDEX.md", "llms-full.txt"} {
		content, err := os.ReadFile(filepath.Join(root, frontDoor))
		if err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(string(content), "QWEN38-27B-LATEST.md") {
			t.Errorf("%s does not route readers to the detailed Qwen3.8 result page", frontDoor)
		}
	}
}

func TestQwenPerformanceIndexCurrentRowsHaveUniqueEnvelopeAndFreshness(t *testing.T) {
	body, err := os.ReadFile(filepath.Join("..", "..", "docs", "benchmarks", "QWEN-PERFORMANCE-INDEX.md"))
	if err != nil {
		t.Fatal(err)
	}
	date := regexp.MustCompile(`observed \*\*\d{4}-\d{2}-\d{2}\*\*; review by \*\*\d{4}-\d{2}-\d{2}\*\*`)
	seen := map[string]bool{}
	for _, line := range strings.Split(string(body), "\n") {
		if !strings.HasPrefix(line, "| **CURRENT** | `") {
			continue
		}
		fields := strings.Split(line, "|")
		if len(fields) < 6 {
			t.Fatalf("malformed CURRENT row: %s", line)
		}
		key := strings.Trim(strings.TrimSpace(fields[2]), "`")
		if seen[key] {
			t.Errorf("duplicate CURRENT envelope key %q", key)
		}
		seen[key] = true
		if !date.MatchString(line) {
			t.Errorf("CURRENT row %q lacks observed and review-by dates", key)
		}
	}
	if len(seen) == 0 {
		t.Fatal("no CURRENT Qwen performance rows found")
	}
}
