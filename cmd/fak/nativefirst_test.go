package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestNativeFirstCanonicalCrossIndex(t *testing.T) {
	root := repoRoot()
	for _, rel := range []string{"README.md", "AGENTS.md", "llms.txt"} {
		b, err := os.ReadFile(filepath.Join(root, rel))
		if err != nil {
			t.Fatal(err)
		}
		if !containsAll(string(b), "docs/native-inference-goal.md", "llama.cpp") {
			t.Fatalf("%s does not cross-index the native inference invariant", rel)
		}
	}
	b, err := os.ReadFile(filepath.Join(root, "docs", "native-inference-goal.md"))
	if err != nil {
		t.Fatal(err)
	}
	if !containsAll(string(b), "must beat llama.cpp", "never selects llama.cpp as a fallback", "explicit external reference", "Did the model execute inside fak") {
		t.Fatal("canonical native inference doctrine is incomplete")
	}
}

func TestQwen38PerformancePreferenceProjected(t *testing.T) {
	root := repoRoot()
	required := []string{
		"New native-performance work prefers Qwen3.8.",
		"Qwen3.6 is allowed only when the task states an explicit task-specific exception, such as regression, compatibility, historical comparison, or a hardware/artifact constraint.",
		"Preserve historical Qwen3.6 artifacts; do not rename or rewrite them as Qwen3.8 evidence.",
	}
	for _, rel := range []string{"AGENTS.md", "docs/native-inference-goal.md"} {
		b, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(rel)))
		if err != nil {
			t.Fatal(err)
		}
		if !containsAll(string(b), required...) {
			t.Fatalf("%s does not project the exact Qwen3.8 performance preference and Qwen3.6 exception rule", rel)
		}
	}
}

func containsAll(s string, wants ...string) bool {
	for _, want := range wants {
		if !strings.Contains(s, want) {
			return false
		}
	}
	return true
}

func TestNativeFirstLintClassifiesContexts(t *testing.T) {
	bad, err := scanNativeFirst(strings.NewReader("Qwen3.8 native performance defaults to llama.cpp.\nNative falls back to llama-server.\n"))
	if err != nil {
		t.Fatal(err)
	}
	if len(bad) != 2 {
		t.Fatalf("findings=%+v", bad)
	}
	good, err := scanNativeFirst(strings.NewReader("Benchmark fak-native against llama.cpp.\nUse llama.cpp explicitly for parity diagnosis.\n"))
	if err != nil {
		t.Fatal(err)
	}
	if len(good) != 0 {
		t.Fatalf("findings=%+v", good)
	}
}
