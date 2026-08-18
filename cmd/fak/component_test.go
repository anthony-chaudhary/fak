package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestComponentCheckComposesPublishedContracts(t *testing.T) {
	root := repoRoot()
	contracts := []string{
		filepath.Join(root, "examples", "component-contracts", "radix-cache.json"),
		filepath.Join(root, "examples", "component-contracts", "paged-attention-kernel.json"),
		filepath.Join(root, "examples", "component-contracts", "cuda-runtime.json"),
	}
	args := []string{"check"}
	for _, path := range contracts {
		args = append(args, "--contract", path)
	}
	args = append(args, "--root", "cache.kv.radix@1")
	var stdout, stderr bytes.Buffer
	if code := runComponent(&stdout, &stderr, args); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	for _, want := range []string{"ALLOW stack", "cache.kv.radix@1", "kernel.paged-attention@2", "runtime.cuda@12.9", "WARN RECOMMENDATION_UNMET", "runtime.cuda.graphs"} {
		if !strings.Contains(stdout.String(), want) {
			t.Fatalf("output missing %q:\n%s", want, stdout.String())
		}
	}
}

func TestComponentCheckRefusesWithoutRequiredRuntime(t *testing.T) {
	root := repoRoot()
	args := []string{"check", "--contract", filepath.Join(root, "examples", "component-contracts", "radix-cache.json"), "--contract", filepath.Join(root, "examples", "component-contracts", "paged-attention-kernel.json"), "--root", "cache.kv.radix@1", "--json"}
	var stdout, stderr bytes.Buffer
	if code := runComponent(&stdout, &stderr, args); code != 3 {
		t.Fatalf("code=%d stdout=%s stderr=%s", code, stdout.String(), stderr.String())
	}
	for _, want := range []string{`"status": "refuse"`, `"wanted": "runtime.cuda.12"`, `"code": "UNSATISFIED_REQUIREMENT"`} {
		if !strings.Contains(stdout.String(), want) {
			t.Fatalf("output missing %q:\n%s", want, stdout.String())
		}
	}
}

func TestComponentCheckUsage(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := runComponent(&stdout, &stderr, nil); code != 2 || !strings.Contains(stderr.String(), "fak component check") {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	if _, err := os.Stat(filepath.Join(repoRoot(), "examples", "component-contracts", "README.md")); err != nil {
		t.Fatal(err)
	}
}
