package main

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestRunTestQualityMalformedBaselineIsHardError(t *testing.T) {
	root := t.TempDir()
	if out, err := exec.Command("git", "-C", root, "init", "--quiet").CombinedOutput(); err != nil {
		t.Fatalf("git init: %v: %s", err, out)
	}
	testPath := filepath.Join(root, "sample_test.go")
	if err := os.WriteFile(testPath, []byte("package sample\n\nimport \"testing\"\n\nfunc TestSample(t *testing.T) { t.Fatal(\"sentinel\") }\n"), 0644); err != nil {
		t.Fatal(err)
	}
	if out, err := exec.Command("git", "-C", root, "add", "sample_test.go").CombinedOutput(); err != nil {
		t.Fatalf("git add: %v: %s", err, out)
	}
	if err := os.MkdirAll(filepath.Join(root, "internal", "testquality"), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "internal", "testquality", "baseline.txt"), []byte("bad row\n"), 0644); err != nil {
		t.Fatal(err)
	}
	var out, errOut bytes.Buffer
	code := runTestQuality(&out, &errOut, []string{"--root", root})
	if code == 0 || !strings.Contains(errOut.String(), "line 1") {
		t.Fatalf("code=%d stderr=%q", code, errOut.String())
	}
}
