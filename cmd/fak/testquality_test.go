package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestRunTestQualityMalformedBaselineIsHardError(t *testing.T) {
	root := t.TempDir()
	if err := os.MkdirAll(filepath.Join(root, "internal", "testquality"), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "internal", "testquality", "baseline.txt"), []byte("bad row\n"), 0644); err != nil {
		t.Fatal(err)
	}
	var out, errOut bytes.Buffer
	code := runTestQuality(root, &out, &errOut, nil)
	if code == 0 || !strings.Contains(errOut.String(), "line 1") {
		t.Fatalf("code=%d stderr=%q", code, errOut.String())
	}
}
