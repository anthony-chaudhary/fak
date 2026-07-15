package main

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
)

func TestAntipatternScorecardPinsCheckerBytes(t *testing.T) {
	root := t.TempDir()
	for _, rel := range []string{"internal/antipattern/antipattern.go", "internal/antipattern/detect.go"} {
		path := filepath.Join(root, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte("package antipattern\n"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	var out, errOut bytes.Buffer
	code := runAntipatternScorecard(&out, &errOut, []string{"--workspace", root, "--commits", "0", "--json"})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
}
