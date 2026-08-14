package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestSpeedABNotYetExit(t *testing.T) {
	path := filepath.Join(t.TempDir(), "m.json")
	if err := os.WriteFile(path, []byte(`{"schema":"fak.speed-ab.v1","runs":[]}`), 0o600); err != nil {
		t.Fatal(err)
	}
	var out strings.Builder
	if code := runSpeedAB(&out, &out, []string{"--manifest", path}); code != 3 {
		t.Fatalf("code=%d out=%s", code, out.String())
	}
	if !strings.Contains(out.String(), `"verdict": "NOT_YET"`) {
		t.Fatalf("out=%s", out.String())
	}
}
