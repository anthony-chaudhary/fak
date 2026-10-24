package main

import (
	"path/filepath"
	"strings"
	"testing"
)

// fak-test:runtime fast est=1s
func TestGuardAllowApproveScopeNamesExactLayer(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "operator", "allow.json")
	t.Setenv(guardAllowOverlayEnv, path)
	var out, errOut strings.Builder
	ov := guardAllowOverlay{}
	if code := runGuardAllowApprove(&out, &errOut, path, &ov, []string{"ap-0123456789ab"}, 0, false, false); code != 0 {
		t.Fatalf("approve exit=%d stderr=%s", code, errOut.String())
	}
	for _, want := range []string{path, "environment override", "same layer", "fak serve approval watcher"} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("approval output missing %q:\n%s", want, out.String())
		}
	}
}
