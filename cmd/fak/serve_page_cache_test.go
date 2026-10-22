package main

import (
	"path/filepath"
	"testing"
)

// A serve without a GGUF, or whose shards cannot be measured, reports no page-cache case.
// fak-test:runtime fast est=5ms lane=default
func TestServeStartupResidencyUnmeasuredIsNil(t *testing.T) {
	if pc := serveStartupResidency(""); pc != nil {
		t.Fatalf("serveStartupResidency(\"\") = %+v, want nil", pc)
	}
	if pc := serveStartupResidency(filepath.Join(t.TempDir(), "absent-00001-of-00002.gguf")); pc != nil {
		t.Fatalf("serveStartupResidency(missing shards) = %+v, want nil", pc)
	}
}
