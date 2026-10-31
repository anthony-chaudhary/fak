package debtlane

import (
	"os"
	"path/filepath"
	"testing"
)

// TestReadRuntimeProofsCapturesHyphenatedLaneName is the before-red/after-green
// regression witness for #14114: the "lane" capture class in readRuntimeProofs
// was [A-Za-z0-9_], which stops at '-', so hyphenated lane names never entered
// the runtime-proof set. It must stay red on the old regex and green once the
// class includes '-'.
func TestReadRuntimeProofsCapturesHyphenatedLaneName(t *testing.T) {
	tmp := t.TempDir()
	matDir := filepath.Join(tmp, "internal", "maturity")
	if err := os.MkdirAll(matDir, 0o755); err != nil {
		t.Fatal(err)
	}
	proofJSON := `[{"lane":"cleancore"},{"lane":"my-lane-x"}]`
	if err := os.WriteFile(filepath.Join(matDir, "runtime-proofs.json"), []byte(proofJSON), 0o644); err != nil {
		t.Fatal(err)
	}

	set := readRuntimeProofs(tmp)
	if !set["cleancore"] {
		t.Errorf("non-hyphenated lane cleancore not captured; got %v", set)
	}
	if !set["my-lane-x"] {
		t.Errorf("hyphenated lane my-lane-x not captured by readRuntimeProofs; got %v", set)
	}
}
