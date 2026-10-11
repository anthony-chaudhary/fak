package compute

import (
	"os"
	"path/filepath"
	"regexp"
	"testing"
)

// TestV41IndexSelectScratchIsCoherent guards the gfx1151 block-drop race: the
// index-select phases exchange Scratch and Status through storage buffers
// across the waves of one workgroup, and barrier() alone does not make another
// wave's writes visible unless the buffer is declared coherent.
// fak-test:runtime fast est=1ms lane=default
func TestV41IndexSelectScratchIsCoherent(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("shaders", "v41_index_select.comp"))
	if err != nil {
		t.Fatalf("read v41_index_select.comp: %v", err)
	}
	for _, block := range []string{"ScratchBuf", "StatusBuf"} {
		decl := regexp.MustCompile(`(?m)^layout\([^)]*\)\s*([\w\s]*?)\bbuffer\s+` + block + `\b`).FindStringSubmatch(string(raw))
		if decl == nil {
			t.Fatalf("v41_index_select.comp declares no %s", block)
		}
		if !regexp.MustCompile(`\bcoherent\b`).MatchString(decl[1]) {
			t.Errorf("%s is shared across waves between barriers and must be declared coherent", block)
		}
	}
}
