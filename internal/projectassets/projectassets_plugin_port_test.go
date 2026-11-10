package projectassets

import (
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"testing"
)

// TestDefaultOpenCodePluginMatchesPrivateGuardSource witnesses that the
// canonical const renders byte-identical to the private tree's committed
// dos-proof-guard.js (closes the flowspec gap "opencode-plugin-divergence":
// SyncOpenCodePlugin generation stays lossless with the private guard).
// The public repo does not track its own .opencode/plugins asset, so the
// comparison source is the sibling private root's tracked copy when present;
// FAK_DOS_PROOF_GUARD_SOURCE overrides it. Absent both, the test skips.
func TestDefaultOpenCodePluginMatchesPrivateGuardSource(t *testing.T) {
	candidates := []string{}
	if p := os.Getenv("FAK_DOS_PROOF_GUARD_SOURCE"); p != "" {
		candidates = append(candidates, p)
	}
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	candidates = append(candidates, filepath.Join(root, "..", "fak-private", ".opencode", "plugins", "dos-proof-guard.js"))

	var src []byte
	for _, c := range candidates {
		if b, err := os.ReadFile(c); err == nil {
			src = b
			break
		}
	}
	if src == nil {
		t.Skip("private dos-proof-guard.js source not present; nothing to compare against")
	}

	got := []byte(DefaultOpenCodePlugin)
	if len(got) != len(src) {
		t.Fatalf("DefaultOpenCodePlugin length %d != source length %d", len(got), len(src))
	}
	for i := range got {
		if got[i] != src[i] {
			t.Fatalf("DefaultOpenCodePlugin diverges from source at byte offset %d", i)
		}
	}
	sum := sha256.Sum256(src)
	t.Logf("const matches private guard source byte-for-byte: len=%d sha256=%s", len(src), hex.EncodeToString(sum[:]))
}
