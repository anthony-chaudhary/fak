package memq

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/memoryread"
)

// buildNotesDigestStore writes a three-note store exercising the REAL
// DefaultArtifactVerifier run in this checkout, mirroring
// cmd/fak/memory_recall_test.go's fixtureMemoryStore: a note naming a path that
// exists (fresh), a note naming a deleted package (stale → withheld), and a
// prose note with nothing checkable (unverified, rendered whole).
func buildNotesDigestStore(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	files := map[string]string{
		"MEMORY.md": "# Memory index\n\n" +
			"- [Gate helper](fresh.md) — still true\n" +
			"- [Moved fix](stale.md) — names a gone artifact\n" +
			"- [Preference](prose.md) — prose only\n",
		"fresh.md": "---\nname: gate-helper\ndescription: where the algebra lives\nmetadata:\n  type: feedback\n---\n\nThe memory algebra executor lives in internal/memq/exec.go.\n",
		"stale.md": "---\nname: moved-fix\ndescription: an old location\nmetadata:\n  type: project\n---\n\nThe fix lives in internal/gonepkg/gone.go.\n",
		"prose.md": "---\nname: preference\ndescription: terse answers\nmetadata:\n  type: user\n---\n\nThe user prefers the outcome stated first.\n",
	}
	for name, body := range files {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}

// #2429 done-condition: RenderNotesDigest — the function `fak memory-read` now
// calls — withholds the stale note (never renders its body) while a fresh
// sibling renders whole, and the withheld footer names the failing claim as
// evidence rather than silently dropping the note.
func TestRenderNotesDigest_withholdsStaleRendersFresh(t *testing.T) {
	dir := buildNotesDigestStore(t)
	out := RenderNotesDigest(dir, false, 60000)
	if !strings.Contains(out, "## Gate helper (fresh.md)") || !strings.Contains(out, "internal/memq/exec.go") {
		t.Fatalf("fresh note must render whole:\n%s", out)
	}
	if strings.Contains(out, "## Moved fix (stale.md)") {
		t.Fatalf("stale note body must never render:\n%s", out)
	}
	if !strings.Contains(out, "withheld (never injected as fact):") || !strings.Contains(out, "stale.md") || !strings.Contains(out, "internal/gonepkg/gone.go") {
		t.Fatalf("stale note must be named in a withheld footer with the failing claim:\n%s", out)
	}
}

func TestRenderNotesDigest_indexOnly(t *testing.T) {
	dir := buildNotesDigestStore(t)
	out := RenderNotesDigest(dir, true, 60000)
	if !strings.Contains(out, "Gate helper") {
		t.Fatalf("index-only digest must still carry the MEMORY.md index text:\n%s", out)
	}
	if strings.Contains(out, "internal/memq/exec.go") {
		t.Fatalf("index-only digest must not expand fact bodies:\n%s", out)
	}
}

// The byte budget and named-overflow behavior (#2430) must survive the switch
// from raw-file RenderDigest to the gated Materialize path unchanged.
func TestRenderNotesDigest_maxBytesOverflowNamed(t *testing.T) {
	dir := buildNotesDigestStore(t)
	out := RenderNotesDigest(dir, false, 1)
	// fresh.md is index-first so the emitted>0 guard always lets it through; the
	// remaining in-budget notes overflow a 1-byte budget and must be NAMED, never
	// an anonymous count.
	if !strings.Contains(out, "internal/memq/exec.go") {
		t.Fatalf("first note must still be emitted under the emitted>0 guard:\n%s", out)
	}
	if !strings.Contains(out, memoryread.OverflowReason) {
		t.Fatalf("over-budget notes must be named under %s:\n%s", memoryread.OverflowReason, out)
	}
}

func TestRenderNotesDigest_directFilePath(t *testing.T) {
	dir := buildNotesDigestStore(t)
	memFile := filepath.Join(dir, "MEMORY.md")
	out := RenderNotesDigest(memFile, false, 60000)
	if !strings.Contains(out, "## Gate helper (fresh.md)") || !strings.Contains(out, "internal/memq/exec.go") {
		t.Fatalf("RenderNotesDigest with direct file path failed:\n%s", out)
	}
}
