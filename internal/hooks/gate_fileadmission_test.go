package hooks

import (
	"os/exec"
	"testing"
)

// gate_fileadmission_test.go — unit cases for the FILE_ADMISSION operator-private extension
// (opsArtifactReason / opsLooseDoc in filesource.go, token data in gate_fileadmission.go).
// The classifier is path+content only, so we exercise it through a tiny in-memory fileReader
// rather than a temp git tree. Verdict-level parity with the Python oracle is covered
// separately by parity_test.go; here we pin the per-file decisions and the live tree's
// cleanliness.

// mapReader is an in-memory fileReader: FileBytes/Size answer from a path->content map,
// Exists from key presence. *StagedDiff and *TrackedTree implement the same surface in prod.
type mapReader map[string]string

func (m mapReader) FileBytes(rel string) ([]byte, bool) { b, ok := m[rel]; return []byte(b), ok }
func (m mapReader) Exists(rel string) bool              { _, ok := m[rel]; return ok }
func (m mapReader) Size(rel string) (int64, bool) {
	b, ok := m[rel]
	if !ok {
		return 0, false
	}
	return int64(len(b)), true
}

func TestFileAdmission_OpsArtifact(t *testing.T) {
	cases := []struct {
		rel     string
		body    string
		wantBad bool
		label   string
	}{
		// (2) BACKSTOP — loose ops doc (infra-noun AND state-noun) is refused.
		{"docs/gpu-reserve-status.md", "# x", true, "loose ops doc under docs/"},
		{"node-availability-status.md", "# x", true, "loose ops doc at repo root"},
		{"docs/fleet-roster.md", "# x", true, "fleet+roster"},
		// BACKSTOP must NOT fire on legitimate docs.
		{"docs/PRODUCT-STATUS.md", "# x", false, "state word, no infra word"},
		{"docs/dispatch-loop.md", "# x", false, "infra word, no state word"},
		{"docs/fleet.md", "# fleet", false, "infra word alone"},
		{"docs/notes/foo-status-9.md", "# x", false, "curated docs/notes/ location"},
		{"experiments/x/STATUS.md", "# x", false, "experiments/ exempt by location"},
		{"docs/fak/node-status.md", "# x", false, "depth-2 curated location"},
		{"docs/whatever.md", "plain", false, "no marker, no ops tokens"},
		// (1) MARKER — any text-like file declaring the token is refused, path-agnostic.
		{"docs/whatever2.md", "---\nfak:operator-private\n---\n", true, "marker in md front-matter"},
		{"notes.json", `{"_audience":"fak:operator-private"}`, true, "marker in json field"},
		{"scratch.txt", "fak:operator-private here", true, "marker in txt"},
		{"a.yaml", "fak:operator-private", true, "marker in yaml"},
		{"b.txt", "FAK:OPERATOR-PRIVATE", true, "marker is case-insensitive"},
		// A binary (non-text-ext) with the token in its bytes is NOT marker-scanned.
		{"assets/data.png", "fak:operator-private", false, "binary not scanned for marker"},
	}
	for _, c := range cases {
		fr := mapReader{c.rel: c.body}
		why := classifyFileWith(fr, c.rel)
		if gotBad := why != ""; gotBad != c.wantBad {
			t.Errorf("%s: classifyFileWith(%q) bad=%v want=%v (why=%q)", c.label, c.rel, gotBad, c.wantBad, why)
		}
	}
}

// TestFileAdmission_OpsLooseDoc pins the location predicate independently of content.
func TestFileAdmission_OpsLooseDoc(t *testing.T) {
	cases := []struct {
		rel  string
		want bool
	}{
		{"gpu-status.md", true},          // repo-root *.md
		{"docs/gpu-status.md", true},     // depth-1 docs/
		{"docs/notes/gpu-status.md", false},
		{"docs/fak/gpu-status.md", false},
		{"experiments/gpu-status.md", false}, // depth-1 but not under docs/
		{"gpu-status.txt", false},            // not .md
	}
	for _, c := range cases {
		if got := opsLooseDoc(c.rel); got != c.want {
			t.Errorf("opsLooseDoc(%q) = %v, want %v", c.rel, got, c.want)
		}
	}
}

// TestFileAdmission_LiveTreeNoOpsArtifact asserts the real tracked tree carries no
// operator-private operational artifact — the false-positive guard on real data and the
// regression guard for the dispatch-status.md leak (untracked in the same change). Skipped
// outside a git checkout.
func TestFileAdmission_LiveTreeNoOpsArtifact(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	tree, err := ReadTrackedTree(repoRoot(t))
	if err != nil {
		t.Skipf("ReadTrackedTree: %v", err)
	}
	for _, p := range tree.Paths {
		if why := opsArtifactReason(tree, p); why != "" {
			t.Errorf("ops-artifact on tracked tree: %s  ->  %s", p, why)
		}
	}
}
