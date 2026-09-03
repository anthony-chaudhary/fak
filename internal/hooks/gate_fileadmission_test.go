package hooks

import (
	"os/exec"
	"strings"
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
		{"llms-full.txt", "fak:operator-private quoted from generated authorities", false, "generated authority corpus may quote marker"},
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
		{"gpu-status.md", true},      // repo-root *.md
		{"docs/gpu-status.md", true}, // depth-1 docs/
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

// TestFileAdmission_OversizedBlobCap pins the size cap's decision boundary: a blob at or under
// the ceiling is admitted, one byte over is refused as an oversized blob. The ceiling is
// overridden to a tiny value here (save/restore) so the test never has to materialize a 25 MiB
// fixture — the wiring under test is the > fileAdmissionMaxBytes comparison, not the constant.
func TestFileAdmission_OversizedBlobCap(t *testing.T) {
	orig := fileAdmissionMaxBytes
	t.Cleanup(func() { fileAdmissionMaxBytes = orig })
	fileAdmissionMaxBytes = 8 // bytes

	// mapReader.Size returns len(body), so body length is the blob size.
	atCap := mapReader{"assets/blob.bin": "12345678"}    // exactly 8 bytes -> admitted
	overCap := mapReader{"assets/blob.bin": "123456789"} // 9 bytes -> refused

	if why := classifyFileWith(atCap, "assets/blob.bin"); why != "" {
		t.Errorf("blob at the cap must be admitted, got refusal: %q", why)
	}
	why := classifyFileWith(overCap, "assets/blob.bin")
	if why == "" {
		t.Fatalf("blob over the cap must be refused, got admission")
	}
	if !strings.Contains(why, "oversized blob") {
		t.Errorf("oversized blob refusal wording drifted: %q", why)
	}
}

// TestFileAdmission_ByMachineHelper pins the staged-only by-machine helper's decisions:
// a raw per-machine run drop (root-hoisted, fak/-nested, and the historically-leaky dgx*
// class) is refused; a normal path or the tracked aggregate catalog is not.
func TestFileAdmission_ByMachineHelper(t *testing.T) {
	// The dgxN-form host segment is assembled at runtime so this source carries no literal
	// private-GPU-host-alias needle — the "no literal needle in tests" convention
	// gate_publicleak.go documents (its own AUDIT_NEEDLES list is split the same way). The
	// by-machine rule keys off the path PREFIX, not the host, so this changes nothing asserted.
	dgxN := "dgx" + "2"
	refused := []string{
		"experiments/benchmark/runs/by-machine/node-macos-a/20260718-x/score.json",
		"fak/experiments/benchmark/runs/by-machine/node-macos-a/20260718-x/score.json",
		"experiments/benchmark/runs/by-machine/dgx-a100-01/20260718-run/witness.json",
		"fak/experiments/benchmark/runs/by-machine/" + dgxN + "/20260718-run/manifest.json",
	}
	for _, p := range refused {
		if !isPrivateByMachineAddition(p) {
			t.Errorf("by-machine addition should be refused: %q", p)
		}
	}
	allowed := []string{
		"internal/foo/bar.go",
		"experiments/benchmark/catalog.json",    // tracked aggregate, OUTSIDE by-machine/
		"experiments/benchmark/runs/summary.md", // sibling under runs/, not by-machine/
	}
	for _, p := range allowed {
		if isPrivateByMachineAddition(p) {
			t.Errorf("normal path must not be a by-machine addition: %q", p)
		}
	}
}

// TestFileAdmission_ByMachineStagedOnly proves the by-machine rule is STAGED-ONLY: the staged
// gate refuses a NEW drop, the shared classifier stays silent (so it is NOT in _classify), and
// the tree twin does NOT refuse a by-machine path — the invariant that keeps the ~50
// grandfathered evidence files green in --audit-tree.
func TestFileAdmission_ByMachineStagedOnly(t *testing.T) {
	p := "experiments/benchmark/runs/by-machine/dgx-a100-01/20260718-run/score.json"

	// STAGED: gateFileAdmission (over AddedRenamedPaths) refuses it.
	d := &StagedDiff{Root: t.TempDir(), AddedRenamedPaths: []string{p}, fileCache: map[string]fileEntry{}}
	findings, err := gateFileAdmission(d)
	if err != nil {
		t.Fatalf("gateFileAdmission: %v", err)
	}
	if !hasFindingFor(findings, "FILE_ADMISSION", "PRIVATE-BY-DEFAULT") {
		t.Fatalf("staged gate must refuse a new by-machine drop; got %+v", findings)
	}

	// SHARED classifier (used by the tree twin) must be SILENT — the rule is staged-only.
	if why := classifyFileWith(d, p); why != "" {
		t.Errorf("classifyFileWith must not fire the by-machine rule (staged-only); got %q", why)
	}

	// TREE twin: a by-machine path in the tracked set is NOT refused (grandfathered evidence).
	tree := &TrackedTree{Root: t.TempDir(), Paths: []string{p}, fileCache: map[string]fileEntry{}}
	tf, err := gateFileAdmissionTree(tree)
	if err != nil {
		t.Fatalf("gateFileAdmissionTree: %v", err)
	}
	if len(tf) != 0 {
		t.Errorf("tree gate must NOT refuse a grandfathered by-machine path; got %+v", tf)
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

func TestFileAdmissionRejectsGeneratedClaudeControlArtifacts(t *testing.T) {
	tests := []string{
		".claude/goal-prompts/frontdoor-6037-recovery.md",
		".claude/goal-prompts/resfleet-6557.md",
		".claude/goal-prompts/resolve-issue-5898-continuation.md",
	}
	for _, path := range tests {
		t.Run(path, func(t *testing.T) {
			d := &StagedDiff{Root: t.TempDir(), AddedRenamedPaths: []string{path}, fileCache: map[string]fileEntry{path: {data: []byte("worker fuel"), exists: true}}}
			got, err := gateFileAdmission(d)
			if err != nil {
				t.Fatal(err)
			}
			if len(got) != 1 || got[0].Gate != "FILE_ADMISSION" {
				t.Fatalf("generated control artifact findings=%#v", got)
			}
		})
	}
}

func TestFileAdmissionAllowsReusableClaudeGoalPrompt(t *testing.T) {
	path := ".claude/goal-prompts/resolve-top-issue-witnessed.md"
	d := &StagedDiff{Root: t.TempDir(), AddedRenamedPaths: []string{path}, fileCache: map[string]fileEntry{path: {data: []byte("reusable project prompt"), exists: true}}}
	got, err := gateFileAdmission(d)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 0 {
		t.Fatalf("reusable prompt must remain admissible: %#v", got)
	}
}

func TestFileAdmission_ScriptAdmission(t *testing.T) {
	cases := []struct {
		rel     string
		wantBad bool
	}{
		{"scripts/dogfood-opencode.ps1", true},
		{"tools/random.ps1", true},
		{"scripts/random.sh", true},
		{"foo.bat", true},
		{"tools/random.cmd", true},
		{"test.ps1", false},
		{"test.sh", false},
		{"scripts/build.sh", false},
		{"tools/build_cuda_windows.ps1", false},
		{"internal/hooks/gate_fileadmission.go", false},
	}
	for _, c := range cases {
		fr := mapReader{c.rel: "content"}
		why := classifyFileWith(fr, c.rel)
		if gotBad := why != ""; gotBad != c.wantBad {
			t.Errorf("classifyFileWith(%q) bad=%v want=%v (why=%q)", c.rel, gotBad, c.wantBad, why)
		}
	}
}
