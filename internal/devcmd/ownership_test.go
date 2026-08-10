package devcmd

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/devindex"
)

func TestRunOwnershipJSON(t *testing.T) {
	var out, errOut bytes.Buffer
	if code := RunOwnership(&out, &errOut, devindex.FindRoot("."), true); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	var got devindex.OwnershipReport
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Schema != "fak-command-ownership/1" || len(got.Commands) == 0 || got.Graph.PackageCount == 0 {
		t.Fatalf("incomplete report: %+v", got)
	}
}

// TestIndexOwnershipCountsTheRuntimeClosure pins the witness to the pattern whose
// closure is the shipped binary. `fak-dev index ownership` once built its own report
// with "./..." — the whole module — and reported ~950 packages / ~586 internal where
// the runtime artifact links ~696 / ~449, so every count published from the CLI was
// wrong (#6022). Comparing the CLI's package_count against an independent
// LoadImportGraph(root, "./cmd/fak") fails loudly if the pattern ever widens again:
// a "./..." report cannot match a "./cmd/fak" graph.
func TestIndexOwnershipCountsTheRuntimeClosure(t *testing.T) {
	root := devindex.FindRoot(".")
	nodes, err := devindex.LoadImportGraph(root, "./cmd/fak")
	if err != nil {
		t.Fatal(err)
	}
	if len(nodes) == 0 {
		t.Fatal("go list -deps ./cmd/fak returned no packages; the comparison would be vacuous")
	}

	var out, errOut bytes.Buffer
	if code := RunIndex(&out, &errOut, []string{"ownership", "--json", "--root", root}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	var got devindex.OwnershipReport
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Graph.Root != "github.com/anthony-chaudhary/fak/cmd/fak" {
		t.Errorf("graph root = %q, want the runtime import root", got.Graph.Root)
	}
	if got.Graph.PackageCount != len(nodes) {
		t.Errorf("index ownership package_count = %d, want the runtime closure %d (did the go-list pattern widen back to \"./...\"?)",
			got.Graph.PackageCount, len(nodes))
	}
	// Leak semantics are pattern-independent (the BFS is rooted at cmd/fak either
	// way); the ratchet from #6022 must still read zero.
	if len(got.Graph.Leaks) != 0 {
		t.Errorf("runtime graph has %d dev-only leak(s): %+v", len(got.Graph.Leaks), got.Graph.Leaks)
	}
}

func TestRunOwnershipText(t *testing.T) {
	var out, errOut bytes.Buffer
	if code := RunOwnership(&out, &errOut, devindex.FindRoot("."), false); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	for _, want := range []string{"command ownership: runtime=", "runtime graph: packages=", "dev-leaks=0"} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("output missing %q:\n%s", want, out.String())
		}
	}
}
