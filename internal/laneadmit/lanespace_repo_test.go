package laneadmit

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// lanespace_repo_test.go — the multiplier witness against the REAL repo, not a fixture.
//
// lanetree.go's claim is that making a lane name path-shaped turns the addressable lane space from
// something a human enumerates in dos.toml into something DERIVED from the tree. That claim is
// only worth anything as a measured number, so this test runs LaneSpace over the actual tracked
// file list at each granularity and reports the ratio against the declared vocabulary. It asserts
// conservative floors (the shape of the result, which cannot regress) and logs the exact counts
// (the number, which moves as the repo grows) — a ratchet, not a brittle golden.
//
// It skips rather than fails when dos.toml or git is unavailable, so it stays honest in a clean
// room or an export where the tracked list cannot be recovered.

func repoRoot(t *testing.T) string {
	t.Helper()
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Skipf("cannot resolve repo root: %v", err)
	}
	if _, err := os.Stat(filepath.Join(root, "dos.toml")); err != nil {
		t.Skipf("dos.toml not readable under %s: %v", root, err)
	}
	return root
}

func trackedPaths(t *testing.T, root string) []string {
	t.Helper()
	cmd := exec.Command("git", "ls-files")
	cmd.Dir = root
	out, err := cmd.Output()
	if err != nil {
		t.Skipf("git ls-files unavailable (clean room / export?): %v", err)
	}
	var paths []string
	for _, line := range strings.Split(strings.ReplaceAll(string(out), "\r\n", "\n"), "\n") {
		if line = strings.TrimSpace(line); line != "" {
			paths = append(paths, line)
		}
	}
	if len(paths) == 0 {
		t.Skip("git ls-files returned nothing")
	}
	return paths
}

// TestRepoLaneSpaceMultiplier measures how much the hierarchical namespace widens the addressable
// lane space over this repo's own tracked tree.
func TestRepoLaneSpaceMultiplier(t *testing.T) {
	root := repoRoot(t)
	data, err := os.ReadFile(filepath.Join(root, "dos.toml"))
	if err != nil {
		t.Skipf("dos.toml unreadable: %v", err)
	}
	tax := ParseTaxonomy(data)
	if !tax.Loaded || len(tax.Trees) == 0 {
		t.Fatalf("dos.toml parsed to an empty taxonomy (Loaded=%v, trees=%d)", tax.Loaded, len(tax.Trees))
	}
	paths := trackedPaths(t, root)

	declared := len(tax.Trees)
	leaf := LaneSpace(paths, tax, GranLeaf)
	dir := LaneSpace(paths, tax, GranDir)
	file := LaneSpace(paths, tax, GranFile)

	t.Logf("tracked paths            : %d", len(paths))
	t.Logf("declared dos.toml lanes  : %d", declared)
	t.Logf("GranLeaf  addressable    : %d  (x%.2f vs declared)", len(leaf), float64(len(leaf))/float64(declared))
	t.Logf("GranDir   addressable    : %d  (x%.2f vs GranLeaf)", len(dir), float64(len(dir))/float64(len(leaf)))
	t.Logf("GranFile  addressable    : %d  (x%.2f vs GranLeaf)", len(file), float64(len(file))/float64(len(leaf)))

	// Shape floors — these are the properties the design guarantees, so they may not regress.
	if len(leaf) == 0 {
		t.Fatal("GranLeaf addressed no lanes at all; the tree index is not matching real paths")
	}
	if len(dir) <= len(leaf) {
		t.Errorf("GranDir (%d) must strictly widen GranLeaf (%d)", len(dir), len(leaf))
	}
	if len(file) <= len(dir) {
		t.Errorf("GranFile (%d) must strictly widen GranDir (%d)", len(file), len(dir))
	}
	// A file lane space smaller than 5x the declared vocabulary would mean the derivation is not
	// actually reaching the tree — a floor loose enough never to fight normal repo churn.
	if got := float64(len(file)) / float64(declared); got < 5 {
		t.Errorf("GranFile lane space is only x%.2f the declared vocabulary; the derivation is not reaching the tree", got)
	}
}

// TestRepoLaneSpaceIsWellFormed checks every derived lane over the real tree round-trips: it
// resolves to a declared ancestor, inherits a tree, and encodes to a lease id that
// internal/leaseref's validID would accept. This is the property that stops a derived lane from
// being unleasable in practice — an unleasable lane is not a lane.
func TestRepoLaneSpaceIsWellFormed(t *testing.T) {
	root := repoRoot(t)
	data, err := os.ReadFile(filepath.Join(root, "dos.toml"))
	if err != nil {
		t.Skipf("dos.toml unreadable: %v", err)
	}
	tax := ParseTaxonomy(data)
	paths := trackedPaths(t, root)

	space := LaneSpace(paths, tax, GranFile)
	badTree, badID := 0, 0
	var firstBadTree, firstBadID string
	for _, lane := range space {
		if len(tax.TreeFor(lane)) == 0 {
			if badTree++; firstBadTree == "" {
				firstBadTree = lane
			}
		}
		id := LeaseID(SurfaceDispatch, lane, "")
		if !refSafeID(id) || LaneOfLeaseID(id) != lane {
			if badID++; firstBadID == "" {
				firstBadID = lane + " -> " + id + " -> " + LaneOfLeaseID(id)
			}
		}
	}
	if badTree > 0 {
		t.Errorf("%d/%d derived lanes resolved to NO tree (first: %q)", badTree, len(space), firstBadTree)
	}
	if badID > 0 {
		t.Errorf("%d/%d derived lanes do not round-trip through a ref-safe lease id (first: %s)", badID, len(space), firstBadID)
	}
}

// refSafeID mirrors internal/leaseref validID: one ref-path segment, no separator, no ref-special
// byte, no leading dash or dot, and no `.lock` suffix (which git's check-ref-format rejects).
// It is duplicated rather than imported to keep laneadmit free of a dependency on the lease store.
func refSafeID(id string) bool {
	if id == "" || len(id) > 200 || strings.HasSuffix(id, ".lock") {
		return false
	}
	if strings.HasPrefix(id, "-") || strings.HasPrefix(id, ".") {
		return false
	}
	for _, c := range []byte(id) {
		switch {
		case c >= 'a' && c <= 'z', c >= 'A' && c <= 'Z', c >= '0' && c <= '9':
		case c == '-' || c == '_' || c == '.':
		default:
			return false
		}
	}
	return true
}
