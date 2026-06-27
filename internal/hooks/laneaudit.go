package hooks

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// laneaudit.go — the whole-tree complement of the per-commit leaf check in commitstamp.go. The
// commit lint catches a bad stamp on ONE commit; this answers the standing question "which real
// leaves have no declared lane at all?". dos.toml's own doctrine is ONE LANE PER LEAF (PARTITION.md:
// "the honest partition is ONE LANE PER LEAF") — every internal/<leaf> package should appear in the
// `[lanes]` taxonomy so its `(fak <leaf>)` ship-stamp binds to a real unit and the arbiter can
// detect a same-tree collision on its edits. A leaf that drifts in WITHOUT a lane silently breaks
// both. This turns that drift from "something an operator has to remember to reconcile" into a
// deterministic, re-runnable count that a gate can ratchet to zero.

// LeafGap names a real Go-package leaf with no declared dos.toml lane.
type LeafGap struct {
	Leaf string `json:"leaf"`
	Base string `json:"base"` // "internal"
}

// UndeclaredLeaves returns every internal/<leaf> that holds a real Go package but has no declared
// dos.toml lane, sorted by name. cmd/<dir> is intentionally NOT audited: the `cmd` lane owns
// `cmd/**` as a single tree (#518), so a cmd demo legitimately has no lane of its own. Returns an
// error when dos.toml (the lane source of truth) cannot be read — the caller decides whether to
// treat that as could-not-run rather than a clean zero.
func UndeclaredLeaves(root string) ([]LeafGap, error) {
	tax := readLaneTaxonomy(root)
	if !tax.loaded {
		return nil, fmt.Errorf("dos.toml not readable under %q (lane taxonomy unavailable)", root)
	}
	dir := filepath.Join(root, "internal")
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	var gaps []LeafGap
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		name := e.Name()
		if tax.declared[strings.ToLower(name)] {
			continue
		}
		if !dirHasGoFiles(filepath.Join(dir, name)) {
			continue // not a Go package (e.g. a testdata-only or doc dir): not a leaf
		}
		gaps = append(gaps, LeafGap{Leaf: name, Base: "internal"})
	}
	sort.Slice(gaps, func(i, j int) bool { return gaps[i].Leaf < gaps[j].Leaf })
	return gaps, nil
}

// dirHasGoFiles reports whether dir directly contains at least one .go file.
func dirHasGoFiles(dir string) bool {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return false
	}
	for _, e := range entries {
		if !e.IsDir() && strings.HasSuffix(e.Name(), ".go") {
			return true
		}
	}
	return false
}
