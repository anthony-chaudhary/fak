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
// "the honest partition is ONE LANE PER LEAF") — every internal/<leaf> package must resolve through
// the `[lanes]` taxonomy so its ship-stamp binds to a real unit and the arbiter can detect a
// same-tree collision on its edits. Most use the package name; an established composite lane may
// own several explicit trees. A leaf that drifts in WITHOUT either form silently breaks both. This
// turns that drift from "something an operator has to remember to reconcile" into a deterministic,
// re-runnable count that a gate can ratchet to zero.

// LeafGap names a real Go-package leaf with no declared dos.toml lane.
type LeafGap struct {
	Leaf string `json:"leaf"`
	Base string `json:"base"` // "internal" or "pkg"
}

// UndeclaredLeaves returns every internal/<leaf> and pkg/<leaf> that holds a real Go package
// but has no declared dos.toml lane, sorted by name. cmd/<dir> is intentionally NOT audited:
// the `cmd` lane owns `cmd/**` as a single tree (#518), so a cmd demo legitimately has no lane
// of its own. Returns an error when dos.toml (the lane source of truth) cannot be read — the
// caller decides whether to treat that as could-not-run rather than a clean zero.
func UndeclaredLeaves(root string) ([]LeafGap, error) {
	tax := readLaneTaxonomy(root)
	if !tax.loaded {
		return nil, fmt.Errorf("dos.toml not readable under %q (lane taxonomy unavailable)", root)
	}
	var gaps []LeafGap
	for _, base := range []string{"internal", "pkg"} {
		dir := filepath.Join(root, base)
		entries, err := os.ReadDir(dir)
		if err != nil {
			if os.IsNotExist(err) && base != "internal" {
				continue
			}
			return nil, err
		}
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
			// Some established composite lanes intentionally differ from the package
			// basename. An explicit tree owner is authoritative even when the lane is
			// named studyreceipt and the package is internal/study.
			probe := filepath.ToSlash(filepath.Join(base, name, "_lane_ownership.go"))
			if explicitTreeLaneForPath(probe, tax) != "" {
				continue
			}
			gaps = append(gaps, LeafGap{Leaf: name, Base: base})
		}
	}
	sort.Slice(gaps, func(i, j int) bool {
		if gaps[i].Leaf != gaps[j].Leaf {
			return gaps[i].Leaf < gaps[j].Leaf
		}
		return gaps[i].Base < gaps[j].Base
	})
	return gaps, nil
}

// explicitTreeLaneForPath is the authored-tree-only half of laneForPath. It has
// no directory-convention fallback, because a fallback cannot prove that the
// arbiter has an actual region to compare for this package.
func explicitTreeLaneForPath(path string, tax laneTaxonomy) string {
	p := strings.ToLower(filepath.ToSlash(filepath.Clean(path)))
	if lane, ok := tax.exact[p]; ok {
		return lane
	}
	best, owner := "", ""
	for prefix, lane := range tax.prefixes {
		if strings.HasPrefix(p, prefix) && len(prefix) > len(best) {
			best, owner = prefix, lane
		}
	}
	return owner
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
