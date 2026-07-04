package issuefanout

import (
	"fmt"
	"sort"
	"strings"
)

// AdoptionSchema identifies the machine-readable adoption report.
const AdoptionSchema = "fak.issue-fanout-adoption.v1"

// LeafAdoption is one shipped leaf's fan-out standing: how many follow-ons the
// fan-out default filed for it, and whether that clears the MinFanout floor.
type LeafAdoption struct {
	Leaf        string `json:"leaf"`
	FanoutFiled int    `json:"fanout_filed"` // distinct fanout-<leaf>-* marker keys seen
	ClearsFloor bool   `json:"clears_floor"` // FanoutFiled >= MinFanout
	Gap         int    `json:"gap"`          // MinFanout - FanoutFiled, floored at 0
}

// AdoptionReport measures the spine-fanout default across a set of shipped
// leaves: which ones had their follow-on backlog filed (cleared the floor) and
// which are gaps — a spine shipped without the 3..50+ follow-ons the default
// demands. OrphanMarkers is the dual gap: a fan-out filed against a leaf that
// is not in the shipped set (a marker with no spine).
//
// It is pure, like the rest of this leaf: the caller gathers the shipped leaves
// (git) and the filed marker keys (gh); Adoption only decides the standing, so
// the meter is scored from witnesses, never from a self-report that the default
// was followed. OK is the one-bit gate a pipeline can drive (no gap leaves).
type AdoptionReport struct {
	Schema        string         `json:"schema"`
	MinFanout     int            `json:"min_fanout"`
	ShippedLeaves int            `json:"shipped_leaves"`
	ClearedLeaves int            `json:"cleared_leaves"`
	GapLeaves     int            `json:"gap_leaves"`
	OK            bool           `json:"ok"`                       // GapLeaves == 0
	Leaves        []LeafAdoption `json:"leaves"`                   // one row per shipped leaf, leaf-sorted
	Gaps          []string       `json:"gaps"`                     // shipped leaves below the floor (leaf-sorted)
	OrphanMarkers []string       `json:"orphan_markers,omitempty"` // fan-out markers matching no shipped leaf
}

// Adoption computes the adoption report for the shipped leaves against the filed
// fan-out marker keys. A marker key has the form fanout-<leaf>-<slug> (see
// expand); each distinct key is credited to the LONGEST shipped leaf whose exact
// "fanout-<leaf>-" prefix it carries, so a leaf whose name prefixes another's
// never steals its count. A fanout-* key matching no shipped leaf is an orphan;
// a non-fanout key is ignored. A shipped leaf with fewer than MinFanout keys is
// a gap. Output is deterministic (leaf-sorted, duplicate leaves/keys collapsed).
func Adoption(shippedLeaves, markerKeys []string) AdoptionReport {
	seenLeaf := map[string]bool{}
	var leaves []string
	for _, l := range shippedLeaves {
		l = strings.TrimSpace(l)
		if l == "" || seenLeaf[l] {
			continue
		}
		seenLeaf[l] = true
		leaves = append(leaves, l)
	}
	sort.Strings(leaves)

	counts := map[string]int{}
	var orphans []string
	seenKey := map[string]bool{}
	for _, k := range markerKeys {
		k = strings.TrimSpace(k)
		if k == "" || seenKey[k] {
			continue
		}
		seenKey[k] = true
		best := ""
		for _, l := range leaves {
			if len(l) > len(best) && strings.HasPrefix(k, "fanout-"+l+"-") {
				best = l
			}
		}
		switch {
		case best != "":
			counts[best]++
		case strings.HasPrefix(k, "fanout-"):
			orphans = append(orphans, k)
		}
	}
	sort.Strings(orphans)

	rep := AdoptionReport{
		Schema:        AdoptionSchema,
		MinFanout:     MinFanout,
		ShippedLeaves: len(leaves),
		OrphanMarkers: orphans,
	}
	for _, l := range leaves {
		n := counts[l]
		clears := n >= MinFanout
		gap := 0
		if !clears {
			gap = MinFanout - n
		}
		rep.Leaves = append(rep.Leaves, LeafAdoption{Leaf: l, FanoutFiled: n, ClearsFloor: clears, Gap: gap})
		if clears {
			rep.ClearedLeaves++
		} else {
			rep.Gaps = append(rep.Gaps, l)
		}
	}
	rep.GapLeaves = len(rep.Gaps)
	rep.OK = rep.GapLeaves == 0
	return rep
}

// RenderAdoption prints the adoption report for a human: the headline ratio, one
// line per shipped leaf, the gap list that names the unfiled fan-outs, and any
// orphan markers.
func RenderAdoption(r AdoptionReport) string {
	var b strings.Builder
	fmt.Fprintf(&b, "spine-fanout adoption: %d/%d shipped leaves cleared the fan-out floor (>=%d), %d gap(s)\n",
		r.ClearedLeaves, r.ShippedLeaves, r.MinFanout, r.GapLeaves)
	for _, l := range r.Leaves {
		mark := "GAP"
		if l.ClearsFloor {
			mark = "ok "
		}
		fmt.Fprintf(&b, "  [%s] %-24s %d/%d follow-on(s) filed\n", mark, l.Leaf, l.FanoutFiled, r.MinFanout)
	}
	if len(r.Gaps) > 0 {
		fmt.Fprintf(&b, "gaps (spine shipped, fan-out not filed): %s\n", strings.Join(r.Gaps, ", "))
	}
	if len(r.OrphanMarkers) > 0 {
		fmt.Fprintf(&b, "orphan markers (fan-out filed, no shipped leaf): %s\n", strings.Join(r.OrphanMarkers, ", "))
	}
	return b.String()
}
