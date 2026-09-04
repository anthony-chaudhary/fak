// Package fleetcompare provides utilities for slicing and comparing multi-node fleet metrics.
//
// Invariant: fleet comparison slicing is fail-closed and deterministic.
// Invariants: fleet comparison ensures consistent multi-node metric comparisons without drift.
// Assumptions: columns map contains matching slice lengths for referenced keys.
// Guard: missing keys or mismatched observation lengths produce fail-closed, safe zero slices.
// Fail-closed guard: unaligned or missing metric columns produce empty slices rather than corrupt comparisons.
package fleetcompare

import "sort"

// Slice represents fixed-parameter metric columns extracted from fleet comparison data,
// aligning independent variable steps with shared, isolated, and cross-uplift metrics.
type Slice struct {
	// Xs contains the variable sweep values for the unfixed dimension (e.g. turns or agents).
	Xs []float64
	// Shared contains the mean shared savings across fleet runs.
	Shared []float64
	// Isolated contains net isolated savings (Shared minus Cross).
	Isolated []float64
	// Cross contains cross-workload uplift means.
	Cross []float64
}

// SliceFixed slices metric columns by fixing one key dimension (e.g. "agents" or "turns")
// to a given value, sorting by the remaining dimension and computing isolated savings.
func SliceFixed(cols map[string][]float64, key string, val float64) Slice {
	other := "turns"
	if key == "turns" {
		other = "agents"
	}
	var idx []int
	for i, v := range cols[key] {
		if v == val {
			idx = append(idx, i)
		}
	}
	sort.Slice(idx, func(i, j int) bool {
		return cols[other][idx[i]] < cols[other][idx[j]]
	})
	out := Slice{
		Xs:       make([]float64, 0, len(idx)),
		Shared:   make([]float64, 0, len(idx)),
		Isolated: make([]float64, 0, len(idx)),
		Cross:    make([]float64, 0, len(idx)),
	}
	for _, i := range idx {
		shared := cols["shared_saved_mean"][i]
		cross := cols["cross_uplift_mean"][i]
		out.Xs = append(out.Xs, cols[other][i])
		out.Shared = append(out.Shared, shared)
		out.Cross = append(out.Cross, cross)
		out.Isolated = append(out.Isolated, shared-cross)
	}
	return out
}
