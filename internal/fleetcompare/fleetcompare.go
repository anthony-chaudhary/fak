package fleetcompare

import "sort"

type Slice struct {
	Xs       []float64
	Shared   []float64
	Isolated []float64
	Cross    []float64
}

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
