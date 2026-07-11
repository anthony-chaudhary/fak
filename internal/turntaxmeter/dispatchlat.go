package turntaxmeter

import "sort"

type DispatchLatency struct {
	Phase   string `json:"phase"`
	Samples int    `json:"samples"`
	P50MS   int64  `json:"p50_ms"`
	P90MS   int64  `json:"p90_ms"`
	P99MS   int64  `json:"p99_ms"`
}

// FoldDispatchLatency converts dispatch tick timing maps into deterministic
// per-phase percentiles. Unknown/missing phases simply do not emit a row.
func FoldDispatchLatency(rows []map[string]int64) []DispatchLatency {
	by := map[string][]int64{}
	for _, row := range rows {
		for phase, ms := range row {
			if ms >= 0 {
				by[phase] = append(by[phase], ms)
			}
		}
	}
	phases := make([]string, 0, len(by))
	for p := range by {
		phases = append(phases, p)
	}
	sort.Strings(phases)
	out := make([]DispatchLatency, 0, len(phases))
	for _, p := range phases {
		v := by[p]
		sort.Slice(v, func(i, j int) bool { return v[i] < v[j] })
		out = append(out, DispatchLatency{p, len(v), pct(v, .50), pct(v, .90), pct(v, .99)})
	}
	return out
}
func pct(v []int64, q float64) int64 {
	if len(v) == 0 {
		return 0
	}
	i := int(q * float64(len(v)-1))
	if i >= len(v) {
		i = len(v) - 1
	}
	return v[i]
}
