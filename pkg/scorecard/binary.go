package scorecard

import "math"

// BinaryResult is the shared input shape for scorecards whose criteria are
// pass/fail rows with a HARD/SOFT classification and an in-axis weight.
type BinaryResult struct {
	Key    string `json:"key"`
	Label  string `json:"label"`
	Hard   bool   `json:"hard"`
	Weight int    `json:"weight"`
	Axis   string `json:"axis"`
	Passed bool   `json:"passed"`
	Detail string `json:"detail"`
}

// BinaryPayload is the stable JSON projection of one BinaryResult.
type BinaryPayload struct {
	KPI     string   `json:"kpi"`
	Group   string   `json:"group"`
	Score   int      `json:"score"`
	Value   float64  `json:"value"`
	Detail  string   `json:"detail"`
	Defects []string `json:"defects"`
	Soft    []string `json:"soft"`
}

// ProjectBinary converts ordered binary results into Fold inputs. axisShares
// names each axis's share of the composite; weights within an axis are
// normalized by that axis's total. extraDefects lets a card retain deliberate
// magnitude debt without duplicating the common HARD/SOFT projection.
func ProjectBinary(rows []BinaryResult, axisShares map[string]float64, extraDefects map[string][]string) ([]KPI, map[string]float64) {
	totals := make(map[string]int, len(axisShares))
	for _, row := range rows {
		totals[row.Axis] += row.Weight
	}
	return projectBinary(rows, axisShares, extraDefects, totals)
}

// ProjectBinaryScaled preserves cards whose existing Fold weights multiply each
// row's in-axis weight directly by an axis scale, without axis normalization.
func ProjectBinaryScaled(rows []BinaryResult, axisScales map[string]float64, extraDefects map[string][]string) ([]KPI, map[string]float64) {
	return projectBinary(rows, axisScales, extraDefects, nil)
}

func projectBinary(rows []BinaryResult, axisWeights map[string]float64, extraDefects map[string][]string, totals map[string]int) ([]KPI, map[string]float64) {
	kpis := make([]KPI, 0, len(rows))
	weights := make(map[string]float64, len(rows))
	for _, row := range rows {
		kpi := BinaryKPI(row)
		kpi.Defects = append(kpi.Defects, extraDefects[row.Key]...)
		kpis = append(kpis, kpi)
		weight := axisWeights[row.Axis] * float64(row.Weight)
		if totals != nil {
			total := totals[row.Axis]
			if total <= 0 {
				continue
			}
			weight /= float64(total)
		}
		weights[row.Key] = weight
	}
	return kpis, weights
}

// BinaryKPI maps one binary result onto the shared Fold shape.
func BinaryKPI(row BinaryResult) KPI {
	kpi := KPI{Key: row.Key, Group: row.Axis, Detail: row.Detail}
	switch {
	case row.Passed:
		kpi.Score = 100
	case row.Hard:
		kpi.Defects = []string{row.Key + ": " + row.Detail}
	default:
		kpi.Soft = []string{row.Key + ": " + row.Detail}
	}
	return kpi
}

// BinaryAxisScore returns the rounded weighted pass percentage for one axis.
func BinaryAxisScore(rows []BinaryResult) int {
	total, passed := 0, 0
	for _, row := range rows {
		total += row.Weight
		if row.Passed {
			passed += row.Weight
		}
	}
	if total == 0 {
		return 0
	}
	return int(math.Round(100 * float64(passed) / float64(total)))
}

// BinaryPayloads preserves row order while projecting results for JSON output.
func BinaryPayloads(rows []BinaryResult) []BinaryPayload {
	out := make([]BinaryPayload, 0, len(rows))
	for _, row := range rows {
		kpi := BinaryKPI(row)
		out = append(out, BinaryPayload{
			KPI:     kpi.Key,
			Group:   kpi.Group,
			Score:   int(kpi.Score),
			Value:   Round3(ValueFromScore(kpi.Score)),
			Detail:  kpi.Detail,
			Defects: kpi.Defects,
			Soft:    kpi.Soft,
		})
	}
	return out
}
