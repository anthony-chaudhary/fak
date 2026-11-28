// Structural adaptation of triton-inference-server/perf_analyzer GenAI-Perf
// percentile statistics and CSV/JSON report schema (csv_exporter.py:44 @ 24f6b9af4db5, BSD-3-Clause).
package perfledger

import (
	"bytes"
	"encoding/csv"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"sort"
)

var ErrEmptyPercentileValues = errors.New("perfledger: empty percentile values")

// PercentileCSVHeader is the fixed column order for GenAI-Perf style CSV exports.
var PercentileCSVHeader = []string{
	"metric", "unit", "avg", "min", "max", "std",
	"p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99",
}

// PercentileRow holds summary and p1..p99 statistics for one named metric.
type PercentileRow struct {
	Metric string  `json:"metric"`
	Unit   string  `json:"unit"`
	Avg    float64 `json:"avg"`
	Min    float64 `json:"min"`
	Max    float64 `json:"max"`
	Std    float64 `json:"std"`
	P1     float64 `json:"p1"`
	P5     float64 `json:"p5"`
	P10    float64 `json:"p10"`
	P25    float64 `json:"p25"`
	P50    float64 `json:"p50"`
	P75    float64 `json:"p75"`
	P90    float64 `json:"p90"`
	P95    float64 `json:"p95"`
	P99    float64 `json:"p99"`
}

// ComputePercentileRow computes summary and p1..p99 linear-interpolated percentiles.
func ComputePercentileRow(metric, unit string, values []float64) (PercentileRow, error) {
	if len(values) == 0 {
		return PercentileRow{}, ErrEmptyPercentileValues
	}
	sorted := append([]float64(nil), values...)
	sort.Float64s(sorted)
	n := float64(len(sorted))
	sum := 0.0
	for _, v := range sorted {
		sum += v
	}
	avg := sum / n
	varSq := 0.0
	for _, v := range sorted {
		d := v - avg
		varSq += d * d
	}
	std := math.Sqrt(varSq / n)

	return PercentileRow{
		Metric: metric,
		Unit:   unit,
		Avg:    avg,
		Min:    sorted[0],
		Max:    sorted[len(sorted)-1],
		Std:    std,
		P1:     reportPercentileLinear(sorted, 1),
		P5:     reportPercentileLinear(sorted, 5),
		P10:    reportPercentileLinear(sorted, 10),
		P25:    reportPercentileLinear(sorted, 25),
		P50:    reportPercentileLinear(sorted, 50),
		P75:    reportPercentileLinear(sorted, 75),
		P90:    reportPercentileLinear(sorted, 90),
		P95:    reportPercentileLinear(sorted, 95),
		P99:    reportPercentileLinear(sorted, 99),
	}, nil
}

func reportPercentileLinear(sorted []float64, p float64) float64 {
	if len(sorted) == 1 {
		return sorted[0]
	}
	rank := (p / 100.0) * float64(len(sorted)-1)
	lo := int(math.Floor(rank))
	hi := int(math.Ceil(rank))
	if lo == hi {
		return sorted[lo]
	}
	frac := rank - float64(lo)
	return sorted[lo]*(1-frac) + sorted[hi]*frac
}

// PercentileReport is the JSON/CSV serializable percentile report.
type PercentileReport struct {
	Rows []PercentileRow `json:"rows"`
}

// MarshalCSV renders r with the fixed PercentileCSVHeader.
func (r PercentileReport) MarshalCSV() ([]byte, error) {
	var buf bytes.Buffer
	w := csv.NewWriter(&buf)
	if err := w.Write(PercentileCSVHeader); err != nil {
		return nil, err
	}
	for _, row := range r.Rows {
		rec := []string{
			row.Metric, row.Unit,
			fmt.Sprintf("%.6g", row.Avg), fmt.Sprintf("%.6g", row.Min),
			fmt.Sprintf("%.6g", row.Max), fmt.Sprintf("%.6g", row.Std),
			fmt.Sprintf("%.6g", row.P1), fmt.Sprintf("%.6g", row.P5),
			fmt.Sprintf("%.6g", row.P10), fmt.Sprintf("%.6g", row.P25),
			fmt.Sprintf("%.6g", row.P50), fmt.Sprintf("%.6g", row.P75),
			fmt.Sprintf("%.6g", row.P90), fmt.Sprintf("%.6g", row.P95),
			fmt.Sprintf("%.6g", row.P99),
		}
		if err := w.Write(rec); err != nil {
			return nil, err
		}
	}
	w.Flush()
	return buf.Bytes(), w.Error()
}

// MarshalJSONBytes renders r as JSON.
func (r PercentileReport) MarshalJSONBytes() ([]byte, error) {
	return json.Marshal(r)
}
