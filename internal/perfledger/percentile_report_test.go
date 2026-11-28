package perfledger

import (
	"encoding/json"
	"strings"
	"testing"
)

// fak-test:runtime fast est=5ms lane=default
func TestPercentileReportSchema(t *testing.T) {
	vals := make([]float64, 100)
	for i := range vals {
		vals[i] = float64(i + 1)
	}
	row, err := ComputePercentileRow("ttft", "ms", vals)
	if err != nil {
		t.Fatal(err)
	}
	if row.Min != 1 || row.Max != 100 || row.P50 < 50 || row.P50 > 51 || row.P99 < 99 {
		t.Fatalf("unexpected percentile row: %+v", row)
	}

	rep := PercentileReport{Rows: []PercentileRow{row}}
	csvBytes, err := rep.MarshalCSV()
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimSpace(string(csvBytes)), "\n")
	if len(lines) != 2 || lines[0] != strings.Join(PercentileCSVHeader, ",") {
		t.Fatalf("unexpected CSV header/rows: %v", lines)
	}
	jsonBytes, err := rep.MarshalJSONBytes()
	if err != nil {
		t.Fatal(err)
	}
	var decoded PercentileReport
	if err := json.Unmarshal(jsonBytes, &decoded); err != nil || len(decoded.Rows) != 1 || decoded.Rows[0].Metric != "ttft" {
		t.Fatalf("unexpected JSON round-trip: %s err=%v", jsonBytes, err)
	}
}
