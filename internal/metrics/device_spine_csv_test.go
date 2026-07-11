package metrics

import (
	"bytes"
	"encoding/csv"
	"strings"
	"testing"
)

// TestRenderCSVSharedTableAndNullOnError proves the CSV surface is a stateless
// reader driven by the SAME metricTable as JSON/Prom: its metric columns are
// exactly the table keys in order (after the identity columns), a present metric
// renders its value, and an unread metric renders an empty cell — the
// null-on-error contract survives to the wire, so an empty cell is
// distinguishable from a measured 0.
func TestRenderCSVSharedTableAndNullOnError(t *testing.T) {
	snap := Collect([]Probe{fakeProbe{backend: "nvml", detect: true, devices: []Device{
		fakeDevice{id: "gpu0", m: map[string]float64{
			"tokens_per_second": 42,
			"utilization_ratio": 0.9,
		}},
	}}})

	out, err := RenderCSV(snap)
	if err != nil {
		t.Fatalf("RenderCSV: %v", err)
	}
	rows, err := csv.NewReader(bytes.NewReader(out)).ReadAll()
	if err != nil {
		t.Fatalf("parse csv: %v", err)
	}
	if len(rows) != 2 {
		t.Fatalf("want header + 1 data row, got %d rows: %v", len(rows), rows)
	}
	header := rows[0]

	// Header metric columns are exactly metricTable keys, in order, after identity.
	wantCols := append([]string{}, csvIdentityColumns...)
	for _, m := range metricTable {
		wantCols = append(wantCols, m.Key)
	}
	if strings.Join(header, ",") != strings.Join(wantCols, ",") {
		t.Fatalf("header not driven by shared table:\n got %v\nwant %v", header, wantCols)
	}

	col := map[string]int{}
	for i, h := range header {
		col[h] = i
	}
	data := rows[1]
	if data[col["backend"]] != "nvml" || data[col["device"]] != "gpu0" {
		t.Fatalf("identity cells wrong: %v", data)
	}
	if data[col["tokens_per_second"]] != "42" {
		t.Fatalf("present metric cell = %q, want 42", data[col["tokens_per_second"]])
	}
	if data[col["utilization_ratio"]] != "0.9" {
		t.Fatalf("present metric cell = %q, want 0.9", data[col["utilization_ratio"]])
	}
	// Unread metrics → empty cells, NOT "0" (null-on-error survives to the wire).
	if data[col["power_watts"]] != "" || data[col["queue_depth"]] != "" {
		t.Fatalf("unread metrics should be empty cells, got power=%q queue=%q",
			data[col["power_watts"]], data[col["queue_depth"]])
	}
}

// TestRenderCSVFederationAndDeterminism proves federated rows carry remote/peer
// columns and that rendering is deterministic (a stateless reader over an
// immutable snapshot).
func TestRenderCSVFederationAndDeterminism(t *testing.T) {
	snap := []DeviceMetrics{
		{Backend: "nvml", DeviceID: "gpu0", PowerWatts: f(120)},
		{Backend: "engine", DeviceID: "vllm0", Remote: true, Peer: "10.0.0.2", QueueDepth: f(3)},
	}
	out, err := RenderCSV(snap)
	if err != nil {
		t.Fatalf("RenderCSV: %v", err)
	}
	rows, err := csv.NewReader(bytes.NewReader(out)).ReadAll()
	if err != nil {
		t.Fatalf("parse csv: %v", err)
	}
	if len(rows) != 3 {
		t.Fatalf("want header + 2 rows, got %d: %v", len(rows), rows)
	}
	col := map[string]int{}
	for i, h := range rows[0] {
		col[h] = i
	}
	local, remote := rows[1], rows[2]
	if local[col["remote"]] != "false" || local[col["peer"]] != "" {
		t.Fatalf("local row federation cells wrong: %v", local)
	}
	if remote[col["remote"]] != "true" || remote[col["peer"]] != "10.0.0.2" {
		t.Fatalf("remote row federation cells wrong: %v", remote)
	}
	if remote[col["queue_depth"]] != "3" {
		t.Fatalf("remote queue_depth = %q, want 3", remote[col["queue_depth"]])
	}

	out2, err := RenderCSV(snap)
	if err != nil {
		t.Fatalf("RenderCSV second: %v", err)
	}
	if !bytes.Equal(out, out2) {
		t.Fatalf("render not deterministic")
	}
}

// TestRenderCSVNilSnapshotHeaderOnly proves a nil snapshot renders the schema
// header row alone (still self-describing), not an error or empty output.
func TestRenderCSVNilSnapshotHeaderOnly(t *testing.T) {
	out, err := RenderCSV(nil)
	if err != nil {
		t.Fatalf("RenderCSV nil: %v", err)
	}
	rows, err := csv.NewReader(bytes.NewReader(out)).ReadAll()
	if err != nil {
		t.Fatalf("parse csv: %v", err)
	}
	if len(rows) != 1 {
		t.Fatalf("nil snapshot should render header only, got %d rows: %v", len(rows), rows)
	}
	if rows[0][0] != "backend" {
		t.Fatalf("header row missing: %v", rows[0])
	}
}
