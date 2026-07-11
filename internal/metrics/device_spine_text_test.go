package metrics

import (
	"strings"
	"testing"
)

// TestRenderTextSharedTableAndAlignment proves the text surface is a stateless
// reader driven by the SAME metricTable as Prom/CSV/JSON: header metric
// columns are exactly the table keys (uppercased, after identity columns),
// unread metrics render "-" not 0, the origin column distinguishes local from
// federated rows, and columns align (every cell in a column starts at the same
// byte offset).
func TestRenderTextSharedTableAndAlignment(t *testing.T) {
	snap := Federate(
		[]DeviceMetrics{{Backend: "nvml", DeviceID: "gpu0", TokensPerSecond: f(42)}},
		[]DeviceMetrics{{Backend: "engine", DeviceID: "vllm0", Remote: true, Peer: "10.0.0.2", QueueDepth: f(3)}},
	)
	out := string(RenderText(snap))
	lines := strings.Split(strings.TrimRight(out, "\n"), "\n")
	if len(lines) != 3 {
		t.Fatalf("want header + 2 rows, got %d lines:\n%s", len(lines), out)
	}

	header := strings.Fields(lines[0])
	want := []string{"BACKEND", "DEVICE", "ORIGIN"}
	for _, m := range metricTable {
		want = append(want, strings.ToUpper(m.Key))
	}
	if strings.Join(header, ",") != strings.Join(want, ",") {
		t.Fatalf("header not driven by shared table:\n got %v\nwant %v", header, want)
	}

	local, remote := strings.Fields(lines[1]), strings.Fields(lines[2])
	if local[0] != "nvml" || local[2] != "local" {
		t.Fatalf("local row wrong: %v", local)
	}
	if remote[0] != "engine" || remote[2] != "10.0.0.2" {
		t.Fatalf("federated row must show peer origin: %v", remote)
	}
	col := map[string]int{}
	for i, h := range header {
		col[h] = i
	}
	if local[col["TOKENS_PER_SECOND"]] != "42" || local[col["QUEUE_DEPTH"]] != "-" {
		t.Fatalf("local metric cells wrong: %v", local)
	}
	if remote[col["QUEUE_DEPTH"]] != "3" || remote[col["TOKENS_PER_SECOND"]] != "-" {
		t.Fatalf("remote metric cells wrong: %v", remote)
	}

	// Alignment: the DEVICE column starts at the same byte offset in every row.
	headerDeviceAt := strings.Index(lines[0], "DEVICE")
	for i, line := range lines[1:] {
		cells := strings.Fields(line)
		if strings.Index(line, cells[1]) != headerDeviceAt {
			t.Fatalf("row %d DEVICE column misaligned:\n%s", i+1, out)
		}
	}
}

// TestRenderTextNilSnapshotHeaderOnly proves a nil snapshot renders the header
// row alone — self-describing, matching RenderCSV's contract.
func TestRenderTextNilSnapshotHeaderOnly(t *testing.T) {
	out := string(RenderText(nil))
	lines := strings.Split(strings.TrimRight(out, "\n"), "\n")
	if len(lines) != 1 {
		t.Fatalf("nil snapshot should render header only, got %d lines:\n%s", len(lines), out)
	}
	if !strings.HasPrefix(lines[0], "BACKEND") {
		t.Fatalf("header row missing: %q", lines[0])
	}
}
