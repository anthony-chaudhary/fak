package dogfoodissues

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func intp(n int) *int { return &n }

// BuildReceipt folds a live Result into per-outcome counts and rows.
func TestBuildReceipt_LiveCounts(t *testing.T) {
	res := Result{
		Report: "/evidence/report.json",
		Planned: []PlanRow{
			{Action: "create", Key: "k1"},
			{Action: "update", Key: "k2", Number: intp(7)},
		},
		Synced: []SyncRow{
			{Key: "k1", Action: "create", OK: true, Number: intp(101)},
			{Key: "k2", Action: "update", OK: false, Number: intp(7)},
		},
		Skipped: []SkippedRow{{Key: "k3", Reason: "vague"}},
	}
	rec := BuildReceipt(res, ReceiptModeLive, time.Date(2026, 7, 1, 12, 0, 0, 0, time.UTC))
	if rec.Schema != ReceiptSchema || rec.Mode != ReceiptModeLive {
		t.Fatalf("schema/mode wrong: %+v", rec)
	}
	if rec.Actions != 3 || rec.Skipped != 1 {
		t.Fatalf("actions/skipped = %d/%d, want 3/1", rec.Actions, rec.Skipped)
	}
	if rec.PlannedCreates != 1 || rec.PlannedUpdates != 1 {
		t.Fatalf("planned creates/updates = %d/%d, want 1/1", rec.PlannedCreates, rec.PlannedUpdates)
	}
	if rec.SyncedOK != 1 || rec.SyncedFailed != 1 {
		t.Fatalf("synced ok/failed = %d/%d, want 1/1", rec.SyncedOK, rec.SyncedFailed)
	}
	if len(rec.Rows) != 2 || rec.Rows[0].Key != "k1" || !rec.Rows[0].OK || rec.Rows[1].OK {
		t.Fatalf("rows wrong: %+v", rec.Rows)
	}
}

// A fetch-existing receipt records the create/update classification; nothing was
// attempted, so no row is a failure.
func TestBuildReceipt_FetchExisting(t *testing.T) {
	res := Result{
		Report: "/evidence/report.json",
		Planned: []PlanRow{
			{Action: "create", Key: "k1"},
			{Action: "update", Key: "k2", Number: intp(9)},
		},
	}
	rec := BuildReceipt(res, ReceiptModeFetchExisting, time.Time{})
	if rec.PlannedCreates != 1 || rec.PlannedUpdates != 1 || rec.SyncedOK != 0 || rec.SyncedFailed != 0 {
		t.Fatalf("counts wrong: %+v", rec)
	}
	if len(rec.Rows) != 2 || !rec.Rows[0].OK || !rec.Rows[1].OK {
		t.Fatalf("fetch-existing rows must be ok=true classifications: %+v", rec.Rows)
	}
	if rec.WrittenAt == "" {
		t.Fatalf("zero now must still stamp WrittenAt")
	}
}

// WriteReceipt round-trips next to the report, under the stable name and with the
// exact JSON field names internal/dogfoodscore reads (the cross-package contract;
// dogfoodscore pins the same shape from its side).
func TestWriteReceipt_RoundTripAndContract(t *testing.T) {
	dir := t.TempDir()
	rec := BuildReceipt(Result{
		Report:  filepath.Join(dir, "report.json"),
		Planned: []PlanRow{{Action: "create", Key: "k1"}},
		Synced:  []SyncRow{{Key: "k1", Action: "create", OK: true, Number: intp(5)}},
	}, ReceiptModeLive, time.Date(2026, 7, 1, 12, 0, 0, 0, time.UTC))
	path, err := WriteReceipt(dir, rec)
	if err != nil {
		t.Fatalf("write receipt: %v", err)
	}
	if filepath.Base(path) != ReceiptName || ReceiptName != "issues-sync.json" {
		t.Fatalf("receipt must be written as issues-sync.json, got %s", path)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read receipt: %v", err)
	}
	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil {
		t.Fatalf("unmarshal receipt: %v", err)
	}
	for _, field := range []string{"schema", "mode", "report", "written_at", "actions", "planned_creates", "planned_updates", "synced_ok", "synced_failed"} {
		if _, ok := m[field]; !ok {
			t.Fatalf("receipt JSON missing contract field %q: %s", field, raw)
		}
	}
	if m["schema"] != ReceiptSchema || m["mode"] != "live" {
		t.Fatalf("schema/mode wrong in JSON: %s", raw)
	}
}

// NewestReport picks the newest stamp's report.json by mtime and names the
// producing command when there is nothing to pick.
func TestNewestReport(t *testing.T) {
	root := t.TempDir()
	if _, err := NewestReport(root); err == nil {
		t.Fatalf("no evidence dir must be an error")
	} else if !strings.Contains(err.Error(), "make dogfood-recent") {
		t.Fatalf("error must name the producing command, got %v", err)
	}

	base := filepath.Join(root, ".fak", "recent-feature-dogfood")
	older := filepath.Join(base, "20260629T000000Z")
	newer := filepath.Join(base, "20260630T000000Z")
	for i, dir := range []string{older, newer} {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatalf("mkdir: %v", err)
		}
		p := filepath.Join(dir, "report.json")
		if err := os.WriteFile(p, []byte("{}"), 0o644); err != nil {
			t.Fatalf("write: %v", err)
		}
		mt := time.Date(2026, 6, 29+i, 0, 0, 0, 0, time.UTC)
		if err := os.Chtimes(p, mt, mt); err != nil {
			t.Fatalf("chtimes: %v", err)
		}
	}
	got, err := NewestReport(root)
	if err != nil {
		t.Fatalf("newest report: %v", err)
	}
	if got != filepath.Join(newer, "report.json") {
		t.Fatalf("newest = %s, want the 20260630 stamp", got)
	}
}

// The receipt line renders on BOTH exits — including the empty-plan early
// return, where a fetch-existing run most often lands (everything skipped or
// already tracked).
func TestRenderShowsReceiptOnEmptyPlan(t *testing.T) {
	out := Render(Result{Mode: "dry-run", Report: "r.json", Receipt: "x/issues-sync.json"})
	if !strings.Contains(out, "receipt: x/issues-sync.json") {
		t.Fatalf("empty-plan render must show the receipt line:\n%s", out)
	}
}
