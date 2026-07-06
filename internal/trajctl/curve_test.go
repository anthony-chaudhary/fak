package trajctl

import (
	"encoding/json"
	"flag"
	"os"
	"path/filepath"
	"testing"
)

// updateGolden regenerates the golden curve reports under testdata/curve. Run
// `go test ./internal/trajctl -run TestCurve -update` (under WSL on this host) to
// refresh them after an intentional schema change, then commit the .json files.
var updateGolden = flag.Bool("update", false, "update curve golden files")

// curveShapes are the four signal shapes named in issue #2538's done condition,
// one fixture ledger each. The objective id is the one the report folds a single
// curve for; want is the signal that fold must derive.
var curveShapes = []struct {
	name      string // testdata/curve/<name>.jsonl and <name>.json
	objective string
	want      Signal
}{
	{"healthy", "traj-healthy", SignalHealthy},
	{"stall", "traj-stall", SignalStall},
	{"drift", "traj-drift", SignalDrift},
	{"detour-overrun", "traj-detour", SignalDetourOverrun},
}

// TestCurveSignalPerShape is the done-condition witness: the fold renders the
// correct closed-vocabulary signal for a fixture ledger of each of the four
// shapes, and the schema-pinned --json report is golden-file stable.
func TestCurveSignalPerShape(t *testing.T) {
	for _, shape := range curveShapes {
		t.Run(shape.name, func(t *testing.T) {
			ledger := filepath.Join("testdata", "curve", shape.name+".jsonl")
			st := Fold(ReadLedgerFile(ledger))

			rep, ok := st.CurveReportFor(shape.objective)
			if !ok {
				t.Fatalf("CurveReportFor(%q): objective missing from %s", shape.objective, ledger)
			}
			if rep.Schema != CurveSchema {
				t.Fatalf("report schema = %q, want %q", rep.Schema, CurveSchema)
			}
			if len(rep.Objectives) != 1 {
				t.Fatalf("report objectives = %d, want 1", len(rep.Objectives))
			}
			if got := rep.Objectives[0].Signal; got != shape.want {
				t.Fatalf("signal = %q, want %q (detail=%q)", got, shape.want, rep.Objectives[0].Detail)
			}

			assertGolden(t, filepath.Join("testdata", "curve", shape.name+".json"), rep)
		})
	}
}

// TestCurveWorstFirstListing pins the no-id shape: folding a ledger that carries
// every open objective lists them worst-first (DETOUR_OVERRUN > DRIFT > STALL >
// HEALTHY), and the closed (met/abandoned) objective is excluded.
func TestCurveWorstFirstListing(t *testing.T) {
	var rows []Row
	for _, shape := range curveShapes {
		rows = append(rows, ReadLedgerFile(filepath.Join("testdata", "curve", shape.name+".jsonl"))...)
	}
	// A closed objective must not appear in the open listing.
	rows = append(rows, ObjectiveRecord(Objective{ID: "traj-done", Statement: "already met", Status: StatusMet}))

	st := Fold(rows)
	rep := st.OpenCurves()

	// traj-epic is the detour's parent; it is paused, which is open (steerable),
	// so it lists too — a HEALTHY with no curve (latest 0.00) sorts ahead of the
	// rising traj-healthy under the lower-progress-is-worse tie-break.
	wantOrder := []struct {
		id  string
		sig Signal
	}{
		{"traj-detour", SignalDetourOverrun},
		{"traj-drift", SignalDrift},
		{"traj-stall", SignalStall},
		{"traj-epic", SignalHealthy},
		{"traj-healthy", SignalHealthy},
	}
	if len(rep.Objectives) != len(wantOrder) {
		t.Fatalf("open curves = %d objectives, want %d (%+v)", len(rep.Objectives), len(wantOrder), rep.Objectives)
	}
	for i, want := range wantOrder {
		got := rep.Objectives[i]
		if got.ObjectiveID != want.id || got.Signal != want.sig {
			t.Fatalf("open curves[%d] = %q/%q, want %q/%q", i, got.ObjectiveID, got.Signal, want.id, want.sig)
		}
	}
	for _, oc := range rep.Objectives {
		if oc.ObjectiveID == "traj-done" {
			t.Fatalf("closed objective leaked into the open listing: %+v", oc)
		}
	}

	assertGolden(t, filepath.Join("testdata", "curve", "worst-first.json"), rep)
}

// TestCurveForUnknownObjective fails closed: folding an id that was never
// declared reports not-found rather than a zero-value curve.
func TestCurveForUnknownObjective(t *testing.T) {
	st := Fold(nil)
	if _, ok := st.CurveReportFor("ghost"); ok {
		t.Fatalf("CurveReportFor(ghost) = ok, want not-found on an empty ledger")
	}
}

// assertGolden compares v's indented JSON to the golden file at path, or rewrites
// the golden when -update is set. The encoding matches the CLI's writeIndentedJSON
// (two-space indent + trailing newline) so a golden doubles as the CLI --json
// contract.
func assertGolden(t *testing.T, path string, v any) {
	t.Helper()
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		t.Fatalf("marshal golden %s: %v", path, err)
	}
	b = append(b, '\n')
	if *updateGolden {
		if err := os.WriteFile(path, b, 0o644); err != nil {
			t.Fatalf("write golden %s: %v", path, err)
		}
		return
	}
	want, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read golden %s: %v (run -update to create it)", path, err)
	}
	if string(want) != string(b) {
		t.Fatalf("golden %s mismatch:\n--- want ---\n%s\n--- got ---\n%s", path, want, b)
	}
}
