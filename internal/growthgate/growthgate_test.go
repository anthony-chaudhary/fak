package growthgate

import "testing"

const mbf = 1 << 20

// TestClassifyPath pins the classification table to the real runtime paths seen
// on the reference box, for both absolute (Windows) and repo-relative spellings.
func TestClassifyPath(t *testing.T) {
	cases := []struct {
		path string
		want Class
	}{
		{`C:\work\fak\.dos\metrics\observations.jsonl`, ClassDosMetrics},
		{`./.dos/metrics/observations.jsonl`, ClassDosMetrics},
		{`./.dos/_dos_park/metrics/observations.jsonl`, ClassDosMetrics},
		{`./.dos/lane-journal.jsonl`, ClassLaneJournal},
		{`./.dos/_dos_park/lane-journal.jsonl`, ClassLaneJournal},
		{`./.dispatch-runs/superloop-200-wide-20260704-0731.log`, ClassDispatchLog},
		{`./.dispatch-runs/overnight-dispatcher.jsonl`, ClassDispatchLog},
		{`./.goal-runs/lab-marathon-runner-20260703.log`, ClassGoalLog},
		{`./.fak/loops.jsonl`, ClassLoops},
		{`./.fak/toolproc/journal.jsonl`, ClassToolproc},
		{`./.fak/toolproc/journal.archived.jsonl`, ClassToolproc},
		{`C:/work/fak-private/fleet-runs/nightrun/watch.out.log`, ClassFleetRun},
		{`./some/other/thing.log`, ClassLog},
		{`./some/other/thing.err`, ClassLog},
		{`./benchmark/enterpriseops/measurements.jsonl`, ClassLedger},
		{`./README.md`, ClassOther},
	}
	for _, c := range cases {
		if got := ClassifyPath(c.path); got != c.want {
			t.Errorf("ClassifyPath(%q) = %q, want %q", c.path, got, c.want)
		}
	}
}

// TestClassifyRealOffenders feeds the census that actually existed on the box and
// asserts the top offenders land as ACTION with the right class + hot flag.
func TestClassifyRealOffenders(t *testing.T) {
	arts := []Artifact{
		{Path: "./.dos/metrics/observations.jsonl", Size: 119546680, ModAgeSec: 3},               // hot, huge
		{Path: "./.dos/_dos_park/metrics/observations.jsonl", Size: 116788783, ModAgeSec: 90000}, // cold parked copy
		{Path: "./.dispatch-runs/superloop-200-wide.log", Size: 47019668, ModAgeSec: 500000},
		{Path: "./.dos/lane-journal.jsonl", Size: 22459548, ModAgeSec: 10},
		{Path: "./.fak/loops.jsonl", Size: 23340033, ModAgeSec: 10},
		{Path: "./README.md", Size: 4096, ModAgeSec: 100},           // unclassified-ish, tiny → OK
		{Path: "./fresh-ledger.jsonl", Size: 1 * mbf, ModAgeSec: 1}, // small ledger → OK
	}
	r := Classify(arts, DefaultBudget())

	if r.Verdict != SevAction {
		t.Fatalf("verdict = %q, want action", r.Verdict)
	}
	if r.Scanned != len(arts) {
		t.Errorf("scanned = %d, want %d", r.Scanned, len(arts))
	}
	// 5 offenders cross a budget; the two tiny files do not.
	if len(r.Findings) != 5 { //boundarylint:ignore CHANGE_DETECTOR_TEST the fixture is constructed with exactly 5 over-budget artifacts
		t.Fatalf("findings = %d, want 5: %+v", len(r.Findings), r.Findings)
	}
	// Worst-first, largest-first: the 119 MB observations.jsonl leads.
	top := r.Findings[0]
	if top.Class != ClassDosMetrics || top.Severity != SevAction {
		t.Errorf("top finding = %+v, want dos-metrics/action", top)
	}
	if !top.Hot {
		t.Errorf("top finding should be HOT (modified 3s ago): %+v", top)
	}
	if top.Remedy == "" {
		t.Errorf("finding should carry a remedy hint")
	}
	// The parked copy is COLD (aged out).
	var park *Finding
	for i := range r.Findings {
		if r.Findings[i].Path == "./.dos/_dos_park/metrics/observations.jsonl" {
			park = &r.Findings[i]
		}
	}
	if park == nil {
		t.Fatal("parked copy should be flagged")
	}
	if park.Hot {
		t.Errorf("parked copy should be COLD: %+v", *park)
	}
}

// TestBudgetBoundaries checks the warn/fail bands are half-open at the threshold.
func TestBudgetBoundaries(t *testing.T) {
	b := DefaultBudget()
	// dos-metrics: warn 4MB, fail 16MB.
	mk := func(size int64) Severity {
		r := Classify([]Artifact{{Path: "./.dos/metrics/observations.jsonl", Size: size}}, b)
		if len(r.Findings) == 0 {
			return SevOK
		}
		return r.Findings[0].Severity
	}
	if s := mk(4*mbf - 1); s != SevOK {
		t.Errorf("just under warn = %q, want ok", s)
	}
	if s := mk(4 * mbf); s != SevWatch {
		t.Errorf("at warn = %q, want watch", s)
	}
	if s := mk(16*mbf - 1); s != SevWatch {
		t.Errorf("just under fail = %q, want watch", s)
	}
	if s := mk(16 * mbf); s != SevAction {
		t.Errorf("at fail = %q, want action", s)
	}
}

// TestPerClassEnvelopes confirms human run logs get a larger envelope than
// telemetry: a 20 MB dispatch log is only WATCH, but a 20 MB observations.jsonl
// is ACTION.
func TestPerClassEnvelopes(t *testing.T) {
	b := DefaultBudget()
	dispatch := Classify([]Artifact{{Path: "./.dispatch-runs/x.log", Size: 20 * mbf}}, b)
	if dispatch.Verdict != SevWatch {
		t.Errorf("20MB dispatch log verdict = %q, want watch", dispatch.Verdict)
	}
	metrics := Classify([]Artifact{{Path: "./.dos/metrics/observations.jsonl", Size: 20 * mbf}}, b)
	if metrics.Verdict != SevAction {
		t.Errorf("20MB observations verdict = %q, want action", metrics.Verdict)
	}
}

// TestByClassTotals checks the per-class rollup sums every scanned file (not just
// flagged ones) and is sorted bytes-desc.
func TestByClassTotals(t *testing.T) {
	arts := []Artifact{
		{Path: "./.dispatch-runs/a.log", Size: 10 * mbf},
		{Path: "./.dispatch-runs/b.log", Size: 30 * mbf},
		{Path: "./.dos/metrics/observations.jsonl", Size: 100 * mbf},
	}
	r := Classify(arts, DefaultBudget())
	if len(r.ByClass) != 2 {
		t.Fatalf("by-class buckets = %d, want 2", len(r.ByClass))
	}
	// dos-metrics (100MB) sorts before dispatch-log (40MB total).
	if r.ByClass[0].Class != ClassDosMetrics || r.ByClass[0].Bytes != 100*mbf {
		t.Errorf("first bucket = %+v, want dos-metrics 100MB", r.ByClass[0])
	}
	if r.ByClass[1].Class != ClassDispatchLog || r.ByClass[1].Bytes != 40*mbf || r.ByClass[1].Count != 2 {
		t.Errorf("second bucket = %+v, want dispatch-log 40MB count 2", r.ByClass[1])
	}
	if r.ByClass[1].MaxBytes != 30*mbf {
		t.Errorf("dispatch-log max = %d, want 30MB", r.ByClass[1].MaxBytes)
	}
}

// TestEmptyCensusIsOK is the fail-open floor: nothing scanned ⇒ ok verdict.
func TestEmptyCensusIsOK(t *testing.T) {
	r := Classify(nil, DefaultBudget())
	if r.Verdict != SevOK || r.Scanned != 0 || len(r.Findings) != 0 {
		t.Errorf("empty census = %+v, want ok/0/none", r)
	}
}

// TestDisposableClasses pins which classes a reaper may hard-delete: pure logs +
// telemetry yes; WALs / chained ledgers never.
func TestDisposableClasses(t *testing.T) {
	disposable := []Class{ClassDosMetrics, ClassDispatchLog, ClassGoalLog, ClassFleetRun, ClassLog}
	protected := []Class{ClassLaneJournal, ClassLoops, ClassToolproc, ClassLedger, ClassOther}
	for _, c := range disposable {
		if !c.Disposable() {
			t.Errorf("class %q should be disposable", c)
		}
	}
	for _, c := range protected {
		if c.Disposable() {
			t.Errorf("class %q must NOT be disposable (WAL/chained/unknown)", c)
		}
	}
}

// TestReapPlan checks the safety partition: only COLD + ACTION + disposable files
// are reapable; HOT files, under-budget files, and WAL/chained ledgers are
// protected even when huge.
func TestReapPlan(t *testing.T) {
	arts := []Artifact{
		{Path: "./.dispatch-runs/cold.log", Size: 80 * mbf, ModAgeSec: 999999},         // COLD disposable ACTION → REAP
		{Path: "./.goal-runs/cold.log", Size: 70 * mbf, ModAgeSec: 999999},             // COLD disposable ACTION → REAP
		{Path: "./.dispatch-runs/hot.log", Size: 80 * mbf, ModAgeSec: 1},               // HOT → protected
		{Path: "./.dos/lane-journal.jsonl", Size: 80 * mbf, ModAgeSec: 999999},         // COLD but WAL → protected
		{Path: "./.fak/loops.jsonl", Size: 80 * mbf, ModAgeSec: 999999},                // COLD but chained → protected
		{Path: "./.dos/metrics/observations.jsonl", Size: 80 * mbf, ModAgeSec: 999999}, // COLD disposable telemetry → REAP
	}
	rep := Classify(arts, DefaultBudget())
	reap, protected := ReapPlan(rep)

	if len(reap) != 3 {
		t.Fatalf("reapable = %d, want 3: %+v", len(reap), reap)
	}
	// Sorted largest-first.
	if !(reap[0].Size >= reap[1].Size && reap[1].Size >= reap[2].Size) {
		t.Errorf("reap not sorted largest-first: %v", []int64{reap[0].Size, reap[1].Size, reap[2].Size})
	}
	for _, f := range reap {
		if !f.Class.Disposable() || f.Hot || f.Severity != SevAction {
			t.Errorf("unsafe file in reap set: %+v", f)
		}
		if f.Class == ClassLaneJournal || f.Class == ClassLoops {
			t.Errorf("WAL/chained ledger must never be reaped: %+v", f)
		}
	}
	// The WAL, the chained loops ledger, and the HOT log are all protected.
	var sawWAL, sawLoops, sawHot bool
	for _, f := range protected {
		switch {
		case f.Class == ClassLaneJournal:
			sawWAL = true
		case f.Class == ClassLoops:
			sawLoops = true
		case f.Hot:
			sawHot = true
		}
	}
	if !sawWAL || !sawLoops || !sawHot {
		t.Errorf("protected set incomplete: WAL=%v loops=%v hot=%v", sawWAL, sawLoops, sawHot)
	}
}

// TestDeterministicOrder runs the same census twice and requires identical
// finding order (the report feeds a JSONL ratchet, so it must be stable).
func TestDeterministicOrder(t *testing.T) {
	arts := []Artifact{
		{Path: "./.fak/loops.jsonl", Size: 25 * mbf},
		{Path: "./.dos/metrics/observations.jsonl", Size: 25 * mbf},
		{Path: "./.dispatch-runs/z.log", Size: 25 * mbf},
	}
	a := Classify(arts, DefaultBudget())
	b := Classify(arts, DefaultBudget())
	if len(a.Findings) != len(b.Findings) {
		t.Fatalf("finding counts differ: %d vs %d", len(a.Findings), len(b.Findings))
	}
	for i := range a.Findings {
		if a.Findings[i].Path != b.Findings[i].Path {
			t.Errorf("order differs at %d: %q vs %q", i, a.Findings[i].Path, b.Findings[i].Path)
		}
	}
}
