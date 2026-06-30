package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// snap is a tiny constructor for a steeringSnapshot with no payload, used to drive
// the pure alert gate without parsing JSON.
func snap(index float64, debt, soft int) steeringSnapshot {
	return steeringSnapshot{index: index, debt: debt, softSignal: soft}
}

func TestShouldAlertHardDebtAlwaysFires(t *testing.T) {
	base := &steeringBaseline{Index: 90, Debt: 0, SoftSignals: 5}
	// Even with the index above the floor and signals unchanged, hard debt fires.
	fire, reason := shouldAlert(snap(95, 1, 5), base, 2.0)
	if !fire {
		t.Fatalf("hard debt > 0 must alert; got no-fire (%s)", reason)
	}
	if !strings.Contains(reason, "debt") {
		t.Fatalf("reason should name the debt: %q", reason)
	}
}

func TestShouldAlertNoBaselineFiresFirstRun(t *testing.T) {
	fire, reason := shouldAlert(snap(90, 0, 5), nil, 2.0)
	if !fire {
		t.Fatalf("a missing floor must fire to establish the baseline; got no-fire (%s)", reason)
	}
}

func TestShouldAlertIndexDropFires(t *testing.T) {
	base := &steeringBaseline{Index: 90, Debt: 0, SoftSignals: 5}
	// 90 -> 87.5 is a 2.5 drop, >= the 2.0 delta.
	fire, reason := shouldAlert(snap(87.5, 0, 5), base, 2.0)
	if !fire {
		t.Fatalf("a 2.5 index drop with delta 2.0 must fire; got no-fire (%s)", reason)
	}
	if !strings.Contains(reason, "index dropped") {
		t.Fatalf("reason should name the index drop: %q", reason)
	}
}

func TestShouldAlertSmallDropDoesNotFire(t *testing.T) {
	base := &steeringBaseline{Index: 90, Debt: 0, SoftSignals: 5}
	// 90 -> 88.5 is a 1.5 drop, below the 2.0 delta, and signals unchanged.
	fire, _ := shouldAlert(snap(88.5, 0, 5), base, 2.0)
	if fire {
		t.Fatal("a 1.5 drop below the 2.0 delta must NOT fire")
	}
}

func TestShouldAlertNewDriftSignalFires(t *testing.T) {
	base := &steeringBaseline{Index: 90, Debt: 0, SoftSignals: 5}
	// Index steady, debt 0, but a NEW soft signal appeared (5 -> 6).
	fire, reason := shouldAlert(snap(90, 0, 6), base, 2.0)
	if !fire {
		t.Fatalf("a new drift signal must fire; got no-fire (%s)", reason)
	}
	if !strings.Contains(reason, "drift signals rose") {
		t.Fatalf("reason should name the drift rise: %q", reason)
	}
}

func TestShouldAlertCleanReadDoesNotFire(t *testing.T) {
	base := &steeringBaseline{Index: 90, Debt: 0, SoftSignals: 5}
	// At the floor on every axis -> no regression.
	fire, reason := shouldAlert(snap(90, 0, 5), base, 2.0)
	if fire {
		t.Fatalf("a clean read at the floor must NOT fire: %s", reason)
	}
	// An improvement (higher index, fewer signals) also does not fire.
	fire, _ = shouldAlert(snap(92, 0, 4), base, 2.0)
	if fire {
		t.Fatal("an improvement must NOT fire an alert")
	}
}

func TestIsImprovement(t *testing.T) {
	base := &steeringBaseline{Index: 90, Debt: 0, SoftSignals: 5}
	if !isImprovement(snap(91, 0, 5), base) {
		t.Fatal("higher index, same debt/signals is an improvement")
	}
	if !isImprovement(snap(90, 0, 4), base) {
		t.Fatal("fewer signals, same index/debt is an improvement")
	}
	if isImprovement(snap(90, 0, 5), base) {
		t.Fatal("identical to the floor is NOT an improvement")
	}
	if isImprovement(snap(95, 1, 5), base) {
		t.Fatal("a higher index but WORSE debt is not a clean improvement")
	}
	if !isImprovement(snap(90, 0, 5), nil) {
		t.Fatal("any read against a missing floor is an improvement (establishes it)")
	}
}

const sampleSteerJSON = `{
  "schema": "fak-steerability-scorecard/1",
  "verdict": "OK",
  "finding": "no hard steerability-debt",
  "corpus": {
    "index": 89.9,
    "score": 89.9,
    "grade": "B",
    "steerability_debt": 0,
    "soft_signals": 5,
    "index_by_group": {"modularity": 81.5, "coupling": 99.0, "navigability": 68.0, "correction": 97.3},
    "breakdown": [
      {"kpi": "func_size_dist", "group": "modularity", "score": 49, "debt": 0, "soft": 0, "detail": "no soft", "index_gain_to_clean": 5.1},
      {"kpi": "package_doc_frac", "group": "navigability", "score": 68, "debt": 0, "soft": 1, "detail": "146/214 packages documented", "index_gain_to_clean": 3.2},
      {"kpi": "god_file_rate", "group": "modularity", "score": 83, "debt": 0, "soft": 1, "detail": "3/858 files > 1500 lines", "index_gain_to_clean": 1.2}
    ]
  },
  "kpis": [
    {"kpi": "func_size_dist", "group": "modularity", "score": 49, "detail": "d", "defects": [], "soft": []},
    {"kpi": "god_file_rate", "group": "modularity", "score": 83, "detail": "d", "defects": [], "soft": ["x"]}
  ]
}`

func TestParseSteeringSnapshot(t *testing.T) {
	s, err := parseSteeringSnapshot([]byte(sampleSteerJSON))
	if err != nil {
		t.Fatal(err)
	}
	if s.index != 89.9 || s.debt != 0 || s.softSignal != 5 {
		t.Fatalf("corpus fold wrong: index=%v debt=%v soft=%v", s.index, s.debt, s.softSignal)
	}
	// drift keeps only KPIs with a soft signal, worst-first.
	if len(s.drift) != 2 {
		t.Fatalf("want 2 drift rows (soft>0), got %d: %+v", len(s.drift), s.drift)
	}
	for _, d := range s.drift {
		if d.Soft <= 0 {
			t.Fatalf("drift row with soft<=0 leaked in: %+v", d)
		}
	}
}

func TestSteeringActionsPointAtOwningSkill(t *testing.T) {
	s, err := parseSteeringSnapshot([]byte(sampleSteerJSON))
	if err != nil {
		t.Fatal(err)
	}
	actions := steeringActions(s)
	if len(actions) == 0 {
		t.Fatal("expected at least the re-measure action")
	}
	joined := ""
	for _, a := range actions {
		joined += a.Label + "|" + a.URL + "\n"
		if a.URL == "" {
			t.Fatalf("a steering action must have a URL (link-button): %+v", a)
		}
	}
	// god_file_rate -> /modularize ; package_doc_frac -> /curate-cluster.
	if !strings.Contains(joined, "modularize") {
		t.Fatalf("god_file_rate drift should point at /modularize; got:\n%s", joined)
	}
	if !strings.Contains(joined, "curate-cluster") {
		t.Fatalf("package_doc_frac drift should point at /curate-cluster; got:\n%s", joined)
	}
	// The re-measure affordance is always present.
	if !strings.Contains(joined, "Re-measure") {
		t.Fatalf("expected a Re-measure action; got:\n%s", joined)
	}
}

func TestBuildSteeringUpdateModes(t *testing.T) {
	s, err := parseSteeringSnapshot([]byte(sampleSteerJSON))
	if err != nil {
		t.Fatal(err)
	}

	// status: headline, no actions, no group line.
	st := buildSteeringUpdate(s, "status", "ci", "")
	if st.Title != "steerability" || len(st.Actions) != 0 {
		t.Fatalf("status mode wrong: title=%q actions=%d", st.Title, len(st.Actions))
	}

	// report: full snapshot with a per-group index line + actions.
	rp := buildSteeringUpdate(s, "report", "ci", "")
	if rp.Title != "steerability report" {
		t.Fatalf("report title wrong: %q", rp.Title)
	}
	if len(rp.Actions) == 0 {
		t.Fatal("report should carry actions")
	}
	foundGroup := false
	for _, l := range rp.Lines {
		if strings.Contains(l, "modularity") && strings.Contains(l, "coupling") {
			foundGroup = true
		}
	}
	if !foundGroup {
		t.Fatalf("report should include the per-group index line; lines=%v", rp.Lines)
	}
	if !strings.Contains(strings.Join(rp.Lines, "\n"), "+3.2 index pts") {
		t.Fatalf("report should include clean-gain guidance; lines=%v", rp.Lines)
	}

	// alert: ACTION verdict, reason folded into Detail, actions present.
	al := buildSteeringUpdate(s, "alert", "ci", "index dropped")
	if al.Verdict != "ACTION" {
		t.Fatalf("alert verdict should be ACTION, got %q", al.Verdict)
	}
	if !strings.Contains(al.Detail, "index dropped") {
		t.Fatalf("alert reason should be in Detail: %q", al.Detail)
	}
	if len(al.Actions) == 0 {
		t.Fatal("alert should carry actions")
	}
}

func TestSteeringBaselineRoundTrip(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "steering_baseline.json")
	s := snap(88.5, 0, 4)
	if err := writeSteeringBaseline(path, s); err != nil {
		t.Fatal(err)
	}
	b, err := readSteeringBaseline(path)
	if err != nil {
		t.Fatal(err)
	}
	if b == nil || b.Index != 88.5 || b.Debt != 0 || b.SoftSignals != 4 {
		t.Fatalf("round-trip lost data: %+v", b)
	}
	if b.Schema != "fak-steering-baseline/1" {
		t.Fatalf("schema not set: %q", b.Schema)
	}
	// The on-disk JSON is indented + newline-terminated (a clean git diff).
	raw, _ := os.ReadFile(path)
	if !strings.HasSuffix(string(raw), "}\n") {
		t.Fatalf("baseline should end with a newline: %q", string(raw[len(raw)-3:]))
	}
	var probe map[string]any
	if err := json.Unmarshal(raw, &probe); err != nil {
		t.Fatalf("baseline is not valid JSON: %v", err)
	}
}

func TestReadSteeringBaselineMissingIsNotError(t *testing.T) {
	b, err := readSteeringBaseline(filepath.Join(t.TempDir(), "nope.json"))
	if err != nil {
		t.Fatalf("a missing floor must not be an error: %v", err)
	}
	if b != nil {
		t.Fatalf("a missing floor should read as nil, got %+v", b)
	}
}

func TestResolveSteeringChannelPrecedence(t *testing.T) {
	// flag wins over everything.
	t.Setenv("FAK_STEERING_CHANNEL", "C_ENV")
	if got := resolveSteeringChannel("C_FLAG"); got != "C_FLAG" {
		t.Fatalf("flag should win: got %q", got)
	}
	// env wins over the default when no flag.
	if got := resolveSteeringChannel(""); got != "C_ENV" {
		t.Fatalf("env should win over default: got %q", got)
	}
	// the built-in default applies when nothing is set (and no .env file is found).
	t.Setenv("FAK_STEERING_CHANNEL", "")
	t.Setenv("FAK_SCOREBOARD_CHANNEL", "")
	chdir(t, t.TempDir())
	if got := resolveSteeringChannel(""); got != steeringChannelDefault {
		t.Fatalf("default should apply: got %q want %q", got, steeringChannelDefault)
	}

	// The generic scoreboard channel must NOT leak into the steering surface — that
	// would misroute the surface to #scoreboard whenever .env.slack.local is sourced.
	t.Setenv("FAK_SCOREBOARD_CHANNEL", "C_SCOREBOARD")
	if got := resolveSteeringChannel(""); got != steeringChannelDefault {
		t.Fatalf("FAK_SCOREBOARD_CHANNEL must not override the steering default: got %q want %q", got, steeringChannelDefault)
	}
}

// chdir switches to dir for the test and restores the prior cwd after.
func chdir(t *testing.T, dir string) {
	t.Helper()
	prev, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Chdir(dir); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Chdir(prev) })
}
