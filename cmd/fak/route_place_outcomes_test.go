package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// Witnesses for `fak route --place --outcomes` (epic #5416, tracks D and F).
//
// This is the end of the automatic path: a journal a fleet appended to, read by the same
// binary that places the work, with nobody asserting a capability anywhere. Which is also
// why it is the easiest place to manufacture one — replay the file, drop the timestamps,
// lower the bar and say nothing — so most of what follows is a test that the surface
// refuses to be flattered by its own input.

// writeJournal appends outcomes through the real writer and returns the file path.
func writeJournal(t *testing.T, outcomes []modelroute.TurnOutcome) string {
	t.Helper()
	var buf bytes.Buffer
	for _, o := range outcomes {
		if err := modelroute.AppendTurnOutcome(&buf, o); err != nil {
			t.Fatal(err)
		}
	}
	path := filepath.Join(t.TempDir(), "turns.jsonl")
	if err := os.WriteFile(path, buf.Bytes(), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

// routineTurns builds n witnessed routine outcomes for one model, `fails` of them failed.
func routineTurns(model string, n, fails int, age time.Duration) []modelroute.TurnOutcome {
	out := make([]modelroute.TurnOutcome, 0, n)
	for i := 0; i < n; i++ {
		out = append(out, modelroute.TurnOutcome{
			ID: fmt.Sprintf("%s-%d", model, i), Model: model, Class: modelroute.ClassRoutine,
			Zone: modelroute.ZoneDevice, Success: i >= fails, Verify: modelroute.VerifyWitness,
			At: time.Now().Add(-age),
		})
	}
	return out
}

func TestAJournalOfRealTurnsPlacesWorkOnTheDeviceRung(t *testing.T) {
	r := routePlaceRoster()
	path := writeJournal(t, routineTurns("rung-device", 40, 2, time.Hour))
	code, out, errOut := routePlaceRunOpts(t, &r, map[string]string{"work_class": "routine"},
		placeOptions{OutcomesPath: path, Since: "30d"})
	if code != 0 {
		t.Fatalf("exit = %d, stderr = %q", code, errOut)
	}
	if !strings.Contains(out, "zone=device") || !strings.Contains(out, "model=rung-device") {
		t.Fatalf("a journal of witnessed turns did not move routine work onto the device rung:\n%s", out)
	}
	if !strings.Contains(out, "40 line(s), 40 counted") {
		t.Errorf("the journal summary does not account for the corpus:\n%s", out)
	}
	if !strings.Contains(out, "38/40 witness") {
		t.Errorf("the grade does not carry the counted turns:\n%s", out)
	}
}

func TestAReplayedJournalCannotManufactureAGrade(t *testing.T) {
	// The same 12 turns appended twice. Counting them naively reads as 24 attempts and
	// clears the 20-attempt floor; counting them honestly leaves 12 and grades nothing.
	// This is the cheapest attack on the whole scheme and it costs one duplicated file.
	r := routePlaceRoster()
	turns := routineTurns("rung-device", 12, 0, time.Hour)
	path := writeJournal(t, append(append([]modelroute.TurnOutcome{}, turns...), turns...))
	code, out, errOut := routePlaceRunOpts(t, &r, map[string]string{"work_class": "routine"},
		placeOptions{OutcomesPath: path})
	if code != 0 {
		t.Fatalf("exit = %d, stderr = %q", code, errOut)
	}
	if !strings.Contains(out, "zone=vendor") {
		t.Fatalf("a replayed journal bought a cheap rung:\n%s", out)
	}
	if !strings.Contains(out, modelroute.ReasonInsufficientSamples) {
		t.Errorf("the shortfall is not reported as a sample shortfall:\n%s", out)
	}
	if !strings.Contains(out, "12 replayed id(s)") {
		t.Errorf("the replay is invisible in the summary:\n%s", out)
	}
}

func TestAnUnstampedJournalIsReportedAsUncheckableForReplay(t *testing.T) {
	// A producer that never sets an id leaves a corpus nobody can check. It still counts —
	// a missing id is not proof of a duplicate — but the operator has to be told that the
	// number they are reading is one their own producer could inflate.
	r := routePlaceRoster()
	turns := routineTurns("rung-device", 30, 1, time.Hour)
	for i := range turns {
		turns[i].ID = ""
	}
	code, out, errOut := routePlaceRunOpts(t, &r, map[string]string{"work_class": "routine"},
		placeOptions{OutcomesPath: writeJournal(t, turns)})
	if code != 0 {
		t.Fatalf("exit = %d, stderr = %q", code, errOut)
	}
	if !strings.Contains(out, "30 counted turn(s) carry no id") {
		t.Errorf("an uncheckable corpus was reported as a clean one:\n%s", out)
	}
}

func TestAWindowExcludesStaleAndUndatedTurnsForDifferentReasons(t *testing.T) {
	// Capability is a property of a model AS DEPLOYED, so a window is a real question. The
	// two exclusions are reported apart because the fixes are opposite: widen the window,
	// or fix the producer that is not stamping its rows.
	r := routePlaceRoster()
	turns := routineTurns("rung-device", 30, 0, 400*24*time.Hour) // last year's model
	fresh := routineTurns("rung-fleet", 4, 0, time.Hour)
	undated := routineTurns("rung-fleet", 30, 0, time.Hour)
	for i := range undated {
		undated[i].ID = "u" + undated[i].ID
		undated[i].At = time.Time{}
	}
	path := writeJournal(t, append(append(turns, fresh...), undated...))
	code, out, errOut := routePlaceRunOpts(t, &r, map[string]string{"work_class": "routine"},
		placeOptions{OutcomesPath: path, Since: "30d"})
	if code != 0 {
		t.Fatalf("exit = %d, stderr = %q", code, errOut)
	}
	if !strings.Contains(out, "zone=vendor") {
		t.Fatalf("stale and undated evidence bought a cheap rung:\n%s", out)
	}
	if !strings.Contains(out, "30 older than 30d") {
		t.Errorf("the stale turns are not reported as stale:\n%s", out)
	}
	if !strings.Contains(out, "30 undated") {
		t.Errorf("the undated turns are not reported as undated:\n%s", out)
	}
	// Without a window, the same journal grades the year-old model — which is exactly why
	// the window exists, and why asking for one has to be the operator's choice.
	code, out, _ = routePlaceRunOpts(t, &r, map[string]string{"work_class": "routine"},
		placeOptions{OutcomesPath: path})
	if code != 0 || !strings.Contains(out, "zone=device") {
		t.Errorf("with no window asked for, the full journal should still grade:\n%s", out)
	}
}

func TestTwoSourcesOfEvidenceAreRefusedRatherThanMerged(t *testing.T) {
	r := routePlaceRoster()
	journal := writeJournal(t, routineTurns("rung-device", 30, 0, time.Hour))
	summary := writeEvidence(t, `{"evidence": {"rung-fleet": [{"class": "routine", "attempts": 60, "successes": 60, "verify": "witness"}]}}`)
	code, out, errOut := routePlaceRunOpts(t, &r, map[string]string{"work_class": "routine"},
		placeOptions{OutcomesPath: journal, EvidencePath: summary})
	if code != 2 {
		t.Fatalf("exit = %d, want 2; stdout=%q", code, out)
	}
	if !strings.Contains(errOut, "--evidence") || !strings.Contains(errOut, "--outcomes") {
		t.Errorf("the refusal does not name both flags: %q", errOut)
	}
	// The journal-only flags do not silently apply to a pre-folded summary either: an
	// --evidence file is already a summary, and nothing here can tell which turns it
	// summarised, so accepting --since against it would be a window nobody applied.
	for _, opts := range []placeOptions{
		{EvidencePath: summary, Since: "30d"},
		{EvidencePath: summary, FloorSpec: "attempts=5"},
	} {
		if code, _, errOut := routePlaceRunOpts(t, &r, map[string]string{"work_class": "routine"}, opts); code != 2 {
			t.Errorf("opts %+v: exit = %d, want 2 (%q)", opts, code, errOut)
		}
	}
}

func TestTheEvidentiaryBarIsTheOperatorsAndAMistypedOneIsRefused(t *testing.T) {
	r := routePlaceRoster()
	path := writeJournal(t, routineTurns("rung-device", 30, 3, time.Hour)) // 90%
	// 90% clears the default 80% bar.
	if code, out, _ := routePlaceRunOpts(t, &r, map[string]string{"work_class": "routine"},
		placeOptions{OutcomesPath: path}); code != 0 || !strings.Contains(out, "zone=device") {
		t.Fatalf("precondition: 27/30 should grade under the default floor:\n%s", out)
	}
	// A fleet that wants 95% gets 95%, and the work goes back to the vendor.
	code, out, errOut := routePlaceRunOpts(t, &r, map[string]string{"work_class": "routine"},
		placeOptions{OutcomesPath: path, FloorSpec: "attempts=20,rate=0.95"})
	if code != 0 {
		t.Fatalf("exit = %d, stderr = %q", code, errOut)
	}
	if !strings.Contains(out, "zone=vendor") || !strings.Contains(out, modelroute.ReasonBelowSuccessFloor) {
		t.Errorf("a stricter bar did not pull the work back:\n%s", out)
	}
	// Provenance can be tightened the same way, and a judge score stops counting.
	judged := routineTurns("rung-device", 40, 0, time.Hour)
	for i := range judged {
		judged[i].Verify = modelroute.VerifyJudge
	}
	code, out, _ = routePlaceRunOpts(t, &r, map[string]string{"work_class": "routine"},
		placeOptions{OutcomesPath: writeJournal(t, judged), FloorSpec: "witness"})
	if code != 0 || !strings.Contains(out, "zone=vendor") || !strings.Contains(out, "40 attempt(s) refused") {
		t.Errorf("--grade-floor witness did not refuse a judge-scored corpus:\n%s", out)
	}
	// A mistyped bar is refused, never applied silently: grading against a bar nobody set
	// is the exact mistake this track exists to prevent.
	// attempts=-5 is in this list because a negative bar is not a harmless typo: every
	// model with any evidence at all clears it, so one lucky turn grades a laptop.
	for _, spec := range []string{"attempts=twenty", "attempts=-5", "rate=90", "atempts=20", "witnes", "attempts"} {
		if code, _, errOut := routePlaceRunOpts(t, &r, map[string]string{"work_class": "routine"},
			placeOptions{OutcomesPath: path, FloorSpec: spec}); code != 2 {
			t.Errorf("--grade-floor %q: exit = %d, want 2 (%q)", spec, code, errOut)
		}
	}
	for _, spec := range []string{"7", "0d", "-3d", "soon"} {
		if code, _, errOut := routePlaceRunOpts(t, &r, map[string]string{"work_class": "routine"},
			placeOptions{OutcomesPath: path, Since: spec}); code != 2 {
			t.Errorf("--since %q: exit = %d, want 2 (%q)", spec, code, errOut)
		}
	}
}

func TestAnEmptyOrMissingJournalIsAUsageErrorNotAnEmptyGrading(t *testing.T) {
	r := routePlaceRoster()
	if code, _, errOut := routePlaceRunOpts(t, &r, map[string]string{"work_class": "routine"},
		placeOptions{OutcomesPath: filepath.Join(t.TempDir(), "nope.jsonl")}); code != 2 {
		t.Fatalf("a missing journal: exit = %d, want 2 (%q)", code, errOut)
	}
	empty := filepath.Join(t.TempDir(), "empty.jsonl")
	if err := os.WriteFile(empty, []byte("\n\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if code, _, errOut := routePlaceRunOpts(t, &r, map[string]string{"work_class": "routine"},
		placeOptions{OutcomesPath: empty}); code != 2 {
		t.Fatalf("an empty journal: exit = %d, want 2 (%q)", code, errOut)
	}
}

func TestATornJournalStillGradesAndSaysWhatItLost(t *testing.T) {
	// A fleet appending to this file will eventually die mid-write. Refusing the whole
	// corpus over one partial row would make the honest path the fragile one.
	r := routePlaceRoster()
	path := writeJournal(t, routineTurns("rung-device", 30, 0, time.Hour))
	f, err := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0o600)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.WriteString(`{"id":"torn","model":"rung-dev`); err != nil {
		t.Fatal(err)
	}
	f.Close()
	code, out, errOut := routePlaceRunOpts(t, &r, map[string]string{"work_class": "routine"},
		placeOptions{OutcomesPath: path})
	if code != 0 {
		t.Fatalf("a torn final line failed the whole run: exit = %d, stderr = %q", code, errOut)
	}
	if !strings.Contains(out, "zone=device") {
		t.Errorf("the 30 good turns were discarded over one torn line:\n%s", out)
	}
	if !strings.Contains(out, "1 unparseable line(s)") {
		t.Errorf("the torn line was swallowed rather than reported:\n%s", out)
	}
}

func TestTheJSONReportCarriesTheJournalAccounting(t *testing.T) {
	r := routePlaceRoster()
	turns := routineTurns("rung-device", 30, 0, time.Hour)
	turns = append(turns, turns[0]) // one replay
	code, out, errOut := routePlaceRunOpts(t, &r, map[string]string{"work_class": "routine"},
		placeOptions{OutcomesPath: writeJournal(t, turns), Since: "30d", JSON: true})
	if code != 0 {
		t.Fatalf("exit = %d, stderr = %q", code, errOut)
	}
	var got placementReport
	if err := json.Unmarshal([]byte(out), &got); err != nil {
		t.Fatalf("--json is not valid JSON: %v\n%s", err, out)
	}
	if got.Journal == nil {
		t.Fatal("the JSON report carries no journal accounting")
	}
	if got.Journal.Journal.Lines != 31 || got.Journal.Fold.Counted != 30 || got.Journal.Fold.Duplicates != 1 {
		t.Errorf("journal accounting = %+v", got.Journal)
	}
	if got.Journal.Floor != modelroute.DefaultGradeFloor() || got.Journal.Since != "30d" {
		t.Errorf("the report does not record the bar it graded against: %+v", got.Journal)
	}
	if got.Placement.Zone != modelroute.ZoneDevice || !got.Placement.Measured {
		t.Errorf("placement = %+v, want a measured device placement", got.Placement)
	}
}
