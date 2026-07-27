package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// Witnesses for `fak route --place --evidence` (epic #5416, track F).
//
// The question this surface answers is the one an operator actually asks: "I have 500
// engineers, I have local hardware, why is everything still going to the vendor?" The
// answers must be distinguishable — nobody measured, the measurement was self-reported,
// there was not enough of it, or the model genuinely failed — because they call for
// completely different responses, and because the tempting bug in all of them is to just
// let the cheap rung serve and hope.

// writeEvidence drops an --evidence file in a temp dir and returns its path.
func writeEvidence(t *testing.T, body string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "outcomes.json")
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestObservedOutcomesMoveWorkOffTheVendorWithoutAnyHandDeclaration(t *testing.T) {
	// The whole epic in one test: nobody asserts anything about the local model. A file
	// of independently verified outcomes is all it takes for routine work to stop
	// costing vendor tokens.
	r := routePlaceRoster()
	path := writeEvidence(t, `{
	  "evidence": {
	    "rung-device": [{"class": "routine", "attempts": 60, "successes": 57, "verify": "witness"}]
	  }
	}`)
	code, out, errOut := routePlaceRunOpts(t, &r, map[string]string{"work_class": "routine"},
		placeOptions{EvidencePath: path})
	if code != 0 {
		t.Fatalf("exit = %d, stderr = %q", code, errOut)
	}
	if !strings.Contains(out, "zone=device") || !strings.Contains(out, "model=rung-device") {
		t.Fatalf("verified outcomes did not move routine work onto the device rung:\n%s", out)
	}
	// And the report says what bought it, so the placement is auditable rather than magic.
	if !strings.Contains(out, "1 of 3 model(s) MEASURED") {
		t.Errorf("the grade summary is missing:\n%s", out)
	}
	if !strings.Contains(out, "57/60 witness") {
		t.Errorf("the grade does not carry its evidence trail:\n%s", out)
	}
}

func TestASelfReportedOutcomeFileChangesNothing(t *testing.T) {
	// Same numbers, same model, one field different: the successes are the model's own
	// word. This must place exactly as if the file were empty — and must say so, rather
	// than leaving the operator to wonder whether the file was even read.
	r := routePlaceRoster()
	path := writeEvidence(t, `{
	  "evidence": {
	    "rung-device": [{"class": "routine", "attempts": 600, "successes": 600, "verify": ""}]
	  }
	}`)
	code, out, errOut := routePlaceRunOpts(t, &r, map[string]string{"work_class": "routine"},
		placeOptions{EvidencePath: path})
	if code != 0 {
		t.Fatalf("exit = %d, stderr = %q", code, errOut)
	}
	if !strings.Contains(out, "zone=vendor") {
		t.Fatalf("600 self-reported successes bought a cheaper rung:\n%s", out)
	}
	if !strings.Contains(out, "0 of 3 model(s) MEASURED") {
		t.Errorf("the summary claims a measurement:\n%s", out)
	}
	if !strings.Contains(out, "600 attempt(s) refused") {
		t.Errorf("the refused evidence is invisible — the operator cannot tell the file was read:\n%s", out)
	}
}

func TestTheReportSeparatesTooLittleEvidenceFromOutrightFailure(t *testing.T) {
	// Two local-ish models that both stayed off the device rung for OPPOSITE reasons:
	// one needs more runs, the other needs to stop being tried. Collapsing these into a
	// single "not eligible" is the failure this test exists to prevent.
	r := routePlaceRoster()
	path := writeEvidence(t, `{
	  "evidence": {
	    "rung-device": [{"class": "routine", "attempts": 4, "successes": 4, "verify": "witness"}],
	    "rung-fleet": [{"class": "routine", "attempts": 90, "successes": 30, "verify": "judge"}]
	  }
	}`)
	code, out, errOut := routePlaceRunOpts(t, &r, map[string]string{"work_class": "routine"},
		placeOptions{EvidencePath: path})
	if code != 0 {
		t.Fatalf("exit = %d, stderr = %q", code, errOut)
	}
	if !strings.Contains(out, "rung-device") || !strings.Contains(out, modelroute.ReasonInsufficientSamples) {
		t.Errorf("4 attempts were not reported as insufficient samples:\n%s", out)
	}
	if !strings.Contains(out, modelroute.ReasonBelowSuccessFloor) {
		t.Errorf("a 33%% success rate was not reported as a failure to clear the floor:\n%s", out)
	}
	if strings.Contains(out, "zone=device") || strings.Contains(out, "zone=fleet") {
		t.Errorf("an ungraded model served a cheap rung anyway:\n%s", out)
	}
}

func TestAnAssertionAndAMeasurementMayNotBothClaimOneModel(t *testing.T) {
	r := routePlaceRoster()
	path := writeEvidence(t, `{
	  "evidence": {"rung-device": [{"class": "routine", "attempts": 60, "successes": 57, "verify": "witness"}]}
	}`)
	code, out, errOut := routePlaceRunOpts(t, &r, map[string]string{"work_class": "routine"},
		placeOptions{CapSpec: "rung-device=t0", EvidencePath: path})
	if code != 2 {
		t.Fatalf("exit = %d, want 2 — two sources claiming one model's grade is a config error, stdout=%q", code, out)
	}
	if !strings.Contains(errOut, "rung-device") {
		t.Errorf("the refusal does not name the conflicting model: %q", errOut)
	}
	if out != "" {
		t.Errorf("a refused configuration still printed a placement: %q", out)
	}
	// The two flags remain composable when they talk about DIFFERENT models.
	code, out, errOut = routePlaceRunOpts(t, &r, map[string]string{"work_class": "normal-impl"},
		placeOptions{CapSpec: "rung-fleet=t1", EvidencePath: path})
	if code != 0 {
		t.Fatalf("exit = %d, stderr = %q", code, errOut)
	}
	if !strings.Contains(out, "zone=fleet") {
		t.Errorf("an asserted fleet grade and a measured device grade did not compose:\n%s", out)
	}
}

func TestAMistypedEvidenceFieldIsRefusedNotSilentlyReadAsZero(t *testing.T) {
	// "success" instead of "successes" would otherwise parse as a perfect record of zero
	// successes: a model that earned its grade would silently lose it, and the report
	// would blame the model.
	r := routePlaceRoster()
	path := writeEvidence(t, `{
	  "evidence": {"rung-device": [{"class": "routine", "attempts": 60, "success": 57, "verify": "witness"}]}
	}`)
	code, out, errOut := routePlaceRunOpts(t, &r, map[string]string{"work_class": "routine"},
		placeOptions{EvidencePath: path})
	if code != 2 {
		t.Fatalf("exit = %d, want 2 for an unknown field; stdout=%q", code, out)
	}
	if !strings.Contains(errOut, "success") {
		t.Errorf("the error does not name the offending field: %q", errOut)
	}
}

func TestAMissingOrEmptyEvidenceFileIsAUsageErrorNotAnEmptyGrading(t *testing.T) {
	r := routePlaceRoster()
	code, _, errOut := routePlaceRunOpts(t, &r, map[string]string{"work_class": "routine"},
		placeOptions{EvidencePath: filepath.Join(t.TempDir(), "nope.json")})
	if code != 2 {
		t.Fatalf("a missing --evidence file: exit = %d, want 2", code)
	}
	if !strings.Contains(errOut, "--evidence") {
		t.Errorf("the error does not name the flag: %q", errOut)
	}
	code, _, errOut = routePlaceRunOpts(t, &r, map[string]string{"work_class": "routine"},
		placeOptions{EvidencePath: writeEvidence(t, `{"evidence": {}}`)})
	if code != 2 {
		t.Fatalf("an empty --evidence file: exit = %d, want 2 — grading nothing is a typo, not a policy", code)
	}
	if !strings.Contains(errOut, "no evidence") {
		t.Errorf("unhelpful error for an empty file: %q", errOut)
	}
}

func TestAnOmittedFloorIsTheDefaultBarAndNotAZeroBar(t *testing.T) {
	// The absent-vs-zero trap: if the omitted floor decoded as all-zeros, ONE verified
	// attempt would grade a model, and every one of the honesty rules above would be
	// bypassed by simply not writing a "floor" key.
	ev, floor, err := loadPlacementEvidence(writeEvidence(t, `{
	  "evidence": {"rung-device": [{"class": "routine", "attempts": 1, "successes": 1, "verify": "witness"}]}
	}`))
	if err != nil {
		t.Fatal(err)
	}
	if floor != modelroute.DefaultGradeFloor() {
		t.Fatalf("floor = %+v, want the default %+v", floor, modelroute.DefaultGradeFloor())
	}
	g := modelroute.GradeCapability("rung-device", ev["rung-device"], floor)
	if g.Measured {
		t.Errorf("a single attempt graded under an omitted floor: %+v", g)
	}
	// An operator who writes a floor gets exactly the one they wrote, including a strict one.
	_, strict, err := loadPlacementEvidence(writeEvidence(t, `{
	  "floor": {"min_attempts": 200, "min_success_rate": 0.95, "require_witness": true},
	  "evidence": {"rung-device": [{"class": "routine", "attempts": 1, "successes": 1, "verify": "witness"}]}
	}`))
	if err != nil {
		t.Fatal(err)
	}
	if want := (modelroute.GradeFloor{MinAttempts: 200, MinSuccessRate: 0.95, RequireWitness: true}); strict != want {
		t.Errorf("floor = %+v, want %+v", strict, want)
	}
	// A nonsense bar is refused rather than applied.
	if _, _, err := loadPlacementEvidence(writeEvidence(t, `{
	  "floor": {"min_attempts": 20, "min_success_rate": 80},
	  "evidence": {"rung-device": [{"class": "routine", "attempts": 60, "successes": 57, "verify": "witness"}]}
	}`)); err == nil {
		t.Error("a success rate of 80 (meaning 8000%) was accepted as a fraction")
	}
}

func TestEvidenceForAModelTheRosterDoesNotBindIsReportedAsBuyingNothing(t *testing.T) {
	// A stale or misspelled model id is the likeliest reason a carefully assembled
	// evidence file changes nothing at all. Silence here reads as "the ladder ignored my
	// measurements"; the note says which ones landed nowhere.
	r := routePlaceRoster()
	path := writeEvidence(t, `{
	  "evidence": {
	    "rung-devise": [{"class": "routine", "attempts": 60, "successes": 60, "verify": "witness"}],
	    "rung-vendor-oldname": [{"class": "routine", "attempts": 60, "successes": 60, "verify": "witness"}]
	  }
	}`)
	code, out, errOut := routePlaceRunOpts(t, &r, map[string]string{"work_class": "routine"},
		placeOptions{EvidencePath: path})
	if code != 0 {
		t.Fatalf("exit = %d, stderr = %q", code, errOut)
	}
	if !strings.Contains(out, "rung-devise") {
		t.Errorf("the typo'd model id is not reported as ungraded:\n%s", out)
	}
	// The compatibility-only spelling is not a second piece of hardware either.
	if !strings.Contains(out, "rung-vendor-oldname") {
		t.Errorf("evidence against a legacy spelling was swallowed:\n%s", out)
	}
	if !strings.Contains(out, "zone=vendor") {
		t.Errorf("evidence for models that are not in the pool moved the placement anyway:\n%s", out)
	}
}

func TestModelsTheFileNeverMentionsAreCountedNotListed(t *testing.T) {
	// On a real roster the unmentioned models outnumber the graded ones several to one.
	// They are still reported — as a count — because "18 bound, 1 measured" is the number
	// that tells an operator how much of their fleet is running on nobody's word.
	r := routePlaceRoster()
	path := writeEvidence(t, `{
	  "evidence": {"rung-device": [{"class": "routine", "attempts": 60, "successes": 57, "verify": "witness"}]}
	}`)
	code, out, errOut := routePlaceRunOpts(t, &r, map[string]string{"work_class": "routine"},
		placeOptions{EvidencePath: path})
	if code != 0 {
		t.Fatalf("exit = %d, stderr = %q", code, errOut)
	}
	if !strings.Contains(out, "2 model(s) the evidence file says nothing about") {
		t.Errorf("silent models were not accounted for:\n%s", out)
	}
	if strings.Contains(out, "rung-vendor  ") {
		t.Errorf("a model with no evidence was listed as a finding:\n%s", out)
	}
	// A model whose evidence was REFUSED is a different thing and keeps its own line.
	path = writeEvidence(t, `{
	  "evidence": {"rung-vendor": [{"class": "routine", "attempts": 90, "successes": 90, "verify": ""}]}
	}`)
	if _, out, _ = routePlaceRunOpts(t, &r, map[string]string{"work_class": "routine"},
		placeOptions{EvidencePath: path}); !strings.Contains(out, "rung-vendor") ||
		!strings.Contains(out, "90 attempt(s) refused") {
		t.Errorf("refused evidence was collapsed into the silent count:\n%s", out)
	}
}

func TestTheJSONReportCarriesTheGradesAndTheirProvenance(t *testing.T) {
	r := routePlaceRoster()
	path := writeEvidence(t, `{
	  "evidence": {
	    "rung-device": [{"class": "routine", "attempts": 60, "successes": 57, "verify": "witness"}],
	    "rung-fleet": [{"class": "normal-impl", "attempts": 40, "successes": 20, "verify": "judge"}]
	  }
	}`)
	code, out, errOut := routePlaceRunOpts(t, &r, map[string]string{"work_class": "routine"},
		placeOptions{EvidencePath: path, JSON: true})
	if code != 0 {
		t.Fatalf("exit = %d, stderr = %q", code, errOut)
	}
	var got placementReport
	if err := json.Unmarshal([]byte(out), &got); err != nil {
		t.Fatalf("--json is not valid JSON: %v\n%s", err, out)
	}
	if len(got.Grades) != 3 {
		t.Fatalf("grades = %d, want one per bound model: %+v", len(got.Grades), got.Grades)
	}
	byModel := map[string]modelroute.Grade{}
	for _, g := range got.Grades {
		byModel[g.Model] = g
	}
	if g := byModel["rung-device"]; !g.Measured || g.Capability != modelroute.TierT2 || g.Verify != modelroute.VerifyWitness {
		t.Errorf("the graded model's row is wrong: %+v", g)
	}
	if g := byModel["rung-fleet"]; g.Measured || g.Reason != modelroute.ReasonBelowSuccessFloor {
		t.Errorf("a 50%% record was not reported as below the floor: %+v", g)
	}
	if got.MeasuredCount != 1 || got.Placement.Zone != modelroute.ZoneDevice || !got.Placement.Measured {
		t.Errorf("the placement half of the report disagrees with the grades: %+v", got.Placement)
	}
}
