package quality

import (
	"strings"
	"testing"
)

// calGroundTruth is the hermetic ground-truth support corpus: one strongly
// supported figure, one weak-evidence forecast, one unsupported speculation.
const calGroundTruth = `[
 {"claim":"Throughput increased 12% week over week","support":"strong"},
 {"claim":"Latency may improve next quarter","support":"weak"},
 {"claim":"Churn is trending down","support":"none"}
]`

const (
	// calFaithfulClaims states calibrated confidence throughout: high on the
	// strong claim, hedged (low) on the weak and unsupported ones.
	calFaithfulClaims = `[
 {"claim":"Throughput increased 12% week over week","confidence":"high"},
 {"claim":"Latency may improve next quarter","confidence":"low"},
 {"claim":"Churn is trending down","confidence":"low"}
]`
	// calOverconfidentClaim is the injected defect: an unsupported claim
	// asserted at high confidence.
	calOverconfidentClaim = "Churn is trending down"
	// calOverconfidentClaims flips only the unsupported claim to high.
	calOverconfidentClaims = `[
 {"claim":"Throughput increased 12% week over week","confidence":"high"},
 {"claim":"Latency may improve next quarter","confidence":"low"},
 {"claim":"Churn is trending down","confidence":"high"}
]`
	// calUnhedgedWeakClaims leaves the weak-evidence forecast unhedged
	// (medium), the second failure class.
	calUnhedgedWeakClaims = `[
 {"claim":"Throughput increased 12% week over week","confidence":"high"},
 {"claim":"Latency may improve next quarter","confidence":"medium"},
 {"claim":"Churn is trending down","confidence":"low"}
]`
)

// calCase builds a valid case whose Reference.Text carries the ground-truth
// support flags the calibration oracle judges stated confidence against.
func calCase(minScore float64) QualityCase {
	return QualityCase{
		Schema:    CaseSchema,
		ID:        "confidence-calibration-exec-report",
		Version:   1,
		Prompt:    "State each rollup claim with a confidence calibrated to its evidence.",
		Params:    SamplingParams{Temperature: 0, MaxTokens: 64},
		Reference: Trace{Text: calGroundTruth},
		Oracles:   []string{"confidence-calibration"},
		Rubric:    RubricSpec{MinScore: minScore},
	}
}

// TestConfidenceCalibrationRegistered proves the oracle registered under its
// stable name and kind, so cases can reference it by name.
func TestConfidenceCalibrationRegistered(t *testing.T) {
	os, err := Lookup([]string{"confidence-calibration"})
	if err != nil {
		t.Fatalf("Lookup(confidence-calibration): %v", err)
	}
	if got := os[0].Kind(); got != "rubric" {
		t.Errorf("Kind() = %q, want rubric", got)
	}
}

// TestConfidenceCalibrationFaithfulPasses is the happy path: every stated
// confidence sits at or under its evidence ceiling, so the oracle passes at
// score 1.0.
func TestConfidenceCalibrationFaithfulPasses(t *testing.T) {
	c := calCase(1)
	v := ConfidenceCalibration{}.Judge(Trace{Text: calGroundTruth}, Trace{Text: calFaithfulClaims}, c)
	if !v.Pass {
		t.Fatalf("faithful calibrated report must pass; got %+v", v)
	}
	if v.Score != 1 {
		t.Errorf("score = %v, want 1.0", v.Score)
	}
}

// TestConfidenceCalibrationOverconfidentUnsupportedFails is the defect
// witness: an unsupported claim asserted at high confidence fails the oracle,
// and Detail names that exact claim and the unsupported class.
func TestConfidenceCalibrationOverconfidentUnsupportedFails(t *testing.T) {
	c := calCase(1)
	v := ConfidenceCalibration{}.Judge(Trace{Text: calGroundTruth}, Trace{Text: calOverconfidentClaims}, c)
	if v.Pass {
		t.Fatalf("overconfident unsupported claim must not pass; got %+v", v)
	}
	if want := 2.0 / 3.0; v.Score != want {
		t.Errorf("score = %v, want %v (2 of 3 claims calibrated)", v.Score, want)
	}
	if !strings.Contains(v.Detail, calOverconfidentClaim) {
		t.Errorf("Detail must name the overconfident claim %q; got %q", calOverconfidentClaim, v.Detail)
	}
	if !strings.Contains(v.Detail, "unsupported") {
		t.Errorf("Detail must name the unsupported class; got %q", v.Detail)
	}
}

// TestConfidenceCalibrationUnhedgedWeakFails is the second failure class: a
// weak-evidence claim stated above the hedged floor fails, Detail says so.
func TestConfidenceCalibrationUnhedgedWeakFails(t *testing.T) {
	c := calCase(1)
	v := ConfidenceCalibration{}.Judge(Trace{Text: calGroundTruth}, Trace{Text: calUnhedgedWeakClaims}, c)
	if v.Pass {
		t.Fatalf("unhedged weak-evidence claim must not pass; got %+v", v)
	}
	if !strings.Contains(v.Detail, "Latency may improve next quarter") {
		t.Errorf("Detail must name the unhedged claim; got %q", v.Detail)
	}
	if !strings.Contains(v.Detail, "unhedged") {
		t.Errorf("Detail must name the unhedged class; got %q", v.Detail)
	}
}

// TestConfidenceCalibrationSpineIntegration runs the overconfident report
// through the full spine: the failure bundle names confidence-calibration as
// the failing oracle and carries the offending claim in its detail.
func TestConfidenceCalibrationSpineIntegration(t *testing.T) {
	c := calCase(1)
	eng := ScriptedRunner{Label: "engine-overconfident", Trace: Trace{Text: calOverconfidentClaims}}
	res, err := RunCase(c, ReferenceRunner{}, eng, oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if res.Pass {
		t.Fatalf("overconfident report must not pass; got %s", Explain(res))
	}
	fb := res.FailureBundle
	if fb == nil {
		t.Fatal("failing run must carry a failure bundle")
	}
	if fb.FailingOracle != "confidence-calibration" {
		t.Errorf("failing oracle = %q, want confidence-calibration", fb.FailingOracle)
	}
	if !strings.Contains(fb.Detail, calOverconfidentClaim) {
		t.Errorf("bundle detail must name the overconfident claim; got %q", fb.Detail)
	}
}

// TestConfidenceCalibrationMinScoreTolerance proves the threshold gate: the
// same overconfident report (2/3 calibrated) passes when the case tolerates
// MinScore 0.5, and the tolerated miscalibration is still named.
func TestConfidenceCalibrationMinScoreTolerance(t *testing.T) {
	c := calCase(0.5)
	v := ConfidenceCalibration{}.Judge(Trace{Text: calGroundTruth}, Trace{Text: calOverconfidentClaims}, c)
	if !v.Pass {
		t.Fatalf("2/3 calibrated must pass at MinScore 0.5; got %+v", v)
	}
	if !strings.Contains(v.Detail, calOverconfidentClaim) {
		t.Errorf("tolerated-miscalibration detail should still name the claim; got %q", v.Detail)
	}
}

// TestConfidenceCalibrationUnknownClaimFailsClosed defines the missing-record
// edge: a claim with no ground-truth row is treated as unsupported, so stating
// it confidently fails while a hedged mention passes.
func TestConfidenceCalibrationUnknownClaimFailsClosed(t *testing.T) {
	c := calCase(1)
	confident := `[{"claim":"Revenue doubled overnight","confidence":"high"}]`
	v := ConfidenceCalibration{}.Judge(Trace{Text: calGroundTruth}, Trace{Text: confident}, c)
	if v.Pass {
		t.Fatalf("confident claim with no ground-truth record must fail closed; got %+v", v)
	}
	if !strings.Contains(v.Detail, "Revenue doubled overnight") {
		t.Errorf("Detail must name the unrecorded claim; got %q", v.Detail)
	}
	hedged := `[{"claim":"Revenue doubled overnight","confidence":"low"}]`
	v = ConfidenceCalibration{}.Judge(Trace{Text: calGroundTruth}, Trace{Text: hedged}, c)
	if !v.Pass {
		t.Errorf("hedged claim with no ground-truth record is calibrated speculation; got %+v", v)
	}
}

// TestConfidenceCalibrationEdges defines the remaining documented edges: a
// report asserting no claims passes at score 1; unparseable claim or support
// payloads fail closed at score 0; an unknown confidence token miscalibrates
// its claim.
func TestConfidenceCalibrationEdges(t *testing.T) {
	c := calCase(1)
	for _, text := range []string{"", "   \n  ", "[]"} {
		v := ConfidenceCalibration{}.Judge(Trace{Text: calGroundTruth}, Trace{Text: text}, c)
		if !v.Pass || v.Score != 1 {
			t.Errorf("empty report %q: got %+v, want pass at score 1", text, v)
		}
	}
	v := ConfidenceCalibration{}.Judge(Trace{Text: calGroundTruth}, Trace{Text: "not json"}, c)
	if v.Pass || v.Score != 0 {
		t.Errorf("unparseable claim payload must fail closed at score 0; got %+v", v)
	}
	v = ConfidenceCalibration{}.Judge(Trace{Text: "not json"}, Trace{Text: calFaithfulClaims}, c)
	if v.Pass || v.Score != 0 {
		t.Errorf("unparseable support payload must fail closed at score 0; got %+v", v)
	}
	badSupport := `[{"claim":"Throughput increased 12% week over week","support":"vibes"}]`
	v = ConfidenceCalibration{}.Judge(Trace{Text: badSupport}, Trace{Text: calFaithfulClaims}, c)
	if v.Pass || !strings.Contains(v.Detail, "vibes") {
		t.Errorf("unknown support flag must fail closed naming the flag; got %+v", v)
	}
	badConf := `[{"claim":"Throughput increased 12% week over week","confidence":"certain"}]`
	v = ConfidenceCalibration{}.Judge(Trace{Text: calGroundTruth}, Trace{Text: badConf}, c)
	if v.Pass || !strings.Contains(v.Detail, "certain") {
		t.Errorf("unknown confidence token must miscalibrate its claim; got %+v", v)
	}
}
