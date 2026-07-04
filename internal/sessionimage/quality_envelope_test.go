package sessionimage

import (
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/session"
)

// TestQualityEnvelopeSurvivesDumpRestore is the #1964 witness (QA-dogfood spine
// #1961/QD-004): a session stamped at start with a QA quality envelope — budget axes,
// witness policy, dogfood probes, and control-pane scorecard membership — must expose
// that SAME envelope unchanged after the image is dumped and restored. The envelope
// rides the drive record (session.json), so this exercises the existing dump/restore
// path with no new image part: DumpDir writes it, LoadDir verifies session.json's
// integrity, and Image.Drive carries the envelope back byte-for-byte.
func TestQualityEnvelopeSurvivesDumpRestore(t *testing.T) {
	dir := t.TempDir()

	env := session.QualityEnvelope{
		Budget:         session.BudgetEnvelope{Budget: session.Budget{TurnsLeft: 20, TokensLeft: 200000}},
		WitnessPolicy:  "proof-by-default",
		DogfoodProbes:  []string{"quality-score", "milestone-score"},
		ScorecardCards: []string{"code_quality", "milestone_scorecard"},
	}
	drive := session.DefaultState("sess-qenv")
	drive.QualityEnvelope = env

	if _, err := DumpDir(dir, Input{SessionID: "sess-qenv", Drive: drive, Now: 1}); err != nil {
		t.Fatalf("DumpDir: %v", err)
	}

	img, err := LoadDir(dir)
	if err != nil {
		t.Fatalf("LoadDir: %v", err)
	}
	got := img.Drive.QualityEnvelope
	if got.IsZero() {
		t.Fatal("restored drive carries no quality envelope; want the stamped one")
	}
	if !reflect.DeepEqual(got, env) {
		t.Fatalf("quality envelope changed across dump/restore:\n got=%+v\n want=%+v", got, env)
	}
	if got.WitnessPolicy != "proof-by-default" {
		t.Fatalf("witness policy lost: %q", got.WitnessPolicy)
	}
	if !reflect.DeepEqual(got.ScorecardCards, []string{"code_quality", "milestone_scorecard"}) {
		t.Fatalf("scorecard membership lost: %v", got.ScorecardCards)
	}
}

// TestQualityEnvelopeAbsentStaysZero anchors the pre-#1964 wire compatibility: a session
// stamped with NO envelope restores as Zero (no QA controls declared), never a phantom
// populated one — the omitzero contract the drive record relies on.
func TestQualityEnvelopeAbsentStaysZero(t *testing.T) {
	dir := t.TempDir()
	if _, err := DumpDir(dir, Input{SessionID: "sess-noqenv", Drive: session.DefaultState("sess-noqenv"), Now: 1}); err != nil {
		t.Fatalf("DumpDir: %v", err)
	}
	img, err := LoadDir(dir)
	if err != nil {
		t.Fatalf("LoadDir: %v", err)
	}
	if !img.Drive.QualityEnvelope.IsZero() {
		t.Fatalf("absent envelope restored non-zero: %+v", img.Drive.QualityEnvelope)
	}
}
