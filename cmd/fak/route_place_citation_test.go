package main

import (
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// Witnesses for the AUDIT PATH a journal-backed placement prints (#5428, epic #5416
// track F).
//
// `--outcomes` already grades from a record nobody hand-wrote. What it could not do was
// answer the next question — which turns? — and a capability an operator cannot re-walk is
// a number they have to take on trust from the same pipeline that produced it. These tests
// pin the citation to the grade beside it, in the two directions it could lie: naming
// turns the grader refused, and naming turns for a grade it never awarded.

func TestThePlacementCitesTheTurnsThatEarnedIt(t *testing.T) {
	r := routePlaceRoster()
	path := writeJournal(t, routineTurns("rung-device", 40, 2, time.Hour))
	code, out, errOut := routePlaceRunOpts(t, &r, map[string]string{"work_class": "routine"},
		placeOptions{OutcomesPath: path, Since: "30d"})
	if code != 0 {
		t.Fatalf("exit = %d, stderr = %q", code, errOut)
	}
	if !strings.Contains(out, "audit") {
		t.Fatalf("a journal-backed placement printed no audit path back to its turns:\n%s", out)
	}
	// The count beside the citation is the grade's own attempt count, so the two cannot
	// drift apart on the page an operator actually reads.
	if !strings.Contains(out, "rung-device routine 38/40 from 40 turn(s)") {
		t.Errorf("the citation does not account for the graded attempts:\n%s", out)
	}
	if !strings.Contains(out, "rung-device-0") {
		t.Errorf("the citation names no turn id from the journal:\n%s", out)
	}
	// A grade can rest on hundreds of turns; the terminal shows a sample and SAYS how many
	// it left out rather than quietly truncating.
	if !strings.Contains(out, "(+36 more)") {
		t.Errorf("the abbreviated citation does not say what it left out:\n%s", out)
	}
}

func TestTheJSONTrailsCarryEveryTurnTheTerminalAbbreviated(t *testing.T) {
	r := routePlaceRoster()
	path := writeJournal(t, routineTurns("rung-device", 40, 2, time.Hour))
	code, out, errOut := routePlaceRunOpts(t, &r, map[string]string{"work_class": "routine"},
		placeOptions{OutcomesPath: path, Since: "30d", JSON: true})
	if code != 0 {
		t.Fatalf("exit = %d, stderr = %q", code, errOut)
	}
	var got struct {
		Journal struct {
			Trails []modelroute.EvidenceTrail `json:"trails"`
		} `json:"journal"`
	}
	if err := json.Unmarshal([]byte(out), &got); err != nil {
		t.Fatalf("--json is not valid JSON: %v\n%s", err, out)
	}
	if len(got.Journal.Trails) != 1 {
		t.Fatalf("want one (model,class,verify) trail, got %+v", got.Journal.Trails)
	}
	tr := got.Journal.Trails[0]
	if tr.Model != "rung-device" || tr.Class != modelroute.ClassRoutine || tr.Verify != modelroute.VerifyWitness {
		t.Errorf("trail lost its key: %+v", tr)
	}
	if len(tr.Turns) != 40 {
		t.Errorf("the archived trail names %d turn(s); the terminal abbreviates, the record must not", len(tr.Turns))
	}
}

// TestAnUnmeasuredModelIsNotCited. The refusal that matters: an unmeasured grade names no
// class, so there is no set of turns that earned it. Printing its turns anyway would read
// as evidence for a capability the grader explicitly declined to award — the per-model
// reason is the honest answer to "why not", and it is already printed.
func TestAnUnmeasuredModelIsNotCited(t *testing.T) {
	r := routePlaceRoster()
	path := writeJournal(t, routineTurns("rung-device", 5, 0, time.Hour)) // under the 20 floor
	code, out, errOut := routePlaceRunOpts(t, &r, map[string]string{"work_class": "routine"},
		placeOptions{OutcomesPath: path, Since: "30d"})
	if code != 0 {
		t.Fatalf("exit = %d, stderr = %q", code, errOut)
	}
	if !strings.Contains(out, modelroute.ReasonInsufficientSamples) {
		t.Fatalf("5 turns bought a grade, or the reason went missing:\n%s", out)
	}
	if strings.Contains(out, "from 5 turn(s)") {
		t.Errorf("an unmeasured model was cited as if its turns had earned something:\n%s", out)
	}
}

// TestTheCitationNeverNamesTurnsTheFloorRefused runs the attack end to end through the
// command: a model with a mountain of self-reported turns and a bare quorum of witnessed
// ones. The grade already drops the self-reported block; the citation printed beside it
// must drop it too, or the audit over-claims in exactly the direction the grader refused.
func TestTheCitationNeverNamesTurnsTheFloorRefused(t *testing.T) {
	r := routePlaceRoster()
	turns := routineTurns("rung-device", 25, 0, time.Hour)
	for i := 0; i < 60; i++ {
		turns = append(turns, modelroute.TurnOutcome{
			ID: "selfreport-" + string(rune('a'+i%26)) + string(rune('a'+i/26)),
			Model: "rung-device", Class: modelroute.ClassRoutine,
			Success: true, Verify: modelroute.VerifyNone, At: time.Now().Add(-time.Hour),
		})
	}
	path := writeJournal(t, turns)
	code, out, errOut := routePlaceRunOpts(t, &r, map[string]string{"work_class": "routine"},
		placeOptions{OutcomesPath: path, Since: "30d"})
	if code != 0 {
		t.Fatalf("exit = %d, stderr = %q", code, errOut)
	}
	if !strings.Contains(out, "from 25 turn(s)") {
		t.Errorf("the citation counted more than the grade did:\n%s", out)
	}
	if strings.Contains(out, "selfreport-") {
		t.Errorf("the citation named a self-reported turn the grade refused to count:\n%s", out)
	}
}
