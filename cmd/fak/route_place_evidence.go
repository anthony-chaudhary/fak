package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// `fak route --place --evidence FILE` — grading the ladder from OBSERVED outcomes instead
// of an operator's word (epic #5416, track F).
//
// --capability is an operator ASSERTING what a model can do. It is honest because it is
// attributable: someone typed it and can be held to it. But it does not scale past a
// handful of models, and it cannot notice when a model gets worse.
//
// This is the other input: a file of what each model was OBSERVED to do, with the
// provenance of each observation, folded by modelroute.GradeCapability into the same
// Candidate the ladder already consumes. The grading rules live in that pure leaf; what
// lives here is the file shape, the refusal to let two sources claim the same model, and
// the rendering of what the evidence bought — including the evidence it did NOT buy,
// which is the number an operator needs when nothing moved.
//
//	fak route --accounts FILE --place --evidence outcomes.json --labels work_class=routine
//
//	{
//	  "floor": {"min_attempts": 20, "min_success_rate": 0.8, "require_witness": false},
//	  "evidence": {
//	    "zone-device": [
//	      {"class": "routine", "attempts": 60, "successes": 57, "verify": "witness"}
//	    ]
//	  }
//	}

// placementEvidenceFile is the on-disk shape of --evidence.
//
// Floor is a POINTER so an omitted floor is distinguishable from one an operator wrote as
// all-zeros. That distinction is the difference between "use the stated default bar" and
// "zero attempts at a zero success rate is enough to grade", which is the same thing as
// having no bar at all.
type placementEvidenceFile struct {
	Floor    *modelroute.GradeFloor                `json:"floor,omitempty"`
	Evidence map[string][]modelroute.ClassEvidence `json:"evidence"`
}

// loadPlacementEvidence reads and validates an --evidence file.
//
// Unknown fields are REFUSED rather than ignored. This file is hand-written, and the
// failure mode of a silently-ignored field is the worst one available here: a misspelled
// "successes" reads as zero successes, which quietly denies a grade a model earned, and
// nothing in the output would say why.
func loadPlacementEvidence(path string) (map[string][]modelroute.ClassEvidence, modelroute.GradeFloor, error) {
	floor := modelroute.DefaultGradeFloor()
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, floor, fmt.Errorf("--evidence %s: %w", path, err)
	}
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	var f placementEvidenceFile
	if err := dec.Decode(&f); err != nil {
		return nil, floor, fmt.Errorf("--evidence %s: %w", path, err)
	}
	if len(f.Evidence) == 0 {
		return nil, floor, fmt.Errorf("--evidence %s: declares no evidence; a file that grades nothing is a typo, not a policy", path)
	}
	if f.Floor != nil {
		floor = *f.Floor
	}
	if floor.MinSuccessRate < 0 || floor.MinSuccessRate > 1 {
		return nil, floor, fmt.Errorf("--evidence %s: min_success_rate %v is not a fraction in [0,1]", path, floor.MinSuccessRate)
	}
	if floor.MinAttempts < 0 {
		return nil, floor, fmt.Errorf("--evidence %s: min_attempts %d is negative", path, floor.MinAttempts)
	}
	return f.Evidence, floor, nil
}

// gradedPool folds evidence into the candidate pool built from the roster's bindings.
//
// A hand declaration and a body of evidence are both legitimate, but not for the SAME
// model: two sources claiming one model's capability is a configuration error, and
// silently preferring either one hides it. The conflict is refused by the caller before
// this runs.
//
// A grade that did not reach the bar leaves its candidate exactly as it was — ungraded,
// and therefore top-rung only. It does NOT downgrade a hand declaration, because failing
// to measure something is not evidence against it.
func gradedPool(base []modelroute.Candidate, evidence map[string][]modelroute.ClassEvidence, floor modelroute.GradeFloor) ([]modelroute.Candidate, []modelroute.Grade, []string) {
	ids := make([]string, 0, len(base))
	bound := map[string]bool{}
	for _, c := range base {
		ids = append(ids, c.Model)
		bound[c.Model] = true
	}
	grades := modelroute.GradeCandidates(ids, evidence, floor)
	measured := map[string]modelroute.Candidate{}
	for _, g := range grades {
		if g.Measured {
			measured[g.Model] = g.Candidate()
		}
	}
	out := make([]modelroute.Candidate, 0, len(base))
	for _, c := range base {
		if m, ok := measured[c.Model]; ok {
			out = append(out, m)
			continue
		}
		out = append(out, c)
	}
	// Evidence for a model the roster does not bind bought nothing. Say so: it is almost
	// always a stale id, and an operator staring at an unmoved placement needs to see it.
	var ignored []string
	for model := range evidence {
		if !bound[model] {
			ignored = append(ignored, model)
		}
	}
	sort.Strings(ignored)
	return out, grades, ignored
}

// conflictingDeclarations returns the models claimed by BOTH --capability and --evidence,
// in a deterministic order.
func conflictingDeclarations(declared map[string]modelroute.WorkTier, evidence map[string][]modelroute.ClassEvidence) []string {
	var out []string
	for model := range declared {
		if _, ok := evidence[model]; ok {
			out = append(out, model)
		}
	}
	sort.Strings(out)
	return out
}

// printGrades renders what the evidence bought, and what it did not.
//
// The unmeasured rows are the point of this block. When a placement did not descend, the
// operator's next question is always "why not, I gave you the numbers" — and the answer is
// one of three closed reasons: the samples were refused as unverifiable, there were not
// enough of them, or the model missed the success floor. Those call for opposite responses,
// so they are never collapsed into one line.
func printGrades(w io.Writer, grades []modelroute.Grade, floor modelroute.GradeFloor, ignored []string) {
	measured := 0
	for _, g := range grades {
		if g.Measured {
			measured++
		}
	}
	witness := ""
	if floor.RequireWitness {
		witness = ", git-witnessed only"
	}
	fmt.Fprintf(w, "  evidence     %d of %d model(s) MEASURED  (floor: %d attempts, %.0f%% verified success%s)\n",
		measured, len(grades), floor.MinAttempts, floor.MinSuccessRate*100, witness)
	silent := 0
	for _, g := range grades {
		// A model the file says nothing about is not a finding. On a real roster those
		// rows outnumber the graded ones several to one, and printing them all buries the
		// three lines the operator came for.
		if noEvidenceAtAll(g) {
			silent++
			continue
		}
		fmt.Fprintf(w, "    %-24s %s\n", g.Model, gradeCell(g))
	}
	if silent > 0 {
		fmt.Fprintf(w, "    %-24s %d model(s) the evidence file says nothing about; they stay top-rung only\n", "(no evidence)", silent)
	}
	if len(ignored) > 0 {
		fmt.Fprintf(w, "  note         evidence for %d model(s) the roster does not bind, so it graded nothing: %s\n",
			len(ignored), strings.Join(ignored, " "))
	}
}

// noEvidenceAtAll reports whether the file simply never mentioned this model, as opposed
// to mentioning it and having the mention refused. The two look identical in the placement
// and are opposite problems: one is a gap in measurement, the other is a gap in TRUST, and
// only the second is worth a line of an operator's attention.
func noEvidenceAtAll(g modelroute.Grade) bool {
	return !g.Measured && g.Dropped == 0 && g.Reason == modelroute.ReasonNoTrustedEvidence
}

// gradeCell renders one model's grade: the tier it earned, or the reason it earned none.
func gradeCell(g modelroute.Grade) string {
	if g.Measured {
		s := fmt.Sprintf("%-3s %s  %d/%d %s", g.Capability, g.Class, g.Successes, g.Attempts, g.Verify.Label())
		return strings.TrimSpace(s)
	}
	s := "UNMEASURED  " + g.Reason
	if g.Dropped > 0 {
		s += fmt.Sprintf(" (%d attempt(s) refused: not independently checked, or tagged with an unknown class)", g.Dropped)
	}
	return s
}
