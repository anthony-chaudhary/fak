package main

import (
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// `fak route --place --outcomes FILE` — grading from the raw turn journal (epic #5416,
// tracks D and F).
//
// --evidence takes a summary somebody assembled. This takes the append-only journal
// itself, one JSON object per turn, and does the counting here — which is the difference
// between a number an operator produced and a number they can be shown. Everything the
// fold refuses (a replayed row, a row that predates the window, a row nobody stamped) is
// counted and printed, because the whole point of grading from a record is that a thin
// grade can be explained without opening the file.
//
//	fak route --accounts FILE --place --outcomes turns.jsonl --since 30d \
//	  --grade-floor attempts=50,rate=0.9,witness

// journalSummary is what a journal-backed grading actually consumed.
type journalSummary struct {
	Path    string                  `json:"path"`
	Since   string                  `json:"since,omitempty"`
	Floor   modelroute.GradeFloor   `json:"floor"`
	Journal modelroute.JournalStats `json:"journal"`
	Fold    modelroute.FoldStats    `json:"fold"`
	Models  int                     `json:"models_with_evidence"`
	// Trails is the audit path from each evidence row back to the turns that produced it.
	// It rides the JSON in full — deliberately, however long it gets — because this is the
	// only form of the record that can be diffed, archived, or handed to someone who was
	// not at the terminal. The human rendering below is the one that abbreviates.
	Trails []modelroute.EvidenceTrail `json:"trails,omitempty"`
}

// placementGradeInputs resolves the two mutually exclusive ways of supplying evidence into
// the single pair the grader needs.
//
// They are mutually exclusive on purpose. Both flags answer "what has this model been
// observed to do", and a run that took some of its answer from a journal and some from a
// hand-written summary produces a grade whose provenance nobody can state afterwards.
func placementGradeInputs(opts placeOptions) (map[string][]modelroute.ClassEvidence, modelroute.GradeFloor, *journalSummary, error) {
	floor := modelroute.DefaultGradeFloor()
	switch {
	case opts.EvidencePath != "" && opts.OutcomesPath != "":
		return nil, floor, nil, fmt.Errorf("--evidence and --outcomes both supply the evidence to grade from; " +
			"pick one, or a grade's provenance cannot be stated afterwards")
	case opts.EvidencePath != "":
		if opts.FloorSpec != "" {
			return nil, floor, nil, fmt.Errorf("--grade-floor applies to --outcomes; an --evidence file carries its own \"floor\" key")
		}
		if opts.Since != "" {
			return nil, floor, nil, fmt.Errorf("--since applies to --outcomes; an --evidence file is already a summary, and this command cannot tell which turns it summarised")
		}
		ev, floor, err := loadPlacementEvidence(opts.EvidencePath)
		return ev, floor, nil, err
	case opts.OutcomesPath == "":
		return nil, floor, nil, nil
	}

	if opts.FloorSpec != "" {
		var err error
		if floor, err = parseGradeFloor(opts.FloorSpec); err != nil {
			return nil, floor, nil, err
		}
	}
	var since time.Time
	if opts.Since != "" {
		d, err := parseSinceWindow(opts.Since)
		if err != nil {
			return nil, floor, nil, err
		}
		since = time.Now().Add(-d)
	}

	f, err := os.Open(opts.OutcomesPath)
	if err != nil {
		return nil, floor, nil, fmt.Errorf("--outcomes: %w", err)
	}
	defer f.Close()
	outcomes, jstats, err := modelroute.ReadTurnOutcomes(f)
	if err != nil {
		return nil, floor, nil, fmt.Errorf("--outcomes %s: %w", opts.OutcomesPath, err)
	}
	if jstats.Lines == 0 {
		return nil, floor, nil, fmt.Errorf("--outcomes %s: the journal is empty; grading nothing is a typo, not a policy", opts.OutcomesPath)
	}
	evidence, fold, trails := modelroute.FoldTurnOutcomesTrail(outcomes, modelroute.FoldOptions{Since: since})
	return evidence, floor, &journalSummary{
		Path: opts.OutcomesPath, Since: opts.Since, Floor: floor,
		Journal: jstats, Fold: fold, Models: len(evidence), Trails: trails,
	}, nil
}

// parseGradeFloor reads `attempts=N,rate=F,witness` — the operator's evidentiary bar.
//
// An unrecognised key is an ERROR rather than an ignored token: the failure mode of
// ignoring one is a run that grades against a bar nobody set and reports no sign of it,
// which is the exact shape of the mistake this whole track exists to prevent.
func parseGradeFloor(spec string) (modelroute.GradeFloor, error) {
	floor := modelroute.DefaultGradeFloor()
	for _, part := range strings.Split(spec, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		if part == "witness" {
			floor.RequireWitness = true
			continue
		}
		key, value, ok := strings.Cut(part, "=")
		if !ok {
			return floor, fmt.Errorf("--grade-floor %q: want attempts=N, rate=0..1, or the bare token witness", part)
		}
		key, value = strings.TrimSpace(key), strings.TrimSpace(value)
		switch key {
		case "attempts":
			n, err := strconv.ParseInt(value, 10, 64)
			if err != nil || n < 0 {
				return floor, fmt.Errorf("--grade-floor attempts=%q: want a non-negative whole number", value)
			}
			floor.MinAttempts = n
		case "rate":
			f, err := strconv.ParseFloat(value, 64)
			if err != nil || f < 0 || f > 1 {
				return floor, fmt.Errorf("--grade-floor rate=%q: want a fraction in [0,1] (0.9, not 90)", value)
			}
			floor.MinSuccessRate = f
		default:
			return floor, fmt.Errorf("--grade-floor %q: unknown key; want attempts=N, rate=0..1, or witness", key)
		}
	}
	return floor, nil
}

// parseSinceWindow accepts a Go duration ("720h") or a whole number of days ("30d"), which
// is how operators actually talk about an evidence window.
func parseSinceWindow(spec string) (time.Duration, error) {
	spec = strings.TrimSpace(spec)
	if days, ok := strings.CutSuffix(spec, "d"); ok {
		n, err := strconv.ParseFloat(days, 64)
		if err != nil || n <= 0 {
			return 0, fmt.Errorf("--since %q: want a positive number of days (30d) or a Go duration (720h)", spec)
		}
		return time.Duration(n * 24 * float64(time.Hour)), nil
	}
	d, err := time.ParseDuration(spec)
	if err != nil || d <= 0 {
		return 0, fmt.Errorf("--since %q: want a positive number of days (30d) or a Go duration (720h)", spec)
	}
	return d, nil
}

// printJournalSummary reports what the journal contributed and what it lost on the way.
//
// The losses are the interesting half. A grade that came out thin has exactly four
// explanations available here — the rows were replayed, they predate the window, nobody
// stamped them, or nobody attributed them to a model — and each has a different fix.
func printJournalSummary(w io.Writer, s journalSummary) {
	fmt.Fprintf(w, "  journal      %s: %d line(s), %d counted, %d model(s) with evidence\n",
		s.Path, s.Journal.Lines, s.Fold.Counted, s.Models)
	var lost []string
	if s.Journal.Malformed > 0 {
		lost = append(lost, fmt.Sprintf("%d unparseable line(s)", s.Journal.Malformed))
	}
	if s.Fold.Duplicates > 0 {
		lost = append(lost, fmt.Sprintf("%d replayed id(s)", s.Fold.Duplicates))
	}
	if s.Fold.OutsideWindow > 0 {
		lost = append(lost, fmt.Sprintf("%d older than %s", s.Fold.OutsideWindow, s.Since))
	}
	if s.Fold.Undated > 0 {
		lost = append(lost, fmt.Sprintf("%d undated (cannot be shown to be inside %s)", s.Fold.Undated, s.Since))
	}
	if s.Fold.Unattributed > 0 {
		lost = append(lost, fmt.Sprintf("%d naming no model", s.Fold.Unattributed))
	}
	if len(lost) > 0 {
		fmt.Fprintf(w, "  not counted  %s\n", strings.Join(lost, ", "))
	}
	if s.Fold.Undeduplicable > 0 {
		fmt.Fprintf(w, "  caution      %d counted turn(s) carry no id, so this corpus cannot be checked for replay\n",
			s.Fold.Undeduplicable)
	}
}

// citationSample bounds how many turn ids one grade prints. A grade can rest on hundreds
// of turns and an operator reading a placement is not reading all of them here — the full
// list is in the --json trails, which is the form an audit actually uses.
const citationSample = 4

// printGradeCitations renders the audit path from each MEASURED grade back to the turns
// that earned it.
//
// This is the half of "grade from a durable record" that makes the record worth keeping.
// A grade printed alone is a number an operator has to take on trust from the same
// pipeline that produced it; a grade printed with the turns behind it can be re-walked
// against the journal, and a producer that double-counted, mis-attributed, or graded a
// model on another model's slots stops being invisible.
//
// Only measured grades are cited, because an unmeasured one names no class and therefore
// has no set of turns that earned it — printing its turns anyway would read as evidence
// for a capability that was explicitly declined. The per-model refusal reason printed by
// printGrades is the answer to that question, and it is already there.
//
// Nothing is silently dropped: an abbreviated list says how many it left out, and turns
// that were counted while carrying no id are reported as unnameable rather than omitted.
func printGradeCitations(w io.Writer, grades []modelroute.Grade, trails []modelroute.EvidenceTrail, floor modelroute.GradeFloor) {
	if len(trails) == 0 {
		return
	}
	header := false
	for _, g := range grades {
		turns, anonymous := modelroute.TurnsBehind(g, trails, floor)
		if len(turns) == 0 && anonymous == 0 {
			continue
		}
		if !header {
			fmt.Fprintf(w, "  audit        which turns earned each measured grade (full list in --json trails)\n")
			header = true
		}
		shown := turns
		elided := 0
		if len(shown) > citationSample {
			shown, elided = shown[:citationSample], len(turns)-citationSample
		}
		line := strings.Join(shown, " ")
		if elided > 0 {
			line = fmt.Sprintf("%s (+%d more)", line, elided)
		}
		if anonymous > 0 {
			line = fmt.Sprintf("%s [%d counted turn(s) carry no id and cannot be named]", line, anonymous)
		}
		fmt.Fprintf(w, "    %s %s %d/%d from %d turn(s): %s\n",
			g.Model, g.Class, g.Successes, g.Attempts, len(turns)+anonymous, line)
	}
}
