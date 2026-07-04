package livecodebench

import (
	"fmt"
	"time"
)

// dateLayout is the YYYY-MM-DD form every LiveCodeBench date uses (contest_date on
// a Problem, start_date/end_date on a Report). Suite.Validate already pins problem
// contest_date to this layout.
const dateLayout = "2006-01-02"

// DateWindow is a contamination-free contest-date window [Start, End]. Scoring a run
// over only the problems whose contest_date falls in the window lets a result state
// a window that post-dates a model's training cut, instead of claiming credit for
// problems the model may have memorized — the LiveCodeBench compute_scores.py
// contamination control.
//
// End defaults to a caller-supplied "now" when the window leaves it open (the
// upstream convention: the window runs up to the present). That now is INJECTED,
// never read from the wall clock, so a windowed score is byte-for-byte reproducible
// in a test or a re-run — the determinism requirement fak carries everywhere.
type DateWindow struct {
	Start time.Time // inclusive lower bound; the zero time means "open" (no lower bound)
	End   time.Time // inclusive upper bound
}

// NewDateWindow parses YYYY-MM-DD start/end/now into a window. An empty start is an
// open lower bound (time zero). An empty end defaults to now, so now is REQUIRED
// (and must parse) whenever end is empty. It errors on an unparseable date or a
// window whose end precedes its start.
func NewDateWindow(startDate, endDate, now string) (DateWindow, error) {
	var w DateWindow
	var err error
	if startDate != "" {
		if w.Start, err = time.Parse(dateLayout, startDate); err != nil {
			return DateWindow{}, fmt.Errorf("livecodebench date window: start_date %q is not YYYY-MM-DD: %w", startDate, err)
		}
	}
	switch {
	case endDate != "":
		if w.End, err = time.Parse(dateLayout, endDate); err != nil {
			return DateWindow{}, fmt.Errorf("livecodebench date window: end_date %q is not YYYY-MM-DD: %w", endDate, err)
		}
	case now != "":
		if w.End, err = time.Parse(dateLayout, now); err != nil {
			return DateWindow{}, fmt.Errorf("livecodebench date window: injected now %q is not YYYY-MM-DD: %w", now, err)
		}
	default:
		return DateWindow{}, fmt.Errorf("livecodebench date window: end_date is empty and no now was injected (now must be supplied, never read from the wall clock)")
	}
	if w.End.Before(w.Start) {
		return DateWindow{}, fmt.Errorf("livecodebench date window: end %s precedes start %s", w.End.Format(dateLayout), w.Start.Format(dateLayout))
	}
	return w, nil
}

// Contains reports whether a problem's contest_date falls within [Start, End]
// inclusive. A problem with no contest_date — or one that does not parse — is in NO
// window: an undated problem cannot be certified contamination-free, so it is
// excluded from the windowed set (and only the windowed set; it still counts in the
// full-set rate). The Start zero value is an open lower bound.
func (w DateWindow) Contains(p Problem) bool {
	if p.ContestDate == "" {
		return false
	}
	d, err := time.Parse(dateLayout, p.ContestDate)
	if err != nil {
		return false
	}
	if !w.Start.IsZero() && d.Before(w.Start) {
		return false
	}
	return !d.After(w.End)
}

// WindowScores is the reportable contamination-window result: pass@k over the full
// problem set AND over the windowed subset, the window bounds, and the two problem
// counts. WindowedPassRate is scored over strictly the in-window problems, so a
// problem outside the window never moves it.
type WindowScores struct {
	StartDate        string  `json:"start_date,omitempty"`
	EndDate          string  `json:"end_date"`
	K                int     `json:"k"`
	TotalProblems    int     `json:"total_problems"`
	WindowedProblems int     `json:"windowed_problems"`
	FullPassRate     float64 `json:"full_pass_rate"`
	WindowedPassRate float64 `json:"windowed_pass_rate"`
}

// WindowedPassAtK joins each per-problem tally to its problem by question_id, then
// scores pass@k twice: over all problems (the full-set rate) and over only those in
// the window (the contamination-free rate). It errors when a tally names a
// question_id absent from problems (the two inputs must describe the same run),
// when problems carries a duplicate question_id (an ambiguous join), or when pass@k
// is invalid. When no problem falls in the window the windowed rate is 0 over 0
// problems — an honest empty window, not an error.
func WindowedPassAtK(problems []Problem, tallies []SampleTally, k int, w DateWindow) (WindowScores, error) {
	byID := make(map[string]Problem, len(problems))
	for _, p := range problems {
		if _, dup := byID[p.QuestionID]; dup {
			return WindowScores{}, fmt.Errorf("livecodebench date window: duplicate question_id %q in problems (ambiguous join)", p.QuestionID)
		}
		byID[p.QuestionID] = p
	}
	windowed := make([]SampleTally, 0, len(tallies))
	for _, t := range tallies {
		p, ok := byID[t.QuestionID]
		if !ok {
			return WindowScores{}, fmt.Errorf("livecodebench date window: tally question_id %q has no matching problem", t.QuestionID)
		}
		if w.Contains(p) {
			windowed = append(windowed, t)
		}
	}
	full, err := MeanPassAtK(tallies, k)
	if err != nil {
		return WindowScores{}, err
	}
	windowedRate := 0.0
	if len(windowed) > 0 {
		if windowedRate, err = MeanPassAtK(windowed, k); err != nil {
			return WindowScores{}, err
		}
	}
	scores := WindowScores{
		EndDate:          w.End.Format(dateLayout),
		K:                k,
		TotalProblems:    len(tallies),
		WindowedProblems: len(windowed),
		FullPassRate:     full,
		WindowedPassRate: windowedRate,
	}
	if !w.Start.IsZero() {
		scores.StartDate = w.Start.Format(dateLayout)
	}
	return scores, nil
}
