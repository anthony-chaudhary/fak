package livecodebench

import (
	"math"
	"testing"
)

func approx(a, b float64) bool { return math.Abs(a-b) < 1e-9 }

// fixture problems for the windowing tests: two in-window, one before the window,
// and one undated. Scenario is set so the values read like a real suite, though the
// windowing functions key only on question_id + contest_date.
func windowFixture() []Problem {
	return []Problem{
		{QuestionID: "in-a", Scenario: ScenarioCodeGeneration, ContestDate: "2025-03-01", Prompt: "p"},
		{QuestionID: "in-b", Scenario: ScenarioCodeGeneration, ContestDate: "2025-05-01", Prompt: "p"},
		{QuestionID: "out-c", Scenario: ScenarioCodeGeneration, ContestDate: "2024-12-01", Prompt: "p"},
		{QuestionID: "no-d", Scenario: ScenarioCodeGeneration, ContestDate: "", Prompt: "p"},
	}
}

// TestWindowedPassAtKExcludesOutOfWindow is the core #2094 witness: a problem whose
// contest_date sits outside the window — even one that PASSED — must not move the
// windowed rate, while the full-set rate still counts it.
func TestWindowedPassAtKExcludesOutOfWindow(t *testing.T) {
	problems := windowFixture()
	tallies := []SampleTally{
		{QuestionID: "in-a", Samples: 2, Correct: 2},  // pass@1 = 1.0, in window
		{QuestionID: "in-b", Samples: 2, Correct: 0},  // pass@1 = 0.0, in window
		{QuestionID: "out-c", Samples: 2, Correct: 2}, // pass@1 = 1.0, OUT of window
		{QuestionID: "no-d", Samples: 2, Correct: 2},  // pass@1 = 1.0, undated -> excluded from window
	}
	// end empty -> defaults to the injected now, never the wall clock.
	w, err := NewDateWindow("2025-01-01", "", "2025-12-31")
	if err != nil {
		t.Fatalf("NewDateWindow: %v", err)
	}
	got, err := WindowedPassAtK(problems, tallies, 1, w)
	if err != nil {
		t.Fatalf("WindowedPassAtK: %v", err)
	}

	if got.TotalProblems != 4 || got.WindowedProblems != 2 {
		t.Errorf("counts = (total %d, windowed %d), want (4, 2)", got.TotalProblems, got.WindowedProblems)
	}
	// full = mean(1,0,1,1) = 0.75 ; windowed = mean(1,0) = 0.5.
	// If out-c (pass 1.0) leaked into the window the windowed rate would be 0.667.
	if !approx(got.FullPassRate, 0.75) {
		t.Errorf("FullPassRate = %v, want 0.75", got.FullPassRate)
	}
	if !approx(got.WindowedPassRate, 0.5) {
		t.Errorf("WindowedPassRate = %v, want 0.5 (out-of-window pass must be excluded)", got.WindowedPassRate)
	}
	if got.StartDate != "2025-01-01" || got.EndDate != "2025-12-31" {
		t.Errorf("bounds = [%s,%s], want [2025-01-01,2025-12-31]", got.StartDate, got.EndDate)
	}
	if got.K != 1 {
		t.Errorf("K = %d, want 1", got.K)
	}

	// Determinism: same inputs (now injected) -> byte-identical scores on re-run.
	again, err := WindowedPassAtK(problems, tallies, 1, w)
	if err != nil {
		t.Fatalf("WindowedPassAtK re-run: %v", err)
	}
	if again != got {
		t.Errorf("non-deterministic result: %+v vs %+v", again, got)
	}
}

// TestReportCarriesWindowScore pins that a Report can carry the windowed + full
// rates and bounds (#2094 acceptance: "Report carries both ... + the window bounds").
func TestReportCarriesWindowScore(t *testing.T) {
	w, err := NewDateWindow("2025-01-01", "2025-06-30", "")
	if err != nil {
		t.Fatalf("NewDateWindow: %v", err)
	}
	scores, err := WindowedPassAtK(windowFixture(), []SampleTally{
		{QuestionID: "in-a", Samples: 1, Correct: 1},
	}, 1, w)
	if err != nil {
		t.Fatalf("WindowedPassAtK: %v", err)
	}
	r := Report{Schema: ReportSchema, WindowScore: &scores}
	if r.WindowScore == nil || r.WindowScore.EndDate != "2025-06-30" {
		t.Fatalf("Report did not carry the window score: %+v", r.WindowScore)
	}
}

func TestNewDateWindowErrors(t *testing.T) {
	cases := []struct {
		name, start, end, now string
	}{
		{"bad_start", "2025-13-01", "2025-12-31", ""},
		{"bad_end", "2025-01-01", "2025-99-01", ""},
		{"bad_now", "2025-01-01", "", "not-a-date"},
		{"empty_end_no_now", "2025-01-01", "", ""},
		{"end_before_start", "2025-06-01", "2025-01-01", ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := NewDateWindow(tc.start, tc.end, tc.now); err == nil {
				t.Errorf("NewDateWindow(%q,%q,%q) = nil error, want error", tc.start, tc.end, tc.now)
			}
		})
	}
}

func TestWindowedPassAtKJoinErrors(t *testing.T) {
	w, _ := NewDateWindow("2025-01-01", "2025-12-31", "")

	// A tally with no matching problem is an error (inputs must describe one run).
	if _, err := WindowedPassAtK(windowFixture(), []SampleTally{{QuestionID: "ghost", Samples: 1, Correct: 1}}, 1, w); err == nil {
		t.Error("unknown tally question_id: got nil error, want error")
	}

	// A duplicate question_id in problems is an ambiguous join -> error.
	dup := append(windowFixture(), Problem{QuestionID: "in-a", Scenario: ScenarioCodeGeneration, ContestDate: "2025-04-01", Prompt: "p"})
	if _, err := WindowedPassAtK(dup, []SampleTally{{QuestionID: "in-a", Samples: 1, Correct: 1}}, 1, w); err == nil {
		t.Error("duplicate question_id: got nil error, want error")
	}
}

// TestEmptyWindowIsHonestZero pins that a window catching no problem scores 0 over 0
// windowed problems rather than erroring or borrowing the full-set rate.
func TestEmptyWindowIsHonestZero(t *testing.T) {
	w, _ := NewDateWindow("2030-01-01", "2030-12-31", "")
	got, err := WindowedPassAtK(windowFixture(), []SampleTally{
		{QuestionID: "in-a", Samples: 2, Correct: 2},
		{QuestionID: "in-b", Samples: 2, Correct: 2},
	}, 1, w)
	if err != nil {
		t.Fatalf("WindowedPassAtK: %v", err)
	}
	if got.WindowedProblems != 0 || !approx(got.WindowedPassRate, 0) {
		t.Errorf("empty window = (n %d, rate %v), want (0, 0)", got.WindowedProblems, got.WindowedPassRate)
	}
	if !approx(got.FullPassRate, 1.0) {
		t.Errorf("FullPassRate = %v, want 1.0 (full set still scored)", got.FullPassRate)
	}
}
