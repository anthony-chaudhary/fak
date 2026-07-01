package attemptbudget

import "testing"

func TestDecide_RepeatedAttemptsMoveDispatchableToHeld(t *testing.T) {
	// The #1777 witness: "a fixture with repeated attempts moves from
	// dispatchable to held."
	base := Input{IssueID: "42", Budget: 3}

	under := base
	under.Attempts = []Attempt{
		{FailureClass: "test_failure", AtUnix: 100},
		{FailureClass: "test_failure", AtUnix: 200},
	}
	got := Decide(under)
	if got.Status != StatusDispatchable {
		t.Fatalf("under budget: want dispatchable, got %q", got.Status)
	}

	atBudget := base
	atBudget.Attempts = []Attempt{
		{FailureClass: "test_failure", AtUnix: 100},
		{FailureClass: "test_failure", AtUnix: 200},
		{FailureClass: "timeout", AtUnix: 300},
	}
	got = Decide(atBudget)
	if got.Status != StatusHeld {
		t.Fatalf("at budget: want held, got %q", got.Status)
	}
	if got.LastFailureClass != "timeout" {
		t.Fatalf("want last failure class %q, got %q", "timeout", got.LastFailureClass)
	}
	if got.AttemptCount != 3 {
		t.Fatalf("want attempt count 3, got %d", got.AttemptCount)
	}
}

func TestDecide_ZeroOrNegativeBudgetIsUnlimited(t *testing.T) {
	for _, budget := range []int{0, -1} {
		in := Input{
			IssueID: "1",
			Budget:  budget,
			Attempts: []Attempt{
				{FailureClass: "x", AtUnix: 1},
				{FailureClass: "x", AtUnix: 2},
				{FailureClass: "x", AtUnix: 3},
				{FailureClass: "x", AtUnix: 4},
			},
		}
		if got := Decide(in); got.Status != StatusDispatchable {
			t.Fatalf("budget=%d: want dispatchable (unlimited), got %q", budget, got.Status)
		}
	}
}

func TestDecide_NoAttempts_DispatchableWithNoFailureClass(t *testing.T) {
	got := Decide(Input{IssueID: "7", Budget: 2})
	if got.Status != StatusDispatchable {
		t.Fatalf("want dispatchable, got %q", got.Status)
	}
	if got.LastFailureClass != "" {
		t.Fatalf("want no failure class, got %q", got.LastFailureClass)
	}
	if got.AttemptCount != 0 {
		t.Fatalf("want attempt count 0, got %d", got.AttemptCount)
	}
}

func TestDecideAll_CountsDispatchableAndHeld(t *testing.T) {
	rep := DecideAll([]Input{
		{IssueID: "1", Budget: 2, Attempts: []Attempt{{FailureClass: "a", AtUnix: 1}}},
		{IssueID: "2", Budget: 2, Attempts: []Attempt{
			{FailureClass: "a", AtUnix: 1},
			{FailureClass: "b", AtUnix: 2},
		}},
		{IssueID: "3", Budget: 0},
	})
	if len(rep.Decisions) != 3 {
		t.Fatalf("want 3 decisions, got %d", len(rep.Decisions))
	}
	if rep.HeldCount != 1 {
		t.Fatalf("want 1 held, got %d", rep.HeldCount)
	}
	if rep.DispatchableCount != 2 {
		t.Fatalf("want 2 dispatchable, got %d", rep.DispatchableCount)
	}
}
