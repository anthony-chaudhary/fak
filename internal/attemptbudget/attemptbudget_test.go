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

// TestBackoff_DistinctWindowsByFailureClass is the #1778 witness: "a policy
// fixture shows different backoff windows by failure class." Four issues,
// identical in every way except the failure class of their one attempt, must
// each carry a DIFFERENT BackoffSeconds under the default policy, and the
// ordering must match the documented cooling-off rationale (auth and
// ambiguous-scope need a human so they cool down longest; merge is moderate;
// test is shortest).
func TestBackoff_DistinctWindowsByFailureClass(t *testing.T) {
	fixture := []struct {
		issueID      string
		failureClass string
		wantClass    FailureClass
	}{
		{"auth-1", "auth_error", FailureClassAuth},
		{"merge-1", "merge_conflict", FailureClassMerge},
		{"test-1", "test_failure", FailureClassTest},
		{"scope-1", "ambiguous_scope", FailureClassAmbiguousScope},
	}

	seen := map[FailureClass]int64{}
	for _, tc := range fixture {
		d := Decide(Input{
			IssueID:  tc.issueID,
			Attempts: []Attempt{{FailureClass: tc.failureClass, AtUnix: 1000}},
		})
		if d.BackoffClass != tc.wantClass {
			t.Fatalf("%s: want classified as %q, got %q", tc.issueID, tc.wantClass, d.BackoffClass)
		}
		if d.BackoffSeconds <= 0 {
			t.Fatalf("%s: want a positive backoff window, got %d", tc.issueID, d.BackoffSeconds)
		}
		if d.CooldownUntilUnix != 1000+d.BackoffSeconds {
			t.Fatalf("%s: want cooldown_until_unix = at_unix + backoff, got %d (backoff %d)",
				tc.issueID, d.CooldownUntilUnix, d.BackoffSeconds)
		}
		seen[tc.wantClass] = d.BackoffSeconds
	}

	// All four classes must carry genuinely distinct windows -- the whole
	// point of the policy is that they do NOT all cool down the same way.
	if len(seen) != len(fixture) {
		t.Fatalf("want %d distinct failure classes recorded, got %d: %+v", len(fixture), len(seen), seen)
	}
	windows := map[int64]bool{}
	for class, secs := range seen {
		if windows[secs] {
			t.Fatalf("want every failure class to carry a distinct backoff window, but %q shares %ds with another class: %+v", class, secs, seen)
		}
		windows[secs] = true
	}

	// The documented ordering: auth and ambiguous-scope (need a human) cool
	// down longer than merge, which cools down longer than test (cheapest,
	// most likely transient, shortest window).
	if !(seen[FailureClassAuth] > seen[FailureClassMerge] &&
		seen[FailureClassAmbiguousScope] > seen[FailureClassMerge] &&
		seen[FailureClassMerge] > seen[FailureClassTest]) {
		t.Fatalf("want auth/ambiguous_scope > merge > test, got %+v", seen)
	}
}

// TestClassify_RateLimitShortWindow proves the rate-limit/overload class: a
// throttled attempt (429/529/overload/rate-limit prose) carries the SHORTEST
// window of the whole policy — before this class existed it fell to
// FailureClassOther's 1h, holding an overload-throttled issue ~6x longer than
// a flaky test even though the capacity window reopens on its own.
func TestClassify_RateLimitShortWindow(t *testing.T) {
	throttled := []string{
		"rate_limit",
		"429 too many requests",
		"upstream 529 overloaded",
		"API rate limit exceeded for installation",
		"quota exhausted for model",
	}
	for _, raw := range throttled {
		d := Decide(Input{IssueID: "rl", Attempts: []Attempt{{FailureClass: raw, AtUnix: 1000}}})
		if d.BackoffClass != FailureClassRateLimit {
			t.Fatalf("%q: want classified %q, got %q", raw, FailureClassRateLimit, d.BackoffClass)
		}
	}
	rl := DefaultBackoffSeconds[FailureClassRateLimit]
	if rl <= 0 {
		t.Fatalf("rate-limit window must be positive, got %d", rl)
	}
	for class, secs := range DefaultBackoffSeconds {
		if class != FailureClassRateLimit && secs <= rl {
			t.Fatalf("rate-limit must carry the shortest window; %q has %ds <= %ds", class, secs, rl)
		}
	}
}

// TestClassify_RateLimitBeatsAuthNeedle pins the ordering trap: GitHub's
// throttling prose mentions authentication ("... authenticated requests get a
// higher rate limit"), which substring-matches the auth needle. Rate-limit is
// classified FIRST, so a reopening capacity window is never cooled 4h as a
// needs-a-human auth failure.
func TestClassify_RateLimitBeatsAuthNeedle(t *testing.T) {
	raw := "API rate limit exceeded (authenticated requests get a higher rate limit)"
	d := Decide(Input{IssueID: "gh", Attempts: []Attempt{{FailureClass: raw, AtUnix: 1}}})
	if d.BackoffClass != FailureClassRateLimit {
		t.Fatalf("want %q, got %q — throttling prose must not be misread as auth", FailureClassRateLimit, d.BackoffClass)
	}
	// A genuine auth failure still classifies auth.
	d = Decide(Input{IssueID: "auth", Attempts: []Attempt{{FailureClass: "auth_error: permission denied", AtUnix: 1}}})
	if d.BackoffClass != FailureClassAuth {
		t.Fatalf("want %q for a genuine auth failure, got %q", FailureClassAuth, d.BackoffClass)
	}
}

// TestDecide_CoolingDownBeforeWindowElapses proves the new StatusCoolingDown
// verdict: under budget, but the last failure's class-specific window has not
// yet elapsed as of the caller-supplied NowUnix.
func TestDecide_CoolingDownBeforeWindowElapses(t *testing.T) {
	in := Input{
		IssueID:  "5",
		Budget:   10, // nowhere near budget-held
		Attempts: []Attempt{{FailureClass: "auth_error", AtUnix: 1000}},
	}
	authWindow := DefaultBackoffSeconds[FailureClassAuth]

	// Just before the window elapses: cooling down.
	in.NowUnix = 1000 + authWindow - 1
	got := Decide(in)
	if got.Status != StatusCoolingDown {
		t.Fatalf("just before window elapses: want cooling_down, got %q", got.Status)
	}

	// At/after the window: dispatchable again.
	in.NowUnix = 1000 + authWindow
	got = Decide(in)
	if got.Status != StatusDispatchable {
		t.Fatalf("at window elapsed: want dispatchable, got %q", got.Status)
	}

	// NowUnix omitted (0): no cooldown timing info, so it must not be
	// reported as actively cooling down even though the window is open.
	in.NowUnix = 0
	got = Decide(in)
	if got.Status != StatusDispatchable {
		t.Fatalf("no clock supplied: want dispatchable (no cooldown claim without a clock), got %q", got.Status)
	}
	if got.BackoffSeconds != authWindow {
		t.Fatalf("want the backoff window still reported without a clock, got %d", got.BackoffSeconds)
	}
}

// TestDecide_HeldOverridesCoolingDown proves Budget is a hard stop: even
// while still inside the class-specific cooldown window, crossing the
// attempt budget reports HELD, not cooling_down.
func TestDecide_HeldOverridesCoolingDown(t *testing.T) {
	got := Decide(Input{
		IssueID: "6",
		Budget:  1,
		NowUnix: 1000,
		Attempts: []Attempt{
			{FailureClass: "test_failure", AtUnix: 999},
		},
	})
	if got.Status != StatusHeld {
		t.Fatalf("want held (budget crossed) even though inside cooldown window, got %q", got.Status)
	}
}

// TestDecide_PerIssueBackoffOverride proves Input.Backoff overrides the
// default policy for a single issue without disturbing the package default.
func TestDecide_PerIssueBackoffOverride(t *testing.T) {
	got := Decide(Input{
		IssueID:  "7",
		Attempts: []Attempt{{FailureClass: "test_failure", AtUnix: 1000}},
		Backoff:  map[FailureClass]int64{FailureClassTest: 5},
	})
	if got.BackoffSeconds != 5 {
		t.Fatalf("want overridden backoff of 5s, got %d", got.BackoffSeconds)
	}
	if DefaultBackoffSeconds[FailureClassTest] == 5 {
		t.Fatalf("override must not mutate the package default")
	}
}

// TestClassify_UnrecognizedFailureClassFallsBackToOther proves an unknown
// raw failure-class string never crashes and never gets silently coerced
// into one of the named classes.
func TestClassify_UnrecognizedFailureClassFallsBackToOther(t *testing.T) {
	got := Decide(Input{
		IssueID:  "8",
		Attempts: []Attempt{{FailureClass: "some_totally_unknown_thing", AtUnix: 1}},
	})
	if got.BackoffClass != FailureClassOther {
		t.Fatalf("want unrecognized failure class to fall back to %q, got %q", FailureClassOther, got.BackoffClass)
	}
	if got.BackoffSeconds != DefaultBackoffSeconds[FailureClassOther] {
		t.Fatalf("want the FailureClassOther default window, got %d", got.BackoffSeconds)
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

func TestDecideAll_CountsCoolingDownSeparatelyFromHeldAndDispatchable(t *testing.T) {
	rep := DecideAll([]Input{
		// Under budget, inside its auth cooldown window as of NowUnix: cooling down.
		{IssueID: "1", Budget: 10, NowUnix: 1000, Attempts: []Attempt{{FailureClass: "auth_error", AtUnix: 999}}},
		// Under budget, its test cooldown window already elapsed: dispatchable.
		{IssueID: "2", Budget: 10, NowUnix: 100000, Attempts: []Attempt{{FailureClass: "test_failure", AtUnix: 1}}},
		// Over budget: held, regardless of cooldown timing.
		{IssueID: "3", Budget: 1, NowUnix: 1000, Attempts: []Attempt{{FailureClass: "auth_error", AtUnix: 999}}},
	})
	if rep.CoolingDownCount != 1 {
		t.Fatalf("want 1 cooling down, got %d (%+v)", rep.CoolingDownCount, rep.Decisions)
	}
	if rep.DispatchableCount != 1 {
		t.Fatalf("want 1 dispatchable, got %d (%+v)", rep.DispatchableCount, rep.Decisions)
	}
	if rep.HeldCount != 1 {
		t.Fatalf("want 1 held, got %d (%+v)", rep.HeldCount, rep.Decisions)
	}
}
