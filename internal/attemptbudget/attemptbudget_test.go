package attemptbudget

import (
	"encoding/json"
	"testing"
)

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

// --- #2860: structured block reason + routing over the attempt history ---

func TestClassifyBlock_SameErrorRepeatedRoutesKnownBad(t *testing.T) {
	// A stable failure signature: every attempt hit the same wall. The issue is
	// genuinely stuck, so it belongs in the known-bad ledger, not the queue.
	reason, route := ClassifyBlock([]Attempt{
		{FailureClass: "merge_conflict", AtUnix: 100},
		{FailureClass: "merge_conflict", AtUnix: 200},
	})
	if reason != BlockReasonSameErrorRepeated {
		t.Fatalf("want %q, got %q", BlockReasonSameErrorRepeated, reason)
	}
	if route != RouteKnownBad {
		t.Fatalf("want route %q, got %q", RouteKnownBad, route)
	}
}

func TestClassifyBlock_NoisyRawStringsStillOneSignature(t *testing.T) {
	// The confusion risk the issue names: a signature that split "test_failure"
	// from "assertion failed" would read a genuinely stuck issue as flaky.
	// Classifying onto the closed FailureClass vocabulary is what keeps the
	// signature stable across the caller's descriptive strings.
	reason, route := ClassifyBlock([]Attempt{
		{FailureClass: "test_failure", AtUnix: 100},
		{FailureClass: "assertion failed in TestFoo", AtUnix: 200},
		{FailureClass: "TEST timeout: assert", AtUnix: 300},
	})
	if reason != BlockReasonSameErrorRepeated {
		t.Fatalf("noisy-but-same-class history: want %q, got %q", BlockReasonSameErrorRepeated, reason)
	}
	if route != RouteKnownBad {
		t.Fatalf("want route %q, got %q", RouteKnownBad, route)
	}
}

func TestClassifyBlock_DistinctErrorsRouteRetry(t *testing.T) {
	// Different walls each time -> no signature to record -> flaky, so retry.
	reason, route := ClassifyBlock([]Attempt{
		{FailureClass: "test_failure", AtUnix: 100},
		{FailureClass: "merge_conflict", AtUnix: 200},
	})
	if reason != BlockReasonDistinctErrors {
		t.Fatalf("want %q, got %q", BlockReasonDistinctErrors, reason)
	}
	if route != RouteRetry {
		t.Fatalf("want route %q, got %q", RouteRetry, route)
	}
}

func TestClassifyBlock_PreconditionUnmetRoutesEscalate(t *testing.T) {
	// Auth and ambiguous-scope are walls a retry of the issue can never clear.
	for _, raw := range []string{"auth_error", "permission denied", "ambiguous scope"} {
		reason, route := ClassifyBlock([]Attempt{
			{FailureClass: "test_failure", AtUnix: 100},
			{FailureClass: raw, AtUnix: 200},
		})
		if reason != BlockReasonPreconditionUnmet {
			t.Fatalf("%q: want %q, got %q", raw, BlockReasonPreconditionUnmet, reason)
		}
		if route != RouteEscalate {
			t.Fatalf("%q: want route %q, got %q", raw, RouteEscalate, route)
		}
	}
}

func TestClassifyBlock_PreconditionBeatsRepetition(t *testing.T) {
	// Three identical auth failures are a precondition problem, not a known-bad
	// code signature: escalate to a human, do not poison the fleet ledger.
	reason, route := ClassifyBlock([]Attempt{
		{FailureClass: "auth_error", AtUnix: 100},
		{FailureClass: "auth_error", AtUnix: 200},
		{FailureClass: "auth_error", AtUnix: 300},
	})
	if reason != BlockReasonPreconditionUnmet {
		t.Fatalf("want %q (precondition first), got %q", BlockReasonPreconditionUnmet, reason)
	}
	if route != RouteEscalate {
		t.Fatalf("want route %q, got %q", RouteEscalate, route)
	}
}

func TestClassifyBlock_SingleAttemptNeverPromotesKnownBad(t *testing.T) {
	// Budget=1 blocks on the first failure. One sample is not repetition, so the
	// classifier must fail safe: retry, never a fleet-wide known-bad signature.
	reason, route := ClassifyBlock([]Attempt{{FailureClass: "test_failure", AtUnix: 100}})
	if reason != BlockReasonDistinctErrors {
		t.Fatalf("single attempt: want %q, got %q", BlockReasonDistinctErrors, reason)
	}
	if route == RouteKnownBad {
		t.Fatalf("single attempt must never route to %q", RouteKnownBad)
	}
}

func TestClassifyBlock_EmptyHistoryExplainsNothing(t *testing.T) {
	reason, route := ClassifyBlock(nil)
	if reason != "" || route != "" {
		t.Fatalf("empty history: want empty reason/route, got %q/%q", reason, route)
	}
}

func TestDecide_BlockReasonStampedOnlyOnHeld(t *testing.T) {
	// Dispatchable and cooling-down issues carry no block reason: there is no
	// block to explain. The held one carries both reason and route.
	under := Decide(Input{IssueID: "1", Budget: 5, Attempts: []Attempt{
		{FailureClass: "test_failure", AtUnix: 100},
		{FailureClass: "test_failure", AtUnix: 200},
	}})
	if under.Status != StatusDispatchable {
		t.Fatalf("setup: want dispatchable, got %q", under.Status)
	}
	if under.BlockReason != "" || under.Route != "" {
		t.Fatalf("dispatchable must carry no block reason, got %q/%q", under.BlockReason, under.Route)
	}

	cooling := Decide(Input{IssueID: "2", Budget: 5, NowUnix: 201, Attempts: []Attempt{
		{FailureClass: "test_failure", AtUnix: 200},
	}})
	if cooling.Status != StatusCoolingDown {
		t.Fatalf("setup: want cooling_down, got %q", cooling.Status)
	}
	if cooling.BlockReason != "" || cooling.Route != "" {
		t.Fatalf("cooling_down must carry no block reason, got %q/%q", cooling.BlockReason, cooling.Route)
	}

	held := Decide(Input{IssueID: "3", Budget: 2, Attempts: []Attempt{
		{FailureClass: "test_failure", AtUnix: 100},
		{FailureClass: "test_failure", AtUnix: 200},
	}})
	if held.Status != StatusHeld {
		t.Fatalf("setup: want held, got %q", held.Status)
	}
	if held.BlockReason != BlockReasonSameErrorRepeated {
		t.Fatalf("held: want %q, got %q", BlockReasonSameErrorRepeated, held.BlockReason)
	}
	if held.Route != RouteKnownBad {
		t.Fatalf("held: want route %q, got %q", RouteKnownBad, held.Route)
	}
	// The repeated class a known-bad signature keys on is already on the
	// Decision -- no duplicate field.
	if held.BackoffClass != FailureClassTest {
		t.Fatalf("held: want backoff class %q, got %q", FailureClassTest, held.BackoffClass)
	}
}

func TestDecide_BlockReasonIsQueryableJSON(t *testing.T) {
	// "First-class, queryable" means it survives the wire the CLI already emits
	// (`fak dispatch attempt-budget --json`), not just the Go struct.
	held := Decide(Input{IssueID: "9", Budget: 2, Attempts: []Attempt{
		{FailureClass: "auth_error", AtUnix: 100},
		{FailureClass: "auth_error", AtUnix: 200},
	}})
	b, err := json.Marshal(held)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var round map[string]any
	if err := json.Unmarshal(b, &round); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if round["block_reason"] != string(BlockReasonPreconditionUnmet) {
		t.Fatalf("want block_reason %q on the wire, got %v", BlockReasonPreconditionUnmet, round["block_reason"])
	}
	if round["route"] != string(RouteEscalate) {
		t.Fatalf("want route %q on the wire, got %v", RouteEscalate, round["route"])
	}

	// A dispatchable decision omits both keys entirely (omitempty), so a reader
	// cannot mistake "not blocked" for "blocked for no reason". Decode into a
	// FRESH map: json.Unmarshal merges into a non-nil map rather than clearing
	// it, so reusing `round` here would still show the held decision's keys.
	ok := Decide(Input{IssueID: "10", Budget: 5})
	b, err = json.Marshal(ok)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	clear := map[string]any{}
	if err := json.Unmarshal(b, &clear); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if _, present := clear["block_reason"]; present {
		t.Fatalf("dispatchable decision must omit block_reason, got %s", b)
	}
	if _, present := clear["route"]; present {
		t.Fatalf("dispatchable decision must omit route, got %s", b)
	}
}
