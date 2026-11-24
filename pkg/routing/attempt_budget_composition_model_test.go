package routing

// Bounded model check of the router attempt-budget COMPOSITION: the request's
// one total budget (AttemptBudget) composed with the per-class same-target
// budget (Failure.RetryBudget) across a chain walk whose per-rung eligibility
// changes from rung to rung.
//
// Why a bounded model check rather than more examples: the two budgets are
// independent axes and a walk's dial count is their composition, so the
// interesting failures live in the interaction - a class that authorizes the
// full physical headroom meeting a request budget that is already spent, a
// late response that must not spend a retry, a later rung with more eligible
// keys than an earlier one. Example tests pin the cases someone thought of;
// this enumerates the declared domain and compares every case against an
// oracle transcribed from the documented contract.
//
// The oracle is INDEPENDENT of the code under test in the two places that
// matter. The class table is transcribed from the doc comments in failure.go
// (DefaultAction / DefaultRetries / AdmissionRefused) rather than read from
// the implementation, and is cross-checked against the implementation in its
// own subtest so a drift in either shows up as a disagreement. The dial count
// is computed in closed form over the whole chain (min of the limit and the
// chain's total demand) rather than by re-running the walk loop, so a control
// flow error in the loop cannot hide behind an identical re-derivation.
//
// The walk driver itself calls the REAL AttemptBudget.Remaining and the REAL
// Failure.RetryBudget; only the loop shape is mirrored from
// platform/gateway/routing_proxy.go walkBackend, which is the consumption site
// the peer study names (docs/research/STUDY-vstack-2026-11-17.md, "2. Router
// failover": "One request's attempts must respect the selected budget even
// when eligibility changes").
//
// Witness: go test ./platform/routing -run '^TestAttemptBudgetCompositionModel$' -count=1 -v

import (
	"errors"
	"fmt"
	"testing"
)

// compositionMaxChain bounds the enumerated chain length. Three rungs is the
// smallest length that can express the property under test: a rung whose
// failure authorizes retries, a rung whose class authorizes none, and a rung
// reached only after the request budget is already spent.
const compositionMaxChain = 3

// compositionClasses is the representative closed set of failure classes: the
// two classes with their own same-target retry budget, the classes that
// authorize the full physical headroom instead, the fatal class, the
// admission-refusal class, and the unclassified default.
var compositionClasses = []FailureClass{
	FailureRateLimited,
	FailureTimeout,
	FailureAuthRejected,
	FailureCreditExhausted,
	FailureBadPayload,
	FailureDeadlineInfeasible,
	FailureUnknown,
}

// backendConfig is one rung of the enumerated chain: the physical credential
// headroom the rung offers, the class of failure it answers with, whether its
// response arrived after the client's deadline, and whether the failure is
// org-wide.
type backendConfig struct {
	headroom int
	class    FailureClass
	late     bool
	orgWide  bool
}

func (b backendConfig) failure() Failure {
	f := newFailure(b.class, "composition model")
	f.OrgWide = b.orgWide
	return f
}

// claimed mirrors walkBackend's `claimed` bool: only a rung that consumed a
// rotation lease may roll to another key, and a rung with no physical headroom
// never claims one, so it advances the chain after a single dial.
func (b backendConfig) claimed() bool { return b.headroom > 0 }

func (b backendConfig) String() string {
	return fmt.Sprintf("{headroom:%d class:%q late:%t orgWide:%t}", b.headroom, b.class, b.late, b.orgWide)
}

// walkDials drives the real budget functions through the walk loop documented
// in platform/gateway/routing_proxy.go walkBackend: one dial per rung, then
// same-backend key retries bounded by Failure.RetryBudget(headroom), with the
// request's AttemptBudget consulted before every dial (the Spend gate). A
// refused spend ends the whole request, not just the rung.
func walkDials(budget AttemptBudget, chain []backendConfig) (total int, perBackend []int, exhausted bool) {
	perBackend = make([]int, len(chain))
	spent := 0
	dial := func(rung int) bool {
		if budget.Remaining(spent) <= 0 {
			return false
		}
		spent++
		perBackend[rung]++
		return true
	}
	for rung, b := range chain {
		remaining, bounded := 0, false
		for {
			if !dial(rung) {
				return spent, perBackend, true
			}
			if !b.claimed() || b.late {
				break
			}
			if !bounded {
				remaining = b.failure().RetryBudget(b.headroom)
				bounded = true
			}
			if remaining <= 0 {
				break
			}
			remaining--
		}
	}
	return spent, perBackend, false
}

// oracleClass is the class table transcribed from the documented contract in
// failure.go, NOT read from the implementation. cap is the class's own
// same-target retry budget, where 0 means the class declares none and the
// physical headroom is the bound instead.
type oracleClass struct {
	cap     int
	fatal   bool
	refused bool
}

var oracleClasses = map[FailureClass]oracleClass{
	FailureRateLimited:        {cap: 2},
	FailureTimeout:            {cap: 1},
	FailureAuthRejected:       {},
	FailureCreditExhausted:    {},
	FailureBadPayload:         {fatal: true},
	FailureDeadlineInfeasible: {refused: true},
	FailureUnknown:            {},
}

// oracleRetries is the same-target retry count the documented contract
// authorizes for one rung, derived from the transcription above.
func oracleRetries(b backendConfig) int {
	if b.headroom <= 0 || b.orgWide || b.late {
		return 0
	}
	c, ok := oracleClasses[b.class]
	if !ok || c.fatal || c.refused {
		return 0
	}
	if c.cap > 0 && c.cap < b.headroom {
		return c.cap
	}
	return b.headroom
}

// oracleDials computes the walk's dial count in closed form: the chain's total
// demand is one dial per rung plus that rung's authorized retries, and the
// walk spends the smaller of that demand and the request's limit. It reports
// exhausted when the demand exceeds the limit, which is exactly when the walk
// reaches a dial the Spend gate refuses.
func oracleDials(limit int, chain []backendConfig) (total int, perBackend []int, exhausted bool) {
	perBackend = make([]int, len(chain))
	demand := 0
	for _, b := range chain {
		demand += 1 + oracleRetries(b)
	}
	total = min(limit, demand)
	exhausted = demand > limit
	left := total
	for i, b := range chain {
		take := min(1+oracleRetries(b), left)
		perBackend[i] = take
		left -= take
	}
	return total, perBackend, exhausted
}

type compositionMismatch struct {
	field  string
	detail string
}

func (m *compositionMismatch) Error() string { return m.field + ": " + m.detail }

// checkComposition compares one walk against the oracle and against the
// safety invariants that must hold whatever the oracle says: the request never
// spends more dials than its selected budget, never spends more on a rung than
// that rung's physical pool can supply, always makes progress on a non-empty
// chain, and accounts for every dial it reports.
func checkComposition(budget AttemptBudget, chain []backendConfig, walk func(AttemptBudget, []backendConfig) (int, []int, bool)) error {
	limit := budget.Max
	if limit <= 0 {
		limit = DefaultMaxAttempts
	}
	wantTotal, wantPer, wantExhausted := oracleDials(limit, chain)
	gotTotal, gotPer, gotExhausted := walk(budget, chain)

	if gotTotal != wantTotal {
		return &compositionMismatch{"total dials", fmt.Sprintf("got %d, want %d (limit %d, chain %v)", gotTotal, wantTotal, limit, chain)}
	}
	// The per-rung split is compared before the exhaustion verdict: a walk
	// that spends the right total on the wrong rungs is the composition
	// defect, and reporting it as "exhausted" would name the symptom.
	for i := range chain {
		if gotPer[i] != wantPer[i] {
			return &compositionMismatch{"per-rung dials", fmt.Sprintf("rung %d got %d, want %d (chain %v)", i, gotPer[i], wantPer[i], chain)}
		}
	}
	if gotExhausted != wantExhausted {
		return &compositionMismatch{"exhausted", fmt.Sprintf("got %t, want %t (limit %d, chain %v)", gotExhausted, wantExhausted, limit, chain)}
	}
	if gotTotal > limit {
		return &compositionMismatch{"budget overrun", fmt.Sprintf("spent %d dials against a limit of %d (chain %v)", gotTotal, limit, chain)}
	}
	if gotTotal < 1 {
		return &compositionMismatch{"no progress", fmt.Sprintf("spent %d dials on a non-empty chain (limit %d)", gotTotal, limit)}
	}
	sum := 0
	for i := range chain {
		if gotPer[i] > 1+chain[i].headroom {
			return &compositionMismatch{"headroom overrun", fmt.Sprintf("rung %d spent %d dials with headroom %d", i, gotPer[i], chain[i].headroom)}
		}
		sum += gotPer[i]
	}
	if sum != gotTotal {
		return &compositionMismatch{"accounting", fmt.Sprintf("per-rung dials sum to %d, total is %d", sum, gotTotal)}
	}
	return nil
}

// unboundedWalk is the negative control: it spends every rung's full retry
// allowance without ever consulting the request budget, so the checker must
// catch it as a budget overrun rather than accept it.
func unboundedWalk(_ AttemptBudget, chain []backendConfig) (int, []int, bool) {
	perBackend := make([]int, len(chain))
	spent := 0
	for rung, b := range chain {
		perBackend[rung]++
		spent++
		if !b.claimed() || b.late {
			continue
		}
		for r := 0; r < b.failure().RetryBudget(b.headroom); r++ {
			perBackend[rung]++
			spent++
		}
	}
	return spent, perBackend, false
}

// headroomOnlyWalk is the second negative control, and the one that matters
// for the composition: it honors the request's total budget exactly, but
// re-derives the same-target bound locally as the full physical headroom
// instead of asking Failure.RetryBudget. That is the defect fak-private#2706
// fixed, and it is invisible to any check that only looks at the total: the
// walk still stops at the request budget, it just hammers a provider that
// already said it was limiting.
func headroomOnlyWalk(budget AttemptBudget, chain []backendConfig) (int, []int, bool) {
	perBackend := make([]int, len(chain))
	spent := 0
	dial := func(rung int) bool {
		if budget.Remaining(spent) <= 0 {
			return false
		}
		spent++
		perBackend[rung]++
		return true
	}
	for rung, b := range chain {
		if !dial(rung) {
			return spent, perBackend, true
		}
		if !b.claimed() || b.late {
			continue
		}
		for r := 0; r < b.headroom; r++ {
			if !dial(rung) {
				return spent, perBackend, true
			}
		}
	}
	return spent, perBackend, false
}

// fak-test:justify why=invariant when=changed:platform/routing/**
// fak-test:runtime fast est=2s
func TestAttemptBudgetCompositionModel(t *testing.T) {
	t.Run("class table transcription", func(t *testing.T) {
		for class, want := range oracleClasses {
			f := newFailure(class, "transcription")
			if f.Retries != want.cap {
				t.Errorf("%s: implementation Retries = %d, transcribed cap = %d", class, f.Retries, want.cap)
			}
			if got := f.Action == ActionFatal; got != want.fatal {
				t.Errorf("%s: implementation fatal = %t, transcribed %t", class, got, want.fatal)
			}
			if got := class.AdmissionRefused(); got != want.refused {
				t.Errorf("%s: implementation AdmissionRefused = %t, transcribed %t", class, got, want.refused)
			}
		}
	})

	t.Run("exhaustive composition", func(t *testing.T) {
		var configs []backendConfig
		for headroom := 0; headroom <= 3; headroom++ {
			for _, class := range compositionClasses {
				for _, late := range []bool{false, true} {
					configs = append(configs, backendConfig{headroom: headroom, class: class, late: late})
				}
			}
		}
		budgets := []AttemptBudget{
			NewAttemptBudget(1), NewAttemptBudget(2), NewAttemptBudget(3), NewAttemptBudget(4),
			{}, // the zero value: a caller that never resolved a decision
		}
		wantChecked := len(budgets) * (len(configs) + len(configs)*len(configs) + len(configs)*len(configs)*len(configs))

		var chain [compositionMaxChain]backendConfig
		var checked int
		var enumerate func(depth int)
		enumerate = func(depth int) {
			if depth > 0 {
				for _, budget := range budgets {
					if err := checkComposition(budget, chain[:depth], walkDials); err != nil {
						t.Fatalf("budget %+v chain %v: %v", budget, chain[:depth], err)
					}
					checked++
				}
			}
			if depth == compositionMaxChain {
				return
			}
			for _, config := range configs {
				chain[depth] = config
				enumerate(depth + 1)
			}
		}
		enumerate(0)
		if checked != wantChecked {
			t.Fatalf("checked %d walks, want %d", checked, wantChecked)
		}
		t.Logf("checked %d walks (rungs %d, rung configs %d, budgets %d)", checked, compositionMaxChain, len(configs), len(budgets))
	})

	t.Run("two credentials with an exhausted budget", func(t *testing.T) {
		// Two eligible keys on the first rung (headroom 1) and one on the
		// second, against a request budget of two dials: the first rung's
		// rollover consumes the whole budget, so the second rung is never
		// dialed and the walk reports exhaustion.
		chain := []backendConfig{
			{headroom: 1, class: FailureAuthRejected},
			{headroom: 1, class: FailureRateLimited},
		}
		budget := NewAttemptBudget(2)
		if err := checkComposition(budget, chain, walkDials); err != nil {
			t.Fatalf("chain %v: %v", chain, err)
		}
		total, per, exhausted := walkDials(budget, chain)
		if total != 2 || per[1] != 0 || !exhausted {
			t.Fatalf("got total %d per %v exhausted %t, want 2 [2 0] true", total, per, exhausted)
		}
	})

	t.Run("late response spends no retry", func(t *testing.T) {
		// A rung whose response arrived after the client's deadline advances
		// the chain without spending a same-target retry, even though its
		// class and headroom would otherwise authorize one.
		chain := []backendConfig{
			{headroom: 3, class: FailureRateLimited, late: true},
			{headroom: 0, class: FailureUnknown},
		}
		budget := NewAttemptBudget(4)
		if err := checkComposition(budget, chain, walkDials); err != nil {
			t.Fatalf("chain %v: %v", chain, err)
		}
		total, per, exhausted := walkDials(budget, chain)
		if total != 2 || per[0] != 1 || exhausted {
			t.Fatalf("got total %d per %v exhausted %t, want 2 [1 1] false", total, per, exhausted)
		}
	})

	t.Run("org-wide failure authorizes no retry", func(t *testing.T) {
		// A bare 405 walls the credential org, so the next sibling key rolls
		// the same wall: the rung authorizes zero retries however much
		// headroom it holds.
		chain := []backendConfig{{headroom: 3, class: FailureUnavailable, orgWide: true}}
		budget := NewAttemptBudget(4)
		if err := checkComposition(budget, chain, walkDials); err != nil {
			t.Fatalf("chain %v: %v", chain, err)
		}
		if total, _, _ := walkDials(budget, chain); total != 1 {
			t.Fatalf("org-wide rung spent %d dials, want 1", total)
		}
	})

	t.Run("unclassified failure authorizes no retry", func(t *testing.T) {
		// The zero Failure is a caller bug: an unidentified failure is not
		// license to spend the pool.
		chain := []backendConfig{{headroom: 3, class: ""}}
		budget := NewAttemptBudget(4)
		if err := checkComposition(budget, chain, walkDials); err != nil {
			t.Fatalf("chain %v: %v", chain, err)
		}
		if total, _, _ := walkDials(budget, chain); total != 1 {
			t.Fatalf("unclassified rung spent %d dials, want 1", total)
		}
	})

	t.Run("detects a walk that ignores the budget", func(t *testing.T) {
		chain := []backendConfig{{headroom: 3, class: FailureRateLimited}}
		err := checkComposition(NewAttemptBudget(1), chain, unboundedWalk)
		var mismatch *compositionMismatch
		if !errors.As(err, &mismatch) || mismatch.field != "total dials" {
			t.Fatalf("want a total-dials counterexample for a budget-ignoring walk, got %v", err)
		}
		t.Logf("counterexample: %v", mismatch)
	})

	t.Run("detects a walk that re-derives the class cap locally", func(t *testing.T) {
		// The request budget is generous enough that both walks spend the
		// same TOTAL; only the per-rung split differs, which is exactly the
		// composition defect. A checker that compared totals alone would pass
		// this walk, so the per-rung comparison is what earns its keep.
		chain := []backendConfig{
			{headroom: 3, class: FailureRateLimited},
			{headroom: 0, class: FailureUnknown},
		}
		budget := NewAttemptBudget(4)
		if err := checkComposition(budget, chain, walkDials); err != nil {
			t.Fatalf("the real walk must satisfy the oracle: %v", err)
		}
		err := checkComposition(budget, chain, headroomOnlyWalk)
		var mismatch *compositionMismatch
		if !errors.As(err, &mismatch) || mismatch.field != "per-rung dials" {
			t.Fatalf("want a per-rung counterexample for a locally re-derived cap, got %v", err)
		}
		controlTotal, _, _ := headroomOnlyWalk(budget, chain)
		realTotal, _, _ := walkDials(budget, chain)
		if controlTotal != realTotal {
			t.Fatalf("control spent %d dials, real walk %d: the control must agree on the total to isolate the split", controlTotal, realTotal)
		}
		t.Logf("counterexample: %v", mismatch)
	})
}
