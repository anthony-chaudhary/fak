package gateway

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"testing"
)

func TestFoldBatchBudgetsMatchesIndependentOracle(t *testing.T) {
	statuses := []batchBudgetStatus{
		batchBudgetAvailable,
		batchBudgetReached,
		batchBudgetExhausted,
	}
	names := []string{"active_tokens", "context_tokens", "sequences"}
	orders := [][]int{
		{0, 1, 2},
		{0, 2, 1},
		{1, 0, 2},
		{1, 2, 0},
		{2, 0, 1},
		{2, 1, 0},
	}

	for _, a := range statuses {
		for _, b := range statuses {
			for _, c := range statuses {
				input := []batchBudgetCheck{
					{status: a, budget: names[0], reason: names[0] + " refused"},
					{status: b, budget: names[1], reason: names[1] + " refused"},
					{status: c, budget: names[2], reason: names[2] + " refused"},
				}
				want := referenceBatchBudgetFold(input)
				for _, order := range orders {
					budgets := make([]batchBudget, 0, len(order))
					for _, idx := range order {
						check := input[idx]
						budgets = append(budgets, func(batchBudgetSnapshot, SeqRequest) batchBudgetCheck {
							return check
						})
					}
					got := foldBatchBudgets(batchBudgetSnapshot{running: 3, tokens: 17}, SeqRequest{Tokens: 5}, budgets)
					if got != want {
						t.Fatalf("statuses=(%d,%d,%d) order=%v: fold=%+v, oracle=%+v", a, b, c, order, got, want)
					}
				}
			}
		}
	}
}

// referenceBatchBudgetFold is intentionally expressed separately from the
// production severity comparison: it classifies the whole input, then selects
// the stable identity from the winning class.
func referenceBatchBudgetFold(checks []batchBudgetCheck) batchBudgetCheck {
	wantStatus := batchBudgetAvailable
	var reached, exhausted []batchBudgetCheck
	for _, check := range checks {
		switch check.status {
		case batchBudgetReached:
			reached = append(reached, check)
		case batchBudgetExhausted:
			exhausted = append(exhausted, check)
		}
	}

	candidates := reached
	if len(exhausted) > 0 {
		wantStatus = batchBudgetExhausted
		candidates = exhausted
	} else if len(reached) > 0 {
		wantStatus = batchBudgetReached
	}
	if len(candidates) == 0 {
		return batchBudgetCheck{status: wantStatus}
	}
	sort.Slice(candidates, func(i, j int) bool {
		if candidates[i].budget != candidates[j].budget {
			return candidates[i].budget < candidates[j].budget
		}
		return candidates[i].reason < candidates[j].reason
	})
	got := candidates[0]
	got.status = wantStatus
	return got
}

func TestAdmissionIndependentBatchBudgetsRefuseRegardlessOfOrder(t *testing.T) {
	for _, order := range []string{"reached-first", "exhausted-first"} {
		t.Run(order, func(t *testing.T) {
			calls := map[string]int{}
			fixed := func(name string, status batchBudgetStatus) batchBudget {
				return func(batchBudgetSnapshot, SeqRequest) batchBudgetCheck {
					calls[name]++
					return batchBudgetCheck{
						status: status,
						budget: name,
						reason: fmt.Sprintf("%s exhausted", name),
					}
				}
			}
			reached := fixed("active_tokens", batchBudgetReached)
			exhausted := fixed("context_tokens", batchBudgetExhausted)
			budgets := []batchBudget{reached, exhausted}
			if order == "exhausted-first" {
				budgets[0], budgets[1] = budgets[1], budgets[0]
			}

			ctl := newAdmissionControllerWithBudgets(AdmissionPolicy{MaxWaiting: 1}, budgets...)
			lease, err := ctl.Acquire(context.Background(), SeqRequest{TraceID: "candidate", Tokens: 4})
			if lease != nil {
				t.Fatal("Acquire returned a lease, want composed exhaustion to refuse admission")
			}
			var admissionErr *AdmissionError
			if !errors.As(err, &admissionErr) {
				t.Fatalf("Acquire error = %v, want typed AdmissionError", err)
			}
			if admissionErr.Verdict != VerdictShed || admissionErr.Budget != "context_tokens" || admissionErr.Reason != "context_tokens exhausted" {
				t.Fatalf("AdmissionError = %+v, want stable context_tokens exhaustion", admissionErr)
			}
			if calls["active_tokens"] != 1 || calls["context_tokens"] != 1 {
				t.Fatalf("budget calls = %#v, want every independent budget evaluated once", calls)
			}
			if stats := ctl.Stats(); stats.Running != 0 || stats.Waiting != 0 || stats.Shed != 1 {
				t.Fatalf("stats = %+v, want refusal before admission or queueing", stats)
			}
		})
	}
}

func TestAdmissionBuiltInBatchBudgetsComposeIndependently(t *testing.T) {
	ctl := NewAdmissionController(AdmissionPolicy{
		MaxNumSeqs:  2,
		TokenBudget: 10,
		MaxWaiting:  2,
	})

	if got := ctl.Offer(SeqRequest{TraceID: "a", Tokens: 6}); got != VerdictAdmitted {
		t.Fatalf("a verdict = %s, want admitted", got)
	}
	// This request reaches both independent budgets exactly: 2 sequences and
	// 10 tokens. It still fits, but the next request must wait.
	if got := ctl.Offer(SeqRequest{TraceID: "b", Tokens: 4}); got != VerdictAdmitted {
		t.Fatalf("b verdict = %s, want admitted at the composed boundary", got)
	}
	if got := ctl.Offer(SeqRequest{TraceID: "c", Tokens: 5}); got != VerdictQueued {
		t.Fatalf("c verdict = %s, want queued with both budgets full", got)
	}
	if stats := ctl.Stats(); stats.Running != 2 || stats.TokensInUse != 10 || stats.Waiting != 1 {
		t.Fatalf("full-boundary stats = %+v, want running=2 tokens=10 waiting=1", stats)
	}

	if !ctl.Complete("a") {
		t.Fatal("Complete(a) reported not running")
	}
	admitted := ctl.Schedule()
	if len(admitted) != 1 || admitted[0].TraceID != "c" {
		t.Fatalf("Schedule after releasing a = %v, want [c]", admitted)
	}
	if stats := ctl.Stats(); stats.Running != 2 || stats.TokensInUse != 9 || stats.Waiting != 0 {
		t.Fatalf("post-promotion stats = %+v, want running=2 tokens=9 waiting=0", stats)
	}
}

func TestFoldBatchBudgetsFailsClosedOnInvalidBudget(t *testing.T) {
	invalidStatus := func(batchBudgetSnapshot, SeqRequest) batchBudgetCheck {
		return batchBudgetCheck{status: batchBudgetStatus(99), budget: "broken"}
	}
	for _, tt := range []struct {
		name    string
		budgets []batchBudget
		budget  string
		reason  string
	}{
		{name: "nil", budgets: []batchBudget{nil}, budget: "invalid_batch_budget", reason: "nil batch budget"},
		{name: "invalid status", budgets: []batchBudget{invalidStatus}, budget: "broken", reason: `batch budget "broken" returned invalid status 99`},
	} {
		t.Run(tt.name, func(t *testing.T) {
			got := foldBatchBudgets(batchBudgetSnapshot{}, SeqRequest{}, tt.budgets)
			if got.status != batchBudgetExhausted || got.budget != tt.budget || got.reason != tt.reason {
				t.Fatalf("fold = %+v, want exhausted budget=%q reason=%q", got, tt.budget, tt.reason)
			}
		})
	}
}
