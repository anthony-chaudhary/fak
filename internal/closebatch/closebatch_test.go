package closebatch

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/mutationbudget"
)

func issueRange(n int) []int {
	out := make([]int, n)
	for i := range out {
		out[i] = 1000 + i
	}
	return out
}

func TestPlan_SplitsIntoBatchesOfBatchSize(t *testing.T) {
	rep := Plan(Input{
		IssueNumbers: issueRange(25),
		BatchSize:    10,
		Budget:       mutationbudget.Budget{Remaining: 5000, Limit: 5000, ResetAtUnix: 4000},
		Reserve:      50,
		NowUnix:      1000,
	})
	if rep.TotalBatches != 3 {
		t.Fatalf("want 3 batches for 25 issues at size 10, got %d", rep.TotalBatches)
	}
	if len(rep.Batches[0].Issues) != 10 || len(rep.Batches[1].Issues) != 10 || len(rep.Batches[2].Issues) != 5 {
		t.Fatalf("want batch sizes 10,10,5, got %d,%d,%d",
			len(rep.Batches[0].Issues), len(rep.Batches[1].Issues), len(rep.Batches[2].Issues))
	}
	if rep.Batches[0].MutationCost != 10 || rep.Batches[2].MutationCost != 5 {
		t.Fatalf("mutation cost should equal batch size, got %+v", rep.Batches)
	}
	if !rep.AllAllow {
		t.Fatalf("ample budget should allow every batch, got %+v", rep)
	}
}

func TestPlan_WitnessTwoBatchesWithCountsAndCosts(t *testing.T) {
	// The #1826 witness: "a sample dry-run shows at least two batches with
	// counts and costs."
	rep := Plan(Input{
		IssueNumbers: issueRange(12),
		BatchSize:    5,
		Budget:       mutationbudget.Budget{Remaining: 500, Limit: 500, ResetAtUnix: 4000},
		Reserve:      10,
		NowUnix:      1000,
	})
	if rep.TotalBatches < 2 {
		t.Fatalf("want at least 2 batches, got %d", rep.TotalBatches)
	}
	for _, b := range rep.Batches {
		if b.MutationCost != len(b.Issues) {
			t.Errorf("batch %d: cost %d != issue count %d", b.Index, b.MutationCost, len(b.Issues))
		}
		if b.RateLimit.Reason == "" {
			t.Errorf("batch %d: want a populated rate-limit reason", b.Index)
		}
	}
}

func TestPlan_EmptyIssueList_ZeroBatches(t *testing.T) {
	rep := Plan(Input{IssueNumbers: nil, Budget: mutationbudget.Budget{Remaining: 100, Limit: 100}})
	if rep.TotalBatches != 0 || len(rep.Batches) != 0 {
		t.Fatalf("want 0 batches for an empty issue list, got %+v", rep)
	}
	if !rep.AllAllow {
		t.Fatalf("no batches means nothing was held, want AllAllow=true, got %+v", rep)
	}
}

func TestPlan_DefaultBatchSize(t *testing.T) {
	rep := Plan(Input{
		IssueNumbers: issueRange(15),
		Budget:       mutationbudget.Budget{Remaining: 500, Limit: 500},
	})
	if rep.BatchSize != DefaultBatchSize {
		t.Fatalf("want default batch size %d when unset, got %d", DefaultBatchSize, rep.BatchSize)
	}
	if rep.TotalBatches != 2 {
		t.Fatalf("want 2 batches of default size %d for 15 issues, got %d", DefaultBatchSize, rep.TotalBatches)
	}
}

func TestPlan_ThinBudgetHoldsLaterBatch(t *testing.T) {
	// Remaining=12, reserve=5: batch 1 (10 issues) leaves 2 < 5 -> HOLD.
	// A held batch spends nothing, so batch 2 is priced against the same
	// thin remaining budget and also HOLDs.
	rep := Plan(Input{
		IssueNumbers: issueRange(20),
		BatchSize:    10,
		Budget:       mutationbudget.Budget{Remaining: 12, Limit: 100, ResetAtUnix: 5000},
		Reserve:      5,
		NowUnix:      1000,
	})
	if rep.TotalBatches != 2 {
		t.Fatalf("want 2 batches, got %d", rep.TotalBatches)
	}
	if rep.Batches[0].RateLimit.Allow {
		t.Fatalf("want batch 0 held (12-10=2 < reserve 5), got %+v", rep.Batches[0].RateLimit)
	}
	if rep.Batches[1].RateLimit.Allow {
		t.Fatalf("want batch 1 also held (budget unspent by the held batch 0), got %+v", rep.Batches[1].RateLimit)
	}
	if rep.Batches[1].RateLimit.Remaining != 12 {
		t.Fatalf("want batch 1 priced against the same remaining=12 (batch 0 never spent), got %d",
			rep.Batches[1].RateLimit.Remaining)
	}
	if rep.AllAllow {
		t.Fatalf("want AllAllow=false when any batch is held, got %+v", rep)
	}
}

func TestPlan_AllowedBatchSpendsBudgetForNextBatch(t *testing.T) {
	// Remaining=20, reserve=5: batch 1 (10) leaves 10 >= 5 -> ALLOW, spends 10.
	// Batch 2 (10) is then priced against remaining=10, leaving 0 < 5 -> HOLD.
	rep := Plan(Input{
		IssueNumbers: issueRange(20),
		BatchSize:    10,
		Budget:       mutationbudget.Budget{Remaining: 20, Limit: 100, ResetAtUnix: 5000},
		Reserve:      5,
		NowUnix:      1000,
	})
	if !rep.Batches[0].RateLimit.Allow {
		t.Fatalf("want batch 0 allowed, got %+v", rep.Batches[0].RateLimit)
	}
	if rep.Batches[1].RateLimit.Remaining != 10 {
		t.Fatalf("want batch 1 priced against remaining=10 after batch 0 spent 10, got %d",
			rep.Batches[1].RateLimit.Remaining)
	}
	if rep.Batches[1].RateLimit.Allow {
		t.Fatalf("want batch 1 held (10-10=0 < reserve 5), got %+v", rep.Batches[1].RateLimit)
	}
}

func TestRollbackNote_NamesGhIssueReopenAndIssues(t *testing.T) {
	rep := Plan(Input{
		IssueNumbers: []int{101, 102, 103},
		BatchSize:    10,
		Budget:       mutationbudget.Budget{Remaining: 100, Limit: 100},
	})
	note := rep.Batches[0].Rollback
	if !strings.Contains(note, "gh issue reopen") {
		t.Errorf("want rollback note to name `gh issue reopen`, got %q", note)
	}
	for _, want := range []string{"101", "102", "103"} {
		if !strings.Contains(note, want) {
			t.Errorf("want rollback note to name issue %s, got %q", want, note)
		}
	}
}
