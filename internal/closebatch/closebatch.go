// Package closebatch groups witnessed-closeable issues into dry-run batches
// before any live close mutates GitHub. Each batch names the issues it would
// close, the mutationbudget rate-limit Decision spending that many closes
// would need, and a rollback note (the `gh issue reopen` recipe) so an
// operator can inspect the whole plan before arming a live close arm. It is
// one leaf of the "safe 400 GitHub issues/hour parallel-agent throughput"
// program (issue #1825, fleet-400iph), following on from mutationbudget's
// per-burst guard (#1825, whose own doc names "wiring this guard into live gh
// calls" as a later ticket) and dispatch progress's witnessed-open detector
// (`fak dispatch progress --json` -> witnessed_numbers).
//
// Plan is a pure fold: witnessed issue numbers + a batch size + a starting
// Budget in, a Report of Batches out. It calls no gh, reads no clock (the
// clock is injected via NowUnix, exactly like mutationbudget itself), and
// never closes anything -- planning is all this package does. Wiring a
// --live close arm on top of an ALLOW batch is later, out-of-scope work.
//
// It is stdlib-plus-mutationbudget only -- off the hot path.
package closebatch

import (
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/mutationbudget"
)

// DefaultBatchSize is used when Input.BatchSize is <= 0.
const DefaultBatchSize = 10

// Input is the pre-gathered facts Plan folds into a Report.
type Input struct {
	// IssueNumbers are the witnessed-closeable issues (e.g. dispatch progress's
	// witnessed_numbers), in the order they should be considered for batching.
	IssueNumbers []int
	// BatchSize is the max issues per batch. <= 0 defaults to DefaultBatchSize.
	BatchSize int
	// Budget is the remaining GitHub API allowance at the start of the plan.
	Budget mutationbudget.Budget
	// Reserve is the minimum budget that must survive each batch (passed
	// through to mutationbudget.Guard unchanged).
	Reserve int
	// NowUnix is the clock, injected so a held batch's reset-window text is
	// deterministic rather than reading a live clock.
	NowUnix int64
}

// Batch is one dry-run group of witnessed closes: the issues it would close,
// the mutation cost that spends, the rate-limit Decision it would need before
// running live, and the rollback note naming how to undo it.
type Batch struct {
	Index        int                     `json:"index"`
	Issues       []int                   `json:"issues"`
	MutationCost int                     `json:"mutation_cost"`
	RateLimit    mutationbudget.Decision `json:"rate_limit"`
	Rollback     string                  `json:"rollback"`
}

// Report is the whole dry-run plan: every batch in order, plus the totals an
// operator reads before arming a live close arm.
type Report struct {
	Batches      []Batch `json:"batches"`
	TotalIssues  int     `json:"total_issues"`
	TotalBatches int     `json:"total_batches"`
	BatchSize    int     `json:"batch_size"`
	AllAllow     bool    `json:"all_allow"`
}

// Plan groups in.IssueNumbers into batches of in.BatchSize (in order), prices
// each batch's mutation cost against a running Budget via mutationbudget.Guard
// (each close spends one mutation), and attaches a rollback note per batch.
//
// The running budget only spends for a batch that Guard ALLOWs: a HELD batch
// never runs (mutationbudget's whole point is refusing a burst before it
// starts), so it spends nothing and the next batch is priced against the same
// remaining budget -- the honest consequence being that once one batch HOLDs,
// every batch after it holds too, until the window resets.
//
// Plan performs no I/O and reads no clock: the same Input always yields the
// same Report.
func Plan(in Input) Report {
	size := in.BatchSize
	if size <= 0 {
		size = DefaultBatchSize
	}
	rep := Report{BatchSize: size, TotalIssues: len(in.IssueNumbers), AllAllow: true}
	remaining := in.Budget
	for start := 0; start < len(in.IssueNumbers); start += size {
		end := start + size
		if end > len(in.IssueNumbers) {
			end = len(in.IssueNumbers)
		}
		issues := append([]int(nil), in.IssueNumbers[start:end]...)
		decision := mutationbudget.Guard(remaining, len(issues), in.Reserve, in.NowUnix)
		rep.Batches = append(rep.Batches, Batch{
			Index:        len(rep.Batches),
			Issues:       issues,
			MutationCost: len(issues),
			RateLimit:    decision,
			Rollback:     rollbackNote(issues),
		})
		if !decision.Allow {
			rep.AllAllow = false
			continue // held: nothing spent, next batch priced against the same remaining budget
		}
		remaining.Remaining = decision.AfterRemaining
	}
	rep.TotalBatches = len(rep.Batches)
	return rep
}

// rollbackNote renders the undo recipe for one batch: reopening every issue in
// it via a single `gh issue reopen` call.
func rollbackNote(issues []int) string {
	if len(issues) == 0 {
		return "no issues in this batch -- nothing to roll back"
	}
	refs := make([]string, len(issues))
	for i, n := range issues {
		refs[i] = fmt.Sprintf("%d", n)
	}
	return fmt.Sprintf("rollback: `gh issue reopen %s` restores every issue in this batch if the close was wrong",
		strings.Join(refs, " "))
}
