package closebatch

import (
	"github.com/anthony-chaudhary/fak/internal/mutationbudget"
	"time"
)

type ComparisonArm struct {
	Name, Kind                                 string
	Available, Correct                         bool
	Latency                                    time.Duration
	Issues, Batches, Allowed, Held, FalsePlans int
	EstimatedRequests                          int
	CPUSeconds                                 float64
	PeakRSSBytes, NetworkBytes                 int64
	OperatorSeconds, CostUSD                   float64
	Note                                       string
}
type ComparisonResult struct {
	Workload string
	Arms     []ComparisonArm
}

var compareInput = Input{IssueNumbers: []int{1, 2, 3, 4, 5, 6, 7}, BatchSize: 3, Budget: mutationbudget.Budget{Remaining: 5, Limit: 5}, Reserve: 1}

func nativeArm() ComparisonArm {
	a := ComparisonArm{Name: "fak native issue-close batch planner", Kind: "native", Available: true, Issues: 7, Note: "deterministic batching, request-cost accounting, budget holds, and rollback commands"}
	st := time.Now()
	r := Plan(compareInput)
	a.Latency = time.Since(st)
	a.Batches = len(r.Batches)
	for _, b := range r.Batches {
		a.EstimatedRequests += b.MutationCost
		if b.RateLimit.Allow {
			a.Allowed++
		} else {
			a.Held++
		}
		if b.Rollback == "" {
			a.FalsePlans++
		}
	}
	a.Correct = a.Batches == 3 && a.Allowed == 2 && a.Held == 1 && a.EstimatedRequests == 7 && a.FalsePlans == 0
	return a
}
func fixedChunkArm() ComparisonArm {
	a := ComparisonArm{Name: "fixed-size chunking only", Kind: "baseline", Available: true, Issues: 7, Note: "tuned baseline chunks deterministically but ignores request budget and rollback"}
	st := time.Now()
	for i := 0; i < len(compareInput.IssueNumbers); i += compareInput.BatchSize {
		a.Batches++
		a.Allowed++
	}
	a.EstimatedRequests = len(compareInput.IssueNumbers)
	a.Latency = time.Since(st)
	a.FalsePlans = 2
	return a
}
func ua(n, k, s string) ComparisonArm { return ComparisonArm{Name: n, Kind: k, Note: s} }
func CompareLocal() ComparisonResult {
	return ComparisonResult{Workload: "plan the same seven issue closures into batches of three under a five-request budget at two requests per issue, including exact holds, costs, and rollback commands", Arms: []ComparisonArm{nativeArm(), fixedChunkArm(), ua("fak + GitHub Issues", "integration", "requires real batched close and independent reopen read-back"), ua("GitHub CLI issue close loop", "external", "requires real gh close loop"), ua("GitHub GraphQL mutation batching", "external", "requires pinned GraphQL mutations and rate-limit accounting"), ua("Jira bulk transition", "external", "requires equivalent Jira bulk workflow"), ua("Linear bulk issue update", "external", "requires equivalent Linear workflow")}}
}
