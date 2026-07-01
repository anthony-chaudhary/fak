package dispatchorder

import (
	"fmt"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/turntaxmeter"
)

func TestFanoutPlanningOverheadBudget(t *testing.T) {
	if raceDetectorEnabled {
		t.Skip("overhead-budget envelope is not meaningful under go test -race instrumentation")
	}
	budget, ok := turntaxmeter.DefaultBudget("dispatch", "plan_fanout")
	if !ok {
		t.Fatalf("dispatch/plan_fanout overhead budget is not declared")
	}

	const candidates = 400
	input := Input{
		NowUnix:         base,
		CooldownSeconds: -1,
		Candidates:      fanoutBudgetCandidates(candidates),
	}
	start := time.Now()
	res := Plan(input)
	elapsed := time.Since(start)

	span := turntaxmeter.Span{
		Rung:      "dispatch",
		Method:    "plan_fanout",
		ElapsedNS: elapsed.Nanoseconds(),
	}
	if breach, reason := turntaxmeter.CheckSpan(span); breach {
		t.Fatalf("dispatch fan-out planning took %s over declared envelope %s (%s); candidates=%d",
			elapsed, time.Duration(budget.MaxNS), reason, candidates)
	}
	if res.KeepCount != candidates || res.SafeConcurrency != candidates || len(res.Collisions) != 0 {
		t.Fatalf("budget fixture changed shape: keep=%d safe=%d collisions=%d, want %d/%d/0",
			res.KeepCount, res.SafeConcurrency, len(res.Collisions), candidates, candidates)
	}
}

func fanoutBudgetCandidates(n int) []Candidate {
	out := make([]Candidate, 0, n)
	for i := 0; i < n; i++ {
		out = append(out, Candidate{
			ID:          fmt.Sprintf("worker-%03d", i),
			Key:         fmt.Sprintf("target-%03d", i),
			Lane:        fmt.Sprintf("lane-%03d", i),
			Tree:        []string{fmt.Sprintf("internal/dispatchbudget/%03d/work.go", i)},
			Mode:        "exclusive",
			CreatedUnix: int64(base - n + i),
			UpdatedUnix: int64(base - n + i),
		})
	}
	return out
}
