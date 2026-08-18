package trajhook

import (
	"github.com/anthony-chaudhary/fak/internal/trajectory"
	"testing"
)

func TestTraceAccumulatorPreservesFirstSeenOrder(t *testing.T) {
	byTrace := map[string]*int{}
	var order []string
	for _, id := range []string{"b", "a", "b"} {
		p := traceAccumulator(byTrace, &order, trajectory.Turn{TraceID: id}, func() *int { return new(int) })
		*p++
	}
	if len(order) != 2 || order[0] != "b" || order[1] != "a" || *byTrace["b"] != 2 {
		t.Fatalf("order=%v counts=%v", order, byTrace)
	}
}
