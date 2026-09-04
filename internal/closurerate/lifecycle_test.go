package closurerate

import (
	"testing"
)

// Invariant: Closure rate metrics computation must distinguish witnessed closes from unwitnessed self-reports.
// Guard: Compute accurately calculates witnessed closure rates and claimed-without-witness counts.

func TestClosureRateLifecycle(t *testing.T) {
	t.Parallel()

	m := Fold(fixtureLedger, 4.0)
	if m.Total != 10 {
		t.Fatalf("expected total 10, got %d", m.Total)
	}
	if m.Closed != 8 {
		t.Fatalf("expected closed 8, got %d", m.Closed)
	}
	if m.Witnessed != 6 {
		t.Fatalf("expected witnessed 6, got %d", m.Witnessed)
	}
	if m.ClaimedWithoutWitness != 2 {
		t.Fatalf("expected 2 unwitnessed closes, got %d", m.ClaimedWithoutWitness)
	}
}
