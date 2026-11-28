package agentbench

import (
	"context"
	"sync/atomic"
	"testing"
	"time"
)

// fak-test:runtime fast est=20ms lane=default
func TestSessionConcurrency(t *testing.T) {
	var inFlight atomic.Int32
	rep, err := RunSessionConcurrency(context.Background(), 4, 3, 2, func(ctx context.Context, sid, turn int) (int, int64, error) {
		inFlight.Add(1)
		defer inFlight.Add(-1)
		time.Sleep(2 * time.Millisecond)
		return 10 + turn, int64((turn + 1) * 1_000_000), nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if rep.CompletedTurns != 12 || rep.FailedTurns != 0 {
		t.Fatalf("completed=%d failed=%d, want 12/0", rep.CompletedTurns, rep.FailedTurns)
	}
	if rep.MaxActiveSessions < 1 || rep.MaxActiveSessions > 2 {
		t.Fatalf("MaxActiveSessions = %d, want in [1, 2]", rep.MaxActiveSessions)
	}
}
