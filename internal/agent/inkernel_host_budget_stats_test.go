package agent

import (
	"context"
	"errors"
	"testing"
	"time"
)

// fak-test:runtime fast est=50ms lane=default
func TestHostMemoryBudgetStatsUnarmedIsZero(t *testing.T) {
	var nilPlanner *InKernelPlanner
	if got := nilPlanner.HostMemoryBudgetStats(); got != (HostMemoryBudgetStats{}) {
		t.Fatalf("nil planner stats = %+v, want zero value", got)
	}
	p := bareHostPlanner()
	if got := p.HostMemoryBudgetStats(); got != (HostMemoryBudgetStats{}) {
		t.Fatalf("unarmed planner stats = %+v, want zero value", got)
	}
	p.SetHostMemoryBudget(1<<30, constHostUsage(0, true).used)
	p.SetHostMemoryBudget(0, constHostUsage(0, true).used)
	if got := p.HostMemoryBudgetStats(); got != (HostMemoryBudgetStats{}) {
		t.Fatalf("disarmed planner stats = %+v, want zero value", got)
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestHostMemoryBudgetStatsSnapshot(t *testing.T) {
	const ceiling = int64(1 << 30)
	for _, tc := range []struct {
		name           string
		used           int64
		known          bool
		wantUsedKnown  bool
		wantAvail      int64
		wantStructural bool
	}{
		{"headroom", ceiling - (64 << 20), true, true, 64 << 20, false},
		{"at-ceiling", ceiling, true, true, 0, true},
		{"over-ceiling", ceiling + (5 << 20), true, true, -(5 << 20), true},
		{"unknown-usage", ceiling + (5 << 20), false, false, 0, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p := bareHostPlanner()
			p.SetHostMemoryBudget(ceiling, constHostUsage(tc.used, tc.known).used)
			st := p.HostMemoryBudgetStats()
			if !st.Armed || st.Ceiling != ceiling {
				t.Fatalf("stats armed=%v ceiling=%d, want armed ceiling %d", st.Armed, st.Ceiling, ceiling)
			}
			if st.UsedKnown != tc.wantUsedKnown {
				t.Fatalf("UsedKnown = %v, want %v", st.UsedKnown, tc.wantUsedKnown)
			}
			if st.AvailSigned != tc.wantAvail {
				t.Fatalf("AvailSigned = %d, want %d", st.AvailSigned, tc.wantAvail)
			}
			if st.Structural != tc.wantStructural {
				t.Fatalf("Structural = %v, want %v", st.Structural, tc.wantStructural)
			}
			if tc.wantUsedKnown && st.Used != tc.used {
				t.Fatalf("Used = %d, want %d", st.Used, tc.used)
			}
			if st.DeclinedTotal != 0 || !st.LastDeclineAt.IsZero() {
				t.Fatalf("fresh budget declined=%d last=%v, want 0/zero", st.DeclinedTotal, st.LastDeclineAt)
			}
		})
	}
}

// A refusal with a turn in flight is transient: the reservation will be released, so the
// budget is not structural even though the second turn is declined.
// fak-test:runtime fast est=50ms lane=default
func TestHostMemoryBudgetStatsInFlightIsNotStructural(t *testing.T) {
	const promptTokens, maxNew = 32, 8
	const ceiling = int64(1 << 30)
	ctx := context.Background()
	p := bareHostPlanner()
	session, _ := hostDemand(t, p, promptTokens, maxNew)
	free := session + session/2
	p.SetHostMemoryBudget(ceiling, constHostUsage(ceiling-free, true).used)

	_, release1, err := p.admitHostMemory(ctx, promptTokens, maxNew)
	if err != nil {
		t.Fatalf("first admission: %v", err)
	}
	defer release1()
	_, release2, err := p.admitHostMemory(ctx, promptTokens, maxNew)
	if release2 != nil {
		defer release2()
	}
	var capErr *InKernelCapacityError
	if !errors.As(err, &capErr) {
		t.Fatalf("second admission error = %T (%v), want *InKernelCapacityError", err, err)
	}
	if capErr.Structural {
		t.Fatal("refusal with a turn in flight marked Structural, want transient")
	}
	if capErr.AvailSigned != free-session || capErr.Avail != free-session {
		t.Fatalf("avail signed/clamped = %d/%d, want %d/%d", capErr.AvailSigned, capErr.Avail, free-session, free-session)
	}
	st := p.HostMemoryBudgetStats()
	if st.Structural {
		t.Fatal("stats Structural with a reservation in flight, want false")
	}
	if st.Reserved != session {
		t.Fatalf("Reserved = %d, want %d", st.Reserved, session)
	}
	if st.DeclinedTotal != 1 {
		t.Fatalf("DeclinedTotal = %d, want 1", st.DeclinedTotal)
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestHostMemoryBudgetStatsCountsStructuralDeclines(t *testing.T) {
	const promptTokens, maxNew = 32, 8
	const ceiling = int64(1 << 30)
	const over = int64(7 << 20)
	ctx := context.Background()
	p := bareHostPlanner()
	p.SetHostMemoryBudget(ceiling, constHostUsage(ceiling+over, true).used)

	before := time.Now()
	for i := 1; i <= 3; i++ {
		_, release, err := p.admitHostMemory(ctx, promptTokens, maxNew)
		if release != nil {
			release()
		}
		var capErr *InKernelCapacityError
		if !errors.As(err, &capErr) {
			t.Fatalf("decline %d: error = %T (%v), want *InKernelCapacityError", i, err, err)
		}
		if !capErr.Structural {
			t.Fatalf("decline %d: Structural = false, want true (idle usage over ceiling)", i)
		}
		if capErr.AvailSigned != -over {
			t.Fatalf("decline %d: AvailSigned = %d, want %d", i, capErr.AvailSigned, -over)
		}
		if capErr.Avail != 0 {
			t.Fatalf("decline %d: Avail = %d, want clamped 0", i, capErr.Avail)
		}
		st := p.HostMemoryBudgetStats()
		if st.DeclinedTotal != int64(i) {
			t.Fatalf("after decline %d: DeclinedTotal = %d", i, st.DeclinedTotal)
		}
		if st.LastDeclineAt.Before(before) || st.LastDeclineAt.After(time.Now()) {
			t.Fatalf("after decline %d: LastDeclineAt = %v outside [%v, now]", i, st.LastDeclineAt, before)
		}
		if !st.Structural || st.AvailSigned != -over {
			t.Fatalf("after decline %d: stats structural=%v avail=%d, want true/%d", i, st.Structural, st.AvailSigned, -over)
		}
	}
}
