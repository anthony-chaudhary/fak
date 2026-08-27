package localappmetrics

import (
	"errors"
	"testing"
	"time"
)

func TestOutcomeJoinKeepsAttemptedAcceptedAndUsedDistinct(t *testing.T) {
	start := time.Unix(1, 0)
	rows := []Operation{{JoinID: "run-a", InstallStarted: start, ReadyAt: start.Add(10 * time.Second), Eligible: true, LocalAttempted: true, LocalAccepted: true, ResultUsed: true, TTFT: 100 * time.Millisecond, EndToEnd: time.Second, PeakMemoryBytes: 4 << 30, PeakDiskBytes: 6 << 30, ForegroundImpact: 5 * time.Millisecond, LocalCost: 1, VerificationCost: .1, CloudBaselineCost: 3}, {JoinID: "run-b", Eligible: true, LocalAttempted: true, LocalAccepted: false, ResultUsed: false, HandoffReason: "quality_reject", TTFT: 300 * time.Millisecond, EndToEnd: 3 * time.Second, PeakMemoryBytes: 5 << 30, RemoteCost: 4, RetryCost: .5, UpdateRollback: true}, {JoinID: "run-c", Eligible: true, LocalAttempted: false, HandoffReason: "pressure", Crash: true}}
	r, err := Aggregate(rows)
	if err != nil {
		t.Fatal(err)
	}
	if r.Eligible != 3 || r.LocalAttempted != 2 || r.LocalAccepted != 1 || r.ResultUsed != 1 {
		t.Fatalf("collapsed funnel: %+v", r)
	}
	if r.Handoffs["quality_reject"] != 1 || r.Handoffs["pressure"] != 1 {
		t.Fatal(r.Handoffs)
	}
	if r.TTFTP50 != 100*time.Millisecond || r.TTFTP95 != 300*time.Millisecond || r.PeakMemoryBytes != 5<<30 || r.CrashFree != 2 || r.UpdateRollbacks != 1 {
		t.Fatalf("bad report: %+v", r)
	}
	if r.NetCloudSavings != 1.9 {
		t.Fatalf("net savings incorrect: %f", r.NetCloudSavings)
	}
}
func TestEphemeralJoinIdentityRequired(t *testing.T) {
	_, err := Aggregate([]Operation{{JoinID: "same"}, {JoinID: "same"}})
	if !errors.Is(err, ErrSensitive) {
		t.Fatal(err)
	}
	_, err = Aggregate([]Operation{{JoinID: ""}})
	if !errors.Is(err, ErrSensitive) {
		t.Fatal(err)
	}
}
