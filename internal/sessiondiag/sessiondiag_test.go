package sessiondiag

import (
	"testing"
	"time"
)

func TestClassifyCapturedCodex0147PressureWithoutOverclaim(t *testing.T) {
	e := Evidence{DBBasename: "logs_2.sqlite", DBBytes: 1755549696, WALBytes: 400748312, PageSize: 4096, PageCount: 428601, FreelistPages: 359146, RecentRows: 15803, QueueDrops: 996, SlowWrites: 200, Integrity: "ok", WindowSeconds: 86400}
	r := Classify(e, time.Unix(1, 0))
	if r.Verdict != "CORRELATED_RUNTIME_PRESSURE" || r.Causality != "not_established" {
		t.Fatalf("report=%+v", r)
	}
	want := map[string]bool{"LOG_STORE_RECLAIMABLE_PRESSURE": false, "LOG_WAL_PRESSURE": false, "LOG_WRITE_CONTENTION": false, "APP_SERVER_EVENT_LOSS": false}
	for _, f := range r.Findings {
		if _, ok := want[f.Code]; ok {
			want[f.Code] = true
		}
	}
	for code, got := range want {
		if !got {
			t.Errorf("missing %s", code)
		}
	}
}
func TestClassifyExplicitFailureAndClean(t *testing.T) {
	r := Classify(Evidence{Integrity: "ok", ExplicitFailures: 2}, time.Time{})
	if r.Verdict != "EXPLICIT_FAILURE_EVIDENCE" || r.Causality != "failure_recorded" {
		t.Fatalf("%+v", r)
	}
	r = Classify(Evidence{Integrity: "ok"}, time.Time{})
	if r.Verdict != "NO_FAULT_EVIDENCE" || r.Findings[0].Code != "INSUFFICIENT_CRASH_EVIDENCE" {
		t.Fatalf("%+v", r)
	}
}
