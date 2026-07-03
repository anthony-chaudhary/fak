package loopdrive

import "testing"

func TestLeaseRefSyncPlanConvergesBeforeDecide(t *testing.T) {
	steps := LeaseRefSyncPlan(LeaseRefSyncPlanInput{Remote: "upstream"})
	if len(steps) != 1 {
		t.Fatalf("steps = %+v, want exactly before-decide sync", steps)
	}
	step := steps[0]
	if step.Boundary != LeaseRefSyncBeforeDecide || step.Direction != LeaseRefSyncFetchOnly {
		t.Fatalf("first step = %+v, want fetch-only before decide", step)
	}
	if step.Remote != "upstream" {
		t.Fatalf("remote = %q, want upstream", step.Remote)
	}
	if step.Required {
		t.Fatalf("sync step must be advisory/nonfatal: %+v", step)
	}
}

func TestLeaseRefSyncPlanPublishesAfterWrite(t *testing.T) {
	steps := LeaseRefSyncPlan(LeaseRefSyncPlanInput{LeaseRefsWritten: true})
	if len(steps) != 2 {
		t.Fatalf("steps = %+v, want before-decide plus after-write", steps)
	}
	if steps[0].Boundary != LeaseRefSyncBeforeDecide || steps[0].Direction != LeaseRefSyncFetchOnly {
		t.Fatalf("first step = %+v, want fetch-only before decide", steps[0])
	}
	if steps[1].Boundary != LeaseRefSyncAfterWrite || steps[1].Direction != LeaseRefSyncPushOnly {
		t.Fatalf("second step = %+v, want push-only after write", steps[1])
	}
	if steps[0].Remote != "origin" || steps[1].Remote != "origin" {
		t.Fatalf("default remotes = %q/%q, want origin", steps[0].Remote, steps[1].Remote)
	}
}

func TestLeaseRefSyncReportSurfacesFailuresWithoutFatalStop(t *testing.T) {
	steps := LeaseRefSyncPlan(LeaseRefSyncPlanInput{Remote: "origin", LeaseRefsWritten: true})
	report := ReportLeaseRefSync([]LeaseRefSyncAttempt{
		{Step: steps[0]},
		{Step: steps[1], Err: "git push exited 128"},
	})
	if report.Outcome != LeaseRefSyncDegraded {
		t.Fatalf("outcome = %q, want degraded", report.Outcome)
	}
	if report.Reason != ReasonLeaseRefSyncTransport {
		t.Fatalf("reason = %q, want %q", report.Reason, ReasonLeaseRefSyncTransport)
	}
	if report.Fatal {
		t.Fatalf("sync transport failure must be surfaced but nonfatal: %+v", report)
	}
	if len(report.Failures) != 1 {
		t.Fatalf("failures = %+v, want one surfaced failure", report.Failures)
	}
}

func TestLeaseRefSyncReportCleanWhenAttemptsPass(t *testing.T) {
	report := ReportLeaseRefSync([]LeaseRefSyncAttempt{{Step: LeaseRefSyncPlan(LeaseRefSyncPlanInput{})[0]}})
	if report.Outcome != LeaseRefSyncOK || report.Fatal || len(report.Failures) != 0 {
		t.Fatalf("report = %+v, want clean nonfatal report", report)
	}
}
