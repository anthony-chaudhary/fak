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
	if step.Surface != LeaseRefSyncSurfaceGeneric {
		t.Fatalf("surface = %q, want generic", step.Surface)
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

func TestLeaseRefSyncPlanForSurfaces(t *testing.T) {
	for _, surface := range []LeaseRefSyncSurface{
		LeaseRefSyncSurfaceDispatchPreflight,
		LeaseRefSyncSurfaceLoopDriveTick,
		LeaseRefSyncSurfaceGardenStaleLease,
	} {
		t.Run(string(surface), func(t *testing.T) {
			steps := LeaseRefSyncPlanForSurface(surface, LeaseRefSyncPlanInput{LeaseRefsWritten: true})
			if len(steps) != 2 {
				t.Fatalf("steps = %+v, want before-decide plus after-write", steps)
			}
			for _, step := range steps {
				if step.Surface != surface {
					t.Fatalf("step surface = %q, want %q in %+v", step.Surface, surface, step)
				}
			}
			if steps[0].Direction != LeaseRefSyncFetchOnly || steps[1].Direction != LeaseRefSyncPushOnly {
				t.Fatalf("directions = %q/%q, want fetch-only then push-only", steps[0].Direction, steps[1].Direction)
			}
		})
	}
}

func TestLeaseRefSyncCommandArgsAndDirections(t *testing.T) {
	steps := LeaseRefSyncPlanForSurface(LeaseRefSyncSurfaceDispatchPreflight, LeaseRefSyncPlanInput{Remote: "origin", LeaseRefsWritten: true})
	args, err := LeaseRefSyncCommandArgs(steps[0], "/repo")
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"leaseref", "sync", "--dir", "/repo", "--remote", "origin", "--fetch-only"}
	if !sameStrings(args, want) {
		t.Fatalf("fetch args = %v, want %v", args, want)
	}
	push, fetch, err := LeaseRefSyncDirections(steps[0])
	if err != nil {
		t.Fatal(err)
	}
	if push || !fetch {
		t.Fatalf("fetch step directions = push:%v fetch:%v, want push:false fetch:true", push, fetch)
	}

	args, err = LeaseRefSyncCommandArgs(steps[1], "")
	if err != nil {
		t.Fatal(err)
	}
	want = []string{"leaseref", "sync", "--remote", "origin", "--push-only"}
	if !sameStrings(args, want) {
		t.Fatalf("push args = %v, want %v", args, want)
	}
	push, fetch, err = LeaseRefSyncDirections(steps[1])
	if err != nil {
		t.Fatal(err)
	}
	if !push || fetch {
		t.Fatalf("push step directions = push:%v fetch:%v, want push:true fetch:false", push, fetch)
	}
}

func TestLeaseRefSyncCommandArgsRejectsUnknownDirection(t *testing.T) {
	step := LeaseRefSyncStep{Direction: LeaseRefSyncDirection("both")}
	if _, err := LeaseRefSyncCommandArgs(step, ""); err == nil {
		t.Fatal("want command arg refusal for unknown direction")
	}
	if _, _, err := LeaseRefSyncDirections(step); err == nil {
		t.Fatal("want direction refusal for unknown direction")
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

func sameStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
