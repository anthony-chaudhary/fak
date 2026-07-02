package preflight

import (
	"context"
	"errors"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/testroute"
)

func workspaceGraph() PackageGraph {
	return PackageGraph{
		TotalPackages: 3,
		FileToPackage: map[string]string{
			"internal/preflight/preflight.go": "example.com/fak/internal/preflight",
			"internal/preflight/workspace.go": "example.com/fak/internal/preflight",
			"cmd/fak/main.go":                 "example.com/fak/cmd/fak",
			"internal/other/other.go":         "example.com/fak/internal/other",
		},
		Edges: map[string][]string{
			"example.com/fak/cmd/fak":        {"example.com/fak/internal/preflight"},
			"example.com/fak/internal/other": {"example.com/fak/internal/preflight"},
		},
	}
}

func TestPlanWorkspacePreflightReadWriteSetReady(t *testing.T) {
	got := PlanWorkspacePreflight(WorkspacePreflightInput{
		TaskID:       "2137",
		Actor:        "worker-b",
		ReadGlobs:    []string{"docs/**", "internal\\preflight\\*.go"},
		WriteGlobs:   []string{"internal/preflight/**"},
		PackageGraph: workspaceGraph(),
		TestProbe: testroute.Probe{
			GOOS:              "windows",
			NativeTestAllowed: false,
			WSLPresent:        true,
		},
		TestArgs: []string{"-short"},
	})

	if got.Verdict != WorkspaceVerdictReady {
		t.Fatalf("verdict = %s reason=%s detail=%s, want READY", got.Verdict, got.Reason, got.Detail)
	}
	if got.LeaseRequest == nil || got.LeaseRequest.ID != "preflight-2137" || got.LeaseRequest.Holder != "worker-b" {
		t.Fatalf("lease request = %+v, want deterministic worker lease", got.LeaseRequest)
	}
	wantDirect := []string{"example.com/fak/internal/preflight"}
	if !reflect.DeepEqual(got.DirectPackages, wantDirect) {
		t.Fatalf("direct packages = %v, want %v", got.DirectPackages, wantDirect)
	}
	wantAffected := []string{
		"example.com/fak/cmd/fak",
		"example.com/fak/internal/other",
		"example.com/fak/internal/preflight",
	}
	if !reflect.DeepEqual(got.AffectedPackages, wantAffected) {
		t.Fatalf("affected packages = %v, want %v", got.AffectedPackages, wantAffected)
	}
	if got.GoBuild == nil || !reflect.DeepEqual(got.GoBuild.Command, []string{"go", "build", "example.com/fak/internal/preflight"}) {
		t.Fatalf("go build plan = %+v", got.GoBuild)
	}
	if !reflect.DeepEqual(got.GoBuild.Env, []string{"GOTOOLCHAIN=auto"}) {
		t.Fatalf("go build env = %v, want GOTOOLCHAIN=auto", got.GoBuild.Env)
	}
	if got.TestRoute.Kind != testroute.KindWSL {
		t.Fatalf("test route = %+v, want WSL", got.TestRoute)
	}
	if len(got.TestCommand) == 0 || got.TestCommand[0] != "powershell" {
		t.Fatalf("test command = %v, want powershell WSL wrapper", got.TestCommand)
	}
	if got.DevIndex == nil || !reflect.DeepEqual(got.DevIndex.Globs, []string{"docs/**", "internal/preflight/*.go"}) {
		t.Fatalf("devindex request = %+v", got.DevIndex)
	}
	if kinds := stepKinds(got.Steps); !reflect.DeepEqual(kinds, []string{StepAcquireWriteLease, StepWarmGoBuildCache, StepResolveTestRoute, StepWarmDevIndex}) {
		t.Fatalf("step kinds = %v", kinds)
	}
}

func TestPlanWorkspacePreflightBlocksCollidingDeclarationBeforePrewarm(t *testing.T) {
	got := PlanWorkspacePreflight(WorkspacePreflightInput{
		TaskID:       "2137",
		WriteGlobs:   []string{"internal/preflight/preflight.go"},
		PackageGraph: workspaceGraph(),
		LiveLeases: []LeaseObservation{
			{ID: "z-docs", Holder: "peer-docs", Tree: []string{"docs/**"}},
			{ID: "a-preflight", Holder: "peer-a", Tree: []string{"internal/preflight/**"}},
		},
	})

	if got.Verdict != WorkspaceVerdictBlockedByLease {
		t.Fatalf("verdict = %s, want BLOCKED_BY_LEASE", got.Verdict)
	}
	if got.Reason != ReasonCollisionRisk {
		t.Fatalf("reason = %s, want %s", got.Reason, ReasonCollisionRisk)
	}
	if got.Conflict == nil || got.Conflict.ID != "a-preflight" || got.Conflict.Holder != "peer-a" {
		t.Fatalf("conflict = %+v, want lexically first colliding peer lease", got.Conflict)
	}
	if len(got.Steps) != 0 || got.GoBuild != nil || got.LeaseRequest != nil {
		t.Fatalf("blocked plan must not schedule prewarm/acquire steps: steps=%v build=%+v lease=%+v", got.Steps, got.GoBuild, got.LeaseRequest)
	}
}

func TestPlanWorkspacePreflightWouldBeRedWhenTestRouteUnavailable(t *testing.T) {
	got := PlanWorkspacePreflight(WorkspacePreflightInput{
		WriteGlobs:   []string{"internal/preflight/**"},
		PackageGraph: workspaceGraph(),
		TestProbe:    testroute.Probe{GOOS: "windows"},
	})
	if got.Verdict != WorkspaceVerdictWouldBeRed || got.Reason != ReasonTestRouteUnavailable {
		t.Fatalf("verdict/reason = %s/%s, want WOULD_BE_RED/%s", got.Verdict, got.Reason, ReasonTestRouteUnavailable)
	}
}

func TestWarmGoBuildCacheReportsWarmDelta(t *testing.T) {
	runner := &fakeGoBuildRunner{runs: []GoBuildRun{
		{ExitCode: 0, ElapsedMS: 61000},
		{ExitCode: 0, ElapsedMS: 450},
	}}
	req := PlanGoBuildWarm([]string{"example.com/fak/internal/preflight"}, true, 2000)
	got := WarmGoBuildCache(context.Background(), runner, req)

	if got.Verdict != WorkspaceVerdictReady || !got.Warm {
		t.Fatalf("warm report = %+v, want READY warm", got)
	}
	if got.ColdElapsedMS != 61000 || got.WarmElapsedMS != 450 || got.DeltaMS != 60550 {
		t.Fatalf("elapsed/delta = cold %d warm %d delta %d", got.ColdElapsedMS, got.WarmElapsedMS, got.DeltaMS)
	}
	if runner.calls != 2 {
		t.Fatalf("runner calls = %d, want two build probes", runner.calls)
	}
}

func TestWarmGoBuildCacheFailureIsWouldBeRed(t *testing.T) {
	runner := &fakeGoBuildRunner{runs: []GoBuildRun{{ExitCode: 1, ElapsedMS: 100, OutputTail: "compile failed"}}}
	req := PlanGoBuildWarm([]string{"example.com/fak/internal/preflight"}, true, 2000)
	got := WarmGoBuildCache(context.Background(), runner, req)
	if got.Verdict != WorkspaceVerdictWouldBeRed || got.Reason != ReasonGoBuildFailed {
		t.Fatalf("report = %+v, want WOULD_BE_RED/%s", got, ReasonGoBuildFailed)
	}
	if len(got.Runs) != 1 {
		t.Fatalf("runs = %d, want no verify run after failed warm", len(got.Runs))
	}
}

func TestPrepareWorkspaceOrdersLeaseBeforeBuildAndDevIndex(t *testing.T) {
	plan := PlanWorkspacePreflight(WorkspacePreflightInput{
		TaskID:       "2140",
		Actor:        "worker-b",
		ReadGlobs:    []string{"internal/preflight/**"},
		WriteGlobs:   []string{"internal/preflight/**"},
		PackageGraph: workspaceGraph(),
		TestProbe: testroute.Probe{
			GOOS:              "linux",
			NativeTestAllowed: true,
		},
	})
	prep := &fakeWorkspacePreparer{}
	got := PrepareWorkspace(context.Background(), plan, prep)

	if got.Verdict != WorkspaceVerdictReady {
		t.Fatalf("prepare verdict = %s reason=%s detail=%s", got.Verdict, got.Reason, got.Detail)
	}
	wantOrder := []string{StepAcquireWriteLease, StepWarmGoBuildCache, StepWarmDevIndex}
	if !reflect.DeepEqual(prep.calls, wantOrder) {
		t.Fatalf("preparer calls = %v, want %v", prep.calls, wantOrder)
	}
	if got.Lease == nil || !got.Lease.Held {
		t.Fatalf("lease result = %+v, want held", got.Lease)
	}
	if got.Build == nil || got.Build.Verdict != WorkspaceVerdictReady {
		t.Fatalf("build result = %+v, want ready", got.Build)
	}
}

func TestPrepareWorkspaceLeaseRefusalStopsBeforeBuild(t *testing.T) {
	plan := PlanWorkspacePreflight(WorkspacePreflightInput{
		WriteGlobs:   []string{"internal/preflight/**"},
		PackageGraph: workspaceGraph(),
		TestProbe:    testroute.Probe{GOOS: "linux", NativeTestAllowed: true},
	})
	prep := &fakeWorkspacePreparer{
		lease: LeaseAcquireResult{
			Held:     false,
			Reason:   ReasonCollisionRisk,
			Conflict: &LeaseObservation{ID: "peer", Holder: "peer", Tree: []string{"internal/preflight/**"}},
		},
	}
	got := PrepareWorkspace(context.Background(), plan, prep)
	if got.Verdict != WorkspaceVerdictBlockedByLease || got.Reason != ReasonCollisionRisk {
		t.Fatalf("prepare verdict/reason = %s/%s, want BLOCKED_BY_LEASE/%s", got.Verdict, got.Reason, ReasonCollisionRisk)
	}
	if !reflect.DeepEqual(prep.calls, []string{StepAcquireWriteLease}) {
		t.Fatalf("calls = %v, want only lease acquire", prep.calls)
	}
}

func TestPrepareWorkspaceBuildFailureIsWouldBeRed(t *testing.T) {
	plan := PlanWorkspacePreflight(WorkspacePreflightInput{
		WriteGlobs:   []string{"internal/preflight/**"},
		PackageGraph: workspaceGraph(),
		TestProbe:    testroute.Probe{GOOS: "linux", NativeTestAllowed: true},
	})
	prep := &fakeWorkspacePreparer{
		build: GoBuildWarmReport{Verdict: WorkspaceVerdictWouldBeRed, Reason: ReasonGoBuildFailed, Detail: "compile failed"},
	}
	got := PrepareWorkspace(context.Background(), plan, prep)
	if got.Verdict != WorkspaceVerdictWouldBeRed || got.Reason != ReasonGoBuildFailed {
		t.Fatalf("prepare verdict/reason = %s/%s, want WOULD_BE_RED/%s", got.Verdict, got.Reason, ReasonGoBuildFailed)
	}
}

type fakeGoBuildRunner struct {
	runs  []GoBuildRun
	err   error
	calls int
}

func (f *fakeGoBuildRunner) RunGoBuild(context.Context, GoBuildWarmRequest) (GoBuildRun, error) {
	f.calls++
	if len(f.runs) == 0 {
		return GoBuildRun{ExitCode: 0}, f.err
	}
	run := f.runs[0]
	f.runs = f.runs[1:]
	return run, f.err
}

type fakeWorkspacePreparer struct {
	calls []string
	lease LeaseAcquireResult
	build GoBuildWarmReport
	dev   WarmStepResult
	errAt string
}

func (f *fakeWorkspacePreparer) AcquireWriteLease(_ context.Context, req LeaseRequest) (LeaseAcquireResult, error) {
	f.calls = append(f.calls, StepAcquireWriteLease)
	if f.errAt == StepAcquireWriteLease {
		return LeaseAcquireResult{}, errors.New("lease boom")
	}
	if f.lease.Reason != "" || f.lease.Conflict != nil || f.lease.Detail != "" || f.lease.LeaseID != "" || f.lease.Held {
		return f.lease, nil
	}
	return LeaseAcquireResult{Held: true, LeaseID: req.ID}, nil
}

func (f *fakeWorkspacePreparer) WarmGoBuild(_ context.Context, req GoBuildWarmRequest) (GoBuildWarmReport, error) {
	f.calls = append(f.calls, StepWarmGoBuildCache)
	if f.errAt == StepWarmGoBuildCache {
		return GoBuildWarmReport{}, errors.New("build boom")
	}
	if f.build.Verdict != "" {
		return f.build, nil
	}
	return GoBuildWarmReport{Request: req, Verdict: WorkspaceVerdictReady, Warm: true, Runs: []GoBuildRun{{ExitCode: 0, ElapsedMS: 500}}}, nil
}

func (f *fakeWorkspacePreparer) WarmDevIndex(_ context.Context, req DevIndexWarmRequest) (WarmStepResult, error) {
	f.calls = append(f.calls, StepWarmDevIndex)
	if f.errAt == StepWarmDevIndex {
		return WarmStepResult{}, errors.New("devindex boom")
	}
	if f.dev.Reason != "" || f.dev.Detail != "" || f.dev.ElapsedMS != 0 {
		return f.dev, nil
	}
	return WarmStepResult{OK: true, ElapsedMS: int64(len(req.Globs))}, nil
}

func stepKinds(steps []PreparationStep) []string {
	out := make([]string, 0, len(steps))
	for _, step := range steps {
		out = append(out, step.Kind)
	}
	return out
}
