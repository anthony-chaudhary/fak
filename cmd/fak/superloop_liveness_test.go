package main

import (
	"errors"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
	"github.com/anthony-chaudhary/fak/internal/fleetmetrics"
	"github.com/anthony-chaudhary/fak/internal/superloop"
)

func stubSuperloopCommitHealth(t *testing.T, metric fleetmetrics.CommitThroughput, active int) {
	t.Helper()
	oldMeasure, oldActive := superloopCommitThroughput, superloopActiveWorkers
	t.Cleanup(func() { superloopCommitThroughput, superloopActiveWorkers = oldMeasure, oldActive })
	superloopCommitThroughput = func(string, time.Time) fleetmetrics.CommitThroughput { return metric }
	superloopActiveWorkers = func(string, time.Time) int { return active }
}

func TestKeepSuperloopAliveRoutesZeroCommitRateBeforeDeclaringDone(t *testing.T) {
	stubResidualReads(t, nil, dispatchtick.RouterPayload{Coverage: dispatchtick.RouterCoverage{Complete: true}}, nil)
	stubSuperloopCommitHealth(t, fleetmetrics.CommitThroughput{Measured: true}, 3)
	got, residual := keepSuperloopAlive(t.TempDir(), cleanDriveDecision())
	if !got.Enter || got.Satisfied || got.Member.Ref != superloop.CommitRecoveryRef {
		t.Fatalf("decision=%+v", got)
	}
	if residual.CommitHealth.State != "blocked" || residual.ActiveWorkers != 3 {
		t.Fatalf("residual=%+v", residual)
	}
}

func TestKeepSuperloopAlivePrioritizesUntrackedWorkOverActionableIssues(t *testing.T) {
	stubResidualReads(t, []byte("cmd/fak/new.go\x00docs/new.md\x00"), routerWithQueues(2,
		dispatchtick.RouterRepairQueue{Kind: "dispatch", Count: 2, NextAction: "dispatch"}), nil)
	oldAttributor := superloopResidualAttributor
	superloopResidualAttributor = func(string, []string) (string, string) { return "OWNED_RECONCILE", "me" }
	t.Cleanup(func() { superloopResidualAttributor = oldAttributor })

	got, residual := keepSuperloopAlive(t.TempDir(), cleanDriveDecision())
	if got.Satisfied || !got.Enter || got.Member.Ref != "owned-untracked-work" {
		t.Fatalf("decision = %#v, want an unsatisfied entered reconciliation member", got)
	}
	if got.Action != "go run ./cmd/fak sweep --json" {
		t.Fatalf("action = %q", got.Action)
	}
	if residual.UntrackedCount != 2 || residual.ActionableIssues != 2 {
		t.Fatalf("residual = %#v", residual)
	}
}

func TestKeepSuperloopAliveRoutesUntrackedOwnershipClasses(t *testing.T) {
	cases := []struct {
		name, class, owner, ref, action string
		enter                           bool
	}{
		{name: "owned", class: "OWNED_RECONCILE", owner: "me", ref: "owned-untracked-work", action: "go run ./cmd/fak sweep --json", enter: true},
		{name: "peer", class: "PEER_ACTIVE", owner: "peer", enter: false},
		{name: "abandoned", class: "ABANDONED_RECOVER", owner: "dead", ref: "abandoned-untracked-work", action: "go run ./cmd/fak tree-doctor --json", enter: true},
		{name: "scratch", class: "SCRATCH_REAP", ref: "untracked-scratch", action: "go run ./cmd/fak tree-doctor --sweep-scratch --dry-run --json", enter: true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			stubResidualReads(t, []byte("cmd/fak/new.go\x00"), routerWithQueues(1, dispatchtick.RouterRepairQueue{Kind: "dispatch", Count: 1}), nil)
			old := superloopResidualAttributor
			superloopResidualAttributor = func(string, []string) (string, string) { return tc.class, tc.owner }
			t.Cleanup(func() { superloopResidualAttributor = old })
			got, residual := keepSuperloopAlive(t.TempDir(), cleanDriveDecision())
			if got.Satisfied || got.Enter != tc.enter || residual.UntrackedClass != tc.class || residual.UntrackedOwner != tc.owner {
				t.Fatalf("decision = %#v residual = %#v", got, residual)
			}
			if tc.enter && (got.Member.Ref != tc.ref || got.Action != tc.action) {
				t.Fatalf("decision = %#v, want ref=%s action=%s", got, tc.ref, tc.action)
			}
		})
	}
}

func TestKeepSuperloopAliveDispatchesActionableIssues(t *testing.T) {
	stubResidualReads(t, nil, routerWithQueues(3,
		dispatchtick.RouterRepairQueue{Kind: "dispatch", Count: 3, NextAction: "dispatch"}), nil)

	got, residual := keepSuperloopAlive(t.TempDir(), cleanDriveDecision())
	if got.Satisfied || !got.Enter || got.Member.Ref != "actionable-issue-backlog" {
		t.Fatalf("decision = %#v, want an unsatisfied entered dispatch member", got)
	}
	if got.Action != "go run ./cmd/fak dispatch sweep --live" || residual.NextQueue != "dispatch" {
		t.Fatalf("decision = %#v residual = %#v", got, residual)
	}
}

func TestKeepSuperloopAliveRepairsHeldBacklogInsteadOfSpinningDispatch(t *testing.T) {
	router := routerWithQueues(0,
		dispatchtick.RouterRepairQueue{Kind: "scope", Count: 7, NextAction: "add worker-ready scope"})
	router.Counts.SkippedHumanBlocked = 7
	stubResidualReads(t, nil, router, nil)

	got, residual := keepSuperloopAlive(t.TempDir(), cleanDriveDecision())
	if got.Satisfied || !got.Enter || got.Member.Ref != "issue-backlog-scope" {
		t.Fatalf("decision = %#v, want scope repair", got)
	}
	if got.Action != "go run ./cmd/fak-dev issue repair --live --json" || residual.NextQueue != "scope" {
		t.Fatalf("decision = %#v residual = %#v", got, residual)
	}
}

func TestKeepSuperloopAliveWaitsForWitnessedHumanOnlyBacklog(t *testing.T) {
	router := routerWithQueues(0,
		dispatchtick.RouterRepairQueue{Kind: "human", Count: 4, NextAction: "wait"})
	router.Counts.SkippedHumanBlocked = 4
	stubResidualReads(t, nil, router, nil)

	got, residual := keepSuperloopAlive(t.TempDir(), cleanDriveDecision())
	if got.Satisfied || got.Enter || residual.NextQueue != "human" {
		t.Fatalf("decision = %#v residual = %#v, want typed wait", got, residual)
	}
}

func TestKeepSuperloopAliveOnlyDeclaresDoneAfterBothSignalsDrain(t *testing.T) {
	stubResidualReads(t, nil, routerWithQueues(0), nil)

	got, residual := keepSuperloopAlive(t.TempDir(), cleanDriveDecision())
	if !got.Satisfied || got.Enter {
		t.Fatalf("decision = %#v, want original clean decision", got)
	}
	if !residual.Checked || !residual.IssueMeasured || !residual.CoverageComplete {
		t.Fatalf("residual = %#v", residual)
	}
}

func TestKeepSuperloopAliveRefusesDoneWhenRouterFailsOrCoverageTruncates(t *testing.T) {
	for _, tc := range []struct {
		name   string
		router dispatchtick.RouterPayload
		err    error
	}{
		{name: "fetch failure", err: errors.New("gh unavailable")},
		{name: "truncated", router: dispatchtick.RouterPayload{Coverage: dispatchtick.RouterCoverage{Complete: false}, Counts: dispatchtick.RouterCounts{Open: 1000}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			stubResidualReads(t, nil, tc.router, tc.err)
			got, residual := keepSuperloopAlive(t.TempDir(), cleanDriveDecision())
			if got.Satisfied || got.Enter {
				t.Fatalf("decision = %#v, want unsatisfied wait", got)
			}
			if residual.MeasureError == "" {
				t.Fatalf("residual = %#v", residual)
			}
		})
	}
}

func stubResidualReads(t *testing.T, gitOut []byte, router dispatchtick.RouterPayload, routerErr error) {
	t.Helper()
	stubSuperloopCommitHealth(t, fleetmetrics.CommitThroughput{Measured: true}, 0)
	oldCommand, oldRouter := superloopResidualCommand, superloopResidualRouter
	t.Cleanup(func() { superloopResidualCommand, superloopResidualRouter = oldCommand, oldRouter })
	superloopResidualCommand = func(_ string, name string, _ ...string) ([]byte, error) {
		if name != "git" {
			t.Fatalf("unexpected command %q", name)
		}
		return gitOut, nil
	}
	superloopResidualRouter = func(string) (dispatchtick.RouterPayload, error) { return router, routerErr }
}

func routerWithQueues(routed int, queues ...dispatchtick.RouterRepairQueue) dispatchtick.RouterPayload {
	return dispatchtick.RouterPayload{
		Coverage:     dispatchtick.RouterCoverage{Complete: true},
		Counts:       dispatchtick.RouterCounts{Open: routed, Routed: routed},
		RepairQueues: queues,
	}
}

func cleanDriveDecision() superloop.DriveDecision {
	return superloop.DriveDecision{Intent: "run-it-all-night", Satisfied: true, Reason: "members clean"}
}
