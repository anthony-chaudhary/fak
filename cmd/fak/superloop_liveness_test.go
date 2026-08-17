package main

import (
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
	"github.com/anthony-chaudhary/fak/internal/superloop"
)

func TestKeepSuperloopAlivePrioritizesUntrackedWorkOverActionableIssues(t *testing.T) {
	stubResidualReads(t, []byte("cmd/fak/new.go\x00docs/new.md\x00"), routerWithQueues(2,
		dispatchtick.RouterRepairQueue{Kind: "dispatch", Count: 2, NextAction: "dispatch"}), nil)

	got, residual := keepSuperloopAlive(t.TempDir(), cleanDriveDecision())
	if got.Satisfied || !got.Enter || got.Member.Ref != "local-untracked-work" {
		t.Fatalf("decision = %#v, want an unsatisfied entered reconciliation member", got)
	}
	if got.Action != "go run ./cmd/fak sweep --json" {
		t.Fatalf("action = %q", got.Action)
	}
	if residual.UntrackedCount != 2 || residual.ActionableIssues != 2 {
		t.Fatalf("residual = %#v", residual)
	}
}

func TestKeepSuperloopAliveDispatchesActionableIssues(t *testing.T) {
	stubResidualReads(t, nil, routerWithQueues(3,
		dispatchtick.RouterRepairQueue{Kind: "dispatch", Count: 3, NextAction: "dispatch"}), nil)

	got, residual := keepSuperloopAlive(t.TempDir(), cleanDriveDecision())
	if got.Satisfied || !got.Enter || got.Member.Ref != "actionable-issue-backlog" {
		t.Fatalf("decision = %#v, want an unsatisfied entered dispatch member", got)
	}
	if got.Action != "go run ./cmd/fak dispatch sweep" || residual.NextQueue != "dispatch" {
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
	if got.Action != "go run ./cmd/fak-dev issue repair --json" || residual.NextQueue != "scope" {
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
