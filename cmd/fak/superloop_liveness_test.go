package main

import (
	"errors"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/superloop"
)

func TestKeepSuperloopAlivePrioritizesUntrackedWorkOverOpenIssues(t *testing.T) {
	old := superloopResidualCommand
	t.Cleanup(func() { superloopResidualCommand = old })
	superloopResidualCommand = func(_ string, name string, args ...string) ([]byte, error) {
		if name == "git" {
			return []byte("cmd/fak/new.go\x00docs/new.md\x00"), nil
		}
		return []byte(`[{"number":42},{"number":43}]`), nil
	}

	got, residual := keepSuperloopAlive(t.TempDir(), cleanDriveDecision())
	if got.Satisfied || !got.Enter || got.Member.Ref != "local-untracked-work" {
		t.Fatalf("decision = %#v, want an unsatisfied entered reconciliation member", got)
	}
	if got.Action != "go run ./cmd/fak sweep --json" {
		t.Fatalf("action = %q", got.Action)
	}
	if residual.UntrackedCount != 2 || residual.OpenIssues != 2 {
		t.Fatalf("residual = %#v", residual)
	}
}

func TestKeepSuperloopAliveDispatchesWhenOpenIssuesRemain(t *testing.T) {
	old := superloopResidualCommand
	t.Cleanup(func() { superloopResidualCommand = old })
	superloopResidualCommand = func(_ string, name string, args ...string) ([]byte, error) {
		if name == "git" {
			return nil, nil
		}
		return []byte(`[{"number":91},{"number":92}]`), nil
	}

	got, residual := keepSuperloopAlive(t.TempDir(), cleanDriveDecision())
	if got.Satisfied || !got.Enter || got.Member.Ref != "open-issue-backlog" {
		t.Fatalf("decision = %#v, want an unsatisfied entered dispatch member", got)
	}
	if got.Action != "go run ./cmd/fak dispatch sweep" {
		t.Fatalf("action = %q", got.Action)
	}
	if !reflect.DeepEqual(residual.IssueSample, []int{91, 92}) {
		t.Fatalf("issue sample = %v", residual.IssueSample)
	}
}

func TestKeepSuperloopAliveOnlyDeclaresDoneAfterBothSignalsDrain(t *testing.T) {
	old := superloopResidualCommand
	t.Cleanup(func() { superloopResidualCommand = old })
	superloopResidualCommand = func(_ string, name string, _ ...string) ([]byte, error) {
		if name == "git" {
			return nil, nil
		}
		return []byte("[]"), nil
	}

	got, residual := keepSuperloopAlive(t.TempDir(), cleanDriveDecision())
	if !got.Satisfied || got.Enter {
		t.Fatalf("decision = %#v, want original clean decision", got)
	}
	if !residual.Checked || !residual.IssueMeasured {
		t.Fatalf("residual = %#v", residual)
	}
}

func TestKeepSuperloopAliveRefusesDoneWhenOpenIssueMeasurementFails(t *testing.T) {
	old := superloopResidualCommand
	t.Cleanup(func() { superloopResidualCommand = old })
	superloopResidualCommand = func(_ string, name string, _ ...string) ([]byte, error) {
		if name == "git" {
			return nil, nil
		}
		return nil, errors.New("gh unavailable")
	}

	got, residual := keepSuperloopAlive(t.TempDir(), cleanDriveDecision())
	if got.Satisfied || got.Enter {
		t.Fatalf("decision = %#v, want unsatisfied wait without an executable member", got)
	}
	if residual.IssueMeasured || residual.MeasureError == "" {
		t.Fatalf("residual = %#v", residual)
	}
}

func cleanDriveDecision() superloop.DriveDecision {
	return superloop.DriveDecision{Intent: "run-it-all-night", Satisfied: true, Reason: "members clean"}
}
