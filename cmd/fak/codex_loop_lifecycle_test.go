package main

import (
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/sessionregistry"
)

func loopDiagnosis(id string) codexLoopDiagnosis {
	return codexLoopDiagnosis{SessionID: id, Verdict: "LOOP"}
}

func TestClassifyCodexLoopLifecyclesMixed(t *testing.T) {
	old := readSessionRows
	readSessionRows = func() ([]sessionregistry.Record, error) {
		return []sessionregistry.Record{
			{State: sessionregistry.StateActive, Identity: sessionregistry.Identity{Runtime: "codex", SessionID: "live"}},
			{State: sessionregistry.StateCompleted, Identity: sessionregistry.Identity{Runtime: "codex", SessionID: "done"}},
		}, nil
	}
	t.Cleanup(func() { readSessionRows = old })
	got := classifyLoopStates(codexLoopRecentReport{Diagnoses: []codexLoopDiagnosis{loopDiagnosis("live"), loopDiagnosis("done"), loopDiagnosis("unknown")}})
	if len(got.Live) != 1 || got.Live[0] != "live" || len(got.Terminal) != 1 || got.Terminal[0] != "done" || len(got.Ambiguous) != 1 || got.Ambiguous[0] != "unknown" {
		t.Fatalf("lifecycle = %+v", got)
	}
}

func TestClassifyCodexLoopLifecyclesRegistryFailureIsAmbiguous(t *testing.T) {
	old := readSessionRows
	readSessionRows = func() ([]sessionregistry.Record, error) { return nil, errors.New("registry corrupt") }
	t.Cleanup(func() { readSessionRows = old })
	got := classifyLoopStates(codexLoopRecentReport{Diagnoses: []codexLoopDiagnosis{loopDiagnosis("x")}})
	if len(got.Ambiguous) != 1 || got.Ambiguous[0] != "x" {
		t.Fatalf("lifecycle = %+v", got)
	}
}

func TestCodexLoopLifecycleUsesLatestIndependentRegistration(t *testing.T) {
	rows := []sessionregistry.Record{
		{State: sessionregistry.StateActive, Identity: sessionregistry.Identity{Runtime: "codex", ThreadID: "x"}},
		{State: sessionregistry.StateFailed, Identity: sessionregistry.Identity{Runtime: "codex", ThreadID: "x"}},
	}
	if got := loopStateForSession(rows, "x"); got != loopStateTerminal {
		t.Fatalf("lifecycle = %q, want terminal", got)
	}
}
