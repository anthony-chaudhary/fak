package main

import (
	"errors"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/journal"
)

func TestAppendGuardChildExitWitnessRecordsCleanAndAbnormal(t *testing.T) {
	for _, tc := range []struct {
		name string
		err  error
		want string
		code int
	}{
		{name: "clean", want: journal.CrashCleanExit},
	} {
		t.Run(tc.name, func(t *testing.T) {
			j := journal.OpenMemory()
			j.AppendAgentEvent("HOOK_DECISION", "codex-loop-hook", "block")
			row := appendGuardChildExitWitness(j, "codex", "guard-4213", tc.err, nil, time.Now().Add(-1500*time.Millisecond))
			if row.Reason != tc.want || row.ChildExit == nil || row.ChildExit.WallTimeMS < 1000 || row.ChildExit.LastHook != "HOOK_DECISION:block" {
				t.Fatalf("exit row = %+v detail=%+v", row, row.ChildExit)
			}
		})
	}
}

func TestAppendGuardChildResourceExitWitnessUsesTypedReason(t *testing.T) {
	j := journal.OpenMemory()
	j.AppendAgentEvent("HOOK_DECISION", "codex-loop-hook", "block")
	row := appendGuardChildExitWitnessWithReason(
		j,
		"codex",
		"guard-9195",
		errors.New("child resource limit: CHILD_RESOURCE_MONITOR_ERROR: private dynamic detail"),
		nil,
		time.Now().Add(-1500*time.Millisecond),
		"CHILD_RESOURCE_MONITOR_ERROR",
	)
	if row.Kind != journal.KindChildCrash || row.Reason != "CHILD_RESOURCE_MONITOR_ERROR" {
		t.Fatalf("resource exit row kind/reason = %q/%q", row.Kind, row.Reason)
	}
	if row.ChildExit == nil || row.ChildExit.WallTimeMS < 1000 || row.ChildExit.LastHook != "HOOK_DECISION:block" {
		t.Fatalf("resource exit detail = %+v", row.ChildExit)
	}
	if _, err := journal.VerifyRows(j.Recent(0)); err != nil {
		t.Fatalf("resource exit row broke audit chain: %v", err)
	}
}
