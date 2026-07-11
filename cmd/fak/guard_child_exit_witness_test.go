package main

import (
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
