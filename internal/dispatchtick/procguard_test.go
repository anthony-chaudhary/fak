package dispatchtick

import "testing"

func procInfo(pid int, name string, threads int) ProcInfo {
	return ProcInfo{PID: pid, Name: name, Threads: IntPtr(threads)}
}

func TestProcGuardFlagsRunawayThreads(t *testing.T) {
	res := EvaluateProcGuard(ProcGuardInput{
		Processes: []ProcInfo{
			procInfo(38264, "llama-cli", 129427),
			procInfo(4, "System", 613),
			procInfo(113628, "python", 90),
		},
	})
	if res.OK || res.ActionableFlaggedCount != 1 || len(res.Flagged) != 1 {
		t.Fatalf("result = %+v, want one actionable flag", res)
	}
	if res.Flagged[0].PID != 38264 || res.Flagged[0].Protected {
		t.Fatalf("flagged = %+v, want non-protected llama-cli", res.Flagged[0])
	}
	if got := res.ActionableNames(); len(got) != 1 || got[0] != "llama-cli(pid 38264)" {
		t.Fatalf("actionable names = %#v", got)
	}
}

func TestProcGuardProtectedOnlyIsNotAction(t *testing.T) {
	res := EvaluateProcGuard(ProcGuardInput{
		Processes: []ProcInfo{procInfo(4, "System", 9000)},
	})
	if !res.OK || res.ActionableFlaggedCount != 0 || len(res.Flagged) != 1 {
		t.Fatalf("protected-only result = %+v, want ok with one non-actionable flag", res)
	}
	if !res.Flagged[0].Protected {
		t.Fatalf("protected flag missing: %+v", res.Flagged[0])
	}
}

func TestProcGuardTerminalHostIsNotActionable(t *testing.T) {
	// The terminal emulator that hosts the fleet's panes accumulates threads in
	// proportion to the panes it hosts; a breach must be reported but must never
	// flip Safe=false, or a busy terminal REFUSE_HOSTs the very fleet it hosts.
	res := EvaluateProcGuard(ProcGuardInput{
		Processes: []ProcInfo{procInfo(35544, "WindowsTerminal", 2284)},
	})
	if !res.OK || res.ActionableFlaggedCount != 0 || len(res.Flagged) != 1 {
		t.Fatalf("terminal-host result = %+v, want ok with one non-actionable flag", res)
	}
	if !res.Flagged[0].Protected {
		t.Fatalf("terminal host should be protected (reported, non-actionable): %+v", res.Flagged[0])
	}
	if got := res.ActionableNames(); len(got) != 0 {
		t.Fatalf("actionable names = %#v, want none for a terminal host", got)
	}
}

func TestProcGuardTerminalHostDoesNotMaskRealRunaway(t *testing.T) {
	// A thread-heavy terminal host sitting next to a genuine runaway must not
	// hide the runaway: the terminal is spared, the inference binary still flags.
	res := EvaluateProcGuard(ProcGuardInput{
		Processes: []ProcInfo{
			procInfo(35544, "WindowsTerminal", 4000),
			procInfo(38264, "llama-cli", 129427),
		},
	})
	if res.OK || res.ActionableFlaggedCount != 1 || len(res.Flagged) != 2 {
		t.Fatalf("mixed result = %+v, want one actionable flag among two", res)
	}
	if got := res.ActionableNames(); len(got) != 1 || got[0] != "llama-cli(pid 38264)" {
		t.Fatalf("actionable names = %#v, want only the runaway", got)
	}
}

func TestProcGuardMissingDimensionsAreSkipped(t *testing.T) {
	res := EvaluateProcGuard(ProcGuardInput{
		Processes: []ProcInfo{{PID: 7, Name: "unknown"}},
	})
	if !res.OK || len(res.Flagged) != 0 {
		t.Fatalf("missing thread dimension result = %+v, want clean", res)
	}
}

func TestProcGuardHandlesAndWorkingSetOptIn(t *testing.T) {
	handles := 50000
	ws := 40000
	row := ProcInfo{PID: 7, Name: "leaky", Threads: IntPtr(10), Handles: &handles, WorkingSetMB: &ws}
	res := EvaluateProcGuard(ProcGuardInput{Processes: []ProcInfo{row}})
	if !res.OK || len(res.Flagged) != 0 {
		t.Fatalf("default thresholds result = %+v, want clean", res)
	}
	res = EvaluateProcGuard(ProcGuardInput{
		Processes: []ProcInfo{row},
		Thresholds: ProcGuardThresholds{
			MaxThreads:      ProcGuardDefaultMaxThreads,
			MaxHandles:      10000,
			MaxWorkingSetMB: 8000,
		},
	})
	if res.OK || len(res.Flagged) != 1 || len(res.Flagged[0].Reasons) != 2 {
		t.Fatalf("opt-in thresholds result = %+v, want handle/ws flag", res)
	}
}

func TestProcGuardAllowlistAndProtectedPID(t *testing.T) {
	res := EvaluateProcGuard(ProcGuardInput{
		Processes:  []ProcInfo{procInfo(9, "BigDB", 50000)},
		AllowNames: []string{"bigdb"},
	})
	if !res.OK || len(res.Flagged) != 0 {
		t.Fatalf("allowlisted result = %+v, want clean", res)
	}

	res = EvaluateProcGuard(ProcGuardInput{
		Processes:     []ProcInfo{procInfo(123, "worker", 50000)},
		ProtectedPIDs: []int{123},
	})
	if !res.OK || len(res.Flagged) != 1 || !res.Flagged[0].Protected {
		t.Fatalf("protected pid result = %+v, want non-actionable protected flag", res)
	}
}

func TestProcGuardCollectErrorIsNotClean(t *testing.T) {
	res := EvaluateProcGuard(ProcGuardInput{CollectError: "scan boom"})
	if res.OK || res.CollectError != "scan boom" {
		t.Fatalf("collect error result = %+v, want not ok", res)
	}
}
