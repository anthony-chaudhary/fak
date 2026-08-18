package main

import "testing"

func TestDispatchMergeMarkedWorkerPIDsIgnoresAmbientCodexUI(t *testing.T) {
	rows := []dispatchCodexProcessRow{
		{PID: 10, PPID: 1, Name: "codex.exe", Cmdline: `codex`},
		{PID: 20, PPID: 1, Name: "node.exe", Cmdline: `node C:\npm\@openai\codex\bin\codex.js`},
		{PID: 21, PPID: 20, Name: "codex.exe", Cmdline: `codex`},
	}
	got := dispatchMergeMarkedWorkerPIDs(map[int]bool{}, "codex", rows)
	if len(got) != 0 {
		t.Fatalf("ambient Codex UI PIDs counted as fleet workers: %v", got)
	}
}

func TestDispatchMergeMarkedWorkerPIDsCountsBackedAndMarkedOnce(t *testing.T) {
	rows := []dispatchCodexProcessRow{
		{PID: 100, PPID: 1, Name: "fak.exe", Cmdline: `fak guard -- codex`},
		{PID: 110, PPID: 100, Name: "node.exe", Cmdline: `node C:\npm\@openai\codex\bin\codex.js`},
		{PID: 111, PPID: 110, Name: "codex.exe", Cmdline: `codex`},
		{PID: 200, PPID: 1, Name: "codex.exe", Cmdline: `codex exec resolve GitHub issue #7827`},
		{PID: 300, PPID: 1, Name: "codex.exe", Cmdline: `codex`},
	}
	got := dispatchMergeMarkedWorkerPIDs(map[int]bool{100: true}, "codex", rows)
	if len(got) != 2 || !got[100] || !got[200] {
		t.Fatalf("worker PIDs = %v, want backed PID 100 and marker PID 200 exactly once", got)
	}
}
