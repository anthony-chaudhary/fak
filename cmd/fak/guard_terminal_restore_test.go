package main

import "testing"

type testGuardChildTerminalRestore struct {
	repairs int
}

func (r *testGuardChildTerminalRestore) RepairAfterStart() bool {
	r.repairs++
	return true
}

func TestGuardChildTerminalRestoreCaptureIncludesClaude(t *testing.T) {
	original := captureGuardChildTerminalRestore
	t.Cleanup(func() { captureGuardChildTerminalRestore = original })

	for _, tc := range []struct {
		name    string
		command []string
		want    int
	}{
		{name: "empty", command: nil, want: 0},
		{name: "other harness", command: []string{"other-agent"}, want: 0},
		{name: "Claude", command: []string{"claude"}, want: 1},
		{name: "Claude Windows launcher", command: []string{`C:\tools\claude.exe`}, want: 1},
		{name: "Codex unchanged", command: []string{"codex", "exec"}, want: 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			captures := 0
			repair := &testGuardChildTerminalRestore{}
			captureGuardChildTerminalRestore = func() guardChildTerminalRestore {
				captures++
				return repair
			}

			got := maybeCaptureGuardChildHarnessTerminalRestore(tc.command)
			if captures != tc.want {
				t.Fatalf("capture count = %d, want %d", captures, tc.want)
			}
			if got == nil {
				t.Fatal("capture returned nil repair")
			}
			if repair.repairs != 0 {
				t.Fatalf("capture repaired before child start %d times", repair.repairs)
			}
		})
	}
}
