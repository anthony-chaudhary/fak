package repoguard

import (
	"strings"
	"testing"
)

func TestClassifySleepWaitFlagsLongForms(t *testing.T) {
	cases := []string{
		"sleep 300",
		"sleep 1500; echo TIMER_DONE_POLL_QWEN", // session f496fd6f audit form
		"for i in $(seq 1 16); do sleep 300; curl -s host; done", // session a150a06c audit form
		"while sleep 300; do probe; done",                        // sleep as the loop condition
		"sleep 5m",                                               // GNU suffix
		"sleep 90 60",                                            // summed operands
		"sleep infinity",                                         // parses as +Inf; a certain hang
		"Start-Sleep -Seconds 300",
		"Start-Sleep -s 300",       // PowerShell prefix matching
		"Start-Sleep -Seconds:300", // colon form
		"start-sleep 300",          // positional form
		"Start-Sleep -Milliseconds 300000",
	}
	for _, cmd := range cases {
		vs := ClassifySleepWait(cmd)
		if len(vs) != 1 {
			t.Errorf("ClassifySleepWait(%q) = %d violations, want 1: %+v", cmd, len(vs), vs)
			continue
		}
		if vs[0].Reason != ReasonForegroundSleep {
			t.Errorf("ClassifySleepWait(%q) reason = %q, want %q", cmd, vs[0].Reason, ReasonForegroundSleep)
		}
		if !strings.Contains(vs[0].Fix, "run_in_background") {
			t.Errorf("ClassifySleepWait(%q) fix should name the background alternatives: %q", cmd, vs[0].Fix)
		}
	}
}

func TestClassifySleepWaitPassesShortAndUnresolvable(t *testing.T) {
	cases := []string{
		"sleep 5",
		"sleep 45; make ci", // below threshold even chained
		"for i in $(seq 1 16); do sleep 45; probe; done", // short per-invocation
		"sleep $T",               // unresolvable — never guess
		"sleep 300 &",            // backgrounded — does not hold the turn
		"sleep 300 2>&1 &",       // redirection & plus job-control &
		"(sleep 600 && probe) &", // backgrounded subshell poll
		"Start-Sleep -Seconds 5",
		"Start-Sleep -Milliseconds 500",
		"Start-Sleep -Seconds $wait",
		"timeout 300 go test ./...", // a bounded real command, not a timer
		"echo sleep 300 is slow | cat",
		"grep -r 'Start-Sleep -Seconds 300' .",
	}
	for _, cmd := range cases {
		if vs := ClassifySleepWait(cmd); len(vs) != 0 {
			t.Errorf("ClassifySleepWait(%q) = %+v, want none", cmd, vs)
		}
	}
}

func TestEvaluateWiresSleepWaitForBashAndPowerShell(t *testing.T) {
	for _, tool := range []string{"Bash", "PowerShell"} {
		vs := Evaluate(tool, map[string]any{"command": "sleep 300"}, "C:/w/fak", nil)
		if len(vs) != 1 || vs[0].Reason != ReasonForegroundSleep {
			t.Errorf("Evaluate(%s, sleep 300) = %+v, want one FOREGROUND_SLEEP", tool, vs)
		}
	}
	if vs := Evaluate("PowerShell", map[string]any{"command": "Start-Sleep -Seconds 600"}, "C:/w/fak", nil); len(vs) != 1 {
		t.Errorf("Evaluate(PowerShell, Start-Sleep 600) = %+v, want one violation", vs)
	}
}

func TestIsAdvisoryReason(t *testing.T) {
	if !IsAdvisoryReason(ReasonForegroundSleep) {
		t.Fatal("FOREGROUND_SLEEP must be advisory")
	}
	for _, denying := range []string{ReasonInteractiveHang, "OUT_OF_TREE_WRITE"} {
		if IsAdvisoryReason(denying) {
			t.Fatalf("%s must NOT be advisory", denying)
		}
	}
}

func TestRenderReasonIncludesSleepBlock(t *testing.T) {
	vs := ClassifySleepWait("sleep 300")
	reason := RenderReason(vs)
	if !strings.Contains(reason, ReasonForegroundSleep) || !strings.Contains(reason, "run_in_background") {
		t.Fatalf("RenderReason(sleep) = %q, want the FOREGROUND_SLEEP block with the fix", reason)
	}
}
