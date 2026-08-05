package stallscan

import (
	"encoding/json"
	"testing"
)

// A sample carrying a single worst handle hog and a single worst thread hog, at
// the counts each test wants to exercise.
func rebootSample(handles, threads int) Sample {
	return Sample{
		TopHandles: []ProcHandles{{PID: 4242, Name: "WindowsTerminal.exe", Handles: handles}},
		TopThreads: []ProcThreads{{PID: 4242, Name: "WindowsTerminal.exe", Threads: threads}},
	}
}

func TestAdviseReboot_belowBothLines_noPage(t *testing.T) {
	// A leak SUSPECT (over the 10k/500 leak line) but under the reboot high-water
	// must NOT advise a reboot — the leak axis warns for days, this axis pages.
	got := AdviseReboot(rebootSample(15000, 900), DefaultRebootThresholds())
	if got.Advised {
		t.Fatalf("15000 handles / 900 threads is under the reboot high-water; want no advice, got %+v", got)
	}
}

func TestAdviseReboot_handleHighWater_pages(t *testing.T) {
	got := AdviseReboot(rebootSample(31000, 100), DefaultRebootThresholds())
	if !got.Advised {
		t.Fatalf("31000 handles crosses the 30k high-water; want advice, got %+v", got)
	}
	if got.Axis != "handle_high_water" {
		t.Fatalf("axis = %q, want handle_high_water", got.Axis)
	}
	if got.Process != "WindowsTerminal.exe" || got.PID != 4242 || got.Count != 31000 || got.Threshold != 30000 {
		t.Fatalf("attribution wrong: %+v", got)
	}
	if got.Reason == "" {
		t.Fatalf("advice must carry an operator-facing reason")
	}
}

func TestAdviseReboot_threadOnly_pagesThreadAxis(t *testing.T) {
	// Handles under their line, threads over theirs: the thread axis pages.
	got := AdviseReboot(rebootSample(5000, 2100), DefaultRebootThresholds())
	if !got.Advised || got.Axis != "thread_high_water" {
		t.Fatalf("want thread_high_water advice, got %+v", got)
	}
	if got.Count != 2100 || got.Threshold != 2000 {
		t.Fatalf("thread attribution wrong: %+v", got)
	}
}

func TestAdviseReboot_bothCross_handleWins(t *testing.T) {
	// When both axes cross, handles win the label — pool exhaustion is closest to
	// a hard freeze, the same precedence Classify uses for the leak Cause.
	got := AdviseReboot(rebootSample(40000, 3000), DefaultRebootThresholds())
	if got.Axis != "handle_high_water" {
		t.Fatalf("both cross; handles must win, got axis %q (%+v)", got.Axis, got)
	}
}

func TestAdviseReboot_atExactLine_pages(t *testing.T) {
	// The line is inclusive (>=), matching worstHandleHog's boundary.
	got := AdviseReboot(rebootSample(30000, 0), DefaultRebootThresholds())
	if !got.Advised || got.Axis != "handle_high_water" {
		t.Fatalf("exactly 30000 handles must page (>= boundary), got %+v", got)
	}
}

func TestAdviseReboot_zeroThreshold_disablesAxis(t *testing.T) {
	// A 0 line disables that axis: a huge handle count is ignored when the handle
	// high-water is 0, and the thread axis still governs.
	got := AdviseReboot(rebootSample(99000, 2100), RebootThresholds{HandleHighWater: 0, ThreadHighWater: 2000})
	if got.Axis != "thread_high_water" {
		t.Fatalf("handle axis disabled by 0 line; thread axis should govern, got %+v", got)
	}
	none := AdviseReboot(rebootSample(99000, 99000), RebootThresholds{})
	if none.Advised {
		t.Fatalf("both lines 0 disables all axes; want no advice, got %+v", none)
	}
}

func TestAdviseReboot_twoIndependentCrossers_bothNamed(t *testing.T) {
	// The measured 2026-07-13 shape on the RDP reference box: a WindowsTerminal at
	// 33,054 handles AND a TermService svchost that crossed the same line on its
	// own. The single-max verdict named only the terminal and masked the svchost,
	// so the operator saw one driver of a two-driver reboot (#4614).
	s := Sample{TopHandles: []ProcHandles{
		{PID: 9001, Name: "svchost.exe", Handles: 31200},
		{PID: 4242, Name: "WindowsTerminal.exe", Handles: 33054},
		{PID: 77, Name: "explorer.exe", Handles: 2100}, // under the line: not a crosser
	}}
	got := AdviseReboot(s, DefaultRebootThresholds())
	if !got.Advised || got.Axis != "handle_high_water" || got.Process != "WindowsTerminal.exe" || got.PID != 4242 || got.Count != 33054 {
		t.Fatalf("headline must stay the worst hog: %+v", got)
	}
	if len(got.Crossers) != 2 {
		t.Fatalf("two processes crossed the high-water; want both in the crosser set, got %d: %+v", len(got.Crossers), got.Crossers)
	}
	if got.Crossers[0].Process != "WindowsTerminal.exe" || got.Crossers[0].Count != 33054 {
		t.Fatalf("crosser set must lead with the headline, got %+v", got.Crossers[0])
	}
	second := got.Crossers[1]
	if second.Process != "svchost.exe" || second.PID != 9001 || second.Count != 31200 || second.Axis != "handle_high_water" || !second.Advised {
		t.Fatalf("the masked svchost must be named with its own attribution, got %+v", second)
	}
	if second.Reason == "" {
		t.Fatalf("every crosser needs its own operator-facing reason: %+v", second)
	}
	if len(got.Crossers[0].Crossers) != 0 || len(second.Crossers) != 0 {
		t.Fatalf("the crosser set is exactly one level deep; nested sets would recurse: %+v", got.Crossers)
	}
	sec := got.SecondaryCrossers()
	if len(sec) != 1 || sec[0].Process != "svchost.exe" {
		t.Fatalf("SecondaryCrossers must drop the headline and keep the rest, got %+v", sec)
	}
}

func TestAdviseReboot_singleCrosser_recordUnchanged(t *testing.T) {
	// Back-compat is load-bearing: one crosser must render exactly the record it
	// rendered before Crossers existed, so single-hog consumers (stallpage,
	// operatorbrief, the --json block) are untouched.
	got := AdviseReboot(rebootSample(31000, 100), DefaultRebootThresholds())
	if len(got.Crossers) != 0 {
		t.Fatalf("one crossing process must leave the crosser set unset, got %+v", got.Crossers)
	}
	if len(got.SecondaryCrossers()) != 0 {
		t.Fatalf("a lone crosser has no secondaries, got %+v", got.SecondaryCrossers())
	}
	// The pre-#4614 record shape, spelled out so this is a contract and not a
	// restatement of whatever the struct happens to be now.
	type preCrosserRecord struct {
		Advised   bool   `json:"advised"`
		Axis      string `json:"axis,omitempty"`
		Process   string `json:"process,omitempty"`
		PID       int    `json:"pid,omitempty"`
		Count     int    `json:"count,omitempty"`
		Threshold int    `json:"threshold,omitempty"`
		Reason    string `json:"reason,omitempty"`
	}
	want, err := json.Marshal(preCrosserRecord{got.Advised, got.Axis, got.Process, got.PID, got.Count, got.Threshold, got.Reason})
	if err != nil {
		t.Fatal(err)
	}
	have, err := json.Marshal(got)
	if err != nil {
		t.Fatal(err)
	}
	if string(have) != string(want) {
		t.Fatalf("single-hog JSON must be byte-for-byte unchanged\n got: %s\nwant: %s", have, want)
	}
}

func TestAdviseReboot_crossersSpanBothAxes(t *testing.T) {
	// A handle crosser and a thread-only crosser are two independent drivers. The
	// process over BOTH lines is one process to reboot away, so it is listed once,
	// on the winning handle axis.
	s := Sample{
		TopHandles: []ProcHandles{{PID: 4242, Name: "WindowsTerminal.exe", Handles: 33054}},
		TopThreads: []ProcThreads{
			{PID: 4242, Name: "WindowsTerminal.exe", Threads: 2400},
			{PID: 9001, Name: "svchost.exe", Threads: 2100},
		},
	}
	got := AdviseReboot(s, DefaultRebootThresholds())
	if got.Axis != "handle_high_water" || got.PID != 4242 {
		t.Fatalf("handle axis must still win the headline, got %+v", got)
	}
	if len(got.Crossers) != 2 {
		t.Fatalf("want the terminal once plus the thread-only svchost, got %+v", got.Crossers)
	}
	if got.Crossers[1].Axis != "thread_high_water" || got.Crossers[1].Process != "svchost.exe" || got.Crossers[1].Count != 2100 || got.Crossers[1].Threshold != 2000 {
		t.Fatalf("the thread-axis crosser is mis-attributed: %+v", got.Crossers[1])
	}

	// A disabled axis contributes no crossers, exactly as it contributes no headline.
	off := AdviseReboot(s, RebootThresholds{HandleHighWater: 0, ThreadHighWater: 2000})
	if len(off.Crossers) != 2 || off.Axis != "thread_high_water" || off.Crossers[0].PID != 4242 || off.Crossers[1].PID != 9001 {
		t.Fatalf("handle axis off: both thread crossers govern, worst first, got %+v (%+v)", off, off.Crossers)
	}
	none := AdviseReboot(s, RebootThresholds{})
	if none.Advised || len(none.Crossers) != 0 {
		t.Fatalf("both lines 0 disables every axis, got %+v", none)
	}
}

func TestAdviseReboot_emptyCensus_noPage(t *testing.T) {
	got := AdviseReboot(Sample{}, DefaultRebootThresholds())
	if got.Advised {
		t.Fatalf("empty census cannot advise a reboot, got %+v", got)
	}
}
