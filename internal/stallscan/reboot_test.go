package stallscan

import "testing"

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

func TestAdviseReboot_emptyCensus_noPage(t *testing.T) {
	got := AdviseReboot(Sample{}, DefaultRebootThresholds())
	if got.Advised {
		t.Fatalf("empty census cannot advise a reboot, got %+v", got)
	}
}
