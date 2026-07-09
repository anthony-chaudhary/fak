package stallscan

import "testing"

// The fixtures below are pinned to LIVE captures from the reference box
// (a Windows fleet host, 2026-07-07) during and between desktop freezes. Each asserts
// that Classify reaches the verdict the human investigation reached from the raw
// counters, so a future change to the thresholds that would re-open the
// misclassification is caught here.

func TestClassify_SoftFaultChurn_isStallNotDisk(t *testing.T) {
	// Live stall frame: ~490k faults/sec, but only ~508 page-reads/sec (hard),
	// 229 GB RAM free, disk queue 0. The human read this as SOFT-fault churn,
	// explicitly NOT disk and NOT memory. The classifier must agree.
	s := Sample{
		TotalFaultsPerSec:      489505,
		HardFaultsPerSec:       508,
		DemandZeroFaultsPerSec: 192374,
		TransitionFaultsPerSec: 136422,
		ContextSwitchesPerSec:  24000,
		AvailableMB:            229257,
		DiskQueueLen:           0,
	}
	v := Classify(s, DefaultThresholds())
	if v.Level != LevelStall {
		t.Fatalf("level = %q, want stall", v.Level)
	}
	if v.Cause != CauseSoftFault {
		t.Fatalf("cause = %q, want %q", v.Cause, CauseSoftFault)
	}
}

func TestClassify_SpawnStorm_winsAttribution(t *testing.T) {
	// Live frame: +9 processes in one 2s interval, context switches 15k->106k,
	// syscalls 165k->733k. The +9 spawn burst is the specific, actionable cause.
	s := Sample{
		TotalFaultsPerSec:     120000,
		HardFaultsPerSec:      50,
		ContextSwitchesPerSec: 106000,
		SystemCallsPerSec:     733000,
		ProcessDelta:          9,
		SpawnBurst:            9,
		AvailableMB:           229000,
	}
	v := Classify(s, DefaultThresholds())
	if v.Level != LevelStall {
		t.Fatalf("level = %q, want stall", v.Level)
	}
	if v.Cause != CauseSpawnStorm {
		t.Fatalf("cause = %q, want %q (spawn burst is the most specific cause)", v.Cause, CauseSpawnStorm)
	}
}

func TestClassify_DiskDominated_isDiskNotSoftChurn(t *testing.T) {
	// A genuinely disk-backed fault storm: high total faults AND a large hard
	// fraction. Must be attributed to disk, not soft churn — this is the guard
	// that stops us mislabeling a real disk stall.
	s := Sample{
		TotalFaultsPerSec: 300000,
		HardFaultsPerSec:  90000, // 30% hard — disk is really being read
		AvailableMB:       200000,
		DiskQueueLen:      1,
	}
	v := Classify(s, DefaultThresholds())
	if v.Cause != CauseDiskIO {
		t.Fatalf("cause = %q, want %q", v.Cause, CauseDiskIO)
	}
}

func TestClassify_DiskQueueBusy_isDisk(t *testing.T) {
	s := Sample{TotalFaultsPerSec: 5000, DiskQueueLen: 6, AvailableMB: 100000}
	v := Classify(s, DefaultThresholds())
	if v.Level != LevelStall || v.Cause != CauseDiskIO {
		t.Fatalf("got level=%q cause=%q, want stall/disk_io", v.Level, v.Cause)
	}
}

func TestClassify_MemoryPressure_isMemNotChurn(t *testing.T) {
	// Low RAM must win over any churn signal — correct attribution matters.
	s := Sample{TotalFaultsPerSec: 500000, HardFaultsPerSec: 100, AvailableMB: 800}
	v := Classify(s, DefaultThresholds())
	if v.Level != LevelStall || v.Cause != CauseMemPressure {
		t.Fatalf("got level=%q cause=%q, want stall/memory_pressure", v.Level, v.Cause)
	}
}

func TestClassify_Calm_lowEverything(t *testing.T) {
	// A quiet frame between bursts: modest faults, RAM free, no spawn. Calm.
	s := Sample{
		TotalFaultsPerSec:     22297,
		HardFaultsPerSec:      10,
		ContextSwitchesPerSec: 20241,
		AvailableMB:           229000,
	}
	v := Classify(s, DefaultThresholds())
	if v.Level != LevelCalm {
		t.Fatalf("level = %q, want calm (reasons: %v)", v.Level, v.Reasons)
	}
}

func TestClassify_Elevated_band(t *testing.T) {
	// Above the elevated fault floor but below stall, nothing else tripping.
	s := Sample{TotalFaultsPerSec: 200000, HardFaultsPerSec: 100, AvailableMB: 229000}
	v := Classify(s, DefaultThresholds())
	if v.Level != LevelElevated {
		t.Fatalf("level = %q, want elevated", v.Level)
	}
}

func TestClassify_TopProcessAttribution(t *testing.T) {
	s := Sample{
		TotalFaultsPerSec: 5000,
		AvailableMB:       229000,
		TopIO: []ProcIO{
			{PID: 6028, Name: "MsMpEng.exe", Ops: 7423},
			{PID: 18456, Name: "AUEPMaster.exe", Ops: 248187},
			{PID: 23056, Name: "CC_Engine_x64.exe", Ops: 2885},
		},
	}
	v := Classify(s, DefaultThresholds())
	if v.TopProcess != "AUEPMaster.exe" || v.TopPID != 18456 {
		t.Fatalf("top = %q/%d, want AUEPMaster.exe/18456", v.TopProcess, v.TopPID)
	}
	if v.TopOps != 248187 {
		t.Fatalf("top ops = %v, want 248187", v.TopOps)
	}
}

func TestClassify_HandleLeak_raisesCalmToElevated(t *testing.T) {
	// Live capture (2026-07-08..11): an otherwise CALM box — low faults, RAM free,
	// disk idle — but WindowsTerminal has accreted 33,418 handles (climbing across
	// days). A leak is a slow-burn warning, so the verdict is ELEVATED (not a
	// stall) with cause handle_leak, and the culprit is named.
	s := Sample{
		TotalFaultsPerSec: 22000,
		HardFaultsPerSec:  10,
		AvailableMB:       135000,
		DiskQueueLen:      0,
		SystemHandleTotal: 510974,
		TopHandles: []ProcHandles{
			{PID: 17836, Name: "WindowsTerminal.exe", Handles: 33418},
			{PID: 4, Name: "System", Handles: 19005},
			{PID: 8492, Name: "svchost.exe", Handles: 14135},
		},
	}
	v := Classify(s, DefaultThresholds())
	if v.Level != LevelElevated {
		t.Fatalf("level = %q, want elevated (reasons: %v)", v.Level, v.Reasons)
	}
	if v.Cause != CauseHandleLeak {
		t.Fatalf("cause = %q, want %q", v.Cause, CauseHandleLeak)
	}
	if v.HandleLeakProcess != "WindowsTerminal.exe" || v.HandleLeakPID != 17836 || v.HandleLeakCount != 33418 {
		t.Fatalf("leak attribution = %q/%d/%d, want WindowsTerminal.exe/17836/33418",
			v.HandleLeakProcess, v.HandleLeakPID, v.HandleLeakCount)
	}
}

func TestClassify_HandleLeak_attributedUnderChurnStall(t *testing.T) {
	// A soft-fault churn stall AND a handle leak at once (the real live frame).
	// The freeze cause wins the Level/Cause (it is the urgent one), but the leak
	// must still be attributed so the operator sees both problems.
	s := Sample{
		TotalFaultsPerSec:      1241214,
		HardFaultsPerSec:       39,
		DemandZeroFaultsPerSec: 554729,
		TransitionFaultsPerSec: 217729,
		ContextSwitchesPerSec:  484244,
		SystemCallsPerSec:      1806032,
		AvailableMB:            135331,
		DiskQueueLen:           0,
		TopHandles:             []ProcHandles{{PID: 17836, Name: "WindowsTerminal.exe", Handles: 33418}},
	}
	v := Classify(s, DefaultThresholds())
	if v.Level != LevelStall || v.Cause != CauseSoftFault {
		t.Fatalf("got level=%q cause=%q, want stall/soft_fault_churn (freeze wins)", v.Level, v.Cause)
	}
	if v.HandleLeakProcess != "WindowsTerminal.exe" || v.HandleLeakCount != 33418 {
		t.Fatalf("leak not attributed under stall: %q/%d", v.HandleLeakProcess, v.HandleLeakCount)
	}
}

func TestClassify_HandleBelowThreshold_noLeak(t *testing.T) {
	// The top holder is under the 10k line — a normal busy process, not a leak.
	// No attribution, and the box stays calm.
	s := Sample{
		TotalFaultsPerSec: 22000,
		AvailableMB:       135000,
		TopHandles:        []ProcHandles{{PID: 100, Name: "fak.exe", Handles: 3241}},
	}
	v := Classify(s, DefaultThresholds())
	if v.Level != LevelCalm {
		t.Fatalf("level = %q, want calm", v.Level)
	}
	if v.HandleLeakProcess != "" {
		t.Fatalf("unexpected leak attribution %q for a 3241-handle process", v.HandleLeakProcess)
	}
}

func TestClassify_SystemHandleHigh_elevatesCalm(t *testing.T) {
	// No single-process leak, but the system-wide handle total crosses the coarse
	// ceiling — a calm box is raised to elevated with cause handle_leak.
	s := Sample{
		TotalFaultsPerSec: 22000,
		AvailableMB:       135000,
		SystemHandleTotal: 1_050_000,
		TopHandles:        []ProcHandles{{PID: 4, Name: "System", Handles: 9000}}, // below per-proc line
	}
	v := Classify(s, DefaultThresholds())
	if v.Level != LevelElevated || v.Cause != CauseHandleLeak {
		t.Fatalf("got level=%q cause=%q, want elevated/handle_leak", v.Level, v.Cause)
	}
	if v.HandleLeakProcess != "" {
		t.Fatalf("no per-process suspect expected, got %q", v.HandleLeakProcess)
	}
}

func TestWorstHandleHog_picksMaxAtOrAboveThreshold(t *testing.T) {
	in := []ProcHandles{
		{PID: 1, Name: "a", Handles: 9000},  // below
		{PID: 2, Name: "b", Handles: 33418}, // worst
		{PID: 3, Name: "c", Handles: 12000},
	}
	got, ok := worstHandleHog(in, 10000)
	if !ok || got.PID != 2 || got.Handles != 33418 {
		t.Fatalf("got %+v ok=%v, want b/33418", got, ok)
	}
	if _, ok := worstHandleHog(in, 0); ok {
		t.Fatalf("threshold 0 must disable the rule")
	}
	if _, ok := worstHandleHog([]ProcHandles{{PID: 1, Handles: 500}}, 10000); ok {
		t.Fatalf("no process above threshold must return ok=false")
	}
}

func TestSortTopIO_capsAndOrders(t *testing.T) {
	in := []ProcIO{
		{PID: 1, Name: "a", Ops: 10},
		{PID: 2, Name: "b", Ops: 500},
		{PID: 3, Name: "c", Ops: 100},
	}
	got := SortTopIO(in, 2)
	if len(got) != 2 || got[0].Name != "b" || got[1].Name != "c" {
		t.Fatalf("got %+v, want [b,c]", got)
	}
}

func TestClassify_isPure_noMutation(t *testing.T) {
	// Classify must not mutate the caller's TopIO slice order.
	in := []ProcIO{{PID: 1, Ops: 10}, {PID: 2, Ops: 999}}
	s := Sample{TopIO: in, AvailableMB: 229000}
	_ = Classify(s, DefaultThresholds())
	if in[0].PID != 1 || in[1].PID != 2 {
		t.Fatalf("Classify mutated caller slice: %+v", in)
	}
}
