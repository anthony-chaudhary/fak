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
