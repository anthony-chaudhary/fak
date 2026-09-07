package stallscan

import (
	"strings"
	"testing"
)

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

func TestClassify_ThreadLeak_raisesCalmToElevated(t *testing.T) {
	// Live capture (2026-07-11): an otherwise CALM box, but WindowsTerminal has
	// accreted 1,577 threads (a thread per PTY/render across a spawn-heavy fleet) —
	// the "terminal thread lag" the operator kept reporting. A thread leak is a
	// slow-burn warning: verdict ELEVATED (not a stall), cause thread_leak, named.
	s := Sample{
		TotalFaultsPerSec: 22000,
		HardFaultsPerSec:  10,
		AvailableMB:       126000,
		DiskQueueLen:      0,
		TopThreads: []ProcThreads{
			{PID: 17836, Name: "WindowsTerminal.exe", Threads: 1577},
			{PID: 4, Name: "System", Threads: 220},
		},
	}
	v := Classify(s, DefaultThresholds())
	if v.Level != LevelElevated {
		t.Fatalf("level = %q, want elevated (reasons: %v)", v.Level, v.Reasons)
	}
	if v.Cause != CauseThreadLeak {
		t.Fatalf("cause = %q, want %q", v.Cause, CauseThreadLeak)
	}
	if v.ThreadLeakProcess != "WindowsTerminal.exe" || v.ThreadLeakPID != 17836 || v.ThreadLeakCount != 1577 {
		t.Fatalf("thread-leak attribution = %q/%d/%d, want WindowsTerminal.exe/17836/1577",
			v.ThreadLeakProcess, v.ThreadLeakPID, v.ThreadLeakCount)
	}
}

func TestClassify_ThreadLeak_attributedUnderChurnStall(t *testing.T) {
	// A churn stall AND a thread leak at once. The freeze cause wins Level/Cause,
	// but the thread leak must still be attributed so the operator sees both.
	s := Sample{
		TotalFaultsPerSec:     911901,
		HardFaultsPerSec:      2,
		ContextSwitchesPerSec: 302996,
		SystemCallsPerSec:     1309273,
		AvailableMB:           126840,
		DiskQueueLen:          0,
		TopThreads:            []ProcThreads{{PID: 17836, Name: "WindowsTerminal.exe", Threads: 1577}},
	}
	v := Classify(s, DefaultThresholds())
	if v.Level != LevelStall || v.Cause != CauseSoftFault {
		t.Fatalf("got level=%q cause=%q, want stall/soft_fault_churn (freeze wins)", v.Level, v.Cause)
	}
	if v.ThreadLeakProcess != "WindowsTerminal.exe" || v.ThreadLeakCount != 1577 {
		t.Fatalf("thread leak not attributed under stall: %q/%d", v.ThreadLeakProcess, v.ThreadLeakCount)
	}
}

func TestClassify_ThreadsBelowThreshold_noLeak(t *testing.T) {
	// A normal multi-threaded process (below the 500 line) is not a leak suspect.
	s := Sample{
		TotalFaultsPerSec: 22000,
		AvailableMB:       126000,
		TopThreads:        []ProcThreads{{PID: 100, Name: "fak.exe", Threads: 48}},
	}
	v := Classify(s, DefaultThresholds())
	if v.Level != LevelCalm {
		t.Fatalf("level = %q, want calm", v.Level)
	}
	if v.ThreadLeakProcess != "" {
		t.Fatalf("unexpected thread-leak attribution %q for a 48-thread process", v.ThreadLeakProcess)
	}
}

func TestWorstThreadHog_picksMaxAtOrAboveThreshold(t *testing.T) {
	in := []ProcThreads{
		{PID: 1, Name: "a", Threads: 200},  // below
		{PID: 2, Name: "b", Threads: 1577}, // worst
		{PID: 3, Name: "c", Threads: 600},
	}
	got, ok := worstThreadHog(in, 500)
	if !ok || got.PID != 2 || got.Threads != 1577 {
		t.Fatalf("got %+v ok=%v, want b/1577", got, ok)
	}
	if _, ok := worstThreadHog(in, 0); ok {
		t.Fatalf("threshold 0 must disable the rule")
	}
	if _, ok := worstThreadHog([]ProcThreads{{PID: 1, Threads: 48}}, 500); ok {
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

// --- Growth (trajectory) axis: ClassifyWithBaseline ---

func TestClassifyWithBaseline_HandleGrowth_flagsClimbBelowAbsoluteLine(t *testing.T) {
	// The whole point of the growth axis: a process CLIMBING while still under the
	// 10k absolute line. WindowsTerminal first seen at 3200 handles, now 5000 —
	// climbed +1800 (>= 1500 growth, current >= 3000 floor). The absolute axis
	// (10k) does NOT fire, but growth flags an otherwise-calm box as elevated with
	// cause handle_leak and names the culprit and its climb.
	base := Sample{TopHandles: []ProcHandles{{PID: 17836, Name: "WindowsTerminal.exe", Handles: 3200}}}
	cur := Sample{
		TotalFaultsPerSec: 22000,
		HardFaultsPerSec:  10,
		AvailableMB:       135000,
		TopHandles:        []ProcHandles{{PID: 17836, Name: "WindowsTerminal.exe", Handles: 5000}},
	}
	v := ClassifyWithBaseline(base, cur, DefaultThresholds())
	if v.Level != LevelElevated {
		t.Fatalf("level = %q, want elevated (reasons: %v)", v.Level, v.Reasons)
	}
	if v.Cause != CauseHandleLeak {
		t.Fatalf("cause = %q, want %q", v.Cause, CauseHandleLeak)
	}
	if v.HandleLeakProcess != "" {
		t.Fatalf("absolute leak axis must NOT fire at 5000 handles, got %q", v.HandleLeakProcess)
	}
	if v.HandleGrowthProcess != "WindowsTerminal.exe" || v.HandleGrowthPID != 17836 ||
		v.HandleGrowthCount != 5000 || v.HandleGrowthDelta != 1800 {
		t.Fatalf("growth attribution = %q/%d/%d/+%d, want WindowsTerminal.exe/17836/5000/+1800",
			v.HandleGrowthProcess, v.HandleGrowthPID, v.HandleGrowthCount, v.HandleGrowthDelta)
	}
}

func TestClassifyWithBaseline_HighButStable_noGrowthFlag(t *testing.T) {
	// A process high AND flat (17000 both samples): the absolute axis fires (>10k),
	// but the growth axis must stay silent — "high" and "climbing" are distinct,
	// and a stable-high process is not on a leak trajectory.
	proc := []ProcHandles{{PID: 17836, Name: "WindowsTerminal.exe", Handles: 17000}}
	base := Sample{TopHandles: proc}
	cur := Sample{TotalFaultsPerSec: 22000, AvailableMB: 135000, TopHandles: proc}
	v := ClassifyWithBaseline(base, cur, DefaultThresholds())
	if v.HandleLeakProcess != "WindowsTerminal.exe" {
		t.Fatalf("absolute axis should fire at 17000 handles, got %q", v.HandleLeakProcess)
	}
	if v.HandleGrowthProcess != "" {
		t.Fatalf("stable-high process must not be a growth suspect, got %q (+%d)", v.HandleGrowthProcess, v.HandleGrowthDelta)
	}
}

func TestClassifyWithBaseline_ThreadGrowth_flagsClimb(t *testing.T) {
	// A terminal accreting threads: 210 -> 400 (+190 >= 100 growth, current >= 200
	// floor), still under the 500 absolute line. Growth raises calm to elevated
	// with cause thread_leak.
	base := Sample{TopThreads: []ProcThreads{{PID: 17836, Name: "WindowsTerminal.exe", Threads: 210}}}
	cur := Sample{
		TotalFaultsPerSec: 22000,
		AvailableMB:       126000,
		TopThreads:        []ProcThreads{{PID: 17836, Name: "WindowsTerminal.exe", Threads: 400}},
	}
	v := ClassifyWithBaseline(base, cur, DefaultThresholds())
	if v.Level != LevelElevated || v.Cause != CauseThreadLeak {
		t.Fatalf("got level=%q cause=%q, want elevated/thread_leak (reasons: %v)", v.Level, v.Cause, v.Reasons)
	}
	if v.ThreadLeakProcess != "" {
		t.Fatalf("absolute thread axis must NOT fire at 400 threads, got %q", v.ThreadLeakProcess)
	}
	if v.ThreadGrowthProcess != "WindowsTerminal.exe" || v.ThreadGrowthCount != 400 || v.ThreadGrowthDelta != 190 {
		t.Fatalf("thread-growth attribution = %q/%d/+%d, want WindowsTerminal.exe/400/+190",
			v.ThreadGrowthProcess, v.ThreadGrowthCount, v.ThreadGrowthDelta)
	}
}

func TestClassifyWithBaseline_belowFloorOrDelta_noFlag(t *testing.T) {
	// Below floor: a small process climbing hard (100 -> 900, +800) but current 900
	// is under the 3000 floor -> not a leak, just normal churn. Below delta: a big
	// process barely moving (5000 -> 5100, +100 < 1500) -> stable, not a trajectory.
	base := Sample{TopHandles: []ProcHandles{
		{PID: 1, Name: "small.exe", Handles: 100},
		{PID: 2, Name: "big.exe", Handles: 5000},
	}}
	cur := Sample{TotalFaultsPerSec: 22000, AvailableMB: 135000, TopHandles: []ProcHandles{
		{PID: 1, Name: "small.exe", Handles: 900},
		{PID: 2, Name: "big.exe", Handles: 5100},
	}}
	v := ClassifyWithBaseline(base, cur, DefaultThresholds())
	if v.Level != LevelCalm {
		t.Fatalf("level = %q, want calm (reasons: %v)", v.Level, v.Reasons)
	}
	if v.HandleGrowthProcess != "" {
		t.Fatalf("no growth suspect expected, got %q (+%d)", v.HandleGrowthProcess, v.HandleGrowthDelta)
	}
}

func TestClassifyWithBaseline_firstSightAndPIDReuse_noFlag(t *testing.T) {
	// A PID absent from the baseline (first sight this session) has no trajectory to
	// measure. A reused PID whose NAME changed is a different process — its climb is
	// meaningless. Neither may be flagged.
	base := Sample{TopHandles: []ProcHandles{{PID: 100, Name: "old.exe", Handles: 3200}}}
	cur := Sample{
		TotalFaultsPerSec: 22000,
		AvailableMB:       135000,
		TopHandles: []ProcHandles{
			{PID: 100, Name: "new.exe", Handles: 9000},   // PID reused by a different process
			{PID: 200, Name: "fresh.exe", Handles: 8000}, // never seen before
		},
	}
	v := ClassifyWithBaseline(base, cur, DefaultThresholds())
	if v.HandleGrowthProcess != "" {
		t.Fatalf("first-sight/PID-reuse must not flag growth, got %q (+%d)", v.HandleGrowthProcess, v.HandleGrowthDelta)
	}
	if v.Level != LevelCalm {
		t.Fatalf("level = %q, want calm", v.Level)
	}
}

func TestClassifyWithBaseline_growthNeverFabricatesOrDowngradesStall(t *testing.T) {
	// A genuine soft-fault churn STALL that also has a climbing handle count. Growth
	// is a WARNING: the freeze must still win Level/Cause, but the growth trajectory
	// must be attributed alongside it (so the operator sees both).
	base := Sample{TopHandles: []ProcHandles{{PID: 17836, Name: "WindowsTerminal.exe", Handles: 3200}}}
	cur := Sample{
		TotalFaultsPerSec:      1241214,
		HardFaultsPerSec:       39,
		DemandZeroFaultsPerSec: 554729,
		AvailableMB:            135000,
		DiskQueueLen:           0,
		TopHandles:             []ProcHandles{{PID: 17836, Name: "WindowsTerminal.exe", Handles: 5000}},
	}
	v := ClassifyWithBaseline(base, cur, DefaultThresholds())
	if v.Level != LevelStall || v.Cause != CauseSoftFault {
		t.Fatalf("got level=%q cause=%q, want stall/soft_fault_churn (freeze wins)", v.Level, v.Cause)
	}
	if v.HandleGrowthProcess != "WindowsTerminal.exe" || v.HandleGrowthDelta != 1800 {
		t.Fatalf("growth not attributed under stall: %q/+%d", v.HandleGrowthProcess, v.HandleGrowthDelta)
	}
}

func TestClassifyWithBaseline_emptyBaseline_matchesClassify(t *testing.T) {
	// With no baseline (empty), ClassifyWithBaseline must reduce to Classify: no
	// growth attribution, identical level/cause. Guards backward-compatibility.
	cur := Sample{
		TotalFaultsPerSec: 22000,
		AvailableMB:       135000,
		TopHandles:        []ProcHandles{{PID: 17836, Name: "WindowsTerminal.exe", Handles: 33418}},
	}
	base := ClassifyWithBaseline(Sample{}, cur, DefaultThresholds())
	plain := Classify(cur, DefaultThresholds())
	if base.Level != plain.Level || base.Cause != plain.Cause {
		t.Fatalf("with empty baseline: got %q/%q, want %q/%q", base.Level, base.Cause, plain.Level, plain.Cause)
	}
	if base.HandleGrowthProcess != "" {
		t.Fatalf("empty baseline must yield no growth attribution, got %q", base.HandleGrowthProcess)
	}
}

func TestClassifyWithBaseline_isPure_noMutation(t *testing.T) {
	baseH := []ProcHandles{{PID: 1, Name: "a", Handles: 3000}, {PID: 2, Name: "b", Handles: 4000}}
	curH := []ProcHandles{{PID: 1, Name: "a", Handles: 3100}, {PID: 2, Name: "b", Handles: 9000}}
	_ = ClassifyWithBaseline(Sample{TopHandles: baseH}, Sample{TopHandles: curH, AvailableMB: 229000}, DefaultThresholds())
	if baseH[0].PID != 1 || baseH[1].PID != 2 || curH[0].PID != 1 || curH[1].PID != 2 {
		t.Fatalf("ClassifyWithBaseline mutated caller slices: base=%+v cur=%+v", baseH, curH)
	}
}

func TestWorstHandleGrowth_picksMaxClimbAboveDeltaAndFloor(t *testing.T) {
	base := []ProcHandles{
		{PID: 1, Name: "a", Handles: 3000},
		{PID: 2, Name: "b", Handles: 3200}, // climbs +1800 -> qualifies
		{PID: 3, Name: "c", Handles: 3000}, // climbs +5000 -> biggest climb
	}
	cur := []ProcHandles{
		{PID: 1, Name: "a", Handles: 3100}, // +100 < 1500 delta
		{PID: 2, Name: "b", Handles: 5000}, // +1800
		{PID: 3, Name: "c", Handles: 8000}, // +5000 (worst)
	}
	got, climb, ok := worstHandleGrowth(base, cur, 1500, 3000)
	if !ok || got.PID != 3 || climb != 5000 {
		t.Fatalf("got %+v climb=%d ok=%v, want c/+5000", got, climb, ok)
	}
	if _, _, ok := worstHandleGrowth(base, cur, 0, 3000); ok {
		t.Fatalf("delta 0 must disable the growth rule")
	}
	// Raise the floor above every current count -> nothing qualifies.
	if _, _, ok := worstHandleGrowth(base, cur, 1500, 100000); ok {
		t.Fatalf("floor above all current counts must return ok=false")
	}
}

func TestClassify_CPUSaturation_busyBox(t *testing.T) {
	s := Sample{
		CPUPercent:           97.5,
		ProcessorQueueLength: 20,
		ProcessorCount:       16,
		AvailableMB:          32000,
		TopCPU: []ProcCPU{
			{PID: 41, Name: "worker.exe", Percent: 21.5},
			{PID: 42, Name: "spinner.exe", Percent: 62.0},
		},
	}
	v := Classify(s, DefaultThresholds())
	if v.Level != LevelStall || v.Cause != CauseCPUSaturation {
		t.Fatalf("got level=%s cause=%s reasons=%v", v.Level, v.Cause, v.Reasons)
	}
	if v.TopCPUProcess != "spinner.exe" || v.TopCPUPID != 42 || v.TopCPUPercent != 62.0 {
		t.Fatalf("missing CPU attribution: %+v", v)
	}
	if len(v.Reasons) != 2 || !strings.Contains(v.Reasons[0], "1.25 runnable waiters/core") || !strings.Contains(v.Reasons[1], "spinner.exe") {
		t.Fatalf("reason does not carry normalized queue witness: %v", v.Reasons)
	}
}

func TestClassify_FullCPUWithoutQueue_isNotStall(t *testing.T) {
	s := Sample{CPUPercent: 99, ProcessorQueueLength: 2, ProcessorCount: 16}
	v := Classify(s, DefaultThresholds())
	if v.Level != LevelElevated || v.Cause != CauseCPUSaturation {
		t.Fatalf("productive full-core use must warn, not claim grind: %+v", v)
	}
}

func TestClassify_QueueWithoutFullCPU_isCalm(t *testing.T) {
	s := Sample{CPUPercent: 45, ProcessorQueueLength: 32, ProcessorCount: 16}
	v := Classify(s, DefaultThresholds())
	if v.Level != LevelCalm {
		t.Fatalf("queue without CPU pressure is not a witnessed busy box: %+v", v)
	}
}

func TestClassify_CPUSaturation_isNormalizedAcrossHostSize(t *testing.T) {
	thresholds := DefaultThresholds()
	for _, s := range []Sample{
		{CPUPercent: 95, ProcessorQueueLength: 4, ProcessorCount: 8},
		{CPUPercent: 95, ProcessorQueueLength: 32, ProcessorCount: 64},
	} {
		v := Classify(s, thresholds)
		if v.Level != LevelStall || v.Cause != CauseCPUSaturation {
			t.Fatalf("sample %+v: %+v", s, v)
		}
	}
}

func TestClassify_GPUMemPressure_Stall(t *testing.T) {
	// Committed VRAM at or above 95% capacity signals an immediate GPU memory pressure stall
	// due to WDDM page thrashing between local VRAM and system memory.
	s := Sample{
		AvailableMB:        32000,
		VRAMTotalBytes:     8 * 1024 * 1024 * 1024,
		VRAMCommittedBytes: 7800 * 1024 * 1024, // ~95.2% committed
		VRAMSharedBytes:    512 * 1024 * 1024,
	}
	v := Classify(s, DefaultThresholds())
	if v.Level != LevelStall {
		t.Fatalf("got level=%q, want stall", v.Level)
	}
	if v.Cause != CauseGPUMemPressure {
		t.Fatalf("got cause=%q, want %q", v.Cause, CauseGPUMemPressure)
	}
	joined := strings.Join(v.Reasons, "; ")
	if !strings.Contains(joined, "VRAM committed") || !strings.Contains(joined, "shared aperture") {
		t.Fatalf("reasons missing VRAM / aperture details: %v", v.Reasons)
	}
}

func TestClassify_GPUMemPressure_Elevated(t *testing.T) {
	// Committed VRAM at 91% capacity (>= 90% elevated threshold, < 95% stall threshold)
	// raises calm box to elevated with cause gpu_memory_pressure.
	s := Sample{
		AvailableMB:        32000,
		VRAMTotalBytes:     8 * 1024 * 1024 * 1024,
		VRAMCommittedBytes: 7500 * 1024 * 1024, // ~91.5% committed
	}
	v := Classify(s, DefaultThresholds())
	if v.Level != LevelElevated {
		t.Fatalf("got level=%q, want elevated", v.Level)
	}
	if v.Cause != CauseGPUMemPressure {
		t.Fatalf("got cause=%q, want %q", v.Cause, CauseGPUMemPressure)
	}
	joined := strings.Join(v.Reasons, "; ")
	if !strings.Contains(joined, "VRAM committed") {
		t.Fatalf("reasons missing VRAM details: %v", v.Reasons)
	}
}

func TestClassify_GPUMemPressure_AttributedUnderChurnStall(t *testing.T) {
	// Under a soft-fault churn stall, elevated VRAM pressure is still recorded in reasons.
	s := Sample{
		TotalFaultsPerSec:  450000,
		HardFaultsPerSec:   100,
		AvailableMB:        32000,
		VRAMTotalBytes:     8 * 1024 * 1024 * 1024,
		VRAMCommittedBytes: 7500 * 1024 * 1024, // ~91.5% committed
	}
	v := Classify(s, DefaultThresholds())
	if v.Level != LevelStall || v.Cause != CauseSoftFault {
		t.Fatalf("got level=%q cause=%q, want stall/soft_fault_churn", v.Level, v.Cause)
	}
	joined := strings.Join(v.Reasons, "; ")
	if !strings.Contains(joined, "VRAM committed") {
		t.Fatalf("reasons missing VRAM warning under churn stall: %v", v.Reasons)
	}
}

func TestClassify_GPUMemPressure_Calm(t *testing.T) {
	// Normal VRAM usage (~50%) stays calm.
	s := Sample{
		AvailableMB:        32000,
		VRAMTotalBytes:     8 * 1024 * 1024 * 1024,
		VRAMCommittedBytes: 4000 * 1024 * 1024,
	}
	v := Classify(s, DefaultThresholds())
	if v.Level != LevelCalm || v.Cause != CauseNone {
		t.Fatalf("got level=%q cause=%q, want calm/none", v.Level, v.Cause)
	}
}

func TestClassify_GPUMemPressure_ZeroTotalDoesNotTrip(t *testing.T) {
	// When VRAM total is 0 (unobserved/no GPU), no warning is triggered.
	s := Sample{
		AvailableMB:        32000,
		VRAMTotalBytes:     0,
		VRAMCommittedBytes: 4000 * 1024 * 1024,
	}
	v := Classify(s, DefaultThresholds())
	if v.Level != LevelCalm || v.Cause != CauseNone {
		t.Fatalf("got level=%q cause=%q, want calm/none", v.Level, v.Cause)
	}
}
