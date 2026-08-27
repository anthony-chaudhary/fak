package systembaseline

import (
	"testing"
	"time"
)

func measuredCgroupFixture() CgroupV2 {
	axis := func(some, full float64) PressureAxis {
		return PressureAxis{
			Some: available(some, "microseconds", "test pressure"),
			Full: available(full, "microseconds", "test pressure"),
		}
	}
	return CgroupV2{
		State: CgroupStateMeasured,
		Membership: CgroupMembership{
			AtomicPlacement: true,
			RootPID:         10,
			AfterStart:      available(1, "processes", "test cgroup.procs"),
			AfterWait:       available(0, "processes", "test cgroup.procs"),
			PlacementSource: "test atomic placement",
		},
		CPU: availableCounterSet(map[string]uint64{
			"usage_usec":  1_000_000,
			"user_usec":   900_000,
			"system_usec": 100_000,
		}, "test cpu.stat"),
		Memory: CgroupMemory{
			CurrentBytes: available(0, "bytes", "test memory.current"),
			PeakBytes:    available(64<<20, "bytes", "test memory.peak"),
			Events:       availableCounterSet(map[string]uint64{"oom": 0, "oom_kill": 0}, "test memory.events"),
		},
		Pressure: CgroupPressure{
			CPU:    axis(10_000, 0),
			Memory: axis(0, 0),
			IO:     axis(0, 0),
		},
		Cleanup: CgroupCleanup{Attempted: true, Empty: true, Removed: true},
	}
}

func TestBuildWithCgroupV2UsesExactCountersAfterShortLivedProcessesExit(t *testing.T) {
	samples := fixture(100e6, 0)
	for i := range samples {
		samples[i].Processes = nil
		samples[i].ProcessEnumerationOK = false
	}
	cgroup := measuredCgroupFixture()
	report := BuildWithCgroupV2(quietFixture(100e6), samples, 10, time.Second, DefaultPolicy(), 0, false, &cgroup)
	if report.Coverage.DescendantAttribution != "exact_cgroup_v2" {
		t.Fatalf("coverage=%q cgroup=%+v", report.Coverage.DescendantAttribution, report.CgroupV2)
	}
	if !report.Attribution.SUTCPUPercentOfHost.Available || report.Attribution.SUTCPUPercentOfHost.Value != 25 {
		t.Fatalf("SUT CPU=%+v, want exact 25%% from cumulative cgroup usage", report.Attribution.SUTCPUPercentOfHost)
	}
	if !report.Attribution.NonSUTCPUPercentOfHost.Available || report.Attribution.NonSUTCPUPercentOfHost.Value != 2.5 {
		t.Fatalf("non-SUT CPU=%+v, want exact 2.5%% residual", report.Attribution.NonSUTCPUPercentOfHost)
	}
	if !report.Attribution.SUTRSSBytes.Available || report.Attribution.SUTRSSBytes.Value != 64<<20 {
		t.Fatalf("SUT memory=%+v", report.Attribution.SUTRSSBytes)
	}
	if report.Verdict != VerdictClean {
		t.Fatalf("verdict=%s findings=%+v", report.Verdict, report.Findings)
	}
	if err := report.Validate(); err != nil {
		t.Fatal(err)
	}
}

func TestCgroupPSIInfluencesPolicyWithoutCorrectingCounters(t *testing.T) {
	cgroup := measuredCgroupFixture()
	cgroup.Pressure.CPU.Some.Value = 100_000
	report := BuildWithCgroupV2(quietFixture(100e6), fixture(100e6, 0), 10, time.Second, DefaultPolicy(), 0, false, &cgroup)
	if report.Verdict != VerdictInvestigate || !hasFindingCode(report, "CPU_PSI_STALL_HIGH") {
		t.Fatalf("verdict=%s findings=%+v", report.Verdict, report.Findings)
	}
	if report.CgroupV2.Pressure.CPU.Some.Value != 100_000 || report.Attribution.SUTCPUPercentOfHost.Value != 25 {
		t.Fatalf("PSI numerically corrected resource counters: cgroup=%+v attribution=%+v", report.CgroupV2, report.Attribution)
	}
	if err := report.Validate(); err != nil {
		t.Fatal(err)
	}
}

func TestUnavailableCgroupPreservesSampledFallbackAndTypedPSI(t *testing.T) {
	cgroup := unavailableCgroup("delegation denied")
	report := BuildWithCgroupV2(quietFixture(100e6), fixture(0, 0), 10, time.Second, DefaultPolicy(), 0, false, &cgroup)
	if report.Coverage.DescendantAttribution != "sampled_pid_ppid_tree" || report.CgroupV2.State != CgroupStateUnavailable {
		t.Fatalf("fallback=%+v", report)
	}
	if report.CgroupV2.Pressure.CPU.Some.Available || report.CgroupV2.Pressure.CPU.Some.Reason != "delegation denied" {
		t.Fatalf("PSI fallback=%+v", report.CgroupV2.Pressure)
	}
	if err := report.Validate(); err != nil {
		t.Fatal(err)
	}
}

func TestCgroupCleanupFailureRequiresInvestigation(t *testing.T) {
	cgroup := measuredCgroupFixture()
	cgroup.Cleanup.Empty = false
	cgroup.Cleanup.Removed = false
	cgroup.Cleanup.Reason = "injected busy cgroup"
	report := BuildWithCgroupV2(quietFixture(100e6), fixture(100e6, 0), 10, time.Second, DefaultPolicy(), 0, false, &cgroup)
	if report.Verdict != VerdictInvestigate || !hasFindingCode(report, "CGROUP_CLEANUP_FAILED") {
		t.Fatalf("verdict=%s findings=%+v", report.Verdict, report.Findings)
	}
	if err := report.Validate(); err != nil {
		t.Fatal(err)
	}
}
