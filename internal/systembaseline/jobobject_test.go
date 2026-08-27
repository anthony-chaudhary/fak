package systembaseline

import (
	"testing"
	"time"
)

func measuredWindowsJobFixture() WindowsJobObject {
	return WindowsJobObject{
		State: WindowsJobStateMeasured,
		Membership: WindowsJobMembership{
			AtomicPlacement: true,
			RootPID:         10,
			RootStartID:     1234,
			AfterStart:      available(1, "processes", "test job membership"),
			AfterWait:       available(0, "processes", "test job membership"),
			PlacementSource: "test suspended assignment",
			IdentitySource:  "test creation identity",
		},
		CPU: availableCounterSet(map[string]uint64{
			"usage_100ns":  10_000_000,
			"user_100ns":   9_000_000,
			"kernel_100ns": 1_000_000,
		}, "test job CPU"),
		Processes: availableCounterSet(map[string]uint64{
			"total_processes":  25,
			"active_processes": 0,
		}, "test job processes"),
		IO: availableCounterSet(map[string]uint64{"read_bytes": 1}, "test job I/O"),
		Memory: WindowsJobMemory{
			PeakJobCommitBytes:     available(64<<20, "bytes", "test peak job commit"),
			PeakProcessCommitBytes: available(32<<20, "bytes", "test peak process commit"),
		},
		Cleanup: WindowsJobCleanup{Attempted: true, Empty: true, Closed: true},
	}
}

func TestBuildWithCommandAttributionUsesOnlyVerifiedJobCounters(t *testing.T) {
	samples := fixture(100e6, 0)
	job := measuredWindowsJobFixture()
	report := BuildWithCommandAttribution(quietFixture(100e6), samples, 10, time.Second, DefaultPolicy(), 0, false, &CommandAttribution{WindowsJobObject: &job})
	if report.Coverage.DescendantAttribution != "job_object" {
		t.Fatalf("coverage=%q job=%+v", report.Coverage.DescendantAttribution, report.WindowsJobObject)
	}
	if !report.Attribution.SUTCPUPercentOfHost.Available || report.Attribution.SUTCPUPercentOfHost.Value != 25 {
		t.Fatalf("SUT CPU=%+v, want exact 25%% from cumulative job usage", report.Attribution.SUTCPUPercentOfHost)
	}
	if report.Attribution.SUTRSSBytes.Value == job.Memory.PeakJobCommitBytes.Value {
		t.Fatalf("job commit was mislabeled as RSS: attribution=%+v job=%+v", report.Attribution.SUTRSSBytes, job.Memory)
	}
	if err := report.Validate(); err != nil {
		t.Fatal(err)
	}

	unverified := job
	unverified.State = WindowsJobStateUnavailable
	unverified.Reason = "injected assignment denial"
	unverified.Membership.AtomicPlacement = false
	unverified.Membership.UnavailableCause = unverified.Reason
	unverified.CPU = unavailableCounterSet(unverified.Reason)
	unverified.Processes = unavailableCounterSet(unverified.Reason)
	unverified.IO = unavailableCounterSet(unverified.Reason)
	unverified.Memory.PeakJobCommitBytes = unavailable("bytes", unverified.Reason)
	unverified.Memory.PeakProcessCommitBytes = unavailable("bytes", unverified.Reason)
	fallback := BuildWithCommandAttribution(quietFixture(100e6), samples, 10, time.Second, DefaultPolicy(), 0, false, &CommandAttribution{WindowsJobObject: &unverified})
	if fallback.Coverage.DescendantAttribution != "sampled_pid_ppid_tree" {
		t.Fatalf("unverified receipt emitted exact coverage: %+v", fallback)
	}
	if err := fallback.Validate(); err != nil {
		t.Fatal(err)
	}
}

func TestWindowsJobCleanupFailureIsInvalidByPolicy(t *testing.T) {
	job := measuredWindowsJobFixture()
	job.State = WindowsJobStateUnavailable
	job.Reason = "injected cleanup timeout"
	job.Membership.UnavailableCause = job.Reason
	job.Cleanup.Empty = false
	job.Cleanup.Reason = job.Reason
	report := BuildWithCommandAttribution(quietFixture(100e6), fixture(100e6, 0), 10, time.Second, DefaultPolicy(), 0, false, &CommandAttribution{WindowsJobObject: &job})
	if report.Verdict != VerdictInvalid || !hasFindingCode(report, "JOB_OBJECT_CLEANUP_FAILED") || report.Coverage.DescendantAttribution != "sampled_pid_ppid_tree" {
		t.Fatalf("cleanup policy=%+v", report)
	}
	if err := report.Validate(); err != nil {
		t.Fatal(err)
	}
}
