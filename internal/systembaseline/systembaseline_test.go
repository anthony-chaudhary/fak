package systembaseline

import (
	"encoding/json"
	"testing"
	"time"
)

func fixture(nonSUTCPU uint64, unreadable int) []Snapshot {
	t0 := time.Unix(100, 0).UTC()
	return []Snapshot{
		{At: t0, Host: HostSample{CPUAvailable: true, TotalCPUNS: 10e9, BusyCPUNS: 2e9, MemoryAvailable: true, MemoryTotal: 1000, MemoryFree: 500}, ProcessEnumerationOK: true, ProcessUnreadable: unreadable, Processes: []ProcessSample{{PID: 10, PPID: 1, StartID: 100, Image: "/tmp/sut", CPUAvailable: true, CPUNS: 100, RSSAvailable: true, RSSBytes: 50}, {PID: 20, PPID: 1, StartID: 200, Image: `C:\host\other.exe`, CPUAvailable: true, CPUNS: 50, RSSAvailable: true, RSSBytes: 100}}},
		{At: t0.Add(time.Second), Host: HostSample{CPUAvailable: true, TotalCPUNS: 14e9, BusyCPUNS: 2e9 + 1e9 + nonSUTCPU, MemoryAvailable: true, MemoryTotal: 1000, MemoryFree: 450}, ProcessEnumerationOK: true, ProcessUnreadable: unreadable, Processes: []ProcessSample{{PID: 10, PPID: 1, StartID: 100, Image: "/tmp/sut", CPUAvailable: true, CPUNS: 1e9 + 100, RSSAvailable: true, RSSBytes: 60}, {PID: 20, PPID: 1, StartID: 200, Image: `C:\host\other.exe`, CPUAvailable: true, CPUNS: 50 + nonSUTCPU, RSSAvailable: true, RSSBytes: 120}}},
	}
}

func quietFixture(busyCPU uint64) []Snapshot {
	samples := fixture(0, 0)
	samples[1].Host.BusyCPUNS = samples[0].Host.BusyCPUNS + busyCPU
	return samples
}

func TestBuildCleanAndScrubsConsumers(t *testing.T) {
	p := DefaultPolicy()
	p.MaximumNonSUTCPUPercent = 20
	p.IncludeTopConsumers = true
	r := Build(quietFixture(100e6), fixture(100e6, 0), 10, time.Second, p, 0, false)
	if r.Verdict != VerdictClean {
		t.Fatalf("verdict=%s findings=%+v", r.Verdict, r.Findings)
	}
	if len(r.TopNonSUT) == 0 || r.TopNonSUT[0].Image != "other.exe" {
		t.Fatalf("top=%+v", r.TopNonSUT)
	}
	if err := r.Validate(); err != nil {
		t.Fatal(err)
	}
	if r.Baseline.Samples != 2 || !r.BaselineHost.CPUPercent.Available {
		t.Fatalf("baseline phase missing: %+v", r.Baseline)
	}
	if r.Coverage.DescendantAttribution != "sampled_pid_ppid_tree" {
		t.Fatalf("descendant attribution=%q", r.Coverage.DescendantAttribution)
	}
}

func TestBuildInvestigatesContaminationAndUnknown(t *testing.T) {
	high := Build(quietFixture(100e6), fixture(2e9, 0), 10, time.Second, DefaultPolicy(), 0, false)
	if high.Verdict != VerdictInvestigate {
		t.Fatalf("high verdict=%s", high.Verdict)
	}
	unknown := Build(quietFixture(100e6), fixture(0, 0), 999, time.Second, DefaultPolicy(), 0, false)
	if unknown.Verdict != VerdictInvestigate || unknown.Attribution.NonSUTCPUPercentOfHost.Available {
		t.Fatalf("unknown=%+v", unknown)
	}
	busyBaseline := Build(quietFixture(4e9), fixture(0, 0), 10, time.Second, DefaultPolicy(), 0, false)
	if busyBaseline.Verdict != VerdictInvestigate || !hasFindingCode(busyBaseline, "BASELINE_HOST_CPU_HIGH") {
		t.Fatalf("busy baseline=%+v", busyBaseline)
	}
}

func TestSamplerDutyExcludesFinalEndpointAndGatesObserverCost(t *testing.T) {
	baseline := quietFixture(100e6)
	baseline[0].CensusWallNS = 50e6
	baseline[1].CensusWallNS = 900e6
	command := fixture(0, 0)
	command[0].CensusWallNS = 100e6
	command[1].CensusWallNS = 900e6
	policy := DefaultPolicy()
	policy.MaximumSamplerDutyPercent = 15
	r := Build(baseline, command, 10, time.Second, policy, 0, false)
	if r.Verdict != VerdictClean || r.BaselineSampler.WallNS != 50e6 || r.CommandSampler.WallNS != 100e6 || r.BaselineSampler.CountedSamples != 1 || r.CommandSampler.CountedSamples != 1 {
		t.Fatalf("endpoint census was counted or clean duty rejected: %+v %+v verdict=%s", r.BaselineSampler, r.CommandSampler, r.Verdict)
	}
	if r.BaselineSampler.DutyPercent.Value != 5 || r.CommandSampler.DutyPercent.Value != 10 {
		t.Fatalf("duty baseline=%g command=%g", r.BaselineSampler.DutyPercent.Value, r.CommandSampler.DutyPercent.Value)
	}

	command[0].CensusWallNS = 200e6
	high := Build(baseline, command, 10, time.Second, policy, 0, false)
	if high.Verdict != VerdictInvestigate || !hasFindingCode(high, "COMMAND_SAMPLER_DUTY_HIGH") {
		t.Fatalf("high sampler duty=%+v findings=%+v", high.CommandSampler, high.Findings)
	}
	high.Verdict = VerdictClean
	high.Seal()
	if err := high.Validate(); err == nil {
		t.Fatal("resealed high-duty clean report validated")
	}
}

func TestBuildInvalidAndDigestTamper(t *testing.T) {
	r := Build(quietFixture(100e6), fixture(0, 0)[:1], 10, time.Second, DefaultPolicy(), 0, false)
	if r.Verdict != VerdictInvalid {
		t.Fatalf("verdict=%s", r.Verdict)
	}
	if err := r.Validate(); err != nil {
		t.Fatal(err)
	}
	b, _ := json.Marshal(r)
	var copy Report
	_ = json.Unmarshal(b, &copy)
	copy.CommandExitCode = 9
	if err := copy.Validate(); err == nil {
		t.Fatal("tampered report validated")
	}
}

func TestPIDReuseAndMissingRSSMakeAttributionUnknown(t *testing.T) {
	samples := fixture(0, 0)
	samples[1].Processes[0].StartID = 101
	r := Build(quietFixture(100e6), samples, 10, time.Second, DefaultPolicy(), 0, false)
	if r.Verdict != VerdictInvestigate || r.Attribution.SUTCPUPercentOfHost.Available {
		t.Fatalf("PID reuse spliced counters: %+v", r.Attribution)
	}

	samples = fixture(0, 0)
	samples[1].Processes[1].RSSAvailable = false
	p := DefaultPolicy()
	p.RequireProcessMemory = true
	r = Build(quietFixture(100e6), samples, 10, time.Second, p, 0, false)
	if r.Attribution.NonSUTRSSBytes.Available || r.Verdict != VerdictInvestigate {
		t.Fatalf("missing RSS treated complete: %+v", r.Attribution)
	}
	incomplete := fixture(0, 1)
	for i := range incomplete {
		incomplete[i].AttributionIncomplete = true
	}
	r = Build(quietFixture(100e6), incomplete, 10, time.Second, DefaultPolicy(), 0, false)
	if r.Attribution.SUTCPUPercentOfHost.Available || r.Verdict != VerdictInvestigate {
		t.Fatalf("unreadable process treated as complete SUT attribution: %+v", r.Attribution)
	}
	partialUnrelated := fixture(0, 1)
	for i := range partialUnrelated {
		partialUnrelated[i].Processes = append(partialUnrelated[i].Processes, ProcessSample{PID: 30, PPID: 1, Image: "protected.exe"})
	}
	r = Build(quietFixture(100e6), partialUnrelated, 10, time.Second, DefaultPolicy(), 0, false)
	if !r.Attribution.SUTCPUPercentOfHost.Available || r.Verdict != VerdictClean {
		t.Fatalf("unrelated protected process invalidated SUT CPU: %+v", r.Attribution)
	}
}

func TestMissingSUTRSSDoesNotInvalidateCPUAttribution(t *testing.T) {
	samples := fixture(0, 0)
	samples[1].Processes[0].RSSAvailable = false
	r := Build(quietFixture(100e6), samples, 10, time.Second, DefaultPolicy(), 0, false)
	if r.Verdict != VerdictClean || !r.Attribution.SUTCPUPercentOfHost.Available || !r.Attribution.NonSUTCPUPercentOfHost.Available {
		t.Fatalf("missing SUT RSS invalidated CPU attribution: verdict=%s attribution=%+v", r.Verdict, r.Attribution)
	}
	if r.Attribution.SUTRSSBytes.Available || r.Attribution.NonSUTRSSBytes.Available {
		t.Fatalf("incomplete RSS reported available: %+v", r.Attribution)
	}

	policy := DefaultPolicy()
	policy.RequireProcessMemory = true
	r = Build(quietFixture(100e6), samples, 10, time.Second, policy, 0, false)
	if r.Verdict != VerdictInvestigate || !r.Attribution.SUTCPUPercentOfHost.Available || r.Attribution.SUTRSSBytes.Available {
		t.Fatalf("memory-required classification=%s attribution=%+v", r.Verdict, r.Attribution)
	}
}

func TestCPUAttributionRequiresAlignedHostEndpointsAndConsistentBusyDelta(t *testing.T) {
	for _, endpoint := range []int{0, 1} {
		t.Run([]string{"first", "last"}[endpoint], func(t *testing.T) {
			samples := fixture(0, 0)
			samples[endpoint].Host.CPUAvailable = false
			r := Build(quietFixture(100e6), samples, 10, time.Second, DefaultPolicy(), 0, false)
			if r.Verdict != VerdictInvestigate || r.Host.CPUPercent.Available || r.Attribution.SUTCPUPercentOfHost.Available || r.Attribution.NonSUTCPUPercentOfHost.Available {
				t.Fatalf("unaligned endpoint accepted: host=%+v attribution=%+v verdict=%s", r.Host, r.Attribution, r.Verdict)
			}
		})
	}
	samples := fixture(0, 0)
	samples[1].Processes[0].CPUNS = 2e9 + 100
	r := Build(quietFixture(100e6), samples, 10, time.Second, DefaultPolicy(), 0, false)
	if r.Verdict != VerdictInvestigate || r.Attribution.SUTCPUPercentOfHost.Available || r.Attribution.NonSUTCPUPercentOfHost.Available {
		t.Fatalf("SUT CPU exceeding host busy was clamped: %+v", r.Attribution)
	}
}

func TestClassifyWindowsProcessAdvance(t *testing.T) {
	for _, test := range []struct {
		name            string
		result, code    uintptr
		done, truncated bool
	}{
		{"continue", 1, 0, false, false},
		{"complete", 0, 18, true, false},
		{"truncated", 0, 5, true, true},
	} {
		t.Run(test.name, func(t *testing.T) {
			done, truncated := classifyWindowsProcessAdvance(test.result, test.code)
			if done != test.done || truncated != test.truncated {
				t.Fatalf("done=%v truncated=%v", done, truncated)
			}
		})
	}
}

func TestFailureTimeoutAndResealedContradictionAreInvalid(t *testing.T) {
	r := Build(quietFixture(100e6), fixture(0, 0), 10, time.Second, DefaultPolicy(), 7, false)
	if r.Verdict != VerdictInvalid {
		t.Fatalf("failed command verdict=%s", r.Verdict)
	}
	r.Verdict = VerdictClean
	r.Seal()
	if err := r.Validate(); err == nil {
		t.Fatal("resealed failed-command clean report validated")
	}
	timed := Build(quietFixture(100e6), fixture(0, 0), 10, time.Second, DefaultPolicy(), 124, true)
	if timed.Verdict != VerdictInvalid {
		t.Fatalf("timeout verdict=%s", timed.Verdict)
	}
}

func TestValidateRejectsResealedCleanContradictions(t *testing.T) {
	tests := []struct {
		name string
		edit func(*Report)
	}{
		{"missing baseline", func(r *Report) { r.Baseline = Window{} }},
		{"required axis unknown", func(r *Report) { r.Host.CPUPercent = Metric{Unit: "percent", Reason: "unknown"} }},
		{"baseline threshold", func(r *Report) { r.BaselineHost.CPUPercent.Value = r.Policy.MaximumNonSUTCPUPercent + 1 }},
		{"sampler threshold", func(r *Report) {
			r.CommandSampler.WallNS = r.Window.DurationNS / 2
			r.CommandSampler.DutyPercent.Value = 50
		}},
		{"non-SUT threshold", func(r *Report) { r.Attribution.NonSUTCPUPercentOfHost.Value = r.Policy.MaximumNonSUTCPUPercent + 1 }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := Build(quietFixture(100e6), fixture(0, 0), 10, time.Second, DefaultPolicy(), 0, false)
			tt.edit(&r)
			r.Verdict = VerdictClean
			r.Seal()
			if err := r.Validate(); err == nil {
				t.Fatal("resealed contradictory clean report validated")
			}
		})
	}
}

func hasFindingCode(report Report, code string) bool {
	for _, finding := range report.Findings {
		if finding.Code == code {
			return true
		}
	}
	return false
}

func TestCanonicalLinuxCPUTicksExcludeGuest(t *testing.T) {
	total, idle, ok := canonicalLinuxCPUTicks([]uint64{100, 20, 30, 40, 5, 1, 2, 3, 90, 10})
	if !ok || total != 201 || idle != 45 {
		t.Fatalf("total=%d idle=%d ok=%v", total, idle, ok)
	}
}
