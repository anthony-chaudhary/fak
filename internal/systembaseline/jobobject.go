package systembaseline

import (
	"fmt"
	"math"
	"strings"
)

const (
	WindowsJobStateMeasured    = "measured"
	WindowsJobStateUnavailable = "unavailable"
)

// WindowsJobMembership records the identity-safe launch boundary. RootStartID
// is the process creation FILETIME read through the still-open process handle;
// it is therefore not vulnerable to a PID being reused later in the run.
type WindowsJobMembership struct {
	AtomicPlacement  bool   `json:"atomic_placement"`
	RootPID          int    `json:"root_pid"`
	RootStartID      uint64 `json:"root_start_id"`
	AfterStart       Metric `json:"members_after_start"`
	AfterWait        Metric `json:"members_after_wait"`
	PlacementSource  string `json:"placement_source,omitempty"`
	IdentitySource   string `json:"identity_source,omitempty"`
	UnavailableCause string `json:"unavailable_reason,omitempty"`
}

// WindowsJobMemory reports peak committed bytes, not RSS. Windows Job Objects
// expose authoritative peak aggregate commit counters but no equivalent exact
// aggregate working-set counter.
type WindowsJobMemory struct {
	PeakJobCommitBytes     Metric `json:"peak_job_commit_bytes"`
	PeakProcessCommitBytes Metric `json:"peak_process_commit_bytes"`
}

type WindowsJobCleanup struct {
	Attempted       bool   `json:"attempted"`
	KilledRemaining bool   `json:"killed_remaining"`
	Empty           bool   `json:"empty"`
	Closed          bool   `json:"closed"`
	Reason          string `json:"reason,omitempty"`
}

// WindowsJobObject is the bounded per-command Windows Job Object receipt. Its
// held-handle creation identity plus lifetime job counters provide the
// PID-reuse-safe equivalent of Process V2 identity/counters. CPU, process, and
// I/O counters include descendants that exit between sampler snapshots.
type WindowsJobObject struct {
	State      string               `json:"state"`
	Reason     string               `json:"reason,omitempty"`
	Membership WindowsJobMembership `json:"membership"`
	CPU        CounterSet           `json:"cpu_accounting"`
	Processes  CounterSet           `json:"process_accounting"`
	IO         CounterSet           `json:"io_accounting"`
	Memory     WindowsJobMemory     `json:"memory"`
	Cleanup    WindowsJobCleanup    `json:"cleanup"`
}

func (j WindowsJobObject) clone() WindowsJobObject {
	j.CPU.Values = cloneCounters(j.CPU.Values)
	j.Processes.Values = cloneCounters(j.Processes.Values)
	j.IO.Values = cloneCounters(j.IO.Values)
	return j
}

func unavailableWindowsJob(reason string) WindowsJobObject {
	reason = strings.TrimSpace(reason)
	if reason == "" {
		reason = "Windows Job Object attribution unavailable"
	}
	metric := func(unit string) Metric { return unavailable(unit, reason) }
	return WindowsJobObject{
		State:  WindowsJobStateUnavailable,
		Reason: reason,
		Membership: WindowsJobMembership{
			AfterStart:       metric("processes"),
			AfterWait:        metric("processes"),
			UnavailableCause: reason,
		},
		CPU:       unavailableCounterSet(reason),
		Processes: unavailableCounterSet(reason),
		IO:        unavailableCounterSet(reason),
		Memory: WindowsJobMemory{
			PeakJobCommitBytes:     metric("bytes"),
			PeakProcessCommitBytes: metric("bytes"),
		},
	}
}

func (r *Report) foldWindowsJobObject(samples []Snapshot) {
	j := r.WindowsJobObject
	if j == nil || j.State != WindowsJobStateMeasured || !j.Membership.AtomicPlacement || !j.CPU.Available || !j.Cleanup.Empty || !j.Cleanup.Closed {
		return
	}
	usage100NS, ok := j.CPU.Values["usage_100ns"]
	if !ok || usage100NS > math.MaxUint64/100 {
		j.State = WindowsJobStateUnavailable
		j.Reason = "JobObjectBasicAccountingInformation lacks a valid usage_100ns counter"
		return
	}
	first, last, hostOK := hostCPUEndpoints(samples)
	if hostOK && last.TotalCPUNS > first.TotalCPUNS && last.BusyCPUNS >= first.BusyCPUNS {
		total := last.TotalCPUNS - first.TotalCPUNS
		busy := last.BusyCPUNS - first.BusyCPUNS
		sutCPU := usage100NS * 100
		if busy <= total && sutCPU <= busy {
			r.Attribution.SUTCPUPercentOfHost = available(float64(sutCPU)/float64(total)*100, "percent", "Windows Job Object cumulative CPU accounting")
			r.Attribution.NonSUTCPUPercentOfHost = available(float64(busy-sutCPU)/float64(total)*100, "percent", "host busy CPU minus Windows Job Object cumulative CPU")
		} else {
			r.Attribution.SUTCPUPercentOfHost = unavailable("percent", "Windows Job Object SUT CPU exceeds aligned host busy CPU")
			r.Attribution.NonSUTCPUPercentOfHost = unavailable("percent", "non-SUT residual is inconsistent with Windows Job Object CPU")
		}
	}
	// Job commit is deliberately not copied into SUTRSSBytes: commit and resident
	// working set are different memory axes.
	r.Coverage.DescendantAttribution = "job_object"
}

func (r *Report) applyWindowsJobPolicy() {
	if r.WindowsJobObject == nil || !r.WindowsJobObject.Cleanup.Attempted {
		return
	}
	if !r.WindowsJobObject.Cleanup.Empty || !r.WindowsJobObject.Cleanup.Closed {
		detail := r.WindowsJobObject.Cleanup.Reason
		if detail == "" {
			detail = "Windows Job Object cleanup did not prove an empty, closed job"
		}
		r.Findings = append(r.Findings, Finding{Code: "JOB_OBJECT_CLEANUP_FAILED", Detail: detail})
		r.Verdict = VerdictInvalid
	}
}

func (j WindowsJobObject) validate() error {
	if j.State != WindowsJobStateMeasured && j.State != WindowsJobStateUnavailable {
		return fmt.Errorf("systembaseline: Windows Job Object state must be measured or unavailable")
	}
	for name, set := range map[string]CounterSet{"job CPU": j.CPU, "job processes": j.Processes, "job I/O": j.IO} {
		if err := validateCounterSet(name, set); err != nil {
			return err
		}
	}
	metrics := []Metric{j.Membership.AfterStart, j.Membership.AfterWait, j.Memory.PeakJobCommitBytes, j.Memory.PeakProcessCommitBytes}
	for _, metric := range metrics {
		if metric.Available && (math.IsNaN(metric.Value) || math.IsInf(metric.Value, 0) || metric.Value < 0 || metric.Unit == "" || metric.Source == "" || metric.Reason != "") {
			return fmt.Errorf("systembaseline: Windows Job Object metric has inconsistent measured state")
		}
		if !metric.Available && (metric.Unit == "" || metric.Reason == "" || metric.Value != 0 || metric.Source != "") {
			return fmt.Errorf("systembaseline: Windows Job Object metric has inconsistent unavailable state")
		}
	}
	if j.State == WindowsJobStateMeasured {
		if j.Reason != "" || !j.Membership.AtomicPlacement || j.Membership.RootPID <= 0 || j.Membership.RootStartID == 0 || !j.CPU.Available || !j.Processes.Available || !j.IO.Available || !j.Cleanup.Attempted || !j.Cleanup.Empty || !j.Cleanup.Closed {
			return fmt.Errorf("systembaseline: measured Windows Job Object attribution lacks verified membership, accounting, or cleanup")
		}
		if _, ok := j.CPU.Values["usage_100ns"]; !ok {
			return fmt.Errorf("systembaseline: measured Windows Job Object CPU accounting lacks usage_100ns")
		}
	} else if j.Reason == "" {
		return fmt.Errorf("systembaseline: unavailable Windows Job Object attribution needs a reason")
	}
	if j.Cleanup.Attempted && j.Cleanup.Empty && j.Cleanup.Closed && j.Cleanup.Reason != "" {
		return fmt.Errorf("systembaseline: successful Windows Job Object cleanup carries a failure reason")
	}
	return nil
}
