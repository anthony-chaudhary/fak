package systembaseline

import (
	"fmt"
	"math"
	"os/exec"
	"strings"
)

const (
	CgroupStateMeasured    = "measured"
	CgroupStateUnavailable = "unavailable"
)

// CounterSet preserves a complete numeric cgroup key/value file as one typed
// measured/unavailable unit.
type CounterSet struct {
	Available bool              `json:"available"`
	Values    map[string]uint64 `json:"values,omitempty"`
	Source    string            `json:"source,omitempty"`
	Reason    string            `json:"reason,omitempty"`
}

// PressureAxis carries PSI cumulative stall time. CPU intentionally reports
// full as unavailable on kernels that expose only the some line.
type PressureAxis struct {
	Some Metric `json:"some_total_us"`
	Full Metric `json:"full_total_us"`
}

type CgroupPressure struct {
	CPU    PressureAxis `json:"cpu"`
	Memory PressureAxis `json:"memory"`
	IO     PressureAxis `json:"io"`
}

type CgroupMembership struct {
	AtomicPlacement  bool   `json:"atomic_placement"`
	RootPID          int    `json:"root_pid"`
	AfterStart       Metric `json:"members_after_start"`
	AfterWait        Metric `json:"members_after_wait"`
	PlacementSource  string `json:"placement_source,omitempty"`
	UnavailableCause string `json:"unavailable_reason,omitempty"`
}

type CgroupMemory struct {
	CurrentBytes Metric     `json:"current_bytes"`
	PeakBytes    Metric     `json:"peak_bytes"`
	Events       CounterSet `json:"events"`
}

// CgroupCPUCapacity keeps sustained quota separate from runtime scheduler width.
// CapacityCPUs is quota/period and is not CPU affinity.
type CgroupCPUCapacity struct {
	Available                     bool    `json:"available"`
	CapacityCPUs                  float64 `json:"capacity_cpus,omitempty"`
	RuntimeWidth                  int     `json:"runtime_width,omitempty"`
	QuotaUS                       uint64  `json:"quota_us,omitempty"`
	PeriodUS                      uint64  `json:"period_us,omitempty"`
	EffectivePath                 string  `json:"effective_path,omitempty"`
	MembershipPath                string  `json:"membership_path,omitempty"`
	RuntimeDefaultMayOverestimate bool    `json:"runtime_default_may_overestimate,omitempty"`
	Source                        string  `json:"source,omitempty"`
	Reason                        string  `json:"reason,omitempty"`
}

type CgroupCleanup struct {
	Attempted       bool   `json:"attempted"`
	KilledRemaining bool   `json:"killed_remaining"`
	Empty           bool   `json:"empty"`
	Removed         bool   `json:"removed"`
	Reason          string `json:"reason,omitempty"`
}

// CgroupV2 is the bounded per-command Linux cgroup attribution receipt. State
// is measured only when atomic membership and all core counters were verified.
type CgroupV2 struct {
	State       string             `json:"state"`
	Reason      string             `json:"reason,omitempty"`
	Membership  CgroupMembership   `json:"membership"`
	CPU         CounterSet         `json:"cpu_stat"`
	CPUCapacity *CgroupCPUCapacity `json:"cpu_capacity,omitempty"`
	Memory      CgroupMemory       `json:"memory"`
	Pressure    CgroupPressure     `json:"pressure"`
	Cleanup     CgroupCleanup      `json:"cleanup"`
}

func (c CgroupV2) clone() CgroupV2 {
	c.CPU.Values = cloneCounters(c.CPU.Values)
	if c.CPUCapacity != nil {
		capacity := *c.CPUCapacity
		c.CPUCapacity = &capacity
	}
	c.Memory.Events.Values = cloneCounters(c.Memory.Events.Values)
	return c
}

func cloneCounters(in map[string]uint64) map[string]uint64 {
	if in == nil {
		return nil
	}
	out := make(map[string]uint64, len(in))
	for key, value := range in {
		out[key] = value
	}
	return out
}

type commandAttributorPlatform interface {
	configure(*exec.Cmd) bool
	active() bool
	started(int) error
	launchFailed(error)
	finish() CgroupV2
}

// CommandAttribution is the OS-neutral result of command-bound attribution.
// Exactly one platform receipt is normally present. The pointers keep an
// unavailable platform from adding a misleading receipt to the report.
type CommandAttribution struct {
	CgroupV2         *CgroupV2         `json:"cgroup_v2,omitempty"`
	WindowsJobObject *WindowsJobObject `json:"windows_job_object,omitempty"`
}

type commandAttributionFinisher interface {
	finishAttribution() CommandAttribution
}

// CommandAttributor owns the optional OS boundary used by the command wrapper.
// Call Configure before Start, Started after a successful Start, and
// FinishAttribution exactly once after Wait. Finish remains the Linux-specific
// compatibility surface.
type CommandAttributor struct {
	platform commandAttributorPlatform
}

func NewCommandAttributor() *CommandAttributor {
	return &CommandAttributor{platform: newCommandAttributorPlatform()}
}

func (a *CommandAttributor) Configure(cmd *exec.Cmd) bool {
	return a != nil && a.platform != nil && a.platform.configure(cmd)
}

func (a *CommandAttributor) Active() bool {
	return a != nil && a.platform != nil && a.platform.active()
}

func (a *CommandAttributor) Started(pid int) error {
	if a != nil && a.platform != nil {
		return a.platform.started(pid)
	}
	return nil
}

func (a *CommandAttributor) LaunchFailed(err error) {
	if a != nil && a.platform != nil {
		a.platform.launchFailed(err)
	}
}

func (a *CommandAttributor) Finish() CgroupV2 {
	if a == nil || a.platform == nil {
		return unavailableCgroup("command attribution adapter unavailable")
	}
	return a.platform.finish()
}

// FinishAttribution finalizes the active platform boundary. It is idempotent;
// callers should prefer it over Finish when constructing a cross-platform
// report.
func (a *CommandAttributor) FinishAttribution() CommandAttribution {
	if a == nil || a.platform == nil {
		return CommandAttribution{}
	}
	if finisher, ok := a.platform.(commandAttributionFinisher); ok {
		return finisher.finishAttribution()
	}
	cgroup := a.platform.finish()
	return CommandAttribution{CgroupV2: &cgroup}
}

func unavailableCgroup(reason string) CgroupV2 {
	reason = strings.TrimSpace(reason)
	if reason == "" {
		reason = "cgroup v2 attribution unavailable"
	}
	metric := func(unit string) Metric { return unavailable(unit, reason) }
	axis := PressureAxis{Some: metric("microseconds"), Full: metric("microseconds")}
	return CgroupV2{
		State:  CgroupStateUnavailable,
		Reason: reason,
		Membership: CgroupMembership{
			AfterStart:       metric("processes"),
			AfterWait:        metric("processes"),
			UnavailableCause: reason,
		},
		CPU: CounterSet{Reason: reason},
		Memory: CgroupMemory{
			CurrentBytes: metric("bytes"),
			PeakBytes:    metric("bytes"),
			Events:       CounterSet{Reason: reason},
		},
		Pressure: CgroupPressure{CPU: axis, Memory: axis, IO: axis},
	}
}

func availableCounterSet(values map[string]uint64, source string) CounterSet {
	return CounterSet{Available: true, Values: cloneCounters(values), Source: source}
}

func unavailableCounterSet(reason string) CounterSet {
	return CounterSet{Reason: reason}
}

func (r *Report) foldCgroupV2(samples []Snapshot) {
	c := r.CgroupV2
	if c == nil || c.State != CgroupStateMeasured || !c.Membership.AtomicPlacement || !c.CPU.Available || !c.Memory.CurrentBytes.Available || !c.Memory.PeakBytes.Available || !c.Memory.Events.Available {
		return
	}
	usageUS, ok := c.CPU.Values["usage_usec"]
	if !ok || usageUS > math.MaxUint64/1000 {
		c.State = CgroupStateUnavailable
		c.Reason = "cpu.stat lacks a valid usage_usec counter"
		return
	}
	first, last, hostOK := hostCPUEndpoints(samples)
	if hostOK && last.TotalCPUNS > first.TotalCPUNS && last.BusyCPUNS >= first.BusyCPUNS {
		total := last.TotalCPUNS - first.TotalCPUNS
		busy := last.BusyCPUNS - first.BusyCPUNS
		sutCPU := usageUS * 1000
		if busy <= total && sutCPU <= busy {
			r.Attribution.SUTCPUPercentOfHost = available(float64(sutCPU)/float64(total)*100, "percent", "delegated cgroup v2 cpu.stat usage_usec")
			r.Attribution.NonSUTCPUPercentOfHost = available(float64(busy-sutCPU)/float64(total)*100, "percent", "host busy CPU minus delegated cgroup v2 CPU")
		} else {
			r.Attribution.SUTCPUPercentOfHost = unavailable("percent", "cgroup SUT CPU exceeds aligned host busy CPU")
			r.Attribution.NonSUTCPUPercentOfHost = unavailable("percent", "non-SUT residual is inconsistent with cgroup CPU")
		}
	}
	r.Attribution.SUTRSSBytes = c.Memory.PeakBytes
	r.Coverage.DescendantAttribution = "exact_cgroup_v2"
}

func pressureStallPercent(metric Metric, durationNS int64) (float64, bool) {
	if !metric.Available || durationNS <= 0 {
		return 0, false
	}
	return metric.Value * 1000 / float64(durationNS) * 100, true
}

func (r *Report) applyCgroupPolicy() {
	if r.CgroupV2 == nil {
		return
	}
	if r.CgroupV2.Cleanup.Attempted && (!r.CgroupV2.Cleanup.Empty || !r.CgroupV2.Cleanup.Removed) {
		r.Findings = append(r.Findings, Finding{Code: "CGROUP_CLEANUP_FAILED", Detail: r.CgroupV2.Cleanup.Reason})
		if r.Verdict != VerdictInvalid {
			r.Verdict = VerdictInvestigate
		}
	}
	axes := []struct {
		name string
		axis PressureAxis
	}{
		{"CPU", r.CgroupV2.Pressure.CPU},
		{"MEMORY", r.CgroupV2.Pressure.Memory},
		{"IO", r.CgroupV2.Pressure.IO},
	}
	for _, item := range axes {
		maxPercent := float64(0)
		measured := false
		for _, metric := range []Metric{item.axis.Some, item.axis.Full} {
			if pct, ok := pressureStallPercent(metric, r.Window.DurationNS); ok {
				measured = true
				if pct > maxPercent {
					maxPercent = pct
				}
			}
		}
		if measured && maxPercent > r.Policy.MaximumPSIStallPercent {
			r.Findings = append(r.Findings, Finding{
				Code:   item.name + "_PSI_STALL_HIGH",
				Detail: fmt.Sprintf("%s cgroup PSI stall %.2f%% exceeds %.2f%%", strings.ToLower(item.name), maxPercent, r.Policy.MaximumPSIStallPercent),
			})
			if r.Verdict != VerdictInvalid {
				r.Verdict = VerdictInvestigate
			}
		}
	}
}

func validateCounterSet(name string, set CounterSet) error {
	if set.Available {
		if set.Values == nil || set.Source == "" || set.Reason != "" {
			return fmt.Errorf("systembaseline: measured %s counters need values and source without an unavailable reason", name)
		}
		return nil
	}
	if set.Reason == "" || len(set.Values) != 0 {
		return fmt.Errorf("systembaseline: unavailable %s counters need a reason and no values", name)
	}
	return nil
}

func (c CgroupV2) validate() error {
	if c.State != CgroupStateMeasured && c.State != CgroupStateUnavailable {
		return errorsNewCgroupState()
	}
	if err := validateCounterSet("cgroup cpu.stat", c.CPU); err != nil {
		return err
	}
	if err := validateCounterSet("cgroup memory.events", c.Memory.Events); err != nil {
		return err
	}
	metrics := []Metric{
		c.Membership.AfterStart, c.Membership.AfterWait,
		c.Memory.CurrentBytes, c.Memory.PeakBytes,
		c.Pressure.CPU.Some, c.Pressure.CPU.Full,
		c.Pressure.Memory.Some, c.Pressure.Memory.Full,
		c.Pressure.IO.Some, c.Pressure.IO.Full,
	}
	for _, metric := range metrics {
		if metric.Available && (math.IsNaN(metric.Value) || math.IsInf(metric.Value, 0) || metric.Value < 0 || metric.Unit == "" || metric.Source == "" || metric.Reason != "") {
			return fmt.Errorf("systembaseline: cgroup metric has inconsistent measured state")
		}
		if !metric.Available && (metric.Unit == "" || metric.Reason == "" || metric.Value != 0 || metric.Source != "") {
			return fmt.Errorf("systembaseline: cgroup metric has inconsistent unavailable state")
		}
	}
	if capacity := c.CPUCapacity; capacity != nil {
		if capacity.Available {
			if capacity.CapacityCPUs <= 0 || math.IsNaN(capacity.CapacityCPUs) || math.IsInf(capacity.CapacityCPUs, 0) || capacity.RuntimeWidth <= 0 || capacity.QuotaUS == 0 || capacity.PeriodUS == 0 || capacity.EffectivePath == "" || capacity.MembershipPath == "" || capacity.Source == "" || capacity.Reason != "" {
				return fmt.Errorf("systembaseline: measured cgroup CPU capacity is incomplete")
			}
		} else if capacity.Reason == "" || capacity.CapacityCPUs != 0 || capacity.RuntimeWidth != 0 || capacity.QuotaUS != 0 || capacity.PeriodUS != 0 || capacity.Source != "" {
			return fmt.Errorf("systembaseline: unavailable cgroup CPU capacity is inconsistent")
		}
	}
	if c.State == CgroupStateMeasured {
		if c.Reason != "" || !c.Membership.AtomicPlacement || c.Membership.RootPID <= 0 || !c.CPU.Available || !c.Memory.CurrentBytes.Available || !c.Memory.PeakBytes.Available || !c.Memory.Events.Available {
			return fmt.Errorf("systembaseline: measured cgroup attribution lacks verified membership or core counters")
		}
		if _, ok := c.CPU.Values["usage_usec"]; !ok {
			return fmt.Errorf("systembaseline: measured cgroup cpu.stat lacks usage_usec")
		}
	} else if c.Reason == "" {
		return fmt.Errorf("systembaseline: unavailable cgroup attribution needs a reason")
	}
	if c.Cleanup.Attempted && c.Cleanup.Empty && c.Cleanup.Removed && c.Cleanup.Reason != "" {
		return fmt.Errorf("systembaseline: successful cgroup cleanup carries a failure reason")
	}
	return nil
}

func errorsNewCgroupState() error {
	return fmt.Errorf("systembaseline: cgroup state must be measured or unavailable")
}
