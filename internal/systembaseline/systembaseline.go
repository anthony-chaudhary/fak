package systembaseline

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

const Schema = "fak.system-baseline/v1"

const (
	linuxClockTicks = uint64(100)
	// Operational reports disclose at most five platform-bounded image names;
	// this generous ceiling keeps hostile Decode readers from growing memory.
	maxEncodedReportBytes = 1 << 20
)

const (
	VerdictClean       = "clean"
	VerdictInvestigate = "investigate"
	VerdictInvalid     = "invalid"
)

type Metric struct {
	Available bool    `json:"available"`
	Value     float64 `json:"value,omitempty"`
	Unit      string  `json:"unit"`
	Source    string  `json:"source,omitempty"`
	Reason    string  `json:"reason,omitempty"`
}

type Policy struct {
	MinimumBaselineSamples    int     `json:"minimum_baseline_samples"`
	MinimumSamples            int     `json:"minimum_samples"`
	MaximumNonSUTCPUPercent   float64 `json:"maximum_non_sut_cpu_percent"`
	MaximumSamplerDutyPercent float64 `json:"maximum_sampler_duty_percent"`
	MaximumPSIStallPercent    float64 `json:"maximum_psi_stall_percent,omitempty"`
	MinimumCensusIntervalNS   int64   `json:"minimum_census_interval_ns"`
	MaximumCensusIntervalNS   int64   `json:"maximum_census_interval_ns"`
	RequireHostCPU            bool    `json:"require_host_cpu"`
	RequireSUTCPU             bool    `json:"require_sut_cpu"`
	RequireProcessMemory      bool    `json:"require_process_memory"`
	IncludeTopConsumers       bool    `json:"include_top_consumers"`
}

func DefaultPolicy() Policy {
	return Policy{MinimumBaselineSamples: 2, MinimumSamples: 2, MaximumNonSUTCPUPercent: 20, MaximumSamplerDutyPercent: 10, MaximumPSIStallPercent: 5, MinimumCensusIntervalNS: int64(10 * time.Millisecond), MaximumCensusIntervalNS: int64(time.Second), RequireHostCPU: true, RequireSUTCPU: true}
}

type Window struct {
	StartedAtUTC string `json:"started_at_utc"`
	EndedAtUTC   string `json:"ended_at_utc"`
	DurationNS   int64  `json:"duration_ns"`
	IntervalNS   int64  `json:"sample_interval_ns"`
	Samples      int    `json:"samples"`
}

type Coverage struct {
	ProcessSnapshots       int    `json:"process_snapshots"`
	ProcessesObserved      int    `json:"processes_observed"`
	ProcessReads           int    `json:"process_reads"`
	ProcessUnreadable      int    `json:"process_unreadable"`
	HostCPUSamples         int    `json:"host_cpu_samples"`
	HostMemorySamples      int    `json:"host_memory_samples"`
	SUTRootPID             int    `json:"sut_root_pid"`
	DescendantAttribution  string `json:"descendant_attribution"`
	ProcessEnumerationNote string `json:"process_enumeration_note,omitempty"`
}

type HostTotals struct {
	CPUPercent           Metric `json:"cpu_percent"`
	MemoryTotalBytes     Metric `json:"memory_total_bytes"`
	MemoryAvailableBytes Metric `json:"memory_available_bytes"`
	ProcessCount         Metric `json:"process_count"`
}

type Attribution struct {
	SUTCPUPercentOfHost    Metric `json:"sut_cpu_percent_of_host"`
	NonSUTCPUPercentOfHost Metric `json:"non_sut_cpu_percent_of_host"`
	SUTRSSBytes            Metric `json:"sut_rss_bytes"`
	NonSUTRSSBytes         Metric `json:"non_sut_rss_bytes"`
}

type Consumer struct {
	Image        string  `json:"image"`
	PID          int     `json:"pid"`
	CPUSeconds   float64 `json:"cpu_seconds,omitempty"`
	RSSBytes     uint64  `json:"rss_bytes,omitempty"`
	CPUAvailable bool    `json:"cpu_available"`
	RSSAvailable bool    `json:"rss_available"`
}

type Finding struct {
	Code   string `json:"code"`
	Detail string `json:"detail"`
}

type SamplerOverhead struct {
	CountedSamples     int              `json:"counted_samples"`
	WallNS             int64            `json:"wall_ns"`
	DutyPercent        Metric           `json:"duty_percent"`
	EffectiveCadenceNS int64            `json:"effective_cadence_ns"`
	Stages             map[string]int64 `json:"stage_wall_ns"`
	CacheHits          int              `json:"stable_identity_cache_hits"`
	CacheMisses        int              `json:"stable_identity_cache_misses"`
	CoverageLimited    bool             `json:"coverage_limited"`
}

type Report struct {
	Schema          string          `json:"schema"`
	Verdict         string          `json:"verdict"`
	Baseline        Window          `json:"baseline_window"`
	BaselineHost    HostTotals      `json:"baseline_host_totals"`
	BaselineSampler SamplerOverhead `json:"baseline_sampler"`
	Window          Window          `json:"command_window"`
	CommandSampler  SamplerOverhead `json:"command_sampler"`
	Coverage        Coverage        `json:"coverage"`
	Host            HostTotals      `json:"host_totals"`
	Attribution     Attribution     `json:"attribution"`
	CgroupV2        *CgroupV2       `json:"cgroup_v2,omitempty"`
	TopNonSUT       []Consumer      `json:"top_non_sut_consumers"`
	Policy          Policy          `json:"policy"`
	Findings        []Finding       `json:"findings"`
	CommandExitCode int             `json:"command_exit_code"`
	TimedOut        bool            `json:"timed_out"`
	Digest          string          `json:"digest"`
}

// HostSample contains cumulative CPU nanoseconds across all logical CPUs and
// instantaneous memory values. Only deltas between snapshots are reported.
type HostSample struct {
	CPUAvailable    bool
	TotalCPUNS      uint64
	BusyCPUNS       uint64
	MemoryAvailable bool
	MemoryTotal     uint64
	MemoryFree      uint64
}

type ProcessSample struct {
	PID, PPID    int
	StartID      uint64
	Image        string
	CPUAvailable bool
	CPUNS        uint64
	RSSAvailable bool
	RSSBytes     uint64
}

type Snapshot struct {
	At                    time.Time
	Host                  HostSample
	Processes             []ProcessSample
	ProcessEnumerationOK  bool
	AttributionIncomplete bool
	CensusWallNS          int64
	CensusStages          map[string]int64
	EffectiveCadenceNS    int64
	StableCacheHits       int
	StableCacheMisses     int
	CoverageLimited       bool
	ProcessUnreadable     int
	ProcessNote           string
}

func Capture() Snapshot {
	started := time.Now()
	snapshot := capturePlatform()
	snapshot.CensusWallNS = time.Since(started).Nanoseconds()
	return snapshot
}

func Build(baselineSamples, samples []Snapshot, rootPID int, interval time.Duration, policy Policy, exitCode int, timedOut bool) Report {
	return BuildWithCgroupV2(baselineSamples, samples, rootPID, interval, policy, exitCode, timedOut, nil)
}

// BuildWithCgroupV2 builds the ordinary host/process report and, when a
// verified delegated cgroup is supplied, replaces sampled SUT CPU and memory
// with the cgroup's cumulative whole-descendant counters.
func BuildWithCgroupV2(baselineSamples, samples []Snapshot, rootPID int, interval time.Duration, policy Policy, exitCode int, timedOut bool, cgroup *CgroupV2) Report {
	r := Report{Schema: Schema, Verdict: VerdictClean, Policy: normalizePolicy(policy), CommandExitCode: exitCode, TimedOut: timedOut, TopNonSUT: []Consumer{}, Findings: []Finding{}}
	if cgroup != nil {
		copy := cgroup.clone()
		r.CgroupV2 = &copy
	}
	if len(baselineSamples) > 0 {
		baselineSamples = append([]Snapshot(nil), baselineSamples...)
		sort.SliceStable(baselineSamples, func(i, j int) bool { return baselineSamples[i].At.Before(baselineSamples[j].At) })
		r.Baseline = windowFor(baselineSamples, interval)
		r.BaselineHost = foldHostTotals(baselineSamples)
		r.BaselineSampler = foldSamplerOverhead(baselineSamples, r.Baseline)
	}
	if len(samples) == 0 {
		r.Verdict = VerdictInvalid
		r.Findings = append(r.Findings, Finding{Code: "NO_SAMPLES", Detail: "no system samples were captured"})
		r.Seal()
		return r
	}
	samples = append([]Snapshot(nil), samples...)
	sort.SliceStable(samples, func(i, j int) bool { return samples[i].At.Before(samples[j].At) })
	r.Window = windowFor(samples, interval)
	r.CommandSampler = foldSamplerOverhead(samples, r.Window)
	r.Coverage.SUTRootPID = rootPID
	r.Coverage.DescendantAttribution = "sampled_pid_ppid_tree"
	maxProcesses := 0
	for _, s := range samples {
		if s.Host.CPUAvailable {
			r.Coverage.HostCPUSamples++
		}
		if s.Host.MemoryAvailable {
			r.Coverage.HostMemorySamples++
		}
		if s.ProcessEnumerationOK {
			r.Coverage.ProcessSnapshots++
		}
		r.Coverage.ProcessesObserved += len(s.Processes)
		for _, p := range s.Processes {
			if p.CPUAvailable || p.RSSAvailable {
				r.Coverage.ProcessReads++
			}
		}
		r.Coverage.ProcessUnreadable += s.ProcessUnreadable
		if len(s.Processes) > maxProcesses {
			maxProcesses = len(s.Processes)
		}
		if s.ProcessNote != "" {
			r.Coverage.ProcessEnumerationNote = s.ProcessNote
		}
	}
	r.foldHost(samples)
	if r.Coverage.ProcessSnapshots > 0 {
		r.Host.ProcessCount = available(float64(maxProcesses), "processes", "platform process census")
	} else {
		r.Host.ProcessCount = unavailable("processes", "process census unavailable")
	}
	r.foldProcesses(samples, rootPID)
	r.foldCgroupV2(samples)
	r.applyPolicy()
	r.Seal()
	return r
}

func normalizePolicy(p Policy) Policy {
	if p.MinimumBaselineSamples <= 0 {
		p.MinimumBaselineSamples = 2
	}
	if p.MinimumSamples <= 0 {
		p.MinimumSamples = 2
	}
	if p.MaximumNonSUTCPUPercent <= 0 || p.MaximumNonSUTCPUPercent > 100 || math.IsNaN(p.MaximumNonSUTCPUPercent) || math.IsInf(p.MaximumNonSUTCPUPercent, 0) {
		p.MaximumNonSUTCPUPercent = 20
	}
	if p.MaximumSamplerDutyPercent <= 0 || p.MaximumSamplerDutyPercent > 100 || math.IsNaN(p.MaximumSamplerDutyPercent) || math.IsInf(p.MaximumSamplerDutyPercent, 0) {
		p.MaximumSamplerDutyPercent = 10
	}
	if p.MaximumPSIStallPercent < 0 || p.MaximumPSIStallPercent > 100 || math.IsNaN(p.MaximumPSIStallPercent) || math.IsInf(p.MaximumPSIStallPercent, 0) {
		p.MaximumPSIStallPercent = 5
	}
	if p.MinimumCensusIntervalNS <= 0 {
		p.MinimumCensusIntervalNS = int64(10 * time.Millisecond)
	}
	if p.MaximumCensusIntervalNS < p.MinimumCensusIntervalNS {
		p.MaximumCensusIntervalNS = int64(time.Second)
	}
	if p.MaximumCensusIntervalNS < p.MinimumCensusIntervalNS {
		p.MaximumCensusIntervalNS = p.MinimumCensusIntervalNS
	}
	return p
}

func foldSamplerOverhead(samples []Snapshot, window Window) SamplerOverhead {
	overhead := SamplerOverhead{Stages: map[string]int64{}}
	if len(samples) < 2 || window.DurationNS <= 0 {
		overhead.DutyPercent = unavailable("percent", "sampler window unavailable")
		return overhead
	}
	overhead.CountedSamples = len(samples) - 1
	for _, sample := range samples[:len(samples)-1] {
		if sample.EffectiveCadenceNS > 0 {
			overhead.EffectiveCadenceNS = sample.EffectiveCadenceNS
		}
		overhead.CacheHits += sample.StableCacheHits
		overhead.CacheMisses += sample.StableCacheMisses
		overhead.CoverageLimited = overhead.CoverageLimited || sample.CoverageLimited
		for stage, ns := range sample.CensusStages {
			overhead.Stages[stage] += ns
		}
		if sample.CensusWallNS < 0 {
			overhead.WallNS = 0
			overhead.DutyPercent = unavailable("percent", "negative census wall duration")
			return overhead
		}
		if sample.CensusWallNS > math.MaxInt64-overhead.WallNS {
			overhead.WallNS = 0
			overhead.DutyPercent = unavailable("percent", "census wall duration overflow")
			return overhead
		}
		overhead.WallNS += sample.CensusWallNS
	}
	overhead.DutyPercent = available(float64(overhead.WallNS)/float64(window.DurationNS)*100, "percent", "sampler census wall time before final counter endpoint")
	return overhead
}
func windowFor(samples []Snapshot, interval time.Duration) Window {
	if len(samples) == 0 {
		return Window{}
	}
	intervalNS := interval.Nanoseconds()
	if intervalNS < 0 {
		intervalNS = 0
	}
	return Window{StartedAtUTC: samples[0].At.UTC().Format(time.RFC3339Nano), EndedAtUTC: samples[len(samples)-1].At.UTC().Format(time.RFC3339Nano), DurationNS: samples[len(samples)-1].At.Sub(samples[0].At).Nanoseconds(), IntervalNS: intervalNS, Samples: len(samples)}
}

// canonicalLinuxCPUTicks excludes guest counters because Linux already includes
// them in user and nice. It lives here so the counter contract is testable on
// every development platform.
func canonicalLinuxCPUTicks(ticks []uint64) (total, idle uint64, ok bool) {
	if len(ticks) < 4 {
		return 0, 0, false
	}
	limit := len(ticks)
	if limit > 8 {
		limit = 8
	}
	for _, tick := range ticks[:limit] {
		if tick > math.MaxUint64-total {
			return 0, 0, false
		}
		total += tick
	}
	idle = ticks[3]
	if len(ticks) > 4 {
		idle += ticks[4]
	}
	return total, idle, total >= idle
}

func linuxCPUTicksToNS(ticks uint64) (uint64, bool) {
	const nsPerSecond = uint64(time.Second)
	whole, remainder := ticks/linuxClockTicks, ticks%linuxClockTicks
	fraction := remainder * nsPerSecond / linuxClockTicks
	if whole > (math.MaxUint64-fraction)/nsPerSecond {
		return 0, false
	}
	return whole*nsPerSecond + fraction, true
}

// canonicalWindowsHostCPUNS keeps the FILETIME conversion testable without a
// Windows kernel. GetSystemTimes includes idle time in the kernel counter.
func canonicalWindowsHostCPUNS(idle, kernel, user uint64) (total, busy uint64, ok bool) {
	if kernel < idle || user > math.MaxUint64-kernel {
		return 0, 0, false
	}
	totalTicks := kernel + user
	busyTicks := kernel - idle
	if user > math.MaxUint64-busyTicks {
		return 0, 0, false
	}
	busyTicks += user
	total, totalOK := windows100NSTicksToNS(totalTicks)
	busy, busyOK := windows100NSTicksToNS(busyTicks)
	if !totalOK || !busyOK {
		return 0, 0, false
	}
	return total, busy, true
}

func canonicalWindowsProcessCPUNS(kernel, user uint64) (uint64, bool) {
	if user > math.MaxUint64-kernel {
		return 0, false
	}
	return windows100NSTicksToNS(kernel + user)
}

func windows100NSTicksToNS(ticks uint64) (uint64, bool) {
	if ticks > math.MaxUint64/100 {
		return 0, false
	}
	return ticks * 100, true
}

func (r *Report) foldHost(samples []Snapshot) {
	r.Host = foldHostTotals(samples)
}

func foldHostTotals(samples []Snapshot) HostTotals {
	var out HostTotals
	first, last, ok := hostCPUEndpoints(samples)
	if ok && last.TotalCPUNS > first.TotalCPUNS && last.BusyCPUNS >= first.BusyCPUNS {
		total, busy := last.TotalCPUNS-first.TotalCPUNS, last.BusyCPUNS-first.BusyCPUNS
		if busy <= total {
			out.CPUPercent = available(float64(busy)/float64(total)*100, "percent", "cumulative host CPU counters")
		} else {
			out.CPUPercent = unavailable("percent", "host busy CPU exceeds total CPU delta")
		}
	} else {
		out.CPUPercent = unavailable("percent", "host CPU counter delta unavailable")
	}
	var total, minFree uint64
	maxProcesses := -1
	for _, s := range samples {
		if s.ProcessEnumerationOK && len(s.Processes) > maxProcesses {
			maxProcesses = len(s.Processes)
		}
		if !s.Host.MemoryAvailable {
			continue
		}
		if total == 0 {
			total = s.Host.MemoryTotal
		}
		if minFree == 0 || s.Host.MemoryFree < minFree {
			minFree = s.Host.MemoryFree
		}
	}
	if total > 0 {
		out.MemoryTotalBytes = available(float64(total), "bytes", "host memory reader")
		out.MemoryAvailableBytes = available(float64(minFree), "bytes", "minimum sampled host-available memory")
	} else {
		out.MemoryTotalBytes = unavailable("bytes", "host memory unavailable")
		out.MemoryAvailableBytes = unavailable("bytes", "host memory unavailable")
	}
	if maxProcesses >= 0 {
		out.ProcessCount = available(float64(maxProcesses), "processes", "platform process census")
	} else {
		out.ProcessCount = unavailable("processes", "process census unavailable")
	}
	return out
}

func hostCPUEndpoints(samples []Snapshot) (HostSample, HostSample, bool) {
	if len(samples) < 2 || !samples[0].Host.CPUAvailable || !samples[len(samples)-1].Host.CPUAvailable {
		return HostSample{}, HostSample{}, false
	}
	return samples[0].Host, samples[len(samples)-1].Host, true
}

type procFold struct {
	firstCPU, lastCPU uint64
	cpuReads          int
	maxRSS            uint64
	haveRSS           bool
	image             string
	pid               int
	sut               bool
}

func (r *Report) foldProcesses(samples []Snapshot, rootPID int) {
	byKey := map[string]*procFold{}
	sutAttributionComplete := r.Coverage.ProcessSnapshots == len(samples)
	allMemoryComplete := sutAttributionComplete && r.Coverage.ProcessUnreadable == 0
	sutIdentityComplete := true
	rootSeen := false
	var peakSUTRSS, peakNonRSS uint64
	for _, snap := range samples {
		sut := descendantSet(snap.Processes, rootPID)
		if snap.AttributionIncomplete {
			sutAttributionComplete = false
		}
		var sutRSS, nonRSS uint64
		for _, p := range snap.Processes {
			if p.StartID == 0 && sut[p.PID] {
				sutIdentityComplete = false
			}
			if !p.RSSAvailable {
				allMemoryComplete = false
			}
			if p.PID == rootPID {
				rootSeen = true
			}
			key := fmt.Sprintf("%d/%d", p.PID, p.StartID)
			f := byKey[key]
			if f == nil {
				f = &procFold{image: scrubImage(p.Image), pid: p.PID}
				byKey[key] = f
			}
			if sut[p.PID] {
				f.sut = true
				if !p.CPUAvailable {
					sutAttributionComplete = false
				}
			}
			if p.CPUAvailable {
				if f.cpuReads == 0 {
					f.firstCPU = p.CPUNS
				}
				f.lastCPU = p.CPUNS
				f.cpuReads++
			}
			if p.RSSAvailable {
				f.haveRSS = true
				if p.RSSBytes > f.maxRSS {
					f.maxRSS = p.RSSBytes
				}
				if sut[p.PID] {
					sutRSS += p.RSSBytes
				} else {
					nonRSS += p.RSSBytes
				}
			}
		}
		if sutRSS > peakSUTRSS {
			peakSUTRSS = sutRSS
		}
		if nonRSS > peakNonRSS {
			peakNonRSS = nonRSS
		}
	}
	if !rootSeen {
		sutAttributionComplete = false
	}
	if !sutIdentityComplete {
		sutAttributionComplete = false
	}
	var sutCPU uint64
	var top []Consumer
	for _, f := range byKey {
		haveCPU := f.cpuReads >= 2 && f.lastCPU >= f.firstCPU
		var delta uint64
		if haveCPU {
			delta = f.lastCPU - f.firstCPU
		}
		if f.sut {
			if !haveCPU {
				sutAttributionComplete = false
			} else {
				sutCPU += delta
			}
			continue
		}
		top = append(top, Consumer{Image: f.image, PID: f.pid, CPUSeconds: float64(delta) / 1e9, RSSBytes: f.maxRSS, CPUAvailable: haveCPU, RSSAvailable: f.haveRSS})
	}
	sort.Slice(top, func(i, j int) bool {
		if top[i].CPUSeconds != top[j].CPUSeconds {
			return top[i].CPUSeconds > top[j].CPUSeconds
		}
		if top[i].RSSBytes != top[j].RSSBytes {
			return top[i].RSSBytes > top[j].RSSBytes
		}
		return top[i].PID < top[j].PID
	})
	if r.Policy.IncludeTopConsumers {
		if len(top) > 5 {
			top = top[:5]
		}
		r.TopNonSUT = top
	}
	first, last, hostOK := hostCPUEndpoints(samples)
	if sutAttributionComplete && hostOK && last.TotalCPUNS > first.TotalCPUNS && last.BusyCPUNS >= first.BusyCPUNS {
		total := last.TotalCPUNS - first.TotalCPUNS
		busy := last.BusyCPUNS - first.BusyCPUNS
		if busy <= total && sutCPU <= busy {
			sutPct := float64(sutCPU) / float64(total) * 100
			nonPct := float64(busy-sutCPU) / float64(total) * 100
			r.Attribution.SUTCPUPercentOfHost = available(sutPct, "percent", "root PID + descendant process CPU deltas")
			r.Attribution.NonSUTCPUPercentOfHost = available(nonPct, "percent", "host busy CPU minus attributed SUT-tree CPU")
		} else {
			r.Attribution.SUTCPUPercentOfHost = unavailable("percent", "attributed SUT CPU exceeds aligned host busy CPU")
			r.Attribution.NonSUTCPUPercentOfHost = unavailable("percent", "non-SUT residual is inconsistent with aligned host CPU")
		}
	} else {
		r.Attribution.SUTCPUPercentOfHost = unavailable("percent", "complete process-tree CPU attribution unavailable")
		r.Attribution.NonSUTCPUPercentOfHost = unavailable("percent", "non-SUT residual requires complete SUT attribution")
	}
	if allMemoryComplete {
		r.Attribution.SUTRSSBytes = available(float64(peakSUTRSS), "bytes", "peak sampled SUT-tree RSS")
		r.Attribution.NonSUTRSSBytes = available(float64(peakNonRSS), "bytes", "peak sampled non-SUT process RSS")
	} else {
		r.Attribution.SUTRSSBytes = unavailable("bytes", "complete process memory attribution unavailable")
		r.Attribution.NonSUTRSSBytes = unavailable("bytes", "complete process memory attribution unavailable")
	}
}

func descendantSet(ps []ProcessSample, root int) map[int]bool {
	out := map[int]bool{}
	if root <= 0 {
		return out
	}
	out[root] = true
	for changed := true; changed; {
		changed = false
		for _, p := range ps {
			if out[p.PID] || !out[p.PPID] {
				continue
			}
			out[p.PID] = true
			changed = true
		}
	}
	return out
}

func classifyWindowsProcessAdvance(result, errorCode uintptr) (done, truncated bool) {
	if result != 0 {
		return false, false
	}
	const errorNoMoreFiles = uintptr(18)
	return true, errorCode != errorNoMoreFiles
}

func (r *Report) applyPolicy() {
	if r.Baseline.Samples < r.Policy.MinimumBaselineSamples || r.Baseline.DurationNS <= 0 || r.Baseline.IntervalNS <= 0 {
		r.Verdict = VerdictInvalid
		r.Findings = append(r.Findings, Finding{Code: "INVALID_BASELINE_WINDOW", Detail: "quiet baseline sample count, duration, or interval is invalid"})
	}
	if r.Window.Samples < r.Policy.MinimumSamples || r.Window.DurationNS <= 0 || r.Window.IntervalNS <= 0 || r.Coverage.SUTRootPID <= 0 {
		r.Verdict = VerdictInvalid
		r.Findings = append(r.Findings, Finding{Code: "INVALID_WINDOW", Detail: "sample count, duration, interval, or SUT root is invalid"})
	}
	unknown := false
	if r.Policy.RequireHostCPU && !r.BaselineHost.CPUPercent.Available {
		unknown = true
		r.Findings = append(r.Findings, Finding{Code: "BASELINE_HOST_CPU_UNKNOWN", Detail: r.BaselineHost.CPUPercent.Reason})
	}
	if r.Policy.RequireHostCPU && !r.Host.CPUPercent.Available {
		unknown = true
		r.Findings = append(r.Findings, Finding{Code: "HOST_CPU_UNKNOWN", Detail: r.Host.CPUPercent.Reason})
	}
	if r.Policy.RequireSUTCPU && (!r.Attribution.SUTCPUPercentOfHost.Available || !r.Attribution.NonSUTCPUPercentOfHost.Available) {
		unknown = true
		r.Findings = append(r.Findings, Finding{Code: "SUT_CPU_UNKNOWN", Detail: r.Attribution.SUTCPUPercentOfHost.Reason})
	}
	if r.Policy.RequireProcessMemory && (!r.Attribution.SUTRSSBytes.Available || !r.Attribution.NonSUTRSSBytes.Available) {
		unknown = true
		r.Findings = append(r.Findings, Finding{Code: "PROCESS_MEMORY_UNKNOWN", Detail: r.Attribution.SUTRSSBytes.Reason})
	}
	if r.Verdict != VerdictInvalid && unknown {
		r.Verdict = VerdictInvestigate
	}
	if !r.BaselineSampler.DutyPercent.Available || !r.CommandSampler.DutyPercent.Available {
		r.Findings = append(r.Findings, Finding{Code: "SAMPLER_DUTY_UNKNOWN", Detail: "baseline or command sampler duty is unavailable"})
		if r.Verdict != VerdictInvalid {
			r.Verdict = VerdictInvestigate
		}
	}
	if r.BaselineSampler.DutyPercent.Available && r.BaselineSampler.DutyPercent.Value > r.Policy.MaximumSamplerDutyPercent {
		r.Findings = append(r.Findings, Finding{Code: "BASELINE_SAMPLER_DUTY_HIGH", Detail: fmt.Sprintf("baseline sampler duty %.2f%% exceeds %.2f%%", r.BaselineSampler.DutyPercent.Value, r.Policy.MaximumSamplerDutyPercent)})
		if r.Verdict != VerdictInvalid {
			r.Verdict = VerdictInvestigate
		}
	}
	if r.CommandSampler.DutyPercent.Available && r.CommandSampler.DutyPercent.Value > r.Policy.MaximumSamplerDutyPercent {
		r.Findings = append(r.Findings, Finding{Code: "COMMAND_SAMPLER_DUTY_HIGH", Detail: fmt.Sprintf("command sampler duty %.2f%% exceeds %.2f%%", r.CommandSampler.DutyPercent.Value, r.Policy.MaximumSamplerDutyPercent)})
		if r.Verdict != VerdictInvalid {
			r.Verdict = VerdictInvestigate
		}
	}
	if r.BaselineSampler.CoverageLimited || r.CommandSampler.CoverageLimited {
		r.Findings = append(r.Findings, Finding{Code: "SAMPLER_COVERAGE_LIMITED", Detail: "required census cadence exceeded the declared maximum; churn coverage is insufficient"})
		r.Verdict = VerdictInvalid
	}
	if r.BaselineHost.CPUPercent.Available && r.BaselineHost.CPUPercent.Value > r.Policy.MaximumNonSUTCPUPercent {
		r.Findings = append(r.Findings, Finding{Code: "BASELINE_HOST_CPU_HIGH", Detail: fmt.Sprintf("quiet baseline host CPU %.2f%% exceeds %.2f%%", r.BaselineHost.CPUPercent.Value, r.Policy.MaximumNonSUTCPUPercent)})
		if r.Verdict != VerdictInvalid {
			r.Verdict = VerdictInvestigate
		}
	}
	if r.Attribution.NonSUTCPUPercentOfHost.Available && r.Attribution.NonSUTCPUPercentOfHost.Value > r.Policy.MaximumNonSUTCPUPercent {
		r.Findings = append(r.Findings, Finding{Code: "NON_SUT_CPU_HIGH", Detail: fmt.Sprintf("non-SUT CPU %.2f%% exceeds %.2f%%", r.Attribution.NonSUTCPUPercentOfHost.Value, r.Policy.MaximumNonSUTCPUPercent)})
		if r.Verdict != VerdictInvalid {
			r.Verdict = VerdictInvestigate
		}
	}
	if r.CommandExitCode != 0 {
		r.Findings = append(r.Findings, Finding{Code: "COMMAND_FAILED", Detail: fmt.Sprintf("child command exited %d", r.CommandExitCode)})
		r.Verdict = VerdictInvalid
	}
	if r.TimedOut {
		r.Findings = append(r.Findings, Finding{Code: "COMMAND_TIMEOUT", Detail: "child command exceeded the configured timeout"})
		r.Verdict = VerdictInvalid
	}
	r.applyCgroupPolicy()
}

func available(v float64, unit, source string) Metric {
	return Metric{Available: true, Value: v, Unit: unit, Source: source}
}
func unavailable(unit, reason string) Metric { return Metric{Unit: unit, Reason: reason} }
func scrubImage(s string) string {
	s = filepath.Base(strings.ReplaceAll(s, "\\", "/"))
	if s == "." || s == "/" || s == "" {
		return "unknown"
	}
	return s
}

func (r *Report) canonicalDigest() string {
	saved := r.Digest
	r.Digest = ""
	b, err := json.Marshal(r)
	r.Digest = saved
	if err != nil {
		return ""
	}
	s := sha256.Sum256(b)
	return "sha256:" + hex.EncodeToString(s[:])
}
func (r *Report) Seal() { r.Digest = r.canonicalDigest() }

func Decode(rd io.Reader) (Report, error) {
	var r Report
	if rd == nil {
		return r, errors.New("systembaseline: decode: nil reader; provide a report reader")
	}
	raw, err := io.ReadAll(io.LimitReader(rd, maxEncodedReportBytes+1))
	if err != nil {
		return r, fmt.Errorf("systembaseline: decode: %w; retry with readable report input", err)
	}
	if len(raw) > maxEncodedReportBytes {
		return r, fmt.Errorf("systembaseline: decode: input exceeds %d bytes; reduce the report to the bounded schema", maxEncodedReportBytes)
	}
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&r); err != nil {
		return r, fmt.Errorf("systembaseline: decode: %w; provide one complete JSON report matching the schema", err)
	}
	var extra any
	if err := dec.Decode(&extra); err != io.EOF {
		if err == nil {
			return r, errors.New("systembaseline: multiple JSON values; provide exactly one JSON report")
		}
		return r, fmt.Errorf("systembaseline: trailing data: %w; remove bytes after the single JSON report", err)
	}
	return r, nil
}

func (r Report) policyVerdict() string {
	p := normalizePolicy(r.Policy)
	if r.Baseline.Samples < p.MinimumBaselineSamples || r.Baseline.DurationNS <= 0 || r.Baseline.IntervalNS <= 0 || r.Window.Samples < p.MinimumSamples || r.Window.DurationNS <= 0 || r.Window.IntervalNS <= 0 || r.Coverage.SUTRootPID <= 0 || r.CommandExitCode != 0 || r.TimedOut {
		return VerdictInvalid
	}
	if p.RequireHostCPU && (!r.BaselineHost.CPUPercent.Available || !r.Host.CPUPercent.Available) {
		return VerdictInvestigate
	}
	if p.RequireSUTCPU && (!r.Attribution.SUTCPUPercentOfHost.Available || !r.Attribution.NonSUTCPUPercentOfHost.Available) {
		return VerdictInvestigate
	}
	if p.RequireProcessMemory && (!r.Attribution.SUTRSSBytes.Available || !r.Attribution.NonSUTRSSBytes.Available) {
		return VerdictInvestigate
	}
	if r.BaselineSampler.CoverageLimited || r.CommandSampler.CoverageLimited {
		return VerdictInvalid
	}
	if !r.BaselineSampler.DutyPercent.Available || !r.CommandSampler.DutyPercent.Available {
		return VerdictInvestigate
	}
	if r.BaselineSampler.DutyPercent.Value > p.MaximumSamplerDutyPercent || r.CommandSampler.DutyPercent.Value > p.MaximumSamplerDutyPercent {
		return VerdictInvestigate
	}
	if r.BaselineHost.CPUPercent.Available && r.BaselineHost.CPUPercent.Value > p.MaximumNonSUTCPUPercent {
		return VerdictInvestigate
	}
	if r.Attribution.NonSUTCPUPercentOfHost.Available && r.Attribution.NonSUTCPUPercentOfHost.Value > p.MaximumNonSUTCPUPercent {
		return VerdictInvestigate
	}
	if r.CgroupV2 != nil {
		if r.CgroupV2.Cleanup.Attempted && (!r.CgroupV2.Cleanup.Empty || !r.CgroupV2.Cleanup.Removed) {
			return VerdictInvestigate
		}
		for _, axis := range []PressureAxis{r.CgroupV2.Pressure.CPU, r.CgroupV2.Pressure.Memory, r.CgroupV2.Pressure.IO} {
			for _, metric := range []Metric{axis.Some, axis.Full} {
				if pct, ok := pressureStallPercent(metric, r.Window.DurationNS); ok && pct > p.MaximumPSIStallPercent {
					return VerdictInvestigate
				}
			}
		}
	}
	return VerdictClean
}

func (r Report) Validate() error {
	if r.Schema != Schema {
		return fmt.Errorf("systembaseline: schema must be %q; set schema to the supported value", Schema)
	}
	if r.Verdict != VerdictClean && r.Verdict != VerdictInvestigate && r.Verdict != VerdictInvalid {
		return errors.New("systembaseline: invalid verdict; set verdict to clean, investigate, or invalid")
	}
	if r.Digest == "" || r.canonicalDigest() != r.Digest {
		return errors.New("systembaseline: canonical digest mismatch; call Seal after the final report edit")
	}
	if r.Baseline.Samples < 0 || r.Baseline.DurationNS < 0 || r.Baseline.IntervalNS < 0 || r.Window.Samples < 0 || r.Window.DurationNS < 0 || r.Window.IntervalNS < 0 || r.Coverage.ProcessSnapshots < 0 || r.Coverage.ProcessesObserved < 0 || r.Coverage.ProcessReads < 0 || r.Coverage.ProcessUnreadable < 0 || r.Coverage.HostCPUSamples < 0 || r.Coverage.HostMemorySamples < 0 || r.BaselineSampler.CountedSamples < 0 || r.BaselineSampler.WallNS < 0 || r.CommandSampler.CountedSamples < 0 || r.CommandSampler.WallNS < 0 {
		return errors.New("systembaseline: negative window or coverage value; use non-negative census values")
	}
	if err := validateSamplerOverhead("baseline", r.BaselineSampler, r.Baseline); err != nil {
		return err
	}
	if err := validateSamplerOverhead("command", r.CommandSampler, r.Window); err != nil {
		return err
	}
	for _, metric := range []Metric{r.BaselineHost.CPUPercent, r.BaselineHost.MemoryTotalBytes, r.BaselineHost.MemoryAvailableBytes, r.BaselineSampler.DutyPercent, r.Host.CPUPercent, r.Host.MemoryTotalBytes, r.Host.MemoryAvailableBytes, r.Host.ProcessCount, r.CommandSampler.DutyPercent, r.Attribution.SUTCPUPercentOfHost, r.Attribution.NonSUTCPUPercentOfHost, r.Attribution.SUTRSSBytes, r.Attribution.NonSUTRSSBytes} {
		if metric.Available && (math.IsNaN(metric.Value) || math.IsInf(metric.Value, 0) || metric.Value < 0 || metric.Unit == "") {
			return errors.New("systembaseline: available metric has invalid value or unit; provide a finite non-negative value and unit or mark it unavailable")
		}
	}
	if r.CgroupV2 != nil {
		if err := r.CgroupV2.validate(); err != nil {
			return err
		}
		if r.Coverage.DescendantAttribution == "exact_cgroup_v2" && r.CgroupV2.State != CgroupStateMeasured {
			return errors.New("systembaseline: exact_cgroup_v2 coverage requires measured cgroup evidence")
		}
	}
	if want := r.policyVerdict(); r.Verdict != want {
		return fmt.Errorf("systembaseline: verdict %q contradicts policy evidence; want %q; rebuild the report or use the policy-derived verdict", r.Verdict, want)
	}
	if len(r.TopNonSUT) > 0 && !r.Policy.IncludeTopConsumers {
		return errors.New("systembaseline: top consumers present without opt-in policy; enable include_top_consumers or remove the list")
	}
	if len(r.TopNonSUT) > 5 {
		return errors.New("systembaseline: top consumers exceed bounded disclosure; trim the list to five entries")
	}
	for _, c := range r.TopNonSUT {
		if c.PID <= 0 || c.Image != scrubImage(c.Image) || strings.ContainsAny(c.Image, "/\\") {
			return errors.New("systembaseline: non-SUT consumer identity is not scrubbed; retain only a basename without path separators")
		}
		if math.IsNaN(c.CPUSeconds) || math.IsInf(c.CPUSeconds, 0) || c.CPUSeconds < 0 {
			return errors.New("systembaseline: non-SUT consumer CPU is invalid; provide a finite non-negative CPU duration")
		}
	}
	return nil
}

func validateSamplerOverhead(name string, overhead SamplerOverhead, window Window) error {
	if window.Samples < 2 || window.DurationNS <= 0 {
		return nil
	}
	if overhead.CountedSamples != window.Samples-1 {
		return fmt.Errorf("systembaseline: %s sampler coverage is inconsistent with its window; set counted samples to window samples minus one", name)
	}
	if !overhead.DutyPercent.Available {
		if overhead.WallNS != 0 || overhead.DutyPercent.Reason == "" {
			return fmt.Errorf("systembaseline: %s sampler unavailable state is inconsistent; set wall time to zero and provide an unavailable reason", name)
		}
		return nil
	}
	want := float64(overhead.WallNS) / float64(window.DurationNS) * 100
	if math.Abs(overhead.DutyPercent.Value-want) > 1e-9 {
		return fmt.Errorf("systembaseline: %s sampler duty does not match census wall time; recompute duty from sampler wall time and window duration", name)
	}
	return nil
}
