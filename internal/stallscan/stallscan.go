// Package stallscan classifies whole-machine "stall" fingerprints from cheap,
// point-in-time system counters — the pure, OS-independent heart of `fak stallscan`.
//
// WHY THIS EXISTS. On a busy multi-session box (many Claude sessions + fak
// workers + per-session MCP servers) the machine can "lock up" for a beat while
// every usage meter reads low: CPU%, RSS, and disk queue all look fine. The
// real contention is inside the kernel's page-fault and scheduler paths — a
// storm of SOFT faults (demand-zero + transition, served from RAM, not disk),
// context switches, and syscalls driven by process/thread CHURN. No "usage"
// dashboard shows that, so a reaper that gates on CPU/RAM never fires.
//
// This package takes a Sample of the signals that DO reveal it and returns a
// Verdict: a level (calm/elevated/stall) plus the dominant cause, decided by
// fixed, documented thresholds. It reads nothing and spawns nothing — the
// caller hands it numbers already gathered — so it is fully testable and cannot
// itself add to the churn it measures.
//
// The classification is deliberately conservative and explainable: a stall is
// only declared when a churn signal crosses a threshold that, in live capture
// on the reference box, coincided with a desktop freeze. See stallscan_test.go
// for the fixtures pinned to those captures.
package stallscan

import (
	"fmt"
	"sort"
)

// Sample is one point-in-time reading of the churn-revealing signals. All rates
// are per-second; counts are absolute. Zero for an unavailable field is treated
// as "not observed" and never trips a threshold on its own.
type Sample struct {
	// Fault subsystem. TotalFaultsPerSec is dominated by soft faults on a
	// healthy box; the split is what matters. HardFaultsPerSec (page reads that
	// actually hit disk) staying tiny while TotalFaults is huge is the signature
	// of allocate/parse/free churn, NOT disk thrash.
	TotalFaultsPerSec      float64 `json:"total_faults_per_sec"`
	HardFaultsPerSec       float64 `json:"hard_faults_per_sec"`
	DemandZeroFaultsPerSec float64 `json:"demand_zero_faults_per_sec"`
	TransitionFaultsPerSec float64 `json:"transition_faults_per_sec"`

	// Scheduler / syscall pressure.
	ContextSwitchesPerSec float64 `json:"context_switches_per_sec"`
	SystemCallsPerSec     float64 `json:"system_calls_per_sec"`

	// Process/thread census + churn since the previous sample.
	ProcessCount int     `json:"process_count"`
	ThreadCount  int     `json:"thread_count"`
	ProcessDelta int     `json:"process_delta"`  // net change in process count vs previous sample
	SpawnBurst   int     `json:"spawn_burst"`    // processes that appeared since previous sample (gross), if known
	AvailableMB  int     `json:"available_mb"`   // free RAM; rules OUT memory exhaustion as the cause
	DiskQueueLen float64 `json:"disk_queue_len"` // current physical-disk queue; rules OUT disk saturation

	// Handle census. A per-process handle count that climbs unbounded is a
	// classic Windows leak (Russinovich's rule of thumb: >10k handles/proc is a
	// likely leak). On this box the terminal emulator has been the top holder,
	// climbing across days — a slow-burn precursor to pool exhaustion that no
	// fault/scheduler counter reveals. Zero = not observed (never trips a rule).
	SystemHandleTotal int           `json:"system_handle_total,omitempty"` // sum of all processes' open handles
	TopHandles        []ProcHandles `json:"top_handles,omitempty"`         // top handle holders at sample time

	// Thread census. A single process whose thread count climbs into the many
	// hundreds/thousands is a thread leak — the "terminal thread lag" failure
	// mode: a terminal emulator accreting a thread per PTY/render across a
	// spawn-heavy fleet until its own UI/scheduling lags (observed at 1500+
	// threads on the reference box). Distinct from handles; a fault or handle
	// axis misses it entirely. Zero = not observed.
	TopThreads []ProcThreads `json:"top_threads,omitempty"` // top thread holders at sample time

	// Top offenders by I/O operations/sec at this instant (already sorted or not).
	TopIO []ProcIO `json:"top_io,omitempty"`
}

// ProcIO is one process's I/O-operation rate at sample time.
type ProcIO struct {
	PID  int     `json:"pid"`
	Name string  `json:"name"`
	Ops  float64 `json:"ops_per_sec"`
}

// ProcHandles is one process's open-handle count at sample time.
type ProcHandles struct {
	PID     int    `json:"pid"`
	Name    string `json:"name"`
	Handles int    `json:"handles"`
}

// ProcThreads is one process's thread count at sample time.
type ProcThreads struct {
	PID     int    `json:"pid"`
	Name    string `json:"name"`
	Threads int    `json:"threads"`
}

// Level is the coarse severity band.
type Level string

const (
	LevelCalm     Level = "calm"
	LevelElevated Level = "elevated"
	LevelStall    Level = "stall"
)

// Cause names the dominant mechanism behind an elevated/stall verdict.
type Cause string

const (
	CauseNone        Cause = "none"
	CauseSoftFault   Cause = "soft_fault_churn" // demand-zero/transition storm at low hard-fault, RAM free
	CauseSpawnStorm  Cause = "spawn_storm"      // process-creation burst driving ctx-switch/syscall spike
	CauseSchedThrash Cause = "scheduler_thrash" // context-switch/syscall storm without a clear spawn burst
	CauseDiskIO      Cause = "disk_io"          // genuine disk-backed pressure (hard faults / disk queue)
	CauseMemPressure Cause = "memory_pressure"  // low available RAM (the classic case, here to distinguish it)
	CauseHandleLeak  Cause = "handle_leak"      // a process's open-handle count climbing unbounded (slow-burn)
	CauseThreadLeak  Cause = "thread_leak"      // a process's thread count climbing into the hundreds/thousands (terminal thread lag)
)

// Thresholds are the decision boundaries. They are exported and defaulted so a
// caller (or a test) can tune them, and so the live self-monitor can be made
// more/less sensitive without code change. Defaults are calibrated to the
// reference box where a desktop freeze coincided with these crossings.
type Thresholds struct {
	// Soft-fault churn: total faults/sec high while hard faults stay low.
	SoftFaultStall    float64 // total faults/sec that alone signals a stall (default 400k)
	SoftFaultElevated float64 // elevated band (default 150k)
	HardFaultDiskFrac float64 // if hard/total exceeds this, it's disk not soft churn (default 0.15)

	// Scheduler/syscall storm.
	ContextSwitchStall float64 // ctx switches/sec (default 100k)
	SysCallStall       float64 // syscalls/sec (default 500k)

	// Spawn burst: net or gross new processes within one sample interval.
	SpawnBurstStall int // default 8

	// Genuine-resource guards (to correctly ATTRIBUTE, not to gate stalls).
	MemLowMB      int     // available RAM below this = memory pressure (default 2048)
	DiskQueueBusy float64 // disk queue at/above this = real disk pressure (default 4)

	// Handle-leak detection. A single process at/above HandleLeakProc handles is
	// a leak suspect (default 10000, Russinovich's line). This is a WARNING axis,
	// not a freeze axis: a leak suspect raises a calm box to *elevated* and names
	// the culprit, but never fabricates a stall — the box runs for days with a
	// leaking terminal before any pool exhausts. SystemHandleHigh is a coarse
	// system-wide informational ceiling (default 1_000_000). Zero disables either.
	HandleLeakProc   int // per-process handle count that flags a leak suspect (default 10000)
	SystemHandleHigh int // system-wide handle total worth flagging as elevated (default 1_000_000)

	// Thread-leak detection. A single process at/above ThreadLeakProc threads is a
	// leak suspect (default 500 — well above a normal process's dozens, below the
	// 1500+ seen on a leaking terminal emulator). Same WARNING semantics as the
	// handle axis: raises a calm box to elevated and names the culprit, but never
	// fabricates a stall. Zero disables it.
	ThreadLeakProc int // per-process thread count that flags a thread-leak suspect (default 500)

	// Growth (trajectory) detection — the axis the absolute HandleLeakProc/
	// ThreadLeakProc lines miss. Absolute thresholds fire only once a process is
	// already huge (>=10k handles / >=500 threads) and cannot tell a process that
	// is high-and-STABLE from one that is CLIMBING. A leak is fundamentally a
	// trajectory: the signal worth alerting on is a process whose count is rising
	// sample-over-sample, even while still under the absolute line. ClassifyWithBaseline
	// compares a baseline sample against the current one and flags a process (matched
	// by PID+name) whose count climbed by at/above *GrowthDelta while its current
	// count is at/above *GrowthFloor. The floor keeps normal small-process churn
	// out; it sits below the absolute line so growth is an EARLIER warning. Same
	// WARNING semantics as the absolute axes: raises calm -> elevated, never a stall.
	// Zero for a Delta disables that growth axis.
	HandleGrowthDelta int // per-process handle CLIMB (cur-baseline) that flags a growth suspect (default 1500)
	HandleGrowthFloor int // only consider processes whose current handle count is at/above this (default 3000)
	ThreadGrowthDelta int // per-process thread CLIMB that flags a growth suspect (default 100)
	ThreadGrowthFloor int // only consider processes whose current thread count is at/above this (default 200)
}

// DefaultThresholds returns the calibrated defaults.
func DefaultThresholds() Thresholds {
	return Thresholds{
		SoftFaultStall:     400000,
		SoftFaultElevated:  150000,
		HardFaultDiskFrac:  0.15,
		ContextSwitchStall: 100000,
		SysCallStall:       500000,
		SpawnBurstStall:    8,
		MemLowMB:           2048,
		DiskQueueBusy:      4,
		HandleLeakProc:     10000,
		SystemHandleHigh:   1000000,
		ThreadLeakProc:     500,
		HandleGrowthDelta:  1500,
		HandleGrowthFloor:  3000,
		ThreadGrowthDelta:  100,
		ThreadGrowthFloor:  200,
	}
}

// Verdict is the classification result.
type Verdict struct {
	Level   Level    `json:"level"`
	Cause   Cause    `json:"cause"`
	Reasons []string `json:"reasons"` // human-readable, each a crossed threshold with its number
	// Attribution: the single process most implicated, if TopIO was provided.
	TopProcess string  `json:"top_process,omitempty"`
	TopPID     int     `json:"top_pid,omitempty"`
	TopOps     float64 `json:"top_ops_per_sec,omitempty"`

	// Handle-leak attribution: the worst per-process handle hog, if any crossed
	// HandleLeakProc. Populated regardless of Level (so an operator sees the
	// leaking process even when a churn stall dominates the Cause).
	HandleLeakProcess string `json:"handle_leak_process,omitempty"`
	HandleLeakPID     int    `json:"handle_leak_pid,omitempty"`
	HandleLeakCount   int    `json:"handle_leak_count,omitempty"`

	// Thread-leak attribution: the worst per-process thread hog, if any crossed
	// ThreadLeakProc. Populated regardless of Level (same rationale as handles).
	ThreadLeakProcess string `json:"thread_leak_process,omitempty"`
	ThreadLeakPID     int    `json:"thread_leak_pid,omitempty"`
	ThreadLeakCount   int    `json:"thread_leak_count,omitempty"`

	// Growth (trajectory) attribution — populated only by ClassifyWithBaseline,
	// which has a baseline to diff against. The process whose handle/thread count
	// CLIMBED the most since the baseline (and crossed the growth delta+floor).
	// Count is the current value; Delta is the climb. Populated regardless of Level.
	HandleGrowthProcess string `json:"handle_growth_process,omitempty"`
	HandleGrowthPID     int    `json:"handle_growth_pid,omitempty"`
	HandleGrowthCount   int    `json:"handle_growth_count,omitempty"` // current handle count
	HandleGrowthDelta   int    `json:"handle_growth_delta,omitempty"` // climb since baseline
	ThreadGrowthProcess string `json:"thread_growth_process,omitempty"`
	ThreadGrowthPID     int    `json:"thread_growth_pid,omitempty"`
	ThreadGrowthCount   int    `json:"thread_growth_count,omitempty"` // current thread count
	ThreadGrowthDelta   int    `json:"thread_growth_delta,omitempty"` // climb since baseline
}

// hardFraction is the share of total faults that actually hit disk. A tiny
// fraction at a huge total is the soft-churn tell.
func (s Sample) hardFraction() float64 {
	if s.TotalFaultsPerSec <= 0 {
		return 0
	}
	return s.HardFaultsPerSec / s.TotalFaultsPerSec
}

// Classify decides the verdict from a sample and thresholds. It is pure: same
// input, same output, no I/O.
//
// Decision order matters. We first rule OUT the genuine-resource explanations
// (memory pressure, disk saturation) so we do not mislabel a real disk stall as
// soft churn. Only then do we test the churn signals. This mirrors the live
// investigation: the whole point is that the machine looks low-usage, so a
// churn verdict must be able to say "and it was NOT disk or RAM."
func Classify(s Sample, t Thresholds) Verdict {
	v := Verdict{Level: LevelCalm, Cause: CauseNone}

	// Attribute the top I/O process regardless of level — useful even when calm.
	if len(s.TopIO) > 0 {
		top := topByOps(s.TopIO)
		v.TopProcess, v.TopPID, v.TopOps = top.Name, top.PID, top.Ops
	}

	// Attribute the worst per-process handle hog regardless of level, so the
	// operator always sees a leaking process — even when a churn stall owns the
	// Cause. A leak is a slow-burn WARNING that rides alongside whatever else is
	// happening; it only raises level below (calm -> elevated), never a stall.
	leak, hasLeak := worstHandleHog(s.TopHandles, t.HandleLeakProc)
	if hasLeak {
		v.HandleLeakProcess, v.HandleLeakPID, v.HandleLeakCount = leak.Name, leak.PID, leak.Handles
	}

	// Same for the worst thread hog — the "terminal thread lag" axis. Also a
	// slow-burn WARNING that rides alongside the Cause and only raises calm ->
	// elevated below, never fabricates a stall.
	threadLeak, hasThreadLeak := worstThreadHog(s.TopThreads, t.ThreadLeakProc)
	if hasThreadLeak {
		v.ThreadLeakProcess, v.ThreadLeakPID, v.ThreadLeakCount = threadLeak.Name, threadLeak.PID, threadLeak.Threads
	}

	// --- Genuine-resource explanations first (highest priority) ---
	if t.MemLowMB > 0 && s.AvailableMB > 0 && s.AvailableMB < t.MemLowMB {
		v.Level = LevelStall
		v.Cause = CauseMemPressure
		v.Reasons = append(v.Reasons, fmt.Sprintf("available RAM %d MB below %d MB", s.AvailableMB, t.MemLowMB))
		return v
	}
	if t.DiskQueueBusy > 0 && s.DiskQueueLen >= t.DiskQueueBusy {
		v.Level = LevelStall
		v.Cause = CauseDiskIO
		v.Reasons = append(v.Reasons, fmt.Sprintf("disk queue %.1f at/above %.1f", s.DiskQueueLen, t.DiskQueueBusy))
		return v
	}
	// Hard-fault-dominated fault storm => disk, not soft churn.
	if s.TotalFaultsPerSec >= t.SoftFaultElevated && s.hardFraction() >= t.HardFaultDiskFrac {
		v.Level = LevelStall
		v.Cause = CauseDiskIO
		v.Reasons = append(v.Reasons, fmt.Sprintf("hard-fault fraction %.2f at %0.f total faults/sec", s.hardFraction(), s.TotalFaultsPerSec))
		return v
	}

	// --- Churn signals (the low-usage stall class) ---
	stall := false

	// Spawn storm: a burst of new processes in one interval. This is the most
	// specific and actionable cause, so it wins attribution when present.
	spawn := s.SpawnBurst
	if spawn == 0 && s.ProcessDelta > 0 {
		spawn = s.ProcessDelta
	}
	if t.SpawnBurstStall > 0 && spawn >= t.SpawnBurstStall {
		stall = true
		v.Cause = CauseSpawnStorm
		v.Reasons = append(v.Reasons, fmt.Sprintf("spawn burst %d processes in one interval (>= %d)", spawn, t.SpawnBurstStall))
	}

	// Soft-fault churn: huge total faults, small hard fraction, RAM free.
	if s.TotalFaultsPerSec >= t.SoftFaultStall && s.hardFraction() < t.HardFaultDiskFrac {
		stall = true
		if v.Cause == CauseNone || v.Cause == CauseSchedThrash {
			v.Cause = CauseSoftFault
		}
		v.Reasons = append(v.Reasons, fmt.Sprintf("%0.f soft faults/sec (hard frac %.2f)", s.TotalFaultsPerSec, s.hardFraction()))
	}

	// Scheduler/syscall thrash.
	if t.ContextSwitchStall > 0 && s.ContextSwitchesPerSec >= t.ContextSwitchStall {
		stall = true
		if v.Cause == CauseNone {
			v.Cause = CauseSchedThrash
		}
		v.Reasons = append(v.Reasons, fmt.Sprintf("%0.f context switches/sec (>= %0.f)", s.ContextSwitchesPerSec, t.ContextSwitchStall))
	}
	if t.SysCallStall > 0 && s.SystemCallsPerSec >= t.SysCallStall {
		stall = true
		if v.Cause == CauseNone {
			v.Cause = CauseSchedThrash
		}
		v.Reasons = append(v.Reasons, fmt.Sprintf("%0.f syscalls/sec (>= %0.f)", s.SystemCallsPerSec, t.SysCallStall))
	}

	// Handle signals are WARNINGS, not freezes: append their reasons now so they
	// surface even under a churn stall, but do NOT set `stall` — a leaking
	// terminal runs for days before any pool exhausts.
	if hasLeak {
		v.Reasons = append(v.Reasons, fmt.Sprintf("%s (pid %d) holds %d handles (>= %d — leak suspect)", leak.Name, leak.PID, leak.Handles, t.HandleLeakProc))
	}
	if hasThreadLeak {
		v.Reasons = append(v.Reasons, fmt.Sprintf("%s (pid %d) holds %d threads (>= %d — thread-leak suspect / terminal lag)", threadLeak.Name, threadLeak.PID, threadLeak.Threads, t.ThreadLeakProc))
	}
	systemHandleHigh := t.SystemHandleHigh > 0 && s.SystemHandleTotal >= t.SystemHandleHigh
	if systemHandleHigh {
		v.Reasons = append(v.Reasons, fmt.Sprintf("%d handles system-wide (>= %d)", s.SystemHandleTotal, t.SystemHandleHigh))
	}

	if stall {
		v.Level = LevelStall
		return v
	}

	// --- Elevated (not yet a stall, but worth logging) ---
	if s.TotalFaultsPerSec >= t.SoftFaultElevated {
		v.Level = LevelElevated
		if v.Cause == CauseNone {
			v.Cause = CauseSoftFault
		}
		v.Reasons = append(v.Reasons, fmt.Sprintf("%0.f faults/sec (elevated >= %0.f)", s.TotalFaultsPerSec, t.SoftFaultElevated))
	}
	// A handle-leak or thread-leak suspect (or a system-wide handle ceiling)
	// raises a calm box to elevated and owns the cause if nothing else claimed it.
	// Handle pressure wins the cause label when both a handle and a thread leak
	// are present (it is the closer-to-exhaustion resource on the reference box).
	if v.Level == LevelCalm && (hasLeak || systemHandleHigh || hasThreadLeak) {
		v.Level = LevelElevated
		if v.Cause == CauseNone {
			if hasLeak || systemHandleHigh {
				v.Cause = CauseHandleLeak
			} else {
				v.Cause = CauseThreadLeak
			}
		}
	}
	return v
}

// worstHandleHog returns the single process with the most open handles among
// those at/above threshold, and ok=false if none qualify (or threshold<=0). Pure
// and non-mutating, like topByOps.
func worstHandleHog(in []ProcHandles, threshold int) (ProcHandles, bool) {
	if threshold <= 0 {
		return ProcHandles{}, false
	}
	worst := ProcHandles{}
	found := false
	for _, p := range in {
		if p.Handles < threshold {
			continue
		}
		if !found || p.Handles > worst.Handles || (p.Handles == worst.Handles && p.PID < worst.PID) {
			worst, found = p, true
		}
	}
	return worst, found
}

// worstThreadHog returns the single process with the most threads among those
// at/above threshold, and ok=false if none qualify (or threshold<=0). Pure and
// non-mutating, mirroring worstHandleHog.
func worstThreadHog(in []ProcThreads, threshold int) (ProcThreads, bool) {
	if threshold <= 0 {
		return ProcThreads{}, false
	}
	worst := ProcThreads{}
	found := false
	for _, p := range in {
		if p.Threads < threshold {
			continue
		}
		if !found || p.Threads > worst.Threads || (p.Threads == worst.Threads && p.PID < worst.PID) {
			worst, found = p, true
		}
	}
	return worst, found
}

// topByOps returns the highest-ops entry (stable on ties by PID for determinism).
func topByOps(in []ProcIO) ProcIO {
	cp := make([]ProcIO, len(in))
	copy(cp, in)
	sort.SliceStable(cp, func(i, j int) bool {
		if cp[i].Ops != cp[j].Ops {
			return cp[i].Ops > cp[j].Ops
		}
		return cp[i].PID < cp[j].PID
	})
	return cp[0]
}

// SortTopIO returns the entries sorted by ops descending, capped at n (n<=0 = all).
// Helper for renderers so the CLI doesn't re-implement the sort.
func SortTopIO(in []ProcIO, n int) []ProcIO {
	cp := make([]ProcIO, len(in))
	copy(cp, in)
	sort.SliceStable(cp, func(i, j int) bool {
		if cp[i].Ops != cp[j].Ops {
			return cp[i].Ops > cp[j].Ops
		}
		return cp[i].PID < cp[j].PID
	})
	if n > 0 && len(cp) > n {
		cp = cp[:n]
	}
	return cp
}
