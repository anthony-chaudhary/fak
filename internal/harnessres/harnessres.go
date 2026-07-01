// Package harnessres samples the hardware-resource use of the fak guard HARNESS
// itself — the guard process (which hosts the in-process gateway on the same PID)
// and the wrapped agent child — so a guarded session can report the CPU, memory, and
// I/O it burned, the same way it already reports its cache/token economy.
//
// It is stdlib-only (per AGENTS.md: no gopsutil). The per-platform readers use
// syscall.Getrusage + /proc on unix and the kernel32/psapi LazyDLL idiom on Windows
// (mirroring cmd/modelbench/rss_*.go); the pure fold + renderers in this file carry
// the value and the tests. Every axis a platform cannot read stays absent behind a
// presence bit, so a missing number renders "n/a" — never a fabricated 0.
//
// Foundation leaf for epic #2044 / #2045: imports nothing internal, off the hot path.
package harnessres

import (
	"encoding/json"
	"fmt"
	"os"
	"runtime"
	"strings"
	"sync"
	"time"
)

// LedgerSchema tags each durable JSONL row (docs/nightrun/harness-resources.jsonl).
const LedgerSchema = "fak-harness-resources/1"

// DefaultLedgerRel is the durable ledger path (sibling of cache-savings.jsonl).
const DefaultLedgerRel = "docs/nightrun/harness-resources.jsonl"

// Half is one side of the harness: the kernel (guard + in-process gateway, one PID,
// sampled continuously) or the agent (the wrapped child, folded from its exit state).
// The Have* bits distinguish an observed zero from an axis the platform cannot read.
type Half struct {
	CPUUser      time.Duration
	CPUSys       time.Duration
	HaveCPU      bool
	RSSBytes     uint64
	HaveRSS      bool
	PeakRSSBytes uint64
	HavePeakRSS  bool
	IOReadBytes  uint64
	IOWriteBytes uint64
	HaveIO       bool
	NetRxBytes   uint64
	NetTxBytes   uint64
	HaveNet      bool
}

// CPUSeconds is the combined user+system CPU time in seconds.
func (h Half) CPUSeconds() float64 { return h.CPUUser.Seconds() + h.CPUSys.Seconds() }

// CPUPercentAvg is the average CPU utilization over elapsed wall time (100% == one
// fully-busy core). ok is false when CPU was not observed or elapsed is non-positive.
func (h Half) CPUPercentAvg(elapsed time.Duration) (pct float64, ok bool) {
	if !h.HaveCPU || elapsed <= 0 {
		return 0, false
	}
	return h.CPUSeconds() / elapsed.Seconds() * 100, true
}

// Snapshot is one folded reading of the whole harness's resource use.
type Snapshot struct {
	Elapsed              time.Duration
	Samples              int
	Kernel               Half // guard + in-process gateway (self, sampled continuously)
	Agent                Half // wrapped child (folded from its exit state; live tree is #2048)
	KernelCPUPercentPeak float64
	HaveKernelCPUPeak    bool
	GoroutinesPeak       int
	GoHeapSysBytes       uint64
	NumCPU               int
	GOMAXPROCS           int
	// GPU/accelerator (present only when the harness runs a model in-kernel via
	// --gguf/--backend; the default proxy path uses no local GPU, so HaveGPU is false).
	GPUVRAMUsedBytes  uint64
	GPUVRAMTotalBytes uint64
	HaveGPU           bool
}

// procSample is a single raw reading of ONE process's OS-level resource use, as
// returned by the per-platform readProcSelf(). Absent axes stay at zero, flagged
// by the have* bits.
type procSample struct {
	cpuUser     time.Duration
	cpuSys      time.Duration
	haveCPU     bool
	rss         uint64
	haveRSS     bool
	peakRSS     uint64
	havePeakRSS bool
	ioRead      uint64
	ioWrite     uint64
	haveIO      bool
}

// Sampler periodically reads the guard process (self) and folds the wrapped child at
// exit into one Snapshot. The zero value is not usable; construct with New.
type Sampler struct {
	nowFn func() time.Time
	start time.Time

	mu       sync.Mutex
	samples  int
	kernel   Half
	goPeak   int
	heapPeak uint64

	// peak instantaneous kernel CPU% across sample ticks
	haveLast    bool
	lastWall    time.Time
	lastCPU     time.Duration
	cpuPeak     float64
	haveCPUPeak bool

	agent Half

	// gpu is the latest accelerator reading (from gpuProvider), snapshot-scoped.
	gpuUsed  uint64
	gpuTotal uint64
	haveGPU  bool

	// netProvider pulls the kernel half's cumulative (rx, tx) upstream network bytes
	// each sample; gpuProvider pulls (used, total) accelerator VRAM. Both nil by default
	// (the leaf reads no network/GPU itself — the host wires them). Set before Start.
	netProvider func() (rx, tx uint64, ok bool)
	gpuProvider func() (used, total uint64, ok bool)

	stopOnce sync.Once
	stop     chan struct{}
	done     chan struct{}
}

// SetNetworkProvider installs the pull source for the kernel half's network bytes
// (fak guard wires this to its upstream CountingRoundTripper). Call before Start.
func (s *Sampler) SetNetworkProvider(fn func() (rx, tx uint64, ok bool)) {
	if s == nil {
		return
	}
	s.mu.Lock()
	s.netProvider = fn
	s.mu.Unlock()
}

// SetGPUProvider installs the pull source for accelerator VRAM (used, total). fak guard
// wires this to compute.DeviceMemoryInfo when a model runs in-kernel. Call before Start.
func (s *Sampler) SetGPUProvider(fn func() (used, total uint64, ok bool)) {
	if s == nil {
		return
	}
	s.mu.Lock()
	s.gpuProvider = fn
	s.mu.Unlock()
}

// New returns a Sampler that reads the wall clock.
func New() *Sampler { return newSampler(time.Now) }

func newSampler(now func() time.Time) *Sampler {
	return &Sampler{nowFn: now, start: now()}
}

// Start begins periodic sampling on a background goroutine. An interval <= 0 falls
// back to 2s. It takes one immediate reading so even a sub-interval session records
// a sample. Calling Start more than once is a no-op after the first.
func (s *Sampler) Start(interval time.Duration) {
	if s == nil || s.stop != nil {
		return
	}
	if interval <= 0 {
		interval = 2 * time.Second
	}
	s.stop = make(chan struct{})
	s.done = make(chan struct{})
	s.sampleOnce()
	go func() {
		t := time.NewTicker(interval)
		defer t.Stop()
		defer close(s.done)
		for {
			select {
			case <-s.stop:
				s.sampleOnce()
				return
			case <-t.C:
				s.sampleOnce()
			}
		}
	}()
}

// Stop halts sampling (taking one final reading) and returns the folded Snapshot.
// Safe to call on a Sampler that was never Started, or more than once.
func (s *Sampler) Stop() Snapshot {
	if s == nil {
		return Snapshot{}
	}
	s.stopOnce.Do(func() {
		if s.stop != nil {
			close(s.stop)
			<-s.done
		}
	})
	return s.Snapshot()
}

func (s *Sampler) sampleOnce() {
	ps := readProcSelf()
	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)
	s.foldProc(ps, s.nowFn(), runtime.NumGoroutine(), ms.HeapSys)
}

// foldProc folds one raw reading into the accumulator. Split out from sampleOnce so
// the peak-tracking logic is unit-testable with synthetic samples + a fake clock.
func (s *Sampler) foldProc(ps procSample, now time.Time, goroutines int, heapSys uint64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.samples++
	if ps.haveCPU {
		total := ps.cpuUser + ps.cpuSys
		if s.haveLast {
			dw := now.Sub(s.lastWall).Seconds()
			dc := (total - s.lastCPU).Seconds()
			if dw > 0 && dc >= 0 {
				if pct := dc / dw * 100; !s.haveCPUPeak || pct > s.cpuPeak {
					s.cpuPeak, s.haveCPUPeak = pct, true
				}
			}
		}
		s.lastWall, s.lastCPU, s.haveLast = now, total, true
		s.kernel.CPUUser, s.kernel.CPUSys, s.kernel.HaveCPU = ps.cpuUser, ps.cpuSys, true
	}
	if ps.haveRSS {
		s.kernel.RSSBytes, s.kernel.HaveRSS = ps.rss, true
		if ps.rss > s.kernel.PeakRSSBytes {
			s.kernel.PeakRSSBytes, s.kernel.HavePeakRSS = ps.rss, true
		}
	}
	if ps.havePeakRSS {
		if ps.peakRSS > s.kernel.PeakRSSBytes {
			s.kernel.PeakRSSBytes = ps.peakRSS
		}
		s.kernel.HavePeakRSS = true
	}
	if ps.haveIO {
		s.kernel.IOReadBytes, s.kernel.IOWriteBytes, s.kernel.HaveIO = ps.ioRead, ps.ioWrite, true
	}
	if goroutines > s.goPeak {
		s.goPeak = goroutines
	}
	if heapSys > s.heapPeak {
		s.heapPeak = heapSys
	}
	if s.netProvider != nil {
		if rx, tx, ok := s.netProvider(); ok {
			s.kernel.NetRxBytes, s.kernel.NetTxBytes, s.kernel.HaveNet = rx, tx, true
		}
	}
	if s.gpuProvider != nil {
		if used, total, ok := s.gpuProvider(); ok {
			s.gpuUsed, s.gpuTotal, s.haveGPU = used, total, true
		}
	}
}

// FoldChildExit records the wrapped agent child's final resource use from its exit
// state. UserTime/SystemTime are cross-platform stdlib; unix additionally exposes the
// child's peak RSS via Rusage.Maxrss (foldChildRusage, per-platform). Nil state is a
// no-op (a child that never started).
func (s *Sampler) FoldChildExit(ps *os.ProcessState) {
	if s == nil || ps == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.agent.CPUUser, s.agent.CPUSys, s.agent.HaveCPU = ps.UserTime(), ps.SystemTime(), true
	foldChildRusage(&s.agent, ps)
}

// Snapshot returns the current folded reading without stopping the sampler.
func (s *Sampler) Snapshot() Snapshot {
	if s == nil {
		return Snapshot{}
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return Snapshot{
		Elapsed:              s.nowFn().Sub(s.start),
		Samples:              s.samples,
		Kernel:               s.kernel,
		Agent:                s.agent,
		KernelCPUPercentPeak: s.cpuPeak,
		HaveKernelCPUPeak:    s.haveCPUPeak,
		GoroutinesPeak:       s.goPeak,
		GoHeapSysBytes:       s.heapPeak,
		NumCPU:               runtime.NumCPU(),
		GOMAXPROCS:           runtime.GOMAXPROCS(0),
		GPUVRAMUsedBytes:     s.gpuUsed,
		GPUVRAMTotalBytes:    s.gpuTotal,
		HaveGPU:              s.haveGPU,
	}
}

// Report renders the one-line human summary for the guard exit summary. It leads with
// "harness resources — " so the caller only prepends its own "fak guard: " prefix.
func (s Snapshot) Report() string {
	var b strings.Builder
	b.WriteString("harness resources — kernel(guard+gateway) ")
	writeHalf(&b, s.Kernel, s.Elapsed, s.KernelCPUPercentPeak, s.HaveKernelCPUPeak)
	b.WriteString("; agent(child) ")
	writeHalf(&b, s.Agent, s.Elapsed, 0, false)
	fmt.Fprintf(&b, "; %d goroutines peak, Go heap %s; %d cores", s.GoroutinesPeak, humanBytes(s.GoHeapSysBytes), s.NumCPU)
	if s.GOMAXPROCS > 0 && s.GOMAXPROCS != s.NumCPU {
		fmt.Fprintf(&b, " (GOMAXPROCS %d)", s.GOMAXPROCS)
	}
	if s.HaveGPU {
		fmt.Fprintf(&b, "; gpu vram %s", humanBytes(s.GPUVRAMUsedBytes))
		if s.GPUVRAMTotalBytes > 0 {
			fmt.Fprintf(&b, "/%s", humanBytes(s.GPUVRAMTotalBytes))
		}
	}
	fmt.Fprintf(&b, "; sampled %dx over %s", s.Samples, humanDur(s.Elapsed))
	return b.String()
}

func writeHalf(b *strings.Builder, h Half, elapsed time.Duration, cpuPeak float64, haveCPUPeak bool) {
	if h.HaveCPU {
		fmt.Fprintf(b, "cpu %.1fs", h.CPUSeconds())
		if pct, ok := h.CPUPercentAvg(elapsed); ok {
			if haveCPUPeak {
				fmt.Fprintf(b, " (%.0f%% avg, %.0f%% peak)", pct, cpuPeak)
			} else {
				fmt.Fprintf(b, " (%.0f%% avg)", pct)
			}
		}
	} else {
		b.WriteString("cpu n/a")
	}
	switch {
	case h.HaveRSS && h.HavePeakRSS:
		fmt.Fprintf(b, ", rss %s (peak %s)", humanBytes(h.RSSBytes), humanBytes(h.PeakRSSBytes))
	case h.HavePeakRSS:
		fmt.Fprintf(b, ", peak rss %s", humanBytes(h.PeakRSSBytes))
	case h.HaveRSS:
		fmt.Fprintf(b, ", rss %s", humanBytes(h.RSSBytes))
	default:
		b.WriteString(", rss n/a")
	}
	if h.HaveIO {
		fmt.Fprintf(b, ", io r/w %s/%s", humanBytes(h.IOReadBytes), humanBytes(h.IOWriteBytes))
	}
	if h.HaveNet {
		fmt.Fprintf(b, ", net rx/tx %s/%s", humanBytes(h.NetRxBytes), humanBytes(h.NetTxBytes))
	}
}

// PrometheusText renders the fak_harness_* gauge family (consumed by the /metrics
// child #2047). Only present axes emit a sample line, so a platform that cannot read
// an axis omits it rather than reporting a fake 0.
func (s Snapshot) PrometheusText() string {
	var b strings.Builder
	writeHelp(&b, "fak_harness_cpu_seconds_total", "CPU seconds consumed by the fak guard harness, by half (kernel=guard+gateway, agent=wrapped child) and mode (user/system).", "gauge")
	for _, hv := range []struct {
		name string
		h    Half
	}{{"kernel", s.Kernel}, {"agent", s.Agent}} {
		if hv.h.HaveCPU {
			fmt.Fprintf(&b, "fak_harness_cpu_seconds_total{half=%q,mode=\"user\"} %s\n", hv.name, promFloat(hv.h.CPUUser.Seconds()))
			fmt.Fprintf(&b, "fak_harness_cpu_seconds_total{half=%q,mode=\"system\"} %s\n", hv.name, promFloat(hv.h.CPUSys.Seconds()))
		}
	}
	writeHelp(&b, "fak_harness_rss_bytes", "Current resident set size of the fak guard harness, by half.", "gauge")
	writeHalfGauge(&b, "fak_harness_rss_bytes", "kernel", s.Kernel.HaveRSS, float64(s.Kernel.RSSBytes))
	writeHalfGauge(&b, "fak_harness_rss_bytes", "agent", s.Agent.HaveRSS, float64(s.Agent.RSSBytes))
	writeHelp(&b, "fak_harness_peak_rss_bytes", "Peak resident set size of the fak guard harness this session, by half.", "gauge")
	writeHalfGauge(&b, "fak_harness_peak_rss_bytes", "kernel", s.Kernel.HavePeakRSS, float64(s.Kernel.PeakRSSBytes))
	writeHalfGauge(&b, "fak_harness_peak_rss_bytes", "agent", s.Agent.HavePeakRSS, float64(s.Agent.PeakRSSBytes))
	writeHelp(&b, "fak_harness_io_bytes_total", "Disk/block I/O bytes by the fak guard harness, by half and direction.", "gauge")
	if s.Kernel.HaveIO {
		fmt.Fprintf(&b, "fak_harness_io_bytes_total{half=\"kernel\",dir=\"read\"} %s\n", promFloat(float64(s.Kernel.IOReadBytes)))
		fmt.Fprintf(&b, "fak_harness_io_bytes_total{half=\"kernel\",dir=\"write\"} %s\n", promFloat(float64(s.Kernel.IOWriteBytes)))
	}
	if s.Agent.HaveIO {
		fmt.Fprintf(&b, "fak_harness_io_bytes_total{half=\"agent\",dir=\"read\"} %s\n", promFloat(float64(s.Agent.IOReadBytes)))
		fmt.Fprintf(&b, "fak_harness_io_bytes_total{half=\"agent\",dir=\"write\"} %s\n", promFloat(float64(s.Agent.IOWriteBytes)))
	}
	writeHelp(&b, "fak_harness_net_bytes_total", "Upstream network bytes by the fak guard harness (the gateway's agent↔LLM proxy traffic), by half and direction.", "gauge")
	if s.Kernel.HaveNet {
		fmt.Fprintf(&b, "fak_harness_net_bytes_total{half=\"kernel\",dir=\"rx\"} %s\n", promFloat(float64(s.Kernel.NetRxBytes)))
		fmt.Fprintf(&b, "fak_harness_net_bytes_total{half=\"kernel\",dir=\"tx\"} %s\n", promFloat(float64(s.Kernel.NetTxBytes)))
	}
	if s.Agent.HaveNet {
		fmt.Fprintf(&b, "fak_harness_net_bytes_total{half=\"agent\",dir=\"rx\"} %s\n", promFloat(float64(s.Agent.NetRxBytes)))
		fmt.Fprintf(&b, "fak_harness_net_bytes_total{half=\"agent\",dir=\"tx\"} %s\n", promFloat(float64(s.Agent.NetTxBytes)))
	}
	if s.HaveGPU {
		writeHelp(&b, "fak_harness_gpu_vram_bytes", "Accelerator VRAM used/total by the fak guard harness when a model runs in-kernel (--gguf/--backend). Absent on the default proxy path (no local GPU).", "gauge")
		fmt.Fprintf(&b, "fak_harness_gpu_vram_bytes{kind=\"used\"} %s\n", promFloat(float64(s.GPUVRAMUsedBytes)))
		if s.GPUVRAMTotalBytes > 0 {
			fmt.Fprintf(&b, "fak_harness_gpu_vram_bytes{kind=\"total\"} %s\n", promFloat(float64(s.GPUVRAMTotalBytes)))
		}
	}
	writeHelp(&b, "fak_harness_goroutines", "Peak goroutine count observed in the fak guard kernel this session.", "gauge")
	fmt.Fprintf(&b, "fak_harness_goroutines %d\n", s.GoroutinesPeak)
	writeHelp(&b, "fak_harness_go_heap_sys_bytes", "Peak Go heap bytes obtained from the OS by the fak guard kernel this session.", "gauge")
	fmt.Fprintf(&b, "fak_harness_go_heap_sys_bytes %s\n", promFloat(float64(s.GoHeapSysBytes)))
	writeHelp(&b, "fak_harness_num_cpu", "Logical CPU count of the host running the fak guard harness.", "gauge")
	fmt.Fprintf(&b, "fak_harness_num_cpu %d\n", s.NumCPU)
	writeHelp(&b, "fak_harness_elapsed_seconds", "Wall-clock seconds the fak guard harness was sampled this session.", "gauge")
	fmt.Fprintf(&b, "fak_harness_elapsed_seconds %s\n", promFloat(s.Elapsed.Seconds()))
	return b.String()
}

func writeHalfGauge(b *strings.Builder, name, half string, have bool, v float64) {
	if have {
		fmt.Fprintf(b, "%s{half=%q} %s\n", name, half, promFloat(v))
	}
}

func writeHelp(b *strings.Builder, name, help, typ string) {
	fmt.Fprintf(b, "# HELP %s %s\n# TYPE %s %s\n", name, help, name, typ)
}

func promFloat(v float64) string {
	return strings.TrimRight(strings.TrimRight(fmt.Sprintf("%.6f", v), "0"), ".")
}

// ledgerRow is the durable JSONL shape. Pointer fields + omitempty keep an axis a
// platform cannot read OUT of the row rather than recording a misleading 0.
type ledgerRow struct {
	Schema           string   `json:"schema"`
	TS               string   `json:"ts"`
	Mode             string   `json:"mode"`
	Provider         string   `json:"provider"`
	AgentName        string   `json:"agent_name"`
	ElapsedS         float64  `json:"elapsed_s"`
	Samples          int      `json:"samples"`
	Kernel           halfJSON `json:"kernel"`
	Agent            halfJSON `json:"agent"`
	KernelCPUPctPeak *float64 `json:"kernel_cpu_pct_peak,omitempty"`
	GoroutinesPeak   int      `json:"goroutines_peak"`
	GoHeapSysBytes   uint64   `json:"go_heap_sys_bytes"`
	NumCPU           int      `json:"num_cpu"`
	GOMAXPROCS       int      `json:"gomaxprocs"`
	GPUVRAMUsed      *uint64  `json:"gpu_vram_used_bytes,omitempty"`
	GPUVRAMTotal     *uint64  `json:"gpu_vram_total_bytes,omitempty"`
}

type halfJSON struct {
	CPUUserS     *float64 `json:"cpu_user_s,omitempty"`
	CPUSysS      *float64 `json:"cpu_sys_s,omitempty"`
	RSSBytes     *uint64  `json:"rss_bytes,omitempty"`
	PeakRSSBytes *uint64  `json:"peak_rss_bytes,omitempty"`
	IOReadBytes  *uint64  `json:"io_read_bytes,omitempty"`
	IOWriteBytes *uint64  `json:"io_write_bytes,omitempty"`
	NetRxBytes   *uint64  `json:"net_rx_bytes,omitempty"`
	NetTxBytes   *uint64  `json:"net_tx_bytes,omitempty"`
}

func (h Half) toJSON() halfJSON {
	var j halfJSON
	if h.HaveCPU {
		u, s := h.CPUUser.Seconds(), h.CPUSys.Seconds()
		j.CPUUserS, j.CPUSysS = &u, &s
	}
	if h.HaveRSS {
		v := h.RSSBytes
		j.RSSBytes = &v
	}
	if h.HavePeakRSS {
		v := h.PeakRSSBytes
		j.PeakRSSBytes = &v
	}
	if h.HaveIO {
		r, w := h.IOReadBytes, h.IOWriteBytes
		j.IOReadBytes, j.IOWriteBytes = &r, &w
	}
	if h.HaveNet {
		rx, tx := h.NetRxBytes, h.NetTxBytes
		j.NetRxBytes, j.NetTxBytes = &rx, &tx
	}
	return j
}

// MarshalLedgerRow renders one durable JSONL row (no trailing newline).
func (s Snapshot) MarshalLedgerRow(mode, provider, agent string, now time.Time) ([]byte, error) {
	row := ledgerRow{
		Schema:         LedgerSchema,
		TS:             now.UTC().Format(time.RFC3339),
		Mode:           mode,
		Provider:       provider,
		AgentName:      agent,
		ElapsedS:       s.Elapsed.Seconds(),
		Samples:        s.Samples,
		Kernel:         s.Kernel.toJSON(),
		Agent:          s.Agent.toJSON(),
		GoroutinesPeak: s.GoroutinesPeak,
		GoHeapSysBytes: s.GoHeapSysBytes,
		NumCPU:         s.NumCPU,
		GOMAXPROCS:     s.GOMAXPROCS,
	}
	if s.HaveKernelCPUPeak {
		p := s.KernelCPUPercentPeak
		row.KernelCPUPctPeak = &p
	}
	if s.HaveGPU {
		used, total := s.GPUVRAMUsedBytes, s.GPUVRAMTotalBytes
		row.GPUVRAMUsed = &used
		if total > 0 {
			row.GPUVRAMTotal = &total
		}
	}
	return json.Marshal(row)
}

func humanBytes(b uint64) string {
	const unit = 1024
	if b < unit {
		return fmt.Sprintf("%d B", b)
	}
	div, exp := uint64(unit), 0
	for n := b / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %ciB", float64(b)/float64(div), "KMGTPE"[exp])
}

func humanDur(d time.Duration) string {
	if d >= time.Second {
		return d.Round(time.Second).String()
	}
	if d <= 0 {
		return "0s"
	}
	return d.Round(time.Millisecond).String()
}
