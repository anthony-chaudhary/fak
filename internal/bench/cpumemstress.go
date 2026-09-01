// cpumemstress.go is the dependency-free, bounded CPU + memory stress witness
// requested by #4371 (gap found during the 24h da33 resource campaign, #4367).
//
// WHY THIS EXISTS. The da33 campaign could inventory services and probe health
// endpoints, but the standard synthetic CPU/memory stress witness was
// unavailable because `stress-ng` is not installed on the box and the campaign
// is not allowed to install packages or take a privileged config. This arm is
// the fallback: it stresses CPU and memory using ONLY the Go standard library
// (crypto/sha256 for the CPU work, a page-touched byte slab for the memory
// work), so it runs anywhere the fak binary or `go test` already runs — no apt,
// no root, no sysctl.
//
// WHAT IT MEASURES (the #4371 done-condition, field by field):
//   - exact duration        — CPUArm.DurationNS is the measured wall time of the
//     hash arm; each MemArm carries alloc + touch times.
//   - concurrency           — CPUArm.Workers (the goroutine fan-out actually run).
//   - allocation size       — MemArm.RequestedBytes / TouchedBytes per slab.
//   - throughput            — CPUArm.HashesPerSec + ThroughputMBs; MemArm.ThroughputMBs.
//   - latency distribution  — exact percentiles over raw per-op samples (p50/p90/
//     p99/max/mean), NOT histogram buckets.
//   - host load/mem before+after — HostSnapshot from /proc/loadavg + /proc/meminfo
//     on Linux (da33), runtime.MemStats always.
//   - artifact digest       — Digest is the SHA-256 of the canonical report JSON
//     with Digest cleared, so the witness is self-verifying.
//
// SAFETY GATES (bounded + no runaway, the third done-condition):
//   - load gate    — refuses to start if /proc/loadavg 1-min load per CPU exceeds
//     MaxLoadPerCPU (default 4.0); we do not pile synthetic load on
//     an already-saturated host.
//   - temperature  — best-effort read of /sys/class/thermal/thermal_zone*/temp
//     (world-readable millidegrees C, no root); refuses over
//     MaxTempMilliC (default 90000 = 90°C). Degrades to "unchecked"
//     where no thermal zone is exposed — an honest RED, not a guess.
//   - timeout      — the whole run is bounded by a context deadline (Timeout) AND
//     each arm is duration/size bounded, so it terminates even if a
//     gate read hangs.
//   - memory cap   — total resident slab is capped at MaxMemFraction of
//     MemAvailable (default 0.5); oversized slabs are SKIPPED and
//     recorded, so a 1 GiB request cannot OOM a smaller box.
//
// FENCE. Numbers land OBSERVED with the host named (FAK_BENCH_HW); this arm
// asserts and flips NO gate — it is an operator stress witness, not a CI floor.
package bench

import (
	"bufio"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

// CPUMemSchema versions the CPU+memory stress artifact.
const CPUMemSchema = "fak.cpumemstress.v1"

// cpuMemFence travels with every artifact so a reader cannot mistake an operator
// stress witness for a CI performance floor.
const cpuMemFence = "OBSERVED operator stress witness (#4371): dependency-free CPU (SHA-256) + memory (page-touch) load, bounded by duration/size/timeout with load+temperature safety gates; host named via FAK_BENCH_HW; NO gate asserted or flipped."

// StressNGGuidance documents when to reach for this arm versus `stress-ng`. It is
// embedded in every report (the "document when to use it versus stress-ng"
// done-condition ships WITH the witness, not just in a doc that can drift away
// from the artifact).
const StressNGGuidance = "Use this dependency-free arm when the host forbids package installation or " +
	"privileged config (the da33 case: no stress-ng, no apt, no root) and you need a bounded, self-" +
	"verifying CPU+memory witness that runs wherever `go test`/the fak binary runs. It hashes with " +
	"crypto/sha256 (CPU) and page-touches a byte slab (memory), records throughput + an exact latency " +
	"distribution + host load/mem before and after, and digests its own report. Prefer `stress-ng` " +
	"when it IS installed and you need its breadth: dozens of stressor methods (VM, cache, I/O, NUMA, " +
	"thermal), --cpu-method micro-kernels, or matched cross-fleet numbers against an external baseline. " +
	"This arm is intentionally two stressors (hash + page-touch), not a stress-ng replacement — it is " +
	"the always-available floor, not the ceiling."

// pageBytes is the touch stride for the memory arm — one write per 4 KiB page
// faults the page in without dirtying every byte (the OS maps in page units).
const pageBytes = 4096

// SystemTestUtilizationProfile names the host role whose resource defaults a
// system witness should use.
type SystemTestUtilizationProfile string

const (
	// SystemTestProfileSharedDevelopment reserves interactive headroom: 80%
	// CPU/GPU requested capacity, with at least one logical CPU left unused when
	// the host has more than one.
	SystemTestProfileSharedDevelopment SystemTestUtilizationProfile = "shared-development"
	// SystemTestProfileRemote is for an unattended remote worker: 90% CPU/GPU.
	SystemTestProfileRemote SystemTestUtilizationProfile = "remote"
	// SystemTestProfileDedicatedInference is for a dedicated inference host:
	// 100% CPU/GPU.
	SystemTestProfileDedicatedInference SystemTestUtilizationProfile = "dedicated-inference"
)

const gpuUtilizationLimitation = "requested policy ceiling only; this CPU/memory witness does not enforce or measure GPU utilization"

// SystemTestUtilizationRequest is the reusable input contract for resolving a
// system witness's requested host capacity. Zero percentages select the
// profile defaults; negative percentages are invalid and values above 100 are
// clamped to 100. Concurrency, when positive, is the final exact CPU override.
type SystemTestUtilizationRequest struct {
	Profile     SystemTestUtilizationProfile
	CPUPercent  int
	GPUPercent  int
	Concurrency int
}

// SystemTestUtilization is the resolved, artifact-ready utilization contract.
// GPUPercent is recorded for a shared vocabulary with GPU witnesses; the
// limitation states honestly that this witness does not apply it yet.
type SystemTestUtilization struct {
	Profile       SystemTestUtilizationProfile `json:"profile"`
	CPUPercent    int                          `json:"cpu_percent"`
	GPUPercent    int                          `json:"gpu_percent"`
	CPUCapacity   int                          `json:"cpu_capacity"`
	GPULimitation string                       `json:"gpu_limitation"`
}

// ResolveSystemTestUtilization resolves a profile and overrides against a
// supplied logical CPU count. Supplying the count makes policy tests and other
// system witnesses deterministic instead of coupling them to the current host.
func ResolveSystemTestUtilization(req SystemTestUtilizationRequest, logicalCPUs int) (SystemTestUtilization, error) {
	if logicalCPUs < 1 {
		return SystemTestUtilization{}, fmt.Errorf("logical CPU count must be positive: %d", logicalCPUs)
	}
	if req.Concurrency < 0 {
		return SystemTestUtilization{}, fmt.Errorf("concurrency must not be negative: %d", req.Concurrency)
	}

	profile := req.Profile
	if profile == "" {
		profile = SystemTestProfileSharedDevelopment
	}
	var defaultPercent int
	switch profile {
	case SystemTestProfileSharedDevelopment:
		defaultPercent = 80
	case SystemTestProfileRemote:
		defaultPercent = 90
	case SystemTestProfileDedicatedInference:
		defaultPercent = 100
	default:
		return SystemTestUtilization{}, fmt.Errorf("unknown system-test utilization profile %q", profile)
	}

	cpuPercent, err := resolveUtilizationPercent("CPU", req.CPUPercent, defaultPercent)
	if err != nil {
		return SystemTestUtilization{}, err
	}
	gpuPercent, err := resolveUtilizationPercent("GPU", req.GPUPercent, defaultPercent)
	if err != nil {
		return SystemTestUtilization{}, err
	}

	capacity := int(int64(logicalCPUs) * int64(cpuPercent) / 100)
	if capacity < 1 {
		capacity = 1
	}
	if capacity > logicalCPUs {
		capacity = logicalCPUs
	}
	if profile == SystemTestProfileSharedDevelopment && logicalCPUs > 1 && capacity >= logicalCPUs {
		capacity = logicalCPUs - 1
	}
	if req.Concurrency > 0 {
		capacity = req.Concurrency
	}

	return SystemTestUtilization{
		Profile:       profile,
		CPUPercent:    cpuPercent,
		GPUPercent:    gpuPercent,
		CPUCapacity:   capacity,
		GPULimitation: gpuUtilizationLimitation,
	}, nil
}

func resolveUtilizationPercent(resource string, requested, defaultPercent int) (int, error) {
	if requested < 0 {
		return 0, fmt.Errorf("%s percent must not be negative: %d", resource, requested)
	}
	if requested == 0 {
		return defaultPercent, nil
	}
	if requested > 100 {
		return 100, nil
	}
	return requested, nil
}

// CPUMemConfig sizes a run. Zero fields take the defaults in withDefaults, so a
// bare CPUMemConfig{} is a shared-development run that reserves host headroom.
type CPUMemConfig struct {
	// Profile selects shared-development (default), remote, or dedicated-inference.
	Profile SystemTestUtilizationProfile `json:"profile,omitempty"`
	// CPUPercent overrides the profile CPU percentage; 0 => profile default,
	// negative => invalid, >100 => clamped to 100.
	CPUPercent int `json:"cpu_percent,omitempty"`
	// GPUPercent overrides the recorded GPU percentage with the same semantics.
	// This witness records but does not enforce or measure GPU utilization.
	GPUPercent int `json:"gpu_percent,omitempty"`
	// Concurrency is the CPU hash fan-out; positive values are exact final overrides.
	Concurrency int `json:"concurrency"`
	// CPUDuration bounds the hash arm's wall time; 0 => 2s.
	CPUDuration time.Duration `json:"cpu_duration"`
	// BlockBytes is one hash op's input size; 0 => 1 MiB.
	BlockBytes int64 `json:"block_bytes"`
	// AllocSizes are the memory-arm slab sizes in bytes; nil => 64 MiB, 256 MiB, 1 GiB.
	AllocSizes []int64 `json:"alloc_sizes"`
	// ChunkBytes is the memory-arm touch-latency sampling granularity; 0 => 16 MiB.
	ChunkBytes int64 `json:"chunk_bytes"`
	// Timeout hard-bounds the whole run; 0 => CPUDuration + 60s.
	Timeout time.Duration `json:"timeout"`
	// MaxLoadPerCPU trips the load safety gate; 0 => 4.0. Negative disables it.
	MaxLoadPerCPU float64 `json:"max_load_per_cpu"`
	// MaxTempMilliC trips the temperature gate (millidegrees C); 0 => 90000. Negative disables it.
	MaxTempMilliC int64 `json:"max_temp_milli_c"`
	// MaxMemFraction caps total slab at this fraction of MemAvailable; 0 => 0.5. Negative disables it.
	MaxMemFraction float64 `json:"max_mem_fraction"`
}

func (c CPUMemConfig) withDefaults(logicalCPUs int) (CPUMemConfig, SystemTestUtilization, error) {
	utilization, err := ResolveSystemTestUtilization(SystemTestUtilizationRequest{
		Profile: c.Profile, CPUPercent: c.CPUPercent, GPUPercent: c.GPUPercent, Concurrency: c.Concurrency,
	}, logicalCPUs)
	if err != nil {
		return CPUMemConfig{}, SystemTestUtilization{}, err
	}
	c.Profile = utilization.Profile
	c.CPUPercent = utilization.CPUPercent
	c.GPUPercent = utilization.GPUPercent
	c.Concurrency = utilization.CPUCapacity
	if c.CPUDuration <= 0 {
		c.CPUDuration = 2 * time.Second
	}
	if c.BlockBytes <= 0 {
		c.BlockBytes = 1 << 20
	}
	if c.AllocSizes == nil {
		c.AllocSizes = []int64{64 << 20, 256 << 20, 1 << 30}
	}
	if c.ChunkBytes <= 0 {
		c.ChunkBytes = 16 << 20
	}
	if c.Timeout <= 0 {
		c.Timeout = c.CPUDuration + 60*time.Second
	}
	if c.MaxLoadPerCPU == 0 {
		c.MaxLoadPerCPU = 4.0
	}
	if c.MaxTempMilliC == 0 {
		c.MaxTempMilliC = 90000
	}
	if c.MaxMemFraction == 0 {
		c.MaxMemFraction = 0.5
	}
	return c, utilization, nil
}

// HostSnapshot is the host's load + memory view at one instant. The /proc fields
// are Linux-only (present on da33); the runtime fields are always populated so a
// non-Linux smoke run still records a before/after delta.
type HostSnapshot struct {
	Source         string  `json:"source"` // "linux:/proc+runtime" | "runtime"
	Load1          float64 `json:"load1,omitempty"`
	Load5          float64 `json:"load5,omitempty"`
	Load15         float64 `json:"load15,omitempty"`
	MemTotalBytes  int64   `json:"mem_total_bytes,omitempty"`
	MemAvailBytes  int64   `json:"mem_available_bytes,omitempty"`
	MemFreeBytes   int64   `json:"mem_free_bytes,omitempty"`
	HeapAllocBytes uint64  `json:"heap_alloc_bytes"`
	HeapSysBytes   uint64  `json:"heap_sys_bytes"`
	NumGC          uint32  `json:"num_gc"`
}

// GateResult records one safety gate's reading and whether it tripped.
type GateResult struct {
	Checked bool    `json:"checked"`
	Value   float64 `json:"value,omitempty"`
	Limit   float64 `json:"limit,omitempty"`
	Tripped bool    `json:"tripped"`
	Note    string  `json:"note,omitempty"`
}

// SafetyGates folds the pre-flight refusal checks.
type SafetyGates struct {
	Load        GateResult `json:"load_gate"`
	Temperature GateResult `json:"temperature_gate"`
	TimeoutNS   int64      `json:"timeout_ns"`
	MemFraction float64    `json:"mem_fraction_cap"`
}

// CPUArm is the hash-throughput arm's result.
type CPUArm struct {
	Workers       int     `json:"workers"`
	BlockBytes    int64   `json:"block_bytes"`
	Ops           int64   `json:"ops"`
	DurationNS    int64   `json:"duration_ns"`
	HashesPerSec  float64 `json:"hashes_per_sec"`
	ThroughputMBs float64 `json:"throughput_mb_s"`
	LatSamples    int     `json:"lat_samples"`
	P50NS         int64   `json:"p50_ns"`
	P90NS         int64   `json:"p90_ns"`
	P99NS         int64   `json:"p99_ns"`
	MaxNS         int64   `json:"max_ns"`
	MeanNS        float64 `json:"mean_ns"`
	// Checksum folds every digest so the compiler cannot elide the hashing.
	Checksum uint64 `json:"checksum"`
}

// MemArm is one slab's alloc + page-touch result.
type MemArm struct {
	RequestedBytes int64   `json:"requested_bytes"`
	TouchedBytes   int64   `json:"touched_bytes"`
	Skipped        bool    `json:"skipped,omitempty"`
	SkipReason     string  `json:"skip_reason,omitempty"`
	PageBytes      int64   `json:"page_bytes"`
	Pages          int64   `json:"pages"`
	AllocNS        int64   `json:"alloc_ns"`
	TouchNS        int64   `json:"touch_ns"`
	ThroughputMBs  float64 `json:"throughput_mb_s"`
	ChunkBytes     int64   `json:"chunk_bytes"`
	P50NS          int64   `json:"p50_ns"`
	P99NS          int64   `json:"p99_ns"`
	MaxNS          int64   `json:"max_ns"`
	// Checksum sums touched bytes so the page-touch loop cannot be optimized away.
	Checksum uint64 `json:"checksum"`
}

// CPUMemReport is the committed, self-verifying witness artifact.
type CPUMemReport struct {
	Schema       string                `json:"schema"`
	Issue        string                `json:"issue"`
	GeneratedAt  string                `json:"generated_at"`
	Hardware     string                `json:"hardware"`
	GoOS         string                `json:"goos"`
	GoArch       string                `json:"goarch"`
	NumCPU       int                   `json:"num_cpu"`
	GoMaxProcs   int                   `json:"gomaxprocs"`
	GoVersion    string                `json:"go_version"`
	Fence        string                `json:"fence"`
	StressNGWhen string                `json:"stress_ng_when"`
	Config       CPUMemConfig          `json:"config"`
	Utilization  SystemTestUtilization `json:"utilization"`
	Gates        SafetyGates           `json:"gates"`
	Refused      bool                  `json:"refused"`
	RefusedWhy   string                `json:"refused_why,omitempty"`
	HostBefore   HostSnapshot          `json:"host_before"`
	HostAfter    HostSnapshot          `json:"host_after"`
	CPU          CPUArm                `json:"cpu"`
	Memory       []MemArm              `json:"memory"`
	Digest       string                `json:"digest"`
}

// RunCPUMemStress runs the bounded CPU + memory stress arm and returns the
// self-verifying report. Hermetic: no network, no model, no filesystem writes
// (the caller decides whether to persist the JSON). A tripped pre-flight gate
// returns the report with Refused=true and a non-nil error — the run is REFUSED,
// not silently skipped, so an operator sees why no load was applied.
func RunCPUMemStress(ctx context.Context, cfg CPUMemConfig) (*CPUMemReport, error) {
	logicalCPUs := runtime.NumCPU()
	cfg, utilization, err := cfg.withDefaults(logicalCPUs)
	if err != nil {
		return nil, fmt.Errorf("cpumemstress: resolve utilization: %w", err)
	}
	ctx, cancel := context.WithTimeout(ctx, cfg.Timeout)
	defer cancel()

	rep := &CPUMemReport{
		Schema:       CPUMemSchema,
		Issue:        "#4371 (parent #4367)",
		GeneratedAt:  time.Now().UTC().Format(time.RFC3339),
		Hardware:     hardwareLabel(),
		GoOS:         runtime.GOOS,
		GoArch:       runtime.GOARCH,
		NumCPU:       logicalCPUs,
		GoMaxProcs:   runtime.GOMAXPROCS(0),
		GoVersion:    runtime.Version(),
		Fence:        cpuMemFence,
		StressNGWhen: StressNGGuidance,
		Config:       cfg,
		Utilization:  utilization,
		Memory:       []MemArm{},
	}
	rep.Gates.TimeoutNS = cfg.Timeout.Nanoseconds()
	rep.Gates.MemFraction = cfg.MaxMemFraction

	rep.HostBefore = snapshotHost()

	// Pre-flight safety gates: refuse BEFORE applying any load.
	rep.Gates.Load = loadGate(rep.HostBefore, cfg.MaxLoadPerCPU, rep.NumCPU)
	rep.Gates.Temperature = temperatureGate(cfg.MaxTempMilliC)
	if rep.Gates.Load.Tripped || rep.Gates.Temperature.Tripped {
		rep.Refused = true
		rep.RefusedWhy = refusalReason(rep.Gates)
		rep.HostAfter = rep.HostBefore
		rep.Digest = rep.computeDigest()
		return rep, fmt.Errorf("cpumemstress: refused by safety gate: %s", rep.RefusedWhy)
	}

	rep.CPU = runCPUArm(ctx, cfg)
	rep.Memory = runMemArms(ctx, cfg, rep.HostBefore)

	rep.HostAfter = snapshotHost()
	rep.Digest = rep.computeDigest()
	return rep, ctx.Err()
}

// runCPUArm fans out cfg.Concurrency workers each hashing a private BlockBytes
// buffer with SHA-256 until the shared deadline, recording per-op latency and a
// digest checksum. Aggregate throughput is total ops over WALL time (not summed
// per-worker), so HashesPerSec reflects real parallel speedup.
func runCPUArm(ctx context.Context, cfg CPUMemConfig) CPUArm {
	deadline := time.Now().Add(cfg.CPUDuration)
	type workerOut struct {
		lat []int64
		sum uint64
	}
	outs := make([]workerOut, cfg.Concurrency)
	var wg sync.WaitGroup
	start := time.Now()
	for w := 0; w < cfg.Concurrency; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			block := make([]byte, cfg.BlockBytes)
			for i := range block {
				block[i] = byte(i*31 + w*7)
			}
			var lat []int64
			var sum uint64
			for time.Now().Before(deadline) {
				if ctx.Err() != nil {
					break
				}
				t0 := time.Now()
				d := sha256.Sum256(block)
				lat = append(lat, time.Since(t0).Nanoseconds())
				sum ^= binary.LittleEndian.Uint64(d[:8])
				// Feed the digest back into the block so each op depends on the
				// last — defeats any hoisting of the constant-input hash.
				block[0] ^= d[0]
			}
			outs[w] = workerOut{lat: lat, sum: sum}
		}(w)
	}
	wg.Wait()
	elapsed := time.Since(start)

	var all []int64
	var ops int64
	var checksum uint64
	for _, o := range outs {
		all = append(all, o.lat...)
		ops += int64(len(o.lat))
		checksum ^= o.sum
	}
	arm := CPUArm{
		Workers:    cfg.Concurrency,
		BlockBytes: cfg.BlockBytes,
		Ops:        ops,
		DurationNS: elapsed.Nanoseconds(),
		LatSamples: len(all),
		Checksum:   checksum,
	}
	if secs := elapsed.Seconds(); secs > 0 {
		arm.HashesPerSec = float64(ops) / secs
		arm.ThroughputMBs = float64(ops*cfg.BlockBytes) / (1 << 20) / secs
	}
	arm.P50NS, arm.P90NS, arm.P99NS, arm.MaxNS, arm.MeanNS = percentiles(all)
	return arm
}

// runMemArms allocates and page-touches each configured slab, honoring the
// memory safety cap derived from MemAvailable. Oversized slabs are recorded as
// skipped rather than risking an OOM on a smaller host.
func runMemArms(ctx context.Context, cfg CPUMemConfig, before HostSnapshot) []MemArm {
	capBytes, capped := memCapBytes(cfg, before)
	arms := make([]MemArm, 0, len(cfg.AllocSizes))
	for _, size := range cfg.AllocSizes {
		arm := MemArm{RequestedBytes: size, PageBytes: pageBytes, ChunkBytes: cfg.ChunkBytes}
		if size <= 0 {
			arm.Skipped, arm.SkipReason = true, "non-positive size"
			arms = append(arms, arm)
			continue
		}
		if capped && size > capBytes {
			arm.Skipped = true
			arm.SkipReason = fmt.Sprintf("size %d > memory cap %d (%.0f%% of MemAvailable)", size, capBytes, cfg.MaxMemFraction*100)
			arms = append(arms, arm)
			continue
		}
		if ctx.Err() != nil {
			arm.Skipped, arm.SkipReason = true, "timeout/cancel before slab"
			arms = append(arms, arm)
			continue
		}
		arms = append(arms, touchSlab(size, cfg.ChunkBytes))
	}
	return arms
}

// touchSlab allocates size bytes, faults in every page with one write, and times
// each ChunkBytes worth of touching for a latency distribution. The touched-byte
// checksum is read back so neither the allocation nor the writes can be elided.
func touchSlab(size, chunkBytes int64) MemArm {
	arm := MemArm{RequestedBytes: size, PageBytes: pageBytes, ChunkBytes: chunkBytes}
	t0 := time.Now()
	buf := make([]byte, size)
	arm.AllocNS = time.Since(t0).Nanoseconds()

	var chunkLat []int64
	var sum uint64
	var pages int64
	chunkStart := time.Now()
	var chunkAccum int64
	touchStart := time.Now()
	for i := int64(0); i < size; i += pageBytes {
		buf[i] = byte(i>>12) + 1
		sum += uint64(buf[i])
		pages++
		chunkAccum += pageBytes
		if chunkAccum >= chunkBytes {
			chunkLat = append(chunkLat, time.Since(chunkStart).Nanoseconds())
			chunkAccum = 0
			chunkStart = time.Now()
		}
	}
	arm.TouchNS = time.Since(touchStart).Nanoseconds()
	arm.TouchedBytes = pages * pageBytes
	arm.Pages = pages
	arm.Checksum = sum
	if secs := time.Duration(arm.TouchNS).Seconds(); secs > 0 {
		arm.ThroughputMBs = float64(arm.TouchedBytes) / (1 << 20) / secs
	}
	arm.P50NS, _, arm.P99NS, arm.MaxNS, _ = percentiles(chunkLat)
	// Drop the slab reference and reclaim eagerly so the next (larger) slab sees
	// the freed pages rather than stacking toward the memory cap.
	buf = nil
	_ = buf
	runtime.GC()
	return arm
}

// memCapBytes is the absolute byte ceiling for a single slab: MaxMemFraction of
// MemAvailable when /proc/meminfo is readable (da33). capped=false means the cap
// is not enforced — the fraction is disabled (< 0) or MemAvailable is unknown
// (non-Linux smoke run), so the caller's AllocSizes are trusted as-is. When
// capped=true the ceiling is enforced even if it rounds to 0 bytes (a
// deliberately tiny fraction must skip every slab, not fall through as uncapped).
func memCapBytes(cfg CPUMemConfig, before HostSnapshot) (int64, bool) {
	if cfg.MaxMemFraction < 0 || before.MemAvailBytes <= 0 {
		return 0, false
	}
	return int64(float64(before.MemAvailBytes) * cfg.MaxMemFraction), true
}

// loadGate refuses to add synthetic load when the host is already busy. It is
// only meaningful where /proc/loadavg is readable (Linux); elsewhere it records
// Checked=false and never trips.
func loadGate(h HostSnapshot, maxPerCPU float64, numCPU int) GateResult {
	if maxPerCPU < 0 {
		return GateResult{Checked: false, Note: "load gate disabled (MaxLoadPerCPU < 0)"}
	}
	if h.Load1 <= 0 || numCPU <= 0 {
		return GateResult{Checked: false, Note: "no /proc/loadavg on this host (non-Linux?); load gate unchecked"}
	}
	perCPU := h.Load1 / float64(numCPU)
	return GateResult{
		Checked: true,
		Value:   perCPU,
		Limit:   maxPerCPU,
		Tripped: perCPU > maxPerCPU,
		Note:    fmt.Sprintf("1-min load %.2f over %d CPUs = %.2f/CPU", h.Load1, numCPU, perCPU),
	}
}

// temperatureGate best-effort reads the hottest /sys/class/thermal zone (no root,
// no lm-sensors) and refuses over the limit. Where no zone is exposed it records
// Checked=false — an honest "could not read" rather than a fabricated pass.
func temperatureGate(maxMilliC int64) GateResult {
	if maxMilliC < 0 {
		return GateResult{Checked: false, Note: "temperature gate disabled (MaxTempMilliC < 0)"}
	}
	hottest, ok := hottestThermalZone()
	if !ok {
		return GateResult{Checked: false, Note: "no readable /sys/class/thermal/thermal_zone*/temp (non-Linux or not exposed); temperature gate unchecked — rely on the load gate + bounded duration"}
	}
	return GateResult{
		Checked: true,
		Value:   float64(hottest) / 1000,
		Limit:   float64(maxMilliC) / 1000,
		Tripped: hottest > maxMilliC,
		Note:    fmt.Sprintf("hottest thermal zone %.1f°C", float64(hottest)/1000),
	}
}

// hottestThermalZone returns the max millidegree-C reading across thermal zones,
// ok=false when none is readable.
func hottestThermalZone() (int64, bool) {
	entries, err := os.ReadDir("/sys/class/thermal")
	if err != nil {
		return 0, false
	}
	var hottest int64
	found := false
	for _, e := range entries {
		if !strings.HasPrefix(e.Name(), "thermal_zone") {
			continue
		}
		b, err := os.ReadFile("/sys/class/thermal/" + e.Name() + "/temp")
		if err != nil {
			continue
		}
		v, err := strconv.ParseInt(strings.TrimSpace(string(b)), 10, 64)
		if err != nil {
			continue
		}
		if !found || v > hottest {
			hottest, found = v, true
		}
	}
	return hottest, found
}

func refusalReason(g SafetyGates) string {
	var parts []string
	if g.Load.Tripped {
		parts = append(parts, "load "+g.Load.Note)
	}
	if g.Temperature.Tripped {
		parts = append(parts, "temperature "+g.Temperature.Note)
	}
	return strings.Join(parts, "; ")
}

// snapshotHost folds /proc load + meminfo (Linux) and the Go runtime heap view
// into one instant snapshot.
func snapshotHost() HostSnapshot {
	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)
	s := HostSnapshot{
		Source:         "runtime",
		HeapAllocBytes: ms.HeapAlloc,
		HeapSysBytes:   ms.HeapSys,
		NumGC:          ms.NumGC,
	}
	l1, l5, l15, okL := readLoadAvg()
	total, avail, free, okM := readMemInfo()
	if okL || okM {
		s.Source = "linux:/proc+runtime"
	}
	if okL {
		s.Load1, s.Load5, s.Load15 = l1, l5, l15
	}
	if okM {
		s.MemTotalBytes, s.MemAvailBytes, s.MemFreeBytes = total, avail, free
	}
	return s
}

// readLoadAvg parses /proc/loadavg (Linux only). ok=false elsewhere.
func readLoadAvg() (l1, l5, l15 float64, ok bool) {
	b, err := os.ReadFile("/proc/loadavg")
	if err != nil {
		return 0, 0, 0, false
	}
	f := strings.Fields(string(b))
	if len(f) < 3 {
		return 0, 0, 0, false
	}
	l1, e1 := strconv.ParseFloat(f[0], 64)
	l5, e5 := strconv.ParseFloat(f[1], 64)
	l15, e15 := strconv.ParseFloat(f[2], 64)
	if e1 != nil || e5 != nil || e15 != nil {
		return 0, 0, 0, false
	}
	return l1, l5, l15, true
}

// readMemInfo parses MemTotal/MemAvailable/MemFree from /proc/meminfo (kB → bytes).
// ok=false where /proc/meminfo is absent (non-Linux).
func readMemInfo() (total, avail, free int64, ok bool) {
	f, err := os.Open("/proc/meminfo")
	if err != nil {
		return 0, 0, 0, false
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		fields := strings.Fields(sc.Text())
		if len(fields) < 2 {
			continue
		}
		kb, err := strconv.ParseInt(fields[1], 10, 64)
		if err != nil {
			continue
		}
		switch fields[0] {
		case "MemTotal:":
			total = kb * 1024
		case "MemAvailable:":
			avail = kb * 1024
		case "MemFree:":
			free = kb * 1024
		}
	}
	return total, avail, free, total > 0
}

// percentileAt is the nearest-rank pick of quantile p from an ALREADY-SORTED
// ascending sample slice: index floor(len*p), clamped to the last element so p=1.0
// (and any float overshoot) lands on the max rather than panicking. It is the single
// definition of "the p-th percentile" for every latency report in this package —
// cpumemstress's percentiles and tailload's foldTailArm both call it, so a published
// number can never mean one rounding here and a different one there. Callers must
// sort first; an empty slice has no percentile and the callers gate on that.
func percentileAt(sorted []int64, p float64) int64 {
	idx := int(float64(len(sorted)) * p)
	if idx >= len(sorted) {
		idx = len(sorted) - 1
	}
	return sorted[idx]
}

// percentiles folds raw nanosecond samples into exact p50/p90/p99/max/mean (no
// bucket quantization). Empty input yields all zeros.
func percentiles(samples []int64) (p50, p90, p99, max int64, mean float64) {
	if len(samples) == 0 {
		return 0, 0, 0, 0, 0
	}
	sorted := make([]int64, len(samples))
	copy(sorted, samples)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })
	var sum int64
	for _, s := range sorted {
		sum += s
	}
	return percentileAt(sorted, 0.50), percentileAt(sorted, 0.90), percentileAt(sorted, 0.99),
		sorted[len(sorted)-1], float64(sum) / float64(len(sorted))
}

// computeDigest returns the SHA-256 hex of the report's canonical JSON with the
// Digest field cleared, so a reader can recompute and confirm the witness was
// not edited after the fact.
func (r *CPUMemReport) computeDigest() string {
	return computeReportDigest(r, &r.Digest)
}

func computeReportDigest(report any, digest *string) string {
	saved := *digest
	*digest = ""
	b, err := json.Marshal(report)
	*digest = saved
	if err != nil {
		return ""
	}
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}
