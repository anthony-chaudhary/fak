package harnessres

import (
	"bufio"
	"bytes"
	"fmt"
	"os"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"
)

// HostSample carries a snapshot of host memory and Linux PSI (Pressure Stall Information)
// metrics used by DensityGovernor to size worker wave concurrency.
type HostSample struct {
	TotalRAMBytes  uint64        `json:"total_ram_bytes"`
	AvailRAMBytes  uint64        `json:"avail_ram_bytes"`
	HaveRAM        bool          `json:"have_ram"`
	TotalSwapBytes uint64        `json:"total_swap_bytes,omitempty"`
	AvailSwapBytes uint64        `json:"avail_swap_bytes,omitempty"`
	HaveSwap       bool          `json:"have_swap"`
	PSISomeTotal   time.Duration `json:"psi_some_total,omitempty"`
	PSIFullTotal   time.Duration `json:"psi_full_total,omitempty"`
	PSIAvg10       float64       `json:"psi_avg10,omitempty"`
	PSIAvg60       float64       `json:"psi_avg60,omitempty"`
	PSIAvg300      float64       `json:"psi_avg300,omitempty"`
	HavePSI        bool          `json:"have_psi"`
	Corrupted      bool          `json:"corrupted,omitempty"`
	Timestamp      time.Time     `json:"timestamp"`
}

// GovernorConfig configures the Vegas / TCP BBR-style density governor.
type GovernorConfig struct {
	MinWorkers               int           `json:"min_workers"`
	MaxWorkers               int           `json:"max_workers"`
	TargetAvailRAMFraction   float64       `json:"target_avail_ram_fraction"`
	BackoffThresholdFraction float64       `json:"backoff_threshold_fraction"`
	PSITriggerDuration       time.Duration `json:"psi_trigger_duration"`
	PSITriggerThreshold      float64       `json:"psi_trigger_threshold"`
	WorkerRAMEstimateBytes   uint64        `json:"worker_ram_estimate_bytes"`
	AdditiveStep             int           `json:"additive_step"`
	MultiplicativeDecay      float64       `json:"multiplicative_decay"`
	RequirePSI               bool          `json:"require_psi"`
}

// DefaultGovernorConfig returns recommended production defaults for host worker waves.
func DefaultGovernorConfig() GovernorConfig {
	return GovernorConfig{
		MinWorkers:               2,
		MaxWorkers:               80,
		TargetAvailRAMFraction:   0.25,
		BackoffThresholdFraction: 0.15,
		PSITriggerDuration:       100 * time.Millisecond,
		PSITriggerThreshold:      10.0,              // 10% stall in 10s window
		WorkerRAMEstimateBytes:   350 * 1024 * 1024, // 350 MiB estimated per worker
		AdditiveStep:             2,
		MultiplicativeDecay:      0.5,
		RequirePSI:               false,
	}
}

// GovernorTelemetry exports point-in-time governor health and admission state.
type GovernorTelemetry struct {
	Concurrency              int           `json:"concurrency"`
	MinWorkers               int           `json:"min_workers"`
	MaxWorkers               int           `json:"max_workers"`
	TargetAvailRAMFraction   float64       `json:"target_avail_ram_fraction"`
	BackoffThresholdFraction float64       `json:"backoff_threshold_fraction"`
	CurrentAvailRAMFraction  float64       `json:"current_avail_ram_fraction"`
	TotalRAMBytes            uint64        `json:"total_ram_bytes"`
	AvailRAMBytes            uint64        `json:"avail_ram_bytes"`
	PSIAvg10                 float64       `json:"psi_avg10"`
	PSIStallDelta            time.Duration `json:"psi_stall_delta_ns"`
	InBackoff                bool          `json:"in_backoff"`
	BackoffReason            string        `json:"backoff_reason,omitempty"`
	LastUpdate               time.Time     `json:"last_update"`
	AdmitAllowed             bool          `json:"admit_allowed"`
	Mode                     string        `json:"mode"`
}

// DensityGovernor regulates co-hosted agent worker wave density dynamically.
// It implements a Vegas / TCP BBR-style feedback controller driven by host memory
// pressure, available RAM fraction, and PSI stall duration.
type DensityGovernor struct {
	mu            sync.RWMutex
	cfg           GovernorConfig
	concurrency   int
	inBackoff     bool
	backoffReason string
	mode          string
	lastSample    HostSample
	hasSample     bool
	lastUpdate    time.Time
	prevSwapUsed  uint64
	hasPrevSwap   bool
	psiStallDelta time.Duration
}

// NewDensityGovernor creates a governor with validated configuration.
func NewDensityGovernor(cfg GovernorConfig) *DensityGovernor {
	norm := cfg
	if norm.MinWorkers <= 0 {
		norm.MinWorkers = 2
	}
	if norm.MaxWorkers < norm.MinWorkers {
		norm.MaxWorkers = norm.MinWorkers
	}
	if norm.TargetAvailRAMFraction <= 0 || norm.TargetAvailRAMFraction >= 1.0 {
		norm.TargetAvailRAMFraction = 0.25
	}
	if norm.BackoffThresholdFraction <= 0 || norm.BackoffThresholdFraction >= norm.TargetAvailRAMFraction {
		norm.BackoffThresholdFraction = 0.15
	}
	if norm.PSITriggerDuration <= 0 {
		norm.PSITriggerDuration = 100 * time.Millisecond
	}
	if norm.PSITriggerThreshold <= 0 {
		norm.PSITriggerThreshold = 10.0
	}
	if norm.WorkerRAMEstimateBytes == 0 {
		norm.WorkerRAMEstimateBytes = 350 * 1024 * 1024
	}
	if norm.AdditiveStep <= 0 {
		norm.AdditiveStep = 2
	}
	if norm.MultiplicativeDecay <= 0 || norm.MultiplicativeDecay >= 1.0 {
		norm.MultiplicativeDecay = 0.5
	}

	return &DensityGovernor{
		cfg:         norm,
		concurrency: norm.MinWorkers,
		mode:        "floor",
		lastUpdate:  time.Now(),
	}
}

// Update evaluates a host resource sample, adjusts the concurrency setpoint, and
// trips/resets the rapid back-off circuit as appropriate.
func (g *DensityGovernor) Update(sample HostSample) int {
	g.mu.Lock()
	defer g.mu.Unlock()

	g.lastUpdate = time.Now()
	if sample.Timestamp.IsZero() {
		sample.Timestamp = g.lastUpdate
	}

	// 1. Fail-closed defense: verify sample sanity.
	if sample.Corrupted || !sample.HaveRAM || sample.TotalRAMBytes == 0 || sample.AvailRAMBytes > sample.TotalRAMBytes || (sample.HaveSwap && sample.AvailSwapBytes > sample.TotalSwapBytes) {
		g.concurrency = g.cfg.MinWorkers
		g.inBackoff = true
		g.backoffReason = "fail-closed: host memory metrics unreadable or corrupted"
		g.mode = "fail_closed"
		g.lastSample = sample
		g.hasSample = true
		return g.concurrency
	}

	if g.cfg.RequirePSI && (!sample.HavePSI || sample.PSIAvg10 < 0 || sample.PSIAvg10 > 100) {
		g.concurrency = g.cfg.MinWorkers
		g.inBackoff = true
		g.backoffReason = "fail-closed: required PSI metrics unreadable or corrupted"
		g.mode = "fail_closed"
		g.lastSample = sample
		g.hasSample = true
		return g.concurrency
	}

	availFrac := float64(sample.AvailRAMBytes) / float64(sample.TotalRAMBytes)

	// 2. Rapid back-off check: Swap surge.
	swapSurge := false
	if sample.HaveSwap && sample.TotalSwapBytes > 0 {
		usedSwap := sample.TotalSwapBytes - sample.AvailSwapBytes
		if g.hasPrevSwap {
			// Swap surge triggered if used swap jumps by > 500 MiB or reaches > 80% while growing.
			if usedSwap > g.prevSwapUsed+(500*1024*1024) ||
				(float64(usedSwap)/float64(sample.TotalSwapBytes) > 0.80 && usedSwap > g.prevSwapUsed) {
				swapSurge = true
			}
		}
		g.prevSwapUsed = usedSwap
		g.hasPrevSwap = true
	}

	// 3. Rapid back-off check: PSI stall duration and stall rate.
	psiSurge := false
	g.psiStallDelta = 0
	if sample.HavePSI {
		if sample.PSIAvg10 >= g.cfg.PSITriggerThreshold {
			psiSurge = true
		}
		if g.hasSample && g.lastSample.HavePSI && sample.PSISomeTotal > g.lastSample.PSISomeTotal {
			g.psiStallDelta = sample.PSISomeTotal - g.lastSample.PSISomeTotal
			if g.psiStallDelta >= g.cfg.PSITriggerDuration {
				psiSurge = true
			}
		}
	}

	// 4. Rapid back-off check: Available RAM below back-off threshold (< 15%).
	ramSqueeze := availFrac < g.cfg.BackoffThresholdFraction

	if ramSqueeze || psiSurge || swapSurge {
		g.inBackoff = true
		g.mode = "backoff"
		if ramSqueeze {
			g.backoffReason = fmt.Sprintf("host available memory below backoff threshold: %.1f%% < %.1f%%", availFrac*100, g.cfg.BackoffThresholdFraction*100)
		} else if psiSurge {
			g.backoffReason = fmt.Sprintf("host PSI stall threshold exceeded: avg10=%.2f%% delta=%v", sample.PSIAvg10, g.psiStallDelta)
		} else {
			g.backoffReason = "host swap activity surge detected"
		}

		// Multiplicative decrease
		newConc := int(float64(g.concurrency) * g.cfg.MultiplicativeDecay)
		if newConc < g.cfg.MinWorkers {
			newConc = g.cfg.MinWorkers
		}
		g.concurrency = newConc
		g.lastSample = sample
		g.hasSample = true
		return g.concurrency
	}

	// 5. Normal operation: Clear back-off.
	g.inBackoff = false
	g.backoffReason = ""

	// 6. Vegas / TCP BBR feedback controller: calculate headroom and capacity ceiling.
	if availFrac >= g.cfg.TargetAvailRAMFraction {
		// Surplus RAM above target headroom
		targetReserveBytes := uint64(float64(sample.TotalRAMBytes) * g.cfg.TargetAvailRAMFraction)
		surplusBytes := uint64(0)
		if sample.AvailRAMBytes > targetReserveBytes {
			surplusBytes = sample.AvailRAMBytes - targetReserveBytes
		}

		workerSeatsFromSurplus := int(surplusBytes / g.cfg.WorkerRAMEstimateBytes)
		capacityCap := g.cfg.MinWorkers + workerSeatsFromSurplus
		if capacityCap > g.cfg.MaxWorkers {
			capacityCap = g.cfg.MaxWorkers
		}
		if capacityCap < g.cfg.MinWorkers {
			capacityCap = g.cfg.MinWorkers
		}

		if g.concurrency < capacityCap {
			// Proportional/additive ramp up towards capacity
			step := g.cfg.AdditiveStep
			headroomDelta := capacityCap - g.concurrency
			dynamicStep := int(float64(headroomDelta) * 0.25)
			if dynamicStep > step {
				step = dynamicStep
			}
			g.concurrency += step
			if g.concurrency > capacityCap {
				g.concurrency = capacityCap
			}
			g.mode = "growing"
		} else if g.concurrency > capacityCap {
			// Gently decay towards capacityCap
			g.concurrency -= g.cfg.AdditiveStep
			if g.concurrency < capacityCap {
				g.concurrency = capacityCap
			}
			if g.concurrency < g.cfg.MinWorkers {
				g.concurrency = g.cfg.MinWorkers
			}
			g.mode = "adjusting"
		} else {
			g.mode = "steady"
		}
	} else {
		// Between BackoffThresholdFraction (15%) and TargetAvailRAMFraction (25%):
		// Gentle downward pressure to prevent crossing into backoff.
		g.concurrency -= g.cfg.AdditiveStep
		if g.concurrency < g.cfg.MinWorkers {
			g.concurrency = g.cfg.MinWorkers
		}
		g.mode = "hedging"
	}

	g.lastSample = sample
	g.hasSample = true
	return g.concurrency
}

// UpdateHost is a convenience adapter wrapping Host into a HostSample.
func (g *DensityGovernor) UpdateHost(h Host) int {
	sample := HostSample{
		TotalRAMBytes: h.TotalRAMBytes,
		AvailRAMBytes: h.AvailRAMBytes,
		HaveRAM:       h.HaveRAM,
		Timestamp:     time.Now(),
	}
	return g.Update(sample)
}

// CurrentConcurrency returns the active worker concurrency limit.
func (g *DensityGovernor) CurrentConcurrency() int {
	g.mu.RLock()
	defer g.mu.RUnlock()
	return g.concurrency
}

// ShouldAdmit tests whether another worker turn admission is permitted.
func (g *DensityGovernor) ShouldAdmit() (admit bool, reason string) {
	g.mu.RLock()
	defer g.mu.RUnlock()
	if g.inBackoff {
		return false, g.backoffReason
	}
	return true, ""
}

// ResolveConcurrency resolves a concurrency argument ("auto", positive integer, or empty).
func (g *DensityGovernor) ResolveConcurrency(concurrencyArg string) int {
	arg := strings.ToLower(strings.TrimSpace(concurrencyArg))
	if arg == "" || arg == "auto" || arg == "dynamic" {
		return g.CurrentConcurrency()
	}
	n, err := strconv.Atoi(arg)
	if err == nil && n > 0 {
		return n
	}
	// Invalid, zero, or negative -> fail safe to conservative floor.
	g.mu.RLock()
	floor := g.cfg.MinWorkers
	g.mu.RUnlock()
	return floor
}

// Telemetry returns a snapshot of governor state for observability or JSON reporting.
func (g *DensityGovernor) Telemetry() GovernorTelemetry {
	g.mu.RLock()
	defer g.mu.RUnlock()

	availFrac := 0.0
	if g.hasSample && g.lastSample.HaveRAM && g.lastSample.TotalRAMBytes > 0 {
		availFrac = float64(g.lastSample.AvailRAMBytes) / float64(g.lastSample.TotalRAMBytes)
	}

	return GovernorTelemetry{
		Concurrency:              g.concurrency,
		MinWorkers:               g.cfg.MinWorkers,
		MaxWorkers:               g.cfg.MaxWorkers,
		TargetAvailRAMFraction:   g.cfg.TargetAvailRAMFraction,
		BackoffThresholdFraction: g.cfg.BackoffThresholdFraction,
		CurrentAvailRAMFraction:  availFrac,
		TotalRAMBytes:            g.lastSample.TotalRAMBytes,
		AvailRAMBytes:            g.lastSample.AvailRAMBytes,
		PSIAvg10:                 g.lastSample.PSIAvg10,
		PSIStallDelta:            g.psiStallDelta,
		InBackoff:                g.inBackoff,
		BackoffReason:            g.backoffReason,
		LastUpdate:               g.lastUpdate,
		AdmitAllowed:             !g.inBackoff,
		Mode:                     g.mode,
	}
}

// platformHostSampleReader is optionally installed by platform-specific init (e.g. Windows).
var platformHostSampleReader func() (HostSample, bool)

// globalHostMu and globalHostProvider allow manual injection or wiring from fak guard.
var (
	globalHostMu       sync.RWMutex
	globalHostProvider func() (Host, bool)
)

// SetGlobalHostProvider sets an optional process-wide Host provider function.
func SetGlobalHostProvider(fn func() (Host, bool)) {
	globalHostMu.Lock()
	globalHostProvider = fn
	globalHostMu.Unlock()
}

// ReadHostSample gathers live host memory and pressure metrics.
// On Windows, it invokes GlobalMemoryStatusEx.
// On Linux, it inspects /proc/meminfo and /proc/pressure/memory.
func ReadHostSample() HostSample {
	if platformHostSampleReader != nil {
		if s, ok := platformHostSampleReader(); ok {
			return s
		}
	}

	if runtime.GOOS == "linux" {
		sample := readLinuxHostSample()
		if sample.HaveRAM {
			return sample
		}
	}

	globalHostMu.RLock()
	provider := globalHostProvider
	globalHostMu.RUnlock()
	if provider != nil {
		if h, ok := provider(); ok && h.HaveRAM {
			return HostSample{
				TotalRAMBytes: h.TotalRAMBytes,
				AvailRAMBytes: h.AvailRAMBytes,
				HaveRAM:       true,
				Timestamp:     time.Now(),
			}
		}
	}

	return HostSample{
		Timestamp: time.Now(),
		HaveRAM:   false,
	}
}

func readLinuxHostSample() HostSample {
	var s HostSample
	s.Timestamp = time.Now()

	memBytes, err := os.ReadFile("/proc/meminfo")
	if err == nil {
		scanner := bufio.NewScanner(bytes.NewReader(memBytes))
		var memTotal, memAvail, swapTotal, swapFree uint64
		var haveTotal, haveAvail, haveSwapTotal, haveSwapFree bool

		for scanner.Scan() {
			line := scanner.Text()
			fields := strings.Fields(line)
			if len(fields) < 2 {
				continue
			}
			key := strings.TrimSuffix(fields[0], ":")
			val, parseErr := strconv.ParseUint(fields[1], 10, 64)
			if parseErr != nil {
				continue
			}
			// /proc/meminfo reports in kB
			valBytes := val * 1024
			switch key {
			case "MemTotal":
				memTotal = valBytes
				haveTotal = true
			case "MemAvailable":
				memAvail = valBytes
				haveAvail = true
			case "SwapTotal":
				swapTotal = valBytes
				haveSwapTotal = true
			case "SwapFree":
				swapFree = valBytes
				haveSwapFree = true
			}
		}

		if haveTotal && haveAvail {
			s.TotalRAMBytes = memTotal
			s.AvailRAMBytes = memAvail
			s.HaveRAM = true
		}
		if haveSwapTotal && haveSwapFree {
			s.TotalSwapBytes = swapTotal
			s.AvailSwapBytes = swapFree
			s.HaveSwap = true
		}
	}

	psiBytes, err := os.ReadFile("/proc/pressure/memory")
	if err == nil {
		scanner := bufio.NewScanner(bytes.NewReader(psiBytes))
		for scanner.Scan() {
			line := scanner.Text()
			fields := strings.Fields(line)
			if len(fields) == 0 {
				continue
			}
			prefix := fields[0]
			if prefix == "some" || prefix == "full" {
				for _, f := range fields[1:] {
					k, v, ok := strings.Cut(f, "=")
					if !ok {
						continue
					}
					switch k {
					case "avg10":
						if prefix == "some" {
							if fval, err := strconv.ParseFloat(v, 64); err == nil {
								s.PSIAvg10 = fval
								s.HavePSI = true
							}
						}
					case "avg60":
						if prefix == "some" {
							if fval, err := strconv.ParseFloat(v, 64); err == nil {
								s.PSIAvg60 = fval
							}
						}
					case "avg300":
						if prefix == "some" {
							if fval, err := strconv.ParseFloat(v, 64); err == nil {
								s.PSIAvg300 = fval
							}
						}
					case "total":
						if uval, err := strconv.ParseUint(v, 10, 64); err == nil {
							d := time.Duration(uval) * time.Microsecond
							if prefix == "some" {
								s.PSISomeTotal = d
								s.HavePSI = true
							} else if prefix == "full" {
								s.PSIFullTotal = d
								s.HavePSI = true
							}
						}
					}
				}
			}
		}
	}

	return s
}

// Package-level default governor for convenience and CLI flag resolution.
var (
	defaultGovernor     *DensityGovernor
	defaultGovernorOnce sync.Once
)

// DefaultGovernor returns the singleton density governor initialized with defaults
// and seeded with a live host reading.
func DefaultGovernor() *DensityGovernor {
	defaultGovernorOnce.Do(func() {
		defaultGovernor = NewDensityGovernor(DefaultGovernorConfig())
		sample := ReadHostSample()
		if sample.HaveRAM {
			defaultGovernor.Update(sample)
		}
	})
	return defaultGovernor
}

// ResolveConcurrency resolves a concurrency argument ("auto" -> dynamic limit,
// positive integer -> parsed value, invalid -> default conservative floor).
func ResolveConcurrency(concurrencyArg string) int {
	return DefaultGovernor().ResolveConcurrency(concurrencyArg)
}
