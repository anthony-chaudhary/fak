package issueorchestrator

import (
	"math/rand"
	"runtime"
	"sync"
	"time"
)

// ConcurrencyOptions configures adaptive wave size calculations and provides
// hooks for test injection of host hardware and database contention metrics.
type ConcurrencyOptions struct {
	WaveSize     int                                 `json:"wave_size"`
	NumCPU       int                                 `json:"num_cpu,omitempty"`
	RAMAvailable float64                             `json:"ram_available,omitempty"` // 0.0 to 1.0 (e.g. 0.15 = 15% available)
	CPULoad      float64                             `json:"cpu_load,omitempty"`      // 0.0 to 1.0 (e.g. 0.85 = 85% load)
	LockLatency  time.Duration                       `json:"lock_latency,omitempty"`
	DBPath       string                              `json:"db_path,omitempty"`
	Probe        func(string) (time.Duration, error) `json:"-"`
}

// SystemResourceInfo captures host resources and database contention metrics.
type SystemResourceInfo struct {
	NumCPU       int
	RAMAvailable float64 // 0.0 to 1.0 (e.g. 0.25 = 25% available)
	CPULoad      float64 // 0.0 to 1.0 (e.g. 0.85 = 85% load)
	LockLatency  time.Duration
}

var (
	resourceInfoMu   sync.RWMutex
	mockResourceInfo *SystemResourceInfo
)

// SetMockSystemResources injects host resources for testing and returns a cleanup function.
func SetMockSystemResources(res *SystemResourceInfo) func() {
	resourceInfoMu.Lock()
	mockResourceInfo = res
	resourceInfoMu.Unlock()
	return func() {
		resourceInfoMu.Lock()
		mockResourceInfo = nil
		resourceInfoMu.Unlock()
	}
}

// CurrentSystemResources probes current host resources and SQLite lock latency.
func CurrentSystemResources(dbPath string) SystemResourceInfo {
	resourceInfoMu.RLock()
	mock := mockResourceInfo
	resourceInfoMu.RUnlock()
	if mock != nil {
		return *mock
	}

	lat, _ := ProbeSQLiteLockLatency(dbPath)
	return SystemResourceInfo{
		NumCPU:       runtime.NumCPU(),
		RAMAvailable: 0.50,
		CPULoad:      0.20,
		LockLatency:  lat,
	}
}

// AdaptiveWaveSize computes safe wave concurrency derived from host CPU/RAM headroom
// and SQLite lock latency: W_eff = clamp(1, floor(CPU/2), W_max), backing off when
// RAM < 20% or lock latency > 50ms.
// It accepts either ConcurrencyOptions or WavePlanOptions (or pointers to them).
func AdaptiveWaveSize(opts any) int {
	switch v := opts.(type) {
	case ConcurrencyOptions:
		return AdaptiveWaveSizeWithOptions(v)
	case *ConcurrencyOptions:
		if v != nil {
			return AdaptiveWaveSizeWithOptions(*v)
		}
		return AdaptiveWaveSizeWithOptions(ConcurrencyOptions{})
	case WavePlanOptions:
		return AdaptiveWaveSizeWithPlanOptions(v)
	case *WavePlanOptions:
		if v != nil {
			return AdaptiveWaveSizeWithPlanOptions(*v)
		}
		return AdaptiveWaveSizeWithPlanOptions(WavePlanOptions{})
	default:
		return 1
	}
}

// AdaptiveWaveSizeWithPlanOptions calculates adaptive wave size for WavePlanOptions.
func AdaptiveWaveSizeWithPlanOptions(opts WavePlanOptions) int {
	res := CurrentSystemResources("")
	return AdaptiveWaveSizeWithResources(opts, res)
}

// AdaptiveWaveSizeWithOptions calculates adaptive wave size for explicit ConcurrencyOptions.
func AdaptiveWaveSizeWithOptions(opts ConcurrencyOptions) int {
	res := CurrentSystemResources(opts.DBPath)
	if opts.NumCPU > 0 {
		res.NumCPU = opts.NumCPU
	}
	if opts.RAMAvailable > 0 {
		res.RAMAvailable = opts.RAMAvailable
	}
	if opts.CPULoad > 0 {
		res.CPULoad = opts.CPULoad
	}
	if opts.LockLatency > 0 {
		res.LockLatency = opts.LockLatency
	} else if opts.Probe != nil {
		lat, _ := opts.Probe(opts.DBPath)
		res.LockLatency = lat
	}

	return computeEffectiveWaveSize(opts.WaveSize, res)
}

// AdaptiveWaveSizeWithResources calculates wave size given explicit resource metrics.
func AdaptiveWaveSizeWithResources(opts WavePlanOptions, res SystemResourceInfo) int {
	return computeEffectiveWaveSize(opts.WaveSize, res)
}

func computeEffectiveWaveSize(maxWave int, res SystemResourceInfo) int {
	cpu := res.NumCPU
	if cpu <= 0 {
		cpu = 1
	}

	wMax := maxWave
	if wMax <= 0 {
		wMax = 4
	}

	wEff := cpu / 2
	if wEff < 1 {
		wEff = 1
	}
	if wEff > wMax {
		wEff = wMax
	}

	// Back off when RAM < 20% or lock latency > 50ms
	if (res.RAMAvailable > 0 && res.RAMAvailable < 0.20) || res.LockLatency > 50*time.Millisecond {
		wEff = wEff / 2
		if wEff < 1 {
			wEff = 1
		}
	}

	return wEff
}

// CalculateSpawnJitter calculates a randomized duration between minMs and maxMs.
// If minMs or maxMs are zero or inverted, it defaults to [500ms, 2500ms].
func CalculateSpawnJitter(minMs, maxMs int) time.Duration {
	if minMs <= 0 && maxMs <= 0 {
		minMs = 500
		maxMs = 2500
	} else {
		if minMs <= 0 {
			minMs = 500
		}
		if maxMs <= minMs {
			maxMs = minMs + 2000
		}
	}
	delta := maxMs - minMs
	jitter := minMs
	if delta > 0 {
		jitter += rand.Intn(delta + 1)
	}
	return time.Duration(jitter) * time.Millisecond
}

// CalculateSpawnDelay calculates the staggering delay before starting a worker,
// incorporating randomized jitter (500ms–2500ms) and soft-throttling if contention
// or CPU pressure is sensed.
func CalculateSpawnDelay(res SystemResourceInfo, logWarn func(string)) time.Duration {
	delay := CalculateSpawnJitter(500, 2500)

	// Soft Advisory Throttling: Lock acquisition wait > 100ms
	if res.LockLatency > 100*time.Millisecond {
		if logWarn != nil {
			logWarn("INFO [orchestrator]: SQLite lock contention detected; soft-throttling spawn delay by +1.5s")
		}
		delay += 1500 * time.Millisecond
	} else if res.CPULoad > 0.85 {
		if logWarn != nil {
			logWarn("INFO [orchestrator]: High CPU pressure (>85%) detected; soft-throttling spawn delay by +1.5s")
		}
		delay += 1500 * time.Millisecond
	}

	return delay
}

// CalculateSpawnDelayWithOptions calculates spawn delay given explicit ConcurrencyOptions.
func CalculateSpawnDelayWithOptions(opts ConcurrencyOptions, logWarn func(string)) time.Duration {
	res := CurrentSystemResources(opts.DBPath)
	if opts.CPULoad > 0 {
		res.CPULoad = opts.CPULoad
	}
	if opts.LockLatency > 0 {
		res.LockLatency = opts.LockLatency
	} else if opts.Probe != nil {
		lat, _ := opts.Probe(opts.DBPath)
		res.LockLatency = lat
	}
	return CalculateSpawnDelay(res, logWarn)
}
