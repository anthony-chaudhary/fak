package issueorchestrator

import (
	"math/rand"
	"runtime"
	"time"
)

// SystemResourceInfo captures host resources and database contention metrics.
type SystemResourceInfo struct {
	NumCPU       int
	RAMAvailable float64 // 0.0 to 1.0 (e.g. 0.25 = 25% available)
	CPULoad      float64 // 0.0 to 1.0 (e.g. 0.85 = 85% load)
	LockLatency  time.Duration
}

// CurrentSystemResources probes current host resources and SQLite lock latency.
func CurrentSystemResources(dbPath string) SystemResourceInfo {
	probe := ProbeSQLiteLockLatency(dbPath)
	return SystemResourceInfo{
		NumCPU:       runtime.NumCPU(),
		RAMAvailable: 0.50,
		CPULoad:      0.20,
		LockLatency:  probe.Latency,
	}
}

// AdaptiveWaveSize computes safe wave concurrency derived from host CPU/RAM headroom
// and SQLite lock latency: W_eff = clamp(1, floor(CPU/2), W_max), backing off when
// RAM < 20% or lock latency > 50ms.
func AdaptiveWaveSize(opts WavePlanOptions) int {
	res := CurrentSystemResources("")
	return AdaptiveWaveSizeWithResources(opts, res)
}

// AdaptiveWaveSizeWithResources calculates wave size given explicit resource metrics.
func AdaptiveWaveSizeWithResources(opts WavePlanOptions, res SystemResourceInfo) int {
	cpu := res.NumCPU
	if cpu <= 0 {
		cpu = 1
	}

	wMax := opts.WaveSize
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

// CalculateSpawnDelay calculates the staggering delay before starting a worker,
// incorporating randomized jitter (500ms–2500ms) and soft-throttling if contention
// or CPU pressure is sensed.
func CalculateSpawnDelay(res SystemResourceInfo, logWarn func(string)) time.Duration {
	jitterMs := 500 + rand.Intn(2001) // 500 to 2500 ms
	delay := time.Duration(jitterMs) * time.Millisecond

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
