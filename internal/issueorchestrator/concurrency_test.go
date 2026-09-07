package issueorchestrator

import (
	"strings"
	"testing"
	"time"
)

func TestAdaptiveConcurrency(t *testing.T) {
	opts := WavePlanOptions{WaveSize: 8}

	// 1. Nominal 16 CPU, healthy RAM, low lock latency -> floor(16/2) = 8, capped by WaveSize=8
	res1 := SystemResourceInfo{
		NumCPU:       16,
		RAMAvailable: 0.50,
		CPULoad:      0.20,
		LockLatency:  5 * time.Millisecond,
	}
	if w := AdaptiveWaveSizeWithResources(opts, res1); w != 8 {
		t.Errorf("expected wave size 8 for 16 CPU, got %d", w)
	}

	// 2. Small host: 2 CPU -> floor(2/2) = 1
	res2 := SystemResourceInfo{
		NumCPU:       2,
		RAMAvailable: 0.50,
		CPULoad:      0.20,
		LockLatency:  5 * time.Millisecond,
	}
	if w := AdaptiveWaveSizeWithResources(opts, res2); w != 1 {
		t.Errorf("expected wave size 1 for 2 CPU, got %d", w)
	}

	// 3. Low RAM pressure (< 20%) -> 16 CPU nominal 8 backs off to 4
	res3 := SystemResourceInfo{
		NumCPU:       16,
		RAMAvailable: 0.15,
		CPULoad:      0.20,
		LockLatency:  5 * time.Millisecond,
	}
	if w := AdaptiveWaveSizeWithResources(opts, res3); w != 4 {
		t.Errorf("expected backed-off wave size 4 under RAM pressure, got %d", w)
	}

	// 4. Database contention (lock latency > 50ms) -> backs off to 4
	res4 := SystemResourceInfo{
		NumCPU:       16,
		RAMAvailable: 0.50,
		CPULoad:      0.20,
		LockLatency:  75 * time.Millisecond,
	}
	if w := AdaptiveWaveSizeWithResources(opts, res4); w != 4 {
		t.Errorf("expected backed-off wave size 4 under lock latency, got %d", w)
	}

	// 5. Jitter calculation without contention (500ms - 2500ms)
	var warnings []string
	logWarn := func(s string) {
		warnings = append(warnings, s)
	}
	delay := CalculateSpawnDelay(res1, logWarn)
	if delay < 500*time.Millisecond || delay > 2500*time.Millisecond {
		t.Errorf("expected delay between 500ms and 2500ms, got %v", delay)
	}
	if len(warnings) != 0 {
		t.Errorf("expected no warnings on healthy resources, got %v", warnings)
	}

	// 6. Soft advisory throttling on SQLite contention > 100ms (+1.5s)
	resContention := SystemResourceInfo{
		NumCPU:       16,
		RAMAvailable: 0.50,
		CPULoad:      0.20,
		LockLatency:  120 * time.Millisecond,
	}
	warnings = nil
	delayContended := CalculateSpawnDelay(resContention, logWarn)
	if delayContended < 2000*time.Millisecond || delayContended > 4000*time.Millisecond {
		t.Errorf("expected delay between 2000ms and 4000ms with +1.5s throttling, got %v", delayContended)
	}
	if len(warnings) == 0 || !strings.Contains(warnings[0], "SQLite lock contention detected") {
		t.Errorf("expected SQLite lock contention warning, got %v", warnings)
	}
}
