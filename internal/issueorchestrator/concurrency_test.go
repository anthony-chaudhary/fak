package issueorchestrator

import (
	"errors"
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
}

func TestAdaptiveConcurrency_WaveSizeScaling(t *testing.T) {
	// 1. Scaling with CPU count
	opts8 := ConcurrencyOptions{
		WaveSize:     8,
		NumCPU:       16,
		RAMAvailable: 0.60,
		LockLatency:  5 * time.Millisecond,
	}
	if w := AdaptiveWaveSize(opts8); w != 8 {
		t.Errorf("expected wave size 8, got %d", w)
	}

	optsSmall := ConcurrencyOptions{
		WaveSize:     8,
		NumCPU:       2,
		RAMAvailable: 0.60,
		LockLatency:  5 * time.Millisecond,
	}
	if w := AdaptiveWaveSize(optsSmall); w != 1 {
		t.Errorf("expected wave size 1 for 2 CPUs, got %d", w)
	}

	// 2. Scaling with RAM pressure (< 20%)
	optsLowRAM := ConcurrencyOptions{
		WaveSize:     8,
		NumCPU:       16,
		RAMAvailable: 0.10, // 10% available
		LockLatency:  5 * time.Millisecond,
	}
	if w := AdaptiveWaveSize(optsLowRAM); w != 4 {
		t.Errorf("expected wave size 4 under low RAM, got %d", w)
	}

	// 3. Scaling with Lock Latency (> 50ms)
	optsContended := ConcurrencyOptions{
		WaveSize:     8,
		NumCPU:       16,
		RAMAvailable: 0.60,
		LockLatency:  80 * time.Millisecond,
	}
	if w := AdaptiveWaveSize(optsContended); w != 4 {
		t.Errorf("expected wave size 4 under lock contention, got %d", w)
	}

	// 4. Test with custom probe hook
	optsProbe := ConcurrencyOptions{
		WaveSize:     8,
		NumCPU:       16,
		RAMAvailable: 0.60,
		Probe: func(dbPath string) (time.Duration, error) {
			return 65 * time.Millisecond, nil
		},
	}
	if w := AdaptiveWaveSize(optsProbe); w != 4 {
		t.Errorf("expected wave size 4 with probe returning 65ms, got %d", w)
	}

	// 5. WavePlanOptions compatibility
	wPlan := AdaptiveWaveSize(WavePlanOptions{WaveSize: 8})
	if wPlan < 1 {
		t.Errorf("expected positive wave size from WavePlanOptions, got %d", wPlan)
	}
}

func TestAdaptiveConcurrency_SpawnJitterBounds(t *testing.T) {
	const iterations = 200
	var minObserved = time.Duration(1<<63 - 1)
	var maxObserved time.Duration

	for i := 0; i < iterations; i++ {
		d := CalculateSpawnJitter(500, 2500)
		if d < 500*time.Millisecond || d > 2500*time.Millisecond {
			t.Fatalf("jitter %v out of bounds [500ms, 2500ms]", d)
		}
		if d < minObserved {
			minObserved = d
		}
		if d > maxObserved {
			maxObserved = d
		}
	}

	// Verify randomness produces variance
	if minObserved == maxObserved {
		t.Errorf("expected jitter variation over %d iterations, got constant %v", iterations, minObserved)
	}

	// Verify default fallback when passing zeros
	for i := 0; i < 50; i++ {
		d := CalculateSpawnJitter(0, 0)
		if d < 500*time.Millisecond || d > 2500*time.Millisecond {
			t.Fatalf("default jitter %v out of bounds [500ms, 2500ms]", d)
		}
	}
}

func TestAdaptiveConcurrency_ContentionThrottling(t *testing.T) {
	var warnings []string
	logWarn := func(s string) {
		warnings = append(warnings, s)
	}

	// 1. Healthy resources: no throttling, no warnings
	healthyRes := SystemResourceInfo{
		NumCPU:       16,
		RAMAvailable: 0.50,
		CPULoad:      0.20,
		LockLatency:  5 * time.Millisecond,
	}
	warnings = nil
	delay := CalculateSpawnDelay(healthyRes, logWarn)
	if delay < 500*time.Millisecond || delay > 2500*time.Millisecond {
		t.Errorf("expected delay between 500ms and 2500ms, got %v", delay)
	}
	if len(warnings) != 0 {
		t.Errorf("expected no warnings on healthy resources, got %v", warnings)
	}

	// 2. SQLite lock contention > 100ms triggers +1.5s throttling and log warning
	contendedRes := SystemResourceInfo{
		NumCPU:       16,
		RAMAvailable: 0.50,
		CPULoad:      0.20,
		LockLatency:  120 * time.Millisecond,
	}
	warnings = nil
	contendedDelay := CalculateSpawnDelay(contendedRes, logWarn)
	if contendedDelay < 2000*time.Millisecond || contendedDelay > 4000*time.Millisecond {
		t.Errorf("expected delay between 2000ms and 4000ms with +1.5s throttling, got %v", contendedDelay)
	}
	if len(warnings) != 1 || !strings.Contains(warnings[0], "INFO [orchestrator]: SQLite lock contention detected; soft-throttling spawn delay by +1.5s") {
		t.Errorf("expected SQLite lock contention warning, got %v", warnings)
	}

	// 3. High CPU pressure > 85% triggers +1.5s throttling
	highCPURes := SystemResourceInfo{
		NumCPU:       16,
		RAMAvailable: 0.50,
		CPULoad:      0.90,
		LockLatency:  5 * time.Millisecond,
	}
	warnings = nil
	highCPUDelay := CalculateSpawnDelay(highCPURes, logWarn)
	if highCPUDelay < 2000*time.Millisecond || highCPUDelay > 4000*time.Millisecond {
		t.Errorf("expected delay between 2000ms and 4000ms with +1.5s high CPU throttling, got %v", highCPUDelay)
	}
	if len(warnings) != 1 || !strings.Contains(warnings[0], "INFO [orchestrator]: High CPU pressure (>85%) detected; soft-throttling spawn delay by +1.5s") {
		t.Errorf("expected high CPU pressure warning, got %v", warnings)
	}

	// 4. Test CalculateSpawnDelayWithOptions
	warnings = nil
	optDelay := CalculateSpawnDelayWithOptions(ConcurrencyOptions{
		LockLatency: 150 * time.Millisecond,
	}, logWarn)
	if optDelay < 2000*time.Millisecond || optDelay > 4000*time.Millisecond {
		t.Errorf("expected delay between 2000ms and 4000ms via ConcurrencyOptions, got %v", optDelay)
	}
	if len(warnings) == 0 {
		t.Errorf("expected contention warning via ConcurrencyOptions")
	}
}

func TestSQLiteProbe_NonBlocking(t *testing.T) {
	// 1. Non-existent file: returns 0 latency, nil error (fail-open)
	start := time.Now()
	lat, err := ProbeSQLiteLockLatency("non_existent_file_path_for_testing_12345.db")
	elapsed := time.Since(start)
	if lat != 0 || err != nil {
		t.Errorf("expected 0 latency and nil error for non-existent file, got lat=%v err=%v", lat, err)
	}
	if elapsed > 100*time.Millisecond {
		t.Errorf("probe took too long (%v), expected non-blocking fast return", elapsed)
	}

	// 2. Mockable probe hook: inject deterministic latency and error
	cleanup := SetSQLiteLockProbe(func(dbPath string) (time.Duration, error) {
		return 35 * time.Millisecond, nil
	})
	mockLat, mockErr := ProbeSQLiteLockLatency("any_path.db")
	if mockLat != 35*time.Millisecond || mockErr != nil {
		t.Errorf("expected mock probe 35ms and nil error, got lat=%v err=%v", mockLat, mockErr)
	}

	detailed := ProbeSQLiteLockLatencyDetailed("any_path.db")
	if detailed.Latency != 35*time.Millisecond || detailed.Contended {
		t.Errorf("expected detailed probe latency 35ms not contended, got %+v", detailed)
	}
	cleanup()

	// 3. Mockable probe with contention & error
	cleanup2 := SetSQLiteLockProbe(func(dbPath string) (time.Duration, error) {
		return 120 * time.Millisecond, errors.New("database is locked")
	})
	contendedLat, contendedErr := ProbeSQLiteLockLatency("contended.db")
	if contendedLat != 120*time.Millisecond || contendedErr == nil || contendedErr.Error() != "database is locked" {
		t.Errorf("expected 120ms and locked error, got lat=%v err=%v", contendedLat, contendedErr)
	}

	detailedContended := ProbeSQLiteLockLatencyDetailed("contended.db")
	if detailedContended.Latency != 120*time.Millisecond || !detailedContended.Contended || detailedContended.Error != "database is locked" {
		t.Errorf("expected detailed probe latency 120ms contended, got %+v", detailedContended)
	}
	cleanup2()

	// 4. Verify original probe restored after cleanup
	restoredLat, restoredErr := ProbeSQLiteLockLatency("non_existent_file_path_for_testing_12345.db")
	if restoredLat != 0 || restoredErr != nil {
		t.Errorf("expected restored probe returning 0 and nil error, got lat=%v err=%v", restoredLat, restoredErr)
	}
}
