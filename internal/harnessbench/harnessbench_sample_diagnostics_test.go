//go:build fak_harnessbench_sample_diagnostics && race

package harnessbench

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/workspaceslot"
)

// This opt-in race-only diagnostic uses a source-pinned derived witness.
// Sample writes occur after the original measured interval and Release. They
// still perturb the surrounding workload; this is not an uninstrumented result.
type spawningInvariantSamples struct {
	EarlyNS    [100]int64 `json:"early_ns"`
	LateNS     [100]int64 `json:"late_ns"`
	EarlyCount int        `json:"early_count"`
	LateCount  int        `json:"late_count"`
}

func requireSpawningDiagnosticSourcePins(t *testing.T) {
	t.Helper()
	pins := []struct {
		path string
		sha  string
	}{
		{"harnessbench.go", "5083b527109eb660758817924f300f69aeae86f77385a755d096c7473297fcca"},
		{"harnessbench_test.go", "8463ca49b529641ee17bb1fe56ff4d04100efb0d90b716f1f030c7d8566119a7"},
		{"receipt.go", "76d321482105eafac86810f56fdfa47a9bbe4c3a1359fe5b89b06d1fccc8311a"},
		{"../workspaceslot/workspaceslot.go", "08475b5bd631320b8e187be8e01acd5ea0cc66fa8c9a35a8d2ab865f716f1e65"},
	}
	for _, pin := range pins {
		raw, err := os.ReadFile(pin.path)
		if err != nil {
			t.Fatalf("diagnostic source pin unavailable %s: %v", pin.path, err)
		}
		if got := fmt.Sprintf("%x", sha256.Sum256(raw)); got != pin.sha {
			t.Fatalf("diagnostic source pin mismatch %s: got %s, want %s", pin.path, got, pin.sha)
		}
	}
}

func checkSpawningDiagnosticSamples(samples *spawningInvariantSamples, receipt *SpawningInvariantReceipt) error {
	if samples == nil || receipt == nil {
		return fmt.Errorf("missing samples or receipt")
	}
	if samples.EarlyCount != len(samples.EarlyNS) || samples.LateCount != len(samples.LateNS) {
		return fmt.Errorf("sample counts: early=%d late=%d, want 100 each", samples.EarlyCount, samples.LateCount)
	}
	var earlyTotal, lateTotal int64
	for i := range samples.EarlyNS {
		earlyTotal += samples.EarlyNS[i]
		lateTotal += samples.LateNS[i]
	}
	if earlyTotal/100 != receipt.Epoch1LatencyNS || lateTotal/100 != receipt.EpochNLatencyNS {
		return fmt.Errorf("sample means disagree with receipt")
	}
	return nil
}

// fak-test:runtime fast est=10ms lane=optin
func TestSpawningInvariantDiagnosticSampleChecks(t *testing.T) {
	var samples spawningInvariantSamples
	samples.EarlyCount, samples.LateCount = 100, 100
	for i := range samples.EarlyNS {
		samples.EarlyNS[i], samples.LateNS[i] = 1000, 2000
	}
	receipt := SpawningInvariantReceipt{Epoch1LatencyNS: 1000, EpochNLatencyNS: 2000}
	if err := checkSpawningDiagnosticSamples(&samples, &receipt); err != nil {
		t.Fatal(err)
	}
	for _, which := range []string{"early_count", "late_count", "early_mean", "late_mean", "nil_samples", "nil_receipt"} {
		t.Run(which, func(t *testing.T) {
			copySamples, copyReceipt := samples, receipt
			s, r := &copySamples, &copyReceipt
			switch which {
			case "early_count":
				s.EarlyCount--
			case "late_count":
				s.LateCount--
			case "early_mean":
				s.EarlyNS[0] += 100
			case "late_mean":
				s.LateNS[0] += 100
			case "nil_samples":
				s = nil
			case "nil_receipt":
				r = nil
			}
			if err := checkSpawningDiagnosticSamples(s, r); err == nil {
				t.Fatalf("%s: inconsistent diagnostic samples accepted", which)
			}
		})
	}
}

// fak-test:runtime integration est=5s lane=optin
func TestSpawningInvariantSampleDiagnostics(t *testing.T) {
	requireSpawningDiagnosticSourcePins(t)
	cycles := 5000
	if !testing.Short() {
		cycles = 10000
	}
	goVersion := runtime.Version()
	procsBefore := runtime.GOMAXPROCS(0)
	var samples spawningInvariantSamples
	receipt, err := runSpawningInvariantSampleDiagnostics(cycles, t.TempDir(), &samples)
	if err != nil {
		t.Fatalf("spawning diagnostic witness: %v", err)
	}
	if err := checkSpawningDiagnosticSamples(&samples, receipt); err != nil {
		t.Fatalf("invalid spawning diagnostics: %v", err)
	}
	latencyPass := receipt.LatencyRatio <= 1.05 || (receipt.EpochNLatencyNS-receipt.Epoch1LatencyNS) < 500
	memoryPass := receipt.AllocDriftBytes < 5*1024*1024
	diagnostic := struct {
		Schema       string                    `json:"schema"`
		Instrumented bool                      `json:"instrumented"`
		RaceBuild    bool                      `json:"race_build"`
		GoVersion    string                    `json:"go_version"`
		GOOS         string                    `json:"goos"`
		GOARCH       string                    `json:"goarch"`
		ProcsBefore  int                       `json:"gomaxprocs_before"`
		ProcsAfter   int                       `json:"gomaxprocs_after"`
		EarlyStart   int                       `json:"early_start_zero_based"`
		EarlyEnd     int                       `json:"early_end_exclusive"`
		LateStart    int                       `json:"late_start_zero_based"`
		LateEnd      int                       `json:"late_end_exclusive"`
		LatencyPass  bool                      `json:"latency_pass"`
		MemoryPass   bool                      `json:"memory_pass"`
		Samples      spawningInvariantSamples  `json:"samples"`
		Receipt      *SpawningInvariantReceipt `json:"receipt"`
	}{
		Schema: "fak.harnessbench.sample_diagnostics.v1", Instrumented: true, RaceBuild: true,
		GoVersion: goVersion, GOOS: runtime.GOOS, GOARCH: runtime.GOARCH,
		ProcsBefore: procsBefore, ProcsAfter: runtime.GOMAXPROCS(0),
		EarlyStart: 200, EarlyEnd: 300, LateStart: receipt.TotalCycles - 100, LateEnd: receipt.TotalCycles,
		LatencyPass: latencyPass, MemoryPass: memoryPass, Samples: samples, Receipt: receipt,
	}
	raw, err := json.Marshal(diagnostic)
	if err != nil {
		t.Fatalf("marshal spawning sample diagnostics: %v", err)
	}
	t.Logf("Spawning Sample Diagnostics: %s", raw)
	if receipt.Pass != (latencyPass && memoryPass) {
		t.Fatal("invalid spawning diagnostics: predicate disagrees with receipt")
	}
	if !receipt.Pass {
		t.Fatalf("spawning invariant witness failed: %+v", receipt)
	}
	if receipt.Schema != SpawningInvariantSchema {
		t.Errorf("schema mismatch: %s", receipt.Schema)
	}
}

// The following function is derived from the pinned production function with
// exactly three substitutions: its name/signature and two sample-recording
// additions. The timed expression, workload, GC calls and predicate are exact.
func runSpawningInvariantSampleDiagnostics(cycles int, baseDir string, samples *spawningInvariantSamples) (*SpawningInvariantReceipt, error) {
	if cycles < 2000 {
		cycles = 2000
	}
	if baseDir == "" {
		tmp, err := os.MkdirTemp("", "fak-harnessbench-slots-*")
		if err != nil {
			return nil, err
		}
		defer os.RemoveAll(tmp)
		baseDir = tmp
	}

	ring, err := workspaceslot.NewSlotRing(baseDir, 8)
	if err != nil {
		return nil, fmt.Errorf("harnessbench: init slot ring: %w", err)
	}
	defer ring.Close()

	ctx := context.Background()
	var epoch1TotalNS int64
	var epochNTotalNS int64
	const sampleCount = 100
	const warmupCycles = 200

	var allocAt1000 uint64
	var allocAtEnd uint64

	for i := 0; i < cycles; i++ {
		t0 := time.Now()
		slot, err := ring.Acquire(ctx, fmt.Sprintf("agent-%d", i))
		launchElapsed := time.Since(t0).Nanoseconds()
		if err != nil {
			return nil, fmt.Errorf("cycle %d acquire: %w", i, err)
		}

		// Simulate step writing scratch data
		scratchFile := filepath.Join(slot.ScratchDir, "step.log")
		_ = os.WriteFile(scratchFile, []byte("step completed"), 0644)

		if err := ring.Release(slot); err != nil {
			return nil, fmt.Errorf("cycle %d release: %w", i, err)
		}

		if i >= warmupCycles && i < warmupCycles+sampleCount {
			epoch1TotalNS += launchElapsed
			samples.EarlyNS[i-warmupCycles] = launchElapsed
			samples.EarlyCount++
		} else if i >= cycles-sampleCount {
			epochNTotalNS += launchElapsed
			samples.LateNS[i-(cycles-sampleCount)] = launchElapsed
			samples.LateCount++
		}

		if i == 1000 {
			runtime.GC()
			var m runtime.MemStats
			runtime.ReadMemStats(&m)
			allocAt1000 = m.Alloc
		}
	}

	runtime.GC()
	var mFinal runtime.MemStats
	runtime.ReadMemStats(&mFinal)
	allocAtEnd = mFinal.Alloc

	avgEpoch1 := epoch1TotalNS / sampleCount
	avgEpochN := epochNTotalNS / sampleCount
	ratio := float64(avgEpochN) / float64(avgEpoch1)
	if avgEpoch1 == 0 {
		ratio = 1.0
	}

	drift := int64(allocAtEnd) - int64(allocAt1000)

	// Invariant: O(1) launch latency - ratio <= 1.05 (allowing for sub-microsecond timer noise floor within 500ns)
	passLatency := ratio <= 1.05 || (avgEpochN-avgEpoch1) < 500
	// Invariant: zero net memory leak (< 5MB heap growth after GC across cycles)
	passMemory := drift < 5*1024*1024

	return &SpawningInvariantReceipt{
		Schema:          SpawningInvariantSchema,
		TotalCycles:     cycles,
		Epoch1LatencyNS: avgEpoch1,
		EpochNLatencyNS: avgEpochN,
		LatencyRatio:    ratio,
		InitialAllocB:   allocAt1000,
		FinalAllocB:     allocAtEnd,
		AllocDriftBytes: drift,
		Pass:            passLatency && passMemory,
		CompletedAt:     time.Now().UTC(),
	}, nil
}
