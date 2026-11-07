package fleet

import (
	"context"
	"errors"
	"os/exec"
	"testing"
)

// fak-test:runtime fast est=5ms lane=default
func TestParseGPUUtilizationCSV(t *testing.T) {
	// 12,45,0,88: busy = 12,45,88 (>= threshold 5); mean = 36.25 -> 36.
	stats, ok := ParseGPUUtilizationCSV("12\n45\n0\n88\n")
	if !ok {
		t.Fatal("valid multi-GPU CSV must parse")
	}
	if stats.Total != 4 || stats.Busy != 3 || stats.UtilPct != 36 {
		t.Fatalf("stats = %+v, want {Total:4 Busy:3 UtilPct:36}", stats)
	}
	// CRLF endings (a Windows nvidia-smi) and trailing blanks are accepted.
	crlf, ok := ParseGPUUtilizationCSV("10\r\n20\r\n\r\n")
	if !ok || crlf.Total != 2 || crlf.Busy != 2 || crlf.UtilPct != 15 {
		t.Fatalf("crlf stats = %+v ok=%v, want {2 2 15}", crlf, ok)
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestParseGPUUtilizationCSVThresholdBoundary(t *testing.T) {
	// Exactly at the threshold counts as busy; one below does not.
	stats, ok := ParseGPUUtilizationCSV("4\n5\n")
	if !ok || stats.Total != 2 || stats.Busy != 1 || stats.UtilPct != 5 {
		t.Fatalf("boundary stats = %+v ok=%v, want {Total:2 Busy:1 UtilPct:5}", stats, ok)
	}
	// A stricter threshold via the internal knob: only 90 is busy now.
	strict, ok := parseGPUUtilizationCSV("4\n5\n90\n", 50)
	if !ok || strict.Total != 3 || strict.Busy != 1 || strict.UtilPct != 33 { // mean of 4,5,90
		t.Fatalf("strict stats = %+v ok=%v, want {Total:3 Busy:1 UtilPct:33}", strict, ok)
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestParseGPUUtilizationCSVRejectsGarbage(t *testing.T) {
	// No valid row: an NVML failure or a missing driver prints text, never numbers.
	for _, out := range []string{
		"",
		"\n",
		"Failed to initialize NVML: Not Found\n",
		"WARNING: infoROM is corrupted\nNVIDIA-SMI has failed\n",
		"-7\n101\n", // out of range on both sides
	} {
		if _, ok := ParseGPUUtilizationCSV(out); ok {
			t.Fatalf("garbage output %q must not parse", out)
		}
	}
	// Mixed: warning lines are skipped, the valid rows still fold.
	stats, ok := ParseGPUUtilizationCSV("WARNING: something\n50\nbad\n")
	if !ok || stats.Total != 1 || stats.Busy != 1 || stats.UtilPct != 50 {
		t.Fatalf("mixed stats = %+v ok=%v, want {1 1 50}", stats, ok)
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestProbeGPUUtilizationStubsRunner(t *testing.T) {
	orig := gpuProbeRunner
	defer func() { gpuProbeRunner = orig }()

	var sawName string
	var sawArgs []string
	gpuProbeRunner = func(ctx context.Context, name string, args ...string) ([]byte, error) {
		sawName, sawArgs = name, args
		return []byte("10\n90\n"), nil
	}
	stats, err := ProbeGPUUtilization(context.Background())
	if err != nil {
		t.Fatalf("stubbed probe failed: %v", err)
	}
	if stats.Total != 2 || stats.Busy != 2 || stats.UtilPct != 50 {
		t.Fatalf("probe stats = %+v, want {2 2 50}", stats)
	}
	if sawName != "nvidia-smi" || len(sawArgs) != 2 || sawArgs[0] != "--query-gpu=utilization.gpu" || sawArgs[1] != "--format=csv,noheader,nounits" {
		t.Fatalf("probe invoked %q %v, want the utilization query", sawName, sawArgs)
	}

	// Missing binary: typed sentinel, no panic, no fabricated reading.
	gpuProbeRunner = func(ctx context.Context, name string, args ...string) ([]byte, error) {
		return nil, &exec.Error{Name: "nvidia-smi", Err: exec.ErrNotFound}
	}
	if _, err := ProbeGPUUtilization(context.Background()); !errors.Is(err, ErrNoGPUSignal) {
		t.Fatalf("missing nvidia-smi should wrap ErrNoGPUSignal, got %v", err)
	}

	// Unparseable stdout: same sentinel.
	gpuProbeRunner = func(ctx context.Context, name string, args ...string) ([]byte, error) {
		return []byte("Driver version mismatch\n"), nil
	}
	if _, err := ProbeGPUUtilization(context.Background()); !errors.Is(err, ErrNoGPUSignal) {
		t.Fatalf("unparseable output should wrap ErrNoGPUSignal, got %v", err)
	}
}
