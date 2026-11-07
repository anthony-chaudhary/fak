package main

import (
	"bytes"
	"encoding/json"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/fleet"
)

// readLabReportGPU decodes the gpu block of a self-report file written by runLab.
func readLabReportGPU(t *testing.T, dir, id string) *fleet.GPUStats {
	t.Helper()
	data, err := os.ReadFile(filepath.Join(dir, id+".json"))
	if err != nil {
		t.Fatalf("read report: %v", err)
	}
	var rep fleet.Report
	if err := json.Unmarshal(data, &rep); err != nil {
		t.Fatalf("report is not JSON: %v\n%s", err, data)
	}
	return rep.GPU
}

// fak-test:runtime fast est=10ms lane=default
func TestLabReportExplicitGPUBlock(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("FAK_FLEET_REPORTS", dir)

	if rc := runLab(io.Discard, io.Discard, []string{
		"report", "--id", "da-cpu", "--state", "live",
		"--gpu-total", "8", "--gpu-busy", "3", "--gpu-util", "41",
	}); rc != 0 {
		t.Fatalf("lab report with explicit gpu flags exited %d, want 0", rc)
	}
	gpu := readLabReportGPU(t, dir, "da-cpu")
	if gpu == nil || gpu.Total != 8 || gpu.Busy != 3 || gpu.UtilPct != 41 {
		t.Fatalf("gpu block = %+v, want {8 3 41}", gpu)
	}

	// No GPU signal at all -> no gpu block in the file (unknown-utilization).
	if rc := runLab(io.Discard, io.Discard, []string{"report", "--id", "box-a", "--state", "idle"}); rc != 0 {
		t.Fatalf("plain lab report exited %d, want 0", rc)
	}
	if gpu := readLabReportGPU(t, dir, "box-a"); gpu != nil {
		t.Fatalf("no gpu flags must leave the block absent, got %+v", gpu)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestLabReportGPUProbeFlagIsFailSoft(t *testing.T) {
	if _, err := exec.LookPath("nvidia-smi"); err == nil {
		t.Skip("host has nvidia-smi; the fail-soft path cannot be witnessed here")
	}
	// --gpu-probe shells out to nvidia-smi; on a box without one it must stay
	// fail-soft: exit 0, state reported, gpu block omitted, cause on stderr.
	dir := t.TempDir()
	t.Setenv("FAK_FLEET_REPORTS", dir)
	var stderr bytes.Buffer
	rc := runLab(io.Discard, &stderr, []string{"report", "--id", "da-cpu", "--state", "live", "--gpu-probe"})
	if rc != 0 {
		t.Fatalf("--gpu-probe must be fail-soft (exit 0), got %d: %s", rc, stderr.String())
	}
	if !strings.Contains(stderr.String(), "gpu probe:") {
		t.Fatalf("probe failure should be named on stderr, got:\n%s", stderr.String())
	}
	if gpu := readLabReportGPU(t, dir, "da-cpu"); gpu != nil {
		t.Fatalf("failed probe must omit the gpu block, got %+v", gpu)
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestMergeGPUFlagsOverrideSemantics(t *testing.T) {
	probed := &fleet.GPUStats{Total: 4, Busy: 2, UtilPct: 50}
	// Probe only: every field comes from the probe.
	if g := mergeGPUFlags(probed, 0, 0, 0, false, false, false); g == nil || *g != *probed {
		t.Fatalf("probe-only merge = %+v, want the probe reading", g)
	}
	// Explicit values win per field over the probe; unset fields keep the probe's.
	if g := mergeGPUFlags(probed, 8, 1, 0, true, true, false); g == nil || g.Total != 8 || g.Busy != 1 || g.UtilPct != 50 {
		t.Fatalf("override merge = %+v, want {Total:8 Busy:1 UtilPct:50}", g)
	}
	// Explicit-only (no probe): a partial block is still a GPU signal.
	if g := mergeGPUFlags(nil, 0, 3, 41, false, true, true); g == nil || g.Total != 0 || g.Busy != 3 || g.UtilPct != 41 {
		t.Fatalf("explicit-only merge = %+v, want {0 3 41}", g)
	}
	// No signal at all: nil, so the report carries no gpu block.
	if g := mergeGPUFlags(nil, 0, 0, 0, false, false, false); g != nil {
		t.Fatalf("no-signal merge = %+v, want nil", g)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestLabReportRejectsInvalidGPUFlags(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("FAK_FLEET_REPORTS", dir)
	for _, args := range [][]string{
		{"report", "--id", "x", "--state", "live", "--gpu-total", "-5"},
		{"report", "--id", "x", "--state", "live", "--gpu-busy", "-3"},
		{"report", "--id", "x", "--state", "live", "--gpu-util", "101"},
		{"report", "--id", "x", "--state", "live", "--gpu-util", "-5"},
		{"report", "--id", "x", "--state", "live", "--gpu-total", "2", "--gpu-busy", "3"},
	} {
		if rc := runLab(io.Discard, io.Discard, args); rc == 0 {
			t.Fatalf("invalid gpu flags %v must be refused", args)
		}
	}
	// -1 is the unset sentinel: it behaves exactly like omitting the flag.
	if rc := runLab(io.Discard, io.Discard, []string{"report", "--id", "x", "--state", "live", "--gpu-total", "-1"}); rc != 0 {
		t.Fatalf("--gpu-total -1 is the unset sentinel and must be accepted, got %d", rc)
	}
	if gpu := readLabReportGPU(t, dir, "x"); gpu != nil {
		t.Fatalf("the sentinel must not create a gpu block, got %+v", gpu)
	}
}
