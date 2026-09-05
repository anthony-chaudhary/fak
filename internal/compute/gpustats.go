package compute

import (
	"context"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

// gpustats.go — the fail-soft `nvidia-smi --query-gpu` accelerator probe (#2052).
//
// DeviceMemoryInfo (capacity.go) reads VRAM through the in-kernel backend's own
// device handle and is the PREFERRED source when a CUDA/Vulkan backend is live.
// It reports memory only — there is no utilization on that seam — and it reports
// nothing at all on a host that has a GPU but no active HAL backend. This is the
// fallback the issue names for both gaps: it shells to nvidia-smi (the one tool
// present on essentially every NVIDIA host) and reads per-device VRAM AND
// utilization. It is fail-soft by construction — no nvidia-smi, a timeout, or
// unparseable output yields present=false so the harness-resource sampler keeps
// the GPU axis honestly absent rather than reporting a fabricated 0.

// GPUStat is one accelerator device's live reading from `nvidia-smi --query-gpu`.
// VRAM is in bytes; UtilizationPct is 0..100.
type GPUStat struct {
	Index          int
	VRAMUsedBytes  uint64
	VRAMTotalBytes uint64
	UtilizationPct float64
}

// nvidiaSMIQueryArgs is the exact query the fallback runs: ordered fields, CSV,
// no header, no units — so memory reads as MiB and utilization as integer
// percent, both pure numbers for the parser.
const nvidiaSMIQueryArgs = "--query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits"

// smiRunner runs nvidia-smi with args and returns stdout. Split out so the parse
// + fold is unit-testable without a GPU; the default execs the real binary.
type smiRunner func(ctx context.Context, args ...string) (string, error)

func execNvidiaSMI(ctx context.Context, args ...string) (string, error) {
	// The program name is a string literal (not a LookPath'd variable) so the
	// interpreter-free request-path invariant stays statically checkable
	// (architest TestRequestPathInterpreterFree / DIRECTION.md). CommandContext
	// resolves it on PATH and fails soft — "executable file not found" — when the
	// host has no nvidia-smi.
	out, err := exec.CommandContext(ctx, "nvidia-smi", args...).Output()
	if err != nil {
		return "", err
	}
	return string(out), nil
}

// NvidiaGPUStats reads per-device VRAM + utilization via nvidia-smi. Fail-soft:
// any error (no nvidia-smi on PATH, timeout, unparseable output, zero devices)
// yields (nil, false) so the caller keeps the GPU axis honestly n/a.
func NvidiaGPUStats() (stats []GPUStat, present bool) {
	return nvidiaGPUStats(execNvidiaSMI)
}

func nvidiaGPUStats(run smiRunner) ([]GPUStat, bool) {
	if run == nil {
		return nil, false
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	out, err := run(ctx, strings.Fields(nvidiaSMIQueryArgs)...)
	if err != nil {
		return nil, false
	}
	stats, ok := parseNvidiaSMIStats(out)
	if !ok {
		return nil, false
	}
	return stats, true
}

// parseNvidiaSMIStats parses `index, memory.used, memory.total, utilization.gpu`
// CSV rows (nounits: memory in MiB, utilization in integer percent). A row that
// lacks four numeric fields is skipped (e.g. a MIG device reporting `[N/A]`);
// ok is false only when no usable row parses at all.
func parseNvidiaSMIStats(csv string) ([]GPUStat, bool) {
	const mib = 1 << 20
	var out []GPUStat
	for _, line := range strings.Split(csv, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		f := strings.Split(line, ",")
		if len(f) < 4 {
			continue
		}
		idx, err1 := strconv.Atoi(strings.TrimSpace(f[0]))
		usedMiB, err2 := strconv.ParseFloat(strings.TrimSpace(f[1]), 64)
		totalMiB, err3 := strconv.ParseFloat(strings.TrimSpace(f[2]), 64)
		util, err4 := strconv.ParseFloat(strings.TrimSpace(f[3]), 64)
		if err1 != nil || err2 != nil || err3 != nil || err4 != nil {
			continue
		}
		out = append(out, GPUStat{
			Index:          idx,
			VRAMUsedBytes:  uint64(usedMiB * mib),
			VRAMTotalBytes: uint64(totalMiB * mib),
			UtilizationPct: util,
		})
	}
	return out, len(out) > 0
}

// AggregateGPUStats folds per-device stats into one harness-level reading: VRAM
// used/total summed across devices, utilization the MAX across devices (the
// busiest device bounds the harness's accelerator pressure). ok is false for an
// empty slice.
func AggregateGPUStats(stats []GPUStat) (usedBytes, totalBytes uint64, utilPct float64, ok bool) {
	if len(stats) == 0 {
		return 0, 0, 0, false
	}
	for _, s := range stats {
		usedBytes += s.VRAMUsedBytes
		totalBytes += s.VRAMTotalBytes
		if s.UtilizationPct > utilPct {
			utilPct = s.UtilizationPct
		}
	}
	return usedBytes, totalBytes, utilPct, true
}

// SystemGPUStats queries accelerator telemetry from available platform probes.
// On Darwin, it checks Apple Silicon IORegistry IOAccelerator telemetry first,
// falling back to nvidia-smi. On other platforms, it queries nvidia-smi.
func SystemGPUStats() (stats []GPUStat, present bool) {
	if stats, ok := AppleSiliconGPUStats(); ok {
		return stats, true
	}
	return NvidiaGPUStats()
}

// HarnessGPUVRAM selects the VRAM reading for the harness-resource sampler (#2052).
// It PREFERS the in-kernel backend's own device handle — deviceTotal/deviceFree/
// deviceKnown exactly as DeviceMemoryInfo returns them — because that is the memory
// the live backend actually sees; it falls back to the nvidia-smi aggregate (summed
// across devices) only when the handle cannot report (deviceKnown=false or a
// non-positive total), which is the "hosts where only nvidia-smi is available" case
// the issue names. used is clamped into [0, total] so a transient free>total or a
// negative free never underflows the unsigned axis. ok is false when neither source
// has a usable reading, so the sampler keeps the VRAM axis honestly n/a.
func HarnessGPUVRAM(deviceTotal, deviceFree int64, deviceKnown bool, smi []GPUStat) (used, total uint64, ok bool) {
	if deviceKnown && deviceTotal > 0 {
		u := deviceTotal - deviceFree
		if deviceFree < 0 || u < 0 {
			u = 0
		}
		if u > deviceTotal {
			u = deviceTotal
		}
		return uint64(u), uint64(deviceTotal), true
	}
	if u, t, _, aok := AggregateGPUStats(smi); aok {
		return u, t, true
	}
	return 0, 0, false
}
