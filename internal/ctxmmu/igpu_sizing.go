// Integrated-device KV sizing. Ported from NVIDIA/TensorRT-LLM
// tensorrt_llm/_torch/pyexecutor/_util.py::_mem_get_info_for_kv_sizing
// (Apache-2.0, @a362600648d451d99a6692554d93bca0dc470149), wired into
// configure_kv_cache_capacity upstream.
//
// On integrated (unified-memory) GPUs the CPU page cache — e.g. the one
// filled by reading the checkpoint — is not reported as free by the device
// free-memory probe, although the kernel reclaims it on demand and device
// allocations can use it. Counting it as used shrinks the KV pool budget.
// There the effective free estimate is max(cudaFree, min(MemAvailable, total)),
// using the kernel's available-memory estimate. Discrete devices keep the raw
// device probe byte-for-byte: the gate below never consults the kernel probe
// for them.

package ctxmmu

import (
	"math"
	"os"
	"runtime"
	"strconv"
	"strings"
)

// EnvHalogenIGPU overrides integrated-device detection for KV sizing:
// 1/true/on/yes force integrated, 0/false/off/no force discrete.
const EnvHalogenIGPU = "HALOGEN_IGPU"

// memInfoPath is the kernel memory report the MemAvailable probe parses.
const memInfoPath = "/proc/meminfo"

// nvidiaDriverMarker marks a discrete NVIDIA GPU driver on the host.
const nvidiaDriverMarker = "/proc/driver/nvidia"

const memInfoKilobyte = int64(1024)

// memInfoReader supplies the kernel memory report text. Tests inject a fixed
// reader so the probe never depends on a real /proc/meminfo existing.
var memInfoReader = func() (string, error) {
	data, err := os.ReadFile(memInfoPath)
	return string(data), err
}

// ParseMemAvailable extracts MemAvailable from /proc/meminfo text in bytes.
// ok is false when the field is absent or unparseable, including values that
// do not fit int64 after the kB-to-bytes conversion.
func ParseMemAvailable(text string) (int64, bool) {
	for _, line := range strings.Split(text, "\n") {
		kv := strings.SplitN(line, ":", 2)
		if len(kv) != 2 || strings.TrimSpace(kv[0]) != "MemAvailable" {
			continue
		}
		fields := strings.Fields(kv[1])
		if len(fields) == 0 {
			continue
		}
		v, err := strconv.ParseUint(fields[0], 10, 64)
		if err != nil {
			return 0, false
		}
		if v > uint64(math.MaxInt64)/uint64(memInfoKilobyte) {
			return 0, false
		}
		return int64(v) * memInfoKilobyte, true
	}
	return 0, false
}

// ProbeMemAvailable returns the kernel MemAvailable in bytes through the
// active reader. ok is false when the report is unreadable or unparsable.
func ProbeMemAvailable() (int64, bool) {
	text, err := memInfoReader()
	if err != nil {
		return 0, false
	}
	return ParseMemAvailable(text)
}

// integratedDetector decides whether the active compute device is
// integrated/UMA. Tests replace it to pin either side of the gate.
var integratedDetector = detectIntegratedDevice

// IsIntegratedDevice reports whether the active compute device is
// integrated/UMA (shares one DRAM aperture with the CPU, e.g. a Strix Halo
// iGPU or DGX Spark). It is consulted at KV pool init only.
func IsIntegratedDevice() bool {
	d := integratedDetector
	if d == nil {
		d = detectIntegratedDevice
	}
	return d()
}

// detectIntegratedDevice is the default detector:
//  1. The HALOGEN_IGPU override wins (operator control, same convention as
//     HALOGEN_KV_POOL_FIT).
//  2. Non-Linux hosts expose no kernel MemAvailable, so the fit path cannot
//     observe reclaimable cache: discrete.
//  3. On Linux the device is integrated when the kernel report is readable
//     and no discrete-GPU driver marker (/proc/driver/nvidia) is present.
func detectIntegratedDevice() bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv(EnvHalogenIGPU))) {
	case "1", "true", "on", "yes":
		return true
	case "0", "false", "off", "no":
		return false
	}
	if runtime.GOOS != "linux" {
		return false
	}
	if _, err := os.Stat(nvidiaDriverMarker); err == nil {
		return false
	}
	_, err := os.Stat(memInfoPath)
	return err == nil
}

// EffectiveFreeForIntegratedDevice is the pure sizing formula for an
// integrated/UMA device: max(cudaFree, min(memAvailable, total)). It never
// lowers the raw device probe and, whenever cudaFree <= total (which holds
// for any real device), never exceeds the total aperture.
func EffectiveFreeForIntegratedDevice(cudaFree, memAvailable, total int64) int64 {
	candidate := memAvailable
	if candidate > total {
		candidate = total
	}
	if cudaFree > candidate {
		return cudaFree
	}
	return candidate
}

// FitIntegratedKVPool applies the integrated-device sizing rule at KV pool
// init. When the active device is integrated/UMA it probes the kernel
// MemAvailable, computes the effective free estimate, and fits the pool via
// FitPoolToAvailableMemory; when the kernel fact is unavailable the raw
// device probe alone drives the fit. On any other device it leaves the pool
// untouched and never consults the kernel probe. It returns the pool's
// effective token ceiling after the call.
func FitIntegratedKVPool(p *SharedTokenPool, cudaFree int64) int {
	if p == nil {
		return 0
	}
	if !IsIntegratedDevice() {
		return p.EffectiveMaxTokens()
	}
	memAvailable, ok := ProbeMemAvailable()
	if !ok {
		memAvailable = cudaFree
	}
	return p.FitPoolToAvailableMemory(EffectiveFreeForIntegratedDevice(cudaFree, memAvailable, p.TotalGTTBytes()))
}
