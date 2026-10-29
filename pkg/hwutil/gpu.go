package hwutil

// Cross-platform GPU collection. NVIDIA devices are read through nvidia-smi
// with a driver-generation-stable core field set (identity, memory,
// utilization, power, SM clock pair, uncorrected ECC); the CSV style follows
// internal/modelperfobs/bandwidth_collect_nvidia.go without importing it.
// PCIe throughput is NOT queried: pcie.rx/tx_throughput are NVML/DCGM-only
// counters, not valid nvidia-smi query fields — querying them makes the whole
// command fail (exit 2, "Field ... is not a valid field to query") and zeros
// the entire NVIDIA surface (witnessed on an RTX 5090 Laptop GPU node).
// PCIERxBytesPerS / PCIETxBytesPerS therefore stay nil for NVIDIA until an
// NVML/DCGM source lands (open follow-on, deliberately not built here). If
// even the core query exits nonzero (a future driver-era field rename), one
// retry with the reduced essential set runs before giving up. AMD devices are
// read from /sys/class/drm on Linux following the internal amdgpu sysfs
// patterns; a DRM card that exposes no readable metric (a platform iGPU PCI
// function) emits no row instead of an all-nil one, and connector nodes
// (card0-DP-1 style) are skipped. A host with no GPU surface yields an empty
// slice, never an error.

import (
	"context"
	"encoding/csv"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
)

var runNvidiaSMI = func(ctx context.Context, args ...string) ([]byte, error) {
	return exec.CommandContext(ctx, "nvidia-smi", args...).Output()
}

// nvidiaCoreFields is the primary nvidia-smi query set: stable across driver
// generations. It deliberately excludes pcie.*_throughput — those counters
// are NVML/DCGM-only and are not valid nvidia-smi fields.
const nvidiaCoreFields = "uuid,name,pci.bus_id,memory.total,memory.used,utilization.gpu,power.draw,clocks.current.sm,clocks.max.sm,ecc.errors.uncorrected.vram"

// nvidiaEssentialFields is the reduced fallback set used for exactly one
// retry when the core query exits nonzero (guards against a future field
// rename breaking the wider set); rows parsed from it leave the SM clock
// pair and ECC total nil.
const nvidiaEssentialFields = "uuid,name,pci.bus_id,memory.total,memory.used,utilization.gpu,power.draw"

// collectGPUs probes every available vendor surface. An absent or failing
// surface contributes nothing; no GPU at all returns an empty (non-nil)
// slice so the wire shape stays an array.
func collectGPUs(ctx context.Context) []GPU {
	out := []GPU{}
	if _, err := exec.LookPath("nvidia-smi"); err == nil {
		out = append(out, nvidiaGPUs(ctx)...)
	}
	if runtime.GOOS == "linux" {
		out = append(out, amdSysfsgpus(ctx)...)
	}
	return out
}

func nvidiaGPUs(ctx context.Context) []GPU {
	fields := nvidiaCoreFields
	rows, err := nvidiaQueryCSV(ctx, fields)
	if err != nil {
		// One retry with the reduced essential set guards against a
		// driver-era field rename breaking the wider core query; only the
		// reduced result is recorded.
		fields = nvidiaEssentialFields
		rows, err = nvidiaQueryCSV(ctx, fields)
	}
	if err != nil || len(rows) == 0 {
		return nil
	}
	gpus := make([]GPU, 0, len(rows))
	for _, r := range rows {
		g, ok := nvidiaRow(fields, r)
		if !ok {
			continue
		}
		g.ThrottleReasons = nvidiaThrottleReasons(ctx, g.UUID)
		gpus = append(gpus, g)
	}
	return gpus
}

// nvidiaRow parses one CSV row against the exact field list that produced
// it, so positional mapping stays honest as the query set evolves.
func nvidiaRow(fields string, r []string) (GPU, bool) {
	names := strings.Split(fields, ",")
	if len(r) != len(names) {
		return GPU{}, false
	}
	v := make(map[string]string, len(names))
	for i, n := range names {
		v[n] = strings.TrimSpace(r[i])
	}
	g := GPU{
		UUID:                v["uuid"],
		Name:                v["name"],
		PCI:                 strings.TrimPrefix(v["pci.bus_id"], "PCI:"),
		MemTotalBytes:       nvidiaMiBtoBytes(v["memory.total"]),
		MemUsedBytes:        nvidiaMiBtoBytes(v["memory.used"]),
		UtilPct:             nvidiaPercent(v["utilization.gpu"]),
		PowerW:              nvidiaFloat(v["power.draw"]),
		SMClockMHz:          nvidiaFloat(v["clocks.current.sm"]),
		MaxSMClockMHz:       nvidiaFloat(v["clocks.max.sm"]),
		ECCUncorrectedTotal: nvidiaInt(v["ecc.errors.uncorrected.vram"]),
	}
	// PCIERxBytesPerS / PCIETxBytesPerS intentionally stay nil: nvidia-smi
	// does not expose PCIe throughput (NVML/DCGM-only counters).
	if g.UUID == "" {
		g.UUID = g.Name
	}
	return g, true
}

// nvidiaThrottleReasons reads one device's active throttle-reason set
// best-effort; the key name moved between driver generations, so both
// spellings are tried and failure leaves the field an empty slice (nothing
// active reported).
func nvidiaThrottleReasons(ctx context.Context, id string) []string {
	for _, key := range []string{"clocks_event_reasons.active", "clocks_throttle_reasons.active"} {
		rows, err := nvidiaQueryCSVID(ctx, id, key)
		if err != nil || len(rows) == 0 || len(rows[0]) != 1 {
			continue
		}
		val := strings.TrimSpace(rows[0][0])
		if val == "" || strings.EqualFold(val, "Not Active") || nvidiaUnreadable(val) {
			return []string{}
		}
		parts := strings.Split(val, ",")
		out := make([]string, 0, len(parts))
		for _, p := range parts {
			if p = strings.TrimSpace(p); p != "" && !strings.EqualFold(p, "Not Active") {
				out = append(out, p)
			}
		}
		return out
	}
	return []string{}
}

func nvidiaQueryCSV(ctx context.Context, fields string) ([][]string, error) {
	return nvidiaQueryCSVID(ctx, "", fields)
}

func nvidiaQueryCSVID(ctx context.Context, id, fields string) ([][]string, error) {
	args := []string{}
	if id != "" {
		args = append(args, "--id="+id)
	}
	args = append(args, "--query-gpu="+fields, "--format=csv,noheader,nounits")
	out, err := runNvidiaSMI(ctx, args...)
	if err != nil {
		return nil, err
	}
	return csv.NewReader(strings.NewReader(strings.TrimSpace(string(out)))).ReadAll()
}

// nvidiaUnreadable matches the [N/A] / [N/A s] placeholders nvidia-smi emits
// for unsupported counters.
func nvidiaUnreadable(s string) bool {
	s = strings.TrimSpace(s)
	return s == "" || strings.HasPrefix(s, "[N/A")
}

func nvidiaFloat(s string) *float64 {
	if nvidiaUnreadable(s) {
		return nil
	}
	v, err := strconv.ParseFloat(strings.TrimSpace(s), 64)
	if err != nil {
		return nil
	}
	return f64(v)
}

func nvidiaInt(s string) *int64 {
	if nvidiaUnreadable(s) {
		return nil
	}
	v, err := strconv.ParseInt(strings.TrimSpace(s), 10, 64)
	if err != nil {
		return nil
	}
	return i64(v)
}

// nvidiaPercent parses a 0..100 percentage; out-of-range values are unread.
func nvidiaPercent(s string) *float64 {
	v := nvidiaFloat(s)
	if v == nil || *v < 0 || *v > 100 {
		return nil
	}
	return v
}

// nvidiaMiBtoBytes converts MiB-valued memory counters to bytes.
func nvidiaMiBtoBytes(s string) *int64 {
	v := nvidiaInt(s)
	if v == nil {
		return nil
	}
	return i64(*v * 1024 * 1024)
}

// drmRootDir is the sysfs class directory scanned for amdgpu card nodes;
// tests point it at a fixture tree.
var drmRootDir = "/sys/class/drm"

// amdSysfsgpus reads /sys/class/drm/card*/device counters (the internal
// amdgpu pattern): busy percent, VRAM info, power, and the PCI BDF from the
// device symlink. Unread files stay nil. A card whose device exposes no
// readable metric (a platform iGPU PCI function with no amdgpu sysfs
// surface) is skipped entirely rather than emitted as an all-nil row, and
// connector nodes (card0-DP-1 style) are not GPUs and never emit rows.
func amdSysfsgpus(ctx context.Context) []GPU {
	cards, err := filepath.Glob(filepath.Join(drmRootDir, "card*"))
	if err != nil || len(cards) == 0 {
		return nil
	}
	var out []GPU
	for _, card := range cards {
		name := filepath.Base(card)
		if strings.Contains(name, "-") {
			continue // connector node (card0-DP-1 style), not a card
		}
		dev := filepath.Join(card, "device")
		if fi, err := os.Lstat(dev); err != nil || fi.Mode()&os.ModeSymlink == 0 {
			continue // not a real GPU card node
		}
		target, err := os.Readlink(dev)
		if err != nil {
			continue
		}
		pci := filepath.Base(target)
		g := GPU{UUID: pci, Name: name + " (amdgpu)", PCI: pci}
		readable := false
		if v, ok := parseFloatFile(filepath.Join(dev, "gpu_busy_percent")); ok && v >= 0 && v <= 100 {
			g.UtilPct = f64(v)
			readable = true
		}
		if v, ok := parseUintFile(filepath.Join(dev, "mem_info_vram_total")); ok && v > 0 {
			g.MemTotalBytes = i64(int64(v))
			readable = true
		}
		if v, ok := parseUintFile(filepath.Join(dev, "mem_info_vram_used")); ok {
			g.MemUsedBytes = i64(int64(v))
			readable = true
		}
		if v, ok := parseFloatFile(filepath.Join(dev, "power")); ok && v > 0 {
			g.PowerW = f64(v)
			readable = true
		}
		if !readable {
			continue // no amdgpu metrics exposed: skip, no all-nil row
		}
		out = append(out, g)
	}
	return out
}
