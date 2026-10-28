package hwutil

// Cross-platform GPU collection. NVIDIA devices are read through nvidia-smi
// with a query set that extends the nine-field model-observe bandwidth probe
// (utilization, memory, power) with SM clocks, PCIe throughput, throttle
// reasons, and ECC errors; the CSV style follows
// internal/modelperfobs/bandwidth_collect_nvidia.go without importing it.
// AMD devices are read from /sys/class/drm on Linux following the internal
// amdgpu sysfs patterns. A host with no GPU surface yields an empty slice,
// never an error.

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

// nvidiaCoreFields extends the legacy nine-field probe with SM clock pair,
// PCIe throughput, and aggregate uncorrected ECC.
const nvidiaCoreFields = "uuid,name,pci.bus_id,memory.total,memory.used,utilization.gpu,power.draw,clocks.current.sm,clocks.max.sm,pcie.tx_throughput,pcie.rx_throughput,ecc.errors.uncorrected.vram"

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
	rows, err := nvidiaQueryCSV(ctx, nvidiaCoreFields)
	if err != nil || len(rows) == 0 {
		return nil
	}
	gpus := make([]GPU, 0, len(rows))
	for _, r := range rows {
		if len(r) != 12 {
			continue
		}
		g := GPU{
			UUID:                strings.TrimSpace(r[0]),
			Name:                strings.TrimSpace(r[1]),
			PCI:                 strings.TrimPrefix(strings.TrimSpace(r[2]), "PCI:"),
			MemTotalBytes:       nvidiaMiBtoBytes(r[3]),
			MemUsedBytes:        nvidiaMiBtoBytes(r[4]),
			UtilPct:             nvidiaPercent(r[5]),
			PowerW:              nvidiaFloat(r[6]),
			SMClockMHz:          nvidiaFloat(r[7]),
			MaxSMClockMHz:       nvidiaFloat(r[8]),
			PCIETxBytesPerS:     nvidiaFloat(r[9]),
			PCIERxBytesPerS:     nvidiaFloat(r[10]),
			ECCUncorrectedTotal: nvidiaInt(r[11]),
		}
		if g.UUID == "" {
			g.UUID = g.Name
		}
		g.ThrottleReasons = nvidiaThrottleReasons(ctx, g.UUID)
		gpus = append(gpus, g)
	}
	return gpus
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

// amdSysfsgpus reads /sys/class/drm/card*/device counters (the internal
// amdgpu pattern): busy percent, VRAM info, power, and the PCI BDF from the
// device symlink. Unread files stay nil.
func amdSysfsgpus(ctx context.Context) []GPU {
	cards, err := filepath.Glob("/sys/class/drm/card*")
	if err != nil || len(cards) == 0 {
		return nil
	}
	var out []GPU
	for _, card := range cards {
		dev := filepath.Join(card, "device")
		if fi, err := os.Lstat(dev); err != nil || fi.Mode()&os.ModeSymlink == 0 {
			continue // not a real GPU card node
		}
		target, err := os.Readlink(dev)
		if err != nil {
			continue
		}
		pci := filepath.Base(target)
		g := GPU{UUID: pci, Name: filepath.Base(card) + " (amdgpu)", PCI: pci}
		if v, ok := parseFloatFile(filepath.Join(dev, "gpu_busy_percent")); ok && v >= 0 && v <= 100 {
			g.UtilPct = f64(v)
		}
		if v, ok := parseUintFile(filepath.Join(dev, "mem_info_vram_total")); ok && v > 0 {
			g.MemTotalBytes = i64(int64(v))
		}
		if v, ok := parseUintFile(filepath.Join(dev, "mem_info_vram_used")); ok {
			g.MemUsedBytes = i64(int64(v))
		}
		if v, ok := parseFloatFile(filepath.Join(dev, "power")); ok && v > 0 {
			g.PowerW = f64(v)
		}
		out = append(out, g)
	}
	return out
}
