package hwutil

// Cross-platform GPU collection. NVIDIA devices are read through nvidia-smi
// with a driver-generation-stable core field set (identity, memory,
// utilization, power, SM clock pair, uncorrected ECC); the CSV style follows
// internal/modelperfobs/bandwidth_collect_nvidia.go without importing it.
// After the core row, each NVIDIA row also carries the curated default
// receipt field selection (ADAPTED from the NVIDIA/dcgm-exporter default
// counter curation, etc/default-counters.csv @ 4e663a526c46, Apache-2.0):
// thermal, ECC SBE/DBE volatile/aggregate totals, NVLink CRC/replay error
// totals on the datacenter NVLink generations, and row-remap counters. That
// pass is one best-effort query per row: a failing or short answer leaves
// every receipt field nil and keeps the core row intact. The DCGM-only
// remainder of the upstream curation (per-reason throttle seconds, per-lane
// NVLink throughput L0-L17, NVLink recovery success/failed counters, fabric
// health summary) has no nvidia-smi source and stays out of the query until
// an NVML/DCGM source lands (open follow-on, deliberately not built here).
// PCIe throughput is NOT queried either: pcie.rx/tx_throughput are
// NVML/DCGM-only counters, not valid nvidia-smi query fields — querying them
// makes the whole command fail (exit 2, "Field ... is not a valid field to
// query") and zeros the entire NVIDIA surface (witnessed on an RTX 5090
// Laptop GPU node). PCIERxBytesPerS / PCIETxBytesPerS therefore stay nil for
// NVIDIA until an NVML/DCGM source lands (open follow-on, deliberately not
// built here). If even the core query exits nonzero (a future driver-era
// field rename), one retry with the reduced essential set runs before giving
// up. AMD devices are read from /sys/class/drm on Linux following the
// internal amdgpu sysfs patterns; a DRM card that exposes no readable metric
// (a platform iGPU PCI function) emits no row instead of an all-nil one, and
// connector nodes (card0-DP-1 style) are skipped. A host with no GPU surface
// yields an empty slice, never an error.

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
		nvidiaReceipt(ctx, &g)
		gpus = append(gpus, g)
	}
	return gpus
}

// nvidiaReceiptGeneration is the closed generation vocabulary used by the
// default receipt field selection.
type nvidiaReceiptGeneration string

const (
	// genDatacenterNVLink marks datacenter SKUs with NVLink health counters
	// (Volta through Blackwell): Tesla V/A/H/B/E series and peers.
	genDatacenterNVLink nvidiaReceiptGeneration = "datacenter-nvlink"
	// genConsumer marks desktop/laptop SKUs without NVLink health counters.
	genConsumer nvidiaReceiptGeneration = "consumer"
)

// nvidiaReceiptGenerationOf classifies one device name into the closed
// generation vocabulary. GeForce names and the desktop RTX series are
// consumer; everything else (Tesla/datacenter series, RTX A-series and
// 6xxx/8xxx workstation SKUs, unknown names) keeps the NVLink receipt fields
// — an unsupported counter answers [N/A] and stays nil, never an error.
func nvidiaReceiptGenerationOf(name string) nvidiaReceiptGeneration {
	n := strings.ToLower(name)
	if strings.Contains(n, "geforce") || rtxDesktopSeries(n) {
		return genConsumer
	}
	return genDatacenterNVLink
}

// rtxDesktopSeries reports whether the name marks a desktop RTX series
// (RTX 3xxx/4xxx/5xxx). Letter-led tokens (RTX A6000) and 6xxx/8xxx
// workstation SKUs keep the NVLink fields.
func rtxDesktopSeries(name string) bool {
	i := strings.Index(name, "rtx ")
	if i < 0 {
		return false
	}
	rest := name[i+len("rtx "):]
	if len(rest) < 2 || rest[0] < '0' || rest[0] > '9' || rest[1] < '0' || rest[1] > '9' {
		return false
	}
	decade, _ := strconv.Atoi(rest[:2])
	return decade < 60
}

// nvidiaReceiptFieldsBase is the curated default receipt field selection
// shared by every generation: thermal, ECC SBE/DBE volatile/aggregate
// totals, and row-remap counters.
var nvidiaReceiptFieldsBase = []string{
	"temperature.gpu",                    // thermal
	"ecc.errors.volatile.correctable",    // ECC SBE volatile
	"ecc.errors.volatile.uncorrectable",  // ECC DBE volatile
	"ecc.errors.aggregate.correctable",   // ECC SBE aggregate
	"ecc.errors.aggregate.uncorrectable", // ECC DBE aggregate
	"retired_pages.pending",              // row remap pending
	"retired_pages.aggregate",            // row remap total
	"retired_pages.double_bit_aggregate", // row remap double-bit
}

// nvidiaReceiptFieldsNVLink extends the base selection for the datacenter
// NVLink generations: NVLink CRC/replay error totals. The upstream curation
// also carries Hopper data-link and Blackwell recovery-success/failed
// counters in this band; those have no nvidia-smi source (DCGM-only) and stay
// out of the query.
var nvidiaReceiptFieldsNVLink = []string{
	"nvlink.errorcount.crc",
	"nvlink.errorcount.replay",
}

// nvidiaReceiptFields returns the exact default receipt field list for one
// generation; the returned slice is owned by the caller.
func nvidiaReceiptFields(gen nvidiaReceiptGeneration) []string {
	out := make([]string, 0, len(nvidiaReceiptFieldsBase)+len(nvidiaReceiptFieldsNVLink))
	out = append(out, nvidiaReceiptFieldsBase...)
	if gen == genDatacenterNVLink {
		out = append(out, nvidiaReceiptFieldsNVLink...)
	}
	return out
}

// nvidiaReceipt runs the curated default receipt field selection for one row:
// a single best-effort nvidia-smi query carrying exactly the generation's
// field list. A failing or short query leaves every receipt field nil (the
// core row is unaffected); [N/A] placeholders and negative values stay nil.
func nvidiaReceipt(ctx context.Context, g *GPU) {
	fields := nvidiaReceiptFields(nvidiaReceiptGenerationOf(g.Name))
	rows, err := nvidiaQueryCSVID(ctx, g.UUID, strings.Join(fields, ","))
	if err != nil || len(rows) == 0 || len(rows[0]) != len(fields) {
		return
	}
	v := make(map[string]string, len(fields))
	for i, n := range fields {
		v[n] = strings.TrimSpace(rows[0][i])
	}
	g.TempC = nvidiaFloat(v["temperature.gpu"])
	g.ECCCorrectableVolatile = nvidiaNonNegInt(v["ecc.errors.volatile.correctable"])
	g.ECCUncorrectableVolatile = nvidiaNonNegInt(v["ecc.errors.volatile.uncorrectable"])
	g.ECCCorrectableAggregate = nvidiaNonNegInt(v["ecc.errors.aggregate.correctable"])
	g.ECCUncorrectableAggregate = nvidiaNonNegInt(v["ecc.errors.aggregate.uncorrectable"])
	g.NVLinkCRCErrorTotal = nvidiaNonNegInt(v["nvlink.errorcount.crc"])
	g.NVLinkReplayErrorTotal = nvidiaNonNegInt(v["nvlink.errorcount.replay"])
	g.RowRemapPendingTotal = nvidiaNonNegInt(v["retired_pages.pending"])
	g.RowRemapTotal = nvidiaNonNegInt(v["retired_pages.aggregate"])
	g.RowRemapDBETotal = nvidiaNonNegInt(v["retired_pages.double_bit_aggregate"])
}

// nvidiaNonNegInt parses a non-negative integer counter; unreadable or
// negative values stay nil (null-on-error at the source edge).
func nvidiaNonNegInt(s string) *int64 {
	v := nvidiaInt(s)
	if v == nil || *v < 0 {
		return nil
	}
	return v
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
