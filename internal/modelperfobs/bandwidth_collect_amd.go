package modelperfobs

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os/exec"
	"strconv"
	"strings"
)

// AMDDeviceSelector is passed to amd-smi --gpu. Empty selects device 0.
type AMDDeviceSelector string

var runAMDSMI = func(ctx context.Context, args ...string) ([]byte, error) {
	return exec.CommandContext(ctx, "amd-smi", args...).Output()
}

var runRDCI = func(ctx context.Context, args ...string) ([]byte, error) {
	return exec.CommandContext(ctx, "rdci", args...).Output()
}

// The RDC convenience field RDC_FI_GPU_MEMORY_CUR_BANDWIDTH is deliberately
// excluded: upstream derives it by multiplying UMC active time by the maximum
// bandwidth. Only the rocprofiler-backed read/write rate fields are byte-traffic
// evidence suitable for LiveBandwidth.
const rdcMemoryRateFields = "RDC_FI_PROF_EVAL_MEM_R_BW,RDC_FI_PROF_EVAL_MEM_W_BW"

type amdSMIMetrics struct {
	gpu      int
	device   DeviceSignals
	capacity CapacitySignals
}

// collectAMDDeviceSnapshot combines low-overhead AMD SMI device state with
// RDC's profiler-backed video-memory read/write rates. Either collector may be
// absent; unsupported throughput stays unavailable rather than becoming zero or
// UMC-percent-times-roofline arithmetic.
func collectAMDDeviceSnapshot(ctx context.Context, selector AMDDeviceSelector) (deviceSnapshot, error) {
	id := strings.TrimSpace(string(selector))
	if id == "" {
		id = "0"
	}
	s := deviceSnapshot{provenanceDevice: "AMD ROCm GPU " + id, provenanceSource: "live-amd-rocm"}
	var collectors []string

	if out, err := runAMDSMI(ctx, "metric", "--gpu", id, "--usage", "--mem-usage", "--json"); err == nil {
		metric, err := parseAMDSMIOutput(out)
		if err != nil {
			return deviceSnapshot{}, err
		}
		s.device = metric.device
		s.capacity = metric.capacity
		s.provenanceDevice = fmt.Sprintf("AMD ROCm GPU %d", metric.gpu)
		s.available = true
		id = strconv.Itoa(metric.gpu)
		collectors = append(collectors, "amd-smi")
	}

	// rdci accepts a numeric GPU index, whereas amd-smi may initially receive a
	// BDF or UUID. A successful AMD SMI parse above resolves those selectors.
	if _, err := strconv.ParseUint(id, 10, 32); err == nil {
		out, runErr := runRDCI(ctx, "dmon", "-i", id, "-e", rdcMemoryRateFields, "-c", "1", "-d", "100")
		if runErr == nil {
			live, err := parseRDCIMemoryRates(out, id)
			if err != nil {
				return deviceSnapshot{}, err
			}
			s.live = live
			s.dramCounters = live.ReadGBS != nil || live.WriteGBS != nil
			s.available = true
			collectors = append(collectors, "rdci-rocp")
		}
	}

	if !s.available {
		return deviceSnapshot{}, nil
	}
	s.collector = strings.Join(collectors, "+")
	return s, nil
}

func parseAMDSMIOutput(data []byte) (amdSMIMetrics, error) {
	var rows []json.RawMessage
	trimmed := bytes.TrimSpace(data)
	if len(trimmed) == 0 {
		return amdSMIMetrics{}, fmt.Errorf("parse amd-smi metric output: empty JSON")
	}
	if trimmed[0] == '[' {
		if err := json.Unmarshal(trimmed, &rows); err != nil {
			return amdSMIMetrics{}, fmt.Errorf("parse amd-smi metric output: %w", err)
		}
	} else {
		var envelope struct {
			GPUData []json.RawMessage `json:"gpu_data"`
		}
		if err := json.Unmarshal(trimmed, &envelope); err != nil {
			return amdSMIMetrics{}, fmt.Errorf("parse amd-smi metric output: %w", err)
		}
		rows = envelope.GPUData
	}
	if len(rows) != 1 {
		return amdSMIMetrics{}, fmt.Errorf("parse amd-smi metric output: expected one GPU row, got %d", len(rows))
	}
	var row struct {
		GPU   json.RawMessage `json:"gpu"`
		Usage struct {
			GFX json.RawMessage `json:"gfx_activity"`
			UMC json.RawMessage `json:"umc_activity"`
		} `json:"usage"`
		MemUsage struct {
			Total json.RawMessage `json:"total_vram"`
			Used  json.RawMessage `json:"used_vram"`
		} `json:"mem_usage"`
	}
	if err := json.Unmarshal(rows[0], &row); err != nil {
		return amdSMIMetrics{}, fmt.Errorf("parse amd-smi GPU row: %w", err)
	}
	gpu, err := parseAMDSMIInt(row.GPU)
	if err != nil {
		return amdSMIMetrics{}, err
	}
	gfx, err := parseAMDSMIRatio(row.Usage.GFX)
	if err != nil {
		return amdSMIMetrics{}, fmt.Errorf("parse amd-smi GFX activity: %w", err)
	}
	umc, err := parseAMDSMIRatio(row.Usage.UMC)
	if err != nil {
		return amdSMIMetrics{}, fmt.Errorf("parse amd-smi UMC activity: %w", err)
	}
	total, err := parseAMDSMIBytes(row.MemUsage.Total)
	if err != nil {
		return amdSMIMetrics{}, fmt.Errorf("parse amd-smi total VRAM: %w", err)
	}
	used, err := parseAMDSMIBytes(row.MemUsage.Used)
	if err != nil {
		return amdSMIMetrics{}, fmt.Errorf("parse amd-smi used VRAM: %w", err)
	}
	if total != nil && used != nil && *used > *total {
		return amdSMIMetrics{}, fmt.Errorf("parse amd-smi VRAM: used bytes exceed total bytes")
	}
	return amdSMIMetrics{gpu: gpu, device: DeviceSignals{ComputeUtilization: gfx, MemoryControllerUtilization: umc}, capacity: CapacitySignals{UsedBytes: used, TotalBytes: total}}, nil
}

func parseAMDSMIInt(raw json.RawMessage) (int, error) {
	var n int
	if err := json.Unmarshal(raw, &n); err == nil && n >= 0 {
		return n, nil
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		v, parseErr := strconv.Atoi(strings.TrimSpace(s))
		if parseErr == nil && v >= 0 {
			return v, nil
		}
	}
	return 0, fmt.Errorf("parse amd-smi GPU index %q", raw)
}

func parseAMDSMIRatio(raw json.RawMessage) (*float64, error) {
	v, unit, err := parseAMDSMIQuantity(raw)
	if err != nil || v == nil {
		return v, err
	}
	if unit != "%" {
		return nil, fmt.Errorf("unexpected unit %q, want %%", unit)
	}
	if *v < 0 || *v > 100 {
		return nil, fmt.Errorf("percent out of range: %v", *v)
	}
	*v /= 100
	return v, nil
}

func parseAMDSMIBytes(raw json.RawMessage) (*uint64, error) {
	v, unit, err := parseAMDSMIQuantity(raw)
	if err != nil || v == nil {
		return nil, err
	}
	var multiplier float64
	switch unit {
	case "B":
		multiplier = 1
	case "MB", "MiB":
		// amd-smi labels this MB after integer division by 1024*1024.
		multiplier = 1024 * 1024
	default:
		return nil, fmt.Errorf("unexpected unit %q, want B or MB", unit)
	}
	bytes := *v * multiplier
	if bytes < 0 || bytes > math.MaxUint64 || math.Trunc(bytes) != bytes {
		return nil, fmt.Errorf("invalid byte quantity %v %s", *v, unit)
	}
	out := uint64(bytes)
	return &out, nil
}

func parseAMDSMIQuantity(raw json.RawMessage) (*float64, string, error) {
	if len(bytes.TrimSpace(raw)) == 0 || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return nil, "", nil
	}
	if unavailable, ok := unavailableMetric(raw); ok {
		if deviceMetricUnavailable(unavailable) {
			return nil, "", nil
		}
		return nil, "", fmt.Errorf("unexpected string %q", unavailable)
	}
	var quantity struct {
		Value json.RawMessage `json:"value"`
		Unit  string          `json:"unit"`
	}
	if err := json.Unmarshal(raw, &quantity); err != nil {
		return nil, "", err
	}
	if len(quantity.Value) == 0 {
		return nil, "", fmt.Errorf("quantity has no value")
	}
	if unavailable, ok := unavailableMetric(quantity.Value); ok {
		if deviceMetricUnavailable(unavailable) {
			return nil, "", nil
		}
		return nil, "", fmt.Errorf("unexpected value %q", unavailable)
	}
	var value float64
	if err := json.Unmarshal(quantity.Value, &value); err != nil {
		return nil, "", err
	}
	if math.IsNaN(value) || math.IsInf(value, 0) {
		return nil, "", fmt.Errorf("non-finite value")
	}
	return &value, strings.TrimSpace(quantity.Unit), nil
}

func unavailableMetric(raw json.RawMessage) (string, bool) {
	var unavailable string
	if err := json.Unmarshal(raw, &unavailable); err != nil {
		return "", false
	}
	return unavailable, true
}

// parseRDCIMemoryRates parses rdci dmon's stable field-name table. RDC's
// profiler fields are expressed as KB/ms; dividing by 1000 converts that rate
// to decimal GB/s without inventing traffic from utilization percentages.
func parseRDCIMemoryRates(data []byte, gpu string) (LiveBandwidth, error) {
	scanner := bufio.NewScanner(bytes.NewReader(data))
	readIndex, writeIndex := -1, -1
	for scanner.Scan() {
		fields := strings.Fields(scanner.Text())
		if len(fields) == 0 {
			continue
		}
		if readIndex < 0 {
			for i, field := range fields {
				switch field {
				case "MEM_R_BW":
					readIndex = i
				case "MEM_W_BW":
					writeIndex = i
				}
			}
			if readIndex >= 0 && writeIndex >= 0 {
				continue
			}
			readIndex, writeIndex = -1, -1
			continue
		}
		maxIndex := readIndex
		if writeIndex > maxIndex {
			maxIndex = writeIndex
		}
		if len(fields) <= maxIndex || fields[0] != gpu {
			continue
		}
		read, err := parseRDCIRate(fields[readIndex])
		if err != nil {
			return LiveBandwidth{}, fmt.Errorf("parse rdci MEM_R_BW: %w", err)
		}
		write, err := parseRDCIRate(fields[writeIndex])
		if err != nil {
			return LiveBandwidth{}, fmt.Errorf("parse rdci MEM_W_BW: %w", err)
		}
		return LiveBandwidth{ReadGBS: read, WriteGBS: write}, nil
	}
	if err := scanner.Err(); err != nil {
		return LiveBandwidth{}, err
	}
	return LiveBandwidth{}, fmt.Errorf("parse rdci dmon: no GPU %s memory-rate row", gpu)
}

func parseRDCIRate(s string) (*float64, error) {
	if deviceMetricUnavailable(s) {
		return nil, nil
	}
	v, err := strconv.ParseFloat(strings.TrimSpace(s), 64)
	if err != nil {
		return nil, err
	}
	if v < 0 || math.IsNaN(v) || math.IsInf(v, 0) {
		return nil, fmt.Errorf("invalid KB/ms rate %v", v)
	}
	v /= 1000
	return &v, nil
}
