package modelperfobs

import (
	"context"
	"encoding/csv"
	"fmt"
	"os/exec"
	"strconv"
	"strings"
)

// NVIDIADeviceSelector is passed to nvidia-smi --id. Empty selects device 0.
type NVIDIADeviceSelector string

type deviceSnapshot struct {
	device           DeviceSignals
	capacity         CapacitySignals
	live             LiveBandwidth
	provenanceDevice string
	provenanceSource string
	collector        string
	available        bool
	dramCounters     bool
}

var runNvidiaSMI = func(ctx context.Context, args ...string) ([]byte, error) {
	return exec.CommandContext(ctx, "nvidia-smi", args...).Output()
}

// collectNVIDIADeviceSnapshot reads low-overhead controller activity and device
// state. utilization.memory is controller active-time percentage, not GB/s; it
// therefore never populates LiveBandwidth.
func collectNVIDIADeviceSnapshot(ctx context.Context, selector NVIDIADeviceSelector) (deviceSnapshot, error) {
	id := string(selector)
	if id == "" {
		id = "0"
	}
	fields := "uuid,name,memory.total,memory.used,utilization.gpu,utilization.memory,clocks.current.memory,power.draw,temperature.gpu"
	out, err := runNvidiaSMI(ctx, "--id="+id, "--query-gpu="+fields, "--format=csv,noheader,nounits")
	if err != nil {
		return deviceSnapshot{}, nil
	}
	rows, err := csv.NewReader(strings.NewReader(strings.TrimSpace(string(out)))).ReadAll()
	if err != nil || len(rows) != 1 || len(rows[0]) != 9 {
		return deviceSnapshot{}, fmt.Errorf("parse nvidia-smi device row: expected 9 fields")
	}
	r := rows[0]
	total, err := parseNvidiaMiB(r[2])
	if err != nil {
		return deviceSnapshot{}, err
	}
	used, err := parseNvidiaMiB(r[3])
	if err != nil {
		return deviceSnapshot{}, err
	}
	gpu, err := parseNvidiaRatio(r[4])
	if err != nil {
		return deviceSnapshot{}, err
	}
	memory, err := parseNvidiaRatio(r[5])
	if err != nil {
		return deviceSnapshot{}, err
	}
	clock, err := parseNvidiaFloat(r[6])
	if err != nil {
		return deviceSnapshot{}, err
	}
	power, err := parseNvidiaFloat(r[7])
	if err != nil {
		return deviceSnapshot{}, err
	}
	temp, err := parseNvidiaFloat(r[8])
	if err != nil {
		return deviceSnapshot{}, err
	}
	name, uuid := strings.TrimSpace(r[1]), strings.TrimSpace(r[0])
	return deviceSnapshot{device: DeviceSignals{ComputeUtilization: gpu, MemoryControllerUtilization: memory, MemoryClockMHz: clock, PowerWatts: power, TemperatureC: temp}, capacity: CapacitySignals{UsedBytes: used, TotalBytes: total}, provenanceDevice: name + " (" + uuid + ")", collector: "nvidia-smi", available: true}, nil
}

func parseNvidiaFloat(s string) (*float64, error) {
	if deviceMetricUnavailable(s) {
		return nil, nil
	}
	v, err := strconv.ParseFloat(strings.TrimSpace(s), 64)
	if err != nil {
		return nil, fmt.Errorf("parse nvidia-smi value %q: %w", s, err)
	}
	return &v, nil
}
func parseNvidiaRatio(s string) (*float64, error) {
	v, err := parseNvidiaFloat(s)
	if err != nil || v == nil {
		return nil, err
	}
	if *v < 0 || *v > 100 {
		return nil, fmt.Errorf("nvidia-smi percent out of range: %v", *v)
	}
	*v /= 100
	return v, nil
}
func parseNvidiaMiB(s string) (*uint64, error) {
	if deviceMetricUnavailable(s) {
		return nil, nil
	}
	v, err := strconv.ParseUint(strings.TrimSpace(s), 10, 64)
	if err != nil {
		return nil, fmt.Errorf("parse nvidia-smi MiB %q: %w", s, err)
	}
	v *= 1024 * 1024
	return &v, nil
}
