package modelperfobs

import (
	"bufio"
	"encoding/csv"
	"errors"
	"fmt"
	"io"
	"math"
	"math/big"
	"strconv"
	"strings"
	"time"
)

const NVIDIAProfileSchema = "fak-nvidia-ncu-bandwidth-profile/1"

var nvidiaProfileMetrics = []string{
	"dram__bytes_read.sum",
	"dram__bytes_write.sum",
	"gpu__time_duration.sum",
}

type NVIDIAProfileOptions struct {
	Phase             RequestPhase
	Shape             RequestShape
	Device            string
	CaptureStartedAt  time.Time
	CaptureEndedAt    time.Time
	TheoreticalGBS    *float64
	MeasuredDeviceGBS *float64
}

// NVIDIAProfileReceipt keeps the profiler evidence separate from the derived
// bandwidth sample. In particular, EngineEvidence makes explicit that raw NCU
// CSV cannot establish which serving engine launched the profiled process.
type NVIDIAProfileReceipt struct {
	Schema                    string   `json:"schema"`
	Engine                    string   `json:"engine"`
	EngineEvidence            string   `json:"engine_evidence"`
	Profiler                  string   `json:"profiler"`
	CounterSource             []string `json:"counter_source"`
	Device                    string   `json:"device"`
	DeviceEvidence            string   `json:"device_evidence"`
	ProfilerDevice            string   `json:"profiler_device,omitempty"`
	ProfilerDeviceEvidence    string   `json:"profiler_device_evidence,omitempty"`
	ProcessID                 string   `json:"process_id"`
	Process                   string   `json:"process"`
	LaunchCount               int      `json:"launch_count"`
	CumulativeReadBytes       *uint64  `json:"cumulative_read_bytes,omitempty"`
	CumulativeWriteBytes      *uint64  `json:"cumulative_write_bytes,omitempty"`
	CumulativeDurationNS      *float64 `json:"cumulative_duration_ns,omitempty"`
	AggregationScope          string   `json:"aggregation_scope"`
	CaptureStartedAt          string   `json:"capture_started_at"`
	CaptureEndedAt            string   `json:"capture_ended_at"`
	CaptureHostEvidence       string   `json:"capture_host_evidence"`
	DeviceRooflineGBS         *float64 `json:"device_roofline_gb_s,omitempty"`
	DeviceRooflineEvidence    string   `json:"device_roofline_evidence,omitempty"`
	ProfiledKernelActiveTime  bool     `json:"profiled_kernel_active_time"`
	UninstrumentedRunRequired bool     `json:"uninstrumented_run_required"`
}

type nvidiaProfileMetric struct {
	seen         bool
	available    bool
	integerValue uint64
	decimalValue float64
}

type nvidiaProfileLaunch struct {
	read     nvidiaProfileMetric
	write    nvidiaProfileMetric
	duration nvidiaProfileMetric
}

// ImportNVIDIAProfile imports the three cumulative base-unit counters needed
// to calculate true DRAM/HBM throughput. Duration is summed once per launch,
// after a complete counter triple has been established.
func ImportNVIDIAProfile(r io.Reader, o NVIDIAProfileOptions) (BandwidthCollection, error) {
	if err := validateNVIDIAProfileOptions(o); err != nil {
		return BandwidthCollection{}, err
	}
	receipt, live, available, err := parseNVIDIAProfileCSV(r, o)
	if err != nil {
		return BandwidthCollection{}, err
	}
	provenance := BandwidthProvenance{
		Source:    "nvidia-nsight-compute-raw-csv",
		Device:    receipt.Device,
		Collector: "nvidia-nsight-compute",
		SampledAt: receipt.CaptureEndedAt,
	}
	sample := BandwidthSample{
		Phase:      o.Phase,
		Shape:      o.Shape,
		Provenance: provenance,
		Rooflines: Rooflines{
			TheoreticalGBS:         cloneFloat(o.TheoreticalGBS),
			MeasuredSustainableGBS: cloneFloat(o.MeasuredDeviceGBS),
		},
		Live: live,
	}
	capture := BandwidthCapture{
		Schema:  BandwidthSchema,
		Engine:  "fak-native",
		Trigger: TriggerConfig{SymptomWindow: 1, ResourceWindow: 1, LatencyThresholdMS: 1e100, ResourceUtilization: 1},
		Samples: []BandwidthSample{sample},
	}
	report, err := AnalyzeBandwidth(capture)
	if err != nil {
		return BandwidthCollection{}, err
	}
	return BandwidthCollection{
		Schema:         BandwidthCollectionSchema,
		Engine:         "fak-native",
		Collector:      "nvidia-nsight-compute",
		Availability:   CollectorAvailability{DRAMCounters: available},
		Capture:        capture,
		Report:         report,
		ProfileReceipt: &receipt,
	}, nil
}

func validateNVIDIAProfileOptions(o NVIDIAProfileOptions) error {
	if strings.TrimSpace(o.Device) == "" {
		return errors.New("NVIDIA profile device is required")
	}
	if o.CaptureStartedAt.IsZero() || o.CaptureEndedAt.IsZero() {
		return errors.New("NVIDIA profile capture start and end times are required")
	}
	if !o.CaptureStartedAt.Before(o.CaptureEndedAt) {
		return errors.New("NVIDIA profile capture end must be after start")
	}
	return validateSample(BandwidthSample{Phase: o.Phase, Shape: o.Shape, Provenance: BandwidthProvenance{Source: "nvidia-nsight-compute-raw-csv"}})
}

func parseNVIDIAProfileCSV(r io.Reader, o NVIDIAProfileOptions) (NVIDIAProfileReceipt, LiveBandwidth, bool, error) {
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 64*1024), 4*1024*1024)
	var header map[string]int
	launches := make(map[string]*nvidiaProfileLaunch)
	var processID, processName, observedDevice, observedHost string
	for scanner.Scan() {
		line := scanner.Text()
		cr := csv.NewReader(strings.NewReader(line))
		cr.FieldsPerRecord = -1
		cr.TrimLeadingSpace = true
		record, err := cr.Read()
		if err != nil {
			if header == nil {
				continue
			}
			return NVIDIAProfileReceipt{}, LiveBandwidth{}, false, fmt.Errorf("parse Nsight Compute CSV: %w", err)
		}
		if header == nil {
			if isNVIDIAProfileHeader(record) {
				header = indexNVIDIAProfileHeader(record)
			}
			continue
		}
		if isNVIDIAProfileHeader(record) {
			return NVIDIAProfileReceipt{}, LiveBandwidth{}, false, errors.New("Nsight Compute CSV contains multiple header blocks")
		}
		if len(record) == 0 || (len(record) == 1 && (strings.TrimSpace(record[0]) == "" || strings.HasPrefix(strings.TrimSpace(record[0]), "==PROF=="))) {
			continue
		}
		metric, ok := fieldAt(record, header, "metric name")
		if !ok {
			return NVIDIAProfileReceipt{}, LiveBandwidth{}, false, errors.New("malformed Nsight Compute CSV data row")
		}
		metric = strings.TrimSpace(metric)
		if !isNVIDIAProfileMetric(metric) {
			continue
		}
		id, okID := fieldAt(record, header, "id")
		pid, okPID := fieldAt(record, header, "process id")
		proc, okProc := fieldAt(record, header, "process name")
		unit, okUnit := fieldAt(record, header, "metric unit")
		value, okValue := fieldAt(record, header, "metric value")
		if !okID || !okPID || !okProc || !okUnit || !okValue || strings.TrimSpace(id) == "" || strings.TrimSpace(pid) == "" || strings.TrimSpace(proc) == "" {
			return NVIDIAProfileReceipt{}, LiveBandwidth{}, false, fmt.Errorf("metric %s has incomplete launch identity", metric)
		}
		id, pid, proc = strings.TrimSpace(id), strings.TrimSpace(pid), strings.TrimSpace(proc)
		if processID == "" {
			processID, processName = pid, proc
		} else if processID != pid || processName != proc {
			return NVIDIAProfileReceipt{}, LiveBandwidth{}, false, fmt.Errorf("mixed-process Nsight Compute capture: %s/%s and %s/%s", processID, processName, pid, proc)
		}
		if host, ok := fieldAt(record, header, "host name"); ok {
			host = strings.TrimSpace(host)
			if observedHost == "" {
				observedHost = host
			} else if observedHost != host {
				return NVIDIAProfileReceipt{}, LiveBandwidth{}, false, errors.New("mixed-host Nsight Compute capture")
			}
		}
		if device, ok := nvidiaProfileDevice(record, header); ok {
			device = strings.TrimSpace(device)
			if device == "" {
				return NVIDIAProfileReceipt{}, LiveBandwidth{}, false, errors.New("Nsight Compute CSV has an empty device field")
			}
			if observedDevice == "" {
				observedDevice = device
			} else if observedDevice != device {
				return NVIDIAProfileReceipt{}, LiveBandwidth{}, false, fmt.Errorf("mixed-device Nsight Compute capture: %q and %q", observedDevice, device)
			}
		}
		key := pid + "\x00" + id
		launch := launches[key]
		if launch == nil {
			launch = &nvidiaProfileLaunch{}
			launches[key] = launch
		}
		var target *nvidiaProfileMetric
		switch metric {
		case nvidiaProfileMetrics[0]:
			target = &launch.read
		case nvidiaProfileMetrics[1]:
			target = &launch.write
		case nvidiaProfileMetrics[2]:
			target = &launch.duration
		}
		if target.seen {
			return NVIDIAProfileReceipt{}, LiveBandwidth{}, false, fmt.Errorf("launch %s has duplicate metric %s", id, metric)
		}
		target.seen = true
		if deviceMetricUnavailable(value) {
			continue
		}
		if !nvidiaProfileBaseUnit(metric, unit) {
			return NVIDIAProfileReceipt{}, LiveBandwidth{}, false, fmt.Errorf("metric %s must use a base unit, got %q", metric, unit)
		}
		if metric == "gpu__time_duration.sum" {
			parsed, err := parseNVIDIAProfileFloat(value)
			if err != nil {
				return NVIDIAProfileReceipt{}, LiveBandwidth{}, false, fmt.Errorf("metric %s value: %w", metric, err)
			}
			target.decimalValue = parsed
		} else {
			parsed, err := parseNVIDIAProfileUint(value)
			if err != nil {
				return NVIDIAProfileReceipt{}, LiveBandwidth{}, false, fmt.Errorf("metric %s value: %w", metric, err)
			}
			target.integerValue = parsed
		}
		target.available = true
	}
	if err := scanner.Err(); err != nil {
		return NVIDIAProfileReceipt{}, LiveBandwidth{}, false, fmt.Errorf("read Nsight Compute CSV: %w", err)
	}
	if header == nil {
		return NVIDIAProfileReceipt{}, LiveBandwidth{}, false, errors.New("Nsight Compute CSV header not found")
	}
	if len(launches) == 0 {
		return NVIDIAProfileReceipt{}, LiveBandwidth{}, false, errors.New("Nsight Compute CSV contains no required bandwidth metrics")
	}
	readAvailable, writeAvailable, durationAvailable := true, true, true
	var readBytes, writeBytes uint64
	var durationNS float64
	for key, launch := range launches {
		if !launch.read.seen || !launch.write.seen || !launch.duration.seen {
			return NVIDIAProfileReceipt{}, LiveBandwidth{}, false, fmt.Errorf("launch %q is missing a required read/write/duration metric", key[strings.IndexByte(key, 0)+1:])
		}
		var ok bool
		if !launch.read.available {
			readAvailable = false
		} else if readBytes, ok = addUint64(readBytes, launch.read.integerValue); !ok {
			return NVIDIAProfileReceipt{}, LiveBandwidth{}, false, errors.New("cumulative read bytes overflow uint64")
		}
		if !launch.write.available {
			writeAvailable = false
		} else if writeBytes, ok = addUint64(writeBytes, launch.write.integerValue); !ok {
			return NVIDIAProfileReceipt{}, LiveBandwidth{}, false, errors.New("cumulative write bytes overflow uint64")
		}
		if !launch.duration.available {
			durationAvailable = false
		} else if durationNS, ok = addNVIDIAProfileFloat(durationNS, launch.duration.decimalValue); !ok {
			return NVIDIAProfileReceipt{}, LiveBandwidth{}, false, errors.New("cumulative duration is not finite")
		}
	}
	hostEvidence := "unavailable"
	if observedHost != "" {
		hostEvidence = "single-host-nsight-csv-value-scrubbed"
	}
	receipt := NVIDIAProfileReceipt{
		Schema:                    NVIDIAProfileSchema,
		Engine:                    "fak-native",
		EngineEvidence:            "operator-asserted-not-proven-by-csv",
		Profiler:                  "nvidia-nsight-compute",
		CounterSource:             append([]string(nil), nvidiaProfileMetrics...),
		Device:                    strings.TrimSpace(o.Device),
		DeviceEvidence:            "operator-declared-capture-device",
		ProfilerDevice:            observedDevice,
		ProcessID:                 processID,
		Process:                   processName,
		LaunchCount:               len(launches),
		AggregationScope:          "sum-cumulative-bytes-over-sum-profiled-kernel-active-nanoseconds",
		CaptureStartedAt:          o.CaptureStartedAt.UTC().Format(time.RFC3339Nano),
		CaptureEndedAt:            o.CaptureEndedAt.UTC().Format(time.RFC3339Nano),
		CaptureHostEvidence:       hostEvidence,
		DeviceRooflineGBS:         cloneFloat(o.MeasuredDeviceGBS),
		ProfiledKernelActiveTime:  true,
		UninstrumentedRunRequired: true,
	}
	if observedDevice != "" {
		receipt.ProfilerDeviceEvidence = "nvidia-nsight-compute-csv"
	}
	if o.MeasuredDeviceGBS != nil {
		receipt.DeviceRooflineEvidence = "operator-supplied-matched-device-measurement"
	}
	if readAvailable {
		receipt.CumulativeReadBytes = uint64p(readBytes)
	}
	if writeAvailable {
		receipt.CumulativeWriteBytes = uint64p(writeBytes)
	}
	if durationAvailable {
		receipt.CumulativeDurationNS = &durationNS
	}
	if durationAvailable && durationNS == 0 && (readAvailable || writeAvailable) {
		return NVIDIAProfileReceipt{}, LiveBandwidth{}, false, errors.New("cumulative profiled kernel duration is zero")
	}
	var live LiveBandwidth
	if durationAvailable && durationNS > 0 {
		if readAvailable {
			readGBS := float64(readBytes) / float64(durationNS)
			live.ReadGBS = &readGBS
		}
		if writeAvailable {
			writeGBS := float64(writeBytes) / float64(durationNS)
			live.WriteGBS = &writeGBS
		}
		if live.ReadGBS != nil && live.WriteGBS != nil {
			totalGBS := *live.ReadGBS + *live.WriteGBS
			live.TotalGBS = &totalGBS
		}
	}
	available := live.ReadGBS != nil || live.WriteGBS != nil
	return receipt, live, available, nil
}

func isNVIDIAProfileHeader(record []string) bool {
	seen := make(map[string]bool)
	for _, field := range record {
		seen[strings.ToLower(strings.TrimSpace(strings.TrimPrefix(field, "\ufeff")))] = true
	}
	for _, required := range []string{"id", "process id", "process name", "metric name", "metric unit", "metric value"} {
		if !seen[required] {
			return false
		}
	}
	return true
}

func indexNVIDIAProfileHeader(record []string) map[string]int {
	header := make(map[string]int, len(record))
	for i, field := range record {
		header[strings.ToLower(strings.TrimSpace(strings.TrimPrefix(field, "\ufeff")))] = i
	}
	return header
}

func fieldAt(record []string, header map[string]int, name string) (string, bool) {
	i, ok := header[name]
	if !ok || i >= len(record) {
		return "", false
	}
	return record[i], true
}

func nvidiaProfileDevice(record []string, header map[string]int) (string, bool) {
	for _, name := range []string{"device", "device name", "device id"} {
		if value, ok := fieldAt(record, header, name); ok {
			return value, true
		}
	}
	return "", false
}

func isNVIDIAProfileMetric(metric string) bool {
	for _, want := range nvidiaProfileMetrics {
		if metric == want {
			return true
		}
	}
	return false
}

func nvidiaProfileBaseUnit(metric, unit string) bool {
	unit = strings.ToLower(strings.TrimSpace(unit))
	if metric == "gpu__time_duration.sum" {
		return unit == "ns" || unit == "nsecond" || unit == "nanosecond"
	}
	return unit == "byte" || unit == "bytes"
}

func parseNVIDIAProfileUint(value string) (uint64, error) {
	clean := strings.ReplaceAll(strings.TrimSpace(value), ",", "")
	v, ok := new(big.Rat).SetString(clean)
	if !ok || !v.IsInt() || v.Sign() < 0 || v.Num().BitLen() > 64 {
		return 0, fmt.Errorf("base-unit byte counter %q is not a non-negative uint64", value)
	}
	return v.Num().Uint64(), nil
}

func parseNVIDIAProfileFloat(value string) (float64, error) {
	clean := strings.ReplaceAll(strings.TrimSpace(value), ",", "")
	v, err := strconv.ParseFloat(clean, 64)
	if err != nil || math.IsNaN(v) || math.IsInf(v, 0) || v < 0 {
		return 0, fmt.Errorf("base-unit duration %q is not a finite non-negative number", value)
	}
	return v, nil
}

func addNVIDIAProfileFloat(a, b float64) (float64, bool) {
	v := a + b
	return v, !math.IsNaN(v) && !math.IsInf(v, 0)
}

func addUint64(a, b uint64) (uint64, bool) {
	if ^uint64(0)-a < b {
		return 0, false
	}
	return a + b, true
}

func uint64p(v uint64) *uint64 { return &v }
