package modelperfobs

import (
	"bytes"
	"encoding/csv"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"strconv"
	"strings"
	"time"
)

const (
	HostControllerImportSchema   = "fak-host-controller-counter-import/1"
	HostControllerArtifactSchema = "fak-host-controller-counters/1"

	// MinHostControllerRunningRatio rejects materially multiplexed generic
	// observations. Perf imports are stricter and require 100% because perf's
	// output does not say whether counter-value was provider-scaled.
	MinHostControllerRunningRatio = 0.90
)

type HostCounterImportFormat string

const (
	HostCounterFormatAuto        HostCounterImportFormat = "auto"
	HostCounterFormatGenericJSON HostCounterImportFormat = "generic-json"
	HostCounterFormatPerfJSON    HostCounterImportFormat = "perf-json"
	HostCounterFormatPerfCSV     HostCounterImportFormat = "perf-csv"
)

type HostControllerScope struct {
	Kind string `json:"kind"`
	ID   string `json:"id,omitempty"`
}

type HostControllerImportOptions struct {
	Format            HostCounterImportFormat
	Provider          string
	Scope             HostControllerScope
	CaptureStartedAt  time.Time
	CaptureEndedAt    time.Time
	PerfBytesPerEvent *uint64
	Phase             RequestPhase
	Shape             RequestShape
	TheoreticalGBS    *float64
	MeasuredHostGBS   *float64
}

// HostControllerCounterArtifact retains the raw provider values beside every
// derivation. Snapshot values are present for generic cumulative counters;
// perf stat's reported interval value is retained in DeltaRawValue. For perf,
// EventRuntime is time_running. Perf reports only a rounded running percentage,
// so TimeEnabledNS is deliberately omitted rather than fabricated from it.
type HostControllerCounterArtifact struct {
	Direction        string  `json:"direction"`
	RawEvent         string  `json:"raw_event"`
	Unit             string  `json:"unit"`
	StartRawValue    *uint64 `json:"start_raw_value,omitempty"`
	EndRawValue      *uint64 `json:"end_raw_value,omitempty"`
	DeltaRawValue    uint64  `json:"delta_raw_value"`
	CounterWidthBits *uint8  `json:"counter_width_bits,omitempty"`
	ResetObserved    *bool   `json:"reset_observed,omitempty"`
	Wrapped          bool    `json:"wrapped,omitempty"`
	BytesPerEvent    *uint64 `json:"bytes_per_event,omitempty"`
	TrafficBytes     uint64  `json:"traffic_bytes"`
	TimeEnabledNS    *uint64 `json:"time_enabled_ns,omitempty"`
	TimeRunningNS    *uint64 `json:"time_running_ns,omitempty"`
	PerfPMUCount     *uint64 `json:"perf_pmu_count,omitempty"`
	RunningRatio     float64 `json:"running_ratio"`
	ByteProvenance   string  `json:"byte_provenance"`
}

type HostControllerArtifact struct {
	Schema           string                          `json:"schema"`
	Provider         string                          `json:"provider"`
	ImportFormat     HostCounterImportFormat         `json:"import_format"`
	Scope            HostControllerScope             `json:"scope"`
	CaptureStartedAt string                          `json:"capture_started_at"`
	CaptureEndedAt   string                          `json:"capture_ended_at"`
	IntervalNS       int64                           `json:"interval_ns"`
	RunningRatio     float64                         `json:"running_ratio"`
	Counters         []HostControllerCounterArtifact `json:"counters"`
	ReadBytes        *uint64                         `json:"read_bytes,omitempty"`
	WriteBytes       *uint64                         `json:"write_bytes,omitempty"`
	TotalBytes       *uint64                         `json:"total_bytes,omitempty"`
}

type genericHostCounterImport struct {
	Schema   string               `json:"schema"`
	Provider string               `json:"provider"`
	Scope    HostControllerScope  `json:"scope"`
	Capture  genericHostCapture   `json:"capture"`
	Counters []genericHostCounter `json:"counters"`
}

type genericHostCapture struct {
	StartedAt string `json:"started_at"`
	EndedAt   string `json:"ended_at"`
}

type genericHostCounter struct {
	Direction        string                `json:"direction"`
	Event            string                `json:"event"`
	Unit             string                `json:"unit"`
	BytesPerEvent    *uint64               `json:"bytes_per_event,omitempty"`
	CounterWidthBits *uint8                `json:"counter_width_bits,omitempty"`
	Scope            HostControllerScope   `json:"scope"`
	TimeEnabledNS    *uint64               `json:"time_enabled_ns,omitempty"`
	TimeRunningNS    *uint64               `json:"time_running_ns,omitempty"`
	RunningRatio     *float64              `json:"running_ratio,omitempty"`
	Snapshots        []genericHostSnapshot `json:"snapshots"`
}

type genericHostSnapshot struct {
	CapturedAt    string `json:"captured_at"`
	Value         uint64 `json:"value"`
	ResetObserved *bool  `json:"reset_observed,omitempty"`
}

// perfStatRecord follows the documented perf stat JSON field spellings and
// adds narrowly scoped metadata fields needed to make a DRAM interpretation
// explicit. Unknown fields are rejected rather than guessed.
type perfStatRecord struct {
	Interval        json.RawMessage `json:"interval,omitempty"`
	CPU             json.RawMessage `json:"cpu,omitempty"`
	Core            json.RawMessage `json:"core,omitempty"`
	Cache           json.RawMessage `json:"cache,omitempty"`
	Cluster         json.RawMessage `json:"cluster,omitempty"`
	Die             json.RawMessage `json:"die,omitempty"`
	Socket          json.RawMessage `json:"socket,omitempty"`
	Node            json.RawMessage `json:"node,omitempty"`
	Thread          json.RawMessage `json:"thread,omitempty"`
	Cgroup          string          `json:"cgroup,omitempty"`
	Counters        json.RawMessage `json:"counters,omitempty"`
	CounterValue    json.RawMessage `json:"counter-value"`
	Unit            string          `json:"unit"`
	Event           string          `json:"event"`
	Variance        json.RawMessage `json:"variance,omitempty"`
	EventRuntime    json.RawMessage `json:"event-runtime,omitempty"`
	PercentRunning  json.RawMessage `json:"pcnt-running"`
	MetricValue     json.RawMessage `json:"metric-value,omitempty"`
	MetricUnit      string          `json:"metric-unit,omitempty"`
	MetricThreshold string          `json:"metric-threshold,omitempty"`
	MetricGroup     string          `json:"metricgroup,omitempty"`
	Direction       string          `json:"direction,omitempty"`
	Scope           string          `json:"scope,omitempty"`
	ScopeID         string          `json:"scope-id,omitempty"`
	Provider        string          `json:"provider,omitempty"`
	BytesPerEvent   *uint64         `json:"bytes-per-event,omitempty"`
}

// ImportHostControllerCounters turns genuine host memory-controller counter
// evidence into one host-memory bandwidth sample and an artifact that preserves
// the raw event names, integer values, scheduling metadata, and capture bounds.
//
// Pinned field borrow:
//   - Intel PCM a211c95843bbe161bdaa8e5feb58cfcf6af1570d, pcm-memory.cpp
//     lines 965-1053 uses two uncore snapshots and an explicit 64-byte CAS
//     conversion. We borrow those principles, not any PCM CSV layout:
//     https://github.com/intel/pcm/blob/a211c95843bbe161bdaa8e5feb58cfcf6af1570d/src/pcm-memory.cpp#L965-L1053
//   - Linux perf 3ba13f5e7180c034b0a1ef7e052fb780856b134e documents the
//     perf stat JSON keys and CSV field order accepted here:
//     https://github.com/torvalds/linux/blob/3ba13f5e7180c034b0a1ef7e052fb780856b134e/tools/perf/Documentation/perf-stat.txt#L568-L611
func ImportHostControllerCounters(r io.Reader, o HostControllerImportOptions) (BandwidthCollection, error) {
	if r == nil {
		return BandwidthCollection{}, errors.New("host controller counter input is nil")
	}
	o.Provider = strings.TrimSpace(o.Provider)
	o.Scope.Kind = strings.TrimSpace(o.Scope.Kind)
	o.Scope.ID = strings.TrimSpace(o.Scope.ID)
	if err := validateHostCounterOptions(o); err != nil {
		return BandwidthCollection{}, err
	}
	const maxHostCounterImportBytes = 16 << 20
	data, err := io.ReadAll(io.LimitReader(r, maxHostCounterImportBytes+1))
	if err != nil {
		return BandwidthCollection{}, err
	}
	if len(data) > maxHostCounterImportBytes {
		return BandwidthCollection{}, fmt.Errorf("host controller counter input exceeds %d bytes", maxHostCounterImportBytes)
	}
	if len(bytes.TrimSpace(data)) == 0 {
		return BandwidthCollection{}, errors.New("host controller counter input is empty")
	}
	format, err := detectHostCounterFormat(data, o.Format)
	if err != nil {
		return BandwidthCollection{}, err
	}
	if format == HostCounterFormatGenericJSON && o.PerfBytesPerEvent != nil {
		return BandwidthCollection{}, errors.New("perf bytes-per-event conversion cannot be applied to generic counters")
	}
	var artifact HostControllerArtifact
	switch format {
	case HostCounterFormatGenericJSON:
		artifact, err = parseGenericHostCounters(data, o)
	case HostCounterFormatPerfJSON:
		artifact, err = parsePerfStatJSON(data, o)
	case HostCounterFormatPerfCSV:
		artifact, err = parsePerfStatCSV(data, o)
	default:
		err = fmt.Errorf("unsupported host counter import format %q", format)
	}
	if err != nil {
		return BandwidthCollection{}, err
	}
	return hostControllerCollection(artifact, o)
}

func validateHostCounterOptions(o HostControllerImportOptions) error {
	if strings.TrimSpace(o.Provider) == "" {
		return errors.New("host counter provider is required")
	}
	if err := validateHostControllerScope(o.Scope); err != nil {
		return err
	}
	if o.PerfBytesPerEvent != nil && *o.PerfBytesPerEvent == 0 {
		return errors.New("perf bytes per event must be positive")
	}
	if o.Scope.Kind != "system" && (o.TheoreticalGBS != nil || o.MeasuredHostGBS != nil) {
		return errors.New("host memory rooflines require system-aggregate host counter scope")
	}
	for name, value := range map[string]*float64{
		"theoretical host memory roofline": o.TheoreticalGBS,
		"measured host memory roofline":    o.MeasuredHostGBS,
	} {
		if value != nil && (*value <= 0 || math.IsNaN(*value) || math.IsInf(*value, 0)) {
			return fmt.Errorf("%s must be positive and finite", name)
		}
	}
	if o.Format == "" {
		o.Format = HostCounterFormatAuto
	}
	switch o.Format {
	case "", HostCounterFormatAuto, HostCounterFormatGenericJSON, HostCounterFormatPerfJSON, HostCounterFormatPerfCSV:
	default:
		return fmt.Errorf("unsupported host counter import format %q", o.Format)
	}
	return validateSample(BandwidthSample{Phase: o.Phase, Shape: o.Shape, Provenance: BandwidthProvenance{Source: "host-controller-counters"}})
}

func validateHostControllerScope(scope HostControllerScope) error {
	scope.Kind = strings.TrimSpace(scope.Kind)
	scope.ID = strings.TrimSpace(scope.ID)
	switch scope.Kind {
	case "system":
		if scope.ID != "" {
			return errors.New("system host counter scope must not have an id")
		}
	case "socket", "controller":
		if scope.ID == "" {
			return fmt.Errorf("%s host counter scope requires an id", scope.Kind)
		}
	default:
		return fmt.Errorf("host counter scope must be system, socket, or controller, got %q", scope.Kind)
	}
	return nil
}

func detectHostCounterFormat(data []byte, requested HostCounterImportFormat) (HostCounterImportFormat, error) {
	if requested != "" && requested != HostCounterFormatAuto {
		return requested, nil
	}
	trimmed := bytes.TrimSpace(data)
	if trimmed[0] == '[' {
		return HostCounterFormatPerfJSON, nil
	}
	if trimmed[0] == '{' {
		var envelope struct {
			Schema string `json:"schema"`
			Event  string `json:"event"`
		}
		if err := json.Unmarshal(trimmed, &envelope); err == nil && envelope.Schema == HostControllerImportSchema {
			return HostCounterFormatGenericJSON, nil
		}
		if envelope.Event != "" || bytes.Contains(trimmed, []byte(`"counter-value"`)) {
			return HostCounterFormatPerfJSON, nil
		}
		return "", errors.New("JSON host counter input has neither the generic schema nor perf stat fields")
	}
	return HostCounterFormatPerfCSV, nil
}

func parseGenericHostCounters(data []byte, o HostControllerImportOptions) (HostControllerArtifact, error) {
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	var input genericHostCounterImport
	if err := dec.Decode(&input); err != nil {
		return HostControllerArtifact{}, fmt.Errorf("parse generic host counters: %w", err)
	}
	if err := requireJSONEOF(dec); err != nil {
		return HostControllerArtifact{}, err
	}
	if input.Schema != HostControllerImportSchema {
		return HostControllerArtifact{}, fmt.Errorf("generic host counter schema must be %q", HostControllerImportSchema)
	}
	if input.Provider == "" || input.Provider != o.Provider {
		return HostControllerArtifact{}, fmt.Errorf("generic host counter provider %q does not match declared provider %q", input.Provider, o.Provider)
	}
	if input.Scope != o.Scope {
		return HostControllerArtifact{}, fmt.Errorf("generic host counter scope %+v does not match declared scope %+v", input.Scope, o.Scope)
	}
	started, err := time.Parse(time.RFC3339Nano, input.Capture.StartedAt)
	if err != nil {
		return HostControllerArtifact{}, fmt.Errorf("generic capture start must be RFC3339: %w", err)
	}
	ended, err := time.Parse(time.RFC3339Nano, input.Capture.EndedAt)
	if err != nil {
		return HostControllerArtifact{}, fmt.Errorf("generic capture end must be RFC3339: %w", err)
	}
	if !o.CaptureStartedAt.IsZero() && !o.CaptureStartedAt.Equal(started) {
		return HostControllerArtifact{}, errors.New("generic capture start conflicts with CLI capture metadata")
	}
	if !o.CaptureEndedAt.IsZero() && !o.CaptureEndedAt.Equal(ended) {
		return HostControllerArtifact{}, errors.New("generic capture end conflicts with CLI capture metadata")
	}
	if err := validateCaptureInterval(started, ended); err != nil {
		return HostControllerArtifact{}, err
	}
	artifact := newHostControllerArtifact(o, HostCounterFormatGenericJSON, started, ended)
	for _, counter := range input.Counters {
		direction, relevant, directionErr := hostCounterDirection(counter.Event, counter.Direction)
		if directionErr != nil {
			return HostControllerArtifact{}, directionErr
		}
		if !relevant || direction != counter.Direction {
			return HostControllerArtifact{}, fmt.Errorf("event %q is not an explicit host DRAM %s counter", counter.Event, counter.Direction)
		}
		if counter.Scope != o.Scope {
			return HostControllerArtifact{}, fmt.Errorf("mixed host counter scopes: %+v and %+v", o.Scope, counter.Scope)
		}
		if len(counter.Snapshots) != 2 {
			return HostControllerArtifact{}, fmt.Errorf("%s counter %q requires exactly two snapshots", counter.Direction, counter.Event)
		}
		if counter.Snapshots[0].CapturedAt != input.Capture.StartedAt || counter.Snapshots[1].CapturedAt != input.Capture.EndedAt {
			return HostControllerArtifact{}, fmt.Errorf("counter %q snapshots must match capture start and end", counter.Event)
		}
		if counter.Snapshots[1].ResetObserved == nil {
			return HostControllerArtifact{}, fmt.Errorf("counter %q reset state is ambiguous", counter.Event)
		}
		if *counter.Snapshots[1].ResetObserved {
			return HostControllerArtifact{}, fmt.Errorf("counter %q reset during capture", counter.Event)
		}
		ratio, err := hostCounterRunningRatio(counter.TimeEnabledNS, counter.TimeRunningNS, counter.RunningRatio)
		if err != nil {
			return HostControllerArtifact{}, fmt.Errorf("counter %q: %w", counter.Event, err)
		}
		if err := validateCounterTiming(counter.TimeEnabledNS, counter.TimeRunningNS, ratio, ended.Sub(started)); err != nil {
			return HostControllerArtifact{}, fmt.Errorf("counter %q: %w", counter.Event, err)
		}
		delta, wrapped, err := cumulativeCounterDelta(counter.Snapshots[0].Value, counter.Snapshots[1].Value, counter.CounterWidthBits)
		if err != nil {
			return HostControllerArtifact{}, fmt.Errorf("counter %q: %w", counter.Event, err)
		}
		traffic, provenance, err := convertHostCounterBytes(delta, counter.Unit, counter.BytesPerEvent)
		if err != nil {
			return HostControllerArtifact{}, fmt.Errorf("counter %q: %w", counter.Event, err)
		}
		start, end, reset := counter.Snapshots[0].Value, counter.Snapshots[1].Value, *counter.Snapshots[1].ResetObserved
		artifact.Counters = append(artifact.Counters, HostControllerCounterArtifact{
			Direction: direction, RawEvent: counter.Event, Unit: counter.Unit,
			StartRawValue: &start, EndRawValue: &end, DeltaRawValue: delta,
			CounterWidthBits: counter.CounterWidthBits, ResetObserved: &reset, Wrapped: wrapped,
			BytesPerEvent: counter.BytesPerEvent, TrafficBytes: traffic,
			TimeEnabledNS: counter.TimeEnabledNS, TimeRunningNS: counter.TimeRunningNS,
			RunningRatio: ratio, ByteProvenance: provenance,
		})
	}
	if err := finishHostControllerArtifact(&artifact); err != nil {
		return HostControllerArtifact{}, err
	}
	return artifact, nil
}

func parsePerfStatJSON(data []byte, o HostControllerImportOptions) (HostControllerArtifact, error) {
	if o.CaptureStartedAt.IsZero() || o.CaptureEndedAt.IsZero() {
		return HostControllerArtifact{}, errors.New("perf stat imports require capture start and end metadata")
	}
	if err := validateCaptureInterval(o.CaptureStartedAt, o.CaptureEndedAt); err != nil {
		return HostControllerArtifact{}, err
	}
	records, err := decodePerfStatRecords(data)
	if err != nil {
		return HostControllerArtifact{}, err
	}
	artifact := newHostControllerArtifact(o, HostCounterFormatPerfJSON, o.CaptureStartedAt, o.CaptureEndedAt)
	for _, record := range records {
		counter, relevant, err := perfRecordCounter(record, o)
		if err != nil {
			return HostControllerArtifact{}, err
		}
		if relevant {
			artifact.Counters = append(artifact.Counters, counter)
		}
	}
	if err := finishHostControllerArtifact(&artifact); err != nil {
		return HostControllerArtifact{}, err
	}
	return artifact, nil
}

func decodePerfStatRecords(data []byte) ([]perfStatRecord, error) {
	trimmed := bytes.TrimSpace(data)
	if trimmed[0] == '[' {
		dec := json.NewDecoder(bytes.NewReader(trimmed))
		dec.DisallowUnknownFields()
		var records []perfStatRecord
		if err := dec.Decode(&records); err != nil {
			return nil, fmt.Errorf("parse perf stat JSON: %w", err)
		}
		if err := requireJSONEOF(dec); err != nil {
			return nil, err
		}
		return records, nil
	}
	dec := json.NewDecoder(bytes.NewReader(trimmed))
	dec.DisallowUnknownFields()
	var records []perfStatRecord
	for {
		var record perfStatRecord
		if err := dec.Decode(&record); err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			return nil, fmt.Errorf("parse perf stat JSON: %w", err)
		}
		records = append(records, record)
	}
	return records, nil
}

func perfRecordCounter(record perfStatRecord, o HostControllerImportOptions) (HostControllerCounterArtifact, bool, error) {
	event := strings.TrimSpace(record.Event)
	direction, relevant, err := hostCounterDirection(event, record.Direction)
	if err != nil || !relevant {
		return HostControllerCounterArtifact{}, relevant, err
	}
	if record.Provider != "" && record.Provider != o.Provider {
		return HostControllerCounterArtifact{}, true, fmt.Errorf("perf record provider %q does not match declared provider %q", record.Provider, o.Provider)
	}
	if err := validatePerfRecordScope(record, o.Scope); err != nil {
		return HostControllerCounterArtifact{}, true, err
	}
	pmuCount, err := perfRecordPMUCount(record, o.Scope)
	if err != nil {
		return HostControllerCounterArtifact{}, true, fmt.Errorf("perf event %q: %w", event, err)
	}
	value, err := rawUint64(record.CounterValue, "counter-value")
	if err != nil {
		return HostControllerCounterArtifact{}, true, fmt.Errorf("perf event %q: %w", event, err)
	}
	ratioPercent, err := rawFloat64(record.PercentRunning, "pcnt-running")
	if err != nil {
		return HostControllerCounterArtifact{}, true, fmt.Errorf("perf event %q: %w", event, err)
	}
	ratio := ratioPercent / 100
	if err := validatePerfRunningRatio(ratio); err != nil {
		return HostControllerCounterArtifact{}, true, fmt.Errorf("perf event %q: %w", event, err)
	}
	bpe := record.BytesPerEvent
	if bpe == nil {
		bpe = o.PerfBytesPerEvent
	} else if o.PerfBytesPerEvent != nil && *bpe != *o.PerfBytesPerEvent {
		return HostControllerCounterArtifact{}, true, fmt.Errorf("perf event %q bytes-per-event conflicts with CLI conversion", event)
	}
	traffic, provenance, err := convertHostCounterBytes(value, record.Unit, bpe)
	if err != nil {
		return HostControllerCounterArtifact{}, true, fmt.Errorf("perf event %q: %w", event, err)
	}
	runningValue, err := rawUint64(record.EventRuntime, "event-runtime")
	if err != nil {
		return HostControllerCounterArtifact{}, true, fmt.Errorf("perf event %q: %w", event, err)
	}
	if runningValue == 0 {
		return HostControllerCounterArtifact{}, true, fmt.Errorf("perf event %q: event-runtime is unavailable", event)
	}
	if err := validatePerfCounterTiming(runningValue, o.CaptureEndedAt.Sub(o.CaptureStartedAt), pmuCount, o.Scope); err != nil {
		return HostControllerCounterArtifact{}, true, fmt.Errorf("perf event %q: %w", event, err)
	}
	return HostControllerCounterArtifact{
		Direction: direction, RawEvent: event, Unit: record.Unit, DeltaRawValue: value,
		BytesPerEvent: bpe, TrafficBytes: traffic, TimeRunningNS: &runningValue, PerfPMUCount: pmuCount,
		RunningRatio: ratio, ByteProvenance: provenance,
	}, true, nil
}

func parsePerfStatCSV(data []byte, o HostControllerImportOptions) (HostControllerArtifact, error) {
	if o.CaptureStartedAt.IsZero() || o.CaptureEndedAt.IsZero() {
		return HostControllerArtifact{}, errors.New("perf stat imports require capture start and end metadata")
	}
	if err := validateCaptureInterval(o.CaptureStartedAt, o.CaptureEndedAt); err != nil {
		return HostControllerArtifact{}, err
	}
	artifact := newHostControllerArtifact(o, HostCounterFormatPerfCSV, o.CaptureStartedAt, o.CaptureEndedAt)
	lines := strings.Split(strings.ReplaceAll(string(data), "\r\n", "\n"), "\n")
	for lineNumber, line := range lines {
		if strings.TrimSpace(line) == "" || strings.HasPrefix(strings.TrimSpace(line), "#") {
			continue
		}
		delimiter := ','
		if strings.Count(line, ";") > strings.Count(line, ",") {
			delimiter = ';'
		}
		reader := csv.NewReader(strings.NewReader(line))
		reader.Comma = delimiter
		reader.FieldsPerRecord = -1
		fields, err := reader.Read()
		if err != nil {
			return HostControllerArtifact{}, fmt.Errorf("parse perf stat CSV line %d: %w", lineNumber+1, err)
		}
		counter, relevant, err := perfCSVRecordCounter(fields, o)
		if err != nil {
			return HostControllerArtifact{}, fmt.Errorf("perf stat CSV line %d: %w", lineNumber+1, err)
		}
		if relevant {
			artifact.Counters = append(artifact.Counters, counter)
		}
	}
	if err := finishHostControllerArtifact(&artifact); err != nil {
		return HostControllerArtifact{}, err
	}
	return artifact, nil
}

func perfCSVRecordCounter(fields []string, o HostControllerImportOptions) (HostControllerCounterArtifact, bool, error) {
	for index, candidate := range fields {
		event := strings.TrimSpace(candidate)
		direction, relevant, err := hostCounterDirection(event, "")
		if err != nil {
			return HostControllerCounterArtifact{}, true, err
		}
		if !relevant {
			continue
		}
		if index < 2 || len(fields) <= index+2 {
			return HostControllerCounterArtifact{}, true, errors.New("DRAM counter row does not have perf stat value/unit/event/runtime/percentage fields")
		}
		pmuCount, err := validatePerfCSVScope(fields[:index-2], event, o.Scope)
		if err != nil {
			return HostControllerCounterArtifact{}, true, err
		}
		value, err := parseUint64Text(fields[index-2], "counter value")
		if err != nil {
			return HostControllerCounterArtifact{}, true, err
		}
		runtimeNS, err := parseUint64Text(fields[index+1], "event runtime")
		if err != nil {
			return HostControllerCounterArtifact{}, true, err
		}
		if runtimeNS == 0 {
			return HostControllerCounterArtifact{}, true, errors.New("event runtime is unavailable")
		}
		percent, err := parseFloatText(fields[index+2], "percentage running")
		if err != nil {
			return HostControllerCounterArtifact{}, true, err
		}
		ratio := percent / 100
		if err := validatePerfRunningRatio(ratio); err != nil {
			return HostControllerCounterArtifact{}, true, err
		}
		if err := validatePerfCounterTiming(runtimeNS, o.CaptureEndedAt.Sub(o.CaptureStartedAt), pmuCount, o.Scope); err != nil {
			return HostControllerCounterArtifact{}, true, err
		}
		unit := strings.TrimSpace(fields[index-1])
		traffic, provenance, err := convertHostCounterBytes(value, unit, o.PerfBytesPerEvent)
		if err != nil {
			return HostControllerCounterArtifact{}, true, err
		}
		return HostControllerCounterArtifact{
			Direction: direction, RawEvent: event, Unit: unit,
			DeltaRawValue: value, BytesPerEvent: o.PerfBytesPerEvent, TrafficBytes: traffic,
			TimeRunningNS: &runtimeNS, PerfPMUCount: pmuCount, RunningRatio: ratio, ByteProvenance: provenance,
		}, true, nil
	}
	return HostControllerCounterArtifact{}, false, nil
}

func validatePerfRecordScope(record perfStatRecord, declared HostControllerScope) error {
	if declared.Kind == "socket" && strings.TrimSpace(record.Scope) == "" && !rawJSONValuePresent(record.Socket) {
		return errors.New("socket-scoped perf JSON requires per-socket metadata")
	}
	if record.Scope != "" || record.ScopeID != "" {
		actual := HostControllerScope{Kind: strings.TrimSpace(record.Scope), ID: strings.TrimSpace(record.ScopeID)}
		if !hostControllerScopesEqual(actual, declared) {
			return fmt.Errorf("mixed host counter scopes: declared %+v, record %+v", declared, actual)
		}
	}
	if strings.TrimSpace(record.Cgroup) != "" {
		return fmt.Errorf("perf cgroup scope %q is not a system/socket/controller DRAM capture", record.Cgroup)
	}
	aggregates := []struct {
		kind string
		raw  json.RawMessage
	}{
		{kind: "cpu", raw: record.CPU}, {kind: "core", raw: record.Core}, {kind: "cache", raw: record.Cache},
		{kind: "cluster", raw: record.Cluster}, {kind: "die", raw: record.Die}, {kind: "socket", raw: record.Socket},
		{kind: "node", raw: record.Node}, {kind: "thread", raw: record.Thread},
	}
	for _, aggregate := range aggregates {
		kind, raw := aggregate.kind, aggregate.raw
		if len(raw) == 0 || string(raw) == "null" || string(raw) == `""` {
			continue
		}
		id, err := rawString(raw)
		if err != nil {
			return fmt.Errorf("perf %s scope: %w", kind, err)
		}
		if kind != declared.Kind || !perfScopeIDsEqual(kind, id, declared.ID) {
			return fmt.Errorf("mixed host counter scopes: declared %+v, perf %s %q", declared, kind, id)
		}
	}
	if declared.Kind == "controller" && !eventNamesController(record.Event, declared.ID) {
		return fmt.Errorf("controller-scoped perf event %q does not identify declared controller %q", record.Event, declared.ID)
	}
	return nil
}

func rawJSONValuePresent(raw json.RawMessage) bool {
	trimmed := strings.TrimSpace(string(raw))
	return trimmed != "" && trimmed != "null" && trimmed != `""`
}

func perfRecordPMUCount(record perfStatRecord, scope HostControllerScope) (*uint64, error) {
	if !rawJSONValuePresent(record.Counters) {
		if scope.Kind == "socket" {
			return nil, errors.New("socket-scoped perf JSON requires aggregate counter count metadata")
		}
		return nil, nil
	}
	count, err := rawUint64(record.Counters, "counters")
	if err != nil {
		return nil, err
	}
	if count == 0 {
		return nil, errors.New("aggregate counter count must be positive")
	}
	return &count, nil
}

func validatePerfCSVScope(prefix []string, event string, declared HostControllerScope) (*uint64, error) {
	for i := range prefix {
		prefix[i] = strings.TrimSpace(prefix[i])
	}
	switch declared.Kind {
	case "system":
		if len(prefix) != 0 {
			return nil, fmt.Errorf("mixed host counter scopes: system perf CSV row has aggregation fields %q", prefix)
		}
	case "socket":
		if len(prefix) != 2 {
			return nil, errors.New("socket-scoped perf CSV requires socket identifier and aggregate counter count fields")
		}
		if !perfScopeIDsEqual("socket", prefix[0], declared.ID) {
			return nil, fmt.Errorf("mixed host counter scopes: declared socket %q, perf socket %q", declared.ID, prefix[0])
		}
		count, err := parseUint64Text(prefix[1], "aggregate counter count")
		if err != nil || count == 0 {
			if err == nil {
				err = errors.New("aggregate counter count must be positive")
			}
			return nil, err
		}
		return &count, nil
	case "controller":
		if len(prefix) != 0 {
			return nil, fmt.Errorf("mixed host counter scopes: controller perf CSV row has aggregation fields %q", prefix)
		}
		if !eventNamesController(event, declared.ID) {
			return nil, fmt.Errorf("controller-scoped perf event %q does not identify declared controller %q", event, declared.ID)
		}
	}
	return nil, nil
}

func hostControllerScopesEqual(left, right HostControllerScope) bool {
	return left.Kind == right.Kind && perfScopeIDsEqual(left.Kind, left.ID, right.ID)
}

func perfScopeIDsEqual(kind, left, right string) bool {
	left = strings.TrimSpace(left)
	right = strings.TrimSpace(right)
	if kind == "socket" {
		left = strings.TrimPrefix(strings.ToUpper(left), "S")
		right = strings.TrimPrefix(strings.ToUpper(right), "S")
	}
	return left == right
}

func eventNamesController(event, id string) bool {
	event = normalizeEventIdentifier(event)
	id = normalizeEventIdentifier(id)
	return id != "" && strings.Contains(event, id)
}

func normalizeEventIdentifier(value string) string {
	var b strings.Builder
	for _, r := range strings.ToLower(value) {
		if r >= 'a' && r <= 'z' || r >= '0' && r <= '9' {
			b.WriteRune(r)
		}
	}
	return b.String()
}

func newHostControllerArtifact(o HostControllerImportOptions, format HostCounterImportFormat, started, ended time.Time) HostControllerArtifact {
	return HostControllerArtifact{
		Schema: HostControllerArtifactSchema, Provider: o.Provider, ImportFormat: format, Scope: o.Scope,
		CaptureStartedAt: started.UTC().Format(time.RFC3339Nano), CaptureEndedAt: ended.UTC().Format(time.RFC3339Nano),
		IntervalNS: ended.Sub(started).Nanoseconds(), Counters: make([]HostControllerCounterArtifact, 0, 2),
	}
}

func finishHostControllerArtifact(artifact *HostControllerArtifact) error {
	if artifact == nil {
		return errors.New("host controller artifact is nil")
	}
	var read, write *HostControllerCounterArtifact
	for i := range artifact.Counters {
		counter := &artifact.Counters[i]
		if strings.TrimSpace(counter.RawEvent) == "" {
			return errors.New("host controller raw event name is required")
		}
		switch counter.Direction {
		case "read":
			if read != nil {
				return errors.New("host controller capture has duplicate read counters")
			}
			read = counter
		case "write":
			if write != nil {
				return errors.New("host controller capture has duplicate write counters")
			}
			write = counter
		default:
			return fmt.Errorf("host controller counter direction must be read or write, got %q", counter.Direction)
		}
	}
	if read == nil || write == nil {
		return errors.New("host controller capture requires one read/write counter pair")
	}
	if read.ByteProvenance != write.ByteProvenance {
		return errors.New("host controller read/write pair has mixed direct-byte and converted-event provenance")
	}
	if (read.PerfPMUCount == nil) != (write.PerfPMUCount == nil) ||
		(read.PerfPMUCount != nil && *read.PerfPMUCount != *write.PerfPMUCount) {
		return errors.New("host controller read/write pair has mixed perf aggregate counter counts")
	}
	artifact.Counters = []HostControllerCounterArtifact{*read, *write}
	artifact.RunningRatio = math.Min(read.RunningRatio, write.RunningRatio)
	readBytes, writeBytes := read.TrafficBytes, write.TrafficBytes
	if readBytes > math.MaxUint64-writeBytes {
		return errors.New("host controller total traffic bytes overflow")
	}
	total := readBytes + writeBytes
	artifact.ReadBytes, artifact.WriteBytes, artifact.TotalBytes = &readBytes, &writeBytes, &total
	return nil
}

func hostControllerCollection(artifact HostControllerArtifact, o HostControllerImportOptions) (BandwidthCollection, error) {
	seconds := float64(artifact.IntervalNS) / float64(time.Second)
	read := float64(*artifact.ReadBytes) / seconds / 1e9
	write := float64(*artifact.WriteBytes) / seconds / 1e9
	total := read + write
	byteSource := "host-controller-direct-bytes"
	if artifact.Counters[0].ByteProvenance == "converted-events" {
		byteSource = "host-controller-converted-events"
	}
	rooflines := Rooflines{}
	if artifact.Scope.Kind == "system" {
		rooflines.TheoreticalGBS = cloneFloat(o.TheoreticalGBS)
		rooflines.MeasuredSustainableGBS = cloneFloat(o.MeasuredHostGBS)
	}
	sample := BandwidthSample{
		Phase: o.Phase, Shape: o.Shape,
		Provenance: BandwidthProvenance{Source: byteSource, Device: "host-memory", Collector: artifact.Provider, SampledAt: artifact.CaptureEndedAt},
		Rooflines:  rooflines,
		Live:       LiveBandwidth{ReadGBS: &read, WriteGBS: &write, TotalGBS: &total},
	}
	capture := BandwidthCapture{
		Schema: BandwidthSchema, Engine: "fak-native",
		Trigger: TriggerConfig{SymptomWindow: 1, ResourceWindow: 1, LatencyThresholdMS: 1e100, ResourceUtilization: 1},
		Samples: []BandwidthSample{sample},
	}
	report, err := AnalyzeBandwidth(capture)
	if err != nil {
		return BandwidthCollection{}, err
	}
	intervalMS := artifact.IntervalNS / int64(time.Millisecond)
	if artifact.IntervalNS%int64(time.Millisecond) != 0 {
		intervalMS++
	}
	return BandwidthCollection{
		Schema: BandwidthCollectionSchema, Engine: "fak-native", Collector: artifact.Provider,
		IntervalMS:   intervalMS,
		Availability: CollectorAvailability{DRAMCounters: true}, Capture: capture, Report: report,
		HostControllerArtifact: &artifact,
	}, nil
}

func hostCounterRunningRatio(enabled, running *uint64, supplied *float64) (float64, error) {
	if (enabled == nil) != (running == nil) {
		return 0, errors.New("time_enabled_ns and time_running_ns must be paired")
	}
	var ratio float64
	if enabled != nil {
		if *enabled == 0 || *running > *enabled {
			return 0, errors.New("invalid time enabled/running values")
		}
		ratio = float64(*running) / float64(*enabled)
		if supplied != nil && math.Abs(*supplied-ratio) > 1e-6 {
			return 0, errors.New("running ratio conflicts with time enabled/running")
		}
	} else if supplied != nil {
		ratio = *supplied
	} else {
		return 0, errors.New("time enabled/running or running ratio is required")
	}
	if err := validateRunningRatio(ratio); err != nil {
		return 0, err
	}
	return ratio, nil
}

func validateRunningRatio(ratio float64) error {
	if math.IsNaN(ratio) || math.IsInf(ratio, 0) || ratio <= 0 || ratio > 1 {
		return fmt.Errorf("running ratio must be in (0,1], got %g", ratio)
	}
	if ratio < MinHostControllerRunningRatio {
		return fmt.Errorf("multiplexed host counter running ratio %.6g is below floor %.2f", ratio, MinHostControllerRunningRatio)
	}
	return nil
}

func validatePerfRunningRatio(ratio float64) error {
	if err := validateRunningRatio(ratio); err != nil {
		return err
	}
	if ratio != 1 {
		return fmt.Errorf("perf stat running ratio %.6g is unavailable: counter-value scaling is ambiguous unless pcnt-running is 100%%", ratio)
	}
	return nil
}

func validateCounterTiming(enabled, running *uint64, ratio float64, captureInterval time.Duration) error {
	if enabled == nil {
		return nil
	}
	if running == nil {
		return errors.New("time_enabled_ns and time_running_ns must be paired")
	}
	if captureInterval <= 0 {
		return errors.New("host counter capture interval must be positive")
	}
	captureNS := uint64(captureInterval)
	tolerance := uint64(time.Microsecond)
	if relative := captureNS / 100; relative > tolerance {
		tolerance = relative
	}
	difference := *enabled
	if difference >= captureNS {
		difference -= captureNS
	} else {
		difference = captureNS - difference
	}
	if difference > tolerance {
		return fmt.Errorf("time_enabled_ns %d conflicts with capture interval %d ns", *enabled, captureNS)
	}
	calculated := float64(*running) / float64(*enabled)
	if math.Abs(calculated-ratio) > 1e-6 {
		return errors.New("running ratio conflicts with time enabled/running")
	}
	return nil
}

func validatePerfCounterTiming(running uint64, captureInterval time.Duration, pmuCount *uint64, scope HostControllerScope) error {
	if captureInterval <= 0 {
		return errors.New("host counter capture interval must be positive")
	}
	multiplier := uint64(0)
	if pmuCount != nil {
		multiplier = *pmuCount
	} else if scope.Kind == "controller" {
		multiplier = 1
	} else {
		// Global perf output may merge multiple uncore PMUs without printing
		// their count. Preserve time_running and the 100% ratio, but do not
		// invent time_enabled or an aggregation factor.
		return nil
	}
	captureNS := uint64(captureInterval)
	if multiplier == 0 || captureNS > math.MaxUint64/multiplier {
		return errors.New("perf aggregate enabled-time calculation overflows")
	}
	expected := captureNS * multiplier
	tolerance := uint64(time.Microsecond)
	if relative := expected / 100; relative > tolerance {
		tolerance = relative
	}
	difference := running
	if difference >= expected {
		difference -= expected
	} else {
		difference = expected - difference
	}
	if difference > tolerance {
		return fmt.Errorf("event-runtime %d conflicts with capture interval %d ns and aggregate count %d", running, captureNS, multiplier)
	}
	return nil
}

func cumulativeCounterDelta(start, end uint64, width *uint8) (uint64, bool, error) {
	if width != nil {
		if *width == 0 || *width > 64 {
			return 0, false, fmt.Errorf("counter width must be 1..64 bits, got %d", *width)
		}
		if *width < 64 {
			limit := uint64(1) << *width
			if start >= limit || end >= limit {
				return 0, false, fmt.Errorf("counter value exceeds %d-bit width", *width)
			}
		}
	}
	if end >= start {
		return end - start, false, nil
	}
	if width == nil {
		return 0, false, errors.New("counter decreased without an explicit width; reset versus wrap is ambiguous")
	}
	if *width < 64 {
		limit := uint64(1) << *width
		return limit - start + end, true, nil
	}
	return math.MaxUint64 - start + 1 + end, true, nil
}

func convertHostCounterBytes(value uint64, unit string, bytesPerEvent *uint64) (uint64, string, error) {
	normalized := strings.ToLower(strings.TrimSpace(unit))
	switch normalized {
	case "b", "byte", "bytes":
		if bytesPerEvent != nil {
			return 0, "", errors.New("bytes-per-event cannot be applied to a byte counter")
		}
		return value, "direct-bytes", nil
	case "", "event", "events", "count", "counts":
		if bytesPerEvent == nil || *bytesPerEvent == 0 {
			return 0, "", errors.New("event counter requires an explicit positive bytes-per-event conversion")
		}
		if value > math.MaxUint64 / *bytesPerEvent {
			return 0, "", errors.New("counter byte conversion overflows uint64")
		}
		return value * *bytesPerEvent, "converted-events", nil
	default:
		return 0, "", fmt.Errorf("unsupported host counter unit %q", unit)
	}
}

func hostCounterDirection(event, explicit string) (string, bool, error) {
	lower := strings.ToLower(strings.TrimSpace(event))
	if lower == "" {
		return "", false, nil
	}
	if lower == "read_bytes" || lower == "write_bytes" || lower == "read-bytes" || lower == "write-bytes" {
		return "", true, fmt.Errorf("event %q is process/storage I/O, not a host DRAM controller counter", event)
	}
	for _, forbidden := range []string{"process", "proc/", "rchar", "wchar", "syscr", "syscw", "disk", "block", "io_read", "io_write"} {
		if strings.Contains(lower, forbidden) {
			return "", true, fmt.Errorf("event %q is process/storage I/O, not a host DRAM controller counter", event)
		}
	}
	direction := strings.ToLower(strings.TrimSpace(explicit))
	if direction != "" && direction != "read" && direction != "write" {
		return "", true, fmt.Errorf("event %q has unsupported direction %q", event, explicit)
	}
	controllerEvent := containsAny(lower, "dram", "uncore", "imc", "memory_controller", "memory-controller", "cas_count", "umc")
	if direction != "" && !controllerEvent {
		return "", false, nil
	}
	if direction == "" {
		read := containsAny(lower, "cas_count_read", "cas-count-read", "read_cas", "dram__bytes_read", "dram_reads", "dram_read", "imc_reads")
		write := containsAny(lower, "cas_count_write", "cas-count-write", "write_cas", "dram__bytes_write", "dram_writes", "dram_write", "imc_writes")
		if read == write {
			return "", false, nil
		}
		if read {
			direction = "read"
		} else {
			direction = "write"
		}
	}
	return direction, true, nil
}

func containsAny(s string, values ...string) bool {
	for _, value := range values {
		if strings.Contains(s, value) {
			return true
		}
	}
	return false
}

func validateCaptureInterval(started, ended time.Time) error {
	if started.IsZero() || ended.IsZero() || !started.Before(ended) {
		return errors.New("host counter capture end must be after start")
	}
	if ended.Sub(started) <= 0 {
		return errors.New("host counter capture interval must be positive")
	}
	return nil
}

func rawUint64(raw json.RawMessage, name string) (uint64, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return 0, fmt.Errorf("%s is required", name)
	}
	return parseUint64Text(string(bytes.Trim(raw, `"`)), name)
}

func rawFloat64(raw json.RawMessage, name string) (float64, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return 0, fmt.Errorf("%s is required", name)
	}
	return parseFloatText(string(bytes.Trim(raw, `"`)), name)
}

func rawString(raw json.RawMessage) (string, error) {
	var text string
	if err := json.Unmarshal(raw, &text); err == nil {
		return strings.TrimSpace(text), nil
	}
	var number json.Number
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	if err := dec.Decode(&number); err != nil {
		return "", err
	}
	return string(number), nil
}

func parseUint64Text(text, name string) (uint64, error) {
	clean := strings.TrimSpace(text)
	if clean == "" || strings.HasPrefix(clean, "<") {
		return 0, fmt.Errorf("%s is unavailable", name)
	}
	// perf stat JSON renders counter-value with %f even for integer hardware
	// counts. Accept only a decimal spelling whose fractional part is exactly
	// zero; parsing through float64 would lose raw counts above 2^53.
	integer := clean
	if dot := strings.IndexByte(clean, '.'); dot >= 0 {
		integer = clean[:dot]
		fraction := clean[dot+1:]
		if fraction == "" || strings.Trim(fraction, "0") != "" {
			return 0, fmt.Errorf("invalid %s %q: fractional hardware count", name, text)
		}
	}
	if integer == "" || strings.ContainsAny(integer, ",eE+-") {
		return 0, fmt.Errorf("invalid %s %q", name, text)
	}
	value, err := strconv.ParseUint(integer, 10, 64)
	if err != nil {
		return 0, fmt.Errorf("invalid %s %q", name, text)
	}
	return value, nil
}

func parseFloatText(text, name string) (float64, error) {
	clean := strings.TrimSpace(strings.TrimSuffix(text, "%"))
	value, err := strconv.ParseFloat(clean, 64)
	if err != nil || math.IsNaN(value) || math.IsInf(value, 0) {
		return 0, fmt.Errorf("invalid %s %q", name, text)
	}
	return value, nil
}

func requireJSONEOF(dec *json.Decoder) error {
	var extra any
	if err := dec.Decode(&extra); !errors.Is(err, io.EOF) {
		if err == nil {
			return errors.New("JSON input contains multiple values")
		}
		return err
	}
	return nil
}
