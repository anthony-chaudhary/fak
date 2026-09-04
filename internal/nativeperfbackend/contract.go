// Package nativeperfbackend defines the bounded Prometheus contract used by the
// fak-native Metal and CUDA backend drill-down dashboard.
package nativeperfbackend

import (
	"fmt"
	"sort"
	"strings"
)

const (
	Schema = "fak-native-backend-telemetry/1"
	Engine = "fak-native"
)

type Backend string

const (
	BackendMetal Backend = "metal"
	BackendCUDA  Backend = "cuda"
)

type UnavailableReason string

const (
	ReasonNone                 UnavailableReason = "none"
	ReasonBackendNotBuilt      UnavailableReason = "backend_not_built"
	ReasonDeviceNotFound       UnavailableReason = "device_not_found"
	ReasonPermissionDenied     UnavailableReason = "permission_denied"
	ReasonDriverUnavailable    UnavailableReason = "driver_unavailable"
	ReasonTelemetryUnsupported UnavailableReason = "telemetry_unsupported"
	ReasonCollectionFailed     UnavailableReason = "collection_failed"
)

type MemoryKind string

const (
	MemoryAllocated MemoryKind = "allocated"
	MemoryResident  MemoryKind = "resident"
)

type DelayKind string

const (
	DelayQueue         DelayKind = "queue"
	DelayStream        DelayKind = "stream"
	DelayCommandBuffer DelayKind = "command_buffer"
)

type Direction string

const (
	DirectionUpload   Direction = "upload"
	DirectionDownload Direction = "download"
)

type KernelFamily string

const (
	KernelMatmul        KernelFamily = "matmul"
	KernelAttention     KernelFamily = "attention"
	KernelNormalization KernelFamily = "normalization"
	KernelEmbedding     KernelFamily = "embedding"
	KernelSampling      KernelFamily = "sampling"
	KernelTransfer      KernelFamily = "transfer"
	KernelOther         KernelFamily = "other"
)

type SyncKind string

const (
	SyncFence      SyncKind = "fence"
	SyncEvent      SyncKind = "event"
	SyncBarrier    SyncKind = "barrier"
	SyncDeviceWait SyncKind = "device_wait"
	SyncOther      SyncKind = "other"
)

type GraphState string

const (
	GraphUnsupported GraphState = "unsupported"
	GraphDisabled    GraphState = "disabled"
	GraphCapturing   GraphState = "capturing"
	GraphReady       GraphState = "ready"
	GraphReplaying   GraphState = "replaying"
	GraphFailed      GraphState = "failed"
)

// Metric describes one metric family and its bounded label dimensions.
type Metric struct {
	Name   string
	Kind   string
	Unit   string
	Labels []string
}

// Metrics returns the stable metric families consumed by the dashboard.
func Metrics() []Metric {
	return []Metric{
		{Name: "fak_native_backend_info", Kind: "gauge", Unit: "info", Labels: []string{"backend", "engine", "model_family", "schema"}},
		{Name: "fak_native_backend_available", Kind: "gauge", Unit: "state", Labels: []string{"backend", "reason"}},
		{Name: "fak_native_backend_device_utilization_ratio", Kind: "gauge", Unit: "ratio", Labels: []string{"backend"}},
		{Name: "fak_native_backend_memory_bytes", Kind: "gauge", Unit: "bytes", Labels: []string{"backend", "kind"}},
		{Name: "fak_native_backend_memory_pressure_ratio", Kind: "gauge", Unit: "ratio", Labels: []string{"backend"}},
		{Name: "fak_native_backend_delay_seconds", Kind: "gauge", Unit: "seconds", Labels: []string{"backend", "kind"}},
		{Name: "fak_native_backend_transfer_bytes_total", Kind: "counter", Unit: "bytes", Labels: []string{"backend", "direction"}},
		{Name: "fak_native_backend_transfer_seconds_total", Kind: "counter", Unit: "seconds", Labels: []string{"backend", "direction"}},
		{Name: "fak_native_backend_kernel_calls_total", Kind: "counter", Unit: "calls", Labels: []string{"backend", "family"}},
		{Name: "fak_native_backend_kernel_seconds_total", Kind: "counter", Unit: "seconds", Labels: []string{"backend", "family"}},
		{Name: "fak_native_backend_sync_events_total", Kind: "counter", Unit: "events", Labels: []string{"backend", "kind"}},
		{Name: "fak_native_backend_sync_seconds_total", Kind: "counter", Unit: "seconds", Labels: []string{"backend", "kind"}},
		{Name: "fak_native_backend_graph_state", Kind: "gauge", Unit: "state", Labels: []string{"backend", "state"}},
	}
}

// Snapshot is one backend scrape. Missing optional values remain absent rather
// than becoming misleading zeros. An unavailable backend carries only identity,
// availability, and an explicit bounded reason.
type Snapshot struct {
	Backend              Backend
	Engine               string
	ModelFamily          string
	Available            bool
	UnavailableReason    UnavailableReason
	DeviceUtilization    *float64
	MemoryBytes          map[MemoryKind]float64
	MemoryPressure       *float64
	DelaySeconds         map[DelayKind]float64
	TransferBytesTotal   map[Direction]float64
	TransferSecondsTotal map[Direction]float64
	KernelCallsTotal     map[KernelFamily]float64
	KernelSecondsTotal   map[KernelFamily]float64
	SyncEventsTotal      map[SyncKind]float64
	SyncSecondsTotal     map[SyncKind]float64
	GraphState           *GraphState
}

var allowedModelFamilies = map[string]bool{"Qwen3.8": true, "other": true}

func validBackend(v Backend) bool { return v == BackendMetal || v == BackendCUDA }

// Validate enforces bounded dimensions, fak-native identity, ratio ranges, and
// honest unavailable-state semantics.
func Validate(s Snapshot) error {
	var problems []string
	if !validBackend(s.Backend) {
		problems = append(problems, "backend must be metal or cuda")
	}
	if s.Engine != Engine {
		problems = append(problems, `engine must be "fak-native"; fallback engines are forbidden`)
	}
	if !allowedModelFamilies[s.ModelFamily] {
		problems = append(problems, "model_family must be Qwen3.8 or other")
	}
	if s.Available {
		if s.UnavailableReason != ReasonNone {
			problems = append(problems, "available backend must use reason=none")
		}
	} else {
		if !validReason(s.UnavailableReason) || s.UnavailableReason == ReasonNone {
			problems = append(problems, "unavailable backend requires an explicit bounded reason")
		}
		if hasMeasurements(s) {
			problems = append(problems, "unavailable backend cannot publish measurement zeros or stale values")
		}
	}
	checkRatio := func(name string, value *float64) {
		if value != nil && (*value < 0 || *value > 1) {
			problems = append(problems, name+" must be within [0,1]")
		}
	}
	checkRatio("device_utilization", s.DeviceUtilization)
	checkRatio("memory_pressure", s.MemoryPressure)
	checkMap := func(name string, values map[string]float64, allowed map[string]bool) {
		for key, value := range values {
			if !allowed[key] {
				problems = append(problems, name+" has unbounded dimension "+key)
			}
			if value < 0 {
				problems = append(problems, name+" values must be non-negative")
			}
		}
	}
	checkMap("memory", memoryMap(s.MemoryBytes), set("allocated", "resident"))
	checkMap("delay", delayMap(s.DelaySeconds), set("queue", "stream", "command_buffer"))
	checkMap("transfer_bytes", directionMap(s.TransferBytesTotal), set("upload", "download"))
	checkMap("transfer_seconds", directionMap(s.TransferSecondsTotal), set("upload", "download"))
	checkMap("kernel_calls", kernelMap(s.KernelCallsTotal), set("matmul", "attention", "normalization", "embedding", "sampling", "transfer", "other"))
	checkMap("kernel_seconds", kernelMap(s.KernelSecondsTotal), set("matmul", "attention", "normalization", "embedding", "sampling", "transfer", "other"))
	checkMap("sync_events", syncMap(s.SyncEventsTotal), set("fence", "event", "barrier", "device_wait", "other"))
	checkMap("sync_seconds", syncMap(s.SyncSecondsTotal), set("fence", "event", "barrier", "device_wait", "other"))
	if s.GraphState != nil && !validGraphState(*s.GraphState) {
		problems = append(problems, "graph_state has unbounded dimension "+string(*s.GraphState))
	}
	if len(problems) != 0 {
		sort.Strings(problems)
		return fmt.Errorf("native backend telemetry invalid:\n- %s", strings.Join(problems, "\n- "))
	}
	return nil
}

func validReason(v UnavailableReason) bool {
	switch v {
	case ReasonNone, ReasonBackendNotBuilt, ReasonDeviceNotFound, ReasonPermissionDenied, ReasonDriverUnavailable, ReasonTelemetryUnsupported, ReasonCollectionFailed:
		return true
	default:
		return false
	}
}

func validGraphState(v GraphState) bool {
	switch v {
	case GraphUnsupported, GraphDisabled, GraphCapturing, GraphReady, GraphReplaying, GraphFailed:
		return true
	default:
		return false
	}
}

func hasMeasurements(s Snapshot) bool {
	return s.DeviceUtilization != nil || s.MemoryPressure != nil || len(s.MemoryBytes) != 0 || len(s.DelaySeconds) != 0 || len(s.TransferBytesTotal) != 0 || len(s.TransferSecondsTotal) != 0 || len(s.KernelCallsTotal) != 0 || len(s.KernelSecondsTotal) != 0 || len(s.SyncEventsTotal) != 0 || len(s.SyncSecondsTotal) != 0 || s.GraphState != nil
}

func set(values ...string) map[string]bool {
	out := make(map[string]bool, len(values))
	for _, value := range values {
		out[value] = true
	}
	return out
}
func stringKeyMap[K ~string](in map[K]float64) map[string]float64 {
	out := make(map[string]float64, len(in))
	for k, v := range in {
		out[string(k)] = v
	}
	return out
}

func memoryMap(in map[MemoryKind]float64) map[string]float64   { return stringKeyMap(in) }
func delayMap(in map[DelayKind]float64) map[string]float64     { return stringKeyMap(in) }
func directionMap(in map[Direction]float64) map[string]float64 { return stringKeyMap(in) }
func kernelMap(in map[KernelFamily]float64) map[string]float64 { return stringKeyMap(in) }
func syncMap(in map[SyncKind]float64) map[string]float64       { return stringKeyMap(in) }
