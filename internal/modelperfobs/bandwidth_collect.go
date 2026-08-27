package modelperfobs

import (
	"context"
	"fmt"
	"runtime"
	"time"
)

const BandwidthCollectionSchema = "fak-model-bandwidth-collection/1"
const MaxCollectionSamples = 120

var MinSampleInterval = 10 * time.Millisecond
var MaxSampleInterval = time.Minute

type HostSignals struct {
	PhysicalTotalBytes     *uint64 `json:"physical_total_bytes,omitempty"`
	PhysicalAvailableBytes *uint64 `json:"physical_available_bytes,omitempty"`
	ProcessResidentBytes   *uint64 `json:"process_resident_bytes,omitempty"`
	ProcessReadBytes       *uint64 `json:"process_read_bytes,omitempty"`
	ProcessWriteBytes      *uint64 `json:"process_write_bytes,omitempty"`
	ProcessIOScope         string  `json:"process_io_scope,omitempty"`
}
type CollectorAvailability struct {
	PhysicalMemory bool `json:"physical_memory"`
	ProcessMemory  bool `json:"process_memory"`
	ProcessIO      bool `json:"process_io"`
	DRAMCounters   bool `json:"dram_counters"`
	DeviceCounters bool `json:"device_counters"`
}
type CollectionOptions struct {
	Count                  int
	Interval               time.Duration
	Phase                  RequestPhase
	Shape                  RequestShape
	TheoreticalGBS         *float64
	MeasuredSustainableGBS *float64
	LatencyMS              *float64
	PromptTokens           *int64
	CompletionTokens       *int64
	LogicalBytes           *uint64
	PhysicalSoftwareRead   *uint64
	PhysicalSoftwareWrite  *uint64
	NVIDIADevice           NVIDIADeviceSelector
}
type BandwidthCollection struct {
	Schema       string                `json:"schema"`
	Engine       string                `json:"engine"`
	MachineClass string                `json:"machine_class"`
	Collector    string                `json:"collector"`
	IntervalMS   int64                 `json:"interval_ms"`
	Availability CollectorAvailability `json:"availability"`
	Capture      BandwidthCapture      `json:"capture"`
	Report       BandwidthReport       `json:"report"`
}
type hostSnapshot struct {
	at           time.Time
	host         HostSignals
	availability CollectorAvailability
	collector    string
}

func ValidateCollectionOptions(o CollectionOptions) error {
	if o.Count < 1 || o.Count > MaxCollectionSamples {
		return fmt.Errorf("sample count must be 1..%d", MaxCollectionSamples)
	}
	if o.Interval < MinSampleInterval || o.Interval > MaxSampleInterval {
		return fmt.Errorf("sample interval must be %s..%s", MinSampleInterval, MaxSampleInterval)
	}
	return validateSample(BandwidthSample{Phase: o.Phase, Shape: o.Shape, Provenance: BandwidthProvenance{Source: "host"}})
}
func CollectBandwidth(ctx context.Context, o CollectionOptions) (BandwidthCollection, error) {
	if err := ValidateCollectionOptions(o); err != nil {
		return BandwidthCollection{}, err
	}
	cap := BandwidthCapture{Schema: BandwidthSchema, Engine: "fak-native", Trigger: TriggerConfig{SymptomWindow: 2, ResourceWindow: 2, LatencyThresholdMS: 1e100, ResourceUtilization: 1}, Samples: make([]BandwidthSample, 0, o.Count)}
	var av CollectorAvailability
	collector := "portable"
	for i := 0; i < o.Count; i++ {
		if i > 0 {
			select {
			case <-ctx.Done():
				return BandwidthCollection{}, ctx.Err()
			case <-time.After(o.Interval):
			}
		}
		snap, err := collectHostSnapshot()
		if err != nil {
			return BandwidthCollection{}, err
		}
		device, err := collectNVIDIADeviceSnapshot(ctx, o.NVIDIADevice)
		if err != nil {
			return BandwidthCollection{}, err
		}
		collector = snap.collector
		av.PhysicalMemory = av.PhysicalMemory || snap.availability.PhysicalMemory
		av.ProcessMemory = av.ProcessMemory || snap.availability.ProcessMemory
		av.ProcessIO = av.ProcessIO || snap.availability.ProcessIO
		av.DeviceCounters = av.DeviceCounters || device.available
		s := BandwidthSample{Phase: o.Phase, Shape: o.Shape, Provenance: BandwidthProvenance{Source: "live-host", Machine: runtime.GOOS + "/" + runtime.GOARCH, Device: "host-memory", Collector: collector, SampledAt: snap.at.UTC().Format(time.RFC3339Nano)}, Rooflines: Rooflines{TheoreticalGBS: cloneFloat(o.TheoreticalGBS), MeasuredSustainableGBS: cloneFloat(o.MeasuredSustainableGBS)}, Request: RequestSignals{LatencyMS: cloneFloat(o.LatencyMS), PromptTokens: cloneI64(o.PromptTokens), CompletionTokens: cloneI64(o.CompletionTokens)}, Host: snap.host, Device: device.device, Software: SoftwareTraffic{LogicalBytes: cloneU64(o.LogicalBytes), PhysicalReadBytes: cloneU64(o.PhysicalSoftwareRead), PhysicalWriteBytes: cloneU64(o.PhysicalSoftwareWrite)}}
		if device.available {
			collector = snap.collector + "+" + device.collector
			s.Capacity = device.capacity
			s.Provenance.Device = device.provenanceDevice
			s.Provenance.Collector = collector
		}
		if !device.available && snap.host.PhysicalTotalBytes != nil && snap.host.PhysicalAvailableBytes != nil && *snap.host.PhysicalTotalBytes >= *snap.host.PhysicalAvailableBytes {
			u := *snap.host.PhysicalTotalBytes - *snap.host.PhysicalAvailableBytes
			s.Capacity.TotalBytes = cloneU64(snap.host.PhysicalTotalBytes)
			s.Capacity.UsedBytes = &u
		}
		cap.Samples = append(cap.Samples, s)
	}
	report, err := AnalyzeBandwidth(cap)
	if err != nil {
		return BandwidthCollection{}, err
	}
	return BandwidthCollection{Schema: BandwidthCollectionSchema, Engine: "fak-native", MachineClass: runtime.GOOS + "/" + runtime.GOARCH, Collector: collector, IntervalMS: o.Interval.Milliseconds(), Availability: av, Capture: cap, Report: report}, nil
}
func cloneI64(v *int64) *int64 {
	if v == nil {
		return nil
	}
	x := *v
	return &x
}
func cloneU64(v *uint64) *uint64 {
	if v == nil {
		return nil
	}
	x := *v
	return &x
}
