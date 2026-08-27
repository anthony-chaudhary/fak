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
	// PhysicalAvailableSemantics labels a collector-specific approximation.
	// Labeled availability remains Host evidence and is not generic Capacity.
	PhysicalAvailableSemantics  string   `json:"physical_available_semantics,omitempty"`
	PhysicalFreeBytes           *uint64  `json:"physical_free_bytes,omitempty"`
	ProcessResidentBytes        *uint64  `json:"process_resident_bytes,omitempty"`
	ProcessReadBytes            *uint64  `json:"process_read_bytes,omitempty"`
	ProcessWriteBytes           *uint64  `json:"process_write_bytes,omitempty"`
	SwapTotalBytes              *uint64  `json:"swap_total_bytes,omitempty"`
	SwapUsedBytes               *uint64  `json:"swap_used_bytes,omitempty"`
	CommitLimitBytes            *uint64  `json:"commit_limit_bytes,omitempty"`
	CommitUsedBytes             *uint64  `json:"commit_used_bytes,omitempty"`
	ProcessMinorFaults          *uint64  `json:"process_minor_faults,omitempty"`
	ProcessMajorFaults          *uint64  `json:"process_major_faults,omitempty"`
	ProcessPageFaults           *uint64  `json:"process_page_faults,omitempty"`
	ProcessMinorFaultsPerSecond *float64 `json:"process_minor_faults_per_second,omitempty"`
	ProcessMajorFaultsPerSecond *float64 `json:"process_major_faults_per_second,omitempty"`
	ProcessPageFaultsPerSecond  *float64 `json:"process_page_faults_per_second,omitempty"`
	// Linux PSI averages are percentages; totals and derived rates are stall
	// microseconds. They are pressure evidence, never inferred DRAM throughput.
	MemoryPressureSomeAvg10Percent               *float64 `json:"memory_pressure_some_avg10_percent,omitempty"`
	MemoryPressureSomeAvg60Percent               *float64 `json:"memory_pressure_some_avg60_percent,omitempty"`
	MemoryPressureSomeAvg300Percent              *float64 `json:"memory_pressure_some_avg300_percent,omitempty"`
	MemoryPressureSomeTotalStallMicroseconds     *uint64  `json:"memory_pressure_some_total_stall_microseconds,omitempty"`
	MemoryPressureSomeStallMicrosecondsPerSecond *float64 `json:"memory_pressure_some_stall_microseconds_per_second,omitempty"`
	MemoryPressureFullAvg10Percent               *float64 `json:"memory_pressure_full_avg10_percent,omitempty"`
	MemoryPressureFullAvg60Percent               *float64 `json:"memory_pressure_full_avg60_percent,omitempty"`
	MemoryPressureFullAvg300Percent              *float64 `json:"memory_pressure_full_avg300_percent,omitempty"`
	MemoryPressureFullTotalStallMicroseconds     *uint64  `json:"memory_pressure_full_total_stall_microseconds,omitempty"`
	MemoryPressureFullStallMicrosecondsPerSecond *float64 `json:"memory_pressure_full_stall_microseconds_per_second,omitempty"`
	// Linux vmstat reclaim and swap counters are cumulative page activity, not
	// byte traffic. Rates are derived only from monotonic two-sample deltas.
	MemoryReclaimKswapdScannedPagesTotal       *uint64  `json:"memory_reclaim_kswapd_scanned_pages_total,omitempty"`
	MemoryReclaimKswapdScannedPagesPerSecond   *float64 `json:"memory_reclaim_kswapd_scanned_pages_per_second,omitempty"`
	MemoryReclaimDirectScannedPagesTotal       *uint64  `json:"memory_reclaim_direct_scanned_pages_total,omitempty"`
	MemoryReclaimDirectScannedPagesPerSecond   *float64 `json:"memory_reclaim_direct_scanned_pages_per_second,omitempty"`
	MemoryReclaimKswapdReclaimedPagesTotal     *uint64  `json:"memory_reclaim_kswapd_reclaimed_pages_total,omitempty"`
	MemoryReclaimKswapdReclaimedPagesPerSecond *float64 `json:"memory_reclaim_kswapd_reclaimed_pages_per_second,omitempty"`
	MemoryReclaimDirectReclaimedPagesTotal     *uint64  `json:"memory_reclaim_direct_reclaimed_pages_total,omitempty"`
	MemoryReclaimDirectReclaimedPagesPerSecond *float64 `json:"memory_reclaim_direct_reclaimed_pages_per_second,omitempty"`
	MemorySwapInPagesTotal                     *uint64  `json:"memory_swap_in_pages_total,omitempty"`
	MemorySwapInPagesPerSecond                 *float64 `json:"memory_swap_in_pages_per_second,omitempty"`
	MemorySwapOutPagesTotal                    *uint64  `json:"memory_swap_out_pages_total,omitempty"`
	MemorySwapOutPagesPerSecond                *float64 `json:"memory_swap_out_pages_per_second,omitempty"`
	// Darwin vm_stat resident-byte states are pressure/occupancy evidence.
	// Page-in/out counters and rates are paging activity, never memory bandwidth.
	MemoryWiredResidentBytes      *uint64  `json:"memory_wired_resident_bytes,omitempty"`
	MemoryCompressedResidentBytes *uint64  `json:"memory_compressed_resident_bytes,omitempty"`
	MemoryPageInPagesTotal        *uint64  `json:"memory_page_in_pages_total,omitempty"`
	MemoryPageInPagesPerSecond    *float64 `json:"memory_page_in_pages_per_second,omitempty"`
	MemoryPageOutPagesTotal       *uint64  `json:"memory_page_out_pages_total,omitempty"`
	MemoryPageOutPagesPerSecond   *float64 `json:"memory_page_out_pages_per_second,omitempty"`
	ProcessIOScope                string   `json:"process_io_scope,omitempty"`
}
type CollectorAvailability struct {
	PhysicalMemory bool `json:"physical_memory"`
	ProcessMemory  bool `json:"process_memory"`
	ProcessIO      bool `json:"process_io"`
	MemoryPressure bool `json:"memory_pressure"`
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
	AMDDevice              AMDDeviceSelector
	MeasureHostRoofline    *RooflineBenchmarkOptions
}
type BandwidthCollection struct {
	Schema              string                `json:"schema"`
	Engine              string                `json:"engine"`
	MachineClass        string                `json:"machine_class"`
	Collector           string                `json:"collector"`
	IntervalMS          int64                 `json:"interval_ms"`
	Availability        CollectorAvailability `json:"availability"`
	Capture             BandwidthCapture      `json:"capture"`
	Report              BandwidthReport       `json:"report"`
	RooflineMeasurement *RooflineMeasurement  `json:"roofline_measurement,omitempty"`
	ProfileReceipt      *NVIDIAProfileReceipt `json:"profile_receipt,omitempty"`
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
	if o.NVIDIADevice != "" && o.AMDDevice != "" {
		return fmt.Errorf("nvidia and AMD device selectors are mutually exclusive")
	}
	return validateSample(BandwidthSample{Phase: o.Phase, Shape: o.Shape, Provenance: BandwidthProvenance{Source: "host"}})
}
func CollectBandwidth(ctx context.Context, o CollectionOptions) (BandwidthCollection, error) {
	if err := ValidateCollectionOptions(o); err != nil {
		return BandwidthCollection{}, err
	}
	var measured *RooflineMeasurement
	if o.MeasureHostRoofline != nil {
		r, err := MeasureHostMemoryRoofline(ctx, *o.MeasureHostRoofline)
		if err != nil {
			return BandwidthCollection{}, err
		}
		measured = &r
		o.MeasuredSustainableGBS = &measured.MeasuredSustainableGBS
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
		device, err := collectDeviceSnapshot(ctx, o)
		if err != nil {
			return BandwidthCollection{}, err
		}
		collector = snap.collector
		av.PhysicalMemory = av.PhysicalMemory || snap.availability.PhysicalMemory
		av.ProcessMemory = av.ProcessMemory || snap.availability.ProcessMemory
		av.ProcessIO = av.ProcessIO || snap.availability.ProcessIO
		av.MemoryPressure = av.MemoryPressure || snap.availability.MemoryPressure
		av.DeviceCounters = av.DeviceCounters || device.available
		av.DRAMCounters = av.DRAMCounters || device.dramCounters
		s := BandwidthSample{Phase: o.Phase, Shape: o.Shape, Provenance: BandwidthProvenance{Source: "live-host", Machine: runtime.GOOS + "/" + runtime.GOARCH, Device: "host-memory", Collector: collector, SampledAt: snap.at.UTC().Format(time.RFC3339Nano)}, Rooflines: Rooflines{TheoreticalGBS: cloneFloat(o.TheoreticalGBS), MeasuredSustainableGBS: cloneFloat(o.MeasuredSustainableGBS)}, Live: device.live, Request: RequestSignals{LatencyMS: cloneFloat(o.LatencyMS), PromptTokens: cloneI64(o.PromptTokens), CompletionTokens: cloneI64(o.CompletionTokens)}, Host: snap.host, Device: device.device, Software: SoftwareTraffic{LogicalBytes: cloneU64(o.LogicalBytes), PhysicalReadBytes: cloneU64(o.PhysicalSoftwareRead), PhysicalWriteBytes: cloneU64(o.PhysicalSoftwareWrite)}}
		if device.available {
			collector = snap.collector + "+" + device.collector
			s.Capacity = device.capacity
			s.Provenance.Device = device.provenanceDevice
			s.Provenance.Collector = collector
			if device.provenanceSource != "" {
				s.Provenance.Source = device.provenanceSource
			}
		}
		if !device.available {
			s.Capacity = capacityFromExactHostAvailability(snap.host)
		}
		if len(cap.Samples) > 0 {
			deriveHostPressureRates(&cap.Samples[len(cap.Samples)-1], &s)
		}
		cap.Samples = append(cap.Samples, s)
	}
	report, err := AnalyzeBandwidth(cap)
	if err != nil {
		return BandwidthCollection{}, err
	}
	return BandwidthCollection{Schema: BandwidthCollectionSchema, Engine: "fak-native", MachineClass: runtime.GOOS + "/" + runtime.GOARCH, Collector: collector, IntervalMS: o.Interval.Milliseconds(), Availability: av, Capture: cap, Report: report, RooflineMeasurement: measured}, nil
}

func capacityFromExactHostAvailability(host HostSignals) CapacitySignals {
	if host.PhysicalAvailableSemantics != "" || host.PhysicalTotalBytes == nil ||
		host.PhysicalAvailableBytes == nil || *host.PhysicalTotalBytes < *host.PhysicalAvailableBytes {
		return CapacitySignals{}
	}
	used := *host.PhysicalTotalBytes - *host.PhysicalAvailableBytes
	return CapacitySignals{TotalBytes: cloneU64(host.PhysicalTotalBytes), UsedBytes: &used}
}

func collectDeviceSnapshot(ctx context.Context, o CollectionOptions) (deviceSnapshot, error) {
	if o.AMDDevice != "" {
		return collectAMDDeviceSnapshot(ctx, o.AMDDevice)
	}
	if o.NVIDIADevice != "" {
		return collectNVIDIADeviceSnapshot(ctx, o.NVIDIADevice)
	}
	s, err := collectNVIDIADeviceSnapshot(ctx, "")
	if err != nil || s.available {
		return s, err
	}
	return collectAMDDeviceSnapshot(ctx, "")
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

func deriveHostPressureRates(previous, current *BandwidthSample) {
	prevAt, e1 := time.Parse(time.RFC3339Nano, previous.Provenance.SampledAt)
	currAt, e2 := time.Parse(time.RFC3339Nano, current.Provenance.SampledAt)
	if e1 != nil || e2 != nil || !currAt.After(prevAt) {
		return
	}
	seconds := currAt.Sub(prevAt).Seconds()
	current.Host.ProcessMinorFaultsPerSecond = counterRate(previous.Host.ProcessMinorFaults, current.Host.ProcessMinorFaults, seconds)
	current.Host.ProcessMajorFaultsPerSecond = counterRate(previous.Host.ProcessMajorFaults, current.Host.ProcessMajorFaults, seconds)
	current.Host.ProcessPageFaultsPerSecond = counterRate(previous.Host.ProcessPageFaults, current.Host.ProcessPageFaults, seconds)
	current.Host.MemoryPressureSomeStallMicrosecondsPerSecond = counterRate(previous.Host.MemoryPressureSomeTotalStallMicroseconds, current.Host.MemoryPressureSomeTotalStallMicroseconds, seconds)
	current.Host.MemoryPressureFullStallMicrosecondsPerSecond = counterRate(previous.Host.MemoryPressureFullTotalStallMicroseconds, current.Host.MemoryPressureFullTotalStallMicroseconds, seconds)
	current.Host.MemoryReclaimKswapdScannedPagesPerSecond = counterRate(previous.Host.MemoryReclaimKswapdScannedPagesTotal, current.Host.MemoryReclaimKswapdScannedPagesTotal, seconds)
	current.Host.MemoryReclaimDirectScannedPagesPerSecond = counterRate(previous.Host.MemoryReclaimDirectScannedPagesTotal, current.Host.MemoryReclaimDirectScannedPagesTotal, seconds)
	current.Host.MemoryReclaimKswapdReclaimedPagesPerSecond = counterRate(previous.Host.MemoryReclaimKswapdReclaimedPagesTotal, current.Host.MemoryReclaimKswapdReclaimedPagesTotal, seconds)
	current.Host.MemoryReclaimDirectReclaimedPagesPerSecond = counterRate(previous.Host.MemoryReclaimDirectReclaimedPagesTotal, current.Host.MemoryReclaimDirectReclaimedPagesTotal, seconds)
	current.Host.MemorySwapInPagesPerSecond = counterRate(previous.Host.MemorySwapInPagesTotal, current.Host.MemorySwapInPagesTotal, seconds)
	current.Host.MemorySwapOutPagesPerSecond = counterRate(previous.Host.MemorySwapOutPagesTotal, current.Host.MemorySwapOutPagesTotal, seconds)
	current.Host.MemoryPageInPagesPerSecond = counterRate(previous.Host.MemoryPageInPagesTotal, current.Host.MemoryPageInPagesTotal, seconds)
	current.Host.MemoryPageOutPagesPerSecond = counterRate(previous.Host.MemoryPageOutPagesTotal, current.Host.MemoryPageOutPagesTotal, seconds)
}

func counterRate(previous, current *uint64, seconds float64) *float64 {
	if previous == nil || current == nil || *current < *previous || seconds <= 0 {
		return nil
	}
	rate := float64(*current-*previous) / seconds
	return &rate
}
