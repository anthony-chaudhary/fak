package modelperfobs

import (
	"errors"
	"fmt"
	"math"
	"strings"
)

const BandwidthSchema = "fak-model-bandwidth/1"

const MaxBandwidthSamples = 256

type RequestPhase string

const (
	PhasePrefill  RequestPhase = "prefill"
	PhaseDecode   RequestPhase = "decode"
	PhaseTransfer RequestPhase = "transfer"
	PhaseOther    RequestPhase = "other"
)

type RequestShape string

const (
	ShapeSmall  RequestShape = "small"
	ShapeMedium RequestShape = "medium"
	ShapeLarge  RequestShape = "large"
)

type Bottleneck string

const (
	BottleneckCapacity Bottleneck = "capacity-pressure"
	BottleneckMemory   Bottleneck = "memory-controller-saturation"
	BottleneckTransfer Bottleneck = "transfer-link"
	BottleneckCompute  Bottleneck = "compute"
	BottleneckUnknown  Bottleneck = "unknown"
)

type BandwidthProvenance struct {
	Source    string `json:"source"`
	Machine   string `json:"machine,omitempty"`
	Device    string `json:"device,omitempty"`
	Collector string `json:"collector,omitempty"`
	SampledAt string `json:"sampled_at,omitempty"`
}

type Rooflines struct {
	TheoreticalGBS         *float64 `json:"theoretical_gb_s,omitempty"`
	MeasuredSustainableGBS *float64 `json:"measured_sustainable_gb_s,omitempty"`
	SelectedGBS            *float64 `json:"selected_gb_s,omitempty"`
	SelectedSource         string   `json:"selected_source,omitempty"`
}

// BandwidthScope identifies whose counters a live rate represents. A phase
// label can temporally align a package/system observation with a request, but
// it does not narrow shared traffic to that process or model.
type BandwidthScope struct {
	Kind string `json:"kind"`
	ID   string `json:"id,omitempty"`
}

type LiveBandwidth struct {
	ReadGBS     *float64        `json:"read_gb_s,omitempty"`
	WriteGBS    *float64        `json:"write_gb_s,omitempty"`
	TotalGBS    *float64        `json:"total_gb_s,omitempty"`
	Utilization *float64        `json:"utilization,omitempty"`
	Scope       *BandwidthScope `json:"scope,omitempty"`
}

type RequestSignals struct {
	LatencyMS        *float64 `json:"latency_ms,omitempty"`
	TTFTMS           *float64 `json:"ttft_ms,omitempty"`
	TPOTMS           *float64 `json:"tpot_ms,omitempty"`
	PromptTokens     *int64   `json:"prompt_tokens,omitempty"`
	CompletionTokens *int64   `json:"completion_tokens,omitempty"`
}

type DeviceSignals struct {
	CoreClockMHz                *float64 `json:"core_clock_mhz,omitempty"`
	MemoryClockMHz              *float64 `json:"memory_clock_mhz,omitempty"`
	PowerWatts                  *float64 `json:"power_watts,omitempty"`
	TemperatureC                *float64 `json:"temperature_c,omitempty"`
	ComputeUtilization          *float64 `json:"compute_utilization,omitempty"`
	MemoryControllerUtilization *float64 `json:"memory_controller_utilization,omitempty"`
	Throttling                  *bool    `json:"throttling,omitempty"`
}

type CapacitySignals struct {
	UsedBytes   *uint64  `json:"used_bytes,omitempty"`
	TotalBytes  *uint64  `json:"total_bytes,omitempty"`
	Utilization *float64 `json:"utilization,omitempty"`
}

type TransferSignals struct {
	HostToDeviceGBS *float64 `json:"host_to_device_gb_s,omitempty"`
	DeviceToHostGBS *float64 `json:"device_to_host_gb_s,omitempty"`
	LinkRoofGBS     *float64 `json:"link_roof_gb_s,omitempty"`
	Utilization     *float64 `json:"utilization,omitempty"`
}

type SoftwareTraffic struct {
	LogicalBytes       *uint64  `json:"logical_bytes,omitempty"`
	PhysicalReadBytes  *uint64  `json:"physical_read_bytes,omitempty"`
	PhysicalWriteBytes *uint64  `json:"physical_write_bytes,omitempty"`
	Amplification      *float64 `json:"amplification,omitempty"`
}

type BandwidthSample struct {
	Phase      RequestPhase        `json:"phase"`
	Shape      RequestShape        `json:"shape"`
	Provenance BandwidthProvenance `json:"provenance"`
	Rooflines  Rooflines           `json:"rooflines"`
	Live       LiveBandwidth       `json:"live"`
	Request    RequestSignals      `json:"request"`
	Device     DeviceSignals       `json:"device"`
	Capacity   CapacitySignals     `json:"capacity"`
	Transfer   TransferSignals     `json:"transfer"`
	Software   SoftwareTraffic     `json:"software"`
	Host       HostSignals         `json:"host"`
}

type TriggerConfig struct {
	SymptomWindow       int     `json:"symptom_window"`
	ResourceWindow      int     `json:"resource_window"`
	LatencyThresholdMS  float64 `json:"latency_threshold_ms"`
	ResourceUtilization float64 `json:"resource_utilization"`
}

type TriggerState struct {
	SymptomStreak  int  `json:"symptom_streak"`
	ResourceStreak int  `json:"resource_streak"`
	Triggered      bool `json:"triggered"`
}

type BandwidthObservation struct {
	Schema      string              `json:"schema"`
	Engine      string              `json:"engine"`
	Phase       RequestPhase        `json:"phase"`
	Shape       RequestShape        `json:"shape"`
	Provenance  BandwidthProvenance `json:"provenance"`
	Rooflines   Rooflines           `json:"rooflines"`
	Live        LiveBandwidth       `json:"live"`
	Request     RequestSignals      `json:"request"`
	Device      DeviceSignals       `json:"device"`
	Capacity    CapacitySignals     `json:"capacity"`
	Transfer    TransferSignals     `json:"transfer"`
	Software    SoftwareTraffic     `json:"software"`
	Host        HostSignals         `json:"host"`
	Bottleneck  Bottleneck          `json:"bottleneck"`
	DeepCapture TriggerState        `json:"deep_capture"`
}

type BandwidthCapture struct {
	Schema  string            `json:"schema,omitempty"`
	Engine  string            `json:"engine"`
	Trigger TriggerConfig     `json:"trigger"`
	Samples []BandwidthSample `json:"samples"`
}

type BandwidthReport struct {
	Schema       string                 `json:"schema"`
	Engine       string                 `json:"engine"`
	Observations []BandwidthObservation `json:"observations"`
	Truncated    bool                   `json:"truncated"`
}

func AnalyzeBandwidth(c BandwidthCapture) (BandwidthReport, error) {
	if c.Engine != "fak-native" {
		return BandwidthReport{}, fmt.Errorf("engine %q is not fak-native", c.Engine)
	}
	if err := validateTrigger(c.Trigger); err != nil {
		return BandwidthReport{}, err
	}
	n := len(c.Samples)
	truncated := n > MaxBandwidthSamples
	if n > MaxBandwidthSamples {
		n = MaxBandwidthSamples
	}
	report := BandwidthReport{Schema: BandwidthSchema, Engine: c.Engine, Truncated: truncated}
	report.Observations = make([]BandwidthObservation, 0, n)
	var state TriggerState
	for i := 0; i < n; i++ {
		s := c.Samples[i]
		if err := validateSample(s); err != nil {
			return BandwidthReport{}, fmt.Errorf("sample %d: %w", i+1, err)
		}
		s.Rooflines = SelectRoofline(s.Rooflines)
		s.Live = deriveLiveUtilization(s.Live, s.Rooflines.SelectedGBS)
		s.Capacity = deriveCapacity(s.Capacity)
		s.Transfer = deriveTransfer(s.Transfer)
		s.Software = deriveAmplification(s.Software)
		bottleneck := ClassifyBottleneck(s)
		state = ObserveTrigger(state, c.Trigger, s)
		report.Observations = append(report.Observations, BandwidthObservation{
			Schema: BandwidthSchema, Engine: c.Engine, Phase: s.Phase, Shape: s.Shape,
			Provenance: s.Provenance, Rooflines: s.Rooflines, Live: s.Live, Request: s.Request,
			Device: s.Device, Capacity: s.Capacity, Transfer: s.Transfer, Software: s.Software, Host: s.Host,
			Bottleneck: bottleneck, DeepCapture: state,
		})
	}
	return report, nil
}

func SelectRoofline(r Rooflines) Rooflines {
	r.SelectedGBS, r.SelectedSource = nil, ""
	if positive(r.MeasuredSustainableGBS) {
		r.SelectedGBS = cloneFloat(r.MeasuredSustainableGBS)
		r.SelectedSource = "measured-sustainable"
	} else if positive(r.TheoreticalGBS) {
		r.SelectedGBS = cloneFloat(r.TheoreticalGBS)
		r.SelectedSource = "theoretical"
	}
	return r
}

func ClassifyBottleneck(s BandwidthSample) Bottleneck {
	if atLeast(s.Capacity.Utilization, .90) {
		return BottleneckCapacity
	}
	if atLeast(s.Transfer.Utilization, .85) {
		return BottleneckTransfer
	}
	if atLeast(s.Live.Utilization, .85) || atLeast(s.Device.MemoryControllerUtilization, .85) {
		return BottleneckMemory
	}
	if atLeast(s.Device.ComputeUtilization, .85) {
		return BottleneckCompute
	}
	return BottleneckUnknown
}

func ObserveTrigger(state TriggerState, cfg TriggerConfig, s BandwidthSample) TriggerState {
	if positive(s.Request.LatencyMS) && *s.Request.LatencyMS >= cfg.LatencyThresholdMS {
		state.SymptomStreak++
	} else {
		state.SymptomStreak = 0
	}
	resource := maxAvailable(s.Live.Utilization, s.Transfer.Utilization, s.Capacity.Utilization, s.Device.ComputeUtilization, s.Device.MemoryControllerUtilization)
	if resource != nil && *resource >= cfg.ResourceUtilization {
		state.ResourceStreak++
	} else {
		state.ResourceStreak = 0
	}
	state.Triggered = state.SymptomStreak >= cfg.SymptomWindow && state.ResourceStreak >= cfg.ResourceWindow
	return state
}

func validateTrigger(c TriggerConfig) error {
	if c.SymptomWindow < 1 || c.ResourceWindow < 1 {
		return errors.New("trigger windows must be at least one sample")
	}
	if c.LatencyThresholdMS <= 0 {
		return errors.New("latency threshold must be positive")
	}
	if c.ResourceUtilization <= 0 || c.ResourceUtilization > 1 {
		return errors.New("resource utilization threshold must be in (0,1]")
	}
	return nil
}

func validateSample(s BandwidthSample) error {
	switch s.Phase {
	case PhasePrefill, PhaseDecode, PhaseTransfer, PhaseOther:
	default:
		return fmt.Errorf("unbounded phase %q", s.Phase)
	}
	switch s.Shape {
	case ShapeSmall, ShapeMedium, ShapeLarge:
	default:
		return fmt.Errorf("unbounded shape %q", s.Shape)
	}
	if strings.TrimSpace(s.Provenance.Source) == "" {
		return errors.New("provenance source is required")
	}
	return nil
}

func deriveLiveUtilization(l LiveBandwidth, roof *float64) LiveBandwidth {
	if l.TotalGBS == nil && l.ReadGBS != nil && l.WriteGBS != nil {
		v := *l.ReadGBS + *l.WriteGBS
		l.TotalGBS = &v
	}
	if l.Utilization == nil && l.TotalGBS != nil && positive(roof) {
		v := *l.TotalGBS / *roof
		l.Utilization = &v
	}
	return l
}

func deriveCapacity(c CapacitySignals) CapacitySignals {
	if c.Utilization == nil && c.UsedBytes != nil && c.TotalBytes != nil && *c.TotalBytes > 0 {
		v := float64(*c.UsedBytes) / float64(*c.TotalBytes)
		c.Utilization = &v
	}
	return c
}

func deriveTransfer(t TransferSignals) TransferSignals {
	if t.Utilization == nil && positive(t.LinkRoofGBS) {
		var total float64
		available := false
		if t.HostToDeviceGBS != nil {
			total += *t.HostToDeviceGBS
			available = true
		}
		if t.DeviceToHostGBS != nil {
			total += *t.DeviceToHostGBS
			available = true
		}
		if available {
			v := total / *t.LinkRoofGBS
			t.Utilization = &v
		}
	}
	return t
}

func deriveAmplification(s SoftwareTraffic) SoftwareTraffic {
	if s.Amplification == nil && s.LogicalBytes != nil && *s.LogicalBytes > 0 && s.PhysicalReadBytes != nil && s.PhysicalWriteBytes != nil {
		v := float64(*s.PhysicalReadBytes+*s.PhysicalWriteBytes) / float64(*s.LogicalBytes)
		s.Amplification = &v
	}
	return s
}

func positive(v *float64) bool                   { return v != nil && *v > 0 && !math.IsNaN(*v) }
func atLeast(v *float64, threshold float64) bool { return v != nil && *v >= threshold }
func cloneFloat(v *float64) *float64 {
	if v == nil {
		return nil
	}
	n := *v
	return &n
}
func maxAvailable(values ...*float64) *float64 {
	var out *float64
	for _, v := range values {
		if v != nil && (out == nil || *v > *out) {
			out = v
		}
	}
	return cloneFloat(out)
}
