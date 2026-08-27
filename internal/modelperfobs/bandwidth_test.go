package modelperfobs

import (
	"encoding/json"
	"strings"
	"testing"
)

func fp(v float64) *float64 { return &v }
func ip(v int64) *int64     { return &v }
func up(v uint64) *uint64   { return &v }

func baseBandwidthSample() BandwidthSample {
	return BandwidthSample{
		Phase: PhaseDecode, Shape: ShapeMedium,
		Provenance: BandwidthProvenance{Source: "fixture", Collector: "unit-test"},
		Request:    RequestSignals{LatencyMS: fp(120), CompletionTokens: ip(16)},
	}
}

func TestBandwidthMissingCountersRemainUnavailable(t *testing.T) {
	report, err := AnalyzeBandwidth(BandwidthCapture{
		Engine: "fak-native", Trigger: TriggerConfig{SymptomWindow: 2, ResourceWindow: 2, LatencyThresholdMS: 100, ResourceUtilization: .8},
		Samples: []BandwidthSample{baseBandwidthSample()},
	})
	if err != nil {
		t.Fatal(err)
	}
	got := report.Observations[0]
	if got.Live.TotalGBS != nil || got.Live.Utilization != nil || got.Rooflines.SelectedGBS != nil {
		t.Fatalf("missing counters became available: %+v", got)
	}
	b, err := json.Marshal(got)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(b), `"total_gb_s":0`) || strings.Contains(string(b), `"utilization":0`) {
		t.Fatalf("missing counter serialized as zero: %s", b)
	}
}

func TestBandwidthBottleneckDistinctions(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*BandwidthSample)
		want   Bottleneck
	}{
		{"capacity", func(s *BandwidthSample) { s.Capacity.Utilization = fp(.92) }, BottleneckCapacity},
		{"memory", func(s *BandwidthSample) { s.Live.Utilization = fp(.91) }, BottleneckMemory},
		{"transfer", func(s *BandwidthSample) { s.Transfer.Utilization = fp(.90) }, BottleneckTransfer},
		{"compute", func(s *BandwidthSample) { s.Device.ComputeUtilization = fp(.95) }, BottleneckCompute},
		{"unknown", func(*BandwidthSample) {}, BottleneckUnknown},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := baseBandwidthSample()
			tt.mutate(&s)
			if got := ClassifyBottleneck(s); got != tt.want {
				t.Fatalf("got %q want %q", got, tt.want)
			}
		})
	}
}

func TestBandwidthRooflineSelection(t *testing.T) {
	r := SelectRoofline(Rooflines{TheoreticalGBS: fp(1000), MeasuredSustainableGBS: fp(760)})
	if r.SelectedGBS == nil || *r.SelectedGBS != 760 || r.SelectedSource != "measured-sustainable" {
		t.Fatalf("measured selection: %+v", r)
	}
	r = SelectRoofline(Rooflines{TheoreticalGBS: fp(1000)})
	if r.SelectedGBS == nil || *r.SelectedGBS != 1000 || r.SelectedSource != "theoretical" {
		t.Fatalf("theoretical fallback: %+v", r)
	}
}

func TestBandwidthDeepCaptureRequiresSustainedSymptomAndResource(t *testing.T) {
	cfg := TriggerConfig{SymptomWindow: 2, ResourceWindow: 3, LatencyThresholdMS: 100, ResourceUtilization: .8}
	var state TriggerState
	for i := 0; i < 2; i++ {
		s := baseBandwidthSample()
		s.Live.Utilization = fp(.9)
		state = ObserveTrigger(state, cfg, s)
	}
	if state.Triggered {
		t.Fatal("triggered before resource window")
	}
	s := baseBandwidthSample()
	s.Live.Utilization = fp(.9)
	state = ObserveTrigger(state, cfg, s)
	if !state.Triggered {
		t.Fatalf("did not trigger after both windows: %+v", state)
	}
	s.Request.LatencyMS = fp(20)
	state = ObserveTrigger(state, cfg, s)
	if state.Triggered {
		t.Fatal("trigger remained set after symptom cleared")
	}
}

func TestBandwidthBoundedCardinalityAndSchema(t *testing.T) {
	samples := make([]BandwidthSample, MaxBandwidthSamples+10)
	for i := range samples {
		samples[i] = baseBandwidthSample()
	}
	report, err := AnalyzeBandwidth(BandwidthCapture{
		Schema: BandwidthSchema, Engine: "fak-native",
		Trigger: TriggerConfig{SymptomWindow: 1, ResourceWindow: 1, LatencyThresholdMS: 100, ResourceUtilization: .8}, Samples: samples,
	})
	if err != nil {
		t.Fatal(err)
	}
	if report.Schema != BandwidthSchema || len(report.Observations) != MaxBandwidthSamples || !report.Truncated {
		t.Fatalf("report bounds/schema: %+v", report)
	}
	for _, row := range report.Observations {
		if row.Schema != BandwidthSchema {
			t.Fatalf("row schema %q", row.Schema)
		}
	}
}

func TestBandwidthRejectsForeignEngineAndUnboundedDimensions(t *testing.T) {
	cfg := TriggerConfig{SymptomWindow: 1, ResourceWindow: 1, LatencyThresholdMS: 100, ResourceUtilization: .8}
	if _, err := AnalyzeBandwidth(BandwidthCapture{Engine: "llama.cpp", Trigger: cfg}); err == nil {
		t.Fatal("foreign engine accepted")
	}
	s := baseBandwidthSample()
	s.Phase = "request-123"
	if _, err := AnalyzeBandwidth(BandwidthCapture{Engine: "fak-native", Trigger: cfg, Samples: []BandwidthSample{s}}); err == nil {
		t.Fatal("unbounded phase accepted")
	}
}

func TestBandwidthDerivesPortableSignalsOnlyWhenInputsExist(t *testing.T) {
	s := baseBandwidthSample()
	s.Rooflines = Rooflines{MeasuredSustainableGBS: fp(800)}
	s.Live = LiveBandwidth{ReadGBS: fp(500), WriteGBS: fp(220)}
	s.Capacity = CapacitySignals{UsedBytes: up(90), TotalBytes: up(100)}
	s.Transfer = TransferSignals{HostToDeviceGBS: fp(20), DeviceToHostGBS: fp(10), LinkRoofGBS: fp(100)}
	s.Software = SoftwareTraffic{LogicalBytes: up(100), PhysicalReadBytes: up(180), PhysicalWriteBytes: up(20)}
	report, err := AnalyzeBandwidth(BandwidthCapture{Engine: "fak-native", Trigger: TriggerConfig{SymptomWindow: 1, ResourceWindow: 1, LatencyThresholdMS: 100, ResourceUtilization: .8}, Samples: []BandwidthSample{s}})
	if err != nil {
		t.Fatal(err)
	}
	got := report.Observations[0]
	if got.Live.TotalGBS == nil || *got.Live.TotalGBS != 720 || got.Live.Utilization == nil || *got.Live.Utilization != .9 {
		t.Fatalf("live derivation: %+v", got.Live)
	}
	if got.Capacity.Utilization == nil || *got.Capacity.Utilization != .9 {
		t.Fatalf("capacity: %+v", got.Capacity)
	}
	if got.Transfer.Utilization == nil || *got.Transfer.Utilization != .3 {
		t.Fatalf("transfer: %+v", got.Transfer)
	}
	if got.Software.Amplification == nil || *got.Software.Amplification != 2 {
		t.Fatalf("software: %+v", got.Software)
	}
}
