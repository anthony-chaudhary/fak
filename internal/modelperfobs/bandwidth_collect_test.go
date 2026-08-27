package modelperfobs

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"
)

func TestCollectionBounds(t *testing.T) {
	base := CollectionOptions{Count: 1, Interval: 10 * time.Millisecond, Phase: PhaseDecode, Shape: ShapeSmall}
	if err := ValidateCollectionOptions(base); err != nil {
		t.Fatal(err)
	}
	base.Count = 121
	if err := ValidateCollectionOptions(base); err == nil {
		t.Fatal("expected count bound")
	}
	base.Count = 1
	base.Interval = time.Millisecond
	if err := ValidateCollectionOptions(base); err == nil {
		t.Fatal("expected interval bound")
	}
	base.Interval = 10 * time.Millisecond
	base.NVIDIADevice = "0"
	base.AMDDevice = "0"
	if err := ValidateCollectionOptions(base); err == nil {
		t.Fatal("expected mutually exclusive device selectors")
	}
}
func TestCollectBandwidthPreservesUnavailableDRAM(t *testing.T) {
	r, err := CollectBandwidth(context.Background(), CollectionOptions{Count: 1, Interval: 10 * time.Millisecond, Phase: PhaseDecode, Shape: ShapeSmall, TheoreticalGBS: fp(100)})
	if err != nil {
		t.Fatal(err)
	}
	if r.Engine != "fak-native" || len(r.Report.Observations) != 1 {
		t.Fatalf("%+v", r)
	}
	if r.Availability.DRAMCounters || r.Report.Observations[0].Live.TotalGBS != nil {
		t.Fatal("host/process signals mislabeled as DRAM bandwidth")
	}
	b, _ := json.Marshal(r)
	if strings.Contains(string(b), `"total_gb_s":0`) {
		t.Fatal(string(b))
	}
	if r.Report.Observations[0].Rooflines.SelectedSource != "theoretical" {
		t.Fatal("roofline not selected")
	}
}

func TestDeriveHostPressureRatesNeedsTwoMonotonicSamples(t *testing.T) {
	minor1, minor2, major1, major2 := uint64(10), uint64(14), uint64(2), uint64(3)
	prev := BandwidthSample{Provenance: BandwidthProvenance{SampledAt: "2026-08-27T00:00:00Z"}, Host: HostSignals{ProcessMinorFaults: &minor1, ProcessMajorFaults: &major1}}
	curr := BandwidthSample{Provenance: BandwidthProvenance{SampledAt: "2026-08-27T00:00:02Z"}, Host: HostSignals{ProcessMinorFaults: &minor2, ProcessMajorFaults: &major2}}
	deriveHostPressureRates(&prev, &curr)
	if curr.Host.ProcessMinorFaultsPerSecond == nil || *curr.Host.ProcessMinorFaultsPerSecond != 2 || curr.Host.ProcessMajorFaultsPerSecond == nil || *curr.Host.ProcessMajorFaultsPerSecond != .5 {
		t.Fatalf("%+v", curr.Host)
	}
	if prev.Host.ProcessMinorFaultsPerSecond != nil {
		t.Fatal("first sample must not fabricate a rate")
	}
}
