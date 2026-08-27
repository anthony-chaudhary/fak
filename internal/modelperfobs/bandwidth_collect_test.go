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

func TestDeriveHostPressureRatesForLinuxCumulativeCounters(t *testing.T) {
	some1, some2 := uint64(1_000_000), uint64(1_500_000)
	full1, full2 := uint64(100_000), uint64(120_000)
	scanKswapd1, scanKswapd2 := uint64(10), uint64(30)
	scanDirect1, scanDirect2 := uint64(4), uint64(10)
	reclaimKswapd1, reclaimKswapd2 := uint64(20), uint64(30)
	reclaimDirect1, reclaimDirect2 := uint64(8), uint64(12)
	swapIn1, swapIn2 := uint64(2), uint64(8)
	swapOut1, swapOut2 := uint64(5), uint64(15)
	prev := BandwidthSample{
		Provenance: BandwidthProvenance{SampledAt: "2026-08-27T00:00:00Z"},
		Host: HostSignals{
			MemoryPressureSomeTotalStallMicroseconds: &some1,
			MemoryPressureFullTotalStallMicroseconds: &full1,
			MemoryReclaimKswapdScannedPagesTotal:     &scanKswapd1,
			MemoryReclaimDirectScannedPagesTotal:     &scanDirect1,
			MemoryReclaimKswapdReclaimedPagesTotal:   &reclaimKswapd1,
			MemoryReclaimDirectReclaimedPagesTotal:   &reclaimDirect1,
			MemorySwapInPagesTotal:                   &swapIn1,
			MemorySwapOutPagesTotal:                  &swapOut1,
		},
	}
	curr := BandwidthSample{
		Provenance: BandwidthProvenance{SampledAt: "2026-08-27T00:00:02Z"},
		Host: HostSignals{
			MemoryPressureSomeTotalStallMicroseconds: &some2,
			MemoryPressureFullTotalStallMicroseconds: &full2,
			MemoryReclaimKswapdScannedPagesTotal:     &scanKswapd2,
			MemoryReclaimDirectScannedPagesTotal:     &scanDirect2,
			MemoryReclaimKswapdReclaimedPagesTotal:   &reclaimKswapd2,
			MemoryReclaimDirectReclaimedPagesTotal:   &reclaimDirect2,
			MemorySwapInPagesTotal:                   &swapIn2,
			MemorySwapOutPagesTotal:                  &swapOut2,
		},
	}

	deriveHostPressureRates(&prev, &curr)

	assertRate(t, "some stall", curr.Host.MemoryPressureSomeStallMicrosecondsPerSecond, 250_000)
	assertRate(t, "full stall", curr.Host.MemoryPressureFullStallMicrosecondsPerSecond, 10_000)
	assertRate(t, "kswapd scanned", curr.Host.MemoryReclaimKswapdScannedPagesPerSecond, 10)
	assertRate(t, "direct scanned", curr.Host.MemoryReclaimDirectScannedPagesPerSecond, 3)
	assertRate(t, "kswapd reclaimed", curr.Host.MemoryReclaimKswapdReclaimedPagesPerSecond, 5)
	assertRate(t, "direct reclaimed", curr.Host.MemoryReclaimDirectReclaimedPagesPerSecond, 2)
	assertRate(t, "swap in", curr.Host.MemorySwapInPagesPerSecond, 3)
	assertRate(t, "swap out", curr.Host.MemorySwapOutPagesPerSecond, 5)
}

func TestDeriveHostPressureRatesOmitsCounterResets(t *testing.T) {
	before, after := uint64(100), uint64(4)
	prev := BandwidthSample{
		Provenance: BandwidthProvenance{SampledAt: "2026-08-27T00:00:00Z"},
		Host: HostSignals{
			MemoryPressureSomeTotalStallMicroseconds: &before,
			MemoryReclaimDirectScannedPagesTotal:     &before,
			MemorySwapOutPagesTotal:                  &before,
		},
	}
	curr := BandwidthSample{
		Provenance: BandwidthProvenance{SampledAt: "2026-08-27T00:00:01Z"},
		Host: HostSignals{
			MemoryPressureSomeTotalStallMicroseconds: &after,
			MemoryReclaimDirectScannedPagesTotal:     &after,
			MemorySwapOutPagesTotal:                  &after,
		},
	}

	deriveHostPressureRates(&prev, &curr)

	if curr.Host.MemoryPressureSomeStallMicrosecondsPerSecond != nil ||
		curr.Host.MemoryReclaimDirectScannedPagesPerSecond != nil ||
		curr.Host.MemorySwapOutPagesPerSecond != nil {
		t.Fatalf("counter resets must be omitted: %+v", curr.Host)
	}
}

func TestLinuxMemoryPressureJSONDoesNotClaimDRAMBandwidth(t *testing.T) {
	stall, reclaimed := uint64(42), uint64(7)
	host := HostSignals{
		MemoryPressureSomeTotalStallMicroseconds: &stall,
		MemoryReclaimKswapdReclaimedPagesTotal:   &reclaimed,
	}
	encoded, err := json.Marshal(host)
	if err != nil {
		t.Fatal(err)
	}
	got := string(encoded)
	if !strings.Contains(got, `"memory_pressure_some_total_stall_microseconds":42`) ||
		!strings.Contains(got, `"memory_reclaim_kswapd_reclaimed_pages_total":7`) {
		t.Fatalf("missing explicit pressure counters: %s", got)
	}
	if strings.Contains(got, "gb_s") || strings.Contains(got, "dram") {
		t.Fatalf("pressure counters mislabeled as DRAM bandwidth: %s", got)
	}
}

func assertRate(t *testing.T, name string, got *float64, want float64) {
	t.Helper()
	if got == nil || *got != want {
		t.Fatalf("%s=%v want %v", name, got, want)
	}
}
