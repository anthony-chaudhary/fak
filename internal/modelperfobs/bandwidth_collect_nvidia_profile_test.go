package modelperfobs

import (
	"fmt"
	"os"
	"strings"
	"testing"
	"time"
)

func nvidiaProfileTestOptions() NVIDIAProfileOptions {
	return NVIDIAProfileOptions{
		Phase:             PhaseDecode,
		Shape:             ShapeLarge,
		Device:            "NVIDIA H100 80GB HBM3 (0)",
		CaptureStartedAt:  time.Date(2026, 8, 27, 10, 0, 0, 0, time.UTC),
		CaptureEndedAt:    time.Date(2026, 8, 27, 10, 1, 0, 0, time.UTC),
		TheoreticalGBS:    fp(3350),
		MeasuredDeviceGBS: fp(1250),
	}
}

func TestImportNVIDIAProfileAggregatesLaunchesOnce(t *testing.T) {
	f, err := os.Open("testdata/nvidia-hbm-ncu.csv")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	got, err := ImportNVIDIAProfile(f, nvidiaProfileTestOptions())
	if err != nil {
		t.Fatal(err)
	}
	if !got.Availability.DRAMCounters || got.ProfileReceipt == nil {
		t.Fatalf("availability=%+v receipt=%+v", got.Availability, got.ProfileReceipt)
	}
	receipt := got.ProfileReceipt
	if receipt.Engine != "fak-native" || receipt.EngineEvidence != "operator-asserted-not-proven-by-csv" || receipt.Profiler != "nvidia-nsight-compute" {
		t.Fatalf("receipt provenance=%+v", receipt)
	}
	if receipt.Device != "NVIDIA H100 80GB HBM3 (0)" || receipt.ProcessID != "4242" || receipt.Process != "fak" || receipt.LaunchCount != 2 {
		t.Fatalf("receipt identity=%+v", receipt)
	}
	if receipt.CumulativeReadBytes == nil || *receipt.CumulativeReadBytes != 1_500_000_000 || receipt.CumulativeWriteBytes == nil || *receipt.CumulativeWriteBytes != 500_000_000 || receipt.CumulativeDurationNS == nil || *receipt.CumulativeDurationNS != 2_000_000 {
		t.Fatalf("receipt counters=%+v", receipt)
	}
	if strings.Join(receipt.CounterSource, ",") != strings.Join(nvidiaProfileMetrics, ",") || receipt.AggregationScope != "sum-cumulative-bytes-over-sum-profiled-kernel-active-nanoseconds" {
		t.Fatalf("receipt counter provenance=%+v", receipt)
	}
	obs := got.Report.Observations[0]
	if obs.Live.ReadGBS == nil || *obs.Live.ReadGBS != 750 || obs.Live.WriteGBS == nil || *obs.Live.WriteGBS != 250 || obs.Live.TotalGBS == nil || *obs.Live.TotalGBS != 1000 {
		t.Fatalf("live bandwidth=%+v", obs.Live)
	}
	if obs.Live.Utilization == nil || *obs.Live.Utilization != .8 || obs.Rooflines.SelectedSource != "measured-sustainable" {
		t.Fatalf("roofline=%+v live=%+v", obs.Rooflines, obs.Live)
	}
	if receipt.DeviceRooflineGBS == nil || *receipt.DeviceRooflineGBS != 1250 || receipt.DeviceRooflineEvidence != "operator-supplied-matched-device-measurement" {
		t.Fatalf("device roofline provenance=%+v", receipt)
	}
	if obs.Provenance.SampledAt != "2026-08-27T10:01:00Z" {
		t.Fatalf("capture time was replaced by parse time: %+v", obs.Provenance)
	}
}

func TestImportNVIDIAProfileUnsupportedDirectionIsUnavailableNotZero(t *testing.T) {
	csv := `"ID","Process ID","Process Name","Device","Metric Name","Metric Unit","Metric Value"
"1","7","fak","GPU-0","dram__bytes_read.sum","byte","N/A"
"1","7","fak","GPU-0","dram__bytes_write.sum","byte","500"
"1","7","fak","GPU-0","gpu__time_duration.sum","nsecond","1000"`
	o := nvidiaProfileTestOptions()
	o.Device = "GPU-0"
	got, err := ImportNVIDIAProfile(strings.NewReader(csv), o)
	if err != nil {
		t.Fatal(err)
	}
	if !got.Availability.DRAMCounters || got.Availability.DeviceCounters || got.ProfileReceipt.CumulativeReadBytes != nil || got.ProfileReceipt.CumulativeWriteBytes == nil || *got.ProfileReceipt.CumulativeWriteBytes != 500 || got.ProfileReceipt.CumulativeDurationNS == nil || *got.ProfileReceipt.CumulativeDurationNS != 1000 {
		t.Fatalf("unsupported counters must be omitted: %+v", got)
	}
	live := got.Report.Observations[0].Live
	if live.ReadGBS != nil || live.WriteGBS == nil || *live.WriteGBS != .5 || live.TotalGBS != nil || live.Utilization != nil {
		t.Fatalf("unsupported bandwidth became zero/value: %+v", live)
	}
}

func TestImportNVIDIAProfilePreservesDecimalDuration(t *testing.T) {
	csv := `"ID","Process ID","Process Name","Device","Metric Name","Metric Unit","Metric Value"
"1","7","fak","GPU-0","dram__bytes_read.sum","byte","1,001"
"1","7","fak","GPU-0","dram__bytes_write.sum","byte","500.0"
"1","7","fak","GPU-0","gpu__time_duration.sum","nsecond","1,000.5"`
	o := nvidiaProfileTestOptions()
	o.Device = "GPU-0"
	got, err := ImportNVIDIAProfile(strings.NewReader(csv), o)
	if err != nil {
		t.Fatal(err)
	}
	if got.ProfileReceipt.CumulativeDurationNS == nil || *got.ProfileReceipt.CumulativeDurationNS != 1000.5 {
		t.Fatalf("decimal duration was not preserved: %+v", got.ProfileReceipt)
	}
	live := got.Report.Observations[0].Live
	if live.ReadGBS == nil || *live.ReadGBS != 1001.0/1000.5 || live.WriteGBS == nil || *live.WriteGBS != 500.0/1000.5 {
		t.Fatalf("decimal duration bandwidth=%+v", live)
	}
	badBytes := strings.Replace(csv, `"500.0"`, `"500.5"`, 1)
	if _, err := ImportNVIDIAProfile(strings.NewReader(badBytes), o); err == nil || !strings.Contains(err.Error(), "not a non-negative uint64") {
		t.Fatalf("fractional byte counter error=%v", err)
	}
}

func TestImportNVIDIAProfileScrubsCaptureHostAndSeparatesDeviceLabel(t *testing.T) {
	csv := `"ID","Process ID","Process Name","Host Name","Device","Metric Name","Metric Unit","Metric Value"
"1","7","fak","private-host","0","dram__bytes_read.sum","byte","100"
"1","7","fak","private-host","0","dram__bytes_write.sum","byte","50"
"1","7","fak","private-host","0","gpu__time_duration.sum","ns","10"`
	o := nvidiaProfileTestOptions()
	o.Device = "NVIDIA L4"
	got, err := ImportNVIDIAProfile(strings.NewReader(csv), o)
	if err != nil {
		t.Fatal(err)
	}
	if got.MachineClass != "" || got.Capture.Samples[0].Provenance.Machine != "" || got.ProfileReceipt.ProfilerDevice != "0" || got.ProfileReceipt.Device != "NVIDIA L4" {
		t.Fatalf("profile identity=%+v provenance=%+v", got.ProfileReceipt, got.Capture.Samples[0].Provenance)
	}
	encoded := fmt.Sprintf("%+v", got)
	if strings.Contains(encoded, "private-host") {
		t.Fatalf("capture host leaked: %s", encoded)
	}
}

func TestImportNVIDIAProfileRejectsPartialAndMixedDeviceCaptures(t *testing.T) {
	partial := `"ID","Process ID","Process Name","Device","Metric Name","Metric Unit","Metric Value"
"1","7","fak","GPU-0","dram__bytes_read.sum","byte","100"
"1","7","fak","GPU-0","gpu__time_duration.sum","ns","10"`
	o := nvidiaProfileTestOptions()
	o.Device = "GPU-0"
	if _, err := ImportNVIDIAProfile(strings.NewReader(partial), o); err == nil || !strings.Contains(err.Error(), "missing a required") {
		t.Fatalf("partial capture error=%v", err)
	}
	mixed := `"ID","Process ID","Process Name","Device","Metric Name","Metric Unit","Metric Value"
"1","7","fak","GPU-0","dram__bytes_read.sum","byte","100"
"1","7","fak","GPU-1","dram__bytes_write.sum","byte","50"
"1","7","fak","GPU-0","gpu__time_duration.sum","ns","10"`
	if _, err := ImportNVIDIAProfile(strings.NewReader(mixed), o); err == nil || !strings.Contains(err.Error(), "mixed-device") {
		t.Fatalf("mixed capture error=%v", err)
	}
}
