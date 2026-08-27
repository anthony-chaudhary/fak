package modelperfobs

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"strings"
	"testing"
	"time"
)

func TestCollectAMDDeviceSnapshotUsesTrueROCmRates(t *testing.T) {
	amdFixture, err := os.ReadFile("testdata/amd-smi-metric.json")
	if err != nil {
		t.Fatal(err)
	}
	rdcFixture, err := os.ReadFile("testdata/rdci-memory-rates.txt")
	if err != nil {
		t.Fatal(err)
	}
	oldAMD, oldRDC := runAMDSMI, runRDCI
	t.Cleanup(func() { runAMDSMI, runRDCI = oldAMD, oldRDC })
	runAMDSMI = func(_ context.Context, args ...string) ([]byte, error) {
		joined := strings.Join(args, " ")
		for _, want := range []string{"metric", "--gpu 0", "--usage", "--mem-usage", "--json"} {
			if !strings.Contains(joined, want) {
				t.Fatalf("amd-smi args %q missing %q", joined, want)
			}
		}
		return amdFixture, nil
	}
	runRDCI = func(_ context.Context, args ...string) ([]byte, error) {
		joined := strings.Join(args, " ")
		for _, want := range []string{"dmon", "-i 0", rdcMemoryRateFields, "-c 1", "-d 100"} {
			if !strings.Contains(joined, want) {
				t.Fatalf("rdci args %q missing %q", joined, want)
			}
		}
		if strings.Contains(joined, "RDC_FI_GPU_MEMORY_CUR_BANDWIDTH") {
			t.Fatalf("rdci args use utilization-times-roofline convenience field: %q", joined)
		}
		return rdcFixture, nil
	}

	got, err := collectAMDDeviceSnapshot(context.Background(), "")
	if err != nil {
		t.Fatal(err)
	}
	if !got.available || !got.dramCounters || got.collector != "amd-smi+rdci-rocp" || got.provenanceSource != "live-amd-rocm" {
		t.Fatalf("collector/provenance = %+v", got)
	}
	if got.device.MemoryControllerUtilization == nil || *got.device.MemoryControllerUtilization != .81 {
		t.Fatalf("UMC activity = %+v", got.device.MemoryControllerUtilization)
	}
	if got.capacity.TotalBytes == nil || *got.capacity.TotalBytes != 65536*1024*1024 || got.capacity.UsedBytes == nil || *got.capacity.UsedBytes != 24576*1024*1024 {
		t.Fatalf("VRAM capacity = %+v", got.capacity)
	}
	if got.live.ReadGBS == nil || *got.live.ReadGBS != 842 || got.live.WriteGBS == nil || *got.live.WriteGBS != 126 {
		t.Fatalf("true ROCm rates = %+v", got.live)
	}
}

func TestCollectBandwidthPublishesAMDROCmProvenance(t *testing.T) {
	amdFixture, err := os.ReadFile("testdata/amd-smi-metric.json")
	if err != nil {
		t.Fatal(err)
	}
	rdcFixture, err := os.ReadFile("testdata/rdci-memory-rates.txt")
	if err != nil {
		t.Fatal(err)
	}
	oldAMD, oldRDC := runAMDSMI, runRDCI
	t.Cleanup(func() { runAMDSMI, runRDCI = oldAMD, oldRDC })
	runAMDSMI = func(context.Context, ...string) ([]byte, error) { return amdFixture, nil }
	runRDCI = func(context.Context, ...string) ([]byte, error) { return rdcFixture, nil }

	got, err := CollectBandwidth(context.Background(), CollectionOptions{Count: 1, Interval: 10 * time.Millisecond, Phase: PhaseDecode, Shape: ShapeMedium, AMDDevice: "0", MeasuredSustainableGBS: fp(1100)})
	if err != nil {
		t.Fatal(err)
	}
	if !got.Availability.DeviceCounters || !got.Availability.DRAMCounters || len(got.Report.Observations) != 1 {
		t.Fatalf("AMD availability/report = %+v", got)
	}
	row := got.Report.Observations[0]
	if row.Provenance.Source != "live-amd-rocm" || row.Provenance.Device != "AMD ROCm GPU 0" || !strings.Contains(row.Provenance.Collector, "amd-smi+rdci-rocp") {
		t.Fatalf("AMD provenance = %+v", row.Provenance)
	}
	if row.Live.ReadGBS == nil || *row.Live.ReadGBS != 842 || row.Live.WriteGBS == nil || *row.Live.WriteGBS != 126 || row.Live.TotalGBS == nil || *row.Live.TotalGBS != 968 {
		t.Fatalf("AMD live rates = %+v", row.Live)
	}
	if row.Capacity.TotalBytes == nil || row.Device.MemoryControllerUtilization == nil {
		t.Fatalf("capacity/controller signals lost: capacity=%+v device=%+v", row.Capacity, row.Device)
	}
	encoded, err := json.Marshal(row)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(encoded), `"process_read_bytes"`) && strings.Contains(string(encoded), `"live":{"process_`) {
		t.Fatalf("process I/O leaked into live DRAM bandwidth: %s", encoded)
	}
}

func TestCollectAMDDeviceSnapshotKeepsUnsupportedBandwidthUnavailable(t *testing.T) {
	oldAMD, oldRDC := runAMDSMI, runRDCI
	t.Cleanup(func() { runAMDSMI, runRDCI = oldAMD, oldRDC })
	runAMDSMI = func(context.Context, ...string) ([]byte, error) {
		return []byte(`[{"gpu":2,"usage":{"gfx_activity":{"value":4,"unit":"%"},"umc_activity":"N/A"},"mem_usage":{"total_vram":{"value":65536,"unit":"MB"},"used_vram":{"value":1024,"unit":"MB"}}}]`), nil
	}
	runRDCI = func(context.Context, ...string) ([]byte, error) {
		return []byte("GPU MEM_R_BW MEM_W_BW\n2 N/A N/A\n"), nil
	}
	got, err := collectAMDDeviceSnapshot(context.Background(), "2")
	if err != nil {
		t.Fatal(err)
	}
	if !got.available || got.dramCounters || got.live.ReadGBS != nil || got.live.WriteGBS != nil || got.device.MemoryControllerUtilization != nil {
		t.Fatalf("unsupported counters became zero/available: %+v", got)
	}
}

func TestCollectAMDDeviceSnapshotUnavailableIsNotZero(t *testing.T) {
	oldAMD, oldRDC := runAMDSMI, runRDCI
	t.Cleanup(func() { runAMDSMI, runRDCI = oldAMD, oldRDC })
	runAMDSMI = func(context.Context, ...string) ([]byte, error) { return nil, errors.New("not installed") }
	runRDCI = func(context.Context, ...string) ([]byte, error) { return nil, errors.New("not installed") }
	got, err := collectAMDDeviceSnapshot(context.Background(), "")
	if err != nil || got.available || got.dramCounters || got.live.TotalGBS != nil {
		t.Fatalf("got=%+v err=%v", got, err)
	}
}

func TestParseAMDSMIRejectsCapacityAsActivity(t *testing.T) {
	_, err := parseAMDSMIOutput([]byte(`[{"gpu":0,"usage":{"gfx_activity":{"value":4,"unit":"%"},"umc_activity":{"value":42,"unit":"MB"}},"mem_usage":{}}]`))
	if err == nil || !strings.Contains(err.Error(), "unexpected unit") {
		t.Fatalf("wrong-unit UMC counter accepted: %v", err)
	}
}

func TestParseRDCIMemoryRatesPreservesPartialSupport(t *testing.T) {
	got, err := parseRDCIMemoryRates([]byte("GPU MEM_R_BW MEM_W_BW\n0 500000 N/A\n"), "0")
	if err != nil {
		t.Fatal(err)
	}
	if got.ReadGBS == nil || *got.ReadGBS != 500 || got.WriteGBS != nil || got.TotalGBS != nil {
		t.Fatalf("partial rates = %+v", got)
	}
}
