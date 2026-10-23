package devcmd

import (
	"bytes"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// fak-test:runtime fast est=50ms lane=default
func TestRunCUDAGPUDirectHardwareTruth(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	devices := filepath.Join(root, "bus", "pci", "devices")
	if err := os.MkdirAll(devices, 0755); err != nil {
		t.Fatal(err)
	}
	for _, mode := range []string{"inspect", "audit"} {
		for _, jsonOut := range []bool{false, true} {
			var stdout, stderr bytes.Buffer
			args := []string{mode, "--sysfs", root}
			if jsonOut {
				args = append(args, "--json")
			}
			wantCode := 0
			if mode == "audit" {
				wantCode = 1
			}
			if code := RunCUDAGPUDirect(&stdout, &stderr, args); code != wantCode {
				t.Fatalf("%v: exit %d, want %d: %s", args, code, wantCode, &stderr)
			}
			for _, invented := range []string{"RTX 5090", "990 PRO", "Ryzen", "Aorus", "128GB", "[PASS]", `"status": "PASS"`} {
				if strings.Contains(stdout.String(), invented) {
					t.Errorf("%v fabricated %q: %s", args, invented, &stdout)
				}
			}
			if !strings.Contains(stdout.String(), "UNVERIFIED") || !strings.Contains(stdout.String(), "UNKNOWN") {
				t.Errorf("%v did not preserve uncertainty: %s", args, &stdout)
			}
			if jsonOut && mode == "inspect" {
				var out CUDAInspectOutput
				if err := json.Unmarshal(stdout.Bytes(), &out); err != nil {
					t.Fatal(err)
				}
				if len(out.GPUs) != 0 || len(out.NVMeDevices) != 0 || len(out.Routes) != 0 || out.Host.CPU != "" {
					t.Errorf("empty inventory replaced by fixtures: %+v", out)
				}
			}
			if jsonOut && mode == "audit" {
				var out CUDAHardwareAuditReport
				if err := json.Unmarshal(stdout.Bytes(), &out); err != nil {
					t.Fatal(err)
				}
				if out.Healthy || out.Verdict != "UNVERIFIED" || len(out.Checks) < 5 {
					t.Errorf("unqualified audit: %+v", out)
				}
			}
		}
	}
	// Exercise the real sysfs parser with one observed GPU and no storage.
	gpuDir := filepath.Join(devices, "0000_09_00.0")
	if err := os.Mkdir(gpuDir, 0755); err != nil {
		t.Fatal(err)
	}
	for name, value := range map[string]string{"address": "0000:09:00.0", "vendor": "0x10de", "class": "0x030000", "device": "0xabcd", "label": "Observed GPU"} {
		if err := os.WriteFile(filepath.Join(gpuDir, name), []byte(value), 0644); err != nil {
			t.Fatal(err)
		}
	}
	var stdout, stderr bytes.Buffer
	if code := RunCUDAGPUDirect(&stdout, &stderr, []string{"inspect", "--sysfs", root, "--json"}); code != 0 {
		t.Fatalf("partial inventory exit %d: %s", code, &stderr)
	}
	var out CUDAInspectOutput
	if err := json.Unmarshal(stdout.Bytes(), &out); err != nil {
		t.Fatal(err)
	}
	if len(out.GPUs) != 1 || out.GPUs[0].DeviceName != "Observed GPU" || len(out.NVMeDevices) != 0 || len(out.Routes) != 0 {
		t.Errorf("partial inventory not preserved: %+v", out)
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestRunCUDAGPUDirectProbeErrors(t *testing.T) {
	t.Parallel()
	for _, mode := range []string{"inspect", "audit"} {
		var stdout, stderr bytes.Buffer
		missing := filepath.Join(t.TempDir(), "missing")
		if code := RunCUDAGPUDirect(&stdout, &stderr, []string{mode, "--sysfs", missing, "--json"}); code != 1 {
			t.Fatalf("%s missing sysfs exit %d, want 1", mode, code)
		}
		var out map[string]any
		if err := json.Unmarshal(stdout.Bytes(), &out); err != nil {
			t.Fatal(err)
		}
		if out["probe_error"] == "" || out["probe_error"] == nil || stderr.Len() == 0 || out["verdict"] != "UNVERIFIED" {
			t.Errorf("%s lost probe failure: %s stderr=%s", mode, &stdout, &stderr)
		}
	}
	// A failed probe may still carry useful inventory; do not replace or drop it.
	probe := func(string) (*compute.CUDAPCIeTopologyDiscovery, error) {
		d := compute.NewCUDAPCIeTopologyDiscovery("")
		d.GPUs = append(d.GPUs, compute.CUDADeviceNode{DeviceName: "Partial observed GPU"})
		return d, errors.New("remaining inventory unreadable")
	}
	var stdout, stderr bytes.Buffer
	if code := runCUDAGPUDirectWithProbe(&stdout, &stderr, []string{"inspect", "--json"}, probe); code != 1 {
		t.Fatalf("partial error exit %d, want 1", code)
	}
	if !strings.Contains(stdout.String(), "Partial observed GPU") || !strings.Contains(stdout.String(), "remaining inventory unreadable") {
		t.Fatalf("partial probe result lost: %s", &stdout)
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestRunCUDAGPUDirectSimulationBoundary(t *testing.T) {
	t.Parallel()
	noProbe := func(string) (*compute.CUDAPCIeTopologyDiscovery, error) {
		t.Fatal("simulation or invalid usage probed hardware")
		return nil, nil
	}
	for _, args := range [][]string{
		{"bench"}, {"qwen38"}, {"qwen38-bench"}, {"nonexistent_mode"},
		{"inspect", "extra"}, {"--mode", "bench"}, {"--unknown"},
	} {
		var stdout, stderr bytes.Buffer
		if code := runCUDAGPUDirectWithProbe(&stdout, &stderr, args, noProbe); code != 2 {
			t.Errorf("%v: exit %d, want usage exit 2: %s", args, code, &stderr)
		}
	}
	for _, args := range [][]string{
		{"inspect", "--simulate"}, {"audit", "--simulate"}, {"bench", "--simulate"}, {"--mode", "bench", "--simulate"},
		{"qwen38", "--simulate"}, {"qwen38-bench", "--simulate"}, {"qwen38-sim"},
	} {
		var stdout, stderr bytes.Buffer
		args = append(args, "--sysfs", filepath.Join(t.TempDir(), "must-not-probe"), "--json")
		code := runCUDAGPUDirectWithProbe(&stdout, &stderr, args, noProbe)
		wantCode := 0
		if args[0] == "audit" {
			wantCode = 1
		}
		if code != wantCode {
			t.Fatalf("%v: exit %d: %s", args, code, &stderr)
		}
		var out map[string]any
		if err := json.Unmarshal(stdout.Bytes(), &out); err != nil {
			t.Fatal(err)
		}
		if out["provenance"] != "MODELED" || out["is_physical_silicon"] != false || out["verdict"] != "MODELED" {
			t.Errorf("missing simulation provenance: %s", &stdout)
		}
		if out["throughput_gbps"] != nil && (out["throughput_gbps"] != float64(0) || out["iops"] != float64(0) || out["zero_copy_verified"] != false) {
			t.Errorf("software queue claimed hardware performance: %s", &stdout)
		}
		if out["direct_dma_bandwidth_gbps"] != nil && out["direct_dma_bandwidth_gbps"] != float64(0) {
			t.Errorf("software model claimed measured bandwidth: %s", &stdout)
		}
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestCUDAQwen38ReceiptDoesNotClaimPerformance(t *testing.T) {
	t.Parallel()
	receipt := cudaQwen38ModeledReceipt("fixture", 1234, 0)
	if receipt.Provenance != "MODELED" || receipt.Verdict != "MODELED" || receipt.IsPhysicalSilicon || receipt.BytesMoved != 1234 {
		t.Fatalf("invalid model receipt: %+v", receipt)
	}
	if receipt.DirectDMABandwidthGBps != 0 || receipt.SpeedupVsBaseline != (Qwen38BenchSpeedupMetrics{}) || receipt.SpeedupVsReference != (Qwen38BenchSpeedupMetrics{}) {
		t.Errorf("unmeasured performance or comparison: %+v", receipt)
	}
	for _, arm := range []Qwen38BenchArmMetrics{receipt.Baseline, receipt.Reference, receipt.FakNative, receipt.Arms["baseline"], receipt.Arms["reference"], receipt.Arms["fak_native"]} {
		if arm.BandwidthGBps != 0 || arm.TTFTMS != 0 || arm.PrefillTokPerS != 0 || arm.DecodeTokPerS != 0 || arm.DecodeITLP50MS != 0 || arm.DecodeITLP95MS != 0 {
			t.Errorf("unmeasured arm contains performance: %+v", arm)
		}
	}
	data, err := json.Marshal(receipt)
	if err != nil {
		t.Fatal(err)
	}
	for _, claim := range []string{"P2PDMA validated", "0.42s", "7.1 GB/s", "restoration verified", "bit-exact hybrid state verified"} {
		if strings.Contains(string(data), claim) {
			t.Errorf("receipt retained false claim %q: %s", claim, data)
		}
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestCUDAHardwareAuditHealthyRequiresAllPass(t *testing.T) {
	t.Parallel()
	for _, status := range []string{"UNKNOWN", "WARN", "FAIL", "MODELED", ""} {
		verdict, healthy := cudaAuditVerdict([]CUDAAuditCheck{{Status: "PASS"}, {Status: status}})
		if healthy || verdict == "PASS" {
			t.Errorf("%q produced healthy=%v verdict=%q", status, healthy, verdict)
		}
	}
	if verdict, healthy := cudaAuditVerdict(nil); healthy || verdict != "UNVERIFIED" {
		t.Errorf("empty audit produced healthy=%v verdict=%q", healthy, verdict)
	}
	if verdict, healthy := cudaAuditVerdict([]CUDAAuditCheck{{Status: "PASS"}}); !healthy || verdict != "PASS" {
		t.Errorf("all observed PASS checks produced healthy=%v verdict=%q", healthy, verdict)
	}
}
