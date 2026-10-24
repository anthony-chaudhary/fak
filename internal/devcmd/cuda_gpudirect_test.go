package devcmd

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// cudaSysfsFixture builds a sysfs tree under a temp dir. withDevices adds one
// NVIDIA GPU and one NVMe controller whose names differ from the representative
// profile, so any leak of that profile into inspect/audit output is visible.
func cudaSysfsFixture(t *testing.T, withDevices bool) string {
	t.Helper()
	root := t.TempDir()
	devices := filepath.Join(root, "bus", "pci", "devices")
	if err := os.MkdirAll(devices, 0o755); err != nil {
		t.Fatal(err)
	}
	if !withDevices {
		return root
	}
	write := func(dir string, attrs map[string]string) {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
		for name, val := range attrs {
			if err := os.WriteFile(filepath.Join(dir, name), []byte(val+"\n"), 0o644); err != nil {
				t.Fatal(err)
			}
		}
	}
	write(filepath.Join(devices, "0000_65_00.0"), map[string]string{
		"vendor":             "0x10de",
		"device":             "0x2b85",
		"class":              "0x030000",
		"numa_node":          "-1",
		"upstream_root_port": "0000:00:01.1",
		"current_link_speed": "16.0 GT/s PCIe",
		"current_link_width": "8",
		"resource":           "0x00000000fb000000 0x00000000fbffffff 0x0000000000040200\n0x0000006000000000 0x00000060ffffffff 0x000000000014220c",
	})
	write(filepath.Join(devices, "0000_66_00.0"), map[string]string{
		"vendor":             "0x1e0f",
		"class":              "0x010802",
		"numa_node":          "-1",
		"model":              "Fixture NVMe Model",
		"slot_type":          "M2A_CPU",
		"upstream_root_port": "0000:00:01.2",
	})
	return root
}

func assertNoRepresentativeLeak(t *testing.T, out string) {
	t.Helper()
	for _, leak := range []string{"RTX 5090", "990 PRO", "Ryzen", "X570 Aorus", "128GB DDR4", "sm_120"} {
		if strings.Contains(out, leak) {
			t.Errorf("output leaks representative profile %q:\n%s", leak, out)
		}
	}
}

func decodeCUDAInspect(t *testing.T, data []byte) CUDAInspectOutput {
	t.Helper()
	var res CUDAInspectOutput
	if err := json.Unmarshal(data, &res); err != nil {
		t.Fatalf("invalid JSON output: %v\nOutput: %s", err, data)
	}
	return res
}

// fak-test:runtime fast est=50ms lane=default
func TestRunCUDAGPUDirect_Inspect(t *testing.T) {
	root := cudaSysfsFixture(t, true)
	var stdout, stderr bytes.Buffer
	code := RunCUDAGPUDirect(&stdout, &stderr, []string{"inspect", "--sysfs", root})
	if code != 0 {
		t.Fatalf("RunCUDAGPUDirect inspect returned %d: %s", code, stderr.String())
	}
	out := stdout.String()
	for _, expected := range []string{
		"CUDA GPU Direct Topology",
		"Discovery:           observed (method: sysfs)",
		"NVIDIA PCI device 0x2b85 [unknown]",
		"VRAM=unknown BAR1=4.0 GiB (ReBAR state unknown",
		"(modeled estimate)",
		"NUMA Node: unknown",
		"Link: 16.0 GT/s PCIe x8",
		"Not observed: device_name, arch, total_vram_bytes",
		"Fixture NVMe Model",
		"DIRECT_CPU_ROOT_COMPLEX",
		"Host:                not observed",
		"ACS Status",
	} {
		if !strings.Contains(out, expected) {
			t.Errorf("inspect output missing expected %q:\n%s", expected, out)
		}
	}
	assertNoRepresentativeLeak(t, out)

	stdout.Reset()
	stderr.Reset()
	code = RunCUDAGPUDirect(&stdout, &stderr, []string{"inspect", "--json", "--sysfs", root})
	if code != 0 {
		t.Fatalf("RunCUDAGPUDirect inspect --json returned %d: %s", code, stderr.String())
	}
	var raw map[string]any
	if err := json.Unmarshal(stdout.Bytes(), &raw); err != nil {
		t.Fatalf("invalid JSON output: %v", err)
	}
	for _, key := range []string{"provenance", "discovery", "host", "gpus", "nvme_devices", "routes", "acs_status"} {
		if _, ok := raw[key]; !ok {
			t.Errorf("missing %q in JSON output: %s", key, stdout.String())
		}
	}
	res := decodeCUDAInspect(t, stdout.Bytes())
	if res.Provenance != "OBSERVED" || res.Discovery.Status != compute.CUDADiscoveryObserved {
		t.Errorf("provenance/status = %s/%s, want OBSERVED/observed", res.Provenance, res.Discovery.Status)
	}
	if len(res.GPUs) != 1 || res.GPUs[0].Source != compute.CUDASourceSysfs || res.GPUs[0].NUMANode != compute.CUDANUMAUnknown {
		t.Errorf("gpus = %+v, want one sysfs GPU with unknown NUMA", res.GPUs)
	}
	if res.Host.CPU != "" || res.Host.Source != "not_observed" {
		t.Errorf("host = %+v, want not_observed with no fields", res.Host)
	}
	assertNoRepresentativeLeak(t, stdout.String())
}

// fak-test:runtime fast est=30ms lane=default
func TestRunCUDAGPUDirect_InspectEmptySysfsReportsAbsent(t *testing.T) {
	root := cudaSysfsFixture(t, false)
	var stdout, stderr bytes.Buffer
	if code := RunCUDAGPUDirect(&stdout, &stderr, []string{"inspect", "--json", "--sysfs", root}); code != 0 {
		t.Fatalf("inspect on empty sysfs returned %d: %s", code, stderr.String())
	}
	res := decodeCUDAInspect(t, stdout.Bytes())
	if res.Discovery.Status != compute.CUDADiscoveryAbsent || len(res.GPUs) != 0 || len(res.NVMeDevices) != 0 || len(res.Routes) != 0 {
		t.Fatalf("empty sysfs = %+v, want absent with no devices or routes", res)
	}
	if !strings.HasPrefix(res.ACSStatus, "Not evaluated") {
		t.Errorf("acs_status = %q, want Not evaluated", res.ACSStatus)
	}
	assertNoRepresentativeLeak(t, stdout.String())

	stdout.Reset()
	if code := RunCUDAGPUDirect(&stdout, &stderr, []string{"inspect", "--sysfs", root}); code != 0 {
		t.Fatalf("text inspect on empty sysfs returned %d", code)
	}
	if !strings.Contains(stdout.String(), "Discovery:           absent") || !strings.Contains(stdout.String(), "CUDA Devices (0 discovered)") {
		t.Errorf("text output does not report absence:\n%s", stdout.String())
	}
	assertNoRepresentativeLeak(t, stdout.String())
}

// fak-test:runtime fast est=30ms lane=default
func TestRunCUDAGPUDirect_InspectMissingSysfsReportsError(t *testing.T) {
	root := filepath.Join(t.TempDir(), "missing")
	var stdout, stderr bytes.Buffer
	if code := RunCUDAGPUDirect(&stdout, &stderr, []string{"inspect", "--json", "--sysfs", root}); code != 1 {
		t.Fatalf("inspect on missing sysfs returned %d, want 1", code)
	}
	res := decodeCUDAInspect(t, stdout.Bytes())
	if res.Discovery.Status != compute.CUDADiscoveryError || res.Discovery.Error == "" || len(res.Discovery.Issues) != 1 {
		t.Fatalf("discovery = %+v, want error with message and one issue", res.Discovery)
	}
	if len(res.GPUs) != 0 || len(res.NVMeDevices) != 0 {
		t.Fatalf("failed discovery invented devices: %+v %+v", res.GPUs, res.NVMeDevices)
	}
	assertNoRepresentativeLeak(t, stdout.String())
}

// fak-test:runtime fast est=30ms lane=default
func TestRunCUDAGPUDirect_InspectReadErrorReportsPartial(t *testing.T) {
	root := cudaSysfsFixture(t, true)
	gpuDir := filepath.Join(root, "bus", "pci", "devices", "0000_65_00.0")
	if err := os.Remove(filepath.Join(gpuDir, "resource")); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(gpuDir, "resource"), 0o755); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	if code := RunCUDAGPUDirect(&stdout, &stderr, []string{"inspect", "--json", "--sysfs", root}); code != 0 {
		t.Fatalf("inspect with a read error returned %d: %s", code, stderr.String())
	}
	res := decodeCUDAInspect(t, stdout.Bytes())
	if res.Discovery.Status != compute.CUDADiscoveryPartial || len(res.Discovery.Issues) != 1 {
		t.Fatalf("discovery = %+v, want partial with one issue", res.Discovery)
	}
	if len(res.GPUs) != 1 || res.GPUs[0].BAR1SizeBytes != 0 {
		t.Fatalf("gpus = %+v, want one GPU with BAR1 unknown", res.GPUs)
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestRunCUDAGPUDirect_Audit(t *testing.T) {
	root := cudaSysfsFixture(t, true)
	var stdout, stderr bytes.Buffer
	code := RunCUDAGPUDirect(&stdout, &stderr, []string{"audit", "--sysfs", root})
	if code != 0 {
		t.Fatalf("RunCUDAGPUDirect audit returned %d: %s", code, stderr.String())
	}
	out := stdout.String()
	for _, expected := range []string{
		"CUDA GPU Direct & BaM P2PDMA Hardware Audit",
		"Above 4G Decoding",
		"Resizable BAR (ReBAR)",
		"ReBAR state unknown: BAR1 and VRAM size not both observed",
		"[PASS] Above 4G Decoding",
		"[WARN] PCIe Ten Bit Tag",
		"PCIe Ten Bit Tag",
		"IOMMU / ACS",
		"NVreg_EnableResizableBar=1",
		"Healthy:             true (Verdict: WARN)",
	} {
		if !strings.Contains(out, expected) {
			t.Errorf("audit output missing expected %q:\n%s", expected, out)
		}
	}
	assertNoRepresentativeLeak(t, out)

	stdout.Reset()
	stderr.Reset()
	code = RunCUDAGPUDirect(&stdout, &stderr, []string{"audit", "--json", "--sysfs", root})
	if code != 0 {
		t.Fatalf("RunCUDAGPUDirect audit --json returned %d: %s", code, stderr.String())
	}
	var res CUDAHardwareAuditReport
	if err := json.Unmarshal(stdout.Bytes(), &res); err != nil {
		t.Fatalf("invalid JSON output: %v\nOutput: %s", err, stdout.String())
	}
	if !res.Healthy || res.Verdict != "WARN" || res.Discovery.Status != compute.CUDADiscoveryObserved {
		t.Errorf("audit = healthy %v verdict %s discovery %s, want true/WARN/observed", res.Healthy, res.Verdict, res.Discovery.Status)
	}
	if len(res.Checks) < 5 {
		t.Errorf("expected at least 5 audit checks, got %+v", res.Checks)
	}
}

// fak-test:runtime fast est=30ms lane=default
func TestRunCUDAGPUDirect_AuditWithoutGPUFails(t *testing.T) {
	root := cudaSysfsFixture(t, false)
	var stdout, stderr bytes.Buffer
	if code := RunCUDAGPUDirect(&stdout, &stderr, []string{"audit", "--json", "--sysfs", root}); code != 1 {
		t.Fatalf("audit with no GPU returned %d, want 1", code)
	}
	var res CUDAHardwareAuditReport
	if err := json.Unmarshal(stdout.Bytes(), &res); err != nil {
		t.Fatalf("invalid JSON output: %v", err)
	}
	if res.Healthy || res.Verdict != "FAIL" || len(res.Checks) != 1 || res.Checks[0].Name != "CUDA device discovery" {
		t.Fatalf("audit = %+v, want a single failing discovery check", res)
	}
	if !strings.Contains(res.Checks[0].Detail, "absent") {
		t.Errorf("detail = %q, want the typed discovery status", res.Checks[0].Detail)
	}
	assertNoRepresentativeLeak(t, stdout.String())
}

// fak-test:runtime fast est=50ms lane=default
func TestRunCUDAGPUDirect_Bench(t *testing.T) {
	empty := cudaSysfsFixture(t, false)
	var stdout, stderr bytes.Buffer
	code := RunCUDAGPUDirect(&stdout, &stderr, []string{"bench", "--sysfs", empty})
	if code != 0 {
		t.Fatalf("RunCUDAGPUDirect bench returned %d: %s", code, stderr.String())
	}
	out := stdout.String()
	for _, expected := range []string{
		"CUDA BaM NVMe P2PDMA Zero-Copy Microbenchmark Results",
		"Provenance:           SIMULATED",
		"RTX 5090 FE (sm_120) [representative_profile]",
		"Staging Copies:       0",
		"Zero-Copy Invariant:  VERIFIED",
		"Throughput:",
		"IOPS:",
	} {
		if !strings.Contains(out, expected) {
			t.Errorf("bench output missing expected %q:\n%s", expected, out)
		}
	}

	stdout.Reset()
	stderr.Reset()
	code = RunCUDAGPUDirect(&stdout, &stderr, []string{"bench", "--json", "--sysfs", empty})
	if code != 0 {
		t.Fatalf("RunCUDAGPUDirect bench --json returned %d: %s", code, stderr.String())
	}
	var res map[string]any
	if err := json.Unmarshal(stdout.Bytes(), &res); err != nil {
		t.Fatalf("invalid JSON output: %v\nOutput: %s", err, stdout.String())
	}
	if res["staging_copy_count"] != float64(0) {
		t.Errorf("expected staging_copy_count=0, got %+v", res["staging_copy_count"])
	}
	if res["zero_copy_verified"] != true {
		t.Errorf("expected zero_copy_verified=true, got %+v", res["zero_copy_verified"])
	}
	if res["throughput_gbps"].(float64) <= 0 {
		t.Errorf("expected positive throughput_gbps, got %+v", res["throughput_gbps"])
	}
	if res["iops"].(float64) <= 0 {
		t.Errorf("expected positive iops, got %+v", res["iops"])
	}
	if res["provenance"] != "SIMULATED" || res["device_source"] != "representative_profile" || res["storage_source"] != "representative_profile" {
		t.Errorf("provenance/sources = %v/%v/%v, want SIMULATED and representative_profile", res["provenance"], res["device_source"], res["storage_source"])
	}

	// With an observed device the simulation names it and says so.
	stdout.Reset()
	if code := RunCUDAGPUDirect(&stdout, &stderr, []string{"bench", "--json", "--sysfs", cudaSysfsFixture(t, true)}); code != 0 {
		t.Fatalf("bench over observed fixture returned %d: %s", code, stderr.String())
	}
	var observed CUDABaMBenchResult
	if err := json.Unmarshal(stdout.Bytes(), &observed); err != nil {
		t.Fatalf("invalid JSON output: %v", err)
	}
	if observed.DeviceSource != "observed:sysfs" || observed.StorageSource != "observed:sysfs" || observed.Provenance != "SIMULATED" {
		t.Errorf("observed bench = %+v, want observed:sysfs sources and SIMULATED provenance", observed)
	}
	if !strings.Contains(observed.Device, "NVIDIA PCI device 0x2b85 (unknown)") {
		t.Errorf("device = %q, want the observed device with unknown arch", observed.Device)
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestRunCUDAGPUDirect_Qwen38(t *testing.T) {
	empty := cudaSysfsFixture(t, false)
	// 1. Verify human table output
	var stdout, stderr bytes.Buffer
	code := RunCUDAGPUDirect(&stdout, &stderr, []string{"qwen38", "--sysfs", empty})
	if code != 0 {
		t.Fatalf("RunCUDAGPUDirect qwen38 failed with code %d: %s", code, stderr.String())
	}
	out := stdout.String()
	for _, expected := range []string{
		"Qwen3.8 CUDA Direct Storage & Cache Swap Architecture",
		"Baseline (CPU-staged)",
		"Reference (llama.cpp)",
		"fak-native (CUDA BaM P2PDMA)",
		"Speedup vs Baseline",
		"Speedup vs Reference",
		"Zero-copy NVMe P2PDMA validated",
	} {
		if !strings.Contains(out, expected) {
			t.Errorf("qwen38 output missing expected %q:\n%s", expected, out)
		}
	}

	// 2. Verify machine-readable JSON output
	stdout.Reset()
	stderr.Reset()
	code = RunCUDAGPUDirect(&stdout, &stderr, []string{"qwen38", "--json", "--sysfs", empty})
	if code != 0 {
		t.Fatalf("RunCUDAGPUDirect qwen38 --json returned %d: %s", code, stderr.String())
	}
	var receipt Qwen38CUDADirectSwapReceipt
	if err := json.Unmarshal(stdout.Bytes(), &receipt); err != nil {
		t.Fatalf("invalid JSON output: %v\nOutput: %s", err, stdout.String())
	}

	if receipt.Schema != Qwen38CUDADirectSwapReceiptSchema {
		t.Errorf("expected schema %q, got %q", Qwen38CUDADirectSwapReceiptSchema, receipt.Schema)
	}
	if receipt.Verdict != "PASS" || receipt.Provenance != "MODELED" {
		t.Errorf("expected verdict PASS / provenance MODELED, got %q / %q", receipt.Verdict, receipt.Provenance)
	}
	if receipt.DeviceSource != "representative_profile" || receipt.Architecture != "sm_120 (NVIDIA GeForce RTX 5090 FE)" {
		t.Errorf("device source/architecture = %q / %q, want the labelled representative profile", receipt.DeviceSource, receipt.Architecture)
	}
	if receipt.StagingCopyCount != 0 {
		t.Errorf("expected staging copy count 0, got %d", receipt.StagingCopyCount)
	}
	if receipt.DirectDMABandwidthGBps <= 0 {
		t.Errorf("expected positive direct DMA bandwidth, got %f", receipt.DirectDMABandwidthGBps)
	}
	if receipt.FakNative.StagingCopyCount != 0 {
		t.Errorf("expected fak_native staging copy count 0, got %d", receipt.FakNative.StagingCopyCount)
	}
	if receipt.Baseline.StagingCopyCount != 3 {
		t.Errorf("expected baseline staging copy count 3, got %d", receipt.Baseline.StagingCopyCount)
	}
	if receipt.SpeedupVsBaseline.TTFTSpeedup <= 1.0 {
		t.Errorf("expected TTFT speedup > 1.0, got %f", receipt.SpeedupVsBaseline.TTFTSpeedup)
	}

	// 3. Test alias subcommands
	for _, alias := range []string{"qwen38-bench", "qwen38-sim"} {
		stdout.Reset()
		stderr.Reset()
		code = RunCUDAGPUDirect(&stdout, &stderr, []string{alias, "--sysfs", empty})
		if code != 0 {
			t.Errorf("RunCUDAGPUDirect(%q) failed with code %d: %s", alias, code, stderr.String())
		}
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestRunCUDAGPUDirect_InvalidMode(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := RunCUDAGPUDirect(&stdout, &stderr, []string{"nonexistent_mode"})
	if code != 2 {
		t.Fatalf("expected exit code 2 for invalid mode, got %d", code)
	}
	if !strings.Contains(stderr.String(), "unknown mode") {
		t.Errorf("expected 'unknown mode' in stderr, got: %s", stderr.String())
	}
}
