package compute

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestCUDAPCIeTopology_DirectCPURootComplex(t *testing.T) {
	gpu := CUDADeviceNode{
		PCIAddress:       "0000:09:00.0",
		DeviceName:       "NVIDIA GeForce RTX 5090",
		Arch:             "sm_120",
		TotalVRAMBytes:   32 * 1024 * 1024 * 1024, // 32 GiB
		BAR1SizeBytes:    32 * 1024 * 1024 * 1024,
		ReBAREnabled:     true,
		UpstreamRootPort: "0000:00:01.1",
		NUMANode:         0,
	}

	nvme := NVMeStorageNode{
		PCIAddress:       "0000:01:00.0",
		Model:            "Samsung SSD 990 PRO 2TB",
		SlotType:         NVMeSlotM2ACPU,
		UpstreamRootPort: "0000:00:01.2",
		PeakReadGBps:     7.45,
	}

	disco := NewCUDAPCIeTopologyDiscovery("")
	verdict := disco.EvaluateP2PRoute(gpu, nvme)

	if verdict.RouteKind != RouteDirectCPURootComplex {
		t.Errorf("RouteKind = %s, want %s", verdict.RouteKind, RouteDirectCPURootComplex)
	}
	if !verdict.IsOptimal {
		t.Errorf("IsOptimal = false, want true")
	}
	if !verdict.ReBARCompliant {
		t.Errorf("ReBARCompliant = false, want true")
	}
	if verdict.ACSStallRisk {
		t.Errorf("ACSStallRisk = true, want false")
	}
	if verdict.LatencyEstimateNs >= 700 || verdict.LatencyEstimateNs == 0 {
		t.Errorf("LatencyEstimateNs = %d, want < 700ns", verdict.LatencyEstimateNs)
	}
	if verdict.MaxBandwidthGBps < 7.4 {
		t.Errorf("MaxBandwidthGBps = %f, want >= 7.4", verdict.MaxBandwidthGBps)
	}
}

func TestCUDAPCIeTopology_ChipsetDownlink(t *testing.T) {
	gpu := CUDADeviceNode{
		PCIAddress:       "0000:09:00.0",
		DeviceName:       "NVIDIA GeForce RTX 5090",
		Arch:             "sm_120",
		TotalVRAMBytes:   32 * 1024 * 1024 * 1024,
		BAR1SizeBytes:    32 * 1024 * 1024 * 1024,
		ReBAREnabled:     true,
		UpstreamRootPort: "0000:00:01.1",
		NUMANode:         0,
	}

	nvme := NVMeStorageNode{
		PCIAddress:       "0000:02:00.0",
		Model:            "Crucial T705 2TB",
		SlotType:         NVMeSlotM2BSB,
		UpstreamRootPort: "0000:00:03.1",
		PeakReadGBps:     14.5,
	}

	disco := NewCUDAPCIeTopologyDiscovery("")
	verdict := disco.EvaluateP2PRoute(gpu, nvme)

	if verdict.RouteKind != RouteChipsetDownlink {
		t.Errorf("RouteKind = %s, want %s", verdict.RouteKind, RouteChipsetDownlink)
	}
	if verdict.IsOptimal {
		t.Errorf("IsOptimal = true, want false for chipset downlink")
	}
	if !verdict.ACSStallRisk {
		t.Errorf("ACSStallRisk = false, want true")
	}
	if verdict.LatencyEstimateNs < 3000 {
		t.Errorf("LatencyEstimateNs = %d, want ~3800ns", verdict.LatencyEstimateNs)
	}
	if verdict.MaxBandwidthGBps > 2.5 {
		t.Errorf("MaxBandwidthGBps = %f, want throttled ~2.1 GB/s", verdict.MaxBandwidthGBps)
	}

	// Verify ACS stall risk diagnostic note
	hasACSNote := false
	for _, note := range verdict.DiagnosticNotes {
		if strings.Contains(note, "ACS") && strings.Contains(note, "stall") {
			hasACSNote = true
			break
		}
	}
	if !hasACSNote {
		t.Errorf("expected ACS stall risk diagnostic note, got: %v", verdict.DiagnosticNotes)
	}
}

func TestCUDAPCIeTopology_SmallBARValidation(t *testing.T) {
	// 256 MiB clamped small BAR
	gpuSmallBAR := CUDADeviceNode{
		PCIAddress:       "0000:09:00.0",
		DeviceName:       "NVIDIA GeForce RTX 5090",
		Arch:             "sm_120",
		TotalVRAMBytes:   32 * 1024 * 1024 * 1024,
		BAR1SizeBytes:    256 * 1024 * 1024, // 256 MiB
		ReBAREnabled:     false,
		UpstreamRootPort: "0000:00:01.1",
		NUMANode:         0,
	}

	nvme := NVMeStorageNode{
		PCIAddress:       "0000:01:00.0",
		Model:            "Samsung SSD 990 PRO 2TB",
		SlotType:         NVMeSlotM2ACPU,
		UpstreamRootPort: "0000:00:01.2",
		PeakReadGBps:     7.45,
	}

	disco := NewCUDAPCIeTopologyDiscovery("")
	verdict := disco.EvaluateP2PRoute(gpuSmallBAR, nvme)

	if verdict.ReBARCompliant {
		t.Errorf("ReBARCompliant = true, want false for 256MB small BAR")
	}
	if verdict.IsOptimal {
		t.Errorf("IsOptimal = true, want false when small BAR is clamped")
	}

	// Must contain warning: "Small BAR clamped (256MB) - run with NVreg_EnableResizableBar=1"
	const expectedWarning = "Small BAR clamped (256MB) - run with NVreg_EnableResizableBar=1"
	hasWarn := false
	for _, note := range verdict.DiagnosticNotes {
		if strings.Contains(note, expectedWarning) {
			hasWarn = true
			break
		}
	}
	if !hasWarn {
		t.Errorf("DiagnosticNotes missing warning %q, got: %v", expectedWarning, verdict.DiagnosticNotes)
	}

	// Now verify large BAR (ReBAR enabled)
	gpuLargeBAR := gpuSmallBAR
	gpuLargeBAR.BAR1SizeBytes = 32 * 1024 * 1024 * 1024
	gpuLargeBAR.ReBAREnabled = true

	verdictLarge := disco.EvaluateP2PRoute(gpuLargeBAR, nvme)
	if !verdictLarge.ReBARCompliant {
		t.Errorf("ReBARCompliant = false, want true for 32GB large BAR")
	}
	if !verdictLarge.IsOptimal {
		t.Errorf("IsOptimal = false, want true for Large BAR + M2A_CPU")
	}
}

func TestCUDAPCIeTopology_SimulatedSysfsParsing(t *testing.T) {
	tmpDir := t.TempDir()

	// 1. Setup NVIDIA RTX 5090 device under /sys/bus/pci/devices/0000_09_00.0
	gpuDir := filepath.Join(tmpDir, "bus", "pci", "devices", "0000_09_00.0")
	if err := os.MkdirAll(gpuDir, 0755); err != nil {
		t.Fatal(err)
	}

	_ = os.WriteFile(filepath.Join(gpuDir, "vendor"), []byte("0x10de\n"), 0644)
	_ = os.WriteFile(filepath.Join(gpuDir, "device"), []byte("0x2b80\n"), 0644)
	_ = os.WriteFile(filepath.Join(gpuDir, "numa_node"), []byte("0\n"), 0644)
	_ = os.WriteFile(filepath.Join(gpuDir, "label"), []byte("NVIDIA GeForce RTX 5090\n"), 0644)
	_ = os.WriteFile(filepath.Join(gpuDir, "upstream_root_port"), []byte("0000:00:01.1\n"), 0644)

	// Resource file: BAR0 (16MB), BAR1 (32 GiB = 0x800000000)
	gpuResource := `0x000000f400000000 0x000000f400ffffff 0x0000000000040200
0x000000c000000000 0x000000c7ffffffff 0x000000000014020c
`
	_ = os.WriteFile(filepath.Join(gpuDir, "resource"), []byte(gpuResource), 0644)

	// 2. Setup NVMe M2A_CPU device under /sys/bus/pci/devices/0000_01_00.0
	nvme1Dir := filepath.Join(tmpDir, "bus", "pci", "devices", "0000_01_00.0")
	if err := os.MkdirAll(nvme1Dir, 0755); err != nil {
		t.Fatal(err)
	}
	_ = os.WriteFile(filepath.Join(nvme1Dir, "vendor"), []byte("0x144d\n"), 0644)
	_ = os.WriteFile(filepath.Join(nvme1Dir, "class"), []byte("0x010802\n"), 0644)
	_ = os.WriteFile(filepath.Join(nvme1Dir, "numa_node"), []byte("0\n"), 0644)
	_ = os.WriteFile(filepath.Join(nvme1Dir, "model"), []byte("Samsung SSD 990 PRO 2TB\n"), 0644)
	_ = os.WriteFile(filepath.Join(nvme1Dir, "slot_type"), []byte("M2A_CPU\n"), 0644)
	_ = os.WriteFile(filepath.Join(nvme1Dir, "upstream_root_port"), []byte("0000:00:01.2\n"), 0644)

	// 3. Setup NVMe M2B_SB device under /sys/bus/pci/devices/0000_02_00.0
	nvme2Dir := filepath.Join(tmpDir, "bus", "pci", "devices", "0000_02_00.0")
	if err := os.MkdirAll(nvme2Dir, 0755); err != nil {
		t.Fatal(err)
	}
	_ = os.WriteFile(filepath.Join(nvme2Dir, "vendor"), []byte("0xc0a9\n"), 0644)
	_ = os.WriteFile(filepath.Join(nvme2Dir, "class"), []byte("0x010802\n"), 0644)
	_ = os.WriteFile(filepath.Join(nvme2Dir, "numa_node"), []byte("0\n"), 0644)
	_ = os.WriteFile(filepath.Join(nvme2Dir, "model"), []byte("Crucial T705 2TB\n"), 0644)
	_ = os.WriteFile(filepath.Join(nvme2Dir, "slot_type"), []byte("M2B_SB\n"), 0644)
	_ = os.WriteFile(filepath.Join(nvme2Dir, "upstream_root_port"), []byte("0000:00:03.1\n"), 0644)

	// Parse mock sysfs
	disco := NewCUDAPCIeTopologyDiscovery(tmpDir)
	if err := disco.Discover(); err != nil {
		t.Fatalf("Discover failed: %v", err)
	}

	if len(disco.GPUs) != 1 {
		t.Fatalf("expected 1 GPU, got %d", len(disco.GPUs))
	}
	if disco.Status != CUDADiscoveryObserved {
		t.Errorf("Status = %s, want %s (issues %v)", disco.Status, CUDADiscoveryObserved, disco.Issues)
	}
	gpu := disco.GPUs[0]
	if gpu.PCIAddress != "0000:09:00.0" {
		t.Errorf("GPU PCIAddress = %s, want 0000:09:00.0", gpu.PCIAddress)
	}
	if gpu.DeviceName != "NVIDIA GeForce RTX 5090" || gpu.DeviceID != "0x2b80" {
		t.Errorf("GPU name/id = %q/%q, want observed label and device id", gpu.DeviceName, gpu.DeviceID)
	}
	// sysfs exposes neither arch nor VRAM: they stay unknown, never a reference card.
	if gpu.Arch != "" || gpu.TotalVRAMBytes != 0 || gpu.ReBAREnabled {
		t.Errorf("GPU arch/vram/rebar = %q/%d/%v, want unobserved zero values", gpu.Arch, gpu.TotalVRAMBytes, gpu.ReBAREnabled)
	}
	if gpu.BAR1SizeBytes != 32*1024*1024*1024 {
		t.Errorf("GPU BAR1SizeBytes = %d, want observed 32 GiB", gpu.BAR1SizeBytes)
	}
	for _, f := range []string{"arch", "total_vram_bytes", "rebar_enabled"} {
		if !containsString(gpu.UnknownFields, f) {
			t.Errorf("UnknownFields %v missing %q", gpu.UnknownFields, f)
		}
	}

	if len(disco.NVMeDevices) != 2 {
		t.Fatalf("expected 2 NVMe devices, got %d", len(disco.NVMeDevices))
	}

	// Route 1: GPU -> NVMe 1 (M2A_CPU)
	verdict1 := disco.EvaluateP2PRoute(gpu, disco.NVMeDevices[0])
	if verdict1.RouteKind != RouteDirectCPURootComplex {
		t.Errorf("verdict1 RouteKind = %s, want %s", verdict1.RouteKind, RouteDirectCPURootComplex)
	}
	if verdict1.IsOptimal || verdict1.ReBARCompliant {
		t.Errorf("verdict1 optimal/rebar = %v/%v, want false while VRAM is unobserved", verdict1.IsOptimal, verdict1.ReBARCompliant)
	}

	// Route 2: GPU -> NVMe 2 (M2B_SB)
	verdict2 := disco.EvaluateP2PRoute(gpu, disco.NVMeDevices[1])
	if verdict2.RouteKind != RouteChipsetDownlink {
		t.Errorf("verdict2 RouteKind = %s, want %s", verdict2.RouteKind, RouteChipsetDownlink)
	}
	if verdict2.IsOptimal {
		t.Errorf("verdict2 IsOptimal = true, want false")
	}
	if !verdict2.ACSStallRisk {
		t.Errorf("verdict2 ACSStallRisk = false, want true")
	}
}

func TestCUDAPCIeTopology_WindowsFallbackProber(t *testing.T) {
	mockDisplayJSON := `[
		{
			"name": "NVIDIA GeForce RTX 5090",
			"driver": "572.16",
			"ram": 34359738368,
			"pnp": "PCI\\VEN_10DE&DEV_2B80&SUBSYS_000010DE&REV_A1\\4&1F2B3C4D&0&0008"
		}
	]`

	gpus, err := ParseWindowsCUDADisplayJSON(mockDisplayJSON)
	if err != nil {
		t.Fatalf("ParseWindowsCUDADisplayJSON failed: %v", err)
	}
	if len(gpus) != 1 {
		t.Fatalf("expected 1 GPU, got %d", len(gpus))
	}
	if gpus[0].DeviceName != "NVIDIA GeForce RTX 5090" {
		t.Errorf("DeviceName = %s, want NVIDIA GeForce RTX 5090", gpus[0].DeviceName)
	}
	if gpus[0].Arch != "" {
		t.Errorf("Arch = %s, want unobserved", gpus[0].Arch)
	}
	// AdapterRAM is a uint32 in WMI; it is never reported as VRAM.
	if gpus[0].TotalVRAMBytes != 0 || !containsString(gpus[0].UnknownFields, "total_vram_bytes") {
		t.Errorf("TotalVRAMBytes = %d, want unknown", gpus[0].TotalVRAMBytes)
	}
	if gpus[0].ReBAREnabled || gpus[0].BAR1SizeBytes != 0 || gpus[0].PCIAddress != "" || gpus[0].UpstreamRootPort != "" {
		t.Errorf("WMI node invented BAR1/ReBAR/BDF/root port: %+v", gpus[0])
	}
	if gpus[0].NUMANode != CUDANUMAUnknown {
		t.Errorf("NUMANode = %d, want unknown", gpus[0].NUMANode)
	}
	if gpus[0].VendorID != "0x10de" || gpus[0].DeviceID != "0x2b80" {
		t.Errorf("vendor/device = %q/%q, want PNP-derived 0x10de/0x2b80", gpus[0].VendorID, gpus[0].DeviceID)
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestCUDAPCIeTopology_WindowsSaturatedAdapterRAMIsUnknown(t *testing.T) {
	// 4293918720 is the saturated value; 2147483648 is what a wrapped 6 GiB card reports.
	for _, ram := range []string{"4293918720", "2147483648"} {
		gpus, err := ParseWindowsCUDADisplayJSON(`{"name":"NVIDIA GeForce RTX 4070 Laptop GPU","ram":` + ram + `,"pnp":"PCI\\VEN_10DE&DEV_2860"}`)
		if err != nil || len(gpus) != 1 {
			t.Fatalf("parse = %v, %d nodes", err, len(gpus))
		}
		if gpus[0].TotalVRAMBytes != 0 || !containsString(gpus[0].UnknownFields, "total_vram_bytes") {
			t.Errorf("uint32 AdapterRAM %s reported as VRAM: %+v", ram, gpus[0])
		}
	}
	gpus, _ := ParseWindowsCUDADisplayJSON(`{"name":"NVIDIA GeForce RTX 4070 Laptop GPU","ram":0,"pnp":"PCI\\VEN_10DE&DEV_2860"}`)
	if gpus[0].DeviceID != "0x2860" {
		t.Errorf("DeviceID = %q, want 0x2860", gpus[0].DeviceID)
	}
}

func TestCUDAPCIeTopology_CrossSocketNUMA(t *testing.T) {
	gpu := CUDADeviceNode{
		PCIAddress:       "0000:09:00.0",
		DeviceName:       "NVIDIA GeForce RTX 5090",
		Arch:             "sm_120",
		TotalVRAMBytes:   32 * 1024 * 1024 * 1024,
		BAR1SizeBytes:    32 * 1024 * 1024 * 1024,
		ReBAREnabled:     true,
		UpstreamRootPort: "0000:00:01.1",
		NUMANode:         0,
	}

	nvme := NVMeStorageNode{
		PCIAddress:       "0000:81:00.0",
		Model:            "Samsung SSD 990 PRO 2TB",
		SlotType:         NVMeSlotM2ACPU,
		UpstreamRootPort: "0000:80:01.1",
		PeakReadGBps:     7.45,
		NUMANode:         1,
	}

	disco := NewCUDAPCIeTopologyDiscovery("")
	verdict := disco.EvaluateP2PRoute(gpu, nvme)

	if verdict.RouteKind != RouteCrossSocketNUMA {
		t.Errorf("RouteKind = %s, want %s", verdict.RouteKind, RouteCrossSocketNUMA)
	}
	if verdict.IsOptimal {
		t.Errorf("IsOptimal = true, want false")
	}
	if verdict.LatencyEstimateNs < 1000 {
		t.Errorf("LatencyEstimateNs = %d, want cross-socket latency >= 1000", verdict.LatencyEstimateNs)
	}
}

func containsString(list []string, want string) bool {
	for _, v := range list {
		if v == want {
			return true
		}
	}
	return false
}

func writeSysfsAttrs(t *testing.T, dir string, attrs map[string]string) {
	t.Helper()
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	for name, val := range attrs {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(val+"\n"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestCUDATopology_EmptySysfsIsAbsent(t *testing.T) {
	root := t.TempDir()
	if err := os.MkdirAll(filepath.Join(root, "bus", "pci", "devices"), 0o755); err != nil {
		t.Fatal(err)
	}
	disco := NewCUDAPCIeTopologyDiscovery(root)
	if err := disco.Discover(); err != nil {
		t.Fatalf("Discover: %v", err)
	}
	if disco.Status != CUDADiscoveryAbsent || len(disco.GPUs) != 0 || len(disco.NVMeDevices) != 0 || len(disco.Issues) != 0 {
		t.Fatalf("empty sysfs = status %s, %d GPUs, %d NVMe, issues %v; want absent and nothing invented",
			disco.Status, len(disco.GPUs), len(disco.NVMeDevices), disco.Issues)
	}
	if routes := disco.EvaluateAllRoutes(); len(routes) != 0 {
		t.Fatalf("routes = %d, want 0", len(routes))
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestCUDATopology_MissingSysfsIsError(t *testing.T) {
	disco := NewCUDAPCIeTopologyDiscovery(filepath.Join(t.TempDir(), "no-such-sysfs"))
	if err := disco.Discover(); err == nil {
		t.Fatal("Discover on a missing sysfs root returned nil error")
	}
	if disco.Status != CUDADiscoveryError || len(disco.Issues) != 1 || len(disco.GPUs) != 0 {
		t.Fatalf("missing sysfs = status %s, issues %v, %d GPUs; want error with one issue", disco.Status, disco.Issues, len(disco.GPUs))
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestCUDATopology_PartialSysfsLeavesFieldsUnknown(t *testing.T) {
	root := t.TempDir()
	devices := filepath.Join(root, "bus", "pci", "devices")
	writeSysfsAttrs(t, filepath.Join(devices, "0000_41_00.0"), map[string]string{
		"vendor":    "0x10de",
		"class":     "0x030000",
		"numa_node": "-1",
	})
	// NVIDIA HD-audio function on the same card is not a GPU.
	writeSysfsAttrs(t, filepath.Join(devices, "0000_41_00.1"), map[string]string{
		"vendor": "0x10de",
		"class":  "0x040300",
	})
	writeSysfsAttrs(t, filepath.Join(devices, "0000_42_00.0"), map[string]string{
		"vendor":    "0x144d",
		"class":     "0x010802",
		"numa_node": "garbage",
	})

	disco := NewCUDAPCIeTopologyDiscovery(root)
	if err := disco.Discover(); err != nil {
		t.Fatalf("Discover: %v", err)
	}
	if disco.Status != CUDADiscoveryObserved {
		t.Fatalf("Status = %s, want observed (issues %v)", disco.Status, disco.Issues)
	}
	if len(disco.GPUs) != 1 {
		t.Fatalf("GPUs = %d, want 1 (audio function must be skipped)", len(disco.GPUs))
	}
	gpu := disco.GPUs[0]
	if gpu.DeviceName != "" || gpu.Arch != "" || gpu.TotalVRAMBytes != 0 || gpu.BAR1SizeBytes != 0 || gpu.UpstreamRootPort != "" {
		t.Errorf("partial GPU carries invented values: %+v", gpu)
	}
	if gpu.NUMANode != CUDANUMAUnknown {
		t.Errorf("kernel numa_node -1 = %d, want CUDANUMAUnknown", gpu.NUMANode)
	}
	for _, f := range []string{"device_id", "device_name", "arch", "total_vram_bytes", "bar1_size_bytes", "upstream_root_port", "numa_node", "pcie_link"} {
		if !containsString(gpu.UnknownFields, f) {
			t.Errorf("GPU UnknownFields %v missing %q", gpu.UnknownFields, f)
		}
	}

	if len(disco.NVMeDevices) != 1 {
		t.Fatalf("NVMe = %d, want 1", len(disco.NVMeDevices))
	}
	nvme := disco.NVMeDevices[0]
	if nvme.Model != "" || nvme.PeakReadGBps != 0 || nvme.SlotType != NVMeSlotUnknown || nvme.UpstreamRootPort != "" || nvme.NUMANode != CUDANUMAUnknown {
		t.Errorf("partial NVMe carries invented values: %+v", nvme)
	}
	for _, f := range []string{"model", "slot_type", "upstream_root_port", "peak_read_gbps", "numa_node"} {
		if !containsString(nvme.UnknownFields, f) {
			t.Errorf("NVMe UnknownFields %v missing %q", nvme.UnknownFields, f)
		}
	}

	v := EvaluateP2PRoute(gpu, nvme)
	if v.RouteKind != RouteUnknown || v.IsOptimal || v.LatencyEstimateNs != 0 || v.MaxBandwidthGBps != 0 {
		t.Errorf("route over unobserved topology = %+v, want UNKNOWN with no latency/bandwidth claim", v)
	}
	for _, n := range v.DiagnosticNotes {
		if strings.Contains(n, "Small BAR clamped") {
			t.Errorf("unobserved VRAM reported as small BAR: %v", v.DiagnosticNotes)
		}
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestCUDATopology_ReadErrorsAreReported(t *testing.T) {
	root := t.TempDir()
	devices := filepath.Join(root, "bus", "pci", "devices")
	gpuDir := filepath.Join(devices, "0000_09_00.0")
	writeSysfsAttrs(t, gpuDir, map[string]string{"vendor": "0x10de", "class": "0x030000", "numa_node": "0"})
	// An unreadable attribute (a directory where a file is expected) is a read
	// error, not a missing attribute.
	if err := os.MkdirAll(filepath.Join(gpuDir, "resource"), 0o755); err != nil {
		t.Fatal(err)
	}

	disco := NewCUDAPCIeTopologyDiscovery(root)
	if err := disco.Discover(); err != nil {
		t.Fatalf("Discover: %v", err)
	}
	if disco.Status != CUDADiscoveryPartial || len(disco.Issues) != 1 {
		t.Fatalf("Status = %s, issues %v; want partial with one read issue", disco.Status, disco.Issues)
	}
	if !strings.HasSuffix(disco.Issues[0].Path, "0000_09_00.0/resource") || disco.Issues[0].Error == "" {
		t.Errorf("issue = %+v, want the resource path and its error", disco.Issues[0])
	}
	if disco.GPUs[0].BAR1SizeBytes != 0 || disco.GPUs[0].NUMANode != 0 {
		t.Errorf("GPU = %+v, want BAR1 unknown and observed NUMA 0", disco.GPUs[0])
	}

	// An unreadable vendor on the only device leaves GPU absence unproven.
	root2 := t.TempDir()
	if err := os.MkdirAll(filepath.Join(root2, "bus", "pci", "devices", "0000_09_00.0", "vendor"), 0o755); err != nil {
		t.Fatal(err)
	}
	disco2 := NewCUDAPCIeTopologyDiscovery(root2)
	if err := disco2.Discover(); err != nil {
		t.Fatalf("Discover: %v", err)
	}
	if disco2.Status != CUDADiscoveryError || len(disco2.GPUs) != 0 {
		t.Fatalf("unreadable vendor = status %s, %d GPUs; want error, not absent", disco2.Status, len(disco2.GPUs))
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestPCIAncestryFromDevicePath(t *testing.T) {
	cases := []struct {
		path     string
		root     string
		ancestry []string
	}{
		{"/sys/devices/pci0000:00/0000:00:01.1/0000:09:00.0", "0000:00:01.1", []string{"0000:00:01.1"}},
		{"/sys/devices/pci0000:40/0000:40:01.1/0000:41:00.0/0000:42:08.0/0000:43:00.0", "0000:40:01.1", []string{"0000:40:01.1", "0000:41:00.0", "0000:42:08.0"}},
		{"C:/tmp/devices/pci0000_00/0000_00_01.2/0000_01_00.0", "0000:00:01.2", []string{"0000:00:01.2"}},
		{"/sys/devices/pci0000:00/0000:00:02.0", "", nil},                        // root-complex integrated endpoint
		{"/sys/bus/pci/devices/0000:09:00.0", "", nil},                           // unresolved link: no host bridge
		{"/sys/devices/platform/soc/0000:01:00.0", "", nil},                      // no pciDDDD:BB element
		{"/sys/devices/pci0000:00/0000:00:01.1/not-a-bdf/0000:09:00.0", "", nil}, // broken chain
	}
	for _, c := range cases {
		root, ancestry := PCIAncestryFromDevicePath(c.path)
		if root != c.root || strings.Join(ancestry, ",") != strings.Join(c.ancestry, ",") {
			t.Errorf("PCIAncestryFromDevicePath(%q) = %q %v, want %q %v", c.path, root, ancestry, c.root, c.ancestry)
		}
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestCUDATopology_RealSysfsAncestryAndLink(t *testing.T) {
	root := t.TempDir()
	real := filepath.Join(root, "devices", "pci0000_00", "0000_00_01.1", "0000_09_00.0")
	writeSysfsAttrs(t, real, map[string]string{
		"vendor":             "0x10de",
		"device":             "0x2b85",
		"class":              "0x030000",
		"numa_node":          "1",
		"current_link_speed": "32.0 GT/s PCIe",
		"current_link_width": "16",
		"max_link_speed":     "32.0 GT/s PCIe",
		"max_link_width":     "16",
	})
	devices := filepath.Join(root, "bus", "pci", "devices")
	if err := os.MkdirAll(devices, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(real, filepath.Join(devices, "0000_09_00.0")); err != nil {
		if runtime.GOOS == "windows" {
			t.Skipf("directory symlinks need SeCreateSymbolicLinkPrivilege on Windows: %v", err)
		}
		t.Fatal(err)
	}

	disco := NewCUDAPCIeTopologyDiscovery(root)
	if err := disco.Discover(); err != nil {
		t.Fatalf("Discover: %v", err)
	}
	if len(disco.GPUs) != 1 {
		t.Fatalf("GPUs = %d, want 1", len(disco.GPUs))
	}
	gpu := disco.GPUs[0]
	if gpu.UpstreamRootPort != "0000:00:01.1" || len(gpu.PCIeAncestry) != 1 {
		t.Errorf("root port/ancestry = %q %v, want 0000:00:01.1 from the resolved link", gpu.UpstreamRootPort, gpu.PCIeAncestry)
	}
	if gpu.NUMANode != 1 || gpu.CurrentLinkSpeed != "32.0 GT/s PCIe" || gpu.CurrentLinkWidth != "16" || gpu.MaxLinkWidth != "16" {
		t.Errorf("NUMA/link = %d %q x%q (max x%q), want observed sysfs values", gpu.NUMANode, gpu.CurrentLinkSpeed, gpu.CurrentLinkWidth, gpu.MaxLinkWidth)
	}
	if containsString(gpu.UnknownFields, "upstream_root_port") || containsString(gpu.UnknownFields, "pcie_link") || containsString(gpu.UnknownFields, "numa_node") {
		t.Errorf("observed fields listed unknown: %v", gpu.UnknownFields)
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestCUDATopology_RouteNeedsDeclaredSlotOrNUMA(t *testing.T) {
	gpu := CUDADeviceNode{UpstreamRootPort: "0000:40:01.1", NUMANode: 0, BAR1SizeBytes: 256 * 1024 * 1024}
	for _, port := range []string{"0000:40:01.2", "0000:00:03.1", "0000:00:06.0"} {
		v := EvaluateP2PRoute(gpu, NVMeStorageNode{UpstreamRootPort: port, SlotType: NVMeSlotUnknown, NUMANode: 0})
		if v.RouteKind != RouteUnknown || v.LatencyEstimateNs != 0 || v.MaxBandwidthGBps != 0 || v.EstimateSource != "" || v.ACSStallRisk {
			t.Errorf("observed root ports %s -> %+v, want UNKNOWN with no estimate", port, v)
		}
	}
	// BAR1 observed but VRAM unknown: no small-BAR claim anywhere in the verdict.
	v := EvaluateP2PRoute(gpu, NVMeStorageNode{SlotType: NVMeSlotM2ACPU, NUMANode: 0})
	if v.RouteKind != RouteDirectCPURootComplex || v.EstimateSource != P2PEstimateModeled || v.ReBARCompliant {
		t.Errorf("declared M2A_CPU = %+v, want modeled DIRECT with ReBAR not established", v)
	}
	for _, n := range v.DiagnosticNotes {
		if strings.Contains(strings.ToLower(n), "small bar") {
			t.Errorf("small-BAR claim with VRAM unobserved: %v", v.DiagnosticNotes)
		}
	}
	// VRAM known but BAR1 unobserved (the WMI shape) is equally unknown.
	if CUDAReBARKnown(CUDADeviceNode{TotalVRAMBytes: 8 << 30}) {
		t.Error("CUDAReBARKnown true with BAR1 unobserved")
	}
}
