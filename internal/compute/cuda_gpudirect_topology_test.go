package compute

import (
	"os"
	"path/filepath"
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
	gpu := disco.GPUs[0]
	if gpu.PCIAddress != "0000:09:00.0" {
		t.Errorf("GPU PCIAddress = %s, want 0000:09:00.0", gpu.PCIAddress)
	}
	if gpu.Arch != "sm_120" {
		t.Errorf("GPU Arch = %s, want sm_120", gpu.Arch)
	}
	if gpu.TotalVRAMBytes != 32*1024*1024*1024 {
		t.Errorf("GPU TotalVRAMBytes = %d, want %d", gpu.TotalVRAMBytes, uint64(32*1024*1024*1024))
	}
	if !gpu.ReBAREnabled {
		t.Errorf("GPU ReBAREnabled = false, want true")
	}

	if len(disco.NVMeDevices) != 2 {
		t.Fatalf("expected 2 NVMe devices, got %d", len(disco.NVMeDevices))
	}

	// Route 1: GPU -> NVMe 1 (M2A_CPU)
	verdict1 := disco.EvaluateP2PRoute(gpu, disco.NVMeDevices[0])
	if verdict1.RouteKind != RouteDirectCPURootComplex {
		t.Errorf("verdict1 RouteKind = %s, want %s", verdict1.RouteKind, RouteDirectCPURootComplex)
	}
	if !verdict1.IsOptimal {
		t.Errorf("verdict1 IsOptimal = false, want true")
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
	if gpus[0].Arch != "sm_120" {
		t.Errorf("Arch = %s, want sm_120", gpus[0].Arch)
	}
	if gpus[0].TotalVRAMBytes != 34359738368 {
		t.Errorf("TotalVRAMBytes = %d, want 34359738368", gpus[0].TotalVRAMBytes)
	}
	if !gpus[0].ReBAREnabled {
		t.Errorf("ReBAREnabled = false, want true")
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
