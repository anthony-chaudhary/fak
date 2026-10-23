package compute

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// fak-test:runtime fast est=1ms lane=default
func TestCUDAPCIeTopology_ExplicitModelOnly(t *testing.T) {
	t.Parallel()
	gpu := CUDADeviceNode{
		TotalVRAMBytes: 32 << 30, BAR1SizeBytes: 32 << 30,
		UpstreamRootPort: "0000:00:01.1", NUMANode: 0,
	}
	for _, tc := range []struct {
		name      string
		nvme      NVMeStorageNode
		kind      P2PRouteKind
		optimal   bool
		acs       bool
		latency   uint32
		bandwidth float64
	}{
		{"direct", NVMeStorageNode{SlotType: NVMeSlotM2ACPU, NUMANode: 0}, RouteDirectCPURootComplex, true, false, 680, 7.4},
		{"chipset", NVMeStorageNode{SlotType: NVMeSlotM2BSB, NUMANode: 0}, RouteChipsetDownlink, false, true, 3800, 2.1},
		{"numa", NVMeStorageNode{NUMANode: 1}, RouteCrossSocketNUMA, false, false, 1800, 4.5},
	} {
		t.Run(tc.name, func(t *testing.T) {
			v := EvaluateModeledP2PRoute(gpu, tc.nvme)
			if v.RouteKind != tc.kind || v.IsOptimal != tc.optimal || v.ACSStallRisk != tc.acs || v.LatencyEstimateNs != tc.latency || v.MaxBandwidthGBps != tc.bandwidth {
				t.Fatalf("unexpected explicit model: %+v", v)
			}
			if v.EvidenceSource != "modeled" || v.P2PValidated || !strings.Contains(v.DiagnosticNotes[0], "Modeled scenario only") {
				t.Fatalf("model was presented as hardware evidence: %+v", v)
			}
		})
	}
	gpu.BAR1SizeBytes = 256 << 20
	v := EvaluateModeledP2PRoute(gpu, NVMeStorageNode{SlotType: NVMeSlotM2ACPU})
	if v.ReBARCompliant || v.IsOptimal {
		t.Fatalf("modeled small aperture was optimal: %+v", v)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestCUDAPCIeTopology_HardwareRoutesRemainUnknown(t *testing.T) {
	t.Parallel()
	gpu := CUDADeviceNode{
		TotalVRAMBytes: 32 << 30, BAR1SizeBytes: 32 << 30, ReBAREnabled: true,
		UpstreamRootPort: "0000:00:01.1", NUMANode: 0,
	}
	d := NewCUDAPCIeTopologyDiscovery("")
	d.GPUs = []CUDADeviceNode{gpu}
	d.NVMeDevices = []NVMeStorageNode{
		{SlotType: NVMeSlotM2ACPU, UpstreamRootPort: "0000:00:01.2", NUMANode: 0},
		{SlotType: NVMeSlotM2BSB, UpstreamRootPort: "0000:00:03.1", NUMANode: 0},
		{SlotType: NVMeSlotM2ACPU, NUMANode: 1},
		{SlotType: NVMeSlotUnknown, NUMANode: -1},
	}
	for _, v := range d.EvaluateAllRoutes() {
		if v.RouteKind != RouteUnknown || v.IsOptimal || v.P2PValidated || v.LatencyEstimateNs != 0 || v.MaxBandwidthGBps != 0 || v.ReBARCompliant {
			t.Fatalf("inventory promoted to hardware evidence: %+v", v)
		}
		if v.EvidenceSource != "hardware_inventory" || !slices.Contains(v.UnknownFields, "acs_stall_risk") || !slices.Contains(v.UnknownFields, "rebar_compliant") {
			t.Fatalf("unknown ACS/ReBAR was reported as a negative observation: %+v", v)
		}
		data, err := json.Marshal(v)
		if err != nil {
			t.Fatal(err)
		}
		var fields map[string]json.RawMessage
		if err := json.Unmarshal(data, &fields); err != nil {
			t.Fatal(err)
		}
		for _, key := range []string{"route_kind", "is_optimal", "latency_estimate_ns", "max_bandwidth_gbps", "acs_stall_risk", "rebar_compliant", "unknown_fields", "p2p_validated"} {
			if _, ok := fields[key]; !ok {
				t.Errorf("missing compatibility key %s", key)
			}
		}
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestCUDAPCIeTopology_SysfsObservedInventory(t *testing.T) {
	t.Parallel()
	root := t.TempDir()
	writeDevice := func(address string, files map[string]string) {
		t.Helper()
		// Colons are not portable directory names on Windows.
		dir := filepath.Join(root, "bus", "pci", "devices", strings.ReplaceAll(address, ":", "_"))
		files["address"] = address
		if err := os.MkdirAll(dir, 0755); err != nil {
			t.Fatal(err)
		}
		for name, data := range files {
			if err := os.WriteFile(filepath.Join(dir, name), []byte(data), 0644); err != nil {
				t.Fatal(err)
			}
		}
	}
	writeDevice("0000:09:00.0", map[string]string{
		"vendor": "0x10de", "class": "0x030200", "device": "0x2684", "label": "NVIDIA GeForce RTX 4090",
		"numa_node": "-1", "resource": "0x10000000 0x10ffffff 0x200\n0x000000c000000000 0x000000c7ffffffff 0x14020c\n",
	})
	writeDevice("0000:0a:00.0", map[string]string{
		"vendor": "0x10de", "class": "0x030000", "numa_node": "malformed", "resource": "malformed\nmalformed\n0x10000000 0x1fffffff 0x200\n",
	})
	writeDevice("0000:09:00.1", map[string]string{"vendor": "0x10de", "class": "0x040300", "model": "GPU audio"})
	writeDevice("0000:09:00.2", map[string]string{"vendor": "0x10de", "model": "missing class"})
	writeDevice("0000:01:00.0", map[string]string{
		"vendor": "0x144d", "class": "0x010802", "model": "Observed 990 PRO", "numa_node": "0",
		"upstream_root_port": "0000:00:01.2", "peak_read_gbps": "7.45",
	})
	writeDevice("0000:02:00.0", map[string]string{
		"vendor": "0x8086", "class": "0x010601", "model": "Not NVMe", "slot_type": "M2A_CPU",
	})
	writeDevice("0000:03:00.0", map[string]string{
		"vendor": "0x8086", "class": "0x010801", "model": "NVMHCI is not NVMe",
	})
	d := NewCUDAPCIeTopologyDiscovery(root)
	for range 2 {
		if err := d.Discover(); err != nil {
			t.Fatal(err)
		}
		if len(d.GPUs) != 2 || len(d.NVMeDevices) != 1 {
			t.Fatalf("class gates or repeat reset failed: GPUs=%+v NVMe=%+v", d.GPUs, d.NVMeDevices)
		}
	}
	gpu := d.GPUs[0]
	if gpu.PCIAddress != "0000:09:00.0" || gpu.DeviceName != "NVIDIA GeForce RTX 4090" || gpu.BAR1SizeBytes != 32<<30 {
		t.Fatalf("lost observed identity/aperture: %+v", gpu)
	}
	for _, gpu := range d.GPUs {
		if gpu.Arch != "" || gpu.TotalVRAMBytes != 0 || gpu.ReBAREnabled || gpu.UpstreamRootPort != "" || gpu.NUMANode != -1 {
			t.Fatalf("invented GPU facts: %+v", gpu)
		}
		for _, field := range []string{"arch", "total_vram_bytes", "rebar_enabled", "upstream_root_port", "numa_node"} {
			if !slices.Contains(gpu.UnknownFields, field) {
				t.Errorf("missing unknown %s: %+v", field, gpu)
			}
		}
	}
	if gpu := d.GPUs[1]; gpu.DeviceName != "" || gpu.BAR1SizeBytes != 0 || !slices.Contains(gpu.UnknownFields, "bar1_size_bytes") {
		t.Fatalf("invented name or shifted malformed BAR resource: %+v", gpu)
	}
	nvme := d.NVMeDevices[0]
	if nvme.Model != "Observed 990 PRO" || nvme.NUMANode != 0 || nvme.UpstreamRootPort != "0000:00:01.2" || nvme.SlotType != NVMeSlotUnknown || nvme.PeakReadGBps != 0 || !slices.Contains(nvme.UnknownFields, "peak_read_gbps") {
		t.Fatalf("lost observed NVMe facts or invented slot/speed: %+v", nvme)
	}
	if err := d.DiscoverLinuxSysfs(filepath.Join(root, "missing")); err == nil || len(d.GPUs) != 0 || len(d.NVMeDevices) != 0 {
		t.Fatalf("failed discovery retained stale inventory: err=%v GPUs=%v NVMe=%v", err, d.GPUs, d.NVMeDevices)
	}
	empty := t.TempDir()
	if err := os.MkdirAll(filepath.Join(empty, "bus", "pci", "devices"), 0755); err != nil {
		t.Fatal(err)
	}
	if err := d.DiscoverLinuxSysfs(empty); err != nil {
		t.Fatalf("successful empty inventory: %v", err)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestCUDAPCIeTopology_WindowsObservedIdentityAndErrors(t *testing.T) {
	t.Parallel()
	const display = `[{
		"name":"NVIDIA GeForce RTX 5090", "driver":"572.16", "ram":4293918720,
		"pnp":"PCI\\VEN_10DE&DEV_2B80&SUBSYS_000010DE&REV_A1\\4&1F2B3C4D&0&0008"
	},{"name":"Not NVIDIA", "ram":123, "pnp":"PCI\\VEN_8086&DEV_0000"}]`
	nodes, err := ParseWindowsCUDADisplayJSON(display)
	if err != nil || len(nodes) != 1 {
		t.Fatalf("nodes=%+v err=%v", nodes, err)
	}
	gpu := nodes[0]
	if gpu.DeviceName != "NVIDIA GeForce RTX 5090" || gpu.PNPDeviceID == "" || gpu.DriverVersion != "572.16" || gpu.ReportedAdapterRAMBytes != 4293918720 {
		t.Fatalf("lost observed Windows identity: %+v", gpu)
	}
	if gpu.PCIAddress != "" || gpu.Arch != "" || gpu.TotalVRAMBytes != 0 || gpu.BAR1SizeBytes != 0 || gpu.ReBAREnabled || gpu.UpstreamRootPort != "" || gpu.NUMANode != -1 || !slices.Contains(gpu.UnknownFields, "total_vram_bytes") {
		t.Fatalf("invented Windows topology/capacity: %+v", gpu)
	}
	d := NewCUDAPCIeTopologyDiscovery("")
	for range 2 {
		if err := d.discoverWindows(func() ([]byte, error) { return []byte(display), nil }); err != nil {
			t.Fatal(err)
		}
		if len(d.GPUs) != 1 || len(d.NVMeDevices) != 0 {
			t.Fatalf("duplicate GPU or fabricated disk: %+v", d)
		}
	}
	queryErr := errors.New("CIM query failed")
	err = d.discoverWindows(func() ([]byte, error) { return nil, queryErr })
	if !errors.Is(err, queryErr) || len(d.GPUs) != 0 || len(d.NVMeDevices) != 0 {
		t.Fatalf("query failure swallowed or stale inventory: err=%v d=%+v", err, d)
	}
	for _, invalid := range []string{"", "not JSON", "{", `[{"ram":"invalid"}]`} {
		if err := d.discoverWindows(func() ([]byte, error) { return []byte(invalid), nil }); err == nil {
			t.Errorf("parse error swallowed for %q", invalid)
		}
	}
	if err := d.discoverWindows(func() ([]byte, error) { return []byte("[]"), nil }); err != nil || len(d.GPUs) != 0 || len(d.NVMeDevices) != 0 {
		t.Fatalf("empty query should be empty inventory, not synthetic hardware: err=%v d=%+v", err, d)
	}
}
