// Package compute implements hardware abstraction, tensor computation, memory slab management,
// and zero-copy device interconnect acceleration for the fak agent kernel.
package compute

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// P2PRouteKind classifies the PCIe topology interconnect path between an NVMe storage device and a GPU.
type P2PRouteKind string

const (
	// RouteUnknown means that inventory has not established a usable P2P route.
	RouteUnknown P2PRouteKind = "UNKNOWN"
	// RouteDirectCPURootComplex represents an optimal peer-to-peer route switched directly within the CPU PCIe root complex.
	RouteDirectCPURootComplex P2PRouteKind = "DIRECT_CPU_ROOT_COMPLEX"
	// RouteChipsetDownlink represents a degraded route crossing motherboard chipset / southbridge interconnects.
	RouteChipsetDownlink P2PRouteKind = "CHIPSET_DOWNLINK"
	// RouteCrossSocketNUMA represents a route crossing inter-socket UPI or Infinity Fabric links.
	RouteCrossSocketNUMA P2PRouteKind = "CROSS_SOCKET_NUMA"
	// RouteIncompatible represents an unsupported or non-functional PCIe interconnect topology.
	RouteIncompatible P2PRouteKind = "INCOMPATIBLE"
)

// NVMeSlotType identifies the motherboard or riser slot PCIe routing origin.
type NVMeSlotType string

const (
	// NVMeSlotM2ACPU represents an M.2 NVMe slot routed directly to CPU PCIe lanes.
	NVMeSlotM2ACPU NVMeSlotType = "M2A_CPU"
	// NVMeSlotM2BSB represents an M.2 NVMe slot routed via the chipset / southbridge downlink.
	NVMeSlotM2BSB NVMeSlotType = "M2B_SB"
	// NVMeSlotChipset represents a generic chipset-connected NVMe storage controller.
	NVMeSlotChipset NVMeSlotType = "Chipset"
	// NVMeSlotUnknown represents an unidentified slot topology.
	NVMeSlotUnknown NVMeSlotType = "UNKNOWN"
)

// CUDADeviceNode describes an NVIDIA GPU device node, its PCIe topology, and memory aperture.
type CUDADeviceNode struct {
	PCIAddress       string `json:"pci_address"`
	DeviceName       string `json:"device_name"`
	Arch             string `json:"arch"`
	TotalVRAMBytes   uint64 `json:"total_vram_bytes"`
	BAR1SizeBytes    uint64 `json:"bar1_size_bytes"`
	ReBAREnabled     bool   `json:"rebar_enabled"`
	UpstreamRootPort string `json:"upstream_root_port"`
	NUMANode         int    `json:"numa_node"`
	PNPDeviceID      string `json:"pnp_device_id,omitempty"`
	DriverVersion    string `json:"driver_version,omitempty"`
	// AdapterRAM is a legacy WMI report, not authoritative modern GPU capacity.
	ReportedAdapterRAMBytes uint64   `json:"reported_adapter_ram_bytes,omitempty"`
	EvidenceSource          string   `json:"evidence_source"`
	UnknownFields           []string `json:"unknown_fields"`
}

// NVMeStorageNode describes a PCIe NVMe storage device and its topology location.
type NVMeStorageNode struct {
	PCIAddress       string       `json:"pci_address"`
	Model            string       `json:"model"`
	SlotType         NVMeSlotType `json:"slot_type"`
	UpstreamRootPort string       `json:"upstream_root_port"`
	PeakReadGBps     float64      `json:"peak_read_gbps"`
	NUMANode         int          `json:"numa_node"`
	EvidenceSource   string       `json:"evidence_source"`
	UnknownFields    []string     `json:"unknown_fields"`
}

// P2PRouteVerdict evaluates the peer-to-peer DMA suitability between an NVMe drive and a CUDA GPU.
type P2PRouteVerdict struct {
	RouteKind         P2PRouteKind `json:"route_kind"`
	IsOptimal         bool         `json:"is_optimal"`
	LatencyEstimateNs uint32       `json:"latency_estimate_ns"`
	MaxBandwidthGBps  float64      `json:"max_bandwidth_gbps"`
	ACSStallRisk      bool         `json:"acs_stall_risk"`
	ReBARCompliant    bool         `json:"rebar_compliant"`
	DiagnosticNotes   []string     `json:"diagnostic_notes"`
	EvidenceSource    string       `json:"evidence_source"`
	UnknownFields     []string     `json:"unknown_fields"`
	P2PValidated      bool         `json:"p2p_validated"`
}

// CUDAPCIeTopologyDiscovery coordinates host-wide discovery and routing adjudication for NVIDIA GPUs and NVMe drives.
type CUDAPCIeTopologyDiscovery struct {
	SysfsRoot   string            `json:"sysfs_root"`
	GPUs        []CUDADeviceNode  `json:"gpus"`
	NVMeDevices []NVMeStorageNode `json:"nvme_devices"`
}

// NewCUDAPCIeTopologyDiscovery constructs a discovery orchestrator targeting the specified sysfs root (or default /sys).
func NewCUDAPCIeTopologyDiscovery(sysfsRoot string) *CUDAPCIeTopologyDiscovery {
	return &CUDAPCIeTopologyDiscovery{
		SysfsRoot:   sysfsRoot,
		GPUs:        make([]CUDADeviceNode, 0),
		NVMeDevices: make([]NVMeStorageNode, 0),
	}
}

// Discover scans the host for NVIDIA GPUs and NVMe controllers using Linux sysfs or Windows fallback.
func (d *CUDAPCIeTopologyDiscovery) Discover() error {
	d.resetInventory()
	if d.SysfsRoot != "" || runtime.GOOS == "linux" {
		root := d.SysfsRoot
		if root == "" {
			root = "/sys"
		}
		return d.DiscoverLinuxSysfs(root)
	}

	if runtime.GOOS == "windows" {
		return d.DiscoverWindows()
	}

	return errors.New("cudatopology: unsupported platform or sysfs unavailable")
}

func (d *CUDAPCIeTopologyDiscovery) resetInventory() {
	d.GPUs = make([]CUDADeviceNode, 0)
	d.NVMeDevices = make([]NVMeStorageNode, 0)
}

// DiscoverLinuxSysfs parses Linux PCIe sysfs tree under sysfsRoot for NVIDIA GPUs and NVMe devices.
func (d *CUDAPCIeTopologyDiscovery) DiscoverLinuxSysfs(sysfsRoot string) error {
	d.resetInventory()
	if sysfsRoot == "" {
		sysfsRoot = "/sys"
	}

	pciDevicesDir := filepath.Join(sysfsRoot, "bus", "pci", "devices")
	entries, err := os.ReadDir(pciDevicesDir)
	if err != nil {
		return fmt.Errorf("cudatopology: reading PCI devices directory: %w", err)
	}

	for _, entry := range entries {
		devDir := filepath.Join(pciDevicesDir, entry.Name())
		read := func(name string) string {
			data, err := os.ReadFile(filepath.Join(devDir, name))
			if err != nil {
				return ""
			}
			return strings.TrimSpace(string(data))
		}
		vendor, vendorErr := strconv.ParseUint(read("vendor"), 0, 16)
		class, classErr := strconv.ParseUint(read("class"), 0, 24)
		// NVIDIA audio, USB, and bridge functions are not CUDA display devices.
		isGPU := vendorErr == nil && vendor == 0x10de && classErr == nil && class>>16 == 0x03
		isNVMe := classErr == nil && class == 0x010802
		if !isGPU && !isNVMe {
			continue
		}

		bdf := entry.Name()
		if address := read("address"); address != "" {
			bdf = address
		}
		numaNode := -1
		if n, err := strconv.Atoi(read("numa_node")); err == nil && n >= 0 {
			numaNode = n
		}
		// An explicit inventory attribute is useful evidence, but its spelling
		// does not prove CPU/chipset routing or working peer-to-peer DMA.
		upstreamPort := read("upstream_root_port")
		if isGPU {
			name := read("label")
			if name == "" {
				name = read("model")
			}
			unknown := []string{"arch", "total_vram_bytes", "rebar_enabled"}
			if name == "" {
				unknown = append(unknown, "device_name")
			}
			resource, _ := os.ReadFile(filepath.Join(devDir, "resource"))
			bar1Size := cudaBAR1ResourceSize(string(resource))
			if bar1Size == 0 {
				unknown = append(unknown, "bar1_size_bytes")
			}
			if upstreamPort == "" {
				unknown = append(unknown, "upstream_root_port")
			}
			if numaNode < 0 {
				unknown = append(unknown, "numa_node")
			}
			d.GPUs = append(d.GPUs, CUDADeviceNode{
				PCIAddress: bdf, DeviceName: name, BAR1SizeBytes: bar1Size,
				UpstreamRootPort: upstreamPort, NUMANode: numaNode,
				EvidenceSource: "linux_sysfs", UnknownFields: unknown,
			})
			continue
		}

		model := read("model")
		if model == "" {
			nvmeSubsystem := filepath.Join(devDir, "nvme")
			if nvmeEntries, err := os.ReadDir(nvmeSubsystem); err == nil {
				for _, ne := range nvmeEntries {
					if data, err := os.ReadFile(filepath.Join(nvmeSubsystem, ne.Name(), "model")); err == nil {
						model = strings.TrimSpace(string(data))
						if model != "" {
							break
						}
					}
				}
			}
		}
		slotType := NVMeSlotUnknown
		switch strings.ToUpper(read("slot_type")) {
		case "M2A_CPU":
			slotType = NVMeSlotM2ACPU
		case "M2B_SB":
			slotType = NVMeSlotM2BSB
		case "CHIPSET":
			slotType = NVMeSlotChipset
		}
		unknown := []string{"peak_read_gbps"}
		if model == "" {
			unknown = append(unknown, "model")
		}
		if slotType == NVMeSlotUnknown {
			unknown = append(unknown, "slot_type")
		}
		if upstreamPort == "" {
			unknown = append(unknown, "upstream_root_port")
		}
		if numaNode < 0 {
			unknown = append(unknown, "numa_node")
		}
		d.NVMeDevices = append(d.NVMeDevices, NVMeStorageNode{
			PCIAddress: bdf, Model: model, SlotType: slotType,
			UpstreamRootPort: upstreamPort, NUMANode: numaNode,
			EvidenceSource: "linux_sysfs", UnknownFields: unknown,
		})
	}
	return nil
}

// Keep physical resource line indices: skipping a malformed BAR0 must not
// promote BAR2 into BAR1. An absent, unassigned, or invalid aperture is unknown.
func cudaBAR1ResourceSize(resource string) uint64 {
	lines := strings.Split(resource, "\n")
	if len(lines) < 2 {
		return 0
	}
	fields := strings.Fields(lines[1])
	if len(fields) < 3 {
		return 0
	}
	start, errStart := strconv.ParseUint(fields[0], 0, 64)
	end, errEnd := strconv.ParseUint(fields[1], 0, 64)
	flags, errFlags := strconv.ParseUint(fields[2], 0, 64)
	if errStart != nil || errEnd != nil || errFlags != nil || start == 0 || end < start || flags&0x200 == 0 {
		return 0
	}
	return end - start + 1
}

type winDisplayInfoRaw struct {
	Name   string `json:"name"`
	Driver string `json:"driver"`
	RAM    int64  `json:"ram"`
	PNP    string `json:"pnp"`
}

// ParseWindowsCUDADisplayJSON parses JSON serialized from Win32_VideoController for NVIDIA GPUs.
func ParseWindowsCUDADisplayJSON(jsonData string) ([]CUDADeviceNode, error) {
	trimmed := strings.TrimSpace(jsonData)
	if trimmed == "" {
		return nil, errors.New("cudatopology: empty display JSON")
	}

	var items []winDisplayInfoRaw
	if strings.HasPrefix(trimmed, "[") {
		if err := json.Unmarshal([]byte(trimmed), &items); err != nil {
			return nil, err
		}
	} else if strings.HasPrefix(trimmed, "{") {
		var single winDisplayInfoRaw
		if err := json.Unmarshal([]byte(trimmed), &single); err != nil {
			return nil, err
		}
		items = append(items, single)
	} else {
		return nil, errors.New("cudatopology: invalid JSON structure")
	}

	nodes := make([]CUDADeviceNode, 0, len(items))
	for _, item := range items {
		pnp := strings.ToUpper(item.PNP)
		parts := strings.Split(pnp, "\\")
		if len(parts) < 2 || parts[0] != "PCI" {
			continue
		}
		isNVIDIA := false
		for _, id := range strings.Split(parts[1], "&") {
			if id == "VEN_10DE" {
				isNVIDIA = true
			}
		}
		if !isNVIDIA {
			continue
		}
		if item.RAM < 0 {
			return nil, errors.New("cudatopology: negative AdapterRAM in display JSON")
		}
		unknown := []string{"pci_address", "arch", "total_vram_bytes", "bar1_size_bytes", "rebar_enabled", "upstream_root_port", "numa_node"}
		if item.Name == "" {
			unknown = append(unknown, "device_name")
		}
		nodes = append(nodes, CUDADeviceNode{
			DeviceName: item.Name, PNPDeviceID: item.PNP, DriverVersion: item.Driver,
			ReportedAdapterRAMBytes: uint64(item.RAM), NUMANode: -1,
			EvidenceSource: "windows_cim", UnknownFields: unknown,
		})
	}
	return nodes, nil
}

// DiscoverWindows probes observed display identity. Disk topology and modern
// GPU memory capacity are not available from this Win32_VideoController query.
func (d *CUDAPCIeTopologyDiscovery) DiscoverWindows() error {
	return d.discoverWindows(func() ([]byte, error) {
		cmd := exec.Command("powershell", "-NoProfile", "-Command",
			"$ErrorActionPreference = 'Stop'; ConvertTo-Json -InputObject @(Get-CimInstance Win32_VideoController | ForEach-Object { [pscustomobject]@{ name = $_.Name; driver = $_.DriverVersion; ram = [int64]$_.AdapterRAM; pnp = $_.PNPDeviceID } }) -Compress")
		windowgate.ConfigureBackgroundCommand(cmd)
		return cmd.Output()
	})
}

func (d *CUDAPCIeTopologyDiscovery) discoverWindows(query func() ([]byte, error)) error {
	d.resetInventory()
	out, err := query()
	if err != nil {
		return fmt.Errorf("cudatopology: querying Windows displays: %w", err)
	}
	gpus, err := ParseWindowsCUDADisplayJSON(string(out))
	if err != nil {
		return fmt.Errorf("cudatopology: parsing Windows displays: %w", err)
	}
	d.GPUs = gpus
	return nil
}

// EvaluateP2PRoute reports whether observed inventory establishes a P2P route.
func (d *CUDAPCIeTopologyDiscovery) EvaluateP2PRoute(gpu CUDADeviceNode, nvme NVMeStorageNode) P2PRouteVerdict {
	return EvaluateP2PRoute(gpu, nvme)
}

// EvaluateP2PRoute never promotes inventory or PCI address patterns into proof
// of peer access. Until a capability/transfer probe exists, even complete-looking
// inventory leaves routing, ACS, ReBAR enablement, and performance unvalidated.
// Existing false/zero scalars are compatibility placeholders; UnknownFields is
// authoritative about missing evidence, not a confirmed negative observation.
func EvaluateP2PRoute(gpu CUDADeviceNode, nvme NVMeStorageNode) P2PRouteVerdict {
	notes := []string{
		"Inventory alone does not validate NVMe-to-GPU peer access or a usable GPUDirect path.",
		"ACS behavior, ReBAR enablement, latency, and bandwidth have not been measured or verified.",
	}
	if gpu.NUMANode >= 0 && nvme.NUMANode >= 0 && gpu.NUMANode != nvme.NUMANode {
		notes = append(notes, "Different NUMA nodes are reported; NUMA identifiers alone do not identify sockets or prove P2P support.")
	}
	return P2PRouteVerdict{
		RouteKind: RouteUnknown, EvidenceSource: "hardware_inventory",
		UnknownFields:   []string{"route_kind", "is_optimal", "latency_estimate_ns", "max_bandwidth_gbps", "acs_stall_risk", "rebar_compliant"},
		DiagnosticNotes: notes,
	}
}

// EvaluateModeledP2PRoute evaluates explicitly supplied hypothetical topology.
// Its constants and classifications are model assumptions, not observations,
// capability checks, or hardware performance measurements.
func EvaluateModeledP2PRoute(gpu CUDADeviceNode, nvme NVMeStorageNode) P2PRouteVerdict {
	verdict := evaluateModeledP2PRoute(gpu, nvme)
	verdict.EvidenceSource = "modeled"
	verdict.UnknownFields = []string{}
	verdict.DiagnosticNotes = append([]string{"Modeled scenario only: no physical P2P transfer or performance measurement was performed."}, verdict.DiagnosticNotes...)
	return verdict
}

func evaluateModeledP2PRoute(gpu CUDADeviceNode, nvme NVMeStorageNode) P2PRouteVerdict {
	notes := make([]string, 0, 4)

	// Step 1: ReBAR (Resizable BAR) validation
	rebarCompliant := gpu.BAR1SizeBytes >= gpu.TotalVRAMBytes && gpu.TotalVRAMBytes > 0
	if !rebarCompliant {
		notes = append(notes, "Modeled BAR1 aperture does not cover modeled VRAM; actual ReBAR enablement is not established")
	}

	// Step 2: Cross-Socket NUMA evaluation
	if gpu.NUMANode >= 0 && nvme.NUMANode >= 0 && gpu.NUMANode != nvme.NUMANode {
		notes = append(notes, "Modeled cross-socket interconnect assumes additional latency and limited P2P throughput")
		return P2PRouteVerdict{
			RouteKind:         RouteCrossSocketNUMA,
			IsOptimal:         false,
			LatencyEstimateNs: 1800,
			MaxBandwidthGBps:  4.5,
			ACSStallRisk:      false,
			ReBARCompliant:    rebarCompliant,
			DiagnosticNotes:   notes,
		}
	}

	// Step 3: Chipset Downlink evaluation (e.g. Gigabyte X570 M2B_SB via southbridge)
	isChipset := nvme.SlotType == NVMeSlotM2BSB ||
		nvme.SlotType == NVMeSlotChipset ||
		strings.Contains(nvme.UpstreamRootPort, "0000:00:03.") ||
		strings.Contains(strings.ToLower(nvme.UpstreamRootPort), "chipset") ||
		strings.Contains(strings.ToLower(nvme.UpstreamRootPort), "southbridge")

	if isChipset {
		notes = append(notes, "Modeled chipset downlink assumes ~2.1 GB/s bandwidth and ~3800ns latency")
		notes = append(notes, "Modeled ACS (Access Control Services) stall risk assumes P2P traffic is redirected through the host bridge")
		return P2PRouteVerdict{
			RouteKind:         RouteChipsetDownlink,
			IsOptimal:         false,
			LatencyEstimateNs: 3800,
			MaxBandwidthGBps:  2.1,
			ACSStallRisk:      true,
			ReBARCompliant:    rebarCompliant,
			DiagnosticNotes:   notes,
		}
	}

	// Step 4: Direct CPU Root Complex evaluation (e.g. Ryzen 5950X PEG 0000:00:01.1 and M.2 CPU 0000:00:01.2)
	isDirectCPU := nvme.SlotType == NVMeSlotM2ACPU ||
		(strings.HasPrefix(gpu.UpstreamRootPort, "0000:00:01.") && strings.HasPrefix(nvme.UpstreamRootPort, "0000:00:01."))

	if isDirectCPU {
		notes = append(notes, "Modeled direct CPU root complex assumes ~680ns latency and ~7.4 GB/s bandwidth")
		isOptimal := rebarCompliant
		if !rebarCompliant {
			notes = append(notes, "P2P DMA constrained by small BAR aperture despite direct root complex connection")
		}

		return P2PRouteVerdict{
			RouteKind:         RouteDirectCPURootComplex,
			IsOptimal:         isOptimal,
			LatencyEstimateNs: 680,
			MaxBandwidthGBps:  7.4,
			ACSStallRisk:      false,
			ReBARCompliant:    rebarCompliant,
			DiagnosticNotes:   notes,
		}
	}

	// Unrecognized modeled topology remains unknown, not proven incompatible.
	notes = append(notes, "No route assumption is available for this modeled topology")
	return P2PRouteVerdict{
		RouteKind:         RouteUnknown,
		IsOptimal:         false,
		LatencyEstimateNs: 0,
		MaxBandwidthGBps:  0.0,
		ACSStallRisk:      false,
		ReBARCompliant:    rebarCompliant,
		DiagnosticNotes:   notes,
	}
}

// EvaluateAllRoutes evaluates all pairwise routes between discovered GPUs and NVMe devices.
func (d *CUDAPCIeTopologyDiscovery) EvaluateAllRoutes() []P2PRouteVerdict {
	verdicts := make([]P2PRouteVerdict, 0, len(d.GPUs)*len(d.NVMeDevices))
	for _, gpu := range d.GPUs {
		for _, nvme := range d.NVMeDevices {
			verdicts = append(verdicts, d.EvaluateP2PRoute(gpu, nvme))
		}
	}
	return verdicts
}
