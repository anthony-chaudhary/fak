// Package compute implements hardware abstraction, tensor computation, memory slab management,
// and zero-copy device interconnect acceleration for the fak agent kernel.
package compute

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
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
	// RouteDirectCPURootComplex represents an optimal peer-to-peer route switched directly within the CPU PCIe root complex.
	RouteDirectCPURootComplex P2PRouteKind = "DIRECT_CPU_ROOT_COMPLEX"
	// RouteChipsetDownlink represents a degraded route crossing motherboard chipset / southbridge interconnects.
	RouteChipsetDownlink P2PRouteKind = "CHIPSET_DOWNLINK"
	// RouteCrossSocketNUMA represents a route crossing inter-socket UPI or Infinity Fabric links.
	RouteCrossSocketNUMA P2PRouteKind = "CROSS_SOCKET_NUMA"
	// RouteIncompatible represents an unsupported or non-functional PCIe interconnect topology.
	RouteIncompatible P2PRouteKind = "INCOMPATIBLE"
	// RouteUnknown means the observed topology is insufficient to classify the route.
	RouteUnknown P2PRouteKind = "UNKNOWN"
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

// CUDADiscoveryStatus is the typed outcome of a CUDA topology discovery pass.
// It separates "the host was enumerated and has no NVIDIA GPU" from "discovery
// could not run" so a failed probe is never read as an empty host.
type CUDADiscoveryStatus string

const (
	// CUDADiscoveryNotRun means no discovery method has been invoked yet.
	CUDADiscoveryNotRun CUDADiscoveryStatus = "not_run"
	// CUDADiscoveryObserved means at least one NVIDIA GPU was observed with no read errors.
	CUDADiscoveryObserved CUDADiscoveryStatus = "observed"
	// CUDADiscoveryPartial means at least one NVIDIA GPU was observed but some reads failed.
	CUDADiscoveryPartial CUDADiscoveryStatus = "partial"
	// CUDADiscoveryAbsent means discovery enumerated every device without error and found no NVIDIA GPU.
	CUDADiscoveryAbsent CUDADiscoveryStatus = "absent"
	// CUDADiscoveryUnsupported means this platform has no discovery method.
	CUDADiscoveryUnsupported CUDADiscoveryStatus = "unsupported"
	// CUDADiscoveryError means discovery could not enumerate devices, or read
	// errors left GPU absence unproven.
	CUDADiscoveryError CUDADiscoveryStatus = "error"
)

// CUDANUMAUnknown is the NUMANode value for a device whose NUMA affinity was not
// observed (missing or unparsable numa_node, or the kernel's own -1).
const CUDANUMAUnknown = -1

// Discovery method / node provenance labels.
const (
	CUDASourceSysfs      = "sysfs"
	CUDASourceWindowsWMI = "windows_wmi"
)

// ErrCUDATopologyUnsupported reports a platform with no CUDA topology discovery method.
var ErrCUDATopologyUnsupported = errors.New("cudatopology: unsupported platform")

// CUDADiscoveryIssue records one read failure observed during discovery. A
// missing optional attribute is not an issue; it is named in the node's
// UnknownFields instead.
type CUDADiscoveryIssue struct {
	Path  string `json:"path"`
	Error string `json:"error"`
}

// CUDADeviceNode describes an NVIDIA GPU device node, its PCIe topology, and memory aperture.
// Zero/empty values mean "not observed" and are named in UnknownFields; NUMANode
// uses CUDANUMAUnknown. Discovery never substitutes a representative device.
type CUDADeviceNode struct {
	PCIAddress       string   `json:"pci_address"`
	DeviceName       string   `json:"device_name"`
	Arch             string   `json:"arch"`
	TotalVRAMBytes   uint64   `json:"total_vram_bytes"`
	BAR1SizeBytes    uint64   `json:"bar1_size_bytes"`
	ReBAREnabled     bool     `json:"rebar_enabled"`
	UpstreamRootPort string   `json:"upstream_root_port"`
	NUMANode         int      `json:"numa_node"`
	VendorID         string   `json:"vendor_id,omitempty"`
	DeviceID         string   `json:"device_id,omitempty"`
	PCIeAncestry     []string `json:"pcie_ancestry,omitempty"`
	CurrentLinkSpeed string   `json:"current_link_speed,omitempty"`
	CurrentLinkWidth string   `json:"current_link_width,omitempty"`
	MaxLinkSpeed     string   `json:"max_link_speed,omitempty"`
	MaxLinkWidth     string   `json:"max_link_width,omitempty"`
	Source           string   `json:"source,omitempty"`
	UnknownFields    []string `json:"unknown_fields,omitempty"`
}

// NVMeStorageNode describes a PCIe NVMe storage device and its topology location.
// Zero/empty values mean "not observed" and are named in UnknownFields.
type NVMeStorageNode struct {
	PCIAddress       string       `json:"pci_address"`
	Model            string       `json:"model"`
	SlotType         NVMeSlotType `json:"slot_type"`
	UpstreamRootPort string       `json:"upstream_root_port"`
	PeakReadGBps     float64      `json:"peak_read_gbps"`
	NUMANode         int          `json:"numa_node"`
	PCIeAncestry     []string     `json:"pcie_ancestry,omitempty"`
	Source           string       `json:"source,omitempty"`
	UnknownFields    []string     `json:"unknown_fields,omitempty"`
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
	// EstimateSource is "modeled" when LatencyEstimateNs/MaxBandwidthGBps are
	// reference-platform estimates rather than measurements; empty when no
	// figure is claimed.
	EstimateSource string `json:"estimate_source,omitempty"`
}

// P2PEstimateModeled marks route latency/bandwidth figures as modeled estimates.
const P2PEstimateModeled = "modeled"

// CUDAPCIeTopologyDiscovery coordinates host-wide discovery and routing adjudication for NVIDIA GPUs and NVMe drives.
type CUDAPCIeTopologyDiscovery struct {
	SysfsRoot   string               `json:"sysfs_root"`
	Status      CUDADiscoveryStatus  `json:"status"`
	Method      string               `json:"method,omitempty"`
	Issues      []CUDADiscoveryIssue `json:"issues,omitempty"`
	GPUs        []CUDADeviceNode     `json:"gpus"`
	NVMeDevices []NVMeStorageNode    `json:"nvme_devices"`
}

// NewCUDAPCIeTopologyDiscovery constructs a discovery orchestrator targeting the specified sysfs root (or default /sys).
func NewCUDAPCIeTopologyDiscovery(sysfsRoot string) *CUDAPCIeTopologyDiscovery {
	return &CUDAPCIeTopologyDiscovery{
		SysfsRoot:   sysfsRoot,
		Status:      CUDADiscoveryNotRun,
		GPUs:        make([]CUDADeviceNode, 0),
		NVMeDevices: make([]NVMeStorageNode, 0),
	}
}

// Discover scans the host for NVIDIA GPUs and NVMe controllers: Linux sysfs when
// a sysfs root is set or the host is Linux, WMI on Windows, otherwise it records
// CUDADiscoveryUnsupported. The typed outcome is left in d.Status.
func (d *CUDAPCIeTopologyDiscovery) Discover() error {
	switch {
	case d.SysfsRoot != "" || runtime.GOOS == "linux":
		root := d.SysfsRoot
		if root == "" {
			root = "/sys"
		}
		return d.DiscoverLinuxSysfs(root)
	case runtime.GOOS == "windows":
		return d.DiscoverWindows()
	default:
		d.Method = "none"
		d.Status = CUDADiscoveryUnsupported
		return ErrCUDATopologyUnsupported
	}
}

func (d *CUDAPCIeTopologyDiscovery) addIssue(path string, err error) {
	d.Issues = append(d.Issues, CUDADiscoveryIssue{Path: filepath.ToSlash(path), Error: err.Error()})
}

// finalizeStatus folds the observed GPU count and read issues into d.Status.
// Zero GPUs with read issues is an error, not absence: an unreadable device
// may have been the GPU.
func (d *CUDAPCIeTopologyDiscovery) finalizeStatus() {
	switch {
	case len(d.GPUs) > 0 && len(d.Issues) > 0:
		d.Status = CUDADiscoveryPartial
	case len(d.GPUs) > 0:
		d.Status = CUDADiscoveryObserved
	case len(d.Issues) > 0:
		d.Status = CUDADiscoveryError
	default:
		d.Status = CUDADiscoveryAbsent
	}
}

// readSysfsAttr reads one trimmed sysfs attribute. A missing attribute returns
// ok=false with no issue; any other read failure is recorded as an issue.
func (d *CUDAPCIeTopologyDiscovery) readSysfsAttr(devDir, name string) (string, bool) {
	path := filepath.Join(devDir, name)
	data, err := os.ReadFile(path)
	if err != nil {
		if !errors.Is(err, fs.ErrNotExist) {
			d.addIssue(path, err)
		}
		return "", false
	}
	return strings.TrimSpace(string(data)), true
}

// isPCIAddressComponent reports whether a path element is a PCI BDF
// (DDDD:BB:DD.F). Underscores stand in for colons on filesystems that forbid them.
func isPCIAddressComponent(elem string) bool {
	parts := strings.Split(strings.ReplaceAll(elem, "_", ":"), ":")
	if len(parts) != 3 || len(parts[0]) != 4 || len(parts[1]) != 2 {
		return false
	}
	devFn := strings.Split(parts[2], ".")
	if len(devFn) != 2 || len(devFn[0]) != 2 || len(devFn[1]) != 1 {
		return false
	}
	for _, h := range []string{parts[0], parts[1], devFn[0], devFn[1]} {
		if _, err := strconv.ParseUint(h, 16, 16); err != nil {
			return false
		}
	}
	return true
}

// isPCIHostBridgeComponent reports whether a path element is a host bridge
// directory (pciDDDD:BB).
func isPCIHostBridgeComponent(elem string) bool {
	if !strings.HasPrefix(elem, "pci") {
		return false
	}
	return isPCIAddressComponent(strings.TrimPrefix(elem, "pci") + ":00.0")
}

// PCIAncestryFromDevicePath extracts the PCIe ancestry from a resolved sysfs
// device path such as /sys/devices/pci0000:00/0000:00:01.1/0000:01:00.0/0000:02:00.0.
// It returns the root port and the chain of upstream bridge addresses (root
// port first, the device itself excluded). A path with no bridge above the
// device (a root-complex integrated endpoint) or no pciDDDD:BB host bridge
// returns an empty root port: ancestry is unknown, never assumed.
func PCIAncestryFromDevicePath(resolved string) (rootPort string, ancestry []string) {
	elems := strings.FieldsFunc(filepath.ToSlash(resolved), func(r rune) bool { return r == '/' })
	host := -1
	for i, e := range elems {
		if isPCIHostBridgeComponent(e) {
			host = i
		}
	}
	if host < 0 {
		return "", nil
	}
	var chain []string
	for _, e := range elems[host+1:] {
		if !isPCIAddressComponent(e) {
			break
		}
		chain = append(chain, strings.ReplaceAll(e, "_", ":"))
	}
	if len(chain) < 2 {
		return "", nil
	}
	ancestry = chain[:len(chain)-1]
	return ancestry[0], ancestry
}

// observeAncestry resolves a sysfs device link to its PCIe ancestry. An
// explicit upstream_root_port attribute (hand-built fixture trees) is honoured
// when the link carries no ancestry; otherwise the root port stays empty.
func (d *CUDAPCIeTopologyDiscovery) observeAncestry(devDir string) (string, []string) {
	if resolved, err := filepath.EvalSymlinks(devDir); err == nil {
		if port, chain := PCIAncestryFromDevicePath(resolved); port != "" {
			return port, chain
		}
	}
	if port, ok := d.readSysfsAttr(devDir, "upstream_root_port"); ok && port != "" {
		return port, nil
	}
	return "", nil
}

// observeNUMA reads numa_node; a missing, unparsable or negative value is unknown.
func (d *CUDAPCIeTopologyDiscovery) observeNUMA(devDir string) int {
	raw, ok := d.readSysfsAttr(devDir, "numa_node")
	if !ok {
		return CUDANUMAUnknown
	}
	n, err := strconv.Atoi(raw)
	if err != nil || n < 0 {
		return CUDANUMAUnknown
	}
	return n
}

// DiscoverLinuxSysfs parses Linux PCIe sysfs tree under sysfsRoot for NVIDIA GPUs and NVMe devices.
// Only attributes present in sysfs are reported; architecture and VRAM are not
// exposed there and stay unknown rather than being filled from a reference card.
func (d *CUDAPCIeTopologyDiscovery) DiscoverLinuxSysfs(sysfsRoot string) error {
	if sysfsRoot == "" {
		sysfsRoot = "/sys"
	}
	d.Method = CUDASourceSysfs

	pciDevicesDir := filepath.Join(sysfsRoot, "bus", "pci", "devices")
	entries, err := os.ReadDir(pciDevicesDir)
	if err != nil {
		d.addIssue(pciDevicesDir, err)
		d.Status = CUDADiscoveryError
		return fmt.Errorf("cudatopology: reading PCI devices directory: %w", err)
	}

	for _, entry := range entries {
		devDir := filepath.Join(pciDevicesDir, entry.Name())
		bdf := strings.ReplaceAll(entry.Name(), "_", ":")
		if addr, ok := d.readSysfsAttr(devDir, "address"); ok && addr != "" {
			bdf = addr
		}

		vendor, ok := d.readSysfsAttr(devDir, "vendor")
		if !ok {
			continue
		}
		class, _ := d.readSysfsAttr(devDir, "class")

		if strings.EqualFold(vendor, "0x10de") {
			if class != "" && !strings.HasPrefix(strings.ToLower(class), "0x03") {
				continue // NVIDIA audio/USB/bridge functions are not GPUs.
			}
			d.GPUs = append(d.GPUs, d.observeGPU(devDir, bdf, vendor))
			continue
		}

		if nvme, ok := d.observeNVMe(devDir, bdf, class); ok {
			d.NVMeDevices = append(d.NVMeDevices, nvme)
		}
	}

	d.finalizeStatus()
	return nil
}

func (d *CUDAPCIeTopologyDiscovery) observeGPU(devDir, bdf, vendor string) CUDADeviceNode {
	gpu := CUDADeviceNode{
		PCIAddress: bdf,
		VendorID:   vendor,
		NUMANode:   d.observeNUMA(devDir),
		Source:     CUDASourceSysfs,
	}
	var unknown []string
	if gpu.DeviceID, _ = d.readSysfsAttr(devDir, "device"); gpu.DeviceID == "" {
		unknown = append(unknown, "device_id")
	}
	if label, ok := d.readSysfsAttr(devDir, "label"); ok && label != "" {
		gpu.DeviceName = label
	} else {
		unknown = append(unknown, "device_name")
	}
	// sysfs exposes neither the SM architecture nor the VRAM size.
	unknown = append(unknown, "arch", "total_vram_bytes")

	if res, ok := d.readSysfsAttr(devDir, "resource"); ok {
		if sizes := ParsePCIeResourceSizes(res); len(sizes) > 1 {
			gpu.BAR1SizeBytes = sizes[1]
		}
	}
	if gpu.BAR1SizeBytes == 0 {
		unknown = append(unknown, "bar1_size_bytes")
	}
	// ReBAR is judged against VRAM, which is unobserved here.
	unknown = append(unknown, "rebar_enabled")

	if gpu.UpstreamRootPort, gpu.PCIeAncestry = d.observeAncestry(devDir); gpu.UpstreamRootPort == "" {
		unknown = append(unknown, "upstream_root_port")
	}
	if gpu.NUMANode == CUDANUMAUnknown {
		unknown = append(unknown, "numa_node")
	}
	gpu.CurrentLinkSpeed, _ = d.readSysfsAttr(devDir, "current_link_speed")
	gpu.CurrentLinkWidth, _ = d.readSysfsAttr(devDir, "current_link_width")
	gpu.MaxLinkSpeed, _ = d.readSysfsAttr(devDir, "max_link_speed")
	gpu.MaxLinkWidth, _ = d.readSysfsAttr(devDir, "max_link_width")
	if gpu.CurrentLinkSpeed == "" || gpu.CurrentLinkWidth == "" {
		unknown = append(unknown, "pcie_link")
	}
	gpu.UnknownFields = unknown
	return gpu
}

func (d *CUDAPCIeTopologyDiscovery) observeNVMe(devDir, bdf, class string) (NVMeStorageNode, bool) {
	isNVMe := strings.HasPrefix(strings.ToLower(class), "0x0108")
	slotTypeRaw, hasSlot := d.readSysfsAttr(devDir, "slot_type")
	model, hasModel := d.readSysfsAttr(devDir, "model")
	isNVMe = isNVMe || hasSlot || hasModel
	if !hasModel {
		// Real controllers expose the model under the nvme class child.
		nvmeSubsystem := filepath.Join(devDir, "nvme")
		nvmeEntries, err := os.ReadDir(nvmeSubsystem)
		if err != nil && !errors.Is(err, fs.ErrNotExist) {
			d.addIssue(nvmeSubsystem, err)
		}
		for _, ne := range nvmeEntries {
			if m, ok := d.readSysfsAttr(filepath.Join(nvmeSubsystem, ne.Name()), "model"); ok {
				model, isNVMe = m, true
				break
			}
		}
	}
	if !isNVMe {
		return NVMeStorageNode{}, false
	}

	nvme := NVMeStorageNode{
		PCIAddress: bdf,
		Model:      model,
		SlotType:   NVMeSlotUnknown,
		NUMANode:   d.observeNUMA(devDir),
		Source:     CUDASourceSysfs,
	}
	var unknown []string
	if nvme.Model == "" {
		unknown = append(unknown, "model")
	}
	switch strings.ToUpper(slotTypeRaw) {
	case "M2A_CPU":
		nvme.SlotType = NVMeSlotM2ACPU
	case "M2B_SB":
		nvme.SlotType = NVMeSlotM2BSB
	case "CHIPSET":
		nvme.SlotType = NVMeSlotChipset
	default:
		unknown = append(unknown, "slot_type")
	}
	if nvme.UpstreamRootPort, nvme.PCIeAncestry = d.observeAncestry(devDir); nvme.UpstreamRootPort == "" {
		unknown = append(unknown, "upstream_root_port")
	}
	if raw, ok := d.readSysfsAttr(devDir, "peak_read_gbps"); ok {
		if p, err := strconv.ParseFloat(raw, 64); err == nil && p > 0 {
			nvme.PeakReadGBps = p
		}
	}
	if nvme.PeakReadGBps == 0 {
		unknown = append(unknown, "peak_read_gbps")
	}
	if nvme.NUMANode == CUDANUMAUnknown {
		unknown = append(unknown, "numa_node")
	}
	nvme.UnknownFields = unknown
	return nvme, true
}

type winDisplayInfoRaw struct {
	Name   string `json:"name"`
	Driver string `json:"driver"`
	RAM    int64  `json:"ram"`
	PNP    string `json:"pnp"`
}

// pnpHardwareID extracts the 4-hex-digit value following key (e.g. "VEN_") in a
// PNP device ID, formatted like sysfs ("0x10de"), or "" when absent.
func pnpHardwareID(pnp, key string) string {
	i := strings.Index(strings.ToUpper(pnp), key)
	if i < 0 || len(pnp) < i+len(key)+4 {
		return ""
	}
	id := pnp[i+len(key) : i+len(key)+4]
	if _, err := strconv.ParseUint(id, 16, 16); err != nil {
		return ""
	}
	return "0x" + strings.ToLower(id)
}

// ParseWindowsCUDADisplayJSON parses JSON serialized from Win32_VideoController for NVIDIA GPUs.
// WMI exposes the name and PNP vendor/device IDs. AdapterRAM is a uint32 that
// saturates or wraps for adapters of 4 GiB and more, so it is never reported as
// VRAM. The PCI address, architecture, BAR1, root port and NUMA node are unknown.
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
		node := CUDADeviceNode{
			DeviceName: item.Name,
			NUMANode:   CUDANUMAUnknown,
			VendorID:   pnpHardwareID(item.PNP, "VEN_"),
			DeviceID:   pnpHardwareID(item.PNP, "DEV_"),
			Source:     CUDASourceWindowsWMI,
		}
		unknown := []string{"pci_address", "arch"}
		if node.DeviceName == "" {
			unknown = append(unknown, "device_name")
		}
		if node.VendorID == "" {
			unknown = append(unknown, "vendor_id")
		}
		if node.DeviceID == "" {
			unknown = append(unknown, "device_id")
		}
		node.UnknownFields = append(unknown, "total_vram_bytes", "bar1_size_bytes", "rebar_enabled", "upstream_root_port", "numa_node", "pcie_link")
		nodes = append(nodes, node)
	}

	return nodes, nil
}

// windowsVideoControllerQuery lists every Win32_VideoController; the NVIDIA
// selection happens in Go (selectNVIDIADisplayNodes) so it is testable.
var windowsVideoControllerQuery = func() ([]byte, error) {
	cmd := exec.Command("powershell", "-NoProfile", "-Command",
		"Get-CimInstance Win32_VideoController | ForEach-Object { [pscustomobject]@{ name = $_.Name; driver = $_.DriverVersion; ram = [int64]$_.AdapterRAM; pnp = $_.PNPDeviceID } } | ConvertTo-Json -Compress")
	windowgate.ConfigureBackgroundCommand(cmd)
	return cmd.Output()
}

// selectNVIDIADisplayNodes keeps adapters by PCI vendor ID, not name: an NVIDIA
// GPU on the Microsoft Basic Display Adapter carries no NVIDIA name but is
// still an NVIDIA device. An adapter with no PNP vendor ID is kept only when
// its name says NVIDIA.
func selectNVIDIADisplayNodes(nodes []CUDADeviceNode) []CUDADeviceNode {
	var out []CUDADeviceNode
	for _, n := range nodes {
		if n.VendorID == "0x10de" || (n.VendorID == "" && strings.Contains(strings.ToUpper(n.DeviceName), "NVIDIA")) {
			out = append(out, n)
		}
	}
	return out
}

// DiscoverWindows probes Windows display adapters via a WMI query. NVMe
// topology is not observable through this path and is left empty.
func (d *CUDAPCIeTopologyDiscovery) DiscoverWindows() error {
	d.Method = CUDASourceWindowsWMI
	out, err := windowsVideoControllerQuery()
	if err != nil {
		d.addIssue("Win32_VideoController", err)
		d.Status = CUDADiscoveryError
		return fmt.Errorf("cudatopology: WMI display query: %w", err)
	}
	if strings.TrimSpace(string(out)) == "" {
		// ConvertTo-Json emits nothing when no adapter matched.
		d.finalizeStatus()
		return nil
	}
	gpus, err := ParseWindowsCUDADisplayJSON(string(out))
	if err != nil {
		d.addIssue("Win32_VideoController", err)
		d.Status = CUDADiscoveryError
		return fmt.Errorf("cudatopology: parsing WMI display query: %w", err)
	}
	d.GPUs = append(d.GPUs, selectNVIDIADisplayNodes(gpus)...)
	d.finalizeStatus()
	return nil
}

// EvaluateP2PRoute computes the topology verdict, bandwidth, latency, and ReBAR compliance for an NVMe -> GPU P2P route.
func (d *CUDAPCIeTopologyDiscovery) EvaluateP2PRoute(gpu CUDADeviceNode, nvme NVMeStorageNode) P2PRouteVerdict {
	return EvaluateP2PRoute(gpu, nvme)
}

// CUDAReBARKnown reports whether both BAR1 and VRAM were observed, the
// precondition for any ReBAR / small-BAR judgment.
func CUDAReBARKnown(gpu CUDADeviceNode) bool {
	return gpu.TotalVRAMBytes > 0 && gpu.BAR1SizeBytes > 0
}

// EvaluateP2PRoute performs standalone PCIe topology routing adjudication.
// A route is classified only from a declared slot type or observed differing
// NUMA nodes; root-port addresses alone do not distinguish CPU from chipset
// lanes across platforms, so they yield RouteUnknown. Latency and bandwidth
// figures are modeled reference-platform estimates (EstimateSource).
func EvaluateP2PRoute(gpu CUDADeviceNode, nvme NVMeStorageNode) P2PRouteVerdict {
	notes := make([]string, 0, 4)

	// Step 1: ReBAR (Resizable BAR) validation
	rebarKnown := CUDAReBARKnown(gpu)
	rebarCompliant := rebarKnown && gpu.BAR1SizeBytes >= gpu.TotalVRAMBytes
	if !rebarKnown {
		notes = append(notes, fmt.Sprintf("ReBAR state unknown: BAR1 (%d bytes) and VRAM (%d bytes) must both be observed", gpu.BAR1SizeBytes, gpu.TotalVRAMBytes))
	} else if !rebarCompliant {
		notes = append(notes, "Small BAR clamped (256MB) - run with NVreg_EnableResizableBar=1")
	}

	// Step 2: Cross-Socket NUMA evaluation
	if gpu.NUMANode >= 0 && nvme.NUMANode >= 0 && gpu.NUMANode != nvme.NUMANode {
		notes = append(notes, "Cross-socket NUMA interconnect detected: crossing inter-socket link adds latency and limits P2P throughput")
		return P2PRouteVerdict{
			RouteKind:         RouteCrossSocketNUMA,
			IsOptimal:         false,
			LatencyEstimateNs: 1800,
			MaxBandwidthGBps:  4.5,
			ACSStallRisk:      false,
			ReBARCompliant:    rebarCompliant,
			DiagnosticNotes:   notes,
			EstimateSource:    P2PEstimateModeled,
		}
	}

	// Step 3: Chipset Downlink evaluation (e.g. Gigabyte X570 M2B_SB via southbridge)
	isChipset := nvme.SlotType == NVMeSlotM2BSB ||
		nvme.SlotType == NVMeSlotChipset ||
		strings.Contains(strings.ToLower(nvme.UpstreamRootPort), "chipset") ||
		strings.Contains(strings.ToLower(nvme.UpstreamRootPort), "southbridge")

	if isChipset {
		notes = append(notes, "Crosses X570 chipset downlink interconnect: PCIe Gen4 x4 uplink throttled to ~2.1 GB/s, high latency ~3800ns")
		notes = append(notes, "ACS (Access Control Services) stall risk detected on chipset downstream ports: P2P DMA traffic forced to loopback through CPU host bridge")
		return P2PRouteVerdict{
			RouteKind:         RouteChipsetDownlink,
			IsOptimal:         false,
			LatencyEstimateNs: 3800,
			MaxBandwidthGBps:  2.1,
			ACSStallRisk:      true,
			ReBARCompliant:    rebarCompliant,
			DiagnosticNotes:   notes,
			EstimateSource:    P2PEstimateModeled,
		}
	}

	// Step 4: Direct CPU Root Complex evaluation (NVMe declared in a CPU-lane slot)
	if nvme.SlotType == NVMeSlotM2ACPU {
		notes = append(notes, "Direct CPU Root Complex P2P route (declared M2A_CPU slot; modeled latency ~680ns, bandwidth ~7.4 GB/s)")
		if rebarKnown && !rebarCompliant {
			notes = append(notes, "P2P DMA constrained by small BAR aperture despite direct root complex connection")
		}

		return P2PRouteVerdict{
			RouteKind:         RouteDirectCPURootComplex,
			IsOptimal:         rebarCompliant,
			LatencyEstimateNs: 680,
			MaxBandwidthGBps:  7.4,
			ACSStallRisk:      false,
			ReBARCompliant:    rebarCompliant,
			DiagnosticNotes:   notes,
			EstimateSource:    P2PEstimateModeled,
		}
	}

	// Step 5: No declared slot and no NUMA split: the attachment is unknown.
	if gpu.UpstreamRootPort == "" || nvme.UpstreamRootPort == "" {
		notes = append(notes, "Route unknown: upstream root port not observed for the GPU or NVMe device; no latency or bandwidth is claimed")
	} else {
		notes = append(notes, fmt.Sprintf("Route unknown: root ports %s (GPU) and %s (NVMe) observed, but CPU vs chipset attachment is not determinable from them; no latency or bandwidth is claimed", gpu.UpstreamRootPort, nvme.UpstreamRootPort))
	}
	return P2PRouteVerdict{
		RouteKind:       RouteUnknown,
		ReBARCompliant:  rebarCompliant,
		DiagnosticNotes: notes,
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
