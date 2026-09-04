package compute

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

// SmallBARThresholdBytes defines the minimum BAR1 size (256 MB) for full Resizable BAR (ReBAR).
// When BAR1 is below 256 MB, the device is operating with legacy small BAR aperture.
const SmallBARThresholdBytes uint64 = 256 * 1024 * 1024

// LinkType categorizes the physical or logical interconnect between peer devices.
type LinkType string

const (
	// LinkTypeXGMI denotes high-speed coherent fabric interconnect (e.g. MI300X/MI250X).
	LinkTypeXGMI LinkType = "xGMI"
	// LinkTypePCIeSwitch denotes peer routing through a PCIe switch downstream ports.
	LinkTypePCIeSwitch LinkType = "PCIeSwitch"
	// LinkTypeHostBridge denotes routing across the CPU host root complex / host bridge.
	LinkTypeHostBridge LinkType = "HostBridge"
)

// AMDDeviceNode describes a single discovered AMD GPU device node.
type AMDDeviceNode struct {
	NodeID     int      `json:"node_id"`
	GPUID      int      `json:"gpu_id"`
	Name       string   `json:"name"`
	Arch       string   `json:"arch,omitempty"`
	PCIAddress string   `json:"pci_address,omitempty"`
	VRAMSize   uint64   `json:"vram_size_bytes"`
	BAR1Size   uint64   `json:"bar1_size_bytes"`
	LargeBAR   bool     `json:"is_large_bar"`
	Warnings   []string `json:"warnings,omitempty"`
}

// IsLargeBAR reports whether the device has a large Resizable BAR (ReBAR) enabled.
func (n *AMDDeviceNode) IsLargeBAR() bool {
	if n == nil {
		return false
	}
	return n.LargeBAR
}

// PeerLink describes the peer-to-peer route between two AMD device nodes.
type PeerLink struct {
	FromNodeID          int      `json:"from_node_id"`
	ToNodeID            int      `json:"to_node_id"`
	From                int      `json:"from"`
	To                  int      `json:"to"`
	Type                LinkType `json:"link_type"`
	P2PSupported        bool     `json:"p2p_supported"`
	Direct              bool     `json:"direct"`
	BandwidthGBs        float64  `json:"bandwidth_gbs,omitempty"`
	Weight              int      `json:"weight,omitempty"`
	IntermediateBridges []string `json:"intermediate_bridges,omitempty"`
	ACSRedirectDetected bool     `json:"acs_redirect_detected"`
	RefusalReason       string   `json:"refusal_reason,omitempty"`
}

// TopologyMatrix captures discovered AMD GPU Direct topology and peer routes.
type TopologyMatrix struct {
	SysfsRoot   string                    `json:"sysfs_root"`
	Nodes       []*AMDDeviceNode          `json:"nodes"`
	Links       []*PeerLink               `json:"links"`
	AuditReport *AuditReport              `json:"audit_report,omitempty"`
	NodeMap     map[int]*AMDDeviceNode    `json:"-"`
	LinkMatrix  map[int]map[int]*PeerLink `json:"-"`
}

// AuditReport details the ReBAR and ACS validation outcome for an AMD GPU Direct topology.
type AuditReport struct {
	Status              string           `json:"status"` // "PASS", "WARN", "REFUSED"
	Passed              bool             `json:"passed"`
	SysfsRoot           string           `json:"sysfs_root"`
	NodeCount           int              `json:"node_count"`
	Nodes               []*AMDDeviceNode `json:"nodes"`
	Links               []*PeerLink      `json:"links"`
	ACSRedirectDetected bool             `json:"acs_redirect_detected"`
	SmallBARDetected    bool             `json:"small_bar_detected"`
	Warnings            []string         `json:"warnings,omitempty"`
	Refusals            []string         `json:"refusals,omitempty"`
	Summary             string           `json:"summary"`
}

// JSON returns the structured JSON representation of the audit report.
func (r *AuditReport) JSON() ([]byte, error) {
	if r == nil {
		return nil, fmt.Errorf("audit report is nil")
	}
	return json.MarshalIndent(r, "", "  ")
}

// JSON returns the structured JSON representation of the topology matrix.
func (m *TopologyMatrix) JSON() ([]byte, error) {
	if m == nil {
		return nil, fmt.Errorf("topology matrix is nil")
	}
	return json.MarshalIndent(m, "", "  ")
}

// ValidateP2PRoute validates whether peer-to-peer DMA / GPU Direct transfer is supported
// between source node `from` and destination node `to`.
// If intermediate PCIe bridges have ACS Request Redirect enabled, fail-closed refusal is returned.
func (m *TopologyMatrix) ValidateP2PRoute(from, to int) (*PeerLink, error) {
	if m == nil {
		return nil, fmt.Errorf("topology matrix is nil")
	}
	if from == to {
		return nil, fmt.Errorf("invalid P2P route: source and destination are identical node %d", from)
	}
	if _, ok := m.NodeMap[from]; !ok {
		return nil, fmt.Errorf("source node %d not found in topology", from)
	}
	if _, ok := m.NodeMap[to]; !ok {
		return nil, fmt.Errorf("destination node %d not found in topology", to)
	}

	var link *PeerLink
	if m.LinkMatrix != nil && m.LinkMatrix[from] != nil {
		link = m.LinkMatrix[from][to]
	}
	if link == nil {
		return nil, fmt.Errorf("no topology link found between node %d and node %d", from, to)
	}

	if link.ACSRedirectDetected {
		return link, fmt.Errorf("P2P route refused from node %d to node %d: %s", from, to, link.RefusalReason)
	}

	if !link.P2PSupported {
		reason := link.RefusalReason
		if reason == "" {
			reason = "P2P not supported on link"
		}
		return link, fmt.Errorf("P2P route refused from node %d to node %d: %s", from, to, reason)
	}

	return link, nil
}

// ValidateP2PRoute is a package-level helper validating a route on the given matrix.
func ValidateP2PRoute(m *TopologyMatrix, from, to int) (*PeerLink, error) {
	if m == nil {
		return nil, fmt.Errorf("topology matrix is nil")
	}
	return m.ValidateP2PRoute(from, to)
}

// Audit evaluates the topology matrix, checking ReBAR posture and ACS redirect conflicts.
func (m *TopologyMatrix) Audit() *AuditReport {
	if m == nil {
		return nil
	}
	if m.AuditReport != nil {
		return m.AuditReport
	}

	r := &AuditReport{
		SysfsRoot: m.SysfsRoot,
		NodeCount: len(m.Nodes),
		Nodes:     m.Nodes,
		Links:     m.Links,
		Status:    "PASS",
		Passed:    true,
	}

	warningSet := make(map[string]struct{})
	for _, n := range m.Nodes {
		if !n.IsLargeBAR() || n.BAR1Size < SmallBARThresholdBytes {
			r.SmallBARDetected = true
		}
		for _, w := range n.Warnings {
			if _, exists := warningSet[w]; !exists {
				warningSet[w] = struct{}{}
				r.Warnings = append(r.Warnings, w)
			}
		}
	}

	refusalSet := make(map[string]struct{})
	for _, l := range m.Links {
		if l.ACSRedirectDetected {
			r.ACSRedirectDetected = true
			r.Passed = false
			if l.RefusalReason != "" {
				if _, exists := refusalSet[l.RefusalReason]; !exists {
					refusalSet[l.RefusalReason] = struct{}{}
					r.Refusals = append(r.Refusals, l.RefusalReason)
				}
			}
		} else if !l.P2PSupported && l.RefusalReason != "" {
			r.Passed = false
			if _, exists := refusalSet[l.RefusalReason]; !exists {
				refusalSet[l.RefusalReason] = struct{}{}
				r.Refusals = append(r.Refusals, l.RefusalReason)
			}
		}
	}

	if r.ACSRedirectDetected || len(r.Refusals) > 0 {
		r.Status = "REFUSED"
		r.Passed = false
	} else if r.SmallBARDetected || len(r.Warnings) > 0 {
		r.Status = "WARN"
	}

	r.Summary = fmt.Sprintf("AMD GPU Direct topology audit: %d nodes, %d links, status: %s (small_bar: %v, acs_redirect: %v)",
		r.NodeCount, len(r.Links), r.Status, r.SmallBARDetected, r.ACSRedirectDetected)

	m.AuditReport = r
	return r
}

// DiscoverTopology scans a mock or real sysfs tree and builds the AMD GPU Direct topology matrix.
func DiscoverTopology(sysfsRoot string) (*TopologyMatrix, error) {
	if sysfsRoot == "" {
		sysfsRoot = "/sys"
	}
	info, err := os.Stat(sysfsRoot)
	if err != nil {
		return nil, fmt.Errorf("read sysfs root %q: %w", sysfsRoot, err)
	}
	if !info.IsDir() {
		return nil, fmt.Errorf("sysfs root %q is not a directory", sysfsRoot)
	}

	nodeDirs := findNodeDirs(sysfsRoot)
	if len(nodeDirs) == 0 {
		return nil, fmt.Errorf("no AMD GPU device nodes found in sysfs root %q", sysfsRoot)
	}

	nodes := make([]*AMDDeviceNode, 0, len(nodeDirs))
	nodeMap := make(map[int]*AMDDeviceNode)
	nodeHiveMap := make(map[int]string)
	nodeSwitchMap := make(map[int]string)

	rawLinks := make([]*PeerLink, 0)

	for _, nDir := range nodeDirs {
		node, hiveID, pcieSwitch, links, err := parseNodeDir(nDir, sysfsRoot)
		if err != nil {
			return nil, fmt.Errorf("parse node dir %q: %w", nDir, err)
		}
		nodes = append(nodes, node)
		nodeMap[node.NodeID] = node
		if hiveID != "" {
			nodeHiveMap[node.NodeID] = hiveID
		}
		if pcieSwitch != "" {
			nodeSwitchMap[node.NodeID] = pcieSwitch
		}
		rawLinks = append(rawLinks, links...)
	}

	// Sort nodes by NodeID for determinism
	sort.Slice(nodes, func(i, j int) bool {
		return nodes[i].NodeID < nodes[j].NodeID
	})

	// If no links were parsed from io_links, derive topology from shared xGMI hive or PCIe switches
	if len(rawLinks) == 0 && len(nodes) > 1 {
		rawLinks = deriveTopologyLinks(nodes, nodeHiveMap, nodeSwitchMap, sysfsRoot)
	}

	// Process and validate ACS redirect on all links
	resolvedLinks, linkMatrix := resolveAndValidateLinks(nodes, rawLinks, sysfsRoot)

	matrix := &TopologyMatrix{
		SysfsRoot:  sysfsRoot,
		Nodes:      nodes,
		Links:      resolvedLinks,
		NodeMap:    nodeMap,
		LinkMatrix: linkMatrix,
	}

	matrix.Audit()
	return matrix, nil
}

// findNodeDirs searches candidate sysfs paths for GPU node directories.
func findNodeDirs(root string) []string {
	candidates := []string{
		filepath.Join(root, "class", "kfd", "kfd", "topology", "nodes"),
		filepath.Join(root, "topology", "nodes"),
		filepath.Join(root, "nodes"),
		root,
	}

	for _, cand := range candidates {
		entries, err := os.ReadDir(cand)
		if err != nil {
			continue
		}
		var found []string
		for _, e := range entries {
			if !e.IsDir() {
				continue
			}
			name := e.Name()
			if strings.HasPrefix(name, "node") || isNumeric(name) {
				found = append(found, filepath.Join(cand, name))
			}
		}
		if len(found) > 0 {
			sort.Strings(found)
			return found
		}
	}
	return nil
}

func isNumeric(s string) bool {
	if s == "" {
		return false
	}
	_, err := strconv.Atoi(s)
	return err == nil
}

// parseNodeDir reads node properties, VRAM, BAR1, and any io_links inside a node directory.
func parseNodeDir(dir string, sysfsRoot string) (*AMDDeviceNode, string, string, []*PeerLink, error) {
	dirName := filepath.Base(dir)
	nodeID := parseNodeID(dirName)

	propsPath := filepath.Join(dir, "properties")
	var props map[string]string
	if data, err := os.ReadFile(propsPath); err == nil {
		props = parseProperties(string(data))
	} else {
		props = make(map[string]string)
	}

	if idStr, ok := props["node_id"]; ok && nodeID == 0 {
		if id, err := strconv.Atoi(idStr); err == nil {
			nodeID = id
		}
	}

	gpuID := nodeID
	if gidStr, ok := props["gpu_id"]; ok {
		if gid, err := strconv.Atoi(gidStr); err == nil {
			gpuID = gid
		}
	} else if data, err := os.ReadFile(filepath.Join(dir, "gpu_id")); err == nil {
		if gid, err := strconv.Atoi(strings.TrimSpace(string(data))); err == nil {
			gpuID = gid
		}
	}

	name := "AMD GPU"
	if n, ok := props["name"]; ok && n != "" {
		name = n
	} else if n, ok := props["product_name"]; ok && n != "" {
		name = n
	} else if data, err := os.ReadFile(filepath.Join(dir, "name")); err == nil {
		n := strings.TrimSpace(string(data))
		if n != "" {
			name = n
		}
	}

	arch := inferArch(name, props)
	pciAddr := inferPCIAddress(dir, props)

	vramSize := readVRAMSize(dir, props)
	bar1Size := readBAR1Size(dir, props, vramSize, pciAddr, sysfsRoot)

	node := &AMDDeviceNode{
		NodeID:     nodeID,
		GPUID:      gpuID,
		Name:       name,
		Arch:       arch,
		PCIAddress: pciAddr,
		VRAMSize:   vramSize,
		BAR1Size:   bar1Size,
		Warnings:   make([]string, 0),
	}

	evaluateReBAR(node)

	hiveID := props["hive_id"]
	pcieSwitch := props["pcie_switch"]
	if pcieSwitch == "" {
		pcieSwitch = props["bridge"]
	}

	links := parseIOLinks(dir, nodeID)

	return node, hiveID, pcieSwitch, links, nil
}

func parseNodeID(s string) int {
	if strings.HasPrefix(s, "node") {
		s = strings.TrimPrefix(s, "node")
	}
	id, _ := strconv.Atoi(s)
	return id
}

func inferArch(name string, props map[string]string) string {
	if a, ok := props["arch"]; ok && a != "" {
		return a
	}
	if a, ok := props["gfx_target_version"]; ok && a != "" {
		return "gfx" + a
	}
	lower := strings.ToLower(name)
	if strings.Contains(lower, "mi300") || strings.Contains(lower, "gfx942") {
		return "gfx942"
	}
	if strings.Contains(lower, "mi250") || strings.Contains(lower, "mi210") || strings.Contains(lower, "gfx90a") {
		return "gfx90a"
	}
	if strings.Contains(lower, "mi100") || strings.Contains(lower, "gfx908") {
		return "gfx908"
	}
	if strings.Contains(lower, "7900") || strings.Contains(lower, "gfx1100") {
		return "gfx1100"
	}
	if strings.Contains(lower, "7600") || strings.Contains(lower, "gfx1102") {
		return "gfx1102"
	}
	return ""
}

func inferPCIAddress(dir string, props map[string]string) string {
	if p, ok := props["pci_address"]; ok && p != "" {
		return p
	}
	if p, ok := props["pci_bus_id"]; ok && p != "" {
		return p
	}
	if p, ok := props["location_id"]; ok && p != "" {
		return p
	}
	if data, err := os.ReadFile(filepath.Join(dir, "pci_address")); err == nil {
		return strings.TrimSpace(string(data))
	}
	if data, err := os.ReadFile(filepath.Join(dir, "pci_bus_id")); err == nil {
		return strings.TrimSpace(string(data))
	}
	return ""
}

func readVRAMSize(dir string, props map[string]string) uint64 {
	// 1. Direct files
	for _, fname := range []string{"vram_size", "vram_size_bytes", "mem_info_vram_total"} {
		if data, err := os.ReadFile(filepath.Join(dir, fname)); err == nil {
			if sz, err := parseByteSize(string(data)); err == nil && sz > 0 {
				return sz
			}
		}
	}
	// 2. Properties
	for _, key := range []string{"vram_size", "vram_size_bytes", "size_in_bytes"} {
		if val, ok := props[key]; ok {
			if sz, err := parseByteSize(val); err == nil && sz > 0 {
				return sz
			}
		}
	}
	// 3. mem_banks
	banksDir := filepath.Join(dir, "mem_banks")
	if entries, err := os.ReadDir(banksDir); err == nil {
		var total uint64
		for _, e := range entries {
			if !e.IsDir() {
				continue
			}
			bpropsData, err := os.ReadFile(filepath.Join(banksDir, e.Name(), "properties"))
			if err != nil {
				continue
			}
			bprops := parseProperties(string(bpropsData))
			heapType := strings.ToLower(bprops["heap_type"])
			if heapType == "0" || heapType == "vram" || heapType == "" {
				if sz, err := parseByteSize(bprops["size_in_bytes"]); err == nil {
					total += sz
				}
			}
		}
		if total > 0 {
			return total
		}
	}
	return 0
}

func readBAR1Size(dir string, props map[string]string, vramSize uint64, pciAddr string, sysfsRoot string) uint64 {
	// 1. Direct files
	for _, fname := range []string{"bar1_size", "bar1_size_bytes"} {
		if data, err := os.ReadFile(filepath.Join(dir, fname)); err == nil {
			if sz, err := parseByteSize(string(data)); err == nil && sz > 0 {
				return sz
			}
		}
	}
	// 2. Properties
	for _, key := range []string{"bar1_size", "bar1_size_bytes", "aperture_size"} {
		if val, ok := props[key]; ok {
			if sz, err := parseByteSize(val); err == nil && sz > 0 {
				return sz
			}
		}
	}
	// 3. Check PCI resource file in node dir or sysfs bus/pci
	if data, err := os.ReadFile(filepath.Join(dir, "resource")); err == nil {
		if sz, err := parsePCIResourceBAR1(string(data)); err == nil && sz > 0 {
			return sz
		}
	}
	if pciAddr != "" {
		pciPaths := []string{
			filepath.Join(sysfsRoot, "bus", "pci", "devices", pciAddr, "resource"),
			filepath.Join(sysfsRoot, "sys", "bus", "pci", "devices", pciAddr, "resource"),
		}
		for _, p := range pciPaths {
			if data, err := os.ReadFile(p); err == nil {
				if sz, err := parsePCIResourceBAR1(string(data)); err == nil && sz > 0 {
					return sz
				}
			}
		}
	}
	// 4. ReBAR flag in properties or file
	if r, ok := props["rebar"]; ok {
		if r == "1" || strings.EqualFold(r, "true") {
			return vramSize
		}
		return SmallBARThresholdBytes
	}
	return 0
}

func evaluateReBAR(node *AMDDeviceNode) {
	if node.BAR1Size == 0 && node.VRAMSize == 0 {
		return
	}

	if node.BAR1Size < SmallBARThresholdBytes {
		node.LargeBAR = false
		node.Warnings = append(node.Warnings, fmt.Sprintf(
			"node %d (%s): small BAR detected (BAR1 size %d bytes < 256MB threshold); Resizable BAR (ReBAR) disabled, GPU Direct P2P will be degraded or restricted",
			node.NodeID, node.Name, node.BAR1Size,
		))
	} else if node.VRAMSize > 0 && node.BAR1Size < node.VRAMSize {
		node.LargeBAR = false
		node.Warnings = append(node.Warnings, fmt.Sprintf(
			"node %d (%s): small BAR detected (BAR1 size %d bytes < VRAM size %d bytes); full-aperture Resizable BAR (ReBAR) disabled",
			node.NodeID, node.Name, node.BAR1Size, node.VRAMSize,
		))
	} else {
		node.LargeBAR = true
	}
}

func parseIOLinks(nodeDir string, fromNodeID int) []*PeerLink {
	linksDir := filepath.Join(nodeDir, "io_links")
	entries, err := os.ReadDir(linksDir)
	if err != nil {
		return nil
	}

	var links []*PeerLink
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		data, err := os.ReadFile(filepath.Join(linksDir, e.Name(), "properties"))
		if err != nil {
			continue
		}
		p := parseProperties(string(data))
		toID, _ := strconv.Atoi(p["node_to"])
		fromID := fromNodeID
		if fStr, ok := p["node_from"]; ok {
			if f, err := strconv.Atoi(fStr); err == nil {
				fromID = f
			}
		}

		lType := LinkTypePCIeSwitch
		typeStr := strings.ToLower(p["type"])
		if typeStr == "2" || strings.Contains(typeStr, "xgmi") {
			lType = LinkTypeXGMI
		} else if typeStr == "3" || strings.Contains(typeStr, "host") || strings.Contains(typeStr, "qpi") {
			lType = LinkTypeHostBridge
		}

		var bw float64
		if b, err := strconv.ParseFloat(p["bandwidth_max"], 64); err == nil {
			bw = b
		} else if b, err := strconv.ParseFloat(p["bandwidth"], 64); err == nil {
			bw = b
		}

		weight, _ := strconv.Atoi(p["weight"])

		var intermediateBridges []string
		if bridgesStr, ok := p["intermediate_bridges"]; ok && bridgesStr != "" {
			for _, b := range strings.Split(bridgesStr, ",") {
				b = strings.TrimSpace(b)
				if b != "" {
					intermediateBridges = append(intermediateBridges, b)
				}
			}
		}

		acsRedirect := false
		if checkACSRedirect(p["acs_redirect"]) || checkACSRedirect(p["acs_flags"]) || checkACSRedirect(p["acs_ctrl"]) {
			acsRedirect = true
		}

		links = append(links, &PeerLink{
			FromNodeID:          fromID,
			ToNodeID:            toID,
			From:                fromID,
			To:                  toID,
			Type:                lType,
			BandwidthGBs:        bw,
			Weight:              weight,
			IntermediateBridges: intermediateBridges,
			ACSRedirectDetected: acsRedirect,
		})
	}
	return links
}

// deriveTopologyLinks derives peer connections when io_links is absent in sysfs.
func deriveTopologyLinks(nodes []*AMDDeviceNode, hives map[int]string, switches map[int]string, sysfsRoot string) []*PeerLink {
	var links []*PeerLink
	n := len(nodes)

	// Check if all nodes belong to same xGMI hive or are MI300X/MI250
	allXGMI := false
	if len(hives) == n {
		h0 := hives[nodes[0].NodeID]
		same := true
		for _, node := range nodes {
			if hives[node.NodeID] != h0 || h0 == "" {
				same = false
				break
			}
		}
		allXGMI = same
	} else {
		miCount := 0
		for _, node := range nodes {
			if strings.Contains(strings.ToLower(node.Name), "mi300") || node.Arch == "gfx942" {
				miCount++
			}
		}
		if miCount == n {
			allXGMI = true
		}
	}

	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if i == j {
				continue
			}
			nFrom := nodes[i]
			nTo := nodes[j]

			if allXGMI {
				links = append(links, &PeerLink{
					FromNodeID:   nFrom.NodeID,
					ToNodeID:     nTo.NodeID,
					From:         nFrom.NodeID,
					To:           nTo.NodeID,
					Type:         LinkTypeXGMI,
					P2PSupported: true,
					Direct:       true,
					BandwidthGBs: 400.0,
				})
				continue
			}

			// Check shared PCIe switch
			swFrom := switches[nFrom.NodeID]
			swTo := switches[nTo.NodeID]
			if swFrom != "" && swFrom == swTo {
				links = append(links, &PeerLink{
					FromNodeID:          nFrom.NodeID,
					ToNodeID:            nTo.NodeID,
					From:                nFrom.NodeID,
					To:                  nTo.NodeID,
					Type:                LinkTypePCIeSwitch,
					IntermediateBridges: []string{swFrom},
					BandwidthGBs:        64.0,
				})
				continue
			}

			// Default: HostBridge
			links = append(links, &PeerLink{
				FromNodeID:    nFrom.NodeID,
				ToNodeID:      nTo.NodeID,
				From:          nFrom.NodeID,
				To:            nTo.NodeID,
				Type:          LinkTypeHostBridge,
				P2PSupported:  false,
				RefusalReason: fmt.Sprintf("P2P route crosses HostBridge between node %d and node %d: direct peer DMA unsupported", nFrom.NodeID, nTo.NodeID),
			})
		}
	}
	return links
}

// resolveAndValidateLinks inspects intermediate bridges for ACS Request Redirect and sets P2P postures.
func resolveAndValidateLinks(nodes []*AMDDeviceNode, rawLinks []*PeerLink, sysfsRoot string) ([]*PeerLink, map[int]map[int]*PeerLink) {
	matrix := make(map[int]map[int]*PeerLink)
	for _, n := range nodes {
		matrix[n.NodeID] = make(map[int]*PeerLink)
	}

	// Index raw links
	for _, l := range rawLinks {
		if matrix[l.FromNodeID] == nil {
			matrix[l.FromNodeID] = make(map[int]*PeerLink)
		}
		matrix[l.FromNodeID][l.ToNodeID] = l
	}

	// Ensure symmetric links
	for _, l := range rawLinks {
		if matrix[l.ToNodeID] != nil && matrix[l.ToNodeID][l.FromNodeID] == nil {
			rev := &PeerLink{
				FromNodeID:          l.ToNodeID,
				ToNodeID:            l.FromNodeID,
				From:                l.ToNodeID,
				To:                  l.FromNodeID,
				Type:                l.Type,
				BandwidthGBs:        l.BandwidthGBs,
				Weight:              l.Weight,
				IntermediateBridges: append([]string(nil), l.IntermediateBridges...),
				ACSRedirectDetected: l.ACSRedirectDetected,
			}
			matrix[l.ToNodeID][l.FromNodeID] = rev
		}
	}

	// Validate each link
	var finalized []*PeerLink
	for fromID := range matrix {
		for toID := range matrix[fromID] {
			link := matrix[fromID][toID]

			// Check intermediate bridges for ACS Request Redirect
			for _, bridge := range link.IntermediateBridges {
				if inspectBridgeACSRedirect(bridge, sysfsRoot) {
					link.ACSRedirectDetected = true
					break
				}
			}

			// Decide P2P support posture
			if link.ACSRedirectDetected {
				link.P2PSupported = false
				link.Direct = false
				link.RefusalReason = fmt.Sprintf("PCIe ACS Request Redirect detected on intermediate bridge (%v): peer-to-peer traffic redirected upstream; fail-closed P2P route refusal", link.IntermediateBridges)
			} else if link.Type == LinkTypeHostBridge {
				link.P2PSupported = false
				link.Direct = false
				if link.RefusalReason == "" {
					link.RefusalReason = fmt.Sprintf("P2P route crosses HostBridge between node %d and node %d: direct peer DMA unsupported", link.FromNodeID, link.ToNodeID)
				}
			} else {
				link.P2PSupported = true
				link.Direct = true
				if link.BandwidthGBs == 0 {
					if link.Type == LinkTypeXGMI {
						link.BandwidthGBs = 400.0
					} else {
						link.BandwidthGBs = 64.0
					}
				}
			}

			finalized = append(finalized, link)
		}
	}

	// Deterministic sorting of links
	sort.Slice(finalized, func(i, j int) bool {
		if finalized[i].FromNodeID != finalized[j].FromNodeID {
			return finalized[i].FromNodeID < finalized[j].FromNodeID
		}
		return finalized[i].ToNodeID < finalized[j].ToNodeID
	})

	return finalized, matrix
}

// inspectBridgeACSRedirect searches sysfs for bridge ACS attributes and checks if Request Redirect is active.
func inspectBridgeACSRedirect(bridgeID string, sysfsRoot string) bool {
	candidateDirs := []string{
		filepath.Join(sysfsRoot, "bridges", bridgeID),
		filepath.Join(sysfsRoot, "bus", "pci", "devices", bridgeID),
		filepath.Join(sysfsRoot, "sys", "bus", "pci", "devices", bridgeID),
		filepath.Join(sysfsRoot, "pci", bridgeID),
		filepath.Join(sysfsRoot, bridgeID),
	}

	for _, bDir := range candidateDirs {
		info, err := os.Stat(bDir)
		if err != nil || !info.IsDir() {
			continue
		}
		for _, fname := range []string{"acs_flags", "acs_redirect", "acs_ctrl", "acs_status", "properties"} {
			if data, err := os.ReadFile(filepath.Join(bDir, fname)); err == nil {
				if checkACSRedirect(string(data)) {
					return true
				}
			}
		}
	}
	return false
}

// checkACSRedirect inspects ACS capability/control text or register bits for active Request Redirect.
func checkACSRedirect(content string) bool {
	content = strings.TrimSpace(content)
	if content == "" {
		return false
	}
	lower := strings.ToLower(content)

	if lower == "1" || lower == "true" || lower == "enabled" || lower == "on" {
		return true
	}
	if lower == "0" || lower == "false" || lower == "disabled" || lower == "off" {
		return false
	}

	// Hex register parsing (PCIe ACS Control Register bit 2 is P2P Request Redirect Enable: 0x0004)
	if strings.HasPrefix(lower, "0x") {
		val, err := strconv.ParseUint(strings.TrimPrefix(lower, "0x"), 16, 32)
		if err == nil {
			return (val & 0x0004) != 0
		}
	}

	// lspci flag formatting: ReqRedir+ vs ReqRedir-
	if strings.Contains(lower, "reqredir+") || strings.Contains(lower, "requestredirect+") {
		return true
	}
	if strings.Contains(lower, "reqredir-") || strings.Contains(lower, "requestredirect-") {
		return false
	}

	// Keywords indicating enabled request redirect
	if strings.Contains(lower, "reqredir") || strings.Contains(lower, "request redirect") || strings.Contains(lower, "request_redirect") {
		if strings.Contains(lower, "disable") || strings.Contains(lower, "off") || strings.Contains(lower, "false") || strings.Contains(lower, "-") {
			return false
		}
		return true
	}

	// Token scan
	tokens := strings.FieldsFunc(content, func(r rune) bool {
		return r == ',' || r == ' ' || r == '\t' || r == '\n' || r == ';'
	})
	for _, tok := range tokens {
		tokLower := strings.ToLower(tok)
		if tokLower == "rr+" || tokLower == "rr:1" || tokLower == "rr:enabled" || tokLower == "rr" {
			return true
		}
	}

	return false
}

func parseByteSize(s string) (uint64, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return 0, fmt.Errorf("empty size string")
	}
	if strings.HasPrefix(s, "0x") || strings.HasPrefix(s, "0X") {
		return strconv.ParseUint(s[2:], 16, 64)
	}

	mult := uint64(1)
	sUpper := strings.ToUpper(s)
	if strings.HasSuffix(sUpper, "B") {
		sUpper = strings.TrimSuffix(sUpper, "B")
	}
	if strings.HasSuffix(sUpper, "K") {
		mult = 1024
		sUpper = strings.TrimSuffix(sUpper, "K")
	} else if strings.HasSuffix(sUpper, "M") {
		mult = 1024 * 1024
		sUpper = strings.TrimSuffix(sUpper, "M")
	} else if strings.HasSuffix(sUpper, "G") {
		mult = 1024 * 1024 * 1024
		sUpper = strings.TrimSuffix(sUpper, "G")
	} else if strings.HasSuffix(sUpper, "T") {
		mult = 1024 * 1024 * 1024 * 1024
		sUpper = strings.TrimSuffix(sUpper, "T")
	}

	val, err := strconv.ParseUint(strings.TrimSpace(sUpper), 10, 64)
	if err != nil {
		return 0, err
	}
	return val * mult, nil
}

func parsePCIResourceBAR1(content string) (uint64, error) {
	lines := strings.Split(strings.TrimSpace(content), "\n")
	if len(lines) < 2 {
		return 0, fmt.Errorf("insufficient lines in resource file")
	}
	// Line 1 is BAR1
	fields := strings.Fields(lines[1])
	if len(fields) < 2 {
		return 0, fmt.Errorf("malformed resource line 1: %q", lines[1])
	}
	start, err := strconv.ParseUint(strings.TrimPrefix(fields[0], "0x"), 16, 64)
	if err != nil {
		return 0, err
	}
	end, err := strconv.ParseUint(strings.TrimPrefix(fields[1], "0x"), 16, 64)
	if err != nil {
		return 0, err
	}
	if end < start {
		return 0, nil
	}
	return end - start + 1, nil
}

func parseProperties(content string) map[string]string {
	res := make(map[string]string)
	scanner := bufio.NewScanner(strings.NewReader(content))
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		var key, val string
		if eqIdx := strings.Index(line, "="); eqIdx != -1 {
			key = strings.TrimSpace(line[:eqIdx])
			val = strings.TrimSpace(line[eqIdx+1:])
		} else if colonIdx := strings.Index(line, ":"); colonIdx != -1 && !strings.Contains(line[:colonIdx], " ") {
			key = strings.TrimSpace(line[:colonIdx])
			val = strings.TrimSpace(line[colonIdx+1:])
		} else {
			parts := strings.Fields(line)
			if len(parts) >= 2 {
				key = parts[0]
				val = strings.TrimSpace(line[len(parts[0]):])
			} else if len(parts) == 1 {
				key = parts[0]
				val = "1"
			}
		}
		if key != "" {
			res[strings.ToLower(key)] = val
		}
	}
	return res
}
