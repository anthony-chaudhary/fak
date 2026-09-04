package compute

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func createMockNode(t *testing.T, sysfsRoot string, nodeID int, name string, arch string, vramBytes uint64, bar1Bytes uint64, pciAddr string, hiveID string, pcieSwitch string) string {
	t.Helper()
	nodeDir := filepath.Join(sysfsRoot, "class", "kfd", "kfd", "topology", "nodes", fmt.Sprintf("node%d", nodeID))
	if err := os.MkdirAll(nodeDir, 0o755); err != nil {
		t.Fatalf("mkdir node dir: %v", err)
	}

	props := []string{
		fmt.Sprintf("node_id %d", nodeID),
		fmt.Sprintf("gpu_id %d", 1000+nodeID),
		fmt.Sprintf("name %s", name),
		fmt.Sprintf("vram_size %d", vramBytes),
		fmt.Sprintf("bar1_size %d", bar1Bytes),
	}
	if arch != "" {
		props = append(props, fmt.Sprintf("arch %s", arch))
	}
	if pciAddr != "" {
		props = append(props, fmt.Sprintf("pci_address %s", pciAddr))
	}
	if hiveID != "" {
		props = append(props, fmt.Sprintf("hive_id %s", hiveID))
	}
	if pcieSwitch != "" {
		props = append(props, fmt.Sprintf("pcie_switch %s", pcieSwitch))
	}

	if err := os.WriteFile(filepath.Join(nodeDir, "properties"), []byte(strings.Join(props, "\n")), 0o644); err != nil {
		t.Fatalf("write properties: %v", err)
	}
	return nodeDir
}

func createMockIOLink(t *testing.T, nodeDir string, linkID int, fromID int, toID int, linkType int, bwMax int, intermediateBridges string, acsRedirect string) {
	t.Helper()
	linkDir := filepath.Join(nodeDir, "io_links", fmt.Sprintf("%d", linkID))
	if err := os.MkdirAll(linkDir, 0o755); err != nil {
		t.Fatalf("mkdir link dir: %v", err)
	}

	props := []string{
		fmt.Sprintf("type %d", linkType),
		fmt.Sprintf("node_from %d", fromID),
		fmt.Sprintf("node_to %d", toID),
		fmt.Sprintf("bandwidth_max %d", bwMax),
		"weight 15",
	}
	if intermediateBridges != "" {
		props = append(props, fmt.Sprintf("intermediate_bridges %s", intermediateBridges))
	}
	if acsRedirect != "" {
		props = append(props, fmt.Sprintf("acs_redirect %s", acsRedirect))
	}

	if err := os.WriteFile(filepath.Join(linkDir, "properties"), []byte(strings.Join(props, "\n")), 0o644); err != nil {
		t.Fatalf("write link properties: %v", err)
	}
}

func createMockBridge(t *testing.T, sysfsRoot string, bridgeID string, acsFlags string, acsRedirect string) {
	t.Helper()
	bridgeDir := filepath.Join(sysfsRoot, "bridges", bridgeID)
	if err := os.MkdirAll(bridgeDir, 0o755); err != nil {
		t.Fatalf("mkdir bridge dir: %v", err)
	}
	if acsFlags != "" {
		if err := os.WriteFile(filepath.Join(bridgeDir, "acs_flags"), []byte(acsFlags), 0o644); err != nil {
			t.Fatalf("write bridge acs_flags: %v", err)
		}
	}
	if acsRedirect != "" {
		if err := os.WriteFile(filepath.Join(bridgeDir, "acs_redirect"), []byte(acsRedirect), 0o644); err != nil {
			t.Fatalf("write bridge acs_redirect: %v", err)
		}
	}
}

// TestAMDGPUDirect is the primary umbrella test runner executing all AMD GPU Direct topology and ReBAR/ACS validations.
func TestAMDGPUDirect(t *testing.T) {
	t.Run("MI300X_xGMI_Mesh", testMI300XxGMIMesh)
	t.Run("RX7900_PCIeSwitch", testRX7900PCIeSwitch)
	t.Run("ACSConflict_FailClosed", testACSConflictFailClosed)
	t.Run("SmallBAR_Detection", testSmallBARDetection)
	t.Run("JSONReport_Serialization", testJSONReportSerialization)
	t.Run("HostBridge_Refusal", testHostBridgeRefusal)
	t.Run("InvalidRoute_Refusal", testInvalidRouteRefusal)
}

func TestAMDGPUDirect_MI300X_xGMI(t *testing.T)       { testMI300XxGMIMesh(t) }
func TestAMDGPUDirect_RX7900_PCIeSwitch(t *testing.T) { testRX7900PCIeSwitch(t) }
func TestAMDGPUDirect_ACSConflict(t *testing.T)       { testACSConflictFailClosed(t) }
func TestAMDGPUDirect_SmallBAR(t *testing.T)          { testSmallBARDetection(t) }
func TestAMDGPUDirect_JSONReport(t *testing.T)        { testJSONReportSerialization(t) }

// testMI300XxGMIMesh tests 4-node MI300X xGMI mesh topology discovery with ReBAR enabled.
func testMI300XxGMIMesh(t *testing.T) {
	root := t.TempDir()

	const mi300xVRAM = uint64(192) * 1024 * 1024 * 1024 // 192 GB
	const mi300xBAR1 = uint64(192) * 1024 * 1024 * 1024 // 192 GB Large BAR
	hiveID := "0x9876543210abcdef"

	nodeDirs := make([]string, 4)
	for i := 0; i < 4; i++ {
		pciAddr := fmt.Sprintf("0000:%02x:00.0", 0x10+i)
		nodeDirs[i] = createMockNode(t, root, i, "AMD Instinct MI300X", "gfx942", mi300xVRAM, mi300xBAR1, pciAddr, hiveID, "")
	}

	// Create xGMI io_links between all pairs
	for i := 0; i < 4; i++ {
		linkIdx := 0
		for j := 0; j < 4; j++ {
			if i == j {
				continue
			}
			// type 2 is xGMI in KFD
			createMockIOLink(t, nodeDirs[i], linkIdx, i, j, 2, 400, "", "0")
			linkIdx++
		}
	}

	topo, err := DiscoverTopology(root)
	if err != nil {
		t.Fatalf("DiscoverTopology failed: %v", err)
	}

	if len(topo.Nodes) != 4 {
		t.Fatalf("expected 4 nodes, got %d", len(topo.Nodes))
	}

	for _, n := range topo.Nodes {
		if !n.IsLargeBAR() {
			t.Errorf("node %d: expected IsLargeBAR to be true", n.NodeID)
		}
		if len(n.Warnings) > 0 {
			t.Errorf("node %d: expected 0 warnings, got %v", n.NodeID, n.Warnings)
		}
	}

	// Validate P2P route between node 0 and node 1
	link, err := topo.ValidateP2PRoute(0, 1)
	if err != nil {
		t.Fatalf("ValidateP2PRoute(0, 1) failed: %v", err)
	}
	if link == nil {
		t.Fatal("expected non-nil PeerLink")
	}
	if link.Type != LinkTypeXGMI {
		t.Errorf("expected link type %s, got %s", LinkTypeXGMI, link.Type)
	}
	if !link.P2PSupported {
		t.Errorf("expected P2PSupported to be true")
	}
	if !link.Direct {
		t.Errorf("expected Direct to be true")
	}
	if link.ACSRedirectDetected {
		t.Errorf("expected ACSRedirectDetected to be false")
	}
	if link.BandwidthGBs < 400.0 {
		t.Errorf("expected bandwidth >= 400 GB/s, got %f", link.BandwidthGBs)
	}

	report := topo.AuditReport
	if report == nil {
		t.Fatal("expected non-nil AuditReport")
	}
	if !report.Passed {
		t.Errorf("expected AuditReport.Passed to be true, got false")
	}
	if report.Status != "PASS" {
		t.Errorf("expected status PASS, got %s", report.Status)
	}
	if report.ACSRedirectDetected {
		t.Errorf("expected ACSRedirectDetected to be false in report")
	}
	if report.SmallBARDetected {
		t.Errorf("expected SmallBARDetected to be false in report")
	}
}

// testRX7900PCIeSwitch tests dual RX 7900 GPUs connected via a PCIe switch with ReBAR and clean ACS posture.
func testRX7900PCIeSwitch(t *testing.T) {
	root := t.TempDir()

	const rx7900VRAM = uint64(24) * 1024 * 1024 * 1024 // 24 GB
	const rx7900BAR1 = uint64(24) * 1024 * 1024 * 1024 // 24 GB Large BAR
	switchBridge := "0000:01:00.0"

	nodeDir0 := createMockNode(t, root, 0, "AMD Radeon RX 7900 XTX", "gfx1100", rx7900VRAM, rx7900BAR1, "0000:03:00.0", "", switchBridge)
	nodeDir1 := createMockNode(t, root, 1, "AMD Radeon RX 7900 XTX", "gfx1100", rx7900VRAM, rx7900BAR1, "0000:04:00.0", "", switchBridge)

	// Create PCIe switch bridge with ACS Request Redirect DISABLED ("ReqRedir-")
	createMockBridge(t, root, switchBridge, "ReqRedir-", "0")

	// io_links with intermediate PCIe switch bridge
	createMockIOLink(t, nodeDir0, 0, 0, 1, 1, 64, switchBridge, "0")
	createMockIOLink(t, nodeDir1, 0, 1, 0, 1, 64, switchBridge, "0")

	topo, err := DiscoverTopology(root)
	if err != nil {
		t.Fatalf("DiscoverTopology failed: %v", err)
	}

	if len(topo.Nodes) != 2 {
		t.Fatalf("expected 2 nodes, got %d", len(topo.Nodes))
	}

	for _, n := range topo.Nodes {
		if !n.IsLargeBAR() {
			t.Errorf("node %d: expected IsLargeBAR to be true", n.NodeID)
		}
	}

	link, err := topo.ValidateP2PRoute(0, 1)
	if err != nil {
		t.Fatalf("ValidateP2PRoute(0, 1) failed: %v", err)
	}
	if link.Type != LinkTypePCIeSwitch {
		t.Errorf("expected link type %s, got %s", LinkTypePCIeSwitch, link.Type)
	}
	if !link.P2PSupported {
		t.Errorf("expected P2PSupported to be true")
	}
	if !link.Direct {
		t.Errorf("expected Direct to be true")
	}
	if link.ACSRedirectDetected {
		t.Errorf("expected ACSRedirectDetected to be false")
	}

	if topo.AuditReport.Status != "PASS" {
		t.Errorf("expected report status PASS, got %s", topo.AuditReport.Status)
	}
}

// testACSConflictFailClosed tests fail-closed refusal when an intermediate bridge has ACS Request Redirect enabled.
func testACSConflictFailClosed(t *testing.T) {
	root := t.TempDir()

	const vram = uint64(24) * 1024 * 1024 * 1024
	const bar1 = uint64(24) * 1024 * 1024 * 1024
	switchBridge := "0000:01:00.0"

	nodeDir0 := createMockNode(t, root, 0, "AMD Radeon RX 7900 XTX", "gfx1100", vram, bar1, "0000:03:00.0", "", switchBridge)
	nodeDir1 := createMockNode(t, root, 1, "AMD Radeon RX 7900 XTX", "gfx1100", vram, bar1, "0000:04:00.0", "", switchBridge)

	// Switch bridge has ACS Request Redirect ENABLED ("ReqRedir+")
	createMockBridge(t, root, switchBridge, "SrcValid+ ReqRedir+ CmpltRedir+", "1")

	createMockIOLink(t, nodeDir0, 0, 0, 1, 1, 64, switchBridge, "")
	createMockIOLink(t, nodeDir1, 0, 1, 0, 1, 64, switchBridge, "")

	topo, err := DiscoverTopology(root)
	if err != nil {
		t.Fatalf("DiscoverTopology failed: %v", err)
	}

	// Route validation MUST refuse fail-closed
	link, err := topo.ValidateP2PRoute(0, 1)
	if err == nil {
		t.Fatal("expected ValidateP2PRoute to fail due to ACS Request Redirect, but got nil error")
	}

	if link == nil {
		t.Fatal("expected link to be returned with refusal details")
	}
	if link.P2PSupported {
		t.Errorf("expected link.P2PSupported to be false under ACS redirect conflict")
	}
	if !link.ACSRedirectDetected {
		t.Errorf("expected link.ACSRedirectDetected to be true")
	}
	if !strings.Contains(link.RefusalReason, "ACS Request Redirect") {
		t.Errorf("expected refusal reason to mention ACS Request Redirect, got: %s", link.RefusalReason)
	}
	if !strings.Contains(err.Error(), "ACS Request Redirect") {
		t.Errorf("expected error string to mention ACS Request Redirect, got: %v", err)
	}

	// Audit report must reflect fail-closed status
	report := topo.AuditReport
	if report.Passed {
		t.Errorf("expected AuditReport.Passed to be false")
	}
	if !report.ACSRedirectDetected {
		t.Errorf("expected AuditReport.ACSRedirectDetected to be true")
	}
	if report.Status != "REFUSED" {
		t.Errorf("expected report status REFUSED, got %s", report.Status)
	}
	if len(report.Refusals) == 0 {
		t.Errorf("expected report.Refusals to contain reasons, got empty")
	}
}

// testSmallBARDetection tests detection of Small BAR (< 256MB) and emission of diagnostic warnings.
func testSmallBARDetection(t *testing.T) {
	root := t.TempDir()

	const vram = uint64(24) * 1024 * 1024 * 1024    // 24 GB
	const smallBAR1 = uint64(128) * 1024 * 1024     // 128 MB (< 256MB)
	const legacy256BAR1 = uint64(256) * 1024 * 1024 // 256 MB (< 24GB VRAM)

	createMockNode(t, root, 0, "AMD Radeon RX 7900 XTX", "gfx1100", vram, smallBAR1, "0000:03:00.0", "hive0", "")
	createMockNode(t, root, 1, "AMD Radeon RX 7900 XTX", "gfx1100", vram, legacy256BAR1, "0000:04:00.0", "hive0", "")

	topo, err := DiscoverTopology(root)
	if err != nil {
		t.Fatalf("DiscoverTopology failed: %v", err)
	}

	node0 := topo.Nodes[0]
	if node0.IsLargeBAR() {
		t.Errorf("node 0: expected IsLargeBAR to be false for 128MB BAR1")
	}
	if len(node0.Warnings) == 0 {
		t.Errorf("node 0: expected diagnostic warnings for small BAR (< 256MB)")
	}
	foundSmallWarn0 := false
	for _, w := range node0.Warnings {
		if strings.Contains(w, "small BAR") || strings.Contains(w, "256MB") {
			foundSmallWarn0 = true
			break
		}
	}
	if !foundSmallWarn0 {
		t.Errorf("node 0: expected warning mentioning small BAR or 256MB, got %v", node0.Warnings)
	}

	node1 := topo.Nodes[1]
	if node1.IsLargeBAR() {
		t.Errorf("node 1: expected IsLargeBAR to be false for 256MB BAR1 vs 24GB VRAM")
	}
	if len(node1.Warnings) == 0 {
		t.Errorf("node 1: expected diagnostic warnings for BAR1 < VRAM")
	}

	// Audit report checks
	report := topo.AuditReport
	if !report.SmallBARDetected {
		t.Errorf("expected report.SmallBARDetected to be true")
	}
	if len(report.Warnings) == 0 {
		t.Errorf("expected report.Warnings to contain small BAR warnings")
	}
	if report.Status != "WARN" {
		t.Errorf("expected report status WARN, got %s", report.Status)
	}
}

// testJSONReportSerialization tests serialization of the structured AuditReport to JSON.
func testJSONReportSerialization(t *testing.T) {
	root := t.TempDir()

	const vram = uint64(192) * 1024 * 1024 * 1024
	const bar1 = uint64(192) * 1024 * 1024 * 1024

	createMockNode(t, root, 0, "AMD Instinct MI300X", "gfx942", vram, bar1, "0000:10:00.0", "0x123", "")
	createMockNode(t, root, 1, "AMD Instinct MI300X", "gfx942", vram, bar1, "0000:11:00.0", "0x123", "")

	topo, err := DiscoverTopology(root)
	if err != nil {
		t.Fatalf("DiscoverTopology failed: %v", err)
	}

	data, err := topo.AuditReport.JSON()
	if err != nil {
		t.Fatalf("AuditReport.JSON() failed: %v", err)
	}
	if len(data) == 0 {
		t.Fatal("AuditReport.JSON() returned empty bytes")
	}

	var parsed map[string]interface{}
	if err := json.Unmarshal(data, &parsed); err != nil {
		t.Fatalf("failed to unmarshal AuditReport JSON: %v", err)
	}

	if parsed["status"] != "PASS" {
		t.Errorf("expected JSON status PASS, got %v", parsed["status"])
	}
	if parsed["node_count"] != float64(2) {
		t.Errorf("expected JSON node_count 2, got %v", parsed["node_count"])
	}

	// Check TopologyMatrix JSON as well
	topoData, err := topo.JSON()
	if err != nil {
		t.Fatalf("TopologyMatrix.JSON() failed: %v", err)
	}
	if len(topoData) == 0 {
		t.Fatal("TopologyMatrix.JSON() returned empty bytes")
	}
}

// testHostBridgeRefusal verifies that routing across a HostBridge produces a refused route.
func testHostBridgeRefusal(t *testing.T) {
	root := t.TempDir()

	const vram = uint64(24) * 1024 * 1024 * 1024
	const bar1 = uint64(24) * 1024 * 1024 * 1024

	node0 := createMockNode(t, root, 0, "AMD Radeon RX 7900 XTX", "gfx1100", vram, bar1, "0000:03:00.0", "", "0000:00:01.0")
	node1 := createMockNode(t, root, 1, "AMD Radeon RX 7900 XTX", "gfx1100", vram, bar1, "0000:83:00.0", "", "0000:80:01.0")

	// io_link type 3 = HostBridge
	createMockIOLink(t, node0, 0, 0, 1, 3, 32, "", "")
	createMockIOLink(t, node1, 0, 1, 0, 3, 32, "", "")

	topo, err := DiscoverTopology(root)
	if err != nil {
		t.Fatalf("DiscoverTopology failed: %v", err)
	}

	link, err := topo.ValidateP2PRoute(0, 1)
	if err == nil {
		t.Fatal("expected route across HostBridge to be refused")
	}
	if link == nil || link.P2PSupported {
		t.Errorf("expected P2PSupported to be false")
	}
	if link.Type != LinkTypeHostBridge {
		t.Errorf("expected link type HostBridge, got %s", link.Type)
	}
}

// testInvalidRouteRefusal tests error behavior on identical source/dest or non-existent nodes.
func testInvalidRouteRefusal(t *testing.T) {
	root := t.TempDir()
	createMockNode(t, root, 0, "AMD Instinct MI300X", "gfx942", 192<<30, 192<<30, "0000:10:00.0", "hive1", "")

	topo, err := DiscoverTopology(root)
	if err != nil {
		t.Fatalf("DiscoverTopology failed: %v", err)
	}

	// Same source and destination
	if _, err := topo.ValidateP2PRoute(0, 0); err == nil {
		t.Error("expected error when from == to")
	}

	// Non-existent node
	if _, err := topo.ValidateP2PRoute(0, 999); err == nil {
		t.Error("expected error when destination does not exist")
	}
	if _, err := topo.ValidateP2PRoute(999, 0); err == nil {
		t.Error("expected error when source does not exist")
	}
}
