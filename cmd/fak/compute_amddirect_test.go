package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

func TestComputeAMDDirect_FlagParsing(t *testing.T) {
	var stdout, stderr bytes.Buffer

	// Empty args -> usage exit 2
	code := runComputeAMDDirect(&stdout, &stderr, []string{})
	if code != 2 {
		t.Fatalf("runComputeAMDDirect with empty args exit = %d, want 2", code)
	}
	if !strings.Contains(stderr.String(), "usage:") {
		t.Errorf("stderr = %q, want usage", stderr.String())
	}

	// Unknown subcommand -> exit 2
	stdout.Reset()
	stderr.Reset()
	code = runComputeAMDDirect(&stdout, &stderr, []string{"unsupported_cmd"})
	if code != 2 {
		t.Fatalf("runComputeAMDDirect with unknown subcommand exit = %d, want 2", code)
	}
	if !strings.Contains(stderr.String(), "unknown subcommand") {
		t.Errorf("stderr = %q, want unknown subcommand error", stderr.String())
	}

	// Unknown flag -> exit 2
	stdout.Reset()
	stderr.Reset()
	code = runComputeAMDDirect(&stdout, &stderr, []string{"status", "--unrecognized-flag"})
	if code != 2 {
		t.Fatalf("runComputeAMDDirect with unknown flag exit = %d, want 2", code)
	}

	// Valid subcommands
	for _, sub := range []string{"status", "audit", "bench"} {
		stdout.Reset()
		stderr.Reset()
		code = runComputeAMDDirect(&stdout, &stderr, []string{sub, "--fixture", "default"})
		if code != 0 {
			t.Fatalf("runComputeAMDDirect(%q) exit = %d, want 0; stderr: %s", sub, code, stderr.String())
		}
		if stdout.Len() == 0 {
			t.Errorf("runComputeAMDDirect(%q) produced empty stdout", sub)
		}
	}

	// Flag before subcommand: --json status
	stdout.Reset()
	stderr.Reset()
	code = runComputeAMDDirect(&stdout, &stderr, []string{"--json", "status", "--fixture", "default"})
	if code != 0 {
		t.Fatalf("runComputeAMDDirect(--json status) exit = %d, want 0", code)
	}
	var st AMDDirectStatusOutput
	if err := json.Unmarshal(stdout.Bytes(), &st); err != nil {
		t.Fatalf("failed to unmarshal JSON from prefix flag: %v", err)
	}
}

func TestComputeAMDDirect_JSONRendering(t *testing.T) {
	var stdout, stderr bytes.Buffer

	// 1. status --json
	stdout.Reset()
	stderr.Reset()
	code := runComputeAMDDirect(&stdout, &stderr, []string{"status", "--json", "--fixture", "default"})
	if code != 0 {
		t.Fatalf("status --json returned %d: %s", code, stderr.String())
	}
	var statusOut AMDDirectStatusOutput
	if err := json.Unmarshal(stdout.Bytes(), &statusOut); err != nil {
		t.Fatalf("invalid JSON output from status: %v\nOutput: %s", err, stdout.String())
	}
	if statusOut.Schema != "fak-compute-amddirect-status/1" {
		t.Errorf("schema = %q, want 'fak-compute-amddirect-status/1'", statusOut.Schema)
	}
	if statusOut.Count != 2 || len(statusOut.Nodes) != 2 {
		t.Errorf("count = %d, nodes = %d, want 2", statusOut.Count, len(statusOut.Nodes))
	}
	for _, n := range statusOut.Nodes {
		if n.DeviceName == "" {
			t.Errorf("node %d has empty device name", n.NodeID)
		}
		if n.Architecture == "" {
			t.Errorf("node %d has empty architecture", n.NodeID)
		}
		if n.PCIeBDF == "" {
			t.Errorf("node %d has empty PCIe BDF", n.NodeID)
		}
		if n.TotalVRAMBytes == 0 {
			t.Errorf("node %d has 0 TotalVRAMBytes", n.NodeID)
		}
		if n.BAR1SizeBytes == 0 {
			t.Errorf("node %d has 0 BAR1SizeBytes", n.NodeID)
		}
		if !n.IsLargeBAR {
			t.Errorf("node %d expected Large BAR on default fixture", n.NodeID)
		}
	}

	// 2. audit --json
	stdout.Reset()
	stderr.Reset()
	code = runComputeAMDDirect(&stdout, &stderr, []string{"audit", "--json", "--fixture", "default"})
	if code != 0 {
		t.Fatalf("audit --json returned %d: %s", code, stderr.String())
	}
	var auditOut AMDDirectAuditOutput
	if err := json.Unmarshal(stdout.Bytes(), &auditOut); err != nil {
		t.Fatalf("invalid JSON output from audit: %v\nOutput: %s", err, stdout.String())
	}
	if auditOut.Schema != "fak-compute-amddirect-audit/1" {
		t.Errorf("schema = %q, want 'fak-compute-amddirect-audit/1'", auditOut.Schema)
	}
	if !auditOut.Healthy {
		t.Errorf("audit expected healthy=true on default fixture")
	}
	if len(auditOut.Checks) < 4 {
		t.Errorf("audit checks count = %d, want >= 4", len(auditOut.Checks))
	}
	expectedChecks := map[string]bool{
		"rebar_aperture":      false,
		"pcie_acs":            false,
		"dmabuf_capabilities": false,
		"rocm_rdma_driver":    false,
	}
	for _, c := range auditOut.Checks {
		expectedChecks[c.Name] = true
	}
	for name, found := range expectedChecks {
		if !found {
			t.Errorf("audit check %q not found in checks list", name)
		}
	}

	// 3. bench --json
	stdout.Reset()
	stderr.Reset()
	code = runComputeAMDDirect(&stdout, &stderr, []string{"bench", "--json", "--fixture", "default"})
	if code != 0 {
		t.Fatalf("bench --json returned %d: %s", code, stderr.String())
	}
	var benchOut AMDDirectBenchOutput
	if err := json.Unmarshal(stdout.Bytes(), &benchOut); err != nil {
		t.Fatalf("invalid JSON output from bench: %v\nOutput: %s", err, stdout.String())
	}
	if benchOut.Schema != "fak-compute-amddirect-bench/1" {
		t.Errorf("schema = %q, want 'fak-compute-amddirect-bench/1'", benchOut.Schema)
	}
	if benchOut.TotalPairs != 2 {
		t.Errorf("total_pairs = %d, want 2", benchOut.TotalPairs)
	}
	if benchOut.FabricBreakdown == nil {
		t.Errorf("missing fabric_breakdown in bench output")
	}
	if benchOut.RDMAMicrobenchmark == nil {
		t.Errorf("missing rdma_microbenchmark in bench output")
	}
	if benchOut.NVMeMicrobenchmark == nil {
		t.Errorf("missing nvme_microbenchmark in bench output")
	}
}

func TestComputeAMDDirect_SyntheticDiagnostics(t *testing.T) {
	var stdout, stderr bytes.Buffer

	// Case 1: Small BAR fixture (<256 MiB active)
	code := runComputeAMDDirect(&stdout, &stderr, []string{"audit", "--fixture", "smallbar"})
	if code != 0 {
		t.Fatalf("audit with smallbar exit = %d, want 0 (warning)", code)
	}
	textOut := stdout.String()
	if !strings.Contains(textOut, "Small BAR") {
		t.Errorf("expected 'Small BAR' in audit output, got:\n%s", textOut)
	}
	if !strings.Contains(textOut, "pci=realloc") {
		t.Errorf("expected kernel parameter 'pci=realloc' in remediation advice, got:\n%s", textOut)
	}
	if !strings.Contains(textOut, "Resizable BAR") && !strings.Contains(textOut, "ReBAR") {
		t.Errorf("expected ReBAR setting in remediation advice, got:\n%s", textOut)
	}

	// JSON check for Small BAR remediation
	stdout.Reset()
	stderr.Reset()
	code = runComputeAMDDirect(&stdout, &stderr, []string{"audit", "--json", "--fixture", "smallbar"})
	if code != 0 {
		t.Fatalf("audit --json with smallbar exit = %d, want 0", code)
	}
	var auditSmall AMDDirectAuditOutput
	if err := json.Unmarshal(stdout.Bytes(), &auditSmall); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}
	if auditSmall.NodesWithSmallBAR != 1 {
		t.Errorf("NodesWithSmallBAR = %d, want 1", auditSmall.NodesWithSmallBAR)
	}
	var foundSmallBARRem bool
	for _, r := range auditSmall.Remediations {
		if r.Issue == "small_bar" {
			foundSmallBARRem = true
			var hasParam bool
			for _, p := range r.BootParameters {
				if p == "pci=realloc" {
					hasParam = true
					break
				}
			}
			if !hasParam {
				t.Errorf("expected 'pci=realloc' in kernel parameters, got %v", r.BootParameters)
			}
			if len(r.BIOSSettings) == 0 {
				t.Errorf("expected BIOS settings in small_bar remediation")
			}
		}
	}
	if !foundSmallBARRem {
		t.Errorf("remediation for 'small_bar' not found in %v", auditSmall.Remediations)
	}

	// Case 2: PCIe ACS conflict fixture (Request Redirect active)
	stdout.Reset()
	stderr.Reset()
	code = runComputeAMDDirect(&stdout, &stderr, []string{"audit", "--fixture", "acs"})
	if code != 1 {
		t.Fatalf("audit with acs conflict exit = %d, want 1 (fail-closed)", code)
	}
	textOut = stdout.String()
	if !strings.Contains(textOut, "ACS Request Redirect") {
		t.Errorf("expected 'ACS Request Redirect' in audit output, got:\n%s", textOut)
	}
	if !strings.Contains(textOut, "pcie_acs_override") {
		t.Errorf("expected 'pcie_acs_override' in remediation advice, got:\n%s", textOut)
	}

	// JSON check for ACS remediation
	stdout.Reset()
	stderr.Reset()
	code = runComputeAMDDirect(&stdout, &stderr, []string{"audit", "--json", "--fixture", "acs"})
	if code != 1 {
		t.Fatalf("audit --json with acs conflict exit = %d, want 1", code)
	}
	var auditACS AMDDirectAuditOutput
	if err := json.Unmarshal(stdout.Bytes(), &auditACS); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}
	if auditACS.Healthy {
		t.Errorf("expected healthy=false on ACS conflict")
	}
	if !auditACS.ACSConflictDetected {
		t.Errorf("expected ACSConflictDetected=true")
	}
	var foundACSRem bool
	for _, r := range auditACS.Remediations {
		if r.Issue == "acs_redirect" {
			foundACSRem = true
			var hasParam bool
			for _, p := range r.BootParameters {
				if strings.Contains(p, "pcie_acs_override") {
					hasParam = true
					break
				}
			}
			if !hasParam {
				t.Errorf("expected pcie_acs_override in kernel parameters, got %v", r.BootParameters)
			}
		}
	}
	if !foundACSRem {
		t.Errorf("remediation for 'acs_redirect' not found in %v", auditACS.Remediations)
	}
}

func TestComputeAMDDirect_BandwidthLinkClassification(t *testing.T) {
	var stdout, stderr bytes.Buffer

	// Run bench on all-fabrics fixture
	code := runComputeAMDDirect(&stdout, &stderr, []string{"bench", "--json", "--fixture", "all-fabrics"})
	if code != 0 {
		t.Fatalf("bench --fixture all-fabrics exit = %d, want 0; stderr: %s", code, stderr.String())
	}

	var benchOut AMDDirectBenchOutput
	if err := json.Unmarshal(stdout.Bytes(), &benchOut); err != nil {
		t.Fatalf("unmarshal error: %v\nOutput: %s", err, stdout.String())
	}

	// Verify all four fabric types are categorized in fabric_breakdown
	expectedFabrics := []string{
		"InfinityFabric_xGMI",
		"PCIe_Switch_P2P",
		"PCIe_Host_Bridge",
		"RoCE_RDMA",
	}

	for _, fab := range expectedFabrics {
		count := benchOut.FabricBreakdown[fab]
		if count <= 0 {
			t.Errorf("fabric_breakdown[%q] = %d, want > 0", fab, count)
		}
	}

	// Verify measured pair details
	var hasXGMI, hasPCIeSwitch, hasHostBridge, hasRoCE bool
	for _, p := range benchOut.GPUPairs {
		if p.StagingCopies != 0 {
			t.Errorf("pair %d -> %d staging copies = %d, want 0", p.SrcNodeID, p.DstNodeID, p.StagingCopies)
		}
		if p.P2PCapable {
			if p.UnidirectionalBandwidthGBps <= 0 {
				t.Errorf("pair %d -> %d unidirectional BW = %.1f, want > 0", p.SrcNodeID, p.DstNodeID, p.UnidirectionalBandwidthGBps)
			}
			if p.BidirectionalBandwidthGBps < p.UnidirectionalBandwidthGBps {
				t.Errorf("pair %d -> %d bidirectional BW = %.1f < unidirectional %.1f",
					p.SrcNodeID, p.DstNodeID, p.BidirectionalBandwidthGBps, p.UnidirectionalBandwidthGBps)
			}
		}

		switch p.Fabric {
		case "InfinityFabric_xGMI":
			hasXGMI = true
			if p.UnidirectionalBandwidthGBps < 800.0 {
				t.Errorf("xGMI BW = %.1f, expected ~896 GB/s", p.UnidirectionalBandwidthGBps)
			}
		case "PCIe_Switch_P2P":
			hasPCIeSwitch = true
			if p.UnidirectionalBandwidthGBps < 50.0 {
				t.Errorf("PCIe switch BW = %.1f, expected ~64 GB/s", p.UnidirectionalBandwidthGBps)
			}
		case "PCIe_Host_Bridge":
			hasHostBridge = true
			if p.UnidirectionalBandwidthGBps < 20.0 {
				t.Errorf("Host bridge BW = %.1f, expected ~32 GB/s", p.UnidirectionalBandwidthGBps)
			}
		case "RoCE_RDMA":
			hasRoCE = true
			if p.UnidirectionalBandwidthGBps < 40.0 {
				t.Errorf("RoCE RDMA BW = %.1f, expected ~50 GB/s", p.UnidirectionalBandwidthGBps)
			}
		}
	}

	if !hasXGMI || !hasPCIeSwitch || !hasHostBridge || !hasRoCE {
		t.Errorf("missing fabric coverage: xgmi=%v pcie_switch=%v host_bridge=%v roce=%v",
			hasXGMI, hasPCIeSwitch, hasHostBridge, hasRoCE)
	}

	// Test text rendering contains all 4 fabrics
	stdout.Reset()
	stderr.Reset()
	code = runComputeAMDDirect(&stdout, &stderr, []string{"bench", "--fixture", "all-fabrics"})
	if code != 0 {
		t.Fatalf("bench text exit = %d, want 0", code)
	}
	text := stdout.String()
	for _, fab := range expectedFabrics {
		if !strings.Contains(text, fab) {
			t.Errorf("bench text output missing fabric %q:\n%s", fab, text)
		}
	}
}

func TestComputeAMDDirect_StatusArchitectureEnumeration(t *testing.T) {
	var stdout, stderr bytes.Buffer

	code := runComputeAMDDirect(&stdout, &stderr, []string{"status", "--json", "--fixture", "all-fabrics"})
	if code != 0 {
		t.Fatalf("status --fixture all-fabrics exit = %d, want 0", code)
	}

	var statusOut AMDDirectStatusOutput
	if err := json.Unmarshal(stdout.Bytes(), &statusOut); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}

	archsFound := make(map[string]bool)
	for _, n := range statusOut.Nodes {
		archsFound[n.Architecture] = true
	}

	for _, wantArch := range []string{"gfx942", "gfx1151", "gfx1100"} {
		if !archsFound[wantArch] {
			t.Errorf("architecture %q not found in enumerated nodes: %v", wantArch, archsFound)
		}
	}
}

func TestComputeAMDDirect_CmdComputeRouting(t *testing.T) {
	var stdout, stderr bytes.Buffer

	// Empty args to compute
	code := runCompute(&stdout, &stderr, []string{})
	if code != 2 {
		t.Fatalf("runCompute([]) exit = %d, want 2", code)
	}

	// Help flag to compute
	stdout.Reset()
	stderr.Reset()
	code = runCompute(&stdout, &stderr, []string{"--help"})
	if code != 0 {
		t.Fatalf("runCompute([--help]) exit = %d, want 0", code)
	}

	// Unknown subcommand to compute
	stdout.Reset()
	stderr.Reset()
	code = runCompute(&stdout, &stderr, []string{"unknown_sub"})
	if code != 2 {
		t.Fatalf("runCompute([unknown_sub]) exit = %d, want 2", code)
	}

	// Route to amd-gpudirect status
	stdout.Reset()
	stderr.Reset()
	code = runCompute(&stdout, &stderr, []string{"amd-gpudirect", "status", "--fixture", "default"})
	if code != 0 {
		t.Fatalf("runCompute(amd-gpudirect status) exit = %d, want 0", code)
	}
	if !strings.Contains(stdout.String(), "AMD GPU Direct Status") {
		t.Errorf("stdout = %q, want 'AMD GPU Direct Status'", stdout.String())
	}
}
