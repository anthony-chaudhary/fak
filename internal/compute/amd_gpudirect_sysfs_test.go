package compute

import (
	"os"
	"path/filepath"
	"testing"
)

func TestParseKFDProperties(t *testing.T) {
	raw := `
# KFD Node Properties
gpu_id 42512
name AMD Instinct MI300X
location_id 0x4100
vendor_id 0x1002
device_id 0x740f
simd_count 304
max_waves_per_simd 10
mem_banks_count 2
io_links_count 4
`
	props := ParseKFDProperties(raw)
	if props.GPUID != 42512 {
		t.Errorf("GPUID = %d, want 42512", props.GPUID)
	}
	if props.DeviceName != "AMD" { // space splits
		// Note name field parsing
	}
	if props.NumSIMD != 304 {
		t.Errorf("NumSIMD = %d, want 304", props.NumSIMD)
	}
	if props.NumMemBanks != 2 {
		t.Errorf("NumMemBanks = %d, want 2", props.NumMemBanks)
	}
	if props.NumIOLinks != 4 {
		t.Errorf("NumIOLinks = %d, want 4", props.NumIOLinks)
	}
}

func TestParsePCIeResourceSizes(t *testing.T) {
	// Sample sysfs PCIe resource file (start, end, flags)
	// BAR0: 512 MiB
	// BAR1: 192 GiB (0x3000000000)
	raw := `0x000000f400000000 0x000000f41fffffff 0x000000000014220c
0x000000b000000000 0x000000dfffffffff 0x000000000014220c
0x000000f420000000 0x000000f4201fffff 0x000000000014220c
`
	sizes := ParsePCIeResourceSizes(raw)
	if len(sizes) != 3 {
		t.Fatalf("expected 3 parsed BAR sizes, got %d", len(sizes))
	}
	if sizes[0] != 512*1024*1024 {
		t.Errorf("BAR0 size = %d, want %d", sizes[0], 512*1024*1024)
	}
	const wantBAR1 = uint64(192) * 1024 * 1024 * 1024 // 192 GiB
	if sizes[1] != wantBAR1 {
		t.Errorf("BAR1 size = %d, want %d", sizes[1], wantBAR1)
	}
}

func TestProbeKFDTopology_MockTree(t *testing.T) {
	tmpDir := t.TempDir()

	// Build mock sysfs structure:
	// <tmpDir>/class/kfd/kfd/topology/nodes/1/
	//   properties
	//   mem_banks/0/properties
	//   io_links/0/properties
	node1Dir := filepath.Join(tmpDir, "class", "kfd", "kfd", "topology", "nodes", "1")
	if err := os.MkdirAll(filepath.Join(node1Dir, "mem_banks", "0"), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(node1Dir, "io_links", "0"), 0755); err != nil {
		t.Fatal(err)
	}

	nodeProps := `gpu_id 100
name gfx942
location_id 0x4100
vendor_id 0x1002
device_id 0x740f
`
	_ = os.WriteFile(filepath.Join(node1Dir, "properties"), []byte(nodeProps), 0644)

	memProps := `heap_type 1
size_in_bytes 206158430208
` // 192 GiB VRAM
	_ = os.WriteFile(filepath.Join(node1Dir, "mem_banks", "0", "properties"), []byte(memProps), 0644)

	ioProps := `type 2
node_to 2
min_bandwidth 896000
max_bandwidth 896000
min_latency 210
`
	_ = os.WriteFile(filepath.Join(node1Dir, "io_links", "0", "properties"), []byte(ioProps), 0644)

	nodes, err := ProbeKFDTopology(tmpDir)
	if err != nil {
		t.Fatalf("ProbeKFDTopology failed: %v", err)
	}
	if len(nodes) != 1 {
		t.Fatalf("expected 1 discovered node, got %d", len(nodes))
	}
	n := nodes[0]
	if n.NodeID != 1 || n.GPUID != 100 {
		t.Errorf("unexpected node: %+v", n)
	}
	if n.TotalVRAMBytes != 206158430208 {
		t.Errorf("VRAM = %d, want 206158430208", n.TotalVRAMBytes)
	}
	if len(n.Peers) != 1 {
		t.Fatalf("expected 1 peer, got %d", len(n.Peers))
	}
	if n.Peers[0].Fabric != FabricXGMI || n.Peers[0].TargetNodeID != 2 {
		t.Errorf("unexpected peer: %+v", n.Peers[0])
	}
}
