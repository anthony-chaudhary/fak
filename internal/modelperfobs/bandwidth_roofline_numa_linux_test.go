//go:build linux

package modelperfobs

import (
	"os"
	"path/filepath"
	"testing"
)

func TestDiscoverNUMATopologyFixture(t *testing.T) {
	root := t.TempDir()
	nodeRoot := filepath.Join(root, "node")
	if err := os.MkdirAll(filepath.Join(nodeRoot, "node0"), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(nodeRoot, "node2"), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(nodeRoot, "node7"), 0755); err != nil {
		t.Fatal(err)
	}
	write := func(p, s string) {
		t.Helper()
		if err := os.WriteFile(p, []byte(s), 0644); err != nil {
			t.Fatal(err)
		}
	}
	write(filepath.Join(nodeRoot, "online"), "0,2,7\n")
	write(filepath.Join(nodeRoot, "node0", "cpulist"), "0-1\n")
	write(filepath.Join(nodeRoot, "node0", "meminfo"), "Node 0 MemTotal: 1024 kB\n")
	write(filepath.Join(nodeRoot, "node2", "cpulist"), "8-9\n")
	write(filepath.Join(nodeRoot, "node2", "meminfo"), "Node 2 MemTotal: 2048 kB\n")
	write(filepath.Join(nodeRoot, "node7", "cpulist"), "\n")
	write(filepath.Join(nodeRoot, "node7", "meminfo"), "Node 7 MemTotal: 0 kB\n")
	status := filepath.Join(root, "status")
	write(status, "Name:\ttest\nCpus_allowed_list:\t0,8\n")
	topo, err := discoverNUMATopology(nodeRoot, status)
	if err != nil {
		t.Fatal(err)
	}
	if !topo.CPUSetRestricted || len(topo.Nodes) != 3 || topo.Nodes[0].AllowedCPUIds[0] != 0 || topo.Nodes[1].AllowedCPUIds[0] != 8 || topo.Nodes[2].OmissionReason != "memoryless" {
		t.Fatalf("topology=%+v", topo)
	}
}
