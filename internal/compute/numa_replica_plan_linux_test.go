//go:build linux

package compute

import (
	"os"
	"path/filepath"
	"strconv"
	"testing"
)

func TestReadNUMAReplicaNodesExactCPUNodeTopology(t *testing.T) {
	root := t.TempDir()
	writeNUMAReplicaTestNode(t, root, 10, "20-23", 200, true)
	writeNUMAReplicaTestNode(t, root, 2, "4-7", 120, true)
	writeNUMAReplicaTestNode(t, root, 4, "", 1000, true) // memory-only: not a target

	nodes, known := readNUMAReplicaNodes(root)
	if !known {
		t.Fatal("synthetic Linux topology unexpectedly unsupported")
	}
	if len(nodes) != 2 {
		t.Fatalf("CPU-bearing target count = %d, want 2: %+v", len(nodes), nodes)
	}
	if nodes[0].id != 2 || nodes[0].freeBytes != 120*1024 || !nodes[0].memoryKnown {
		t.Fatalf("node[0] = %+v, want node 2 with exact known free memory", nodes[0])
	}
	if nodes[1].id != 10 || nodes[1].freeBytes != 200*1024 || !nodes[1].memoryKnown {
		t.Fatalf("node[1] = %+v, want node 10 with exact known free memory", nodes[1])
	}
}

func TestReadNUMAReplicaNodesPreservesUnknownTargetMemory(t *testing.T) {
	root := t.TempDir()
	writeNUMAReplicaTestNode(t, root, 0, "0-3", 0, false)
	writeNUMAReplicaTestNode(t, root, 1, "4-7", 100, true)

	nodes, known := readNUMAReplicaNodes(root)
	if !known || len(nodes) != 2 {
		t.Fatalf("topology = known:%v nodes:%+v, want two visible targets", known, nodes)
	}
	if nodes[0].memoryKnown {
		t.Fatalf("unreadable target memory was silently treated as known: %+v", nodes[0])
	}
	got := planNUMAReplicas(numaReplicaSnapshot{
		policy:        numaReplicaPolicyUnconstrained,
		topologyKnown: known,
		nodes:         nodes,
	}, 1, 0)
	assertNUMAReplicaRefusal(t, got, NUMAReplicaPlanUnknownNodeMemory)
}

func TestReadNUMAReplicaNodesFailsClosedWithoutCPUIdentity(t *testing.T) {
	root := t.TempDir()
	dir := filepath.Join(root, "node0")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if nodes, known := readNUMAReplicaNodes(root); known || nodes != nil {
		t.Fatalf("missing cpulist = known:%v nodes:%+v, want unsupported topology", known, nodes)
	}
	if nodes, known := readNUMAReplicaNodes(filepath.Join(root, "missing")); known || nodes != nil {
		t.Fatalf("missing sysfs root = known:%v nodes:%+v, want unsupported topology", known, nodes)
	}
}

func writeNUMAReplicaTestNode(t *testing.T, root string, id int, cpulist string, freeKB int64, withMemory bool) {
	t.Helper()
	dir := filepath.Join(root, "node"+strconv.Itoa(id))
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "cpulist"), []byte(cpulist+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if !withMemory {
		return
	}
	meminfo := "Node " + strconv.Itoa(id) + " MemTotal: 4096 kB\n" +
		"Node " + strconv.Itoa(id) + " MemFree: " + strconv.FormatInt(freeKB, 10) + " kB\n"
	if err := os.WriteFile(filepath.Join(dir, "meminfo"), []byte(meminfo), 0o644); err != nil {
		t.Fatal(err)
	}
}
