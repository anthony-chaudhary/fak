//go:build linux

package compute

import (
	"os"
	"path/filepath"
	"strconv"
	"testing"
)

// writeFakeNode materializes one node<N> directory in a synthetic sysfs tree: its
// cpulist ("" for a memory-only node) and a per-node meminfo in the kernel's
// "Node <N> <Field>: <kB> kB" shape.
func writeFakeNode(t *testing.T, root string, id int, cpulist string, totalKB, freeKB int64) {
	t.Helper()
	dir := filepath.Join(root, "node"+itoa(id))
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	meminfo := "Node " + itoa(id) + " MemTotal:       " + itoa64(totalKB) + " kB\n" +
		"Node " + itoa(id) + " MemFree:        " + itoa64(freeKB) + " kB\n" +
		"Node " + itoa(id) + " MemUsed:        " + itoa64(totalKB-freeKB) + " kB\n"
	if err := os.WriteFile(filepath.Join(dir, "meminfo"), []byte(meminfo), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "cpulist"), []byte(cpulist+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
}

func itoa(v int) string { return strconv.Itoa(v) }

func itoa64(v int64) string { return strconv.FormatInt(v, 10) }

// TestFarMemoryFromNodesTwoSocketsPlusCXL is the classifier's core case: a two-socket
// box (node0 local, node1 far) with one memory-only expansion node (node2). node1
// feeds NUMA-far, node2 feeds CXL, and node0 (local) contributes to neither.
func TestFarMemoryFromNodesTwoSocketsPlusCXL(t *testing.T) {
	root := t.TempDir()
	writeFakeNode(t, root, 0, "0-31", 512<<20, 100<<20)  // local socket (kB units)
	writeFakeNode(t, root, 1, "32-63", 512<<20, 300<<20) // far socket
	writeFakeNode(t, root, 2, "", 1<<30, 1<<29)          // memory-only expansion node

	nodes := readNodeMemory(root)
	if len(nodes) != 3 {
		t.Fatalf("want 3 parsed nodes, got %d: %+v", len(nodes), nodes)
	}
	numaTotal, numaFree, numaOK, cxlTotal, cxlFree, cxlOK := farMemoryFromNodes(nodes)
	if !numaOK || !cxlOK {
		t.Fatalf("both far classes should be confirmed, got numa=%v cxl=%v", numaOK, cxlOK)
	}
	if numaTotal != (512<<20)*1024 || numaFree != (300<<20)*1024 {
		t.Fatalf("NUMA-far must be exactly the far socket: total=%d free=%d", numaTotal, numaFree)
	}
	if cxlTotal != (1<<30)*1024 || cxlFree != (1<<29)*1024 {
		t.Fatalf("CXL must be exactly the memory-only node: total=%d free=%d", cxlTotal, cxlFree)
	}
}

// TestFarMemoryFromNodesSingleSocketFailsOpen: one CPU-ful node and no memory-only
// node confirms NEITHER far class — a plain single-socket box is byte-identical to
// today (the #1470 fence: no fabricated far tier).
func TestFarMemoryFromNodesSingleSocketFailsOpen(t *testing.T) {
	root := t.TempDir()
	writeFakeNode(t, root, 0, "0-15", 256<<20, 64<<20)

	_, numaFree, numaOK, _, cxlFree, cxlOK := farMemoryFromNodes(readNodeMemory(root))
	if numaOK || cxlOK {
		t.Fatalf("single-socket box must confirm no far memory, got numa=%v cxl=%v", numaOK, cxlOK)
	}
	if numaFree != FreeUnknown || cxlFree != FreeUnknown {
		t.Fatalf("unconfirmed classes must report FreeUnknown, got numa=%d cxl=%d", numaFree, cxlFree)
	}
}

// TestFarMemoryFromNodesUnreadableRootFailsOpen: a missing topology directory parses
// to no nodes and both classes unknown.
func TestFarMemoryFromNodesUnreadableRootFailsOpen(t *testing.T) {
	nodes := readNodeMemory(filepath.Join(t.TempDir(), "does-not-exist"))
	if nodes != nil {
		t.Fatalf("missing root must parse to no nodes, got %+v", nodes)
	}
	_, _, numaOK, _, _, cxlOK := farMemoryFromNodes(nodes)
	if numaOK || cxlOK {
		t.Fatalf("missing topology must confirm nothing, got numa=%v cxl=%v", numaOK, cxlOK)
	}
}

// TestParseNodeMeminfoRejectsPartial: a meminfo missing MemFree (or MemTotal) is not
// half-trusted — the node is skipped rather than probed with a fabricated field.
func TestParseNodeMeminfoRejectsPartial(t *testing.T) {
	path := filepath.Join(t.TempDir(), "meminfo")
	if err := os.WriteFile(path, []byte("Node 0 MemTotal: 1000 kB\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, _, ok := parseNodeMeminfo(path); ok {
		t.Fatal("meminfo without MemFree must not parse ok")
	}
}

// TestNUMAFarMemoryLive smokes the real sysfs on the test host: whatever it reports
// must satisfy the public contract (checked in TestFarMemoryInfoContract); here we
// only assert it does not panic and stays self-consistent with a re-read.
func TestNUMAFarMemoryLive(t *testing.T) {
	if _, err := os.Stat(sysNodeRoot); err != nil {
		t.Skipf("no NUMA topology on this host: %v", err)
	}
	t1, _, k1 := NUMAFarMemoryInfo()
	t2, _, k2 := NUMAFarMemoryInfo()
	if k1 != k2 || t1 != t2 {
		t.Fatalf("far-memory probe not stable across reads: (%d,%v) vs (%d,%v)", t1, k1, t2, k2)
	}
}
