//go:build darwin

package procguard

import (
	"os/exec"
	"strings"
	"testing"
)

func TestJoinDarwinMemorySnapshotIncludesRootAndCycleSafeDescendants(t *testing.T) {
	census := []Proc{
		{PID: 100, Name: "root", WSMB: IntPtr(11)},
		{PID: 101, Name: "child", WSMB: IntPtr(7)},
		{PID: 102, Name: "grandchild", WSMB: IntPtr(5)},
		{PID: 999, Name: "unowned", WSMB: IntPtr(1000)},
	}
	relations := []Proc{
		{PID: 100, PPID: IntPtr(102), Name: "root", Cmdline: "root --owned"},
		{PID: 101, PPID: IntPtr(100), Name: "child", Cmdline: "child --owned"},
		{PID: 102, PPID: IntPtr(101), Name: "grandchild", Cmdline: "grandchild --owned"},
		{PID: 999, PPID: IntPtr(1), Name: "unowned", Cmdline: "unowned"},
	}

	snapshot, detail := joinDarwinMemorySnapshot(100, census, relations)
	if detail != "" {
		t.Fatalf("detail=%q", detail)
	}
	if snapshot.Metric != MemoryMetricRSS || snapshot.TreeBytes != 23<<20 {
		t.Fatalf("snapshot=%+v", snapshot)
	}
	if len(snapshot.Processes) != 3 || snapshot.Processes[0].PID != 100 || snapshot.Processes[1].PID != 101 || snapshot.Processes[2].PID != 102 {
		t.Fatalf("owned traversal=%+v", snapshot.Processes)
	}
}

func TestJoinDarwinMemorySnapshotRejectsIncompleteRows(t *testing.T) {
	census := []Proc{{PID: 100, Name: "root", WSMB: IntPtr(11)}}
	relations := []Proc{
		{PID: 100, PPID: IntPtr(1), Name: "root"},
		{PID: 101, PPID: IntPtr(100), Name: "child"},
	}
	snapshot, detail := joinDarwinMemorySnapshot(100, census, relations)
	if !strings.Contains(detail, "owned pids missing from rss census: [101]") {
		t.Fatalf("detail=%q snapshot=%+v", detail, snapshot)
	}
	if len(snapshot.Processes) != 2 {
		t.Fatalf("partial ownership must remain available for fail-closed reap: %+v", snapshot.Processes)
	}
	if _, detail := joinDarwinMemorySnapshot(100, census, relations[1:]); !strings.Contains(detail, "root pid 100 missing") {
		t.Fatalf("missing root relation detail=%q", detail)
	}
}

func TestCollectMemorySnapshotOwnProcessUsesRSS(t *testing.T) {
	child := exec.Command("sleep", "5")
	if err := child.Start(); err != nil {
		t.Fatalf("start snapshot target: %v", err)
	}
	t.Cleanup(func() {
		_ = child.Process.Kill()
		_, _ = child.Process.Wait()
	})
	snapshot, supported, detail := CollectMemorySnapshot(child.Process.Pid)
	if !supported || detail != "" {
		t.Fatalf("supported=%v detail=%q snapshot=%+v", supported, detail, snapshot)
	}
	if snapshot.Metric != MemoryMetricRSS || snapshot.TreeBytes == 0 || snapshot.HostPhysicalBytes == 0 {
		t.Fatalf("Darwin snapshot lacks rss/host bytes: %+v", snapshot)
	}
	found := false
	for _, process := range snapshot.Processes {
		found = found || process.PID == child.Process.Pid
	}
	if !found {
		t.Fatalf("root pid %d missing from %+v", child.Process.Pid, snapshot.Processes)
	}
	if _, supported, _ := CollectCommitSnapshot(child.Process.Pid); supported {
		t.Fatal("Darwin RSS must not be exposed through the legacy commit API")
	}
}
