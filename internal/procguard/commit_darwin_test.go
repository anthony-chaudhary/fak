//go:build darwin

package procguard

import (
	"errors"
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
	snapshot, detail := joinDarwinMemorySnapshotWithProbe(100, census, relations, func(int) (bool, error) {
		return true, nil
	})
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

func TestJoinDarwinMemorySnapshotSkipsExitedDescendants(t *testing.T) {
	census := []Proc{
		{PID: 100, Name: "root", WSMB: IntPtr(11)},
		{PID: 103, Name: "live-child", WSMB: IntPtr(7)},
	}
	relations := []Proc{
		{PID: 100, PPID: IntPtr(1), Name: "root"},
		{PID: 101, PPID: IntPtr(100), Name: "exited-child"},
		{PID: 102, PPID: IntPtr(101), Name: "exited-grandchild"},
		{PID: 103, PPID: IntPtr(100), Name: "live-child"},
	}
	probed := make(map[int]int)
	snapshot, detail := joinDarwinMemorySnapshotWithProbe(100, census, relations, func(pid int) (bool, error) {
		probed[pid]++
		return false, nil
	})
	if detail != "" {
		t.Fatalf("normal descendant exit churn must not fail the snapshot: %q", detail)
	}
	if snapshot.TreeBytes != 18<<20 {
		t.Fatalf("tree bytes=%d snapshot=%+v", snapshot.TreeBytes, snapshot)
	}
	if len(snapshot.Processes) != 2 || snapshot.Processes[0].PID != 100 || snapshot.Processes[1].PID != 103 {
		t.Fatalf("exited descendants must be omitted: %+v", snapshot.Processes)
	}
	if probed[101] != 1 || probed[102] != 1 || len(probed) != 2 {
		t.Fatalf("only missing descendant rows should be probed: %+v", probed)
	}
}

func TestJoinDarwinMemorySnapshotKeepsRootAndProbeFailuresFatal(t *testing.T) {
	relations := []Proc{
		{PID: 100, PPID: IntPtr(1), Name: "root"},
		{PID: 101, PPID: IntPtr(100), Name: "child"},
	}
	if _, detail := joinDarwinMemorySnapshotWithProbe(100, nil, relations, func(int) (bool, error) {
		// Descendants may still be reconciled so the partial ownership snapshot
		// remains useful, but an exited root itself must never be suppressed.
		return false, nil
	}); !strings.Contains(detail, "owned pids missing from rss census: [100]") {
		t.Fatalf("missing root detail=%q", detail)
	}

	probeErr := errors.New("liveness unavailable")
	census := []Proc{{PID: 100, Name: "root", WSMB: IntPtr(11)}}
	if _, detail := joinDarwinMemorySnapshotWithProbe(100, census, relations, func(int) (bool, error) {
		return false, probeErr
	}); !strings.Contains(detail, "probe missing rss pid 101: liveness unavailable") {
		t.Fatalf("probe failure detail=%q", detail)
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
