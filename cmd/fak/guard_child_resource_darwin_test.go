//go:build darwin

package main

import (
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/procguard"
)

func TestGuardChildResourceMonitorDarwinReapsOwnedTreeAndReceipts(t *testing.T) {
	if mode := os.Getenv("FAK_DARWIN_RESOURCE_WITNESS_HELPER"); mode != "" {
		runDarwinGuardResourceWitnessHelper(t, mode)
		return
	}

	dir := t.TempDir()
	childPIDPath := filepath.Join(dir, "child.pid")
	grandchildPIDPath := filepath.Join(dir, "grandchild.pid")
	journal := filepath.Join(dir, "child-resource.jsonl")
	cmd := exec.CommandContext(context.Background(), os.Args[0], "-test.run=^TestGuardChildResourceMonitorDarwinReapsOwnedTreeAndReceipts$")
	cmd.Env = append(os.Environ(),
		"FAK_DARWIN_RESOURCE_WITNESS_HELPER=root",
		"FAK_DARWIN_RESOURCE_WITNESS_CHILD_PID_PATH="+childPIDPath,
		"FAK_DARWIN_RESOURCE_WITNESS_GRANDCHILD_PID_PATH="+grandchildPIDPath,
	)
	procguard.ConfigureProcessTreeCancel(cmd)
	if err := cmd.Start(); err != nil {
		t.Fatalf("start contained Darwin witness tree: %v", err)
	}
	wait := make(chan error, 1)
	go func() { wait <- cmd.Wait() }()
	stopped := false
	t.Cleanup(func() {
		if !stopped {
			_, _ = procguard.KillPID(cmd.Process.Pid)
			select {
			case <-wait:
			case <-time.After(5 * time.Second):
			}
		}
	})

	childPID := waitForDarwinWitnessPID(t, childPIDPath, 5*time.Second)
	grandchildPID := waitForDarwinWitnessPID(t, grandchildPIDPath, 5*time.Second)
	const threshold = uint64(48) << 20
	before, supported, detail := procguard.CollectMemorySnapshot(cmd.Process.Pid)
	if !supported || detail != "" {
		t.Fatalf("read fak-owned witness tree: supported=%v detail=%q snapshot=%+v", supported, detail, before)
	}
	if before.Metric != procguard.MemoryMetricRSS || before.TreeBytes <= threshold {
		t.Fatalf("bounded witness did not cross RSS threshold: %+v", before)
	}
	wantOwned := map[int]bool{cmd.Process.Pid: false, childPID: false, grandchildPID: false}
	for _, process := range before.Processes {
		if _, ok := wantOwned[process.PID]; ok {
			wantOwned[process.PID] = true
		}
	}
	for pid, found := range wantOwned {
		if !found {
			t.Fatalf("fak-owned PID %d missing from pre-intervention snapshot: %+v", pid, before.Processes)
		}
	}

	stop := make(chan struct{})
	started := time.Now()
	resource := startGuardChildResourceMonitor(cmd.Process.Pid, "trace-darwin-witness", "codex", guardResourcePolicy{
		PollInterval: 100 * time.Millisecond,
		Metric:       procguard.MemoryMetricRSS,
		MaxTreeBytes: threshold,
		Stop:         stop,
	})
	ev := <-resource
	close(stop)
	if ev.Kind != guardChildResourceLimit || ev.Resource == nil || !ev.Resource.Stop || ev.Resource.Reason != "CHILD_TREE_RSS_LIMIT" {
		t.Fatalf("resource event=%+v", ev)
	}
	if elapsed := time.Since(started); elapsed > 3*time.Second {
		t.Fatalf("intervention took %v, want <=3s", elapsed)
	}

	_ = stopGuardChild(cmd, wait, 0)
	stopped = true
	t.Setenv("FAK_CHILD_RESOURCE_JOURNAL", journal)
	if err := guardWriteResourceReceipt(ev, "trace-darwin-witness", "codex", cmd.Process.Pid); err != nil {
		t.Fatalf("write Darwin resource receipt: %v", err)
	}
	assertDarwinWitnessPIDsGone(t, 3*time.Second, cmd.Process.Pid, childPID, grandchildPID)

	data, err := os.ReadFile(journal)
	if err != nil {
		t.Fatalf("read Darwin resource receipt: %v", err)
	}
	var receipt guardResourceReceipt
	if err := json.Unmarshal(data, &receipt); err != nil {
		t.Fatalf("decode Darwin resource receipt: %v\n%s", err, data)
	}
	if receipt.RootPID != cmd.Process.Pid || receipt.OffenderPID == 0 || receipt.ThresholdBytes != threshold || receipt.TreeRSSBytes == nil || *receipt.TreeRSSBytes <= receipt.ThresholdBytes {
		t.Fatalf("receipt lacks identity/threshold/RSS evidence: %+v", receipt)
	}
	if receipt.MemoryMetric != string(procguard.MemoryMetricRSS) || receipt.TreeCommitBytes != nil {
		t.Fatalf("Darwin RSS was mislabeled as commit: %+v", receipt)
	}
	if receipt.Action != "reap_tree" || receipt.DescendantsSurvive {
		t.Fatalf("receipt lacks successful reap readback: %+v", receipt)
	}
}

func TestGuardChildResourceMonitorDarwinSurfacesCollectorError(t *testing.T) {
	stop := make(chan struct{})
	resource := startGuardChildResourceMonitor(-1, "trace-darwin-error", "codex", guardResourcePolicy{
		PollInterval: 100 * time.Millisecond,
		Metric:       procguard.MemoryMetricRSS,
		MaxTreeBytes: 1 << 30,
		Stop:         stop,
	})
	defer close(stop)
	select {
	case event := <-resource:
		if event.Resource == nil || event.Resource.Reason != "CHILD_RESOURCE_MONITOR_ERROR" || event.Resource.Metric != procguard.MemoryMetricRSS {
			t.Fatalf("collector error was not typed: %+v", event)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("Darwin collector error was silent")
	}
}

func runDarwinGuardResourceWitnessHelper(t *testing.T, mode string) {
	const allocation = 16 << 20
	memory := make([]byte, allocation)
	for i := 0; i < len(memory); i += 4096 {
		memory[i] = byte(i)
	}
	if mode == "root" {
		child := exec.Command(os.Args[0], "-test.run=^TestGuardChildResourceMonitorDarwinReapsOwnedTreeAndReceipts$")
		child.Env = append(os.Environ(),
			"FAK_DARWIN_RESOURCE_WITNESS_HELPER=child",
			"FAK_DARWIN_RESOURCE_WITNESS_GRANDCHILD_PID_PATH="+os.Getenv("FAK_DARWIN_RESOURCE_WITNESS_GRANDCHILD_PID_PATH"),
		)
		if err := child.Start(); err != nil {
			t.Fatalf("start Darwin child: %v", err)
		}
		if err := os.WriteFile(os.Getenv("FAK_DARWIN_RESOURCE_WITNESS_CHILD_PID_PATH"), []byte(strconv.Itoa(child.Process.Pid)), 0o600); err != nil {
			t.Fatalf("write Darwin child pid: %v", err)
		}
	} else if mode == "child" {
		grandchild := exec.Command(os.Args[0], "-test.run=^TestGuardChildResourceMonitorDarwinReapsOwnedTreeAndReceipts$")
		grandchild.Env = append(os.Environ(), "FAK_DARWIN_RESOURCE_WITNESS_HELPER=grandchild")
		if err := grandchild.Start(); err != nil {
			t.Fatalf("start Darwin grandchild: %v", err)
		}
		if err := os.WriteFile(os.Getenv("FAK_DARWIN_RESOURCE_WITNESS_GRANDCHILD_PID_PATH"), []byte(strconv.Itoa(grandchild.Process.Pid)), 0o600); err != nil {
			t.Fatalf("write Darwin grandchild pid: %v", err)
		}
	}
	for {
		time.Sleep(time.Second)
		memory[0]++
	}
}

func waitForDarwinWitnessPID(t *testing.T, path string, timeout time.Duration) int {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if data, err := os.ReadFile(path); err == nil {
			pid, err := strconv.Atoi(string(data))
			if err == nil && pid > 0 {
				return pid
			}
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatalf("Darwin witness pid was not published within %v", timeout)
	return 0
}

func assertDarwinWitnessPIDsGone(t *testing.T, timeout time.Duration, pids ...int) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for {
		rows, detail := procguard.CollectRelations()
		if detail != "" {
			t.Fatalf("Darwin process readback: %s", detail)
		}
		alive := map[int]bool{}
		for _, row := range rows {
			alive[row.PID] = true
		}
		any := false
		for _, pid := range pids {
			any = any || alive[pid]
		}
		if !any {
			return
		}
		if time.Now().After(deadline) {
			t.Fatalf("owned Darwin process IDs still alive after %v: %v", timeout, pids)
		}
		time.Sleep(25 * time.Millisecond)
	}
}
