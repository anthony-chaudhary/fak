//go:build windows

package main

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/procguard"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

func TestGuardChildResourceMonitorReapsOwnedTreeAndReceipts(t *testing.T) {
	if mode := os.Getenv("FAK_RESOURCE_WITNESS_HELPER"); mode != "" {
		runGuardResourceWitnessHelper(t, mode)
		return
	}

	dir := t.TempDir()
	pidPath := filepath.Join(dir, "descendant.pid")
	grandchildPIDPath := filepath.Join(dir, "grandchild.pid")
	journal := filepath.Join(dir, "child-resource.jsonl")
	cmd := exec.Command(os.Args[0], "-test.run=^TestGuardChildResourceMonitorReapsOwnedTreeAndReceipts$", "codex-resource-witness")
	cmd.Env = append(os.Environ(), "FAK_RESOURCE_WITNESS_HELPER=root", "FAK_RESOURCE_WITNESS_PID_PATH="+pidPath, "FAK_RESOURCE_WITNESS_GRANDCHILD_PID_PATH="+grandchildPIDPath)
	job, err := windowgate.StartInNewJob(cmd)
	if err != nil {
		t.Fatalf("start contained witness tree: %v", err)
	}
	defer job.Close()
	wait := make(chan error, 1)
	go func() { wait <- cmd.Wait() }()

	descendantPID := waitForWitnessPID(t, pidPath, 5*time.Second)
	grandchildPID := waitForWitnessPID(t, grandchildPIDPath, 5*time.Second)
	stop := make(chan struct{})
	started := time.Now()
	resource := startGuardChildResourceMonitor(cmd.Process.Pid, "trace-witness", "codex", guardResourcePolicy{
		PollInterval:      100 * time.Millisecond,
		Metric:            procguard.MemoryMetricCommit,
		MaxTreeBytes:      96 << 20,
		MinSystemHeadroom: 1,
		Stop:              stop,
	})
	ev := <-resource
	close(stop)
	if ev.Kind != guardChildResourceLimit || ev.Resource == nil || !ev.Resource.Stop {
		t.Fatalf("resource event=%+v", ev)
	}
	if elapsed := time.Since(started); elapsed > 3*time.Second {
		t.Fatalf("intervention took %v, want <=3s", elapsed)
	}

	stopGuardChild(cmd, wait, 0)
	oldConfig := guardResourceConfigured
	setGuardResourceConfig(guardResourceConfig{ReceiptPath: journal})
	t.Cleanup(func() { setGuardResourceConfig(oldConfig) })
	guardWriteResourceReceipt(ev, "trace-witness", "codex", cmd.Process.Pid)

	assertWitnessPIDsGone(t, 3*time.Second, cmd.Process.Pid, descendantPID, grandchildPID)
	data, err := os.ReadFile(journal)
	if err != nil {
		t.Fatalf("read resource receipt: %v", err)
	}
	var receipt guardResourceReceipt
	if err := json.Unmarshal(data, &receipt); err != nil {
		t.Fatalf("decode resource receipt: %v\n%s", err, data)
	}
	if receipt.RootPID != cmd.Process.Pid || receipt.OffenderPID == 0 || receipt.ThresholdBytes != 96<<20 || receipt.TreeCommitBytes == nil || *receipt.TreeCommitBytes <= receipt.ThresholdBytes {
		t.Fatalf("receipt lacks identity/threshold/usage evidence: %+v", receipt)
	}
	if receipt.MemoryMetric != string(procguard.MemoryMetricCommit) || receipt.TreeRSSBytes != nil {
		t.Fatalf("Windows receipt changed metric schema: %+v", receipt)
	}
	if receipt.Action != "reap_tree" || receipt.DescendantsSurvive {
		t.Fatalf("receipt lacks successful reap readback: %+v", receipt)
	}
}

func runGuardResourceWitnessHelper(t *testing.T, mode string) {
	const allocation = 64 << 20
	memory := make([]byte, allocation)
	for i := 0; i < len(memory); i += 4096 {
		memory[i] = byte(i)
	}
	if mode == "root" {
		child := exec.Command(os.Args[0], "-test.run=^TestGuardChildResourceMonitorReapsOwnedTreeAndReceipts$", "codex-resource-descendant")
		child.Env = append(os.Environ(), "FAK_RESOURCE_WITNESS_HELPER=leaf", "FAK_RESOURCE_WITNESS_GRANDCHILD_PID_PATH="+os.Getenv("FAK_RESOURCE_WITNESS_GRANDCHILD_PID_PATH"))
		if err := child.Start(); err != nil {
			t.Fatalf("start descendant: %v", err)
		}
		if err := os.WriteFile(os.Getenv("FAK_RESOURCE_WITNESS_PID_PATH"), []byte(strconv.Itoa(child.Process.Pid)), 0o600); err != nil {
			t.Fatalf("write descendant pid: %v", err)
		}
	} else if mode == "leaf" {
		grandchild := exec.Command(os.Args[0], "-test.run=^TestGuardChildResourceMonitorReapsOwnedTreeAndReceipts$", "codex-resource-grandchild")
		grandchild.Env = append(os.Environ(), "FAK_RESOURCE_WITNESS_HELPER=grandchild")
		if err := grandchild.Start(); err != nil {
			t.Fatalf("start grandchild: %v", err)
		}
		if err := os.WriteFile(os.Getenv("FAK_RESOURCE_WITNESS_GRANDCHILD_PID_PATH"), []byte(strconv.Itoa(grandchild.Process.Pid)), 0o600); err != nil {
			t.Fatalf("write grandchild pid: %v", err)
		}
	}
	for {
		time.Sleep(time.Second)
		memory[0]++
	}
}

func waitForWitnessPID(t *testing.T, path string, timeout time.Duration) int {
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
	t.Fatalf("descendant pid was not published within %v", timeout)
	return 0
}

func assertWitnessPIDsGone(t *testing.T, timeout time.Duration, pids ...int) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for {
		rows, detail := procguard.CollectRelations()
		if detail != "" {
			t.Fatalf("process readback: %s", detail)
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
			t.Fatalf("owned process IDs still alive after %v: %v", timeout, pids)
		}
		time.Sleep(25 * time.Millisecond)
	}
}
