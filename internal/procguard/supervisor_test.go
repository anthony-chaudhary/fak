package procguard

import (
	"context"
	"fmt"
	"os/exec"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/processalive"
)

func TestSupervisor_SubprocessTrackingAndReaping(t *testing.T) {
	supervisor := NewProcessSupervisor(WithTickInterval(20 * time.Millisecond))
	defer func() {
		if err := supervisor.Close(); err != nil {
			t.Errorf("supervisor.Close error: %v", err)
		}
	}()

	// Spawn a parent process that sleeps so it stays alive during test
	var cmd *exec.Cmd
	if runtime.GOOS == "windows" {
		cmd = exec.Command("ping", "127.0.0.1", "-n", "30")
	} else {
		cmd = exec.Command("sleep", "30")
	}
	supervisor.ConfigureCommand(cmd, "session-tree-1")
	if err := cmd.Start(); err != nil {
		t.Fatalf("failed to start command: %v", err)
	}

	pid := cmd.Process.Pid
	supervisor.TrackProcess("session-tree-1", pid)

	// Verify the process is active
	active := supervisor.ActiveProcesses()
	if len(active) == 0 {
		t.Fatalf("expected active processes, got 0")
	}
	found := false
	for _, p := range active {
		if p.PID == pid {
			found = true
			if p.SessionID != "session-tree-1" {
				t.Errorf("expected session session-tree-1, got %s", p.SessionID)
			}
		}
	}
	if !found {
		t.Fatalf("pid %d not found in active processes: %+v", pid, active)
	}

	// Also simulate an explicit child PID in the tree
	childPID := pid + 100000 // synthetic non-existent child PID to test tracking
	supervisor.TrackChildPID("session-tree-1", childPID)

	// Reap the session
	reaped, err := supervisor.ReapSession("session-tree-1")
	if err != nil {
		t.Fatalf("ReapSession returned error: %v", err)
	}

	// Verify root was reaped
	reapedRoot := false
	for _, p := range reaped {
		if p == pid {
			reapedRoot = true
		}
	}
	if !reapedRoot {
		t.Fatalf("root PID %d not in reaped list: %v", pid, reaped)
	}

	// Verify process is no longer alive
	time.Sleep(50 * time.Millisecond)
	if processalive.Check(pid) {
		t.Fatalf("pid %d is still alive after ReapSession", pid)
	}

	// Verify supervisor's active process list is now empty
	if remaining := supervisor.ActiveProcesses(); len(remaining) != 0 {
		t.Fatalf("expected 0 active processes after reap, got %d: %+v", len(remaining), remaining)
	}
}

func TestSupervisor_1000SyntheticToolCommandsZeroLeaks(t *testing.T) {
	// Thread-safe synthetic process table to simulate 1,000 command executions with process trees
	var (
		mu          sync.Mutex
		alivePIDs   = make(map[int]bool)
		descendants = make(map[int][]int)
	)

	mockAlive := func(pid int) bool {
		mu.Lock()
		defer mu.Unlock()
		return alivePIDs[pid]
	}

	mockKill := func(pid int) (bool, string) {
		mu.Lock()
		defer mu.Unlock()
		delete(alivePIDs, pid)
		return true, "mock killed"
	}

	mockDescendant := func(pid int) ([]int, string) {
		mu.Lock()
		defer mu.Unlock()
		if ch, ok := descendants[pid]; ok {
			return append([]int(nil), ch...), ""
		}
		return nil, ""
	}

	supervisor := NewProcessSupervisor(
		WithTickInterval(10*time.Millisecond),
		WithKillFunc(mockKill),
		WithAliveFunc(mockAlive),
		WithDescendantFunc(mockDescendant),
	)
	defer supervisor.Close()

	const totalCommands = 1000
	var wg sync.WaitGroup
	numWorkers := 20
	workChan := make(chan int, totalCommands)

	for i := 0; i < totalCommands; i++ {
		workChan <- i
	}
	close(workChan)

	var completedCount int64

	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range workChan {
				sessionID := fmt.Sprintf("synthetic-tool-session-%d", i)
				rootPID := 10000 + i*10
				child1 := rootPID + 1
				child2 := rootPID + 2

				// Register synthetic processes as alive
				mu.Lock()
				alivePIDs[rootPID] = true
				alivePIDs[child1] = true
				alivePIDs[child2] = true
				descendants[rootPID] = []int{child1, child2}
				mu.Unlock()

				// Track root process and port
				port := 20000 + (i % 2000)
				_ = supervisor.TrackPort(sessionID, port)
				supervisor.TrackProcess(sessionID, rootPID)

				// Stagger some execution paths:
				// - 600: regular completion then reap
				// - 200: timeout with deadline auto-reap
				// - 200: immediate reap
				switch {
				case i%5 == 0:
					// Deadline timeout path
					supervisor.TrackProcessTimeout(sessionID, rootPID, 15*time.Millisecond)
					time.Sleep(25 * time.Millisecond)
					// Let watchdog reap, or ensure reap
					_, _ = supervisor.ReapSession(sessionID)
				case i%5 == 1:
					// Immediate reap
					_, _ = supervisor.ReapSession(sessionID)
				default:
					// Normal simulated work
					time.Sleep(1 * time.Millisecond)
					reaped, err := supervisor.ReapSession(sessionID)
					if err != nil {
						t.Errorf("error reaping session %s: %v", sessionID, err)
					}
					if len(reaped) == 0 {
						t.Errorf("session %s: expected reaped PIDs, got 0", sessionID)
					}
				}

				atomic.AddInt64(&completedCount, 1)
			}
		}()
	}

	wg.Wait()

	if completed := atomic.LoadInt64(&completedCount); completed != totalCommands {
		t.Fatalf("expected %d completed commands, got %d", totalCommands, completed)
	}

	// Allow any watchdog cycles to settle
	time.Sleep(50 * time.Millisecond)

	// Invariant Check 1: Supervisor active process table must be completely drained
	active := supervisor.ActiveProcesses()
	if len(active) != 0 {
		t.Fatalf("leak detected: supervisor still has %d active processes: %+v", len(active), active)
	}

	// Invariant Check 2: All simulated PIDs in mock process table must be killed
	mu.Lock()
	leakCount := len(alivePIDs)
	mu.Unlock()

	if leakCount != 0 {
		t.Fatalf("leak detected: %d living processes remained un-reaped in process table", leakCount)
	}
}

func TestSupervisor_DeadlineTimeoutEnforcement(t *testing.T) {
	var timeoutFired int64
	var timedOutSession string
	var mu sync.Mutex

	supervisor := NewProcessSupervisor(
		WithTickInterval(15*time.Millisecond),
		WithOnTimeout(func(sessionID string, pids []int) {
			atomic.AddInt64(&timeoutFired, 1)
			mu.Lock()
			timedOutSession = sessionID
			mu.Unlock()
		}),
	)
	defer supervisor.Close()

	// Spawn a real long-running sleep command
	var cmd *exec.Cmd
	if runtime.GOOS == "windows" {
		cmd = exec.Command("ping", "127.0.0.1", "-n", "60")
	} else {
		cmd = exec.Command("sleep", "60")
	}
	supervisor.ConfigureCommand(cmd, "deadline-session-1")
	if err := cmd.Start(); err != nil {
		t.Fatalf("failed to start command: %v", err)
	}
	pid := cmd.Process.Pid

	// Track port and process with 60ms deadline
	port := 28765
	if err := supervisor.TrackPort("deadline-session-1", port); err != nil {
		t.Fatalf("TrackPort failed: %v", err)
	}
	supervisor.TrackProcessTimeout("deadline-session-1", pid, 60*time.Millisecond)

	// Verify port is registered
	ports := supervisor.SessionPorts("deadline-session-1")
	if len(ports) != 1 || ports[0] != port {
		t.Fatalf("expected port %d registered, got %v", port, ports)
	}

	// Wait for watchdog to detect deadline expiration and auto-reap
	deadlineWait := 250 * time.Millisecond
	start := time.Now()
	for time.Since(start) < deadlineWait {
		if atomic.LoadInt64(&timeoutFired) > 0 {
			break
		}
		time.Sleep(15 * time.Millisecond)
	}

	if atomic.LoadInt64(&timeoutFired) == 0 {
		t.Fatalf("expected timeout watchdog to fire within %v", deadlineWait)
	}

	mu.Lock()
	sessID := timedOutSession
	mu.Unlock()
	if sessID != "deadline-session-1" {
		t.Errorf("expected timeout for deadline-session-1, got %s", sessID)
	}

	// Verify process was terminated by the auto-reaper
	time.Sleep(30 * time.Millisecond)
	if processalive.Check(pid) {
		t.Fatalf("pid %d is still alive after deadline auto-reap", pid)
	}

	// Verify port was released
	if err := supervisor.TrackPort("another-session", port); err != nil {
		t.Fatalf("failed to re-acquire port %d after reap: %v", port, err)
	}

	// Verify active processes are zero
	if active := supervisor.ActiveProcesses(); len(active) != 0 {
		t.Fatalf("expected 0 active processes after timeout reap, got %d", len(active))
	}
}

func TestSupervisor_PortIsolation(t *testing.T) {
	supervisor := NewProcessSupervisor()
	defer supervisor.Close()

	const port = 19876

	// Session A claims port
	if err := supervisor.TrackPort("session-a", port); err != nil {
		t.Fatalf("session-a failed to claim port: %v", err)
	}

	// Re-claiming by same session is idempotent
	if err := supervisor.TrackPort("session-a", port); err != nil {
		t.Fatalf("session-a re-claim failed: %v", err)
	}

	// Session B trying to claim same port fails with ErrPortConflict
	err := supervisor.TrackPort("session-b", port)
	if err == nil {
		t.Fatalf("expected ErrPortConflict when session-b claims held port, got nil")
	}

	// Invalid ports rejected
	if err := supervisor.TrackPort("session-a", -1); err != ErrInvalidPort {
		t.Errorf("expected ErrInvalidPort for -1, got %v", err)
	}
	if err := supervisor.TrackPort("session-a", 70000); err != ErrInvalidPort {
		t.Errorf("expected ErrInvalidPort for 70000, got %v", err)
	}

	// Release port from session-a
	supervisor.ReleasePort("session-a", port)

	// Session B can now claim it
	if err := supervisor.TrackPort("session-b", port); err != nil {
		t.Fatalf("session-b failed to claim released port: %v", err)
	}

	// Reaping session-b frees the port
	_, _ = supervisor.ReapSession("session-b")
	if ports := supervisor.SessionPorts("session-b"); len(ports) != 0 {
		t.Errorf("expected 0 ports for session-b after reap, got %v", ports)
	}
}

func TestSupervisor_ExecuteCommand(t *testing.T) {
	supervisor := NewProcessSupervisor()
	defer supervisor.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	var cmd *exec.Cmd
	if runtime.GOOS == "windows" {
		cmd = exec.Command("cmd.exe", "/c", "echo", "fak-procguard-governance")
	} else {
		cmd = exec.Command("echo", "fak-procguard-governance")
	}
	if err := supervisor.ExecuteCommand(ctx, "exec-session-1", cmd); err != nil {
		t.Fatalf("ExecuteCommand failed: %v", err)
	}

	// Verify zero processes leaked after command completes
	if active := supervisor.ActiveProcesses(); len(active) != 0 {
		t.Fatalf("expected 0 active processes after ExecuteCommand, got %d: %+v", len(active), active)
	}
}
