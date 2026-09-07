package issueorchestrator

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestAdaptiveSupervisor_ActiveFileWritesExtendDeadline(t *testing.T) {
	tmpDir := t.TempDir()
	simTime := time.Date(2026, 9, 6, 12, 0, 0, 0, time.UTC)
	nowFunc := func() time.Time { return simTime }

	cfg := WorkerSupervisorConfig{
		PID:             1001,
		WorktreeDir:     tmpDir,
		InitialDeadline: 15 * time.Minute,
		ExtendIncrement: 10 * time.Minute,
		MaxDeadline:     60 * time.Minute,
		LivenessProbe:   func(pid int) bool { return true },
		NowFunc:         nowFunc,
	}

	sup := NewWorkerSupervisor(cfg)

	// Baseline deadline is startTime + 15m
	initialDeadline := sup.CurrentDeadline()
	expectedInitial := simTime.Add(15 * time.Minute)
	if !initialDeadline.Equal(expectedInitial) {
		t.Fatalf("expected initial deadline %v, got %v", expectedInitial, initialDeadline)
	}

	// Advance time to 16m (past the initial 15m deadline)
	simTime = simTime.Add(16 * time.Minute)

	// Active file write in worktree demonstrates ongoing progress
	err := os.WriteFile(filepath.Join(tmpDir, "progress.txt"), []byte("active implementation work"), 0o644)
	if err != nil {
		t.Fatalf("write file: %v", err)
	}

	verdict := sup.EvaluateStatus()
	if verdict != SUPERVISOR_EXTENDED {
		t.Fatalf("expected verdict SUPERVISOR_EXTENDED, got %v", verdict)
	}

	// Verify that deadline has extended past initial 15m to 25m
	extendedDeadline := sup.CurrentDeadline()
	if !extendedDeadline.After(initialDeadline) {
		t.Fatalf("expected extended deadline > initial (%v), got %v", initialDeadline, extendedDeadline)
	}
	expectedExtended := simTime.Add(-16 * time.Minute).Add(25 * time.Minute)
	if !extendedDeadline.Equal(expectedExtended) {
		t.Fatalf("expected extended deadline %v, got %v", expectedExtended, extendedDeadline)
	}
	if sup.Reaped() {
		t.Fatalf("expected worker to remain active and un-reaped")
	}
}

func TestAdaptiveSupervisor_SilenceSoftWarning(t *testing.T) {
	tmpDir := t.TempDir()
	simTime := time.Date(2026, 9, 6, 12, 0, 0, 0, time.UTC)
	nowFunc := func() time.Time { return simTime }

	cfg := WorkerSupervisorConfig{
		PID:             1002,
		WorktreeDir:     tmpDir,
		InitialDeadline: 15 * time.Minute,
		SilenceWarn:     10 * time.Minute,
		SilenceReap:     25 * time.Minute,
		LivenessProbe:   func(pid int) bool { return true },
		NowFunc:         nowFunc,
	}

	sup := NewWorkerSupervisor(cfg)

	// Advance time by 11m without any file/log activity (silent)
	simTime = simTime.Add(11 * time.Minute)

	verdict := sup.EvaluateStatus()
	if verdict != SUPERVISOR_WARN_SILENCE {
		t.Fatalf("expected verdict SUPERVISOR_WARN_SILENCE, got %v", verdict)
	}

	if sup.Reaped() {
		t.Fatalf("process should NOT be reaped on soft warning")
	}

	warnings := sup.Warnings()
	if len(warnings) == 0 {
		t.Fatalf("expected soft warning to be recorded")
	}
	expectedWarn := fmt.Sprintf("WARN [supervisor]: worker PID 1002 silent for 10m; monitoring activity before reaping")
	if !strings.Contains(warnings[0], expectedWarn) {
		t.Fatalf("expected warning %q, got %q", expectedWarn, warnings[0])
	}
}

func TestAdaptiveSupervisor_DeadProcessReaped(t *testing.T) {
	simTime := time.Date(2026, 9, 6, 12, 0, 0, 0, time.UTC)
	nowFunc := func() time.Time { return simTime }

	reapedPID := 0
	reaper := func(pid int) error {
		reapedPID = pid
		return nil
	}

	cfg := WorkerSupervisorConfig{
		PID:           1003,
		LivenessProbe: func(pid int) bool { return false }, // PID confirmed dead
		NowFunc:       nowFunc,
		ReaperFunc:    reaper,
	}

	sup := NewWorkerSupervisor(cfg)
	verdict := sup.EvaluateStatus()

	if verdict != SUPERVISOR_REAP_DEAD {
		t.Fatalf("expected verdict SUPERVISOR_REAP_DEAD, got %v", verdict)
	}
	if reapedPID != 1003 {
		t.Fatalf("expected reaper to be called for PID 1003, got %d", reapedPID)
	}
	if !sup.Reaped() {
		t.Fatalf("expected supervisor to mark process as reaped")
	}
}

func TestAdaptiveSupervisor_UnresponsivePastGraceReaped(t *testing.T) {
	tmpDir := t.TempDir()
	simTime := time.Date(2026, 9, 6, 12, 0, 0, 0, time.UTC)
	nowFunc := func() time.Time { return simTime }

	reapedPID := 0
	reaper := func(pid int) error {
		reapedPID = pid
		return nil
	}

	cfg := WorkerSupervisorConfig{
		PID:           1004,
		WorktreeDir:   tmpDir,
		SilenceWarn:   10 * time.Minute,
		SilenceReap:   25 * time.Minute,
		LivenessProbe: func(pid int) bool { return true }, // alive
		NowFunc:       nowFunc,
		ReaperFunc:    reaper,
	}

	sup := NewWorkerSupervisor(cfg)

	// Advance past 25m silence grace limit (26m) with zero child activity
	simTime = simTime.Add(26 * time.Minute)

	verdict := sup.EvaluateStatus()
	if verdict != SUPERVISOR_REAP_WEDGED {
		t.Fatalf("expected verdict SUPERVISOR_REAP_WEDGED, got %v", verdict)
	}
	if reapedPID != 1004 {
		t.Fatalf("expected reaper to be called for PID 1004, got %d", reapedPID)
	}
	if !sup.Reaped() {
		t.Fatalf("expected supervisor to mark process as reaped")
	}
}

func TestAdaptiveSupervisor_MultiChannelHeartbeats(t *testing.T) {
	tmpDir := t.TempDir()
	logFile := filepath.Join(tmpDir, "worker.log")
	simTime := time.Date(2026, 9, 6, 12, 0, 0, 0, time.UTC)
	nowFunc := func() time.Time { return simTime }

	childActive := false
	childProbe := func(pid int) bool { return childActive }

	cfg := WorkerSupervisorConfig{
		PID:             1005,
		WorktreeDir:     tmpDir,
		LogFile:         logFile,
		InitialDeadline: 15 * time.Minute,
		ExtendIncrement: 10 * time.Minute,
		MaxDeadline:     60 * time.Minute,
		LivenessProbe:   func(pid int) bool { return true },
		ChildProbe:      childProbe,
		NowFunc:         nowFunc,
	}

	sup := NewWorkerSupervisor(cfg)

	// Channel 1: Log appends
	_ = os.WriteFile(logFile, []byte("line 1\n"), 0o644)
	hb, err := sup.SampleHeartbeat()
	if err != nil {
		t.Fatalf("SampleHeartbeat error: %v", err)
	}
	if !hb.LogAppended || !hb.AnyProgress {
		t.Fatalf("expected LogAppended and AnyProgress to be true")
	}

	// Channel 2: Lease update
	leaseFile := filepath.Join(tmpDir, "lease.json")
	_ = os.WriteFile(leaseFile, []byte(`{"pid":1005,"ts":"2026-09-06T12:00:00Z"}`), 0o644)
	hb, err = sup.SampleHeartbeat()
	if err != nil {
		t.Fatalf("SampleHeartbeat error: %v", err)
	}
	if !hb.LeaseUpdated || !hb.AnyProgress {
		t.Fatalf("expected LeaseUpdated and AnyProgress to be true")
	}

	// Channel 3: Child compiler/test activity
	childActive = true
	hb, err = sup.SampleHeartbeat()
	if err != nil {
		t.Fatalf("SampleHeartbeat error: %v", err)
	}
	if !hb.ChildActive || !hb.AnyProgress {
		t.Fatalf("expected ChildActive and AnyProgress to be true")
	}
}
