package main

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gpulease"
)

func TestLoadLocalLauncherModelWithMetalLeaseRefusesBeforeLoadAndReleasesAfterServe(t *testing.T) {
	const holderEnv = "FAK_LOCAL_LAUNCHER_METAL_LEASE_HOLDER_TEST"
	if path := os.Getenv(holderEnv); path != "" {
		lease, err := gpulease.Acquire(gpulease.Options{Path: path, NoWait: true})
		if err != nil {
			fmt.Fprintln(os.Stderr, "child lease acquire:", err)
			os.Exit(3)
		}
		fmt.Fprintf(os.Stdout, "READY %d\n", os.Getpid())
		_ = os.Stdout.Sync()
		_, _ = io.Copy(io.Discard, os.Stdin)
		runtime.KeepAlive(lease)
		os.Exit(0) // The OS, not an in-process Release call, drops the child flock.
	}

	path := filepath.Join(t.TempDir(), "gpu.lease")
	t.Setenv("FAK_GPU_LEASE", path)
	t.Setenv("FAK_ADMISSION_POLICY", "dev")
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	t.Cleanup(cancel)
	child := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestLoadLocalLauncherModelWithMetalLeaseRefusesBeforeLoadAndReleasesAfterServe$")
	child.Env = append(os.Environ(), holderEnv+"="+path)
	childIn, err := child.StdinPipe()
	if err != nil {
		t.Fatalf("child stdin: %v", err)
	}
	childOut, err := child.StdoutPipe()
	if err != nil {
		t.Fatalf("child stdout: %v", err)
	}
	var childStderr strings.Builder
	child.Stderr = &childStderr
	if err := child.Start(); err != nil {
		t.Fatalf("start unrelated lease holder: %v", err)
	}
	waited := false
	t.Cleanup(func() {
		_ = childIn.Close()
		if !waited && child.Process != nil {
			_ = child.Process.Kill()
			_ = child.Wait()
		}
	})
	ready, err := bufio.NewReader(childOut).ReadString('\n')
	if err != nil {
		t.Fatalf("wait for child lease holder: %v; stderr=%s", err, childStderr.String())
	}
	wantReady := "READY " + strconv.Itoa(child.Process.Pid)
	if strings.TrimSpace(ready) != wantReady {
		t.Fatalf("child readiness = %q, want %q; stderr=%s", strings.TrimSpace(ready), wantReady, childStderr.String())
	}

	loads := 0
	release, err := loadLocalLauncherModelWithMetalLease(true, "qwen3.8-27b-q4_k_m.gguf", gpulease.Options{}, func() {
		loads++
	})
	if err == nil {
		t.Fatal("Metal serve admission succeeded while modelbench-compatible lease was held")
	}
	if !errors.Is(err, gpulease.ErrBusy) {
		t.Fatalf("busy admission error = %v, want errors.Is(ErrBusy)", err)
	}
	for _, want := range []string{path, "pid " + strconv.Itoa(child.Process.Pid), "before model load", "stop the holder process"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("busy admission error %q does not contain %q", err, want)
		}
	}
	if loads != 0 {
		t.Fatalf("load callback calls while lease held = %d, want 0", loads)
	}
	release()

	if err := childIn.Close(); err != nil {
		t.Fatalf("signal holder exit: %v", err)
	}
	if err := child.Wait(); err != nil {
		t.Fatalf("holder exit: %v; stderr=%s", err, childStderr.String())
	}
	waited = true
	release, err = loadLocalLauncherModelWithMetalLease(true, "qwen3.8-27b-q4_k_m.gguf", gpulease.Options{}, func() {
		loads++
	})
	if err != nil {
		t.Fatalf("admit after holder release: %v", err)
	}
	if loads != 1 {
		t.Fatalf("load callback calls after admission = %d, want 1", loads)
	}
	if _, err := gpulease.Acquire(gpulease.Options{NoWait: true}); !errors.Is(err, gpulease.ErrBusy) {
		t.Fatalf("serve lease was not retained after load callback: got %v, want ErrBusy", err)
	}

	release()
	reacquired, err := gpulease.Acquire(gpulease.Options{NoWait: true})
	if err != nil {
		t.Fatalf("reacquire after serve cleanup: %v", err)
	}
	reacquired.Release()
}

func TestLoadLocalLauncherModelWithMetalLeaseLeavesCPUAndEmptyModelUnserialized(t *testing.T) {
	tests := []struct {
		name     string
		metal    bool
		ggufPath string
	}{
		{name: "CPU model", metal: false, ggufPath: "model.gguf"},
		{name: "Metal proxy without local model", metal: true, ggufPath: ""},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "gpu.lease")
			held, err := gpulease.Acquire(gpulease.Options{Path: path})
			if err != nil {
				t.Fatalf("hold unrelated lease: %v", err)
			}
			defer held.Release()

			loads := 0
			release, err := loadLocalLauncherModelWithMetalLease(tc.metal, tc.ggufPath, gpulease.Options{Path: path}, func() { loads++ })
			if err != nil {
				t.Fatalf("unserialized path: %v", err)
			}
			defer release()
			if loads != 1 {
				t.Fatalf("load callback calls = %d, want 1", loads)
			}
		})
	}
}

func TestLoadLocalLauncherModelWithMetalLeasePressurePolicy(t *testing.T) {
	path := filepath.Join(t.TempDir(), "gpu.lease")
	t.Setenv("FAK_GPU_LEASE", path)
	t.Setenv("FAK_NATIVE_ADMISSION", "exclusive")

	loaded := false
	release, err := loadLocalLauncherModelWithMetalLease(true, "test.gguf", gpulease.Options{Path: path}, func() {
		loaded = true
	})
	if err != nil {
		t.Fatalf("exclusive rollback should admit without reservation gate: %v", err)
	}
	defer release()
	if !loaded {
		t.Fatal("expected load to run under exclusive admission")
	}
}

func TestLoadLocalLauncherModelWithMetalLeaseCoexistsForSmallModels(t *testing.T) {
	resDir := filepath.Join(t.TempDir(), "reservations")
	leasePath := filepath.Join(t.TempDir(), "gpu.lease")
	t.Setenv("FAK_RESERVATION_DIR", resDir)
	t.Setenv("FAK_GPU_LEASE", leasePath)
	t.Setenv("FAK_ADMISSION_POLICY", "dev")
	t.Setenv("FAK_NATIVE_ADMISSION", "aggregate")
	t.Setenv("FAK_TEST_STARTUP_PEAK_BYTES", "536870912") // 512 MiB
	t.Setenv("FAK_TEST_STEADY_BYTES", "268435456")       // 256 MiB

	loads1 := 0
	release1, err := loadLocalLauncherModelWithMetalLease(true, "small-model-1.gguf", gpulease.Options{}, func() {
		loads1++
	})
	if err != nil {
		t.Fatalf("first small model load failed: %v", err)
	}
	defer release1()
	if loads1 != 1 {
		t.Fatalf("first small model expected 1 load, got %d", loads1)
	}

	loads2 := 0
	release2, err := loadLocalLauncherModelWithMetalLease(true, "small-model-2.gguf", gpulease.Options{}, func() {
		loads2++
	})
	if err != nil {
		t.Fatalf("second small model should coexist when aggregate fits: %v", err)
	}
	defer release2()
	if loads2 != 1 {
		t.Fatalf("second small model expected 1 load, got %d", loads2)
	}

	// Verify ledger recorded both coexisting reservations in steady state.
	ledgerPath := filepath.Join(resDir, "reservations.json")
	data, err := os.ReadFile(ledgerPath)
	if err != nil {
		t.Fatalf("read ledger: %v", err)
	}
	content := string(data)
	if !strings.Contains(content, "steady") {
		t.Fatalf("ledger should record steady phase: %s", content)
	}

	// Release first model, verify second remains active.
	release1()
	dataAfter1, err := os.ReadFile(ledgerPath)
	if err != nil {
		t.Fatalf("read ledger after release1: %v", err)
	}
	if !strings.Contains(string(dataAfter1), "steady") {
		t.Fatalf("ledger should still contain second model after release1: %s", string(dataAfter1))
	}

	// Release second model, verify cleanup.
	release2()
	dataAfter2, err := os.ReadFile(ledgerPath)
	if err != nil {
		t.Fatalf("read ledger after release2: %v", err)
	}
	if strings.Contains(string(dataAfter2), "\"held_bytes\":268435456") {
		t.Fatalf("ledger should have cleaned up second reservation: %s", string(dataAfter2))
	}
}

func TestLoadLocalLauncherModelWithMetalLeaseRefusesOvercommitBeforeLoader(t *testing.T) {
	resDir := filepath.Join(t.TempDir(), "reservations")
	leasePath := filepath.Join(t.TempDir(), "gpu.lease")
	t.Setenv("FAK_RESERVATION_DIR", resDir)
	t.Setenv("FAK_GPU_LEASE", leasePath)
	t.Setenv("FAK_ADMISSION_POLICY", "dev")

	// Set impossible peak bytes (e.g. 500 GiB on Apple Silicon)
	t.Setenv("FAK_TEST_STARTUP_PEAK_BYTES", strconv.FormatInt(500<<30, 10))
	t.Setenv("FAK_TEST_STEADY_BYTES", strconv.FormatInt(400<<30, 10))

	loads := 0
	release, err := loadLocalLauncherModelWithMetalLease(true, "huge-model.gguf", gpulease.Options{}, func() {
		loads++
	})
	if err == nil {
		release()
		t.Fatal("expected overcommit to refuse before load, got success")
	}
	if loads != 0 {
		t.Fatalf("loader must not be called on refusal, got %d loads", loads)
	}
	if !strings.Contains(err.Error(), "aggregate_capacity") && !strings.Contains(err.Error(), "refused") {
		t.Fatalf("unexpected error message: %v", err)
	}
}

func TestLoadLocalLauncherModelWithMetalLeaseLoadFailureReleasesReservation(t *testing.T) {
	resDir := filepath.Join(t.TempDir(), "reservations")
	leasePath := filepath.Join(t.TempDir(), "gpu.lease")
	t.Setenv("FAK_RESERVATION_DIR", resDir)
	t.Setenv("FAK_GPU_LEASE", leasePath)
	t.Setenv("FAK_ADMISSION_POLICY", "dev")
	t.Setenv("FAK_TEST_STARTUP_PEAK_BYTES", "536870912")
	t.Setenv("FAK_TEST_STEADY_BYTES", "268435456")

	defer func() {
		_ = recover()
		// After panic in loader, check that ledger was reaped / released
		ledgerPath := filepath.Join(resDir, "reservations.json")
		data, err := os.ReadFile(ledgerPath)
		if err == nil && strings.Contains(string(data), "\"phase\":\"startup\"") {
			t.Fatalf("aborted load must not leak startup reservation in ledger: %s", string(data))
		}
	}()

	_, _ = loadLocalLauncherModelWithMetalLease(true, "panicking-model.gguf", gpulease.Options{}, func() {
		panic("simulated loader panic")
	})
}
