package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
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

	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/gpulease"
	"github.com/anthony-chaudhary/fak/internal/localadmission"
	"github.com/anthony-chaudhary/fak/internal/memgate"
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

	modelPath := writeServePackedEmbeddingFixture(t, "lease-model.gguf", false)
	path := filepath.Join(t.TempDir(), "gpu.lease")
	t.Setenv("FAK_GPU_LEASE", path)
	t.Setenv("FAK_RESERVATION_DIR", t.TempDir())
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
	release, err := loadLocalLauncherModelWithMetalLease(true, modelPath, gpulease.Options{}, func() {
		loads++
	})
	if err == nil {
		t.Fatal("Metal serve admission succeeded while modelbench-compatible lease was held")
	}
	if !errors.Is(err, gpulease.ErrBusy) {
		t.Fatalf("busy admission error = %v, want errors.Is(ErrBusy)", err)
	}
	for _, want := range []string{path, "pid " + strconv.Itoa(child.Process.Pid), "before model load", "stop the holder process", "fak doctor serve"} {
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
	release, err = loadLocalLauncherModelWithMetalLease(true, modelPath, gpulease.Options{}, func() {
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

	modelPath := writeServePackedEmbeddingFixture(t, "lease-model.gguf", false)
	loaded := false
	release, err := loadLocalLauncherModelWithMetalLease(true, modelPath, gpulease.Options{Path: path}, func() {
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

func TestLoadLocalLauncherModelWithVulkanLeaseRefusesBeforeLoadAndReleasesAfterServe(t *testing.T) {
	const holderEnv = "FAK_LOCAL_LAUNCHER_VULKAN_LEASE_HOLDER_TEST"
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
		os.Exit(0)
	}

	path := filepath.Join(t.TempDir(), "gpu.lease")
	t.Setenv("FAK_GPU_LEASE", path)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	t.Cleanup(cancel)
	child := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestLoadLocalLauncherModelWithVulkanLeaseRefusesBeforeLoadAndReleasesAfterServe$")
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
	release, err := loadLocalLauncherModelWithVulkanLease(true, "qwen3.8-27b-q4_k_m.gguf", gpulease.Options{}, func() {
		loads++
	})
	if err == nil {
		t.Fatal("Vulkan serve admission succeeded while lease was held")
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

	release, err = loadLocalLauncherModelWithVulkanLease(true, "qwen3.8-27b-q4_k_m.gguf", gpulease.Options{}, func() {
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

	// Verify lease file contains our pid
	lockData, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read lease file: %v", err)
	}
	if strings.TrimSpace(string(lockData)) != strconv.Itoa(os.Getpid()) {
		t.Fatalf("lease holder in file = %q, want %d", strings.TrimSpace(string(lockData)), os.Getpid())
	}

	release()
	reacquired, err := gpulease.Acquire(gpulease.Options{NoWait: true})
	if err != nil {
		t.Fatalf("reacquire after serve cleanup: %v", err)
	}
	reacquired.Release()
}

func TestLoadLocalLauncherModelWithVulkanLeaseLeavesCPUAndEmptyModelUnserialized(t *testing.T) {
	tests := []struct {
		name      string
		vulkan    bool
		modelPath string
	}{
		{name: "CPU model", vulkan: false, modelPath: "model.gguf"},
		{name: "Vulkan proxy without local model", vulkan: true, modelPath: ""},
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
			release, err := loadLocalLauncherModelWithVulkanLease(tc.vulkan, tc.modelPath, gpulease.Options{Path: path}, func() { loads++ })
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

func TestVulkanServiceProcessLeaseLifetime(t *testing.T) {
	const serviceEnv = "FAK_VULKAN_SERVICE_PROCESS_REGRESSION_TEST"
	if path := os.Getenv(serviceEnv); path != "" {
		modelPath := os.Getenv("FAK_VULKAN_SERVICE_MODEL")
		if modelPath == "" {
			modelPath = "qwen3.8-27b-q4_k_m.gguf"
		}
		loaded := false
		release, err := loadLocalLauncherModelWithVulkanLease(true, modelPath, gpulease.Options{Path: path, NoWait: true}, func() {
			loaded = true
		})
		if err != nil {
			fmt.Fprintf(os.Stderr, "service start refused: %v\n", err)
			os.Exit(2)
		}
		if !loaded {
			fmt.Fprintln(os.Stderr, "service start error: load callback never ran")
			os.Exit(3)
		}
		fmt.Fprintf(os.Stdout, "SERVING %d\n", os.Getpid())
		_ = os.Stdout.Sync()
		_, _ = io.Copy(io.Discard, os.Stdin)
		release()
		os.Exit(0)
	}

	path := filepath.Join(t.TempDir(), "fak-gpu.lease")
	t.Setenv("FAK_GPU_LEASE", path)

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	t.Cleanup(cancel)

	// Step 1: Launch service process 1 with temporary lease.
	child1 := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestVulkanServiceProcessLeaseLifetime$")
	child1.Env = append(os.Environ(), serviceEnv+"="+path)
	child1In, err := child1.StdinPipe()
	if err != nil {
		t.Fatalf("child1 stdin pipe: %v", err)
	}
	child1Out, err := child1.StdoutPipe()
	if err != nil {
		t.Fatalf("child1 stdout pipe: %v", err)
	}
	var child1Stderr strings.Builder
	child1.Stderr = &child1Stderr
	if err := child1.Start(); err != nil {
		t.Fatalf("start service child1: %v", err)
	}
	waited1 := false
	t.Cleanup(func() {
		_ = child1In.Close()
		if !waited1 && child1.Process != nil {
			_ = child1.Process.Kill()
			_ = child1.Wait()
		}
	})

	reader1 := bufio.NewReader(child1Out)
	line, err := reader1.ReadString('\n')
	if err != nil {
		t.Fatalf("read service readiness: %v; stderr=%s", err, child1Stderr.String())
	}
	wantServing := "SERVING " + strconv.Itoa(child1.Process.Pid)
	if strings.TrimSpace(line) != wantServing {
		t.Fatalf("service readiness = %q, want %q; stderr=%s", strings.TrimSpace(line), wantServing, child1Stderr.String())
	}

	// Step 2: Prove the service PID owns the canonical lease on disk.
	lockData, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read lease file %s: %v", path, err)
	}
	recordedPID, err := strconv.Atoi(strings.TrimSpace(string(lockData)))
	if err != nil || recordedPID != child1.Process.Pid {
		t.Fatalf("lease file records pid %q (parsed %d), want service PID %d", string(lockData), recordedPID, child1.Process.Pid)
	}

	// Step 3: Prove a competing acquisition fails while service runs.
	competingLease, compErr := gpulease.Acquire(gpulease.Options{Path: path, NoWait: true})
	if compErr == nil {
		competingLease.Release()
		t.Fatal("competing lease acquisition succeeded while service was running")
	}
	if !errors.Is(compErr, gpulease.ErrBusy) {
		t.Fatalf("competing error = %v, want errors.Is(ErrBusy)", compErr)
	}

	// Step 4: Prove a competing service instance fails before model load.
	child2 := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestVulkanServiceProcessLeaseLifetime$")
	child2.Env = append(os.Environ(), serviceEnv+"="+path)
	var child2Stdout, child2Stderr strings.Builder
	child2.Stdout = &child2Stdout
	child2.Stderr = &child2Stderr
	child2Err := child2.Run()
	if child2Err == nil {
		t.Fatal("competing service child2 succeeded while service child1 held lease")
	}
	if !strings.Contains(child2Stderr.String(), "Vulkan residency admission refused before model load") {
		t.Fatalf("child2 stderr %q does not contain refusal notice", child2Stderr.String())
	}
	if !strings.Contains(child2Stderr.String(), strconv.Itoa(child1.Process.Pid)) {
		t.Fatalf("child2 stderr %q does not identify incumbent PID %d", child2Stderr.String(), child1.Process.Pid)
	}

	// Step 5: Close child 1 and prove lease is released on clean process exit.
	if err := child1In.Close(); err != nil {
		t.Fatalf("close child1 stdin: %v", err)
	}
	if err := child1.Wait(); err != nil {
		t.Fatalf("child1 wait: %v; stderr=%s", err, child1Stderr.String())
	}
	waited1 = true

	// Step 6: Verify lease is free and can be acquired immediately.
	afterLease, err := gpulease.Acquire(gpulease.Options{Path: path, NoWait: true})
	if err != nil {
		t.Fatalf("acquire lease after service exit: %v", err)
	}
	afterLease.Release()
}

func TestIsServeVulkan(t *testing.T) {
	t.Setenv("FAK_BACKEND", "")
	if isServeVulkan(nil) {
		t.Error("nil serve flags should not be vulkan")
	}

	backendName := "vulkan"
	baseURL := "http://127.0.0.1:8080/v1"
	sfProxy := &serveFlags{backendName: &backendName, baseURL: &baseURL}
	if isServeVulkan(sfProxy) {
		t.Error("sf with baseURL should not be vulkan (proxy mode is lease-free)")
	}

	cpu := "cpu"
	if isServeVulkan(&serveFlags{backendName: &cpu}) {
		t.Error("explicit CPU floor should not be vulkan")
	}
	unknown := "definitely-not-a-backend"
	if isServeVulkan(&serveFlags{backendName: &unknown}) {
		t.Error("unavailable backend should not be vulkan")
	}
	if isServeVulkan(&serveFlags{}) {
		t.Error("nil backendName should not be vulkan")
	}
}

func TestLoadServeModelWithVulkanLeaseUsesEffectiveBackendAndGGUF(t *testing.T) {
	autoLease := runtime.GOOS == "linux" || runtime.GOOS == "windows"
	tests := []struct {
		name       string
		backend    string
		envBackend string
		registered string
		gguf       string
		baseURL    string
		wantLease  bool
	}{
		{name: "explicit local Vulkan GGUF", backend: "vulkan", registered: "vulkan", gguf: "model.gguf", wantLease: true},
		{name: "empty GGUF never substitutes default model", backend: "vulkan", registered: "vulkan"},
		{name: "uppercase backend normalized", backend: "VULKAN", registered: "vulkan", gguf: "model.gguf", wantLease: true},
		{name: "whitespace backend normalized", backend: " vulkan ", registered: "vulkan", gguf: "model.gguf", wantLease: true},
		{name: "automatic registered Vulkan", registered: "vulkan", gguf: "model.gguf", wantLease: autoLease},
		{name: "environment Vulkan", envBackend: "vulkan", registered: "vulkan", gguf: "model.gguf", wantLease: true},
		{name: "proxy bypass", backend: "vulkan", registered: "vulkan", gguf: "model.gguf", baseURL: "http://127.0.0.1:8080/v1"},
		{name: "CPU reference bypass", backend: "cpu", registered: "vulkan", gguf: "model.gguf"},
		{name: "nonempty whitespace GGUF admits before load validation", backend: "vulkan", registered: "vulkan", gguf: " ", wantLease: true},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
			defer cancel()
			cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestServeBackendAutoChild$")
			cmd.Env = backendAutoChildEnvironment(map[string]string{
				backendAutoChildEnv:                 "1",
				"FAK_TEST_SERVE_BACKEND_ACTION":     "lease-load",
				"FAK_TEST_SERVE_BACKEND_REQUESTED":  tc.backend,
				"FAK_TEST_SERVE_BACKEND_REGISTERED": tc.registered,
				"FAK_TEST_SERVE_BACKEND_BASE_URL":   tc.baseURL,
				"FAK_TEST_SERVE_BACKEND_GGUF":       tc.gguf,
				"FAK_TEST_SERVE_BACKEND_LEASE_PATH": filepath.Join(t.TempDir(), "gpu.lease"),
				"FAK_TEST_SERVE_BACKEND_WANT":       strconv.FormatBool(tc.wantLease),
				"FAK_BACKEND":                       tc.envBackend,
				"FAK_VULKAN_SPIRV":                  "",
			})
			if out, err := cmd.CombinedOutput(); err != nil {
				if ctx.Err() != nil {
					t.Fatalf("child lease lifecycle timed out: %v", ctx.Err())
				}
				t.Fatalf("child lease lifecycle failed: %v\n%s", err, out)
			}
		})
	}
}

func TestLoadLocalLauncherModelWithMetalLeaseRefusesExpandingQ8LoadOn36GBHost(t *testing.T) {
	origRead := serveReadMemory
	defer func() { serveReadMemory = origRead }()
	serveReadMemory = func() (memgate.Memory, error) {
		mem, err := origRead()
		if err == nil && mem.AvailableBytes < 15*(1<<30) {
			mem.TotalBytes = 36 * (1 << 30)
			mem.AvailableBytes = 22 * (1 << 30)
			mem.FreeBytes = 20 * (1 << 30)
		}
		return mem, err
	}

	resDir := filepath.Join(t.TempDir(), "reservations")
	leasePath := filepath.Join(t.TempDir(), "gpu.lease")
	t.Setenv("FAK_RESERVATION_DIR", resDir)
	t.Setenv("FAK_GPU_LEASE", leasePath)
	t.Setenv("FAK_ADMISSION_POLICY", "dev")

	udPath := filepath.Join(t.TempDir(), "qwen38-27b-ud-q2kxl.gguf")
	writeSynth27BGGUF(t, udPath, true)

	t.Run("resident UD-Q2_K_XL admits on host", func(t *testing.T) {
		loads := 0
		release, err := loadLocalLauncherModelWithMetalLease(true, udPath, gpulease.Options{}, func() {
			loads++
		})
		if err != nil {
			t.Fatalf("expected resident load of 27B UD-Q2_K_XL to admit, got: %v", err)
		}
		defer release()
		if loads != 1 {
			t.Fatalf("expected 1 load, got %d", loads)
		}
	})

	t.Run("FAK_Q4K=0 forces expanding Q8 load and refuses", func(t *testing.T) {
		t.Setenv("FAK_Q4K", "0")
		loads := 0
		release, err := loadLocalLauncherModelWithMetalLease(true, udPath, gpulease.Options{}, func() {
			loads++
		})
		if err == nil {
			release()
			t.Fatal("expected expanding Q8 load of 27B model to be refused before load on 36 GiB host, got success")
		}
		if loads != 0 {
			t.Fatalf("loader must not be called on refusal, got %d loads", loads)
		}
		if !strings.Contains(err.Error(), "refused") {
			t.Fatalf("unexpected error message: %v", err)
		}
		if !strings.Contains(err.Error(), "exceeds available allocatable capacity") {
			t.Fatalf("expected detailed remedy hint in error message, got: %v", err)
		}
	})
}

func TestStreamedQ4KReservationUsesProcessPeakNotHostFloor(t *testing.T) {
	t.Setenv("FAK_STREAM_Q4K", "1")
	t.Setenv("FAK_Q4K_FREE_CPU", "1")
	t.Setenv("FAK_Q4K", "1")

	path := filepath.Join(t.TempDir(), "Qwen3.8-27B-Q4_K_M.gguf")
	writeSynth27BGGUF(t, path, false)
	if err := os.Truncate(path, qwen38Q4KMArtifactBytes); err != nil {
		t.Fatal(err)
	}
	plan, err := estimateMetalModelMemoryBounds(path)
	if err != nil {
		t.Fatal(err)
	}
	if plan.StartupPeakBytes != streamedQ4KFreeCPUReservationPeakBytes {
		t.Fatalf("reservation startup peak = %d, want conservative FreeCPU process bound %d", plan.StartupPeakBytes, streamedQ4KFreeCPUReservationPeakBytes)
	}
	if plan.SteadyBytes <= 0 || plan.SteadyBytes > plan.StartupPeakBytes {
		t.Fatalf("invalid reservation bounds: %+v", plan)
	}
	required, refuse, mode := streamedQ4KMetalCapacity(36<<30, true, true)
	if required != 36<<30 || refuse || mode != streamedQ4KModeFreeCPU {
		t.Fatalf("host floor = (%d, %v, %q), want (%d, false, %q)", required, refuse, mode, int64(36<<30), streamedQ4KModeFreeCPU)
	}
	otherPath := filepath.Join(filepath.Dir(path), "unwitnessed-q4.gguf")
	if err := os.Rename(path, otherPath); err != nil {
		t.Fatal(err)
	}
	other, err := estimateMetalModelMemoryBounds(otherPath)
	if err != nil {
		t.Fatal(err)
	}
	if other.StartupPeakBytes != 36<<30 {
		t.Fatalf("unmatched streamed profile startup peak = %d, want preserved 36 GiB host-floor reservation", other.StartupPeakBytes)
	}
}

func TestLoadLocalLauncherModelWithMetalLeaseWarningPressureAdvisory(t *testing.T) {
	origRead := serveReadMemory
	defer func() { serveReadMemory = origRead }()

	serveReadMemory = func() (memgate.Memory, error) {
		return memgate.Memory{
			TotalBytes:      40 * (1 << 30),
			AvailableBytes:  25 * (1 << 30),
			CompressedBytes: 5 * (1 << 30),
			WiredBytes:      2 * (1 << 30),
		}, nil
	}

	path := filepath.Join(t.TempDir(), "gpu.lease")
	t.Setenv("FAK_GPU_LEASE", path)
	t.Setenv("FAK_ADMISSION_POLICY", "dev")
	t.Setenv("FAK_TEST_STARTUP_PEAK_BYTES", "536870912")
	t.Setenv("FAK_TEST_STEADY_BYTES", "268435456")

	oldStderr := os.Stderr
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	os.Stderr = w

	loaded := false
	release, err := loadLocalLauncherModelWithMetalLease(true, "small-model.gguf", gpulease.Options{Path: path}, func() {
		loaded = true
	})

	_ = w.Close()
	os.Stderr = oldStderr

	var buf bytes.Buffer
	_, _ = io.Copy(&buf, r)
	_ = r.Close()

	if err != nil {
		t.Fatalf("load failed: %v", err)
	}
	defer release()
	if !loaded {
		t.Fatal("expected model to load")
	}

	stderrOutput := buf.String()
	want := "fak local launcher: advisory: ambient memory pressure is warning"
	if !strings.Contains(stderrOutput, want) {
		t.Fatalf("stderr %q does not contain %q", stderrOutput, want)
	}
	if !strings.Contains(stderrOutput, "close background apps if paging occurs") {
		t.Fatalf("stderr %q does not contain close background apps advice", stderrOutput)
	}
}

// stubServeAllocatable replaces the host memory probe for the duration of the test so
// the reservation capacity is deterministic. allocatable and total are injected bytes.
func stubServeAllocatable(t *testing.T, total, allocatable int64) {
	t.Helper()
	orig := serveReadMemory
	t.Cleanup(func() { serveReadMemory = orig })
	serveReadMemory = func() (memgate.Memory, error) {
		return memgate.Memory{
			TotalBytes:      total,
			FreeBytes:       allocatable,
			AvailableBytes:  allocatable,
			CompressedBytes: 0,
			WiredBytes:      0,
		}, nil
	}
}

// TestLoadLocalLauncherModelWithMetalLeaseRefusesWhenWeightsFitButStateEnvelopeDoesNot
// is the issue #9587 RED/GREEN witness. The weights-only plan for a small model fits the
// injected allocatable capacity, but once FAK_ADMISSION_STATE_ENVELOPE=1 widens the
// reservation to the aggregate session/cache envelope the same load must be REFUSED
// before the loader runs. With the envelope off, the identical setup is ADMITTED, proving
// the envelope — not the weights — changed the verdict.
func TestLoadLocalLauncherModelWithMetalLeaseRefusesWhenWeightsFitButStateEnvelopeDoesNot(t *testing.T) {
	const (
		smallPeak      = 512 << 20 // 512 MiB weights-only startup peak
		smallSteady    = 256 << 20 // 256 MiB weights-only steady
		tightAllocable = 768 << 20 // 0.75 GiB: fits weights-only, not the state envelope
		roomyAllocable = 64 << 30  // 64 GiB: envelope fits
	)

	t.Run("state envelope refuses a weights-only-fitting load", func(t *testing.T) {
		stubServeAllocatable(t, 2<<30, tightAllocable)
		resDir := filepath.Join(t.TempDir(), "reservations")
		t.Setenv("FAK_RESERVATION_DIR", resDir)
		t.Setenv("FAK_GPU_LEASE", filepath.Join(t.TempDir(), "gpu.lease"))
		t.Setenv("FAK_ADMISSION_POLICY", "dev")
		t.Setenv("FAK_NATIVE_ADMISSION", "aggregate")
		t.Setenv("FAK_ADMISSION_STATE_ENVELOPE", "1")
		t.Setenv("FAK_TEST_STARTUP_PEAK_BYTES", strconv.Itoa(smallPeak))
		t.Setenv("FAK_TEST_STEADY_BYTES", strconv.Itoa(smallSteady))

		loads := 0
		release, err := loadLocalLauncherModelWithMetalLease(true, "small-state-heavy.gguf", gpulease.Options{}, func() {
			loads++
		})
		if err == nil {
			release()
			t.Fatal("expected state-envelope reservation to refuse a weights-only-fitting load, got success")
		}
		if loads != 0 {
			t.Fatalf("loader must not run on state-envelope refusal, got %d loads", loads)
		}
		if !strings.Contains(err.Error(), "aggregate_capacity") {
			t.Fatalf("refusal %q does not name aggregate capacity", err)
		}
		if !strings.Contains(err.Error(), "exceeds available allocatable capacity") {
			t.Fatalf("refusal %q does not carry the aggregate-capacity remedy hint", err)
		}
	})

	t.Run("same load is admitted when the envelope is off", func(t *testing.T) {
		stubServeAllocatable(t, 2<<30, tightAllocable)
		t.Setenv("FAK_RESERVATION_DIR", filepath.Join(t.TempDir(), "reservations"))
		t.Setenv("FAK_GPU_LEASE", filepath.Join(t.TempDir(), "gpu.lease"))
		t.Setenv("FAK_ADMISSION_POLICY", "dev")
		t.Setenv("FAK_NATIVE_ADMISSION", "aggregate")
		t.Setenv("FAK_ADMISSION_STATE_ENVELOPE", "0")
		t.Setenv("FAK_TEST_STARTUP_PEAK_BYTES", strconv.Itoa(smallPeak))
		t.Setenv("FAK_TEST_STEADY_BYTES", strconv.Itoa(smallSteady))

		loads := 0
		release, err := loadLocalLauncherModelWithMetalLease(true, "small-state-heavy.gguf", gpulease.Options{}, func() {
			loads++
		})
		if err != nil {
			t.Fatalf("weights-only load should be admitted with the envelope off: %v", err)
		}
		defer release()
		if loads != 1 {
			t.Fatalf("weights-only load callback calls = %d, want 1", loads)
		}
	})

	t.Run("fits and records class breakdown retained through steady", func(t *testing.T) {
		stubServeAllocatable(t, 128<<30, roomyAllocable)
		resDir := filepath.Join(t.TempDir(), "reservations")
		t.Setenv("FAK_RESERVATION_DIR", resDir)
		t.Setenv("FAK_GPU_LEASE", filepath.Join(t.TempDir(), "gpu.lease"))
		t.Setenv("FAK_ADMISSION_POLICY", "dev")
		t.Setenv("FAK_NATIVE_ADMISSION", "aggregate")
		t.Setenv("FAK_ADMISSION_STATE_ENVELOPE", "1")
		t.Setenv("FAK_TEST_STARTUP_PEAK_BYTES", strconv.Itoa(smallPeak))
		t.Setenv("FAK_TEST_STEADY_BYTES", strconv.Itoa(smallSteady))

		loads := 0
		release, err := loadLocalLauncherModelWithMetalLease(true, "small-state-light.gguf", gpulease.Options{}, func() {
			loads++
		})
		if err != nil {
			t.Fatalf("state envelope should fit in roomy allocatable memory: %v", err)
		}
		defer release()
		if loads != 1 {
			t.Fatalf("load callback calls = %d, want 1", loads)
		}

		ledgerPath := filepath.Join(resDir, "reservations.json")
		data, err := os.ReadFile(ledgerPath)
		if err != nil {
			t.Fatalf("read ledger: %v", err)
		}
		content := string(data)
		if !strings.Contains(content, "kv_cache") {
			t.Fatalf("ledger did not record the kv_cache class breakdown: %s", content)
		}
		if !strings.Contains(content, "steady") {
			t.Fatalf("ledger did not retain the reservation through steady: %s", content)
		}
		if strings.Contains(content, "\"phase\": \"startup\"") {
			t.Fatalf("ledger leaked a startup-phase reservation after MarkSteady: %s", content)
		}
	})
}

// TestEstimateModelStateEnvelopeIsMonotonicAndConservative pins the pure envelope helper:
// it accounts for weights plus every enforced session plus cohort scratch, never reports
// startup below steady, and grows with the session count.
func TestEstimateModelStateEnvelopeIsMonotonicAndConservative(t *testing.T) {
	const (
		weights     = 256 << 20
		tokenBudget = 8192
	)
	env := estimateModelStateEnvelope(weights, tokenBudget, 8)
	if env.StartupPeakBytes() < env.SteadyBytes() {
		t.Fatalf("startup peak %d < steady %d", env.StartupPeakBytes(), env.SteadyBytes())
	}
	if env.SteadyBytes() <= int64(weights) {
		t.Fatalf("envelope steady %d must exceed weights alone %d", env.SteadyBytes(), weights)
	}
	classes := env.Classes()
	if classes[localadmission.MemClassKVCache] <= 0 {
		t.Fatalf("envelope classes missing kv_cache: %v", classes)
	}
	if classes[localadmission.MemClassActivation] <= 0 {
		t.Fatalf("envelope classes missing activation scratch: %v", classes)
	}
	if classes[localadmission.MemClassWeights] <= 0 {
		t.Fatalf("envelope classes missing weights: %v", classes)
	}

	one := estimateModelStateEnvelope(weights, tokenBudget, 1)
	if one.SteadyBytes() >= env.SteadyBytes() {
		t.Fatalf("envelope with 1 session (%d) must be smaller than 8 sessions (%d)", one.SteadyBytes(), env.SteadyBytes())
	}
}

// TestSaturatingEnvelopeMulFailsTowardTooBig pins the overflow guard: a product
// that would wrap must saturate high so the KV term stays conservative, never
// collapse toward zero and under-reserve.
func TestSaturatingEnvelopeMulFailsTowardTooBig(t *testing.T) {
	if got := saturatingEnvelopeMul(0, 1<<20); got != 0 {
		t.Fatalf("zero operand = %d, want 0", got)
	}
	if got := saturatingEnvelopeMul(-1, 1<<20); got != 0 {
		t.Fatalf("negative operand = %d, want 0", got)
	}
	if got := saturatingEnvelopeMul(4, 5); got != 20 {
		t.Fatalf("normal product = %d, want 20", got)
	}
	// Largest token budget * the per-token scalar overflows int64; the guard must
	// return MaxInt64 rather than a wrapped, possibly-small value.
	if got := saturatingEnvelopeMul(1<<47, stateEnvelopeKVBytesPerToken); got != int64(^uint64(0)>>1) {
		t.Fatalf("overflowing product = %d, want MaxInt64", got)
	}
}

// This regression calls the unchanged launcher seam, reads its real on-disk
// reservation ledger during allocation and after downshift, and reacquires its
// actual GPU lease after errors and cleanup.
func TestMetalLauncherReservationUsesStoredWeightsAndReleasesOnEstimateError(t *testing.T) {
	t.Setenv("FAK_Q4K", "1")
	t.Setenv("FAK_STREAM_Q4K", "")
	t.Setenv("FAK_METAL_STREAM_Q4K", "")
	t.Setenv("FAK_TEST_STARTUP_PEAK_BYTES", "")
	t.Setenv("FAK_TEST_STEADY_BYTES", "")
	t.Setenv("FAK_NATIVE_ADMISSION", "aggregate")
	t.Setenv("FAK_ADMISSION_POLICY", "dev")
	original := serveReadMemory
	serveReadMemory = func() (memgate.Memory, error) {
		return memgate.Memory{TotalBytes: 64 << 30, AvailableBytes: 60 << 30}, nil
	}
	t.Cleanup(func() { serveReadMemory = original })
	for _, name := range []string{"stored-weights", "unsupported-architecture", "malformed-header", "missing-file"} {
		t.Run(name, func(t *testing.T) {
			dir := t.TempDir()
			t.Setenv("FAK_RESERVATION_DIR", dir)
			leasePath := filepath.Join(dir, "gpu.lease")
			t.Setenv("FAK_GPU_LEASE", leasePath)
			readLedger := func() []localadmission.Reservation {
				t.Helper()
				b, err := os.ReadFile(filepath.Join(dir, "reservations.json"))
				if os.IsNotExist(err) {
					return nil
				}
				if err != nil {
					t.Fatal(err)
				}
				var ledger struct {
					Reservations []localadmission.Reservation `json:"reservations"`
				}
				if err := json.Unmarshal(b, &ledger); err != nil {
					t.Fatal(err)
				}
				return ledger.Reservations
			}
			path := filepath.Join(dir, "missing.gguf")
			const wantStored int64 = 3*256*144 + 3*210 + 3*256*4
			wantSteady := wantStored
			admitted := name == "stored-weights" || name == "unsupported-architecture"
			if name == "stored-weights" {
				path = writeServePackedEmbeddingFixture(t, "mixed-qwen.gguf", false)
				// The startup peak is the shared
				// metalServeStartupPeakBytes resident bound: max(transformed,
				// raw payload) steady plus the 1 GiB staging scratch. Refuse one
				// byte below it and admit exactly at it.
				const rawPayload int64 = 3*256*144 + 3*210 + 3*144
				peak := max(wantStored, rawPayload) + (1 << 30)
				if err := refuseOversubscribedMetalGGUFForHost(path, peak, true); err != nil {
					t.Errorf("historical/transformed peak boundary refused: %v", err)
				}
				if err := refuseOversubscribedMetalGGUFForHost(path, peak-1, true); err == nil || !strings.Contains(err.Error(), "METAL_GGUF_PEAK_TOO_BIG") {
					t.Errorf("below the corrected weight peak bound: error=%v", err)
				}

			} else if name == "unsupported-architecture" {
				// Gemma2 is outside this estimator's qualified architecture set.
				// Keep a complete untied configuration and valid tensor directory;
				// only the qualified estimator refuses; admission keeps its prior policy.
				const rawPayload int64 = 3*256*144 + 3*210 + 3*144
				wantSteady = rawPayload
				path = writeServePackedEmbeddingFixture(t, "unqualified-gemma.gguf", false)
				b, err := os.ReadFile(path)
				if err != nil {
					t.Fatal(err)
				}
				b = bytes.ReplaceAll(b, []byte("qwen35"), []byte("gemma2"))
				if err := os.WriteFile(path, b, 0600); err != nil {
					t.Fatal(err)
				}
				ws, err := ggufload.OpenWeights(path)
				if err != nil {
					t.Fatal(err)
				}
				cfg, cfgErr := ws.File.Config()
				arm := resolveMetalServeLoadArm(ws)
				_, estimateErr := ws.EstimateQ4KLoadMemoryPlan()
				_ = ws.Close()
				if cfgErr != nil || cfg.ModelType != "gemma2" || cfg.TieWordEmbeddings || arm != serveLoadArmResidentQ4K {
					t.Fatalf("expected untied unqualified Gemma2 resident route: cfg=%+v arm=%q error=%v", cfg, arm, cfgErr)
				}
				if !errors.Is(estimateErr, ggufload.ErrQ4KLoadEstimateUnsupported) {
					t.Fatalf("qualified estimator error=%v, want typed unsupported", estimateErr)
				}
				// The unqualified route falls back to the raw
				// payload plan, still judged by the shared resident bound (raw
				// steady plus the 1 GiB staging scratch).
				if err := refuseOversubscribedMetalGGUFForHost(path, rawPayload+(1<<30), true); err != nil {
					t.Errorf("historical raw-payload peak boundary refused: %v", err)
				}
				if err := refuseOversubscribedMetalGGUFForHost(path, rawPayload+(1<<30)-1, true); err == nil || !strings.Contains(err.Error(), "METAL_GGUF_PEAK_TOO_BIG") {
					t.Errorf("below historical raw-payload peak: error=%v", err)
				}
			} else if name == "malformed-header" {
				if err := os.WriteFile(path, []byte("invalid GGUF header"), 0600); err != nil {
					t.Fatal(err)
				}
			}
			loads := 0
			release, err := loadLocalLauncherModelWithMetalLease(true, path, gpulease.Options{}, func() {
				loads++
				if !admitted {
					return
				}
				ledger := readLedger()
				if len(ledger) != 1 {
					t.Fatalf("startup ledger=%+v", ledger)
				}
				if ledger[0].Phase != "startup" || ledger[0].SteadyBytes != wantSteady || ledger[0].HeldBytes != ledger[0].StartupPeakBytes {
					t.Errorf("startup reservation=%+v, admission basis=%d", ledger[0], wantSteady)
				}
				if wantPeak := max(wantSteady*7/2, wantSteady+(1<<30)); ledger[0].StartupPeakBytes != wantPeak {
					t.Errorf("startup reservation=%d want historical peak/staging bound=%d", ledger[0].StartupPeakBytes, wantPeak)
				}
				if name == "unsupported-architecture" {
					return // Callback execution and ledger witness admission, not exact Gemma storage.
				}
				m, loadErr := ggufload.LoadModelQ4KProfileOptions(path, nil)
				if loadErr != nil {
					t.Fatal(loadErr)
				}
				defer m.CloseWeights()
				if got := m.ResidentReport().TotalResidentBytes; got != wantStored {
					t.Fatalf("actual loader=%d, independent=%d", got, wantStored)
				}
			})
			t.Cleanup(release)
			if admitted {
				if err != nil || loads != 1 {
					t.Fatalf("load count=%d error=%v", loads, err)
				}
				ledger := readLedger()
				if len(ledger) != 1 || ledger[0].Phase != "steady" || ledger[0].HeldBytes != wantSteady {
					t.Errorf("steady ledger=%+v, admission basis=%d", ledger, wantSteady)
				}
			} else {
				if err == nil || loads != 0 {
					t.Errorf("invalid model reached load: callbacks=%d error=%v", loads, err)
				}
				if name == "missing-file" && !errors.Is(err, os.ErrNotExist) {
					t.Errorf("missing-file estimate error=%v, want original path error", err)
				}
				// Check before calling returned cleanup: the failing seam itself must release.
				if ledger := readLedger(); len(ledger) != 0 {
					t.Errorf("failed estimate retained reservation: %+v", ledger)
				}
				reacquired, acquireErr := gpulease.Acquire(gpulease.Options{Path: leasePath, NoWait: true})
				if acquireErr != nil {
					t.Errorf("failed estimate retained lease: %v", acquireErr)
				} else {
					reacquired.Release()
				}
			}
			release()
			if ledger := readLedger(); len(ledger) != 0 {
				t.Errorf("cleanup retained ledger: %+v", ledger)
			}
			reacquired, acquireErr := gpulease.Acquire(gpulease.Options{Path: leasePath, NoWait: true})
			if acquireErr != nil {
				t.Fatal(acquireErr)
			}
			reacquired.Release()
		})
	}
}
