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
	if isServeVulkan(nil) {
		t.Error("nil serve flags should not be vulkan")
	}

	backendName := "vulkan"
	sf := &serveFlags{backendName: &backendName}
	if !isServeVulkan(sf) {
		t.Error("sf with backendName=vulkan should be vulkan")
	}

	upperBackend := "VULKAN"
	sfUpper := &serveFlags{backendName: &upperBackend}
	if isServeVulkan(sfUpper) {
		t.Error("sf with backendName=VULKAN should not be exact vulkan")
	}

	spacedBackend := " vulkan "
	sfSpaced := &serveFlags{backendName: &spacedBackend}
	if isServeVulkan(sfSpaced) {
		t.Error("sf with whitespace-padded backendName should not be exact vulkan")
	}

	otherBackend := "metal"
	sfOther := &serveFlags{backendName: &otherBackend}
	if isServeVulkan(sfOther) {
		t.Error("sf with backendName=metal should not be vulkan")
	}

	emptyBackend := ""
	sfEmpty := &serveFlags{backendName: &emptyBackend}
	if isServeVulkan(sfEmpty) {
		t.Error("empty backendName should not derive Vulkan identity from runtime state")
	}

	baseURL := "http://127.0.0.1:8080/v1"
	sfProxy := &serveFlags{backendName: &backendName, baseURL: &baseURL}
	if isServeVulkan(sfProxy) {
		t.Error("sf with baseURL should not be vulkan (proxy mode is lease-free)")
	}

	if isServeVulkan(&serveFlags{}) {
		t.Error("nil backendName should not be vulkan")
	}
}

func TestLoadServeModelWithVulkanLeaseUsesExactBackendAndGGUF(t *testing.T) {
	tests := []struct {
		name      string
		backend   string
		gguf      string
		model     string
		baseURL   string
		wantLease bool
	}{
		{name: "exact local Vulkan GGUF", backend: "vulkan", gguf: "model.gguf", model: "mock", wantLease: true},
		{name: "empty GGUF never substitutes default model", backend: "vulkan", model: "mock"},
		{name: "uppercase backend rejected", backend: "VULKAN", gguf: "model.gguf"},
		{name: "whitespace backend rejected", backend: " vulkan ", gguf: "model.gguf"},
		{name: "runtime identity unavailable without exact flag", gguf: "model.gguf"},
		{name: "proxy bypass", backend: "vulkan", gguf: "model.gguf", baseURL: "http://127.0.0.1:8080/v1"},
		{name: "CPU reference bypass", gguf: "model.gguf"},
		{name: "exact nonempty whitespace GGUF admits before load validation", backend: "vulkan", gguf: " ", wantLease: true},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "gpu.lease")
			backend, gguf, model, baseURL := tc.backend, tc.gguf, tc.model, tc.baseURL
			sf := &serveFlags{backendName: &backend, ggufPath: &gguf, model: &model, baseURL: &baseURL}
			loads := 0

			if !tc.wantLease {
				held, err := gpulease.Acquire(gpulease.Options{Path: path, NoWait: true})
				if err != nil {
					t.Fatalf("hold unrelated lease: %v", err)
				}
				defer held.Release()
			}

			release, err := loadServeModelWithVulkanLease(sf, gpulease.Options{Path: path}, func() { loads++ })
			if err != nil {
				t.Fatalf("serve admission: %v", err)
			}
			defer release()
			if loads != 1 {
				t.Fatalf("load callback calls = %d, want 1", loads)
			}

			if tc.wantLease {
				if competing, err := gpulease.Acquire(gpulease.Options{Path: path, NoWait: true}); !errors.Is(err, gpulease.ErrBusy) {
					if err == nil {
						competing.Release()
					}
					t.Fatalf("exact Vulkan GGUF did not retain lease: %v", err)
				}
			}
		})
	}
}

func TestLoadLocalLauncherModelWithMetalLeaseRefusesExpandingQ8LoadOn36GBHost(t *testing.T) {
	resDir := filepath.Join(t.TempDir(), "reservations")
	leasePath := filepath.Join(t.TempDir(), "gpu.lease")
	t.Setenv("FAK_RESERVATION_DIR", resDir)
	t.Setenv("FAK_GPU_LEASE", leasePath)
	t.Setenv("FAK_ADMISSION_POLICY", "dev")

	udPath := filepath.Join(t.TempDir(), "qwen38-27b-ud-q2kxl.gguf")
	writeSynth27BGGUF(t, udPath, true)

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
}
