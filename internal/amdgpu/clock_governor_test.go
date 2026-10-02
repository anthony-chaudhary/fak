package amdgpu

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// Hardware migration preserves private generic clock acceptance; injected I/O is software evidence.
// fak-test:runtime integration est=2s lane=default
func TestHardwareClockGovernor(t *testing.T) {
	// 1. Platform-specific command synthesis for AMD DPM and NVIDIA NVML
	t.Run("PlatformCommandSynthesis_AMD_and_NVIDIA", func(t *testing.T) {
		// AMD DPM
		govAMD := NewHardwareClockGovernor(
			WithClockGovernorPlatform(PlatformAMD),
			WithClockGovernorTargetClocks(2200, 1000),
			WithClockGovernorSysfsDRMRoot("/sys/class/drm"),
			WithClockGovernorDeviceCards("card1", "card0"),
		)
		amdLockCmds := govAMD.SynthesizeLockCommands()
		if len(amdLockCmds) != 3 {
			t.Fatalf("expected 3 AMD lock commands, got %d: %v", len(amdLockCmds), amdLockCmds)
		}
		if !strings.Contains(amdLockCmds[0], "power_dpm_force_performance_level") || !strings.Contains(amdLockCmds[0], "manual") {
			t.Errorf("expected manual DPM in AMD command 0, got: %s", amdLockCmds[0])
		}
		if !strings.Contains(amdLockCmds[1], "pp_dpm_sclk") {
			t.Errorf("expected pp_dpm_sclk in AMD command 1, got: %s", amdLockCmds[1])
		}
		if !strings.Contains(amdLockCmds[2], "pp_dpm_mclk") {
			t.Errorf("expected pp_dpm_mclk in AMD command 2, got: %s", amdLockCmds[2])
		}

		amdUnlockCmds := govAMD.SynthesizeUnlockCommands()
		if len(amdUnlockCmds) != 1 || !strings.Contains(amdUnlockCmds[0], "auto") {
			t.Errorf("expected auto DPM in AMD unlock command, got: %v", amdUnlockCmds)
		}

		// NVIDIA NVML
		govNV := NewHardwareClockGovernor(
			WithClockGovernorPlatform(PlatformNVIDIA),
			WithClockGovernorTargetClocks(2100, 1215),
		)
		nvLockCmds := govNV.SynthesizeLockCommands()
		if len(nvLockCmds) != 3 {
			t.Fatalf("expected 3 NVIDIA lock commands, got %d: %v", len(nvLockCmds), nvLockCmds)
		}
		if nvLockCmds[0] != "nvidia-smi -pm 1" {
			t.Errorf("expected persistence mode command, got: %s", nvLockCmds[0])
		}
		if !strings.Contains(nvLockCmds[1], "nvidia-smi -lgc") || !strings.Contains(nvLockCmds[1], "2100") {
			t.Errorf("expected locked graphics clock command with 2100, got: %s", nvLockCmds[1])
		}
		if !strings.Contains(nvLockCmds[2], "nvidia-smi -lmc") || !strings.Contains(nvLockCmds[2], "1215") {
			t.Errorf("expected locked memory clock command with 1215, got: %s", nvLockCmds[2])
		}

		nvUnlockCmds := govNV.SynthesizeUnlockCommands()
		if len(nvUnlockCmds) != 3 {
			t.Fatalf("expected 3 NVIDIA unlock commands, got %d: %v", len(nvUnlockCmds), nvUnlockCmds)
		}
		if nvUnlockCmds[0] != "nvidia-smi -rgc" || nvUnlockCmds[1] != "nvidia-smi -rmc" || nvUnlockCmds[2] != "nvidia-smi -pm 0" {
			t.Errorf("unexpected NVIDIA unlock commands: %v", nvUnlockCmds)
		}

		// Apple Silicon QoS
		govApple := NewHardwareClockGovernor(
			WithClockGovernorPlatform(PlatformApple),
		)
		appleLockCmds := govApple.SynthesizeLockCommands()
		if len(appleLockCmds) != 1 || !strings.Contains(appleLockCmds[0], "taskpolicy") || !strings.Contains(appleLockCmds[0], "user_interactive") {
			t.Errorf("expected taskpolicy user_interactive, got: %v", appleLockCmds)
		}
		appleUnlockCmds := govApple.SynthesizeUnlockCommands()
		if len(appleUnlockCmds) != 1 || !strings.Contains(appleUnlockCmds[0], "default") {
			t.Errorf("expected taskpolicy default, got: %v", appleUnlockCmds)
		}
	})

	// 2. Anti-flapping hysteresis holds performance clock states through 5000ms of simulated inter-token idle
	t.Run("AntiFlappingHysteresis_5000ms_SimulatedIdle", func(t *testing.T) {
		virtualNow := time.Date(2026, 9, 8, 12, 0, 0, 0, time.UTC)
		nowFn := func() time.Time {
			return virtualNow
		}

		gov := NewHardwareClockGovernor(
			WithClockGovernorPlatform(PlatformNVIDIA),
			WithClockGovernorHysteresis(5000*time.Millisecond),
			WithClockGovernorTimeFunc(nowFn),
		)

		if gov.IsLocked() {
			t.Fatal("expected governor initially quiescent / unlocked")
		}

		// First token burst at t=0
		gov.RecordActivity()
		if !gov.IsLocked() {
			t.Fatal("expected governor to lock immediately upon token activity")
		}
		status := gov.Status()
		if status.State != "LOCKED_PERFORMANCE" {
			t.Errorf("expected LOCKED_PERFORMANCE state, got %s", status.State)
		}

		// Inter-token pause 1: 15ms (typical decode delay)
		virtualNow = virtualNow.Add(15 * time.Millisecond)
		if unlocked := gov.CheckHysteresis(virtualNow); unlocked {
			t.Fatal("anti-flapping hysteresis must NOT unlock after 15ms inter-token idle")
		}
		if !gov.IsLocked() {
			t.Fatal("expected performance clock state to be held locked after 15ms")
		}
		if gov.Status().State != "HYSTERESIS_HOLD" {
			t.Errorf("expected HYSTERESIS_HOLD state, got %s", gov.Status().State)
		}

		// Next token arrives at t=15ms
		gov.RecordActivity()
		if gov.Status().State != "LOCKED_PERFORMANCE" {
			t.Errorf("expected return to LOCKED_PERFORMANCE state, got %s", gov.Status().State)
		}

		// Inter-token pause 2: 50ms pause
		virtualNow = virtualNow.Add(50 * time.Millisecond)
		if unlocked := gov.CheckHysteresis(virtualNow); unlocked {
			t.Fatal("anti-flapping hysteresis must NOT unlock after 50ms idle")
		}
		if !gov.IsLocked() {
			t.Fatal("expected clock states held locked after 50ms")
		}

		// Token arrives at t=65ms
		gov.RecordActivity()

		// Inter-token pause 3: 4500ms extended pause (still within 5000ms window)
		virtualNow = virtualNow.Add(4500 * time.Millisecond)
		if unlocked := gov.CheckHysteresis(virtualNow); unlocked {
			t.Fatal("anti-flapping hysteresis must NOT unlock after 4500ms (< 5000ms window)")
		}
		if !gov.IsLocked() {
			t.Fatal("expected clock states held locked after 4500ms (< 5000ms window)")
		}

		// Advance 499ms more -> total 4999ms since last token (at t=65ms)
		virtualNow = virtualNow.Add(499 * time.Millisecond)
		if unlocked := gov.CheckHysteresis(virtualNow); unlocked {
			t.Fatal("anti-flapping hysteresis must NOT unlock at 4999ms (< 5000ms)")
		}
		if !gov.IsLocked() {
			t.Fatal("expected clock states still locked at 4999ms")
		}

		// Now advance past the 5000ms window (e.g. +2ms -> 5001ms idle)
		virtualNow = virtualNow.Add(2 * time.Millisecond)
		if unlocked := gov.CheckHysteresis(virtualNow); !unlocked {
			t.Fatal("expected governor to unlock after > 5000ms of idle without activity")
		}
		if gov.IsLocked() {
			t.Fatal("expected governor to be unlocked to quiescent state after 5000ms hysteresis expired")
		}
		if gov.Status().State != "QUIESCENT_IDLE" {
			t.Errorf("expected QUIESCENT_IDLE state, got %s", gov.Status().State)
		}
	})

	// 3. Sysfs State Management (AMD)
	t.Run("Sysfs_StateManagement_AMD", func(t *testing.T) {
		tempDir := t.TempDir()
		card1Dir := filepath.Join(tempDir, "card1", "device")
		if err := os.MkdirAll(card1Dir, 0755); err != nil {
			t.Fatalf("failed to create mock sysfs: %v", err)
		}

		dpmFile := filepath.Join(card1Dir, "power_dpm_force_performance_level")
		sclkFile := filepath.Join(card1Dir, "pp_dpm_sclk")
		mclkFile := filepath.Join(card1Dir, "pp_dpm_mclk")

		_ = os.WriteFile(dpmFile, []byte("auto\n"), 0644)
		_ = os.WriteFile(sclkFile, []byte("0: 600Mhz\n1: 1100Mhz\n2: 2200Mhz *\n"), 0644)
		_ = os.WriteFile(mclkFile, []byte("0: 400Mhz\n1: 800Mhz\n2: 1000Mhz *\n"), 0644)

		gov := NewHardwareClockGovernor(
			WithClockGovernorPlatform(PlatformAMD),
			WithClockGovernorSysfsDRMRoot(tempDir),
			WithClockGovernorDeviceCards("card1", "card0"),
			WithClockGovernorTargetClocks(2200, 1000),
			WithClockGovernorTargetMemMT(8000),
		)

		// Initial inspection
		core, mem, locked, err := gov.Inspect()
		if err != nil {
			t.Fatalf("Inspect failed: %v", err)
		}
		if core != 2200 || mem != 1000 || locked {
			t.Errorf("unexpected initial inspection: core=%d mem=%d locked=%v", core, mem, locked)
		}

		// Lock performance state
		if err := gov.Lock(); err != nil {
			t.Fatalf("Lock failed: %v", err)
		}
		if !gov.IsLocked() {
			t.Fatal("expected governor to be locked")
		}

		// Verify file contents on disk
		dpmData, _ := os.ReadFile(dpmFile)
		if strings.TrimSpace(string(dpmData)) != "manual" {
			t.Errorf("expected dpm file to be 'manual', got: %s", string(dpmData))
		}
		sclkData, _ := os.ReadFile(sclkFile)
		if strings.TrimSpace(string(sclkData)) != "2" {
			t.Errorf("expected sclk file to be '2', got: %s", string(sclkData))
		}
		mclkData, _ := os.ReadFile(mclkFile)
		if strings.TrimSpace(string(mclkData)) != "2" {
			t.Errorf("expected mclk file to be '2', got: %s", string(mclkData))
		}

		// Unlock
		if err := gov.Unlock(); err != nil {
			t.Fatalf("Unlock failed: %v", err)
		}
		if gov.IsLocked() {
			t.Fatal("expected governor to be unlocked")
		}
		dpmData, _ = os.ReadFile(dpmFile)
		if strings.TrimSpace(string(dpmData)) != "auto" {
			t.Errorf("expected dpm file reset to 'auto', got: %s", string(dpmData))
		}
	})

	// 4. NVML State Management (NVIDIA)
	t.Run("NVML_StateManagement_NVIDIA", func(t *testing.T) {
		var executedCmds []string
		mockRunner := func(name string, args ...string) (string, error) {
			executedCmds = append(executedCmds, name+" "+strings.Join(args, " "))
			return "", nil
		}

		gov := NewHardwareClockGovernor(
			WithClockGovernorPlatform(PlatformNVIDIA),
			WithClockGovernorTargetClocks(2100, 1215),
			WithClockGovernorCommandRunner(mockRunner),
		)

		// Lock
		if err := gov.Lock(); err != nil {
			t.Fatalf("Lock failed: %v", err)
		}
		if !gov.IsLocked() {
			t.Fatal("expected governor locked")
		}

		// Verify executed nvidia-smi commands
		expectedCmds := []string{
			"nvidia-smi -pm 1",
			"nvidia-smi -lgc 2100,2100",
			"nvidia-smi -lmc 1215",
		}
		if len(executedCmds) != 3 {
			t.Fatalf("expected 3 executed commands, got %d: %v", len(executedCmds), executedCmds)
		}
		for i, exp := range expectedCmds {
			if executedCmds[i] != exp {
				t.Errorf("expected command %d to be %q, got %q", i, exp, executedCmds[i])
			}
		}

		// Inspect while locked
		core, mem, locked, err := gov.Inspect()
		if err != nil {
			t.Fatalf("Inspect failed: %v", err)
		}
		if core != 2100 || mem != 1215 || !locked {
			t.Errorf("unexpected NVML inspection: core=%d mem=%d locked=%v", core, mem, locked)
		}

		// Unlock
		executedCmds = nil
		if err := gov.Unlock(); err != nil {
			t.Fatalf("Unlock failed: %v", err)
		}
		if gov.IsLocked() {
			t.Fatal("expected governor unlocked")
		}
		expectedUnlock := []string{
			"nvidia-smi -rgc",
			"nvidia-smi -rmc",
			"nvidia-smi -pm 0",
		}
		if len(executedCmds) != 3 {
			t.Fatalf("expected 3 unlock commands, got %d: %v", len(executedCmds), executedCmds)
		}
		for i, exp := range expectedUnlock {
			if executedCmds[i] != exp {
				t.Errorf("expected unlock command %d to be %q, got %q", i, exp, executedCmds[i])
			}
		}
	})

	// 5. Prometheus Metrics Export
	t.Run("Prometheus_Metrics_Export", func(t *testing.T) {
		gov := NewHardwareClockGovernor(
			WithClockGovernorPlatform(PlatformAMD),
			WithClockGovernorTargetClocks(2200, 1000),
		)
		_ = gov.Lock()
		gov.RecordThrottleEvent()

		metrics := gov.Metrics()
		if !strings.Contains(metrics, "hardware_clock_mhz") {
			t.Errorf("expected hardware_clock_mhz in metrics, got:\n%s", metrics)
		}
		if !strings.Contains(metrics, "memory_clock_mhz") {
			t.Errorf("expected memory_clock_mhz in metrics, got:\n%s", metrics)
		}
		if !strings.Contains(metrics, "throttle_events_total") {
			t.Errorf("expected throttle_events_total in metrics, got:\n%s", metrics)
		}
		if !strings.Contains(metrics, `throttle_events_total{platform="amd",device="card1"} 1`) {
			t.Errorf("expected throttle_events_total count of 1 in metrics, got:\n%s", metrics)
		}
	})

	// 6. Background Start and Stop
	t.Run("BackgroundDaemon_StartAndStop", func(t *testing.T) {
		gov := NewHardwareClockGovernor(
			WithClockGovernorPlatform(PlatformAMD),
			WithClockGovernorPollInterval(10*time.Millisecond),
		)
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		if err := gov.Start(ctx); err != nil {
			t.Fatalf("Start failed: %v", err)
		}
		// Second start fails
		if err := gov.Start(ctx); err == nil {
			t.Fatal("expected error on duplicate Start")
		}

		gov.RecordActivity()
		if !gov.IsLocked() {
			t.Fatal("expected locked after RecordActivity")
		}

		if err := gov.Stop(); err != nil {
			t.Fatalf("Stop failed: %v", err)
		}
		if gov.IsLocked() {
			t.Fatal("expected unlocked after Stop")
		}
	})
}

// cb11AssertReadersComplete releases and joins both goroutines before reporting
// a blocked state reader, including when the original implementation regresses.
func cb11AssertReadersComplete(t *testing.T, entered, release chan struct{}, operation <-chan error, read func()) {
	t.Helper()
	released := false
	defer func() {
		if !released {
			close(release)
		}
	}()
	select {
	case <-entered:
	case err := <-operation:
		t.Fatalf("I/O operation ended before blocked hook: %v", err)
	case <-time.After(time.Second):
		t.Fatal("I/O hook not reached")
	}
	readerDone := make(chan struct{})
	go func() { read(); close(readerDone) }()
	blocked := false
	select {
	case <-readerDone:
	case <-time.After(200 * time.Millisecond):
		blocked = true
	}
	close(release)
	released = true
	select {
	case err := <-operation:
		if err != nil {
			t.Errorf("operation: %v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("released operation did not join")
	}
	select {
	case <-readerDone:
	case <-time.After(time.Second):
		t.Fatal("state reader did not join after release")
	}
	if blocked {
		t.Error("state readers blocked behind hardware I/O")
	}
}

// fak-test:runtime integration est=2s lane=default
func TestCB11HardwareLockUnlockReadersDuringCommand(t *testing.T) {
	for _, platform := range []HardwarePlatform{PlatformNVIDIA, PlatformApple} {
		for _, unlock := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/unlock=%v", platform, unlock), func(t *testing.T) {
				entered := make(chan struct{})
				release := make(chan struct{})
				var once sync.Once
				runner := func(string, ...string) (string, error) { once.Do(func() { close(entered) }); <-release; return "", nil }
				g := NewHardwareClockGovernor(WithClockGovernorPlatform(platform))
				if unlock {
					g.commandRunner = func(string, ...string) (string, error) { return "", nil }
					if err := g.Lock(); err != nil {
						t.Fatal(err)
					}
				}
				g.commandRunner = runner
				operation := make(chan error, 1)
				go func() {
					if unlock {
						operation <- g.Unlock()
					} else {
						operation <- g.Lock()
					}
				}()
				cb11AssertReadersComplete(t, entered, release, operation, func() { _ = g.Status(); _ = g.Metrics(); _ = g.IsLocked() })
				if g.IsLocked() == unlock {
					t.Fatalf("final locked state=%v unlock=%v", g.IsLocked(), unlock)
				}
			})
		}
	}
}

// fak-test:runtime integration est=2s lane=default
func TestCB11AMDCommandFallbackDoesNotDeadlock(t *testing.T) {
	if os.Getenv("FAK_CB11_AMD_HELPER") == "1" {
		calls := 0
		g := NewHardwareClockGovernor(WithClockGovernorPlatform(PlatformAMD), WithClockGovernorSysfsDRMRoot(t.TempDir()), WithClockGovernorCommandRunner(func(string, ...string) (string, error) { calls++; return "", nil }))
		if err := g.Lock(); err != nil {
			t.Fatal(err)
		}
		if err := g.Unlock(); err != nil {
			t.Fatal(err)
		}
		if calls < 4 {
			t.Fatalf("AMD command fallback not invoked: %d", calls)
		}
		return
	}
	// Isolate an irrecoverable recursive mutex regression; CommandContext kills
	// and Wait joins this child rather than leaking a permanently blocked goroutine.
	bin, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, bin, "-test.run=^TestCB11AMDCommandFallbackDoesNotDeadlock$", "-test.timeout=5s")
	cmd.Env = append(os.Environ(), "FAK_CB11_AMD_HELPER=1", "GORACE=atexit_sleep_ms=0")
	if raw, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("AMD command synthesis deadlocked or failed: %v %s", err, raw)
	}
}

// fak-test:runtime integration est=1s lane=default
func TestCB11AMDFileWriterAllowsReaders(t *testing.T) {
	for _, unlock := range []bool{false, true} {
		t.Run(fmt.Sprintf("unlock=%v", unlock), func(t *testing.T) {
			root := t.TempDir()
			device := filepath.Join(root, "card0", "device")
			if err := os.MkdirAll(device, 0755); err != nil {
				t.Fatal(err)
			}
			g := NewHardwareClockGovernor(WithClockGovernorPlatform(PlatformAMD), WithClockGovernorSysfsDRMRoot(root), WithClockGovernorDeviceCards("card0", "card1"))
			if unlock {
				if err := g.Lock(); err != nil {
					t.Fatal(err)
				}
			}
			entered := make(chan struct{})
			release := make(chan struct{})
			var once sync.Once
			g.fileWriter = func(path string, data []byte, mode os.FileMode) error {
				once.Do(func() { close(entered); <-release })
				return os.WriteFile(path, data, mode)
			}
			done := make(chan error, 1)
			go func() {
				if unlock {
					done <- g.Unlock()
				} else {
					done <- g.Lock()
				}
			}()
			cb11AssertReadersComplete(t, entered, release, done, func() { _ = g.Status(); _ = g.Metrics() })
		})
	}
}

// fak-test:runtime integration est=1s lane=default
func TestCB11LatestActivitySurvivesBlockedUnlock(t *testing.T) {
	base := time.Date(2026, 9, 30, 0, 0, 0, 0, time.UTC)
	var stamp atomic.Int64
	stamp.Store(base.UnixNano())
	g := NewHardwareClockGovernor(WithClockGovernorPlatform(PlatformNVIDIA), WithClockGovernorHysteresis(time.Second), WithClockGovernorTimeFunc(func() time.Time { return time.Unix(0, stamp.Load()) }), WithClockGovernorCommandRunner(func(string, ...string) (string, error) { return "", nil }))
	if err := g.Lock(); err != nil {
		t.Fatal(err)
	}
	g.RecordActivity()
	entered := make(chan struct{})
	release := make(chan struct{})
	var once sync.Once
	g.commandRunner = func(string, ...string) (string, error) { once.Do(func() { close(entered); <-release }); return "", nil }
	done := make(chan bool, 1)
	go func() { done <- g.CheckHysteresis(base.Add(2 * time.Second)) }()
	select {
	case <-entered:
	case <-time.After(time.Second):
		close(release)
		<-done
		t.Fatal("hysteresis unlock I/O did not start")
	}
	stamp.Store(base.Add(2 * time.Second).UnixNano())
	activity := make(chan struct{})
	go func() { g.RecordActivity(); close(activity) }()
	statusDone := make(chan HardwareGovernorStatus, 1)
	go func() { statusDone <- g.Status() }()
	blocked := false
	select {
	case <-statusDone:
	case <-time.After(200 * time.Millisecond):
		blocked = true
	}
	close(release)
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("hysteresis unlock failed to join")
	}
	select {
	case <-activity:
	case <-time.After(time.Second):
		t.Fatal("latest activity failed to join")
	}
	if blocked {
		select {
		case <-statusDone:
		case <-time.After(time.Second):
			t.Fatal("status failed to join")
		}
		t.Error("latest-activity state reader blocked behind unlock")
	}
	if !g.IsLocked() {
		t.Error("older unlock overwrote newer serving activity")
	}
	if st := g.Status(); st.LastActivityAgoMs != 0 || st.State != "LOCKED_PERFORMANCE" {
		t.Fatalf("latest activity lost: %+v", st)
	}
}
