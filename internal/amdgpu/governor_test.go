package amdgpu

import (
	"bytes"
	"encoding/json"
	"errors"
	"os"
	"path"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

// MockFS is an in-memory filesystem for testing sysfs and proc interactions.
type MockFS struct {
	mu    sync.RWMutex
	files map[string][]byte
}

func NewMockFS() *MockFS {
	return &MockFS{files: make(map[string][]byte)}
}

func (m *MockFS) ReadFile(p string) ([]byte, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	clean := filepath.ToSlash(filepath.Clean(p))
	data, ok := m.files[clean]
	if !ok {
		return nil, os.ErrNotExist
	}
	cp := make([]byte, len(data))
	copy(cp, data)
	return cp, nil
}

func (m *MockFS) WriteFile(p string, d []byte, perm os.FileMode) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	clean := filepath.ToSlash(filepath.Clean(p))
	cp := make([]byte, len(d))
	copy(cp, d)
	m.files[clean] = cp
	return nil
}

func (m *MockFS) Glob(pattern string) ([]string, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	cleanPat := filepath.ToSlash(pattern)
	var matches []string
	for k := range m.files {
		matched, err := path.Match(cleanPat, k)
		if err == nil && matched {
			matches = append(matches, k)
		}
	}
	return matches, nil
}

func (m *MockFS) Stat(p string) (os.FileInfo, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	clean := filepath.ToSlash(filepath.Clean(p))
	data, ok := m.files[clean]
	if !ok {
		return nil, os.ErrNotExist
	}
	return mockFileInfo{name: filepath.Base(clean), size: int64(len(data))}, nil
}

type mockFileInfo struct {
	name string
	size int64
}

func (m mockFileInfo) Name() string       { return m.name }
func (m mockFileInfo) Size() int64        { return m.size }
func (m mockFileInfo) Mode() os.FileMode  { return 0644 }
func (m mockFileInfo) ModTime() time.Time { return time.Time{} }
func (m mockFileInfo) IsDir() bool        { return false }
func (m mockFileInfo) Sys() any           { return nil }

const (
	gibBytes uint64 = 1024 * 1024 * 1024
)

func TestCalculateRecommendedPagesLimit(t *testing.T) {
	ram64 := 64 * gibBytes
	bytes90 := uint64(float64(ram64) * 0.90)
	bytes875 := uint64(float64(ram64) * 0.875)

	tests := []struct {
		name        string
		totalRAM    uint64
		pageSize    uint64
		ratio       float64
		wantPages   uint64
		wantBytes   uint64
		minRatio    float64
		maxRatio    float64
		expectError bool
	}{
		{
			name:      "128 GiB system (q38rocm APU expansion: ~120 GiB)",
			totalRAM:  128 * gibBytes,
			pageSize:  DefaultPageSize,
			ratio:     0,
			wantPages: 31457280, // 120 GiB / 4096
			wantBytes: 120 * gibBytes,
			minRatio:  0.937,
			maxRatio:  0.938,
		},
		{
			name:      "64 GiB system (q38rocm APU expansion: ~56 GiB)",
			totalRAM:  64 * gibBytes,
			pageSize:  DefaultPageSize,
			ratio:     0,
			wantPages: 14680064, // 56 GiB / 4096
			wantBytes: 56 * gibBytes,
			minRatio:  0.874,
			maxRatio:  0.876,
		},
		{
			name:      "32 GiB system (reserve 6 GiB: ~26 GiB)",
			totalRAM:  32 * gibBytes,
			pageSize:  DefaultPageSize,
			ratio:     0,
			wantPages: 6815744, // 26 GiB / 4096
			wantBytes: 26 * gibBytes,
			minRatio:  0.812,
			maxRatio:  0.813,
		},
		{
			name:      "16 GiB system (reserve 4 GiB: ~12 GiB)",
			totalRAM:  16 * gibBytes,
			pageSize:  DefaultPageSize,
			ratio:     0,
			wantPages: 31457280 / 10, // 12 GiB / 4096 = 3145728
			wantBytes: 12 * gibBytes,
			minRatio:  0.749,
			maxRatio:  0.751,
		},
		{
			name:      "Custom ratio 90% on 64 GiB",
			totalRAM:  ram64,
			pageSize:  DefaultPageSize,
			ratio:     0.90,
			wantBytes: bytes90,
			wantPages: bytes90 / DefaultPageSize,
			minRatio:  0.899,
			maxRatio:  0.901,
		},
		{
			name:      "Custom ratio 87.5% on 64 GiB",
			totalRAM:  ram64,
			pageSize:  DefaultPageSize,
			ratio:     0.875,
			wantBytes: bytes875,
			wantPages: bytes875 / DefaultPageSize,
			minRatio:  0.874,
			maxRatio:  0.876,
		},
		{
			name:        "Unsafe ratio > 95% rejected",
			totalRAM:    64 * gibBytes,
			pageSize:    DefaultPageSize,
			ratio:       0.98,
			expectError: true,
		},
		{
			name:        "Negative ratio rejected",
			totalRAM:    64 * gibBytes,
			pageSize:    DefaultPageSize,
			ratio:       -0.5,
			expectError: true,
		},
		{
			name:        "Zero total RAM rejected",
			totalRAM:    0,
			pageSize:    DefaultPageSize,
			ratio:       0,
			expectError: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			pages, bytesLimit, ratio, err := CalculateRecommendedPagesLimit(tc.totalRAM, tc.pageSize, tc.ratio)
			if tc.expectError {
				if err == nil {
					t.Fatalf("expected error, got pages=%d, bytes=%d", pages, bytesLimit)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if tc.wantPages > 0 && pages != tc.wantPages {
				t.Errorf("pages = %d, want %d", pages, tc.wantPages)
			}
			if tc.wantBytes > 0 && bytesLimit != tc.wantBytes {
				t.Errorf("bytes = %d, want %d", bytesLimit, tc.wantBytes)
			}
			if ratio < tc.minRatio || ratio > tc.maxRatio {
				t.Errorf("ratio = %.4f, want between %.4f and %.4f", ratio, tc.minRatio, tc.maxRatio)
			}
		})
	}
}

func TestRecommendedOSReserveBytes(t *testing.T) {
	if got := RecommendedOSReserveBytes(128 * gibBytes); got != 8*gibBytes {
		t.Errorf("128GB reserve = %d, want %d", got, 8*gibBytes)
	}
	if got := RecommendedOSReserveBytes(64 * gibBytes); got != 8*gibBytes {
		t.Errorf("64GB reserve = %d, want %d", got, 8*gibBytes)
	}
	if got := RecommendedOSReserveBytes(32 * gibBytes); got != 6*gibBytes {
		t.Errorf("32GB reserve = %d, want %d", got, 6*gibBytes)
	}
	if got := RecommendedOSReserveBytes(16 * gibBytes); got != 4*gibBytes {
		t.Errorf("16GB reserve = %d, want %d", got, 4*gibBytes)
	}
	if got := RecommendedOSReserveBytes(8 * gibBytes); got != 2*gibBytes {
		t.Errorf("8GB reserve = %d, want %d", got, 2*gibBytes)
	}
	if got := RecommendedOSReserveBytes(2 * gibBytes); got != 1*gibBytes {
		t.Errorf("2GB reserve = %d, want %d", got, 1*gibBytes)
	}
}

func TestParseMemTotalFromProcMeminfo(t *testing.T) {
	content := `MemTotal:       65675200 kB
MemFree:        45123000 kB
MemAvailable:   58000000 kB
Buffers:         1200000 kB
Cached:          8500000 kB
`
	got, err := ParseMemTotalFromProcMeminfo(content)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	want := uint64(65675200) * 1024
	if got != want {
		t.Fatalf("got %d bytes, want %d", got, want)
	}

	// Missing MemTotal
	_, err = ParseMemTotalFromProcMeminfo("MemFree: 4000 kB\n")
	if err == nil {
		t.Fatal("expected error for missing MemTotal")
	}

	// Malformed MemTotal
	_, err = ParseMemTotalFromProcMeminfo("MemTotal: not_a_number kB\n")
	if err == nil {
		t.Fatal("expected error for malformed MemTotal")
	}
}

func TestValidateGovernorConfig(t *testing.T) {
	for _, lvl := range []string{"high", "auto", "low", "manual", "profile_peak"} {
		if err := ValidateGovernorConfig(GovernorConfig{TargetDPMLevel: lvl}); err != nil {
			t.Errorf("level %q should be valid, got: %v", lvl, err)
		}
	}

	if err := ValidateGovernorConfig(GovernorConfig{TargetDPMLevel: "turbo"}); err == nil {
		t.Error("level 'turbo' should be invalid")
	}

	if err := ValidateGovernorConfig(GovernorConfig{TargetDPMLevel: "high", TargetRatio: 0.98}); err == nil {
		t.Error("ratio 0.98 should be rejected (> 0.95)")
	}

	if err := ValidateGovernorConfig(GovernorConfig{TargetDPMLevel: "high", TargetRatio: -0.1}); err == nil {
		t.Error("negative ratio should be rejected")
	}
}

func setupMockLinuxSysfs() *MockFS {
	m := NewMockFS()
	m.files["/proc/meminfo"] = []byte("MemTotal:       67108864 kB\n") // 64 GiB = 68719476736 bytes
	m.files["/sys/class/drm/card0/device/power_dpm_force_performance_level"] = []byte("auto\n")
	m.files["/sys/module/ttm/parameters/pages_limit"] = []byte("0\n") // default 50%
	return m
}

func TestLinuxPlanDryRunNeedsUpdate(t *testing.T) {
	mockFS := setupMockLinuxSysfs()

	cfg := GovernorConfig{
		TargetDPMLevel: "high",
		SysfsRoot:      "/sys",
		ProcRoot:       "/proc",
	}

	plan, err := BuildPlan(cfg,
		WithFS(mockFS),
		WithGOOS("linux"),
		WithElevation(func() bool { return false }),
	)
	if err != nil {
		t.Fatalf("BuildPlan failed: %v", err)
	}

	if plan.Ready {
		t.Fatal("plan should NOT be ready when updates are pending")
	}
	if !plan.NeedsElevation {
		t.Fatal("plan should require elevation when non-root")
	}
	if plan.RecommendedRunAs != "sudo fak-dev amd-setup --apply" {
		t.Fatalf("recommended run as = %q, want %q", plan.RecommendedRunAs, "sudo fak-dev amd-setup --apply")
	}
	if len(plan.Cards) != 1 {
		t.Fatalf("expected 1 card, got %d", len(plan.Cards))
	}
	if plan.Cards[0].CurrentLevel != "auto" || plan.Cards[0].TargetLevel != "high" || !plan.Cards[0].NeedsUpdate {
		t.Fatalf("unexpected card status: %+v", plan.Cards[0])
	}

	if !plan.TTM.Available {
		t.Fatal("TTM should be available")
	}
	if !plan.TTM.Default50PercentMode {
		t.Fatal("TTM should be recognized as default 50% mode (current pages_limit = 0)")
	}
	if plan.TTM.CurrentLimitGiB != 32.0 {
		t.Fatalf("current limit GiB = %v, want 32.0", plan.TTM.CurrentLimitGiB)
	}
	if plan.TTM.TargetLimitGiB != 56.0 {
		t.Fatalf("target limit GiB = %v, want 56.0", plan.TTM.TargetLimitGiB)
	}
	if plan.TTM.TargetPagesLimit != 14680064 {
		t.Fatalf("target pages = %d, want 14680064", plan.TTM.TargetPagesLimit)
	}

	if len(plan.Actions) != 2 {
		t.Fatalf("expected 2 actions, got %d", len(plan.Actions))
	}

	// Verify JSON output
	data, err := plan.JSON()
	if err != nil {
		t.Fatalf("JSON marshal error: %v", err)
	}
	var decoded GovernorReport
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("JSON unmarshal error: %v", err)
	}
	if decoded.Platform != "linux" || len(decoded.Actions) != 2 {
		t.Fatalf("decoded JSON mismatch: %+v", decoded)
	}
}

func TestLinuxPlanAlreadyAtTarget(t *testing.T) {
	mockFS := NewMockFS()
	mockFS.files["/proc/meminfo"] = []byte("MemTotal:       67108864 kB\n")
	mockFS.files["/sys/class/drm/card0/device/power_dpm_force_performance_level"] = []byte("high\n")
	mockFS.files["/sys/module/ttm/parameters/pages_limit"] = []byte("14680064\n")

	cfg := GovernorConfig{
		TargetDPMLevel: "high",
		SysfsRoot:      "/sys",
		ProcRoot:       "/proc",
	}

	plan, err := BuildPlan(cfg,
		WithFS(mockFS),
		WithGOOS("linux"),
		WithElevation(func() bool { return false }),
	)
	if err != nil {
		t.Fatalf("BuildPlan failed: %v", err)
	}

	if !plan.Ready {
		t.Fatal("plan should be ready when all parameters are at target")
	}
	if plan.NeedsElevation {
		t.Fatal("elevation should not be needed when no actions pending")
	}
	if len(plan.Actions) != 0 {
		t.Fatalf("expected 0 actions, got %d", len(plan.Actions))
	}
}

func TestLinuxApplyRefusalWithoutElevation(t *testing.T) {
	mockFS := setupMockLinuxSysfs()

	cfg := GovernorConfig{
		TargetDPMLevel: "high",
		SysfsRoot:      "/sys",
		ProcRoot:       "/proc",
	}

	plan, err := BuildPlan(cfg,
		WithFS(mockFS),
		WithGOOS("linux"),
		WithElevation(func() bool { return false }),
	)
	if err != nil {
		t.Fatalf("BuildPlan failed: %v", err)
	}

	result, err := Apply(plan,
		WithFS(mockFS),
		WithElevation(func() bool { return false }),
	)
	if err == nil {
		t.Fatal("Apply should have failed without elevation")
	}
	if !strings.Contains(err.Error(), "sudo fak-dev amd-setup --apply") {
		t.Fatalf("expected error to contain %q, got %q", "sudo fak-dev amd-setup --apply", err.Error())
	}
	if result.Success {
		t.Fatal("result.Success should be false")
	}
	if result.AppliedCount != 0 {
		t.Fatalf("applied count = %d, want 0", result.AppliedCount)
	}

	// Verify files were not modified
	if string(mockFS.files["/sys/class/drm/card0/device/power_dpm_force_performance_level"]) != "auto\n" {
		t.Fatal("card0 power_dpm should remain unmodified")
	}
	if string(mockFS.files["/sys/module/ttm/parameters/pages_limit"]) != "0\n" {
		t.Fatal("ttm pages_limit should remain unmodified")
	}
}

func TestLinuxApplySuccessWithElevation(t *testing.T) {
	mockFS := setupMockLinuxSysfs()

	cfg := GovernorConfig{
		TargetDPMLevel: "high",
		SysfsRoot:      "/sys",
		ProcRoot:       "/proc",
	}

	plan, err := BuildPlan(cfg,
		WithFS(mockFS),
		WithGOOS("linux"),
		WithElevation(func() bool { return true }),
	)
	if err != nil {
		t.Fatalf("BuildPlan failed: %v", err)
	}

	result, err := Apply(plan,
		WithFS(mockFS),
		WithElevation(func() bool { return true }),
	)
	if err != nil {
		t.Fatalf("Apply failed: %v", err)
	}
	if !result.Success {
		t.Fatal("Apply result.Success should be true")
	}
	if result.AppliedCount != 2 {
		t.Fatalf("applied count = %d, want 2", result.AppliedCount)
	}
	if !result.Verified {
		t.Fatal("result.Verified should be true")
	}

	// Verify files on mock FS
	if strings.TrimSpace(string(mockFS.files["/sys/class/drm/card0/device/power_dpm_force_performance_level"])) != "high" {
		t.Fatalf("power_dpm was not updated: %s", string(mockFS.files["/sys/class/drm/card0/device/power_dpm_force_performance_level"]))
	}
	if strings.TrimSpace(string(mockFS.files["/sys/module/ttm/parameters/pages_limit"])) != "14680064" {
		t.Fatalf("pages_limit was not updated: %s", string(mockFS.files["/sys/module/ttm/parameters/pages_limit"]))
	}

	// Subsequent plan should show READY
	replan, err := BuildPlan(cfg,
		WithFS(mockFS),
		WithGOOS("linux"),
		WithElevation(func() bool { return true }),
	)
	if err != nil {
		t.Fatalf("replan failed: %v", err)
	}
	if !replan.Ready {
		t.Fatal("replan should be READY after apply")
	}
	if len(replan.Actions) != 0 {
		t.Fatalf("replan actions = %d, want 0", len(replan.Actions))
	}
}

// faultyWriteFS simulates a sysfs node that rejects a write (read-back mismatch).
type faultyWriteFS struct {
	*MockFS
	faultyPath string
}

func (f *faultyWriteFS) WriteFile(p string, d []byte, perm os.FileMode) error {
	if filepath.ToSlash(filepath.Clean(p)) == f.faultyPath {
		// Kernel silently ignores or sets different value
		return nil
	}
	return f.MockFS.WriteFile(p, d, perm)
}

func TestLinuxApplyReadBackVerificationFailure(t *testing.T) {
	mockFS := setupMockLinuxSysfs()
	faulty := &faultyWriteFS{
		MockFS:     mockFS,
		faultyPath: "/sys/module/ttm/parameters/pages_limit",
	}

	cfg := GovernorConfig{
		TargetDPMLevel: "high",
		SysfsRoot:      "/sys",
		ProcRoot:       "/proc",
	}

	plan, err := BuildPlan(cfg,
		WithFS(faulty),
		WithGOOS("linux"),
		WithElevation(func() bool { return true }),
	)
	if err != nil {
		t.Fatalf("BuildPlan failed: %v", err)
	}

	result, err := Apply(plan,
		WithFS(faulty),
		WithElevation(func() bool { return true }),
	)
	if err == nil {
		t.Fatal("expected error due to read-back mismatch")
	}
	if result.Success {
		t.Fatal("result.Success should be false on read-back mismatch")
	}
	if result.Verified {
		t.Fatal("result.Verified should be false")
	}
	if len(result.Errors) == 0 {
		t.Fatal("expected error details in result.Errors")
	}
	if !strings.Contains(err.Error(), "rollback succeeded") {
		t.Fatalf("expected error to indicate rollback succeeded, got: %v", err)
	}

	// Verify that card0 was restored to its original value "auto" via rollback
	card0Val := strings.TrimSpace(string(mockFS.files["/sys/class/drm/card0/device/power_dpm_force_performance_level"]))
	if card0Val != "auto" {
		t.Fatalf("card0 sysfs file not restored to original value; got %q, want %q", card0Val, "auto")
	}
}

func TestMultipleCardsDPM(t *testing.T) {
	mockFS := NewMockFS()
	mockFS.files["/proc/meminfo"] = []byte("MemTotal:       67108864 kB\n")
	mockFS.files["/sys/class/drm/card0/device/power_dpm_force_performance_level"] = []byte("auto\n")
	mockFS.files["/sys/class/drm/card1/device/power_dpm_force_performance_level"] = []byte("low\n")
	mockFS.files["/sys/module/ttm/parameters/pages_limit"] = []byte("14680064\n") // TTM already at target

	cfg := GovernorConfig{
		TargetDPMLevel: "high",
		SysfsRoot:      "/sys",
		ProcRoot:       "/proc",
	}

	plan, err := BuildPlan(cfg,
		WithFS(mockFS),
		WithGOOS("linux"),
		WithElevation(func() bool { return true }),
	)
	if err != nil {
		t.Fatalf("BuildPlan failed: %v", err)
	}

	if len(plan.Cards) != 2 {
		t.Fatalf("expected 2 cards, got %d", len(plan.Cards))
	}
	if len(plan.Actions) != 2 {
		t.Fatalf("expected 2 card actions, got %d", len(plan.Actions))
	}

	result, err := Apply(plan, WithFS(mockFS), WithElevation(func() bool { return true }))
	if err != nil {
		t.Fatalf("Apply failed: %v", err)
	}
	if result.AppliedCount != 2 {
		t.Fatalf("applied count = %d, want 2", result.AppliedCount)
	}

	if strings.TrimSpace(string(mockFS.files["/sys/class/drm/card0/device/power_dpm_force_performance_level"])) != "high" {
		t.Error("card0 was not set to high")
	}
	if strings.TrimSpace(string(mockFS.files["/sys/class/drm/card1/device/power_dpm_force_performance_level"])) != "high" {
		t.Error("card1 was not set to high")
	}
}

func TestWindowsStatusReporting(t *testing.T) {
	runner := fakeRunner(true, goodAMD, "")

	cfg := GovernorConfig{}
	plan, err := BuildPlan(cfg,
		WithGOOS("windows"),
		WithRunner(runner),
	)
	if err != nil {
		t.Fatalf("BuildPlan failed: %v", err)
	}

	if plan.Platform != "windows" {
		t.Fatalf("platform = %q, want windows", plan.Platform)
	}
	if plan.Windows == nil {
		t.Fatal("expected non-nil Windows status")
	}
	if !plan.Windows.Available {
		t.Fatalf("Windows status unavailable: %s", plan.Windows.Error)
	}
	if plan.Windows.DeviceName != "AMD Radeon RX 7600" {
		t.Errorf("device name = %q, want AMD Radeon RX 7600", plan.Windows.DeviceName)
	}
	if plan.Windows.DriverVersion != "32.0.31019.2002" { //boundarylint:ignore CHANGE_DETECTOR_TEST driver version string invariant
		t.Errorf("driver version = %q", plan.Windows.DriverVersion)
	}
	if plan.Windows.BusiestUnit != "Graphics" {
		t.Errorf("busiest unit = %q", plan.Windows.BusiestUnit)
	}
	if !strings.Contains(plan.Windows.PowerClockProfile, "Windows WDDM driver") {
		t.Errorf("unexpected PowerClockProfile: %s", plan.Windows.PowerClockProfile)
	}
	if !strings.Contains(plan.Windows.MemoryTuningNote, "Windows WDDM") {
		t.Errorf("unexpected MemoryTuningNote: %s", plan.Windows.MemoryTuningNote)
	}
	if !plan.Ready {
		t.Fatal("Windows should be marked ready by default as sysfs is Linux-only")
	}
}

func TestWindowsStatusUnavailable(t *testing.T) {
	runner := fakeRunner(false, "", "No AMD GPU found")

	plan, err := BuildPlan(GovernorConfig{},
		WithGOOS("windows"),
		WithRunner(runner),
	)
	if err != nil {
		t.Fatalf("BuildPlan failed: %v", err)
	}
	if plan.Windows == nil || plan.Windows.Available {
		t.Fatal("expected Windows status to be unavailable")
	}
	if len(plan.Warnings) == 0 {
		t.Fatal("expected warning when probe is unavailable")
	}
}

func TestCLIRunner(t *testing.T) {
	mockFS := setupMockLinuxSysfs()

	t.Run("dry-run text output", func(t *testing.T) {
		var stdout, stderr bytes.Buffer
		code := RunCLI(&stdout, &stderr, []string{"--dry-run", "--sysfs-root", "/sys", "--proc-root", "/proc"},
			WithFS(mockFS),
			WithGOOS("linux"),
			WithElevation(func() bool { return false }),
		)
		if code != 0 {
			t.Fatalf("code = %d, want 0; stderr: %s", code, stderr.String())
		}
		out := stdout.String()
		if !strings.Contains(out, "AMD GPU Hardware Governor & TTM Memory Ceiling Plan:") {
			t.Errorf("missing header in output:\n%s", out)
		}
		if !strings.Contains(out, "power_dpm_force_performance_level") {
			t.Errorf("missing card action in output:\n%s", out)
		}
		if !strings.Contains(out, "Elevation Required:") {
			t.Errorf("missing elevation notice in output:\n%s", out)
		}
	})

	t.Run("dry-run json output", func(t *testing.T) {
		var stdout, stderr bytes.Buffer
		code := RunCLI(&stdout, &stderr, []string{"--json", "--sysfs-root", "/sys", "--proc-root", "/proc"},
			WithFS(mockFS),
			WithGOOS("linux"),
			WithElevation(func() bool { return false }),
		)
		if code != 0 {
			t.Fatalf("code = %d, want 0; stderr: %s", code, stderr.String())
		}
		var parsed GovernorReport
		if err := json.Unmarshal(stdout.Bytes(), &parsed); err != nil {
			t.Fatalf("failed to parse JSON stdout: %v\n%s", err, stdout.String())
		}
		if len(parsed.Actions) != 2 {
			t.Fatalf("expected 2 actions in parsed JSON, got %d", len(parsed.Actions))
		}
	})

	t.Run("apply without elevation fails", func(t *testing.T) {
		var stdout, stderr bytes.Buffer
		code := RunCLI(&stdout, &stderr, []string{"--apply", "--sysfs-root", "/sys", "--proc-root", "/proc"},
			WithFS(mockFS),
			WithGOOS("linux"),
			WithElevation(func() bool { return false }),
		)
		if code != 1 {
			t.Fatalf("code = %d, want 1; stderr: %s", code, stderr.String())
		}
		if !strings.Contains(stderr.String(), "elevation required") {
			t.Fatalf("stderr missing elevation error: %s", stderr.String())
		}
	})

	t.Run("apply with elevation succeeds", func(t *testing.T) {
		freshFS := setupMockLinuxSysfs()
		var stdout, stderr bytes.Buffer
		code := RunCLI(&stdout, &stderr, []string{"--apply", "--sysfs-root", "/sys", "--proc-root", "/proc"},
			WithFS(freshFS),
			WithGOOS("linux"),
			WithElevation(func() bool { return true }),
		)
		if code != 0 {
			t.Fatalf("code = %d, want 0; stderr: %s", code, stderr.String())
		}
		if !strings.Contains(stdout.String(), "READY:") {
			t.Fatalf("stdout missing READY:\n%s", stdout.String())
		}
	})

	t.Run("apply with json output", func(t *testing.T) {
		freshFS := setupMockLinuxSysfs()
		var stdout, stderr bytes.Buffer
		code := RunCLI(&stdout, &stderr, []string{"--apply", "--json", "--sysfs-root", "/sys", "--proc-root", "/proc"},
			WithFS(freshFS),
			WithGOOS("linux"),
			WithElevation(func() bool { return true }),
		)
		if code != 0 {
			t.Fatalf("code = %d, want 0; stderr: %s", code, stderr.String())
		}
		var res ApplyResult
		if err := json.Unmarshal(stdout.Bytes(), &res); err != nil {
			t.Fatalf("failed to unmarshal JSON apply result: %v\n%s", err, stdout.String())
		}
		if !res.Success || res.AppliedCount != 2 {
			t.Fatalf("unexpected apply result: %+v", res)
		}
	})

	t.Run("invalid flag exits 2", func(t *testing.T) {
		var stdout, stderr bytes.Buffer
		code := RunCLI(&stdout, &stderr, []string{"--nonexistent-flag"})
		if code != 2 {
			t.Fatalf("code = %d, want 2", code)
		}
	})

	t.Run("unexpected arguments exits 2", func(t *testing.T) {
		var stdout, stderr bytes.Buffer
		code := RunCLI(&stdout, &stderr, []string{"unexpected-arg"})
		if code != 2 {
			t.Fatalf("code = %d, want 2", code)
		}
	})

	t.Run("invalid DPM level exits 1", func(t *testing.T) {
		var stdout, stderr bytes.Buffer
		code := RunCLI(&stdout, &stderr, []string{"--level", "invalid-level"})
		if code != 1 {
			t.Fatalf("code = %d, want 1", code)
		}
	})

	t.Run("explicit pages flag via CLI", func(t *testing.T) {
		var stdout, stderr bytes.Buffer
		code := RunCLI(&stdout, &stderr, []string{"--json", "--pages", "10000000", "--sysfs-root", "/sys", "--proc-root", "/proc"},
			WithFS(mockFS),
			WithGOOS("linux"),
			WithElevation(func() bool { return false }),
		)
		if code != 0 {
			t.Fatalf("code = %d, want 0; stderr: %s", code, stderr.String())
		}
		var parsed GovernorReport
		if err := json.Unmarshal(stdout.Bytes(), &parsed); err != nil {
			t.Fatalf("failed to parse JSON stdout: %v", err)
		}
		if parsed.TTM.TargetPagesLimit != 10000000 {
			t.Fatalf("TargetPagesLimit = %d, want 10000000", parsed.TTM.TargetPagesLimit)
		}
	})

	t.Run("explicit ratio flag via CLI", func(t *testing.T) {
		var stdout, stderr bytes.Buffer
		code := RunCLI(&stdout, &stderr, []string{"--json", "--ratio", "0.85", "--sysfs-root", "/sys", "--proc-root", "/proc"},
			WithFS(mockFS),
			WithGOOS("linux"),
			WithElevation(func() bool { return false }),
		)
		if code != 0 {
			t.Fatalf("code = %d, want 0; stderr: %s", code, stderr.String())
		}
		var parsed GovernorReport
		if err := json.Unmarshal(stdout.Bytes(), &parsed); err != nil {
			t.Fatalf("failed to parse JSON stdout: %v", err)
		}
		if parsed.TTM.TargetPercentOfRAM != 85.0 {
			t.Fatalf("TargetPercentOfRAM = %v, want 85.0", parsed.TTM.TargetPercentOfRAM)
		}
	})
}

func TestLinuxPlanMissingTTM(t *testing.T) {
	mockFS := NewMockFS()
	mockFS.files["/proc/meminfo"] = []byte("MemTotal:       67108864 kB\n")
	mockFS.files["/sys/class/drm/card0/device/power_dpm_force_performance_level"] = []byte("auto\n")
	// TTM node omitted

	plan, err := BuildPlan(GovernorConfig{SysfsRoot: "/sys", ProcRoot: "/proc"},
		WithFS(mockFS),
		WithGOOS("linux"),
	)
	if err != nil {
		t.Fatalf("BuildPlan failed: %v", err)
	}
	if plan.TTM.Available {
		t.Fatal("TTM should not be available when sysfs node is missing")
	}
	if len(plan.Warnings) == 0 {
		t.Fatal("expected warning when TTM is missing")
	}
}

func TestLinuxPlanNoCards(t *testing.T) {
	mockFS := NewMockFS()
	mockFS.files["/proc/meminfo"] = []byte("MemTotal:       67108864 kB\n")
	mockFS.files["/sys/module/ttm/parameters/pages_limit"] = []byte("0\n")
	// Cards omitted

	plan, err := BuildPlan(GovernorConfig{SysfsRoot: "/sys", ProcRoot: "/proc"},
		WithFS(mockFS),
		WithGOOS("linux"),
	)
	if err != nil {
		t.Fatalf("BuildPlan failed: %v", err)
	}
	if len(plan.Cards) != 0 {
		t.Fatalf("cards = %d, want 0", len(plan.Cards))
	}
	if len(plan.Warnings) == 0 {
		t.Fatal("expected warning when no cards found")
	}
}

func TestLinuxPlanTargetPagesExceedsSafeRAM(t *testing.T) {
	mockFS := setupMockLinuxSysfs()

	// 64 GiB RAM is 16,777,216 pages. 20,000,000 pages > 95%.
	cfg := GovernorConfig{
		TargetPagesLimit: 20000000,
		SysfsRoot:        "/sys",
		ProcRoot:         "/proc",
	}

	plan, err := BuildPlan(cfg,
		WithFS(mockFS),
		WithGOOS("linux"),
		WithElevation(func() bool { return true }),
	)
	if err != nil {
		t.Fatalf("BuildPlan failed: %v", err)
	}
	if len(plan.Errors) == 0 {
		t.Fatal("expected error when TargetPagesLimit exceeds 95% of RAM")
	}

	res, err := Apply(plan, WithFS(mockFS), WithElevation(func() bool { return true }))
	if err == nil {
		t.Fatal("Apply should fail when plan has errors")
	}
	if res.Success {
		t.Fatal("res.Success should be false")
	}
}

func TestApplyNilPlan(t *testing.T) {
	_, err := Apply(nil)
	if err == nil {
		t.Fatal("expected error for nil plan")
	}
}

func TestExtractCardName(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		{"/sys/class/drm/card0/device/power_dpm_force_performance_level", "card0"},
		{"/sys/class/drm/card12/device/power_dpm_force_performance_level", "card12"},
		{"card3", "card3"},
	}
	for _, tc := range tests {
		if got := extractCardName(tc.input); got != tc.want {
			t.Errorf("extractCardName(%q) = %q, want %q", tc.input, got, tc.want)
		}
	}
}

func TestValidDPMLevelsString(t *testing.T) {
	s := validDPMLevelsString()
	for _, expected := range []string{"auto", "high", "low", "manual", "profile_peak"} {
		if !strings.Contains(s, expected) {
			t.Errorf("validDPMLevelsString missing %q: %s", expected, s)
		}
	}
}

func TestApplyResultJSON(t *testing.T) {
	res := &ApplyResult{
		Success:      true,
		AppliedCount: 1,
		Actions: []ActionItem{
			{Parameter: "power_dpm", Path: "/sys/card0", Current: "auto", Target: "high"},
		},
		Verified: true,
	}
	data, err := res.JSON()
	if err != nil {
		t.Fatalf("JSON error: %v", err)
	}
	var decoded ApplyResult
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("Unmarshal error: %v", err)
	}
	if !decoded.Success || decoded.AppliedCount != 1 || !decoded.Verified {
		t.Fatalf("decoded mismatch: %+v", decoded)
	}
}

// failingPathFS simulates a write failure for a specific path.
type failingPathFS struct {
	*MockFS
	failingPath string
	err         error
}

func (f *failingPathFS) WriteFile(p string, d []byte, perm os.FileMode) error {
	if filepath.ToSlash(filepath.Clean(p)) == f.failingPath {
		return f.err
	}
	return f.MockFS.WriteFile(p, d, perm)
}

// dynamicFailFS allows intercepting WriteFile with custom logic (e.g. failing during rollback).
type dynamicFailFS struct {
	*MockFS
	writeHook func(p string, d []byte) error
}

func (f *dynamicFailFS) WriteFile(p string, d []byte, perm os.FileMode) error {
	if f.writeHook != nil {
		if err := f.writeHook(p, d); err != nil {
			return err
		}
	}
	return f.MockFS.WriteFile(p, d, perm)
}

func TestApply_RollbackOnFailure(t *testing.T) {
	t.Run("second action write failure restores first action sysfs file", func(t *testing.T) {
		mockFS := setupMockLinuxSysfs()
		failFS := &failingPathFS{
			MockFS:      mockFS,
			failingPath: "/sys/module/ttm/parameters/pages_limit",
			err:         errors.New("simulated write error: permission denied"),
		}

		cfg := GovernorConfig{
			TargetDPMLevel: "high",
			SysfsRoot:      "/sys",
			ProcRoot:       "/proc",
		}

		plan, err := BuildPlan(cfg,
			WithFS(failFS),
			WithGOOS("linux"),
			WithElevation(func() bool { return true }),
		)
		if err != nil {
			t.Fatalf("BuildPlan failed: %v", err)
		}
		if len(plan.Actions) != 2 {
			t.Fatalf("expected 2 actions, got %d", len(plan.Actions))
		}

		result, err := Apply(plan,
			WithFS(failFS),
			WithElevation(func() bool { return true }),
		)
		if err == nil {
			t.Fatal("expected Apply to fail")
		}
		if result.Success {
			t.Fatal("result.Success should be false")
		}
		if result.Verified {
			t.Fatal("result.Verified should be false")
		}
		if result.AppliedCount != 0 {
			t.Fatalf("result.AppliedCount = %d, want 0", result.AppliedCount)
		}
		if !strings.Contains(err.Error(), "simulated write error: permission denied") {
			t.Fatalf("expected error to contain write failure, got: %v", err)
		}
		if !strings.Contains(err.Error(), "rollback succeeded") {
			t.Fatalf("expected error to indicate rollback succeeded, got: %v", err)
		}

		// Verify that card0 was restored to its original value "auto"
		card0Val := strings.TrimSpace(string(mockFS.files["/sys/class/drm/card0/device/power_dpm_force_performance_level"]))
		if card0Val != "auto" {
			t.Fatalf("card0 sysfs file was not restored to original value; got %q, want %q", card0Val, "auto")
		}
	})

	t.Run("rollback error details included when rollback fails", func(t *testing.T) {
		mockFS := setupMockLinuxSysfs()
		card0Path := "/sys/class/drm/card0/device/power_dpm_force_performance_level"
		ttmPath := "/sys/module/ttm/parameters/pages_limit"

		card0Writes := 0
		dynFS := &dynamicFailFS{
			MockFS: mockFS,
			writeHook: func(p string, d []byte) error {
				clean := filepath.ToSlash(filepath.Clean(p))
				if clean == card0Path {
					card0Writes++
					if card0Writes > 1 {
						return errors.New("simulated rollback permission error on card0")
					}
				}
				if clean == ttmPath {
					return errors.New("simulated ttm apply error")
				}
				return nil
			},
		}

		cfg := GovernorConfig{
			TargetDPMLevel: "high",
			SysfsRoot:      "/sys",
			ProcRoot:       "/proc",
		}

		plan, err := BuildPlan(cfg,
			WithFS(dynFS),
			WithGOOS("linux"),
			WithElevation(func() bool { return true }),
		)
		if err != nil {
			t.Fatalf("BuildPlan failed: %v", err)
		}

		result, err := Apply(plan,
			WithFS(dynFS),
			WithElevation(func() bool { return true }),
		)
		if err == nil {
			t.Fatal("expected Apply to fail")
		}
		if result.Success {
			t.Fatal("result.Success should be false")
		}
		if !strings.Contains(err.Error(), "rollback failed") {
			t.Fatalf("expected error to indicate rollback failed, got: %v", err)
		}
		if !strings.Contains(err.Error(), "simulated rollback permission error on card0") {
			t.Fatalf("expected error to include rollback failure details, got: %v", err)
		}
	})
}
