//go:build windows

package windowgate

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"log"
	"os/exec"
	"strings"
	"testing"
	"time"
)

func TestPowerShellFlags(t *testing.T) {
	script := "Get-Process"
	args := BuildPowerShellArgs(script)

	expected := []string{
		"-NoLogo",
		"-NoProfile",
		"-NonInteractive",
		"-ExecutionPolicy",
		"Bypass",
		"-Command",
		script,
	}

	if len(args) != len(expected) {
		t.Fatalf("BuildPowerShellArgs length = %d, want %d: %v", len(args), len(expected), args)
	}
	for i, exp := range expected {
		if args[i] != exp {
			t.Errorf("arg[%d] = %q, want %q", i, args[i], exp)
		}
	}

	cmd := exec.Command("powershell.exe", args...)
	ConfigureBackgroundCommand(cmd)

	if cmd.SysProcAttr == nil {
		t.Fatal("ConfigureBackgroundCommand did not allocate SysProcAttr")
	}
	if !cmd.SysProcAttr.HideWindow {
		t.Error("SysProcAttr.HideWindow is false, want true")
	}
	if cmd.SysProcAttr.CreationFlags&CreateNoWindow == 0 {
		t.Errorf("SysProcAttr.CreationFlags = %#x, missing CreateNoWindow (%#x)", cmd.SysProcAttr.CreationFlags, CreateNoWindow)
	}
}

func TestPowerShellCategoryDeadlines(t *testing.T) {
	tests := []struct {
		name     string
		script   string
		category string
		want     time.Duration
	}{
		{
			name:     "explicit lightweight",
			script:   "Write-Output test",
			category: CategoryLightweight,
			want:     DeadlineLightweight,
		},
		{
			name:     "explicit medium",
			script:   "Write-Output test",
			category: CategoryMedium,
			want:     DeadlineMedium,
		},
		{
			name:     "explicit heavy",
			script:   "Write-Output test",
			category: CategoryHeavy,
			want:     DeadlineHeavy,
		},
		{
			name:     "explicit default",
			script:   "Write-Output test",
			category: CategoryDefault,
			want:     DeadlineDefault,
		},
		{
			name:     "auto lightweight Get-ChildItem",
			script:   "Get-ChildItem -Path C:\\",
			category: "",
			want:     DeadlineLightweight,
		},
		{
			name:     "auto lightweight Get-Date",
			script:   "Get-Date",
			category: "",
			want:     DeadlineLightweight,
		},
		{
			name:     "auto lightweight Test-Path",
			script:   "Test-Path -Path 'C:\\Windows'",
			category: "",
			want:     DeadlineLightweight,
		},
		{
			name:     "auto medium Get-CimInstance",
			script:   "Get-CimInstance -ClassName Win32_OperatingSystem",
			category: "",
			want:     DeadlineMedium,
		},
		{
			name:     "auto medium Get-WmiObject",
			script:   "Get-WmiObject -Class Win32_Service",
			category: "",
			want:     DeadlineMedium,
		},
		{
			name:     "auto medium Get-Service",
			script:   "Get-Service -Name wuauserv",
			category: "",
			want:     DeadlineMedium,
		},
		{
			name:     "auto heavy dism",
			script:   "dism.exe /Online /Get-Features",
			category: "",
			want:     DeadlineHeavy,
		},
		{
			name:     "auto heavy Install-Module",
			script:   "Install-Module -Name Pester -Force",
			category: "",
			want:     DeadlineHeavy,
		},
		{
			name:     "auto heavy Import-Module",
			script:   "Import-Module ActiveDirectory",
			category: "",
			want:     DeadlineHeavy,
		},
		{
			name:     "auto default generic",
			script:   "Write-Output 'Generic query'",
			category: "",
			want:     DeadlineDefault,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := ResolveBaseDeadline(tc.script, tc.category)
			if got != tc.want {
				t.Errorf("ResolveBaseDeadline(%q, %q) = %v, want %v", tc.script, tc.category, got, tc.want)
			}
		})
	}
}

func TestPowerShellDynamicDeadlines(t *testing.T) {
	ctx := context.Background()
	script := "Write-Output 'load check'"

	// High load detected: base deadline should scale by 1.5x and log an advisory
	res, err := RunPowerShell(ctx, script,
		WithCategory(CategoryLightweight),
		WithDetectHighLoad(func() bool { return true }),
	)
	if err != nil {
		t.Fatalf("RunPowerShell with high load: %v", err)
	}

	var foundAdvisory bool
	for _, adv := range res.Advisories {
		if strings.Contains(adv, "high system load detected") {
			foundAdvisory = true
			break
		}
	}
	if !foundAdvisory {
		t.Errorf("expected high system load advisory, got advisories: %v", res.Advisories)
	}

	// Normal load: no scaling advisory
	resNormal, err := RunPowerShell(ctx, script,
		WithCategory(CategoryLightweight),
		WithDetectHighLoad(func() bool { return false }),
	)
	if err != nil {
		t.Fatalf("RunPowerShell with normal load: %v", err)
	}
	for _, adv := range resNormal.Advisories {
		if strings.Contains(adv, "high system load detected") {
			t.Errorf("unexpected high load advisory under normal load: %s", adv)
		}
	}
}

func TestPowerShellStreamingExtension(t *testing.T) {
	ctx := context.Background()

	// PowerShell starts in ~250ms on Windows.
	// Total duration is ~1050ms with outputs at ~250ms, ~650ms, and ~1050ms.
	// Base timeout is 750ms. Without streaming extension, it would time out.
	// With streaming extension (slices of 500ms), it completes successfully.
	script := `Write-Output 'stage 1'; [Console]::Out.Flush(); Start-Sleep -Milliseconds 400; Write-Output 'stage 2'; [Console]::Out.Flush(); Start-Sleep -Milliseconds 400; Write-Output 'stage 3'`

	res, err := RunPowerShell(ctx, script,
		WithPreferredEngine("powershell.exe"),
		WithTimeout(750*time.Millisecond),
		WithExtensionSlice(500*time.Millisecond),
		WithMaxDeadline(5*time.Second),
	)
	if err != nil {
		t.Fatalf("RunPowerShell streaming extension failed: %v, stdout=%q, stderr=%q", err, res.Stdout, res.Stderr)
	}
	if res.TimedOut {
		t.Fatal("command was unexpectedly timed out despite streaming activity")
	}
	if res.ExitCode != 0 {
		t.Fatalf("unexpected exit code: %d", res.ExitCode)
	}
	if res.Extensions <= 0 {
		t.Errorf("expected Extensions > 0, got %d", res.Extensions)
	}
	if !strings.Contains(res.Stdout, "stage 3") {
		t.Errorf("expected stdout to contain 'stage 3', got: %q", res.Stdout)
	}
}

func TestPowerShellSoftProgressiveWarning(t *testing.T) {
	ctx := context.Background()

	// Base timeout: 2000ms. 70% threshold is 1400ms.
	// Command sleeps 1500ms, guaranteeing crossing the 1400ms warning threshold
	// while finishing well before the 2000ms deadline.
	script := `Start-Sleep -Milliseconds 1500; Write-Output 'progressive complete'`

	res, err := RunPowerShell(ctx, script,
		WithPreferredEngine("powershell.exe"),
		WithTimeout(2000*time.Millisecond),
		WithWarningThreshold(0.70),
		WithExtensionSlice(5*time.Second),
	)
	if err != nil {
		t.Fatalf("RunPowerShell failed: %v", err)
	}
	if res.ExitCode != 0 {
		t.Fatalf("unexpected exit code: %d", res.ExitCode)
	}

	var foundWarning bool
	for _, adv := range res.Advisories {
		if strings.Contains(adv, "command reached 70% of deadline") {
			foundWarning = true
			break
		}
	}
	if !foundWarning {
		t.Errorf("expected 70%% deadline advisory warning, got advisories: %v", res.Advisories)
	}
}

func TestPowerShellDualEngineProbeFallback(t *testing.T) {
	ctx := context.Background()
	script := "Write-Output 'FallbackProbeSuccess'"

	// Mock LookPath to report that pwsh does not exist, triggering fallback to powershell.exe
	res, err := RunPowerShell(ctx, script,
		WithLookPath(func(name string) (string, error) {
			if strings.HasPrefix(name, "pwsh") {
				return "", exec.ErrNotFound
			}
			return exec.LookPath(name)
		}),
	)
	if err != nil {
		t.Fatalf("RunPowerShell with probe fallback failed: %v", err)
	}
	if res.ExitCode != 0 {
		t.Fatalf("unexpected exit code: %d", res.ExitCode)
	}
	if !strings.Contains(res.Stdout, "FallbackProbeSuccess") {
		t.Fatalf("expected stdout 'FallbackProbeSuccess', got: %q", res.Stdout)
	}
	if !strings.EqualFold(res.Engine, "powershell.exe") {
		t.Errorf("expected engine 'powershell.exe', got: %q", res.Engine)
	}

	var foundFallbackAdvisory bool
	for _, adv := range res.Advisories {
		if strings.Contains(adv, "pwsh not found in PATH; falling back to powershell.exe") {
			foundFallbackAdvisory = true
			break
		}
	}
	if !foundFallbackAdvisory {
		t.Errorf("expected pwsh not found fallback advisory, got advisories: %v", res.Advisories)
	}
}

func TestPowerShellDualEngineStartFallback(t *testing.T) {
	ctx := context.Background()
	script := "Write-Output 'StartFallbackSuccess'"

	// Mock CommandContext such that when pwsh is attempted, it points to a non-existent binary that fails Start
	res, err := RunPowerShell(ctx, script,
		WithCommandContext(func(cmdCtx context.Context, name string, args ...string) *exec.Cmd {
			if strings.HasPrefix(name, "pwsh") {
				return exec.CommandContext(cmdCtx, "C:\\nonexistent_pwsh_binary_for_test.exe", args...)
			}
			return exec.CommandContext(cmdCtx, name, args...)
		}),
	)
	if err != nil {
		t.Fatalf("RunPowerShell with start fallback failed: %v", err)
	}
	if res.ExitCode != 0 {
		t.Fatalf("unexpected exit code: %d", res.ExitCode)
	}
	if !strings.Contains(res.Stdout, "StartFallbackSuccess") {
		t.Fatalf("expected stdout 'StartFallbackSuccess', got: %q", res.Stdout)
	}
	if !strings.EqualFold(res.Engine, "powershell.exe") {
		t.Errorf("expected engine 'powershell.exe', got: %q", res.Engine)
	}

	var foundStartAdvisory bool
	for _, adv := range res.Advisories {
		if strings.Contains(adv, "failed to start pwsh") && strings.Contains(adv, "falling back to powershell.exe") {
			foundStartAdvisory = true
			break
		}
	}
	if !foundStartAdvisory {
		t.Errorf("expected start failure fallback advisory, got advisories: %v", res.Advisories)
	}
}

func TestPowerShellContextCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())

	// Cancel context after 60ms while command sleeps for 30s
	go func() {
		time.Sleep(60 * time.Millisecond)
		cancel()
	}()

	res, err := RunPowerShell(ctx, "Start-Sleep -Seconds 30")
	if err == nil {
		t.Fatal("expected error on context cancellation, got nil")
	}
	if !errors.Is(err, context.Canceled) {
		t.Errorf("expected context.Canceled error, got: %v", err)
	}
	if res.TimedOut {
		t.Errorf("expected TimedOut false on context cancellation, got true")
	}
}

func TestPowerShellTimeoutHandling(t *testing.T) {
	ctx := context.Background()

	// Command sleeps 30s with no output, but deadline is 150ms
	res, err := RunPowerShell(ctx, "Start-Sleep -Seconds 30",
		WithTimeout(150*time.Millisecond),
		WithExtensionSlice(10*time.Millisecond),
	)
	if err == nil {
		t.Fatal("expected error on timeout, got nil")
	}
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("expected context.DeadlineExceeded, got: %v", err)
	}
	if !res.TimedOut {
		t.Errorf("expected TimedOut true, got false")
	}
	if res.ExitCode != -1 {
		t.Errorf("expected ExitCode -1 on timeout, got %d", res.ExitCode)
	}
}

func TestPowerShellExecution(t *testing.T) {
	ctx := context.Background()
	res, err := RunPowerShell(ctx, "Write-Output 'Hello Windowgate'")
	if err != nil {
		t.Fatalf("RunPowerShell basic execution failed: %v", err)
	}
	if res.ExitCode != 0 {
		t.Fatalf("expected ExitCode 0, got %d", res.ExitCode)
	}
	if !strings.Contains(res.Stdout, "Hello Windowgate") {
		t.Fatalf("expected stdout containing 'Hello Windowgate', got: %q", res.Stdout)
	}
	if res.Duration <= 0 {
		t.Errorf("expected Duration > 0, got %v", res.Duration)
	}
	if res.Engine == "" {
		t.Errorf("expected non-empty Engine")
	}
}

func TestPowerShellSuppressesAdvisoryLogSpam(t *testing.T) {
	ctx := context.Background()

	// 1. Without WithLogger, advisories must be recorded in res.Advisories but NOT spam standard log output
	var logBuf bytes.Buffer
	origLogWriter := log.Writer()
	log.SetOutput(&logBuf)
	defer log.SetOutput(origLogWriter)

	res, err := RunPowerShell(ctx, "Write-Output 'SilentAdvisory'",
		WithPreferredEngine("powershell.exe"),
		WithDetectHighLoad(func() bool { return true }),
	)
	if err != nil {
		t.Fatalf("RunPowerShell failed: %v", err)
	}
	if len(res.Advisories) == 0 {
		t.Fatalf("expected high load advisory in res.Advisories, got none")
	}
	if logBuf.Len() > 0 {
		t.Fatalf("expected zero log output when Logger is nil, got: %q", logBuf.String())
	}

	// 2. When WithLogger is explicitly configured, it should receive the advisory
	var customLogged []string
	resLogged, err := RunPowerShell(ctx, "Write-Output 'LoggedAdvisory'",
		WithPreferredEngine("powershell.exe"),
		WithDetectHighLoad(func() bool { return true }),
		WithLogger(func(format string, args ...any) {
			customLogged = append(customLogged, fmt.Sprintf(format, args...))
		}),
	)
	if err != nil {
		t.Fatalf("RunPowerShell with custom logger failed: %v", err)
	}
	if len(resLogged.Advisories) == 0 {
		t.Fatalf("expected high load advisory in resLogged.Advisories, got none")
	}
	if len(customLogged) == 0 {
		t.Fatalf("expected custom logger to receive advisory, got none")
	}
}
