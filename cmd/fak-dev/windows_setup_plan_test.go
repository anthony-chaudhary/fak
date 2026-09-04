package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestDefaultPlanCoversNativeTestsAndFleetSpine(t *testing.T) {
	local := filepath.Join(t.TempDir(), "Local")
	t.Setenv("LOCALAPPDATA", local)
	t.Setenv("GOPATH", filepath.Join(t.TempDir(), "go"))
	repo := t.TempDir()
	if err := os.WriteFile(filepath.Join(repo, "go.mod"), []byte("module example"), 0o600); err != nil {
		t.Fatal(err)
	}
	p, err := buildWindowsSetupSpec(repo)
	if err != nil {
		t.Fatal(err)
	}
	home, err := os.UserHomeDir()
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		repo,
		filepath.Join(local, "go-build"),
		filepath.Join(local, "Fleet"),
		filepath.Join(local, "fak"),
		filepath.Join(home, ".codex"),
		filepath.Join(home, ".claude"),
	} {
		found := false
		for _, got := range p.Paths {
			if strings.EqualFold(filepath.Clean(got), filepath.Clean(want)) {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("paths omit %q: %v", want, p.Paths)
		}
	}
	procs := strings.ToLower(strings.Join(p.Processes, " "))
	for _, want := range []string{"go.exe", "compile.exe", "link.exe", "fak.exe", "codex.exe", "claude.exe"} {
		if !strings.Contains(procs, want) {
			t.Errorf("processes omit %s", want)
		}
	}
	if p.Group != FleetGroup || p.Port != FleetPort {
		t.Fatalf("fleet spine endpoint = %s:%d", p.Group, p.Port)
	}
}

func TestPlanUsesFleetSpineEndpointEnvironment(t *testing.T) {
	t.Setenv(FleetGroupEnv, "239.1.2.3")
	t.Setenv(FleetPortEnv, "9876")
	repo := t.TempDir()
	if err := os.WriteFile(filepath.Join(repo, "go.mod"), []byte("module example"), 0o600); err != nil {
		t.Fatal(err)
	}
	p, err := buildWindowsSetupSpec(repo)
	if err != nil {
		t.Fatal(err)
	}
	if p.Group != "239.1.2.3" || p.Port != 9876 {
		t.Fatalf("fleet spine endpoint = %s:%d, want environment endpoint", p.Group, p.Port)
	}
	script := PowerShell(p, `C:\tmp\result.json`, true)
	for _, want := range []string{"239.1.2.3", "9876"} {
		if !strings.Contains(script, want) {
			t.Errorf("script omits configured endpoint %q", want)
		}
	}
	for _, stale := range []string{FleetGroup, "4765"} {
		if strings.Contains(script, stale) {
			t.Errorf("script retains hard-coded endpoint %q", stale)
		}
	}
}

func TestPlanFallsBackForInvalidFleetSpinePortEnvironment(t *testing.T) {
	for _, value := range []string{"", "not-a-port", "0", "65536"} {
		t.Run(value, func(t *testing.T) {
			t.Setenv(FleetPortEnv, value)
			_, port := fleetSpineEndpointFromEnv()
			if port != FleetPort {
				t.Fatalf("port = %d, want default %d", port, FleetPort)
			}
		})
	}
}

func TestDefaultPlanRefusesBroadNonRepositoryPath(t *testing.T) {
	if _, err := buildWindowsSetupSpec(t.TempDir()); err == nil {
		t.Fatal("accepted directory without go.mod")
	}
}

func TestPowerShellIsIdempotentAndReadBackDriven(t *testing.T) {
	p := SetupSpec{Paths: []string{`C:\src\fak`}, Processes: []string{"go.exe"}, Group: "239.1.2.3", Port: 9876}
	script := PowerShell(p, `C:\tmp\result.json`, true)
	for _, want := range []string{"Add-MpPreference", "Set-NetFirewallProfile", "-DefaultInboundAction Block", "-DefaultOutboundAction Allow", "-NotifyOnListen False", "-AllowInboundRules True", "-AllowLocalFirewallRules True", "Get-NetFirewallRule", "Remove-NetFirewallRule", "New-NetFirewallRule", "Get-MpPreference", "Get-NetFirewallProfile", "239.1.2.3", "9876", "ConvertTo-Json"} {
		if !strings.Contains(script, want) {
			t.Errorf("script omits %q", want)
		}
	}
}

func TestResultCompleteRequiresEveryReadBack(t *testing.T) {
	r := Result{
		Paths: []Item{{Present: true}}, Processes: []Item{{Present: true}},
		Firewall: []Item{{Present: true}, {Present: true}},
		Profiles: []Item{{Present: true}, {Present: true}, {Present: true}},
	}
	if !r.Complete() {
		t.Fatal("complete result rejected")
	}
	r.Firewall[1].Present = false
	if r.Complete() {
		t.Fatal("partial firewall result accepted")
	}
	r.Firewall[1].Present = true
	r.Profiles[2].Present = false
	if r.Complete() {
		t.Fatal("partial profile result accepted")
	}
	b, _ := json.Marshal(r)
	if !strings.Contains(string(b), "firewall") {
		t.Fatal("result not json stable")
	}
}

func TestDefaultPlanIncludesHighPerformancePowerAndLongPaths(t *testing.T) {
	repo := t.TempDir()
	if err := os.WriteFile(filepath.Join(repo, "go.mod"), []byte("module example"), 0o600); err != nil {
		t.Fatal(err)
	}
	p, err := buildWindowsSetupSpec(repo)
	if err != nil {
		t.Fatal(err)
	}
	if !p.TunePower {
		t.Error("expected TunePower to be true by default")
	}
	if !p.LongPaths {
		t.Error("expected LongPaths to be true by default")
	}
	procs := strings.ToLower(strings.Join(p.Processes, " "))
	for _, proc := range []string{"gh.exe", "wsl.exe", "python.exe", "ninja.exe"} {
		if !strings.Contains(procs, proc) {
			t.Errorf("processes omit %s", proc)
		}
	}
	allPaths := strings.ToLower(strings.Join(p.Paths, ";"))
	for _, sub := range []string{"opencode", "fak-gotmp"} {
		if !strings.Contains(allPaths, sub) {
			t.Errorf("paths omit %s", sub)
		}
	}
}

func TestPowerShellGeneratesHighPerformanceAndLongPathsScript(t *testing.T) {
	p := SetupSpec{
		Paths:     []string{`C:\src\fak`},
		Processes: []string{"go.exe"},
		Group:     "239.1.2.3",
		Port:      9876,
		TunePower: true,
		LongPaths: true,
	}
	script := PowerShell(p, `C:\tmp\result.json`, true)
	for _, want := range []string{
		"VIDEOCONLOCK", "AWAYMODE", "STANDBYIDLE", "PROCTHROTTLEMIN", "SYSCOOLPOL",
		"LongPathsEnabled", "active_scheme_high_performance", "videoconlock_ac_disabled",
	} {
		if !strings.Contains(script, want) {
			t.Errorf("script omits %q", want)
		}
	}
}

func TestResultCompleteRequiresPowerAndSettingsWhenPresent(t *testing.T) {
	r := Result{
		Paths:     []Item{{Present: true}},
		Processes: []Item{{Present: true}},
		Firewall:  []Item{{Present: true}, {Present: true}},
		Profiles:  []Item{{Present: true}, {Present: true}, {Present: true}},
		Power:     []Item{{Value: "active_scheme_high_performance", Present: true}},
		Settings:  []Item{{Value: "filesystem_long_paths_enabled", Present: true}},
	}
	if !r.Complete() {
		t.Fatal("complete result with power and settings rejected")
	}
	r.Power[0].Present = false
	if r.Complete() {
		t.Fatal("incomplete power result accepted")
	}
	r.Power[0].Present = true
	r.Settings[0].Present = false
	if r.Complete() {
		t.Fatal("incomplete settings result accepted")
	}
}

func TestDefaultPlanSetsLaptopSafeProcThrottleMin(t *testing.T) {
	repo := t.TempDir()
	if err := os.WriteFile(filepath.Join(repo, "go.mod"), []byte("module example"), 0o600); err != nil {
		t.Fatal(err)
	}
	p, err := buildWindowsSetupSpec(repo)
	if err != nil {
		t.Fatal(err)
	}
	if p.ProcThrottleMin != DefaultProcThrottleMin {
		t.Fatalf("ProcThrottleMin = %d, want %d", p.ProcThrottleMin, DefaultProcThrottleMin)
	}
}

func TestPowerShellProcThrottleConfiguration(t *testing.T) {
	t.Run("default laptop safe 5 percent", func(t *testing.T) {
		p := SetupSpec{
			Paths:           []string{`C:\src\fak`},
			Processes:       []string{"go.exe"},
			Group:           "239.1.2.3",
			Port:            9876,
			TunePower:       true,
			ProcThrottleMin: 5,
		}
		script := PowerShell(p, `C:\tmp\result.json`, true)
		if !strings.Contains(script, "$throttleMin=5") {
			t.Error("script omits $throttleMin=5")
		}
		if !strings.Contains(script, "PROCTHROTTLEMIN $throttleMin") {
			t.Error("script omits PROCTHROTTLEMIN $throttleMin")
		}
		if !strings.Contains(script, "PROCTHROTTLEMAX 100") {
			t.Error("script omits PROCTHROTTLEMAX 100")
		}
	})

	t.Run("explicit override to 100 percent", func(t *testing.T) {
		p := SetupSpec{
			Paths:           []string{`C:\src\fak`},
			Processes:       []string{"go.exe"},
			Group:           "239.1.2.3",
			Port:            9876,
			TunePower:       true,
			ProcThrottleMin: 100,
		}
		script := PowerShell(p, `C:\tmp\result.json`, true)
		if !strings.Contains(script, "$throttleMin=100") {
			t.Error("script omits $throttleMin=100")
		}
		if !strings.Contains(script, "PROCTHROTTLEMIN $throttleMin") {
			t.Error("script omits PROCTHROTTLEMIN $throttleMin")
		}
		if !strings.Contains(script, "PROCTHROTTLEMAX 100") {
			t.Error("script omits PROCTHROTTLEMAX 100")
		}
	})

	t.Run("unspecified proc throttle min auto detects laptop or safe 5", func(t *testing.T) {
		p := SetupSpec{
			Paths:     []string{`C:\src\fak`},
			Processes: []string{"go.exe"},
			Group:     "239.1.2.3",
			Port:      9876,
			TunePower: true,
		}
		script := PowerShell(p, `C:\tmp\result.json`, true)
		for _, want := range []string{"Win32_Battery", "Win32_SystemEnclosure", "PROCTHROTTLEMIN $throttleMin", "PROCTHROTTLEMAX 100"} {
			if !strings.Contains(script, want) {
				t.Errorf("script omits %q", want)
			}
		}
	})
}

func TestOneDriveSyncDetectionAndWarning(t *testing.T) {
	oneDriveRoot := filepath.Join(t.TempDir(), "FakeOneDrive")
	t.Setenv("OneDrive", oneDriveRoot)

	// Subfolder inside OneDrive
	repoInside := filepath.Join(oneDriveRoot, "Desktop", "my-repo")
	if err := os.MkdirAll(repoInside, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(repoInside, "go.mod"), []byte("module example"), 0o600); err != nil {
		t.Fatal(err)
	}

	specInside, err := buildWindowsSetupSpec(repoInside)
	if err != nil {
		t.Fatal(err)
	}
	foundWarning := false
	for _, w := range specInside.Warnings {
		if strings.Contains(w, "inside OneDrive sync path") && strings.Contains(w, "filesystem latency") {
			foundWarning = true
			break
		}
	}
	if !foundWarning {
		t.Errorf("expected OneDrive warning for repo inside %s, got warnings: %v", oneDriveRoot, specInside.Warnings)
	}

	// Folder outside OneDrive
	repoOutside := filepath.Join(t.TempDir(), "local-repo")
	if err := os.MkdirAll(repoOutside, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(repoOutside, "go.mod"), []byte("module example"), 0o600); err != nil {
		t.Fatal(err)
	}
	specOutside, err := buildWindowsSetupSpec(repoOutside)
	if err != nil {
		t.Fatal(err)
	}
	for _, w := range specOutside.Warnings {
		if strings.Contains(w, "OneDrive sync path") {
			t.Errorf("unexpected OneDrive warning for outside repo: %s", w)
		}
	}
}

func TestIsSubpathEdgeCases(t *testing.T) {
	cases := []struct {
		parent string
		child  string
		want   bool
	}{
		{`C:\Users\dev\OneDrive`, `C:\Users\dev\OneDrive\Desktop\repo`, true},
		{`C:\Users\dev\OneDrive`, `C:\Users\dev\OneDrive`, true},
		{`c:\users\dev\onedrive`, `C:\USERS\DEV\ONEDRIVE\repo`, true},
		{`C:\Users\dev\OneDrive`, `C:\Users\dev\OneDriveFake\repo`, false},
		{`C:\Users\dev\OneDrive`, `C:\src\fak`, false},
		{`/home/dev/OneDrive`, `/home/dev/OneDrive/work/fak`, true},
		{`/home/dev/OneDrive`, `/home/dev/work/fak`, false},
	}
	for _, tc := range cases {
		t.Run(fmt.Sprintf("%s in %s", tc.child, tc.parent), func(t *testing.T) {
			got := isSubpath(tc.parent, tc.child)
			if got != tc.want {
				t.Errorf("isSubpath(%q, %q) = %v, want %v", tc.parent, tc.child, got, tc.want)
			}
		})
	}
}

func TestInspectAndReapStaleTempDirs(t *testing.T) {
	tempRoot := t.TempDir()
	now := time.Now()

	stale1 := filepath.Join(tempRoot, "fak-buildcheck-111")
	stale2 := filepath.Join(tempRoot, "fak-windows-setup-222")
	fresh := filepath.Join(tempRoot, "fak-live-333")
	otherStale := filepath.Join(tempRoot, "notfak-stale-444")

	for _, d := range []string{stale1, stale2, fresh, otherStale} {
		if err := os.MkdirAll(d, 0o755); err != nil {
			t.Fatal(err)
		}
	}

	// Set modification times
	oldTime := now.Add(-50 * time.Hour)
	freshTime := now.Add(-2 * time.Hour)
	if err := os.Chtimes(stale1, oldTime, oldTime); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(stale2, oldTime, oldTime); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(fresh, freshTime, freshTime); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(otherStale, oldTime, oldTime); err != nil {
		t.Fatal(err)
	}

	// 1. Inspect stale
	staleDirs := inspectStaleTempDirs(tempRoot, 48*time.Hour, now)
	if len(staleDirs) != 2 {
		t.Fatalf("inspectStaleTempDirs found %d dirs, want 2: %v", len(staleDirs), staleDirs)
	}
	for _, d := range staleDirs {
		if filepath.Base(d) != "fak-buildcheck-111" && filepath.Base(d) != "fak-windows-setup-222" {
			t.Errorf("unexpected stale dir: %s", d)
		}
	}

	// 2. Reap stale
	reaped, err := reapStaleTempDirs(tempRoot, 48*time.Hour, now)
	if err != nil {
		t.Fatalf("reapStaleTempDirs failed: %v", err)
	}
	if len(reaped) != 2 {
		t.Fatalf("reaped %d dirs, want 2: %v", len(reaped), reaped)
	}

	// Stale dirs must be removed
	for _, d := range []string{stale1, stale2} {
		if _, err := os.Stat(d); !os.IsNotExist(err) {
			t.Errorf("expected %s to be reaped", d)
		}
	}

	// Fresh dir and other dir must be preserved
	for _, d := range []string{fresh, otherStale} {
		if _, err := os.Stat(d); err != nil {
			t.Errorf("expected %s to be preserved: %v", d, err)
		}
	}
}
