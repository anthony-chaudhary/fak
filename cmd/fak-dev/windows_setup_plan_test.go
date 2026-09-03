package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
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
