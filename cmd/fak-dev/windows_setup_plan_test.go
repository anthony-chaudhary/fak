package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestDefaultPlanCoversNativeTestsAndFleetSpine(t *testing.T) {
	t.Setenv("LOCALAPPDATA", filepath.Join(t.TempDir(), "Local"))
	t.Setenv("GOPATH", filepath.Join(t.TempDir(), "go"))
	repo := t.TempDir()
	if err := os.WriteFile(filepath.Join(repo, "go.mod"), []byte("module example"), 0o600); err != nil {
		t.Fatal(err)
	}
	p, err := buildWindowsSetupSpec(repo)
	if err != nil {
		t.Fatal(err)
	}
	joined := strings.ToLower(strings.Join(p.Paths, "\n"))
	for _, want := range []string{strings.ToLower(repo), `go-build`, `\fleet`, `\fak`, `.codex`, `.claude`} {
		if !strings.Contains(joined, strings.ToLower(want)) {
			t.Errorf("paths omit %q: %v", want, p.Paths)
		}
	}
	procs := strings.ToLower(strings.Join(p.Processes, " "))
	for _, want := range []string{"go.exe", "compile.exe", "link.exe", "fak.exe", "codex.exe", "claude.exe"} {
		if !strings.Contains(procs, want) {
			t.Errorf("processes omit %s", want)
		}
	}
	if p.Group != "239.255.70.65" || p.Port != 4765 {
		t.Fatalf("fleet spine endpoint = %s:%d", p.Group, p.Port)
	}
}

func TestDefaultPlanRefusesBroadNonRepositoryPath(t *testing.T) {
	if _, err := buildWindowsSetupSpec(t.TempDir()); err == nil {
		t.Fatal("accepted directory without go.mod")
	}
}

func TestPowerShellIsIdempotentAndReadBackDriven(t *testing.T) {
	p := SetupSpec{Paths: []string{`C:\src\fak`}, Processes: []string{"go.exe"}, Group: FleetGroup, Port: FleetPort}
	script := PowerShell(p, `C:\tmp\result.json`, true)
	for _, want := range []string{"Add-MpPreference", "Set-NetFirewallProfile", "-DefaultInboundAction Allow", "-DefaultOutboundAction Allow", "-NotifyOnListen False", "-AllowInboundRules True", "-AllowLocalFirewallRules True", "Get-NetFirewallRule", "Remove-NetFirewallRule", "New-NetFirewallRule", "Get-MpPreference", "Get-NetFirewallProfile", "239.255.70.65", "4765", "ConvertTo-Json"} {
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
