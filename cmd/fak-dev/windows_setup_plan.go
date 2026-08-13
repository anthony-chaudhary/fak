// Package windowsdevsetup installs and verifies the Windows Security exceptions
// needed by fak's native development loop.
package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

const (
	FleetGroup = "239.255.70.65"
	FleetPort  = 4765
)

type SetupSpec struct {
	Paths     []string `json:"paths"`
	Processes []string `json:"processes"`
	Group     string   `json:"fleet_group"`
	Port      int      `json:"fleet_port"`
}

type Item struct {
	Value   string `json:"value"`
	Present bool   `json:"present"`
}

type Result struct {
	AppliedAt string `json:"applied_at"`
	Paths     []Item `json:"paths"`
	Processes []Item `json:"processes"`
	Firewall  []Item `json:"firewall"`
	Profiles  []Item `json:"profiles"`
}

func (r Result) Complete() bool {
	for _, set := range [][]Item{r.Paths, r.Processes, r.Firewall, r.Profiles} {
		for _, item := range set {
			if !item.Present {
				return false
			}
		}
	}
	return len(r.Paths) > 0 && len(r.Processes) > 0 && len(r.Firewall) == 2 && len(r.Profiles) == 3
}

func buildWindowsSetupSpec(repo string) (SetupSpec, error) {
	if strings.TrimSpace(repo) == "" {
		return SetupSpec{}, fmt.Errorf("repository root is required")
	}
	if _, err := os.Stat(filepath.Join(repo, "go.mod")); err != nil {
		return SetupSpec{}, fmt.Errorf("repository root must contain go.mod: %w", err)
	}
	abs, err := filepath.Abs(repo)
	if err != nil {
		return SetupSpec{}, err
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return SetupSpec{}, err
	}
	local := os.Getenv("LOCALAPPDATA")
	if local == "" {
		local = filepath.Join(home, "AppData", "Local")
	}
	temp := os.TempDir()
	goPath := os.Getenv("GOPATH")
	if goPath == "" {
		goPath = filepath.Join(home, "go")
	}
	paths := []string{
		abs, temp, filepath.Join(local, "go-build"), goPath,
		filepath.Join(local, "Fleet"), filepath.Join(local, "fak"),
		filepath.Join(local, "fak-maintenance"), filepath.Join(local, "fak-models"),
		filepath.Join(local, "fak-reaper"), filepath.Join(local, "fak-watchdog-audit"),
		filepath.Join(home, ".codex"), filepath.Join(home, ".claude"),
	}
	paths = uniqueClean(paths)
	return SetupSpec{Paths: paths, Processes: []string{
		"go.exe", "compile.exe", "link.exe", "fak.exe", "fak-dev.exe",
		"codex.exe", "claude.exe", "pwsh.exe", "powershell.exe", "node.exe",
		"git.exe", "bash.exe", "sh.exe",
	}, Group: FleetGroup, Port: FleetPort}, nil
}

func uniqueClean(in []string) []string {
	seen := map[string]bool{}
	out := make([]string, 0, len(in))
	for _, p := range in {
		p = filepath.Clean(p)
		k := strings.ToLower(p)
		if p != "." && !seen[k] {
			seen[k] = true
			out = append(out, p)
		}
	}
	sort.Strings(out)
	return out
}

func (p SetupSpec) JSON() ([]byte, error) { return json.MarshalIndent(p, "", "  ") }

func PowerShell(p SetupSpec, resultPath string, apply bool) string {
	q := func(s string) string { return "'" + strings.ReplaceAll(s, "'", "''") + "'" }
	arr := func(xs []string) string {
		ys := make([]string, len(xs))
		for i, x := range xs {
			ys[i] = q(x)
		}
		return "@(" + strings.Join(ys, ",") + ")"
	}
	mode := "$false"
	if apply {
		mode = "$true"
	}
	return fmt.Sprintf(`$ErrorActionPreference='Stop'
$apply=%s
$paths=%s
$processes=%s
if($apply){
 foreach($x in $paths){if(Test-Path -LiteralPath $x){Add-MpPreference -ExclusionPath $x}}
 foreach($x in $processes){Add-MpPreference -ExclusionProcess $x}
 Set-NetFirewallProfile -Profile Domain,Private,Public -DefaultInboundAction Allow -DefaultOutboundAction Allow -NotifyOnListen False -AllowInboundRules True -AllowLocalFirewallRules True
 foreach($d in @('Inbound','Outbound')){
  $n='fak-fleet-spine-'+$d.ToLowerInvariant()
  Get-NetFirewallRule -Name $n -ErrorAction SilentlyContinue|Remove-NetFirewallRule
  New-NetFirewallRule -Name $n -DisplayName ('fak fleet spine multicast ('+$d.ToLowerInvariant()+')') -Direction $d -Action Allow -Enabled True -Profile Any -Protocol UDP -RemoteAddress %s -LocalPort %d -RemotePort %d|Out-Null
 }
}
$p=Get-MpPreference
$r=[ordered]@{applied_at=(Get-Date).ToString('o');paths=@();processes=@();firewall=@();profiles=@()}
foreach($x in $paths){$r.paths += [ordered]@{value=$x;present=(@($p.ExclusionPath)-contains $x)}}
foreach($x in $processes){$r.processes += [ordered]@{value=$x;present=(@($p.ExclusionProcess)-contains $x)}}
foreach($d in @('inbound','outbound')){$n='fak-fleet-spine-'+$d;$f=Get-NetFirewallRule -Name $n -ErrorAction SilentlyContinue;$a=$f|Get-NetFirewallAddressFilter -ErrorAction SilentlyContinue;$pt=$f|Get-NetFirewallPortFilter -ErrorAction SilentlyContinue;$ok=[bool]($f -and $f.Enabled -eq 'True' -and $f.Action -eq 'Allow' -and @($a.RemoteAddress)-contains %s -and @($pt.LocalPort)-contains '%d');$r.firewall += [ordered]@{value=$n;present=$ok}}
foreach($name in @('Domain','Private','Public')){$fp=Get-NetFirewallProfile -Name $name;$ok=[bool]($fp.DefaultInboundAction -eq 'Allow' -and $fp.DefaultOutboundAction -eq 'Allow' -and $fp.NotifyOnListen -eq $false -and $fp.AllowInboundRules -eq 'True' -and $fp.AllowLocalFirewallRules -eq 'True');$r.profiles += [ordered]@{value=$name;present=$ok}}
$r|ConvertTo-Json -Depth 5|Set-Content -LiteralPath %s -Encoding utf8
`, mode, arr(p.Paths), arr(p.Processes), q(p.Group), p.Port, p.Port, q(p.Group), p.Port, q(resultPath))
}
