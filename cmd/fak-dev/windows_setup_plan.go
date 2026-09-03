// Package windowsdevsetup installs and verifies the Windows Security exceptions
// needed by fak's native development loop.
package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

const (
	FleetGroup    = "239.255.70.65"
	FleetPort     = 4765
	FleetGroupEnv = "FLEET_SPINE_GROUP"
	FleetPortEnv  = "FLEET_SPINE_PORT"
)

type SetupSpec struct {
	Paths     []string `json:"paths"`
	Processes []string `json:"processes"`
	Group     string   `json:"fleet_group"`
	Port      int      `json:"fleet_port"`
	TunePower bool     `json:"tune_power"`
	LongPaths bool     `json:"long_paths"`
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
	Power     []Item `json:"power,omitempty"`
	Settings  []Item `json:"settings,omitempty"`
}

func (r Result) Complete() bool {
	for _, set := range [][]Item{r.Paths, r.Processes, r.Firewall, r.Profiles, r.Power, r.Settings} {
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
		filepath.Join(local, "fak-gotmp"),
		filepath.Join(temp, "opencode"),
		filepath.Join(home, ".opencode"), filepath.Join(local, "opencode"),
		filepath.Join(home, ".cargo"), filepath.Join(home, ".rustup"),
		filepath.Join(home, ".cache"), filepath.Join(local, "pip", "cache"),
		filepath.Join(home, ".codex"), filepath.Join(home, ".claude"),
	}
	paths = uniqueClean(paths)
	group, port := fleetSpineEndpointFromEnv()
	return SetupSpec{Paths: paths, Processes: []string{
		"go.exe", "compile.exe", "link.exe", "fak.exe", "fak-dev.exe",
		"codex.exe", "claude.exe", "pwsh.exe", "powershell.exe", "node.exe",
		"git.exe", "bash.exe", "sh.exe",
		"gh.exe", "wsl.exe", "wslhost.exe", "python.exe", "python3.exe",
		"ninja.exe", "cmake.exe", "nvcc.exe", "cl.exe", "conhost.exe",
	}, Group: group, Port: port, TunePower: true, LongPaths: true}, nil
}

func fleetSpineEndpointFromEnv() (string, int) {
	group := strings.TrimSpace(os.Getenv(FleetGroupEnv))
	if group == "" {
		group = FleetGroup
	}
	port := FleetPort
	if configured, err := strconv.Atoi(strings.TrimSpace(os.Getenv(FleetPortEnv))); err == nil && configured > 0 && configured <= 65535 {
		port = configured
	}
	return group, port
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
	tunePower := "$false"
	if p.TunePower {
		tunePower = "$true"
	}
	longPaths := "$false"
	if p.LongPaths {
		longPaths = "$true"
	}
	return fmt.Sprintf(`$ErrorActionPreference='Stop'
$apply=%s
$tunePower=%s
$longPaths=%s
$paths=%s
$processes=%s
if($apply){
 foreach($x in $paths){
  if(-not (Test-Path -LiteralPath $x)){
   try { New-Item -ItemType Directory -Path $x -Force -ErrorAction SilentlyContinue|Out-Null } catch {}
  }
  Add-MpPreference -ExclusionPath $x -ErrorAction SilentlyContinue
 }
 foreach($x in $processes){Add-MpPreference -ExclusionProcess $x -ErrorAction SilentlyContinue}
 Set-NetFirewallProfile -Profile Domain,Private,Public -DefaultInboundAction Block -DefaultOutboundAction Allow -NotifyOnListen False -AllowInboundRules True -AllowLocalFirewallRules True
 foreach($d in @('Inbound','Outbound')){
  $n='fak-fleet-spine-'+$d.ToLowerInvariant()
  Get-NetFirewallRule -Name $n -ErrorAction SilentlyContinue|Remove-NetFirewallRule
  New-NetFirewallRule -Name $n -DisplayName ('fak fleet spine multicast ('+$d.ToLowerInvariant()+')') -Direction $d -Action Allow -Enabled True -Profile Any -Protocol UDP -RemoteAddress %s -LocalPort %d -RemotePort %d|Out-Null
 }
 if($tunePower){
  $schemes=powercfg /list
  $targetScheme=$null
  if($schemes -match '([a-f0-9\-]{36})\s+\(Ultimate Performance\)'){
   $targetScheme=$matches[1]
  }elseif($schemes -match '([a-f0-9\-]{36})\s+\(High performance\)'){
   $targetScheme=$matches[1]
  }else{
   $dup=powercfg -duplicatescheme e9a42b02-d5df-448d-aa00-03f14749eb61 2>&1
   if($dup -match '([a-f0-9\-]{36})'){
    $targetScheme=$matches[1]
   }else{
    $dup2=powercfg -duplicatescheme 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c 2>&1
    if($dup2 -match '([a-f0-9\-]{36})'){$targetScheme=$matches[1]}
   }
  }
  if($targetScheme){powercfg /setactive $targetScheme 2>$null}
  powercfg /setacvalueindex SCHEME_CURRENT SUB_VIDEO VIDEOCONLOCK 0 2>$null
  powercfg /setacvalueindex SCHEME_CURRENT SUB_SLEEP AWAYMODE 1 2>$null
  powercfg /setacvalueindex SCHEME_CURRENT SUB_SLEEP STANDBYIDLE 0 2>$null
  powercfg /setacvalueindex SCHEME_CURRENT SUB_SLEEP 7bc4a2f9-d8fc-4469-b07b-33eb785aaca0 0 2>$null
  powercfg /setacvalueindex SCHEME_CURRENT SUB_DISK DISKIDLE 0 2>$null
  powercfg /setacvalueindex SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMIN 100 2>$null
  powercfg /setacvalueindex SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMAX 100 2>$null
  powercfg /setacvalueindex SCHEME_CURRENT SUB_PROCESSOR SYSCOOLPOL 1 2>$null
  powercfg /setactive SCHEME_CURRENT 2>$null
 }
 if($longPaths){
  Set-ItemProperty -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' -Name 'LongPathsEnabled' -Value 1 -ErrorAction SilentlyContinue
 }
}
$p=Get-MpPreference
$r=[ordered]@{applied_at=(Get-Date).ToString('o');paths=@();processes=@();firewall=@();profiles=@();power=@();settings=@()}
foreach($x in $paths){$r.paths += [ordered]@{value=$x;present=(@($p.ExclusionPath)-contains $x)}}
foreach($x in $processes){$r.processes += [ordered]@{value=$x;present=(@($p.ExclusionProcess)-contains $x)}}
foreach($d in @('inbound','outbound')){$n='fak-fleet-spine-'+$d;$f=Get-NetFirewallRule -Name $n -ErrorAction SilentlyContinue;$a=$f|Get-NetFirewallAddressFilter -ErrorAction SilentlyContinue;$pt=$f|Get-NetFirewallPortFilter -ErrorAction SilentlyContinue;$ok=[bool]($f -and $f.Enabled -eq 'True' -and $f.Action -eq 'Allow' -and @($a.RemoteAddress)-contains %s -and @($pt.LocalPort)-contains '%d');$r.firewall += [ordered]@{value=$n;present=$ok}}
foreach($name in @('Domain','Private','Public')){$fp=Get-NetFirewallProfile -Name $name;$ok=[bool]($fp.DefaultInboundAction -eq 'Block' -and $fp.DefaultOutboundAction -eq 'Allow' -and $fp.NotifyOnListen -eq $false -and $fp.AllowInboundRules -eq 'True' -and $fp.AllowLocalFirewallRules -eq 'True');$r.profiles += [ordered]@{value=$name;present=$ok}}
if($tunePower){
 $act=powercfg /getactivescheme
 $r.power += [ordered]@{value='active_scheme_high_performance';present=[bool]($act -match 'High performance' -or $act -match 'Ultimate Performance')}
 $vcon=powercfg /qh SCHEME_CURRENT SUB_VIDEO VIDEOCONLOCK
 $r.power += [ordered]@{value='videoconlock_ac_disabled';present=[bool](@($vcon -match 'Current AC Power Setting Index:\s+0x00000000').Count -gt 0)}
 $away=powercfg /qh SCHEME_CURRENT SUB_SLEEP AWAYMODE
 $r.power += [ordered]@{value='awaymode_ac_enabled';present=[bool](@($away -match 'Current AC Power Setting Index:\s+0x00000001').Count -gt 0)}
 $standby=powercfg /qh SCHEME_CURRENT SUB_SLEEP STANDBYIDLE
 $r.power += [ordered]@{value='standbyidle_ac_disabled';present=[bool](@($standby -match 'Current AC Power Setting Index:\s+0x00000000').Count -gt 0)}
}
if($longPaths){
 $lp=Get-ItemPropertyValue -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' -Name 'LongPathsEnabled' -ErrorAction SilentlyContinue
 $r.settings += [ordered]@{value='filesystem_long_paths_enabled';present=($lp -eq 1)}
}
$r|ConvertTo-Json -Depth 5|Set-Content -LiteralPath %s -Encoding utf8
`, mode, tunePower, longPaths, arr(p.Paths), arr(p.Processes), q(p.Group), p.Port, p.Port, q(p.Group), p.Port, q(resultPath))
}
