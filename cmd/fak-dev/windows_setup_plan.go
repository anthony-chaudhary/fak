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
	"time"
)

const (
	FleetGroup             = "239.255.70.65"
	FleetPort              = 4765
	FleetGroupEnv          = "FLEET_SPINE_GROUP"
	FleetPortEnv           = "FLEET_SPINE_PORT"
	DefaultProcThrottleMin = 5
	DefaultStaleTempAge    = 48 * time.Hour
)

var windowsSetupTempDir = os.TempDir

type SetupSpec struct {
	Paths           []string `json:"paths"`
	Processes       []string `json:"processes"`
	Group           string   `json:"fleet_group"`
	Port            int      `json:"fleet_port"`
	TunePower       bool     `json:"tune_power"`
	LongPaths       bool     `json:"long_paths"`
	ProcThrottleMin int      `json:"proc_throttle_min"`
	Warnings        []string `json:"warnings,omitempty"`
	StaleTempDirs   []string `json:"stale_temp_dirs,omitempty"`
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
	temp := windowsSetupTempDir()
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

	var warnings []string
	if oneDrive, matched := checkOneDriveSync(abs); oneDrive {
		warnings = append(warnings, fmt.Sprintf("repository root %q is inside OneDrive sync path %q; OneDrive sync causes silent filesystem latency, file locking collisions, and git index stalls — consider locating checkouts on a dedicated local partition or un-synced root", abs, matched))
	}
	staleDirs := inspectStaleTempDirs(temp, DefaultStaleTempAge, time.Now())
	if len(staleDirs) > 0 {
		warnings = append(warnings, fmt.Sprintf("found %d stale fak-* temporary directory(ies) in %%TEMP%% older than 48 hours; run windows-setup --apply to reap", len(staleDirs)))
	}

	return SetupSpec{
		Paths: paths,
		Processes: []string{
			"go.exe", "compile.exe", "link.exe", "fak.exe", "fak-dev.exe",
			"codex.exe", "claude.exe", "pwsh.exe", "powershell.exe", "node.exe",
			"git.exe", "bash.exe", "sh.exe",
			"gh.exe", "wsl.exe", "wslhost.exe", "python.exe", "python3.exe",
			"ninja.exe", "cmake.exe", "nvcc.exe", "cl.exe", "conhost.exe",
		},
		Group:           group,
		Port:            port,
		TunePower:       true,
		LongPaths:       true,
		ProcThrottleMin: DefaultProcThrottleMin,
		Warnings:        warnings,
		StaleTempDirs:   staleDirs,
	}, nil
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

func isSubpath(parent, child string) bool {
	p := filepath.Clean(parent)
	c := filepath.Clean(child)
	if strings.EqualFold(p, c) {
		return true
	}
	pLower := strings.ToLower(p)
	cLower := strings.ToLower(c)
	if !strings.HasSuffix(pLower, string(filepath.Separator)) && !strings.HasSuffix(pLower, "/") {
		pLower += string(filepath.Separator)
	}
	return strings.HasPrefix(cLower, pLower)
}

func checkOneDriveSync(path string) (bool, string) {
	roots := collectOneDriveSyncRoots()
	for _, root := range roots {
		if isSubpath(root, path) {
			return true, root
		}
	}
	return false, ""
}

func collectOneDriveSyncRoots() []string {
	var roots []string
	for _, env := range []string{"OneDrive", "OneDriveCommercial", "OneDriveConsumer"} {
		if v := strings.TrimSpace(os.Getenv(env)); v != "" {
			roots = append(roots, filepath.Clean(v))
		}
	}
	if home, err := os.UserHomeDir(); err == nil && home != "" {
		roots = append(roots, filepath.Join(home, "OneDrive"))
		if entries, err := os.ReadDir(home); err == nil {
			for _, entry := range entries {
				if entry.IsDir() && strings.HasPrefix(strings.ToLower(entry.Name()), "onedrive") {
					roots = append(roots, filepath.Join(home, entry.Name()))
				}
			}
		}
	}
	roots = append(roots, queryUserShellFoldersOneDrive()...)
	return uniqueClean(roots)
}

func queryUserShellFoldersOneDrive() []string {
	if windowsSetupGOOS != "windows" {
		return nil
	}
	cmd := windowsSetupCommand("reg", "query", `HKCU\Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders`)
	out, err := cmd.Output()
	if err != nil {
		return nil
	}
	var results []string
	for _, line := range strings.Split(string(out), "\n") {
		fields := strings.Fields(line)
		if len(fields) >= 3 && (fields[1] == "REG_SZ" || fields[1] == "REG_EXPAND_SZ") {
			val := strings.Join(fields[2:], " ")
			val = os.ExpandEnv(val)
			if strings.Contains(strings.ToLower(val), "onedrive") {
				results = append(results, filepath.Clean(val))
			}
		}
	}
	return results
}

func inspectStaleTempDirs(tempRoot string, maxAge time.Duration, now time.Time) []string {
	if strings.TrimSpace(tempRoot) == "" {
		tempRoot = windowsSetupTempDir()
	}
	entries, err := os.ReadDir(tempRoot)
	if err != nil {
		return nil
	}
	var stale []string
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		name := entry.Name()
		if !strings.HasPrefix(name, "fak-") {
			continue
		}
		info, err := entry.Info()
		if err != nil {
			continue
		}
		if now.Sub(info.ModTime()) >= maxAge {
			stale = append(stale, filepath.Join(tempRoot, name))
		}
	}
	sort.Strings(stale)
	return stale
}

func reapStaleTempDirs(tempRoot string, maxAge time.Duration, now time.Time) ([]string, error) {
	stale := inspectStaleTempDirs(tempRoot, maxAge, now)
	var reaped []string
	var errs []string
	for _, dir := range stale {
		if err := os.RemoveAll(dir); err != nil {
			errs = append(errs, fmt.Sprintf("%s: %v", dir, err))
		} else {
			reaped = append(reaped, dir)
		}
	}
	if len(errs) > 0 {
		return reaped, fmt.Errorf("failed to remove %d stale temp directories: %s", len(errs), strings.Join(errs, "; "))
	}
	return reaped, nil
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
	throttleMin := p.ProcThrottleMin
	if throttleMin <= 0 {
		throttleMin = DefaultProcThrottleMin
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
  $throttleMin=%d
  if($throttleMin -le 0){
   $isLaptop=[bool](Get-CimInstance -ClassName Win32_Battery -ErrorAction SilentlyContinue)
   if(-not $isLaptop){
    $ct=(Get-CimInstance -ClassName Win32_SystemEnclosure -ErrorAction SilentlyContinue).ChassisTypes
    if($ct -and ($ct|Where-Object {$_ -in 8,9,10,11,12,14,30,31,32})){$isLaptop=$true}
   }
   if($isLaptop){$throttleMin=5}else{$throttleMin=5}
  }
  powercfg /setacvalueindex SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMIN $throttleMin 2>$null
  powercfg /setacvalueindex SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMAX 100 2>$null
  powercfg /setacvalueindex SCHEME_CURRENT SUB_PROCESSOR SYSCOOLPOL 1 2>$null
  powercfg /setactive SCHEME_CURRENT 2>$null
 }
 if($longPaths){
  Set-ItemProperty -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' -Name 'LongPathsEnabled' -Value 1 -ErrorAction SilentlyContinue
 }
 $tempRoot=[System.IO.Path]::GetTempPath()
 if($env:TEMP -and (Test-Path -LiteralPath $env:TEMP)){$tempRoot=$env:TEMP}
 if($tempRoot -and (Test-Path -LiteralPath $tempRoot)){
  $staleCutoff=(Get-Date).AddHours(-48)
  Get-ChildItem -LiteralPath $tempRoot -Filter 'fak-*' -Directory -ErrorAction SilentlyContinue|Where-Object {$_.LastWriteTime -lt $staleCutoff}|ForEach-Object {
   try { Remove-Item -LiteralPath $_.FullName -Recurse -Force -ErrorAction SilentlyContinue } catch {}
  }
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
`, mode, tunePower, longPaths, arr(p.Paths), arr(p.Processes), q(p.Group), p.Port, p.Port, throttleMin, q(p.Group), p.Port, q(resultPath))
}
