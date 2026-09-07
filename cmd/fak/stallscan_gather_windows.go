//go:build windows

package main

// stallscan_gather_windows.go — the Windows OS-reading layer for `fak stallscan`.
// It gathers exactly ONE Get-Counter batch (the fault split, scheduler, memory
// and disk-queue signals) and ONE two-sample Get-CimInstance process I/O
// snapshot (to compute per-process ops/sec), then hands the numbers to the pure
// stallscan.Classify. Everything expensive lives in this single PowerShell
// invocation so the cost of a sample is one process spawn, not N.
//
// Deliberately NO per-process Get-Process/Threads.Count walk (that is what makes
// a naive monitor part of the problem). The process snapshot reads only
// Read/Write/OtherOperationCount from Win32_Process, twice, one second apart.

import (
	"bytes"
	"context"
	"encoding/json"
	"os/exec"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/stallscan"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// runStallTool runs one probe command with a hard timeout. Local to this file
// (procguard's equivalent is unexported); returns (stdout, "") or ("", errmsg).
func runStallTool(timeout time.Duration, name string, args ...string) (string, string) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, name, args...)
	windowgate.ConfigureBackgroundCommand(cmd)
	out, err := cmd.Output()
	if err != nil {
		return "", err.Error()
	}
	return string(out), ""
}

// stallPS is the one-shot PowerShell script. It emits a single compact JSON
// object: the counter batch plus the top process I/O deltas over a 1s window.
// It is intentionally self-contained (no functions, no modules) so it starts
// cold fast.
const stallPS = `
$ErrorActionPreference='SilentlyContinue'
# One counter batch — fault split, scheduler, memory, disk queue, WDDM VRAM.
$paths = @(
 '\Memory\Page Faults/sec','\Memory\Page Reads/sec',
 '\Memory\Demand Zero Faults/sec','\Memory\Transition Faults/sec',
 '\System\Context Switches/sec','\System\System Calls/sec',
 '\Processor(_Total)\% Processor Time','\System\Processor Queue Length',
 '\Memory\Committed Bytes','\Memory\Commit Limit',
 '\System\Processes','\System\Threads',
 '\Memory\Available MBytes','\PhysicalDisk(_Total)\Current Disk Queue Length',
 '\GPU Adapter Memory(*)\Dedicated Usage',
 '\GPU Adapter Memory(*)\Shared Usage',
 '\GPU Adapter Memory(*)\Total Committed'
)
$os = Get-CimInstance Win32_OperatingSystem
$vcs = Get-CimInstance Win32_VideoController
$c = Get-Counter -Counter $paths
$h = @{}
$vramDedicated = 0; $vramShared = 0; $vramCommitted = 0
foreach ($s in $c.CounterSamples) {
  $leaf = $s.Path.Split([char]92)[-1]
  $h[$leaf] = [math]::Round($s.CookedValue,2)
  if ($leaf -match '(?i)dedicated usage') { $vramDedicated += [uint64]$s.CookedValue }
  elseif ($leaf -match '(?i)shared usage') { $vramShared += [uint64]$s.CookedValue }
  elseif ($leaf -match '(?i)total committed') { $vramCommitted += [uint64]$s.CookedValue }
}
if ($vramCommitted -eq 0 -and ($vramDedicated -gt 0 -or $vramShared -gt 0)) {
  $vramCommitted = $vramDedicated + $vramShared
}
$vramTotal = 0
foreach ($vc in $vcs) {
  if ($vc.AdapterRAM -and $vc.AdapterRAM -gt 0) { $vramTotal += [uint64]$vc.AdapterRAM }
}
# Two process-IO snapshots 1s apart -> ops/sec per process. The second pass also
# carries HandleCount, so the handle census (total + top holders) is computed
# from the SAME enumeration — no extra Get-Process/Get-CimInstance walk.
$snap = { Get-CimInstance Win32_Process | ForEach-Object {
  [pscustomobject]@{ pid=$_.ProcessId; name=$_.Name; handles=[int]$_.HandleCount; threads=[int]$_.ThreadCount;
    cpu=[int64]$_.('Kernel'+'Mode'+'Time') + [int64]$_.('User'+'Mode'+'Time');
    ops=[int64]$_.ReadOperationCount + [int64]$_.WriteOperationCount + [int64]$_.OtherOperationCount } } }
# Under the very churn stall this tool diagnoses, Get-CimInstance Win32_Process
# intermittently returns an empty/degraded set — which would blank the handle,
# thread, and IO census exactly when it matters most. Retry until the snapshot
# looks plausible (>50 processes) before trusting it, so the diagnostic does not
# go blind during a stall. Fault/scheduler counters come from Get-Counter and are
# unaffected. Get-Snap yields [] on a transient failure, never throws.
function Get-Snap { $r=@(& $snap); return ,$r }
$first=@(); foreach($try in 1..4){ $first=Get-Snap; if($first.Count -gt 50){break}; Start-Sleep -Milliseconds 250 }
# A degraded FIRST snapshot would make every process in the second look newly
# born, fabricating a spawn storm exactly when the box is already struggling.
# Only report the birth count when the baseline passed the same plausibility bar.
$spawnKnown = ($first.Count -gt 50)
$a=@{}; foreach($p in $first){ $a[$p.pid]=$p }
# Time the ACTUAL window rather than assuming the nominal 1s sleep. The two
# enumerations are not free (and get slower under the very stall this diagnoses,
# where the retry loop above may add seconds), so the true span can be several
# times the sleep. A birth COUNT divided by an assumed window is a fabricated
# rate; dividing by the measured one is a real one.
$sw = [System.Diagnostics.Stopwatch]::StartNew()
Start-Sleep -Seconds 1
$second=@(); foreach($try in 1..4){ $second=Get-Snap; if($second.Count -gt 50){break}; Start-Sleep -Milliseconds 250 }
$sw.Stop()
$spawnWindow = [math]::Round($sw.Elapsed.TotalSeconds, 3)
$top=@(); $topCPU=@(); $hlist=@(); $handleTotal=0; $spawned=0
$logicalCPU=[Environment]::ProcessorCount
foreach($p in $second){
  $handleTotal += $p.handles
  $hlist += [pscustomobject]@{ pid=$p.pid; name=$p.name; handles=$p.handles; threads=$p.threads }
  $prev = $a[$p.pid]
  # A PID in the second snapshot but not the first was BORN inside the window.
  # This is the gross birth count the spawn axis needs; the net process delta
  # cancels it away (a burst of short-lived spawns is born AND reaped between
  # samples). Free here: both enumerations already exist for the I/O delta.
  if ($prev) {
    $d = $p.ops - $prev.ops; if ($d -gt 0) { $top += [pscustomobject]@{ pid=$p.pid; name=$p.name; ops=$d } }
    $dcpu = $p.cpu - $prev.cpu
    if ($p.pid -ne 0 -and $dcpu -gt 0 -and $spawnWindow -gt 0 -and $logicalCPU -gt 0) {
      $cpuPct = 100 * $dcpu / ($spawnWindow * 10000000 * $logicalCPU)
      $topCPU += [pscustomobject]@{ pid=$p.pid; name=$p.name; percent=[math]::Round($cpuPct,2) }
    }
  }
  else { $spawned++ }
}
$topCPU = $topCPU | Sort-Object percent -Descending | Select-Object -First 12
$top   = $top   | Sort-Object ops     -Descending | Select-Object -First 12
$topH  = $hlist | Sort-Object handles -Descending | Select-Object -First 12
$topT  = $hlist | Sort-Object threads -Descending | Select-Object -First 12
[pscustomobject]@{
  timestamp    = (Get-Date).ToUniversalTime().ToString('o')
  boot_time    = $os.LastBootUpTime.ToUniversalTime().ToString('o')
  commit_bytes = [uint64]$h['Committed Bytes']
  commit_limit = [uint64]$h['Commit Limit']
  available_bytes = [uint64]($h['Available MBytes'] * 1MB)
  vram_committed_bytes = [uint64]$vramCommitted
  vram_total_bytes     = [uint64]$vramTotal
  vram_shared_bytes    = [uint64]$vramShared
  faults      = $h['Page Faults/sec']
  hard        = $h['Page Reads/sec']
  demandZero  = $h['Demand Zero Faults/sec']
  transition  = $h['Transition Faults/sec']
  ctxsw       = $h['Context Switches/sec']
  syscalls    = $h['System Calls/sec']
  cpuPct      = $h['% Processor Time']
  cpuQueue    = $h['Processor Queue Length']
  logicalCPU  = [int]$logicalCPU
  procs       = [int]$h['Processes']
  threads     = [int]$h['Threads']
  availMB     = [int]$h['Available MBytes']
  diskQ       = $h['Current Disk Queue Length']
  handleTotal = [int64]$handleTotal
  spawned     = [int]$spawned
  spawnKnown  = [bool]$spawnKnown
  spawnWindow = [double]$spawnWindow
  topCPU      = $topCPU
  top         = $top
  topHandles  = $topH
  topThreads  = $topT
} | ConvertTo-Json -Compress -Depth 4
`

type stallTopCPU struct {
	PID     int     `json:"pid"`
	Name    string  `json:"name"`
	Percent float64 `json:"percent"`
}

type stallTopIO struct {
	PID  int     `json:"pid"`
	Name string  `json:"name"`
	Ops  float64 `json:"ops"`
}

type stallTopHandle struct {
	PID     int    `json:"pid"`
	Name    string `json:"name"`
	Handles int    `json:"handles"`
}

type stallTopThread struct {
	PID     int    `json:"pid"`
	Name    string `json:"name"`
	Threads int    `json:"threads"`
}

type stallRaw struct {
	Timestamp          string  `json:"timestamp"`
	BootTime           string  `json:"boot_time"`
	CommitBytes        uint64  `json:"commit_bytes"`
	CommitLimit        uint64  `json:"commit_limit"`
	AvailableBytes     uint64  `json:"available_bytes"`
	VRAMCommittedBytes uint64  `json:"vram_committed_bytes"`
	VRAMTotalBytes     uint64  `json:"vram_total_bytes"`
	VRAMSharedBytes    uint64  `json:"vram_shared_bytes"`
	Faults             float64 `json:"faults"`
	Hard               float64 `json:"hard"`
	DemandZero         float64 `json:"demandZero"`
	Transition         float64 `json:"transition"`
	Ctxsw              float64 `json:"ctxsw"`
	Syscalls           float64 `json:"syscalls"`
	CPUPct             float64 `json:"cpuPct"`
	CPUQueue           float64 `json:"cpuQueue"`
	LogicalCPU         int     `json:"logicalCPU"`
	Procs              int     `json:"procs"`
	Threads            int     `json:"threads"`
	AvailMB            int     `json:"availMB"`
	DiskQ              float64 `json:"diskQ"`
	HandleTotal        int64   `json:"handleTotal"`
	// Spawned is the GROSS count of processes born inside the probe's own
	// snapshot window (PIDs present in the second enumeration and absent from
	// the first). SpawnKnown is false when the first enumeration came back
	// degraded, in which case Spawned is meaningless and must not be trusted.
	// SpawnWindow is the MEASURED wall-clock span between the two enumerations,
	// which is what makes Spawned convertible to a rate. It is not the nominal
	// sleep: the enumerations themselves cost time, and more of it under a stall.
	Spawned     int     `json:"spawned"`
	SpawnKnown  bool    `json:"spawnKnown"`
	SpawnWindow float64 `json:"spawnWindow"`
	// TopCPU, Top, TopHandles, and TopThreads are json.RawMessage because PowerShell's ConvertTo-Json
	// unwraps a SINGLE-element array into a bare object — so on a quiet box (only
	// one process with a positive IO delta) these arrive as `{...}`, not `[...]`.
	// psList decodes either shape, so the monitor never crashes when the box calms.
	TopCPU     json.RawMessage `json:"topCPU"`
	Top        json.RawMessage `json:"top"`
	TopHandles json.RawMessage `json:"topHandles"`
	TopThreads json.RawMessage `json:"topThreads"`
}

// psList unmarshals a PowerShell ConvertTo-Json value that may be a single object
// (PS unwrapped a 1-element array), a real array, or absent/null, into dst (a
// pointer to a slice). Empty/null yields an empty slice, not an error.
func psList(raw json.RawMessage, dst any) error {
	t := bytes.TrimSpace(raw)
	if len(t) == 0 || string(t) == "null" {
		return nil
	}
	if t[0] == '[' {
		return json.Unmarshal(t, dst)
	}
	wrapped := make([]byte, 0, len(t)+2)
	wrapped = append(wrapped, '[')
	wrapped = append(wrapped, t...)
	wrapped = append(wrapped, ']')
	return json.Unmarshal(wrapped, dst)
}

// prevProcCount remembers the last census so we can report a process delta
// across snapshot invocations (best-effort; 0 on first call).
var prevProcCount int

// gatherStallSample runs the one-shot probe and returns a classified-ready
// Sample. Second return is a non-empty error string on failure.
func gatherStallSample(topN int) (stallscan.Sample, string) {
	out, errStr := runStallTool(30*time.Second, "powershell", "-NoProfile", "-NonInteractive", "-Command", stallPS)
	if errStr != "" {
		return stallscan.Sample{}, errStr
	}
	raw, perr := parseStallRaw(out)
	if perr != "" {
		return stallscan.Sample{}, perr
	}
	boot, _ := time.Parse(time.RFC3339Nano, raw.BootTime)
	s := stallscan.Sample{
		BootTime:               boot,
		CommitBytes:            raw.CommitBytes,
		CommitLimit:            raw.CommitLimit,
		AvailableBytes:         raw.AvailableBytes,
		VRAMCommittedBytes:     raw.VRAMCommittedBytes,
		VRAMTotalBytes:         raw.VRAMTotalBytes,
		VRAMSharedBytes:        raw.VRAMSharedBytes,
		TotalFaultsPerSec:      raw.Faults,
		HardFaultsPerSec:       raw.Hard,
		DemandZeroFaultsPerSec: raw.DemandZero,
		TransitionFaultsPerSec: raw.Transition,
		ContextSwitchesPerSec:  raw.Ctxsw,
		SystemCallsPerSec:      raw.Syscalls,
		CPUPercent:             raw.CPUPct,
		ProcessorQueueLength:   raw.CPUQueue,
		ProcessorCount:         raw.LogicalCPU,
		ProcessCount:           raw.Procs,
		ThreadCount:            raw.Threads,
		AvailableMB:            raw.AvailMB,
		DiskQueueLen:           raw.DiskQ,
		SystemHandleTotal:      int(raw.HandleTotal),
	}
	if prevProcCount > 0 {
		s.ProcessDelta = raw.Procs - prevProcCount
	}
	prevProcCount = raw.Procs
	// The spawn axis wants GROSS births, not the net census delta. A storm of
	// short-lived spawns (git/pwsh/tasklist bursts — the reference freeze's own
	// signature) is born and reaped inside one interval, so the net delta reads
	// ~0 and the axis stays blind to exactly the shape it was built for. The
	// probe now counts PIDs that appeared between its two enumerations, and
	// reports the MEASURED span of that window alongside the count so Classify
	// can compare a births/sec RATE. Handing over the count alone would be worse
	// than useless: on this box ordinary fleet load runs 22 gross births/sec
	// (p95 63), which clears the legacy count threshold of 8 on 95% of ticks.
	//
	// SpawnKnown false = the baseline enumeration was degraded, so every process
	// would look newly born. Leave SpawnBurst at zero rather than fabricate a
	// storm; Classify then falls back to ProcessDelta exactly as before. A
	// non-positive measured window is treated the same way — a count we cannot
	// convert is not evidence, and must not be smuggled onto the count path
	// where its calibration does not apply.
	if raw.SpawnKnown && raw.Spawned > 0 && raw.SpawnWindow > 0 {
		s.SpawnBurst = raw.Spawned
		s.SpawnWindowSeconds = raw.SpawnWindow
	}
	var topCPU []stallTopCPU
	if err := psList(raw.TopCPU, &topCPU); err == nil {
		for _, p := range topCPU {
			s.TopCPU = append(s.TopCPU, stallscan.ProcCPU{PID: p.PID, Name: p.Name, Percent: p.Percent})
		}
	}
	var topIO []stallTopIO
	if err := psList(raw.Top, &topIO); err == nil {
		for _, p := range topIO {
			s.TopIO = append(s.TopIO, stallscan.ProcIO{PID: p.PID, Name: p.Name, Ops: p.Ops})
		}
	}
	var topH []stallTopHandle
	if err := psList(raw.TopHandles, &topH); err == nil {
		for _, p := range topH {
			s.TopHandles = append(s.TopHandles, stallscan.ProcHandles{PID: p.PID, Name: p.Name, Handles: p.Handles})
		}
	}
	var topT []stallTopThread
	if err := psList(raw.TopThreads, &topT); err == nil {
		for _, p := range topT {
			s.TopThreads = append(s.TopThreads, stallscan.ProcThreads{PID: p.PID, Name: p.Name, Threads: p.Threads})
		}
	}
	return s, ""
}

// parseStallRaw tolerates ConvertTo-Json emitting a bare object (our case) and
// trims any BOM/whitespace.
func parseStallRaw(text string) (stallRaw, string) {
	t := strings.TrimSpace(strings.TrimPrefix(text, "\ufeff"))
	if t == "" {
		return stallRaw{}, "empty counter output (Get-Counter unavailable?)"
	}
	var raw stallRaw
	if err := json.Unmarshal([]byte(t), &raw); err != nil {
		return stallRaw{}, "parse counters: " + err.Error()
	}
	return raw, ""
}
