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
	"context"
	"encoding/json"
	"os/exec"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/stallscan"
)

// runStallTool runs one probe command with a hard timeout. Local to this file
// (procguard's equivalent is unexported); returns (stdout, "") or ("", errmsg).
func runStallTool(timeout time.Duration, name string, args ...string) (string, string) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	out, err := exec.CommandContext(ctx, name, args...).Output()
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
# One counter batch — fault split, scheduler, memory, disk queue.
$paths = @(
 '\Memory\Page Faults/sec','\Memory\Page Reads/sec',
 '\Memory\Demand Zero Faults/sec','\Memory\Transition Faults/sec',
 '\System\Context Switches/sec','\System\System Calls/sec',
 '\System\Processes','\System\Threads',
 '\Memory\Available MBytes','\PhysicalDisk(_Total)\Current Disk Queue Length'
)
$c = Get-Counter -Counter $paths
$h = @{}
foreach ($s in $c.CounterSamples) { $h[$s.Path.Split([char]92)[-1]] = [math]::Round($s.CookedValue,2) }
# Two process-IO snapshots 1s apart -> ops/sec per process.
$snap = { Get-CimInstance Win32_Process | ForEach-Object {
  [pscustomobject]@{ pid=$_.ProcessId; name=$_.Name;
    ops=[int64]$_.ReadOperationCount + [int64]$_.WriteOperationCount + [int64]$_.OtherOperationCount } } }
$a=@{}; & $snap | ForEach-Object { $a[$_.pid]=$_ }
Start-Sleep -Seconds 1
$top=@()
& $snap | ForEach-Object {
  $prev = $a[$_.pid]
  if ($prev) { $d = $_.ops - $prev.ops; if ($d -gt 0) { $top += [pscustomobject]@{ pid=$_.pid; name=$_.name; ops=$d } } }
}
$top = $top | Sort-Object ops -Descending | Select-Object -First 12
[pscustomobject]@{
  faults      = $h['Page Faults/sec']
  hard        = $h['Page Reads/sec']
  demandZero  = $h['Demand Zero Faults/sec']
  transition  = $h['Transition Faults/sec']
  ctxsw       = $h['Context Switches/sec']
  syscalls    = $h['System Calls/sec']
  procs       = [int]$h['Processes']
  threads     = [int]$h['Threads']
  availMB     = [int]$h['Available MBytes']
  diskQ       = $h['Current Disk Queue Length']
  top         = $top
} | ConvertTo-Json -Compress -Depth 4
`

type stallRaw struct {
	Faults     float64 `json:"faults"`
	Hard       float64 `json:"hard"`
	DemandZero float64 `json:"demandZero"`
	Transition float64 `json:"transition"`
	Ctxsw      float64 `json:"ctxsw"`
	Syscalls   float64 `json:"syscalls"`
	Procs      int     `json:"procs"`
	Threads    int     `json:"threads"`
	AvailMB    int     `json:"availMB"`
	DiskQ      float64 `json:"diskQ"`
	Top        []struct {
		PID  int     `json:"pid"`
		Name string  `json:"name"`
		Ops  float64 `json:"ops"`
	} `json:"top"`
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
	s := stallscan.Sample{
		TotalFaultsPerSec:      raw.Faults,
		HardFaultsPerSec:       raw.Hard,
		DemandZeroFaultsPerSec: raw.DemandZero,
		TransitionFaultsPerSec: raw.Transition,
		ContextSwitchesPerSec:  raw.Ctxsw,
		SystemCallsPerSec:      raw.Syscalls,
		ProcessCount:           raw.Procs,
		ThreadCount:            raw.Threads,
		AvailableMB:            raw.AvailMB,
		DiskQueueLen:           raw.DiskQ,
	}
	if prevProcCount > 0 {
		s.ProcessDelta = raw.Procs - prevProcCount
	}
	prevProcCount = raw.Procs
	for _, p := range raw.Top {
		s.TopIO = append(s.TopIO, stallscan.ProcIO{PID: p.PID, Name: p.Name, Ops: p.Ops})
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
