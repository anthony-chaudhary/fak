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
# Two process-IO snapshots 1s apart -> ops/sec per process. The second pass also
# carries HandleCount, so the handle census (total + top holders) is computed
# from the SAME enumeration — no extra Get-Process/Get-CimInstance walk.
$snap = { Get-CimInstance Win32_Process | ForEach-Object {
  [pscustomobject]@{ pid=$_.ProcessId; name=$_.Name; handles=[int]$_.HandleCount;
    ops=[int64]$_.ReadOperationCount + [int64]$_.WriteOperationCount + [int64]$_.OtherOperationCount } } }
$a=@{}; & $snap | ForEach-Object { $a[$_.pid]=$_ }
Start-Sleep -Seconds 1
$top=@(); $hlist=@(); $handleTotal=0
& $snap | ForEach-Object {
  $handleTotal += $_.handles
  $hlist += [pscustomobject]@{ pid=$_.pid; name=$_.name; handles=$_.handles }
  $prev = $a[$_.pid]
  if ($prev) { $d = $_.ops - $prev.ops; if ($d -gt 0) { $top += [pscustomobject]@{ pid=$_.pid; name=$_.name; ops=$d } } }
}
$top   = $top   | Sort-Object ops     -Descending | Select-Object -First 12
$topH  = $hlist | Sort-Object handles -Descending | Select-Object -First 12
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
  handleTotal = [int64]$handleTotal
  top         = $top
  topHandles  = $topH
} | ConvertTo-Json -Compress -Depth 4
`

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

type stallRaw struct {
	Faults      float64 `json:"faults"`
	Hard        float64 `json:"hard"`
	DemandZero  float64 `json:"demandZero"`
	Transition  float64 `json:"transition"`
	Ctxsw       float64 `json:"ctxsw"`
	Syscalls    float64 `json:"syscalls"`
	Procs       int     `json:"procs"`
	Threads     int     `json:"threads"`
	AvailMB     int     `json:"availMB"`
	DiskQ       float64 `json:"diskQ"`
	HandleTotal int64   `json:"handleTotal"`
	// Top and TopHandles are json.RawMessage because PowerShell's ConvertTo-Json
	// unwraps a SINGLE-element array into a bare object — so on a quiet box (only
	// one process with a positive IO delta) these arrive as `{...}`, not `[...]`.
	// psList decodes either shape, so the monitor never crashes when the box calms.
	Top        json.RawMessage `json:"top"`
	TopHandles json.RawMessage `json:"topHandles"`
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
		SystemHandleTotal:      int(raw.HandleTotal),
	}
	if prevProcCount > 0 {
		s.ProcessDelta = raw.Procs - prevProcCount
	}
	prevProcCount = raw.Procs
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
