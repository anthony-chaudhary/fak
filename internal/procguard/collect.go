package procguard

import (
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"time"
)

// CollectProcesses returns one process snapshot (no CPU rate) via the platform's
// own tool (PowerShell on Windows, ps on POSIX); no third-party deps. A scan error
// is returned as a string so the caller can fold it into the payload's
// collect_error (the dimension is skipped, never treated as a clean host).
func CollectProcesses() ([]Proc, string) {
	if runtime.GOOS == "windows" {
		return collectWindows()
	}
	return collectPOSIX()
}

// CollectProcessesCPU enriches each row with a sustained per-core CPUPct measured
// over `samples` snapshots taken `windowSec` apart. The LAST snapshot is the
// returned set, annotated with CPUPctSustained (the minimum across windows). Used
// only when the CPU dimension is enabled, so the common path pays no extra scan.
// sleeper is injectable for hermetic tests.
func CollectProcessesCPU(windowSec float64, samples int, sleeper func(time.Duration)) ([]Proc, string) {
	if sleeper == nil {
		sleeper = time.Sleep
	}
	n := samples
	if n < 2 {
		n = 2
	}
	snaps := make([]map[int]float64, 0, n)
	var last []Proc
	for i := 0; i < n; i++ {
		if i > 0 {
			w := windowSec
			if w < 0.1 {
				w = 0.1
			}
			sleeper(time.Duration(w * float64(time.Second)))
		}
		procs, cpuSecs, err := collectWithCPUSeconds()
		if err != "" {
			return procs, err
		}
		last = procs
		snaps = append(snaps, cpuSecs)
	}
	pct := CPUPctSustained(snaps, windowSec)
	for i := range last {
		if v, ok := pct[last[i].PID]; ok {
			vv := v
			last[i].CPUPct = &vv
		}
	}
	return last, ""
}

// CollectRelations returns ppid/cmdline/age rows — only run when an orphan mode is on.
func CollectRelations() ([]Proc, string) {
	if runtime.GOOS == "windows" {
		return collectWindowsRelations()
	}
	return collectPOSIXRelations()
}

func collectWithCPUSeconds() ([]Proc, map[int]float64, string) {
	if runtime.GOOS == "windows" {
		procs, secs, err := collectWindowsCPU()
		return procs, secs, err
	}
	procs, secs, err := collectPOSIXCPU()
	return procs, secs, err
}

// --- Windows collectors --------------------------------------------------- //

type winRow struct {
	PID     int     `json:"pid"`
	Name    string  `json:"name"`
	Threads int     `json:"threads"`
	Handles int     `json:"handles"`
	WS      int64   `json:"ws"`
	CPU     float64 `json:"cpu"`
	Start   string  `json:"start"`
}

func collectWindows() ([]Proc, string) {
	procs, _, err := collectWindowsCPU()
	return procs, err
}

func collectWindowsCPU() ([]Proc, map[int]float64, string) {
	script := "Get-Process -ErrorAction SilentlyContinue | ForEach-Object { " +
		"try { $st=''; try { $st=$_.StartTime.ToUniversalTime().ToString('o') } catch {}; " +
		"[pscustomobject]@{ pid=$_.Id; name=$_.ProcessName; " +
		"threads=$_.Threads.Count; handles=$_.HandleCount; ws=[int64]$_.WorkingSet64; " +
		"cpu=$_.CPU; start=$st } } catch {} " +
		"} | ConvertTo-Json -Compress"
	out, err := runTool(60*time.Second, "powershell", "-NoProfile", "-NonInteractive", "-Command", script)
	if err != "" {
		return nil, nil, err
	}
	rows := parseJSONRows[winRow](out)
	procs := make([]Proc, 0, len(rows))
	secs := map[int]float64{}
	for _, r := range rows {
		p := Proc{
			PID: r.PID, Name: r.Name,
			Threads: IntPtr(r.Threads), Handles: IntPtr(r.Handles),
			WSMB:  IntPtr(int(r.WS / (1024 * 1024))),
			Start: r.Start,
		}
		procs = append(procs, p)
		secs[r.PID] = r.CPU
	}
	return procs, secs, ""
}

type winRelRow struct {
	PID  int    `json:"pid"`
	PPID int    `json:"ppid"`
	Name string `json:"name"`
	Cmd  string `json:"cmd"`
	Age  int    `json:"age"`
}

func collectWindowsRelations() ([]Proc, string) {
	script := "$now=Get-Date; Get-CimInstance Win32_Process -ErrorAction SilentlyContinue " +
		"| ForEach-Object { try { " +
		"$a = if ($_.CreationDate) { [int](New-TimeSpan -Start $_.CreationDate -End $now).TotalSeconds } else { -1 }; " +
		"[pscustomobject]@{ pid=$_.ProcessId; ppid=$_.ParentProcessId; name=$_.Name; cmd=$_.CommandLine; age=$a } " +
		"} catch {} } | ConvertTo-Json -Compress"
	out, err := runTool(90*time.Second, "powershell", "-NoProfile", "-NonInteractive", "-Command", script)
	if err != "" {
		return nil, err
	}
	rows := parseJSONRows[winRelRow](out)
	procs := make([]Proc, 0, len(rows))
	for _, r := range rows {
		p := Proc{PID: r.PID, PPID: IntPtr(r.PPID), Name: stripExe(r.Name), Cmdline: r.Cmd}
		if r.Age >= 0 {
			p.AgeSec = IntPtr(r.Age)
		}
		procs = append(procs, p)
	}
	return procs, ""
}

// --- POSIX collectors ----------------------------------------------------- //

func collectPOSIX() ([]Proc, string) {
	procs, _, err := collectPOSIXCPU()
	return procs, err
}

func collectPOSIXCPU() ([]Proc, map[int]float64, string) {
	out, err := runTool(30*time.Second, "ps", "-eo", "pid=,nlwp=,rss=,cputimes=,comm=")
	if err != "" {
		return nil, nil, err
	}
	procs := []Proc{}
	secs := map[int]float64{}
	for _, line := range strings.Split(out, "\n") {
		parts := splitN(line, 5)
		var pidS, nlwp, rss, cputimes, comm string
		switch {
		case len(parts) >= 5:
			pidS, nlwp, rss, cputimes, comm = parts[0], parts[1], parts[2], parts[3], parts[4]
		case len(parts) == 4:
			pidS, nlwp, rss, comm = parts[0], parts[1], parts[2], parts[3]
		default:
			continue
		}
		pid, perr := strconv.Atoi(pidS)
		if perr != nil {
			continue
		}
		p := Proc{PID: pid, Name: filepath.Base(strings.TrimSpace(comm))}
		if t, e := strconv.Atoi(nlwp); e == nil {
			p.Threads = IntPtr(t)
		}
		if rk, e := strconv.Atoi(rss); e == nil {
			p.WSMB = IntPtr(rk / 1024)
		}
		procs = append(procs, p)
		if cputimes != "" {
			if c, e := strconv.ParseFloat(cputimes, 64); e == nil {
				secs[pid] = c
			}
		}
	}
	return procs, secs, ""
}

func collectPOSIXRelations() ([]Proc, string) {
	out, err := runTool(30*time.Second, "ps", "-eo", "pid=,ppid=,etimes=,comm=,args=")
	if err != "" {
		return nil, err
	}
	procs := []Proc{}
	for _, line := range strings.Split(out, "\n") {
		parts := splitN(line, 5)
		if len(parts) < 4 {
			continue
		}
		pid, perr := strconv.Atoi(parts[0])
		if perr != nil {
			continue
		}
		ppid, _ := strconv.Atoi(parts[1])
		args := parts[3]
		if len(parts) > 4 {
			args = parts[4]
		}
		p := Proc{PID: pid, PPID: IntPtr(ppid), Name: filepath.Base(strings.TrimSpace(parts[3])), Cmdline: args}
		if a, e := strconv.Atoi(parts[2]); e == nil {
			p.AgeSec = IntPtr(a)
		}
		procs = append(procs, p)
	}
	return procs, ""
}

// KillPID is the destructive reaper (taskkill /T /F on Windows, SIGKILL on POSIX).
func KillPID(pid int) (bool, string) {
	if pid <= 0 {
		return false, "invalid pid"
	}
	if runtime.GOOS == "windows" {
		out, err := runTool(30*time.Second, "taskkill", "/PID", strconv.Itoa(pid), "/T", "/F")
		return err == "", trimTo(strings.TrimSpace(out), 200)
	}
	if err := killSignal(pid); err != nil {
		return false, err.Error()
	}
	return true, "SIGKILL sent"
}

// --- streak ledger persistence ------------------------------------------- //

// LoadCPUStreaks reads the cross-tick CPU-pin streak ledger. Any error (absent or
// corrupt) yields an empty ledger — a lost ledger means a pin must re-accumulate
// its streak, the safe direction (a missed reap, never a wrong one).
func LoadCPUStreaks(logDir string) map[string]int {
	raw, err := os.ReadFile(filepath.Join(logDir, CPUStreakLedger))
	if err != nil {
		return map[string]int{}
	}
	var m map[string]int
	if json.Unmarshal(raw, &m) != nil || m == nil {
		return map[string]int{}
	}
	return m
}

// SaveCPUStreaks persists the updated streak ledger; errors are swallowed (a lost
// ledger is the safe direction).
func SaveCPUStreaks(logDir string, streaks map[string]int) {
	if err := os.MkdirAll(logDir, 0o755); err != nil {
		return
	}
	if data, err := json.Marshal(streaks); err == nil {
		_ = os.WriteFile(filepath.Join(logDir, CPUStreakLedger), data, 0o644)
	}
}

// --- shared I/O helpers --------------------------------------------------- //

func runTool(timeout time.Duration, name string, args ...string) (string, string) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	out, err := exec.CommandContext(ctx, name, args...).Output()
	if err != nil {
		return "", err.Error()
	}
	return string(out), ""
}

func parseJSONRows[T any](text string) []T {
	text = strings.TrimSpace(text)
	if text == "" {
		return nil
	}
	var rows []T
	if json.Unmarshal([]byte(text), &rows) == nil {
		return rows
	}
	var one T
	if json.Unmarshal([]byte(text), &one) == nil {
		return []T{one}
	}
	return nil
}

func stripExe(name string) string {
	if strings.HasSuffix(strings.ToLower(name), ".exe") {
		return name[:len(name)-4]
	}
	return name
}

// splitN reproduces Python's str.split(None, n-1): split on runs of whitespace
// into AT MOST n fields, the last of which keeps the remainder verbatim (so a
// space-bearing command name / args column stays one field).
func splitN(line string, n int) []string {
	line = strings.TrimSpace(line)
	if line == "" {
		return nil
	}
	out := []string{}
	for len(out) < n-1 {
		idx := strings.IndexFunc(line, isSpace)
		if idx < 0 {
			break
		}
		out = append(out, line[:idx])
		line = strings.TrimLeft(line[idx:], " \t\r\n\v\f")
	}
	if line != "" {
		out = append(out, line)
	}
	return out
}

func isSpace(r rune) bool {
	switch r {
	case ' ', '\t', '\r', '\n', '\v', '\f':
		return true
	}
	return false
}

func trimTo(s string, n int) string {
	if len(s) > n {
		return s[:n]
	}
	return s
}

// IntPtr returns a pointer to n (local helper so the package is self-contained and
// does not depend on dispatchtick.IntPtr's signature).
func IntPtr(n int) *int { return &n }
