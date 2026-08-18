package procguard

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/strictjson"
)

// CollectProcesses returns one process snapshot (no CPU rate) via the platform's
// own tool (PowerShell on Windows, ps on POSIX); no third-party deps. A scan error
// is returned as a string so the caller can fold it into the payload's
// collect_error (the dimension is skipped, never treated as a clean host).
//
// The error string carries one meaning and only one: THERE IS NO CENSUS. Every
// collector below returns it exactly when the tool yielded no usable row, so a
// caller may keep reading `err == "" && len(procs) > 0` as "this snapshot is
// usable" (#5385). A tool that printed a complete table and then exited non-zero
// is a usable snapshot and is reported as one.
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
	out, toolErr := runTool(60*time.Second, "powershell", "-NoProfile", "-NonInteractive", "-Command", script)
	rows := strictjson.Rows[winRow](out)
	// Truncated or absent JSON does not parse, so censusError cannot dress a broken scan
	// up as a clean host here: no rows come back and the error stands.
	if e := censusError(len(rows), toolErr); e != "" {
		return nil, nil, e
	}
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
	if procs, handled, err := collectWindowsRelationsNative(); handled {
		return procs, err
	}
	script := "$now=Get-Date; Get-CimInstance Win32_Process -ErrorAction SilentlyContinue " +
		"| ForEach-Object { try { " +
		"$a = if ($_.CreationDate) { [int](New-TimeSpan -Start $_.CreationDate -End $now).TotalSeconds } else { -1 }; " +
		"[pscustomobject]@{ pid=$_.ProcessId; ppid=$_.ParentProcessId; name=$_.Name; cmd=$_.CommandLine; age=$a } " +
		"} catch {} } | ConvertTo-Json -Compress"
	out, toolErr := runTool(90*time.Second, "powershell", "-NoProfile", "-NonInteractive", "-Command", script)
	rows := strictjson.Rows[winRelRow](out)
	if e := censusError(len(rows), toolErr); e != "" {
		return nil, e
	}
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

// psNoColumn marks a field the host's `ps` dialect cannot supply AT ALL. Every row then
// leaves that field nil, which Classify skips as an unread dimension
// (TestClassifyMissingDimensionIsSkipped); a fabricated 0 would instead read as a real
// measurement of zero. Where a platform cannot answer, the honest answer is "unknown".
const psNoColumn = -1

// psSpec is one `ps` dialect: the exact argv to run plus where each field lands in the
// table that comes back. It exists because BSD and GNU `ps` do not share a vocabulary —
// `nlwp`, `cputimes` and `etimes` are procps-ng extensions that BSD/macOS `ps` rejects
// outright, printing a complete table and exiting 1 (#5385) — so the INVOCATION has to
// vary by host, not just the parsing. Keeping the invocation in a value rather than
// inline in the collectors is what lets a test pin the per-GOOS argv without a POSIX host
// and without running `ps`.
type psSpec struct {
	// args is the argv handed to `ps` after the program name.
	args []string
	// n is the number of whitespace-separated fields to cut each line into; the last one
	// keeps the remainder verbatim (a command line contains spaces).
	n int
	// Column indexes into the split line, or psNoColumn when this dialect has no keyword
	// for that field.
	pid, threads, rssKB, cpuTime, comm, ppid, elapsed, argv int
	// optional names the one column that may be ABSENT from a short line, in which case
	// every column after it shifts one to the left. `ps` emits nothing at all for a value
	// it has no measurement of (a zombie has neither accumulated CPU time nor an argv),
	// and the pre-#5385 parser already tolerated exactly this one-field-short shape; the
	// field keeps that tolerance instead of silently dropping those rows.
	optional int
}

// column returns one spec column from a line already split into fields. It answers ""
// for a column this dialect does not have, and for the optional column on a line that
// came back one field short — the case where the later columns shift left, because `ps`
// drops the value, not just its contents.
func (s psSpec) column(parts []string, i int) string {
	if i == psNoColumn {
		return ""
	}
	if len(parts) < s.n && s.optional != psNoColumn {
		switch {
		case i == s.optional:
			return ""
		case i > s.optional:
			i--
		}
	}
	if i < 0 || i >= len(parts) {
		return ""
	}
	return parts[i]
}

// psBSD reports whether a GOOS speaks the BSD `ps` dialect. ONLY darwin is listed, and
// that is deliberate: darwin is the platform #5385 actually witnessed rejecting the
// procps keywords. The other BSDs very probably behave the same way — but "very probably"
// is precisely the assumption that shipped this bug, so an un-witnessed GOOS keeps the
// invocation that is known to work on the hosts fak is known to run on, and runTool's
// salvage below is what keeps such a host from reading as an empty machine in the
// meantime. Adding a GOOS here should come with a pasted `ps` transcript from it.
func psBSD(goos string) bool { return goos == "darwin" }

// psCensusSpec returns the resource-census dialect for a GOOS: pid, thread count,
// resident set and cumulative CPU seconds.
//
// Why a per-GOOS branch and not one column set both dialects accept: the two vocabularies
// DO intersect (`pid`, `rss`, `comm`, `time`), but the intersection contains no
// thread-count keyword whatsoever — BSD `ps` has no `nlwp` equivalent. A shared column set
// would therefore have to drop `nlwp`, silently disabling the thread dimension on Linux —
// the exact dimension this package was built for (the witnessed incident was one process
// at ~129,427 threads). Fixing macOS by blinding Linux is not a fix. So Linux keeps its
// invocation byte-for-byte, and darwin gets the best set BSD `ps` can actually answer:
// thread count is simply not among them, and stays nil.
func psCensusSpec(goos string) psSpec {
	if psBSD(goos) {
		// `time` is BSD's cumulative CPU column and is FORMATTED ([[dd-]hh:]mm:ss[.ff]),
		// not the bare seconds procps' `cputimes` emits — see parsePSDuration.
		return psSpec{
			args:     []string{"-eo", "pid=,rss=,time=,comm="},
			n:        4,
			pid:      0,
			threads:  psNoColumn, // BSD `ps` has no thread-count keyword; Threads stays nil
			rssKB:    1,
			cpuTime:  2,
			comm:     3,
			ppid:     psNoColumn,
			elapsed:  psNoColumn,
			argv:     psNoColumn,
			optional: 2,
		}
	}
	return psSpec{
		args:     []string{"-eo", "pid=,nlwp=,rss=,cputimes=,comm="},
		n:        5,
		pid:      0,
		threads:  1,
		rssKB:    2,
		cpuTime:  3,
		comm:     4,
		ppid:     psNoColumn,
		elapsed:  psNoColumn,
		argv:     psNoColumn,
		optional: 3,
	}
}

// psRelationSpec returns the relations dialect for a GOOS: pid, ppid, elapsed age,
// command name and full command line. macOS has `etime` (formatted) where procps has
// `etimes` (seconds); everything else in this row is common vocabulary, which is why the
// darwin argv differs from the Linux one by exactly one keyword.
func psRelationSpec(goos string) psSpec {
	elapsed := "etimes=," // procps-ng: whole seconds
	if psBSD(goos) {
		elapsed = "etime=," // BSD: [[dd-]hh:]mm:ss
	}
	return psSpec{
		args:     []string{"-eo", "pid=,ppid=," + elapsed + "comm=,args="},
		n:        5,
		pid:      0,
		threads:  psNoColumn,
		rssKB:    psNoColumn,
		cpuTime:  psNoColumn,
		ppid:     1,
		elapsed:  2,
		comm:     3,
		argv:     4,
		optional: 4, // a zombie has no argv; Cmdline then falls back to the command name
	}
}

// parsePSDuration converts one `ps` time column to seconds. procps-ng's `etimes` and
// `cputimes` are bare integers, but BSD `ps` has no such keyword at all — its `etime` and
// `time` columns are FORMATTED as [[dd-]hh:]mm:ss[.ff]. A keyword rename alone would
// therefore have produced ages of 0 on macOS, so the darwin column set is only half a fix
// without this. One parser reads both dialects: a bare integer is the degenerate
// (seconds-only) case of the same grammar, so the Linux path still yields exactly the
// integer it yielded before.
//
// ok=false for anything it cannot read (a `-` placeholder, a leaked header, an unexpected
// dialect). The caller then leaves the field nil instead of recording a zero: a zero age
// means "started this instant" and a zero CPU time means "never ran", both of which are
// claims, and neither is witnessed by an unparseable string.
func parsePSDuration(s string) (float64, bool) {
	s = strings.TrimSpace(s)
	if s == "" {
		return 0, false
	}
	days := 0.0
	if i := strings.IndexByte(s, '-'); i >= 0 { // dd-hh:mm:ss
		d, err := strconv.ParseFloat(s[:i], 64)
		if err != nil || d < 0 {
			return 0, false
		}
		days, s = d, s[i+1:]
	}
	fields := strings.Split(s, ":")
	if len(fields) > 3 { // nothing in either dialect is deeper than hh:mm:ss
		return 0, false
	}
	total := 0.0
	for _, f := range fields {
		// Digits and one decimal point only. ParseFloat alone would accept "NaN", "Inf"
		// and "1e9" — a census column is never any of those, and a NaN age would poison
		// every comparison downstream instead of being rejected here.
		for _, r := range f {
			if (r < '0' || r > '9') && r != '.' {
				return 0, false
			}
		}
		v, err := strconv.ParseFloat(f, 64)
		if err != nil || v < 0 {
			return 0, false
		}
		total = total*60 + v
	}
	return days*86400 + total, true
}

// parsePSCensus folds a `ps` resource table into rows plus the pid -> cumulative CPU
// seconds map the sustained-CPU dimension differences across snapshots. Pure — the caller
// supplies the bytes — so both dialects' table shapes are testable on any host, including
// the one that cannot run `ps` at all.
func parsePSCensus(out string, spec psSpec) ([]Proc, map[int]float64) {
	procs := []Proc{}
	secs := map[int]float64{}
	for _, line := range strings.Split(out, "\n") {
		parts := splitN(line, spec.n)
		if len(parts) < spec.n-1 {
			continue
		}
		pid, perr := strconv.Atoi(spec.column(parts, spec.pid))
		if perr != nil {
			continue
		}
		p := Proc{PID: pid, Name: filepath.Base(strings.TrimSpace(spec.column(parts, spec.comm)))}
		if t, e := strconv.Atoi(spec.column(parts, spec.threads)); e == nil {
			p.Threads = IntPtr(t)
		}
		if rk, e := strconv.Atoi(spec.column(parts, spec.rssKB)); e == nil {
			p.WSMB = IntPtr(rk / 1024)
		}
		procs = append(procs, p)
		if c, ok := parsePSDuration(spec.column(parts, spec.cpuTime)); ok {
			secs[pid] = c
		}
	}
	return procs, secs
}

// parsePSRelations folds a `ps` relations table into ppid/cmdline/age rows. Pure, for the
// same reason parsePSCensus is.
func parsePSRelations(out string, spec psSpec) []Proc {
	procs := []Proc{}
	for _, line := range strings.Split(out, "\n") {
		parts := splitN(line, spec.n)
		if len(parts) < spec.n-1 {
			continue
		}
		pid, perr := strconv.Atoi(spec.column(parts, spec.pid))
		if perr != nil {
			continue
		}
		ppid, _ := strconv.Atoi(spec.column(parts, spec.ppid))
		comm := spec.column(parts, spec.comm)
		args := spec.column(parts, spec.argv)
		if args == "" {
			// An empty argv column (a zombie keeps its accounting name and nothing else)
			// falls back to the command name, exactly as the pre-#5385 parser did.
			// Downstream, an empty Cmdline means "the platform would not name a command
			// line" and the row is counted as unexaminable
			// (cmd/fak/resume_stopped_liveness.go), so leaving it empty for a row we CAN
			// name would manufacture that shape instead of reporting what we have.
			args = comm
		}
		p := Proc{PID: pid, PPID: IntPtr(ppid), Name: filepath.Base(strings.TrimSpace(comm)), Cmdline: args}
		if a, ok := parsePSDuration(spec.column(parts, spec.elapsed)); ok {
			p.AgeSec = IntPtr(int(a))
		}
		procs = append(procs, p)
	}
	return procs
}

func collectPOSIX() ([]Proc, string) {
	procs, _, err := collectPOSIXCPU()
	return procs, err
}

// censusError is the ONE rule every collector applies to a tool that failed: keep the
// failure only when nothing usable came out of it.
//
// Both halves matter and they pull in opposite directions.
//
// Rows and an error: report the census, drop the error. A `ps` that printed rows and then
// exited non-zero over one unknown keyword has answered the question, and this is the
// case #5385 is about. Reporting the error ANYWAY would be worse than the
// original bug in one specific way: consumers read a non-empty error as "there is no
// census at all" — cmd/fak/resume_stopped_liveness.go defines its `readable` as no-error
// AND at least one row — so rows-with-an-error would turn a host whose census WORKED into
// one where driver liveness is "not observable", silently disabling a witness that had
// just started working. Hence: rows win, and `readable` keeps meaning exactly what it
// meant before this change.
//
// No rows and an error: report the error. Returning zero rows and no error would state
// that the host is quiet, which is the one claim this package must never make on evidence
// it does not have — a guard that sees nothing must say it saw nothing, not that there
// was nothing to see.
//
// The residual case this rule cannot separate is a tool that died PART WAY through
// printing: those rows are kept and look complete. That is the same direction the rest of
// the package already fails in (a short census flags fewer runaways: a missed reap, never
// a wrong one), and it is bounded — with the dialect-correct column sets above, a
// non-zero exit from `ps` is no longer the expected case on any supported host.
func censusError(rows int, toolErr string) string {
	if rows > 0 {
		return ""
	}
	return toolErr
}

func collectPOSIXCPU() ([]Proc, map[int]float64, string) {
	spec := psCensusSpec(runtime.GOOS)
	out, toolErr := runTool(30*time.Second, "ps", spec.args...)
	procs, secs := parsePSCensus(out, spec)
	if e := censusError(len(procs), toolErr); e != "" {
		return nil, nil, e
	}
	return procs, secs, ""
}

func collectPOSIXRelations() ([]Proc, string) {
	spec := psRelationSpec(runtime.GOOS)
	out, toolErr := runTool(30*time.Second, "ps", spec.args...)
	procs := parsePSRelations(out, spec)
	if e := censusError(len(procs), toolErr); e != "" {
		return nil, e
	}
	return procs, ""
}

// KillPID is the destructive reaper (native process-tree termination on Windows,
// process-group or descendant-walk SIGKILL on POSIX).
func KillPID(pid int) (bool, string) {
	if pid <= 0 {
		return false, "invalid pid"
	}
	if runtime.GOOS == "windows" {
		if ok, detail, handled := killTreeWindowsNative(pid); handled {
			return ok, detail
		}
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

// runTool runs a census tool and returns BOTH its stdout and, if it failed, the failure
// text. The two are not exclusive, which is the whole point (#5385): .Output() fills the
// buffer before the wait status is ever examined, and a `ps` that does not recognise one
// requested keyword still prints every ROW of the table — in the columns it did
// recognise — and only then exits 1. Discarding that stdout is what made every POSIX
// census on BSD/macOS report an empty machine on a host running hundreds of processes.
//
// It is deliberately the CALLER that decides what a non-zero exit means, because only the
// caller can see whether the bytes parsed into anything. Every collector in this file
// applies the same rule: keep the error only when no usable row came back, so the error
// string still means "no census" and never "a partial one".
//
// The tool's stderr rides along in the error text when there is one. `exit status 1` alone
// was true and useless — the sentence that would have named this bug on sight,
// `ps: etimes: keyword not found`, was being thrown away with it.
func runTool(timeout time.Duration, name string, args ...string) (string, string) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	out, err := exec.CommandContext(ctx, name, args...).Output()
	if err != nil {
		detail := err.Error()
		var exitErr *exec.ExitError
		if errors.As(err, &exitErr) {
			if msg := strings.TrimSpace(string(exitErr.Stderr)); msg != "" {
				detail += ": " + trimTo(msg, 200)
			}
		}
		return string(out), detail
	}
	return string(out), ""
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
