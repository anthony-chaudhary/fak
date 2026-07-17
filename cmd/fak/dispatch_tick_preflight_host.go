package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
	"github.com/anthony-chaudhary/fak/internal/procguard"
)

func dispatchRunExternalJSONImpl(root string, timeout time.Duration, name string, args ...string) (map[string]any, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, name, args...)
	cmd.Dir = root
	// Bound the post-deadline pipe wait: `dos` is a pip console-script shim
	// whose real work runs in a python.exe GRANDCHILD holding the inherited
	// stdout pipe. When the context fires, Go kills only the shim; without a
	// WaitDelay, CombinedOutput() then blocks unboundedly until the grandchild
	// exits -- the dispatch tick "hangs past its timeout" class.
	cmd.WaitDelay = 10 * time.Second
	configureDispatchHelperCommand(cmd)
	out, err := cmd.CombinedOutput()
	if obj, perr := lastJSONObject(out); perr == nil {
		return obj, nil
	}
	if err != nil {
		return nil, err
	}
	return nil, errors.New("no JSON object in helper output")
}

func dispatchProbeProcessesNative() dispatchtick.ProcGuardInput {
	procs, err := dispatchScanProcesses()
	collectError := ""
	if err != nil {
		collectError = err.Error()
	}
	return dispatchtick.ProcGuardInput{
		Processes:     procs,
		CollectError:  collectError,
		Thresholds:    dispatchtick.DefaultProcGuardThresholds(),
		ProtectedPIDs: []int{os.Getpid(), os.Getppid()},
	}
}

func dispatchScanProcesses() ([]dispatchtick.ProcInfo, error) {
	if runtime.GOOS == "windows" {
		return dispatchScanProcessesWindows()
	}
	return dispatchScanProcessesPOSIX()
}

func dispatchScanProcessesWindows() ([]dispatchtick.ProcInfo, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "powershell", "-NoProfile", "-NonInteractive", "-Command",
		"Get-Process -ErrorAction SilentlyContinue | ForEach-Object { "+
			"try { [pscustomobject]@{ pid=$_.Id; name=$_.ProcessName; threads=$_.Threads.Count; handles=$_.HandleCount; ws_mb=[int64]($_.WorkingSet64 / 1MB) } } catch {} "+
			"} | ConvertTo-Json -Compress")
	configureDispatchHelperCommand(cmd)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return nil, err
	}
	var rows []struct {
		PID     int    `json:"pid"`
		Name    string `json:"name"`
		Threads int    `json:"threads"`
		Handles int    `json:"handles"`
		WSMB    int    `json:"ws_mb"`
	}
	if uerr := json.Unmarshal(out, &rows); uerr != nil {
		var one struct {
			PID     int    `json:"pid"`
			Name    string `json:"name"`
			Threads int    `json:"threads"`
			Handles int    `json:"handles"`
			WSMB    int    `json:"ws_mb"`
		}
		if oerr := json.Unmarshal(out, &one); oerr != nil {
			return nil, uerr
		}
		rows = []struct {
			PID     int    `json:"pid"`
			Name    string `json:"name"`
			Threads int    `json:"threads"`
			Handles int    `json:"handles"`
			WSMB    int    `json:"ws_mb"`
		}{one}
	}
	procs := make([]dispatchtick.ProcInfo, 0, len(rows))
	for _, row := range rows {
		procs = append(procs, dispatchtick.ProcInfo{
			PID:          row.PID,
			Name:         row.Name,
			Threads:      dispatchtick.IntPtr(row.Threads),
			Handles:      dispatchtick.IntPtr(row.Handles),
			WorkingSetMB: dispatchtick.IntPtr(row.WSMB),
		})
	}
	return procs, nil
}

func dispatchScanProcessesPOSIX() ([]dispatchtick.ProcInfo, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "ps", "-eo", "pid=,nlwp=,rss=,comm=")
	configureDispatchHelperCommand(cmd)
	out, err := cmd.Output()
	if err != nil {
		return nil, err
	}
	procs := []dispatchtick.ProcInfo{}
	for _, line := range strings.Split(string(out), "\n") {
		fields := strings.Fields(line)
		if len(fields) < 4 {
			continue
		}
		pid, perr := strconv.Atoi(fields[0])
		threads, terr := strconv.Atoi(fields[1])
		rssKB, rerr := strconv.Atoi(fields[2])
		if perr != nil {
			continue
		}
		name := strings.Join(fields[3:], " ")
		proc := dispatchtick.ProcInfo{PID: pid, Name: name}
		if terr == nil {
			proc.Threads = dispatchtick.IntPtr(threads)
		}
		if rerr == nil {
			proc.WorkingSetMB = dispatchtick.IntPtr(rssKB / 1024)
		}
		procs = append(procs, proc)
	}
	return procs, nil
}

var dispatchBuildHostResources = dispatchPreflightHostResourcesFromProcesses

func dispatchPreflightHostResources() dispatchtick.HostResources {
	return dispatchPreflightHostResourcesFromProcesses(dispatchProbeProcesses())
}

func dispatchPreflightHostResourcesFromProcesses(processes dispatchtick.ProcGuardInput) dispatchtick.HostResources {
	cores := runtime.NumCPU()
	freeRAM := dispatchFreeRAM()
	totalThreads := 0
	seenThreads := false
	for _, proc := range processes.Processes {
		if proc.Threads != nil {
			totalThreads += *proc.Threads
			seenThreads = true
		}
	}
	var threads *int
	if seenThreads {
		threads = &totalThreads
	}
	return dispatchtick.HostResources{Cores: &cores, FreeRAMMB: freeRAM, TotalThreads: threads}
}

func dispatchFreeRAM() *int {
	if runtime.GOOS != "windows" {
		free, _ := dispatchRAMAndThreadsPOSIX()
		return free
	}
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "powershell", "-NoProfile", "-NonInteractive", "-Command", "$os=Get-CimInstance Win32_OperatingSystem; [int64]$os.FreePhysicalMemory")
	configureDispatchHelperCommand(cmd)
	out, err := cmd.Output()
	if err != nil {
		return nil
	}
	kb, err := strconv.ParseInt(strings.TrimSpace(string(out)), 10, 64)
	if err != nil {
		return nil
	}
	mb := int(kb / 1024)
	return &mb
}

func dispatchRAMAndThreads() (*int, *int) {
	if runtime.GOOS == "windows" {
		return dispatchRAMAndThreadsWindows()
	}
	return dispatchRAMAndThreadsPOSIX()
}

func dispatchRAMAndThreadsWindows() (*int, *int) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "powershell", "-NoProfile", "-NonInteractive", "-Command",
		"$os = Get-CimInstance Win32_OperatingSystem; "+
			"$t = (Get-Process -ErrorAction SilentlyContinue | ForEach-Object { $_.Threads.Count } | Measure-Object -Sum).Sum; "+
			"[pscustomobject]@{ free_kb = [int64]$os.FreePhysicalMemory; threads = [int]$t } | ConvertTo-Json -Compress")
	configureDispatchHelperCommand(cmd)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return nil, nil
	}
	doc, err := lastJSONObject(out)
	if err != nil {
		return nil, nil
	}
	freeKB := intPtrFromAny(doc["free_kb"])
	threads := intPtrFromAny(doc["threads"])
	if freeKB != nil {
		mb := *freeKB / 1024
		freeKB = &mb
	}
	return freeKB, threads
}

func dispatchRAMAndThreadsPOSIX() (*int, *int) {
	var freeRAM *int
	if b, err := os.ReadFile("/proc/meminfo"); err == nil {
		for _, line := range strings.Split(string(b), "\n") {
			if strings.HasPrefix(line, "MemAvailable:") {
				fields := strings.Fields(line)
				if len(fields) >= 2 {
					if kb, err := strconv.Atoi(fields[1]); err == nil {
						mb := kb / 1024
						freeRAM = &mb
					}
				}
				break
			}
		}
	}
	var threads *int
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "ps", "-eo", "nlwp=")
	configureDispatchHelperCommand(cmd)
	out, err := cmd.Output()
	if err == nil {
		total := 0
		seen := false
		for _, tok := range strings.Fields(string(out)) {
			if n, err := strconv.Atoi(tok); err == nil {
				total += n
				seen = true
			}
		}
		if seen {
			threads = &total
		}
	}
	return freeRAM, threads
}

func dispatchProductWorkerCount(root, product string) int {
	return len(dispatchProductWorkerPIDs(root, product))
}

// dispatchProductWorkerPIDs is the identity behind dispatchProductWorkerCount: the set of
// live worker PIDs for a product -- lease-tracked resolve/repair pidfiles, goal-run
// breadcrumbs, cmdline-marked workers (`resolve GitHub issue #` / `dos-dispatch-loop`),
// plus codex ambient sessions. The count is len() of this set; exposing the set lets the
// #3109 self-heal name the exact orphan PIDs preflight counts as unattributed_live.
func dispatchProductWorkerPIDs(root, product string) map[int]bool {
	pids := dispatchLiveResolveWorkerPIDs(filepath.Join(root, dispatchtick.RunsDirName), product)
	// Snapshot the host worker-process table ONCE per call and share it: both the
	// goal-breadcrumb and cmdline-marker passes below classify the same Win32_Process
	// rows, yet each used to spawn its OWN dispatchProbeWorkerProcessRows() -- a cold
	// PowerShell start + full-table Get-CimInstance enumeration (~0.3-1.5s on a busy
	// box) -- so a claude/unscoped preflight paid the identical scan twice every tick.
	// One scan serves both; a scan error yields nil rows, which folds each pass to the
	// same empty result the old per-caller `if err != nil { return out }` produced.
	rows, _ := dispatchProbeWorkerProcessRows()
	for pid := range dispatchLiveGoalWorkerPIDs(filepath.Join(root, dispatchGoalRunsDirName), product, rows) {
		pids[pid] = true
	}
	for pid := range dispatchCmdlineWorkerPIDs(product, rows) {
		pids[pid] = true
	}
	if product == "codex" {
		for pid := range dispatchAmbientCodexPIDsExcludingSidecarParents(pids) {
			pids[pid] = true
		}
	}
	return pids
}

// dispatchLeasedWorkerPIDs is the set of worker PIDs that hold a LIVE seat lease -- the
// resolve/repair pidfiles under the runs dir whose PID is still alive. It is the "carries
// a live lease" half of the unattributed_live predicate: a PID in the worker set but NOT
// in this set is an orphan with no seat attribution, the exact thing preflight depletes
// the pool on (#3109). Reads the same leases dispatchPreflightSeat feeds to BuildSeatPool.
func dispatchLeasedWorkerPIDs(root string) map[int]bool {
	out := map[int]bool{}
	for _, lease := range dispatchLiveSeatLeases(filepath.Join(root, dispatchtick.RunsDirName)) {
		if lease.PID > 0 {
			out[lease.PID] = true
		}
	}
	return out
}

// dispatchUnattributedWorklist is the conservative reap worklist for #3109: the sorted
// PIDs that carry the dispatch-worker marker (they are in workerPIDs) AND hold no live
// seat lease (they are absent from leasedPIDs) -- exactly the set preflight counts as
// unattributed_live. A leased worker or an unrelated (non-marker) process can never
// appear here, so the janitor can never sweep something it should not. Pure; no I/O.
func dispatchUnattributedWorklist(workerPIDs, leasedPIDs map[int]bool) []int {
	out := make([]int, 0, len(workerPIDs))
	for pid := range workerPIDs {
		if pid > 0 && !leasedPIDs[pid] {
			out = append(out, pid)
		}
	}
	sort.Ints(out)
	return out
}

// dispatchReapPID is the destructive TREE reaper the #3109 self-heal routes each orphan
// PID through. It defaults to procguard.KillPID -- a process-tree kill (native job
// termination / taskkill /T on Windows, process-group/descendant SIGKILL on POSIX) -- so
// an orphan's own descendants (the node runtime + MCP/tool subprocesses a `claude`
// spawns) are reaped too; a bare kill would leave that subtree behind and re-poison the
// count. Injectable for tests. Mirrors fleetKillPID (fleet.go) / guardChildTreeKill.
var dispatchReapPID = procguard.KillPID

// dispatchReapOutcome records the result of tree-reaping one orphan PID from the janitor
// worklist -- surfaced on the refused dispatch-tick payload as an audit trail.
type dispatchReapOutcome struct {
	PID    int    `json:"pid"`
	OK     bool   `json:"ok"`
	Detail string `json:"detail,omitempty"`
}

// dispatchReapWorklist tree-reaps every PID in a #3109 janitor worklist through
// dispatchReapPID and returns the per-PID outcome. The dispatch tick calls this only on a
// LIVE tick after preflight has already refused (next-tick recovery), never inside the
// hot admission path -- so a mis-attributed kill under lease TOCTOU is impossible.
func dispatchReapWorklist(worklist []int) []dispatchReapOutcome {
	out := make([]dispatchReapOutcome, 0, len(worklist))
	for _, pid := range worklist {
		if pid <= 0 {
			continue
		}
		ok, detail := dispatchReapPID(pid)
		out = append(out, dispatchReapOutcome{PID: pid, OK: ok, Detail: detail})
	}
	return out
}

const dispatchGoalRunsDirName = ".goal-runs"

type dispatchCodexProcessRow struct {
	PID     int    `json:"pid"`
	PPID    int    `json:"ppid"`
	Name    string `json:"name"`
	Cmdline string `json:"cmdline"`
}

func dispatchAmbientCodexProcessCount() int {
	return len(dispatchAmbientCodexPIDs())
}

func dispatchAmbientCodexPIDs() map[int]bool {
	rows, err := dispatchProbeCodexProcessRows()
	if err != nil {
		return map[int]bool{}
	}
	return dispatchCodexProcessPIDs(rows)
}

func dispatchCodexProcessPIDs(rows []dispatchCodexProcessRow) map[int]bool {
	return dispatchCodexProcessPIDsExcludingParents(rows, nil)
}

func dispatchAmbientCodexPIDsExcludingSidecarParents(sidecarPIDs map[int]bool) map[int]bool {
	rows, err := dispatchProbeCodexProcessRows()
	if err != nil {
		return map[int]bool{}
	}
	return dispatchCodexProcessPIDsExcludingParents(rows, sidecarPIDs)
}

func dispatchCodexProcessPIDsExcludingParents(rows []dispatchCodexProcessRow, excludedParents map[int]bool) map[int]bool {
	native := map[int]bool{}
	wrappers := map[int]bool{}
	parent := map[int]int{}
	for _, row := range rows {
		if row.PID <= 0 {
			continue
		}
		parent[row.PID] = row.PPID
		switch {
		case dispatchIsCodexNativeImage(row.Name):
			native[row.PID] = true
		case dispatchIsCodexNodeWrapper(row.Name, row.Cmdline):
			wrappers[row.PID] = true
		}
	}
	wrappersWithNativeChild := map[int]bool{}
	for pid := range native {
		if ppid := parent[pid]; ppid > 0 {
			wrappersWithNativeChild[ppid] = true
		}
	}
	out := map[int]bool{}
	for pid := range native {
		if excludedParents != nil && dispatchPIDHasAncestor(pid, parent, excludedParents) {
			continue
		}
		out[pid] = true
	}
	for pid := range wrappers {
		if !wrappersWithNativeChild[pid] {
			if excludedParents != nil && dispatchPIDHasAncestor(pid, parent, excludedParents) {
				continue
			}
			out[pid] = true
		}
	}
	return out
}

func dispatchPIDHasAncestor(pid int, parents map[int]int, ancestors map[int]bool) bool {
	seen := map[int]bool{}
	for pid > 0 && !seen[pid] {
		seen[pid] = true
		parent := parents[pid]
		if ancestors[parent] {
			return true
		}
		pid = parent
	}
	return false
}

const (
	dispatchWorkerCmdMarker       = "dos-dispatch-loop"
	dispatchIssueResolveCmdMarker = "resolve GitHub issue #"
)

// dispatchCmdlineWorkerPIDs classifies a caller-supplied host process table (one
// Win32_Process snapshot shared across the preflight's worker-PID passes, see
// dispatchProductWorkerPIDs) into the marker-cmdline workers for a product. Nil rows
// (the scan errored, or nothing ran) yield an empty set.
func dispatchCmdlineWorkerPIDs(product string, rows []dispatchCodexProcessRow) map[int]bool {
	out := map[int]bool{}
	for _, row := range rows {
		if row.PID <= 0 || !dispatchIsWorkerCmdline(row.Cmdline) {
			continue
		}
		if product != "" && !dispatchProcessImageMatchesProduct(row.Name, product) {
			continue
		}
		out[row.PID] = true
	}
	return out
}

func dispatchIsWorkerCmdline(cmdline string) bool {
	low := strings.ToLower(cmdline)
	return strings.Contains(low, dispatchWorkerCmdMarker) ||
		strings.Contains(low, strings.ToLower(dispatchIssueResolveCmdMarker))
}

func dispatchProcessImageMatchesProduct(name, product string) bool {
	stem := dispatchProcessNameStem(name)
	if stem == "" {
		return false
	}
	for _, backend := range dispatchProductBackends(product) {
		backend = strings.TrimSpace(backend)
		if backend != "" && (stem == backend || strings.HasPrefix(stem, backend)) {
			return true
		}
	}
	return false
}

func dispatchIsCodexNativeImage(name string) bool {
	return dispatchProcessNameStem(name) == "codex"
}

func dispatchIsCodexNodeWrapper(name, cmdline string) bool {
	if dispatchProcessNameStem(name) != "node" {
		return false
	}
	low := strings.ToLower(strings.ReplaceAll(cmdline, "\\", "/"))
	return strings.Contains(low, "@openai/codex") || strings.Contains(low, "codex/bin/codex.js")
}

func dispatchProcessNameStem(name string) string {
	base := strings.ToLower(strings.Trim(strings.TrimSpace(name), `"`))
	base = strings.ReplaceAll(base, "\\", "/")
	if i := strings.LastIndex(base, "/"); i >= 0 {
		base = base[i+1:]
	}
	for _, ext := range []string{".exe", ".cmd", ".bat"} {
		if strings.HasSuffix(base, ext) {
			base = strings.TrimSuffix(base, ext)
			break
		}
	}
	return base
}

func dispatchScanCodexProcessRowsNative() ([]dispatchCodexProcessRow, error) {
	if runtime.GOOS == "windows" {
		return dispatchScanCodexProcessRowsWindows()
	}
	return dispatchScanCodexProcessRowsPOSIX()
}

func dispatchScanWorkerProcessRowsNative() ([]dispatchCodexProcessRow, error) {
	if runtime.GOOS == "windows" {
		return dispatchScanWorkerProcessRowsWindows()
	}
	return dispatchScanWorkerProcessRowsPOSIX()
}

func dispatchScanWorkerProcessRowsWindows() ([]dispatchCodexProcessRow, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "powershell", "-NoProfile", "-NonInteractive", "-Command",
		"$rows = @(Get-CimInstance Win32_Process "+
			"-Filter \"Name = 'claude.exe' OR Name = 'opencode.exe' OR Name = 'codex.exe' OR Name = 'node.exe'\" | "+
			"Select-Object @{n='pid';e={$_.ProcessId}},@{n='ppid';e={$_.ParentProcessId}},@{n='name';e={$_.Name}},@{n='cmdline';e={$_.CommandLine}}); "+
			"$rows | ConvertTo-Json -Compress")
	configureDispatchHelperCommand(cmd)
	out, err := cmd.Output()
	if err != nil {
		return nil, err
	}
	return decodeDispatchCodexProcessRows(out)
}

func dispatchScanWorkerProcessRowsPOSIX() ([]dispatchCodexProcessRow, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "ps", "-eo", "pid=,ppid=,comm=,args=")
	configureDispatchHelperCommand(cmd)
	out, err := cmd.Output()
	if err != nil {
		return nil, err
	}
	rows := []dispatchCodexProcessRow{}
	for _, line := range strings.Split(string(out), "\n") {
		fields := strings.Fields(line)
		if len(fields) < 3 {
			continue
		}
		pid, perr := strconv.Atoi(fields[0])
		ppid, pperr := strconv.Atoi(fields[1])
		if perr != nil || pperr != nil {
			continue
		}
		name := fields[2]
		cmdline := name
		if len(fields) > 3 {
			cmdline = strings.Join(fields[3:], " ")
		}
		rows = append(rows, dispatchCodexProcessRow{PID: pid, PPID: ppid, Name: name, Cmdline: cmdline})
	}
	return rows, nil
}

func dispatchScanCodexProcessRowsWindows() ([]dispatchCodexProcessRow, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "powershell", "-NoProfile", "-NonInteractive", "-Command",
		"$rows = @(Get-CimInstance Win32_Process "+
			"-Filter \"Name = 'codex.exe' OR Name = 'node.exe'\" | "+
			"Select-Object @{n='pid';e={$_.ProcessId}},@{n='ppid';e={$_.ParentProcessId}},@{n='name';e={$_.Name}},@{n='cmdline';e={$_.CommandLine}}); "+
			"$rows | ConvertTo-Json -Compress")
	configureDispatchHelperCommand(cmd)
	out, err := cmd.Output()
	if err != nil {
		return nil, err
	}
	return decodeDispatchCodexProcessRows(out)
}

func dispatchScanCodexProcessRowsPOSIX() ([]dispatchCodexProcessRow, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "ps", "-eo", "pid=,ppid=,comm=,args=")
	configureDispatchHelperCommand(cmd)
	out, err := cmd.Output()
	if err != nil {
		return nil, err
	}
	rows := []dispatchCodexProcessRow{}
	for _, line := range strings.Split(string(out), "\n") {
		fields := strings.Fields(line)
		if len(fields) < 3 {
			continue
		}
		pid, perr := strconv.Atoi(fields[0])
		ppid, pperr := strconv.Atoi(fields[1])
		if perr != nil || pperr != nil {
			continue
		}
		name := fields[2]
		cmdline := name
		if len(fields) > 3 {
			cmdline = strings.Join(fields[3:], " ")
		}
		if dispatchIsCodexNativeImage(name) || dispatchIsCodexNodeWrapper(name, cmdline) {
			rows = append(rows, dispatchCodexProcessRow{PID: pid, PPID: ppid, Name: name, Cmdline: cmdline})
		}
	}
	return rows, nil
}

func decodeDispatchCodexProcessRows(out []byte) ([]dispatchCodexProcessRow, error) {
	text := strings.TrimSpace(string(out))
	if text == "" {
		return nil, nil
	}
	var rows []dispatchCodexProcessRow
	if err := json.Unmarshal([]byte(text), &rows); err == nil {
		return rows, nil
	}
	var one dispatchCodexProcessRow
	if err := json.Unmarshal([]byte(text), &one); err != nil {
		return nil, err
	}
	return []dispatchCodexProcessRow{one}, nil
}

func dispatchLiveResolveWorkerPIDs(runsDir, product string) map[int]bool {
	out := map[int]bool{}
	if st, err := os.Stat(runsDir); err != nil || !st.IsDir() {
		return out
	}
	for _, pidFile := range dispatchWorkerPIDFiles(runsDir) {
		if !dispatchResolvePIDRE.MatchString(filepath.Base(pidFile)) {
			continue
		}
		if product != "" && !dispatchBackendInProduct(dispatchReadBackendSidecar(pidFile), product) {
			continue
		}
		pid, ok := readPID(pidFile)
		if ok && dispatchPIDAlive(pid) {
			out[pid] = true
		}
	}
	return out
}

func dispatchWorkerPIDFiles(runsDir string) []string {
	matches := []string{}
	for _, pattern := range []string{"resolve-*.pid", "repair-*.pid"} {
		got, _ := filepath.Glob(filepath.Join(runsDir, pattern))
		matches = append(matches, got...)
	}
	sort.Strings(matches)
	return matches
}

// dispatchLiveGoalWorkerPIDs maps live goal-run breadcrumbs to worker PIDs, verifying
// each against the caller-supplied host process table (the single Win32_Process
// snapshot shared across the preflight's worker-PID passes, see
// dispatchProductWorkerPIDs). Nil rows (scan errored, or nothing ran) yield an empty
// set -- the same result the old per-caller scan-error early-return produced.
func dispatchLiveGoalWorkerPIDs(goalRunsDir, product string, rows []dispatchCodexProcessRow) map[int]bool {
	out := map[int]bool{}
	if st, err := os.Stat(goalRunsDir); err != nil || !st.IsDir() {
		return out
	}
	// tools/launch_goal_detached.ps1 is a Claude launcher; its breadcrumbs have
	// no backend sidecar, so a product-scoped count can only assign them to the
	// Claude pool. Empty product is the unscoped/global fold.
	if product != "" && product != "claude" {
		return out
	}
	byPID := map[int]dispatchCodexProcessRow{}
	for _, row := range rows {
		if row.PID > 0 {
			byPID[row.PID] = row
		}
	}
	matches, _ := filepath.Glob(filepath.Join(goalRunsDir, "*.pid"))
	sort.Strings(matches)
	for _, pidFile := range matches {
		if !dispatchGoalPIDRE.MatchString(filepath.Base(pidFile)) {
			continue
		}
		pid, ok := readPID(pidFile)
		if !ok || !dispatchPIDAlive(pid) {
			continue
		}
		row, ok := byPID[pid]
		if !ok || !dispatchProcessImageMatchesProduct(row.Name, "claude") {
			continue
		}
		// A stale breadcrumb reused by an unrelated system process must not
		// consume a worker slot. The launcher starts Claude, so require the
		// current PID to resolve to a Claude worker image before counting it.
		out[pid] = true
	}
	return out
}

func dispatchReadBackendSidecar(pidFile string) string {
	b, err := os.ReadFile(strings.TrimSuffix(pidFile, filepath.Ext(pidFile)) + ".backend")
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(b))
}

func dispatchBackendInProduct(backend, product string) bool {
	backend = strings.TrimSpace(backend)
	for _, candidate := range dispatchProductBackends(product) {
		if backend == candidate {
			return true
		}
	}
	return false
}

func dispatchProductBackends(product string) []string {
	switch product {
	case "claude":
		return []string{"claude"}
	case "opencode":
		return []string{"opencode"}
	case "codex":
		return []string{"codex"}
	default:
		return []string{product}
	}
}

var dispatchTreeBuildCommand = func(root string) (string, error) {
	goBin, err := exec.LookPath("go")
	if err != nil {
		return "", err
	}
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, goBin, "build", "-o", os.DevNull, "./cmd/fak")
	cmd.Dir = root
	configureDispatchHelperCommand(cmd)
	out, err := cmd.CombinedOutput()
	if err != nil {
		if ctx.Err() == context.DeadlineExceeded {
			// A 90s wall-clock kill is infrastructure (a loaded host, a slow disk,
			// an OOM reap), not a compiler diagnostic. Wrap it so the probe can
			// recognize the timeout via errors.Is and fail open, instead of
			// misreading a killed build as a poisoned tree and freezing the fleet.
			return string(out), fmt.Errorf(
				"tree build probe timed out after 90s: %w", context.DeadlineExceeded)
		}
		return string(out), err
	}
	return string(out), nil
}

func dispatchProbeTreeBuild(root string) dispatchtick.TreeCheck {
	out, err := dispatchTreeBuildCommand(root)
	if err == nil {
		return dispatchtick.TreeCheck{}
	}
	// Missing toolchain/probe infrastructure fails open; a real compiler diagnostic
	// names a package/file and is the poison witness. A probe root without a Go
	// module (a bare temp dir, a misconfigured root) is infrastructure-missing, not
	// a red tree, so it must fail open too -- otherwise the #3583 poison gate freezes
	// the fleet over a moduleless probe root, and every dispatch test that ticks a
	// temp workspace refuses with TREE_POISONED. `go build` names the missing module
	// two ways: "go.mod file not found ..." in a bare dir, and "cannot find main
	// module, but found .git/config ..." once the root is git-init'd (as the tick
	// test harness leaves it). Both land in combined output, not err.Error().
	//
	// A build that timed out or was killed (context.DeadlineExceeded from the 90s
	// cap, or a SIGKILL "signal: killed" from an OOM reap under fleet load)
	// produced NO compiler diagnostic -- it is infrastructure, not a poison
	// witness -- so it fails open too. Otherwise a slow host quietly converts a
	// healthy tree into a fleet-wide freeze.
	lowered := strings.ToLower(err.Error() + "\n" + out)
	if errors.Is(err, exec.ErrNotFound) ||
		errors.Is(err, context.DeadlineExceeded) ||
		strings.Contains(lowered, "executable file not found") ||
		strings.Contains(lowered, "go.mod file not found") ||
		strings.Contains(lowered, "cannot find main module") ||
		strings.Contains(lowered, "signal: killed") {
		detail := strings.TrimSpace(out)
		if detail == "" {
			detail = err.Error()
		}
		return dispatchtick.TreeCheck{Error: detail}
	}
	line := ""
	for _, candidate := range strings.Split(strings.TrimSpace(out), "\n") {
		candidate = strings.TrimSpace(candidate)
		if candidate != "" {
			line = candidate
			break
		}
	}
	if line == "" {
		line = err.Error()
	}
	return dispatchtick.TreeCheck{Poisoned: true, Package: line, Error: err.Error()}
}
