package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"time"

	configaccounts "github.com/anthony-chaudhary/fak/internal/accounts"
	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
)

func dispatchRefreshRegistry(root string, stderr io.Writer) map[string]any {
	obj, err := dispatchRunJSON(root, stderr, 120*time.Second, filepath.Join("tools", "fleet_sessions.py"), "registry")
	if err != nil {
		return map[string]any{"ok": false, "error": err.Error()}
	}
	obj["ok"] = obj["_error"] == nil
	return obj
}

func dispatchPreflight(root string, stderr io.Writer, maxWorkers int, workKind, product string) (map[string]any, error) {
	in := dispatchtick.PreflightInput{
		Workspace:     root,
		MaxWorkers:    maxWorkers,
		Host:          dispatchPreflightHost(root, stderr),
		Account:       dispatchPreflightAccount(root, stderr, workKind, product),
		Kernel:        dispatchPreflightKernel(root),
		Seat:          dispatchPreflightSeat(root, stderr, product),
		Resources:     dispatchProbeHostResources(),
		OSWorkerProcs: dispatchProbeWorkerCount(root, product),
	}
	return dispatchtick.EvaluatePreflight(in).Map(), nil
}

func dispatchPreflightHost(_ string, _ io.Writer) dispatchtick.HostCheck {
	res := dispatchtick.EvaluateProcGuard(dispatchProbeProcesses())
	return dispatchtick.HostCheck{
		Safe:         res.OK,
		Error:        res.CollectError,
		Flagged:      res.ActionableFlaggedCount,
		FlaggedNames: res.ActionableNames(),
	}
}

func dispatchPreflightAccount(root string, _ io.Writer, workKind, product string) dispatchtick.AccountCheck {
	if product == "codex" {
		return dispatchCodexAmbientAccount()
	}
	rows, err := dispatchReadAccountRoster(root)
	if err != nil {
		return dispatchtick.AccountCheck{Available: false, Error: err.Error()}
	}
	route := dispatchtick.RouteAccount(dispatchtick.AccountRouteInput{Rows: rows, Product: product, WorkKind: workKind})
	blocked := make([]string, 0, len(route.BlockedTargetAccounts))
	for _, row := range route.BlockedTargetAccounts {
		if row.Tag != "" {
			blocked = append(blocked, row.Tag)
		}
	}
	return dispatchtick.AccountCheck{
		Available:   route.OK,
		Tag:         route.Account.Tag,
		Dir:         route.Account.Dir,
		Tier:        route.SelectedTier,
		Model:       route.Account.Model,
		Reason:      route.Reason,
		Blocked:     blocked,
		LoginStatus: route.Account.LoginStatus,
		CanServe:    route.Account.CanServe,
	}
}

func dispatchCodexAmbientAccount() dispatchtick.AccountCheck {
	home, err := os.UserHomeDir()
	if err != nil || strings.TrimSpace(home) == "" {
		return dispatchtick.AccountCheck{Available: false, Reason: "could not resolve home directory for codex ambient login"}
	}
	dir := filepath.Join(home, ".codex")
	if _, err := os.Stat(filepath.Join(dir, "auth.json")); err == nil {
		return dispatchtick.AccountCheck{Available: true, Tag: "codex-ambient", Dir: dir, Tier: 1, Reason: "ambient ~/.codex login"}
	}
	return dispatchtick.AccountCheck{Available: false, Reason: "no ~/.codex/auth.json - run `codex login`"}
}

func dispatchPreflightSeat(root string, _ io.Writer, product string) dispatchtick.SeatCheck {
	if product == "codex" {
		live := dispatchAmbientCodexProcessCount()
		leased := 0
		if live > 0 {
			leased = 1
		}
		return dispatchtick.SeatCheck{
			Total:    dispatchtick.IntPtr(1),
			Free:     dispatchtick.IntPtr(1 - leased),
			Leased:   dispatchtick.IntPtr(leased),
			Depleted: leased > 0,
		}
	}
	rows, err := dispatchReadAccountRoster(root)
	if err != nil {
		return dispatchtick.SeatCheck{Error: err.Error()}
	}
	pool := dispatchtick.BuildSeatPool(rows, dispatchLiveSeatLeases(filepath.Join(root, dispatchtick.RunsDirName)), product)
	return dispatchtick.SeatCheck{
		Total:    dispatchtick.IntPtr(pool.TotalSeats),
		Free:     dispatchtick.IntPtr(pool.FreeSeats),
		Leased:   dispatchtick.IntPtr(pool.LeasedSeats),
		Depleted: pool.Depleted,
	}
}

func dispatchPreflightKernel(root string) dispatchtick.KernelCheck {
	doc, err := dispatchRunExternalJSON(root, 60*time.Second, "dos", "loop", "--workspace", root, "--json")
	if err != nil {
		return dispatchtick.KernelCheck{Error: err.Error()}
	}
	return dispatchtick.KernelCheck{
		Alive:   intPtrFromAny(doc["alive"]),
		Target:  intPtrFromAny(doc["target"]),
		Verdict: dispatchMapString(doc, "verdict"),
	}
}

var dispatchRunExternalJSON = dispatchRunExternalJSONImpl
var dispatchProbeHostResources = dispatchPreflightHostResources
var dispatchProbeWorkerCount = dispatchProductWorkerCount
var dispatchProbeProcesses = dispatchProbeProcessesNative
var dispatchProbeCodexProcessRows = dispatchScanCodexProcessRowsNative
var dispatchReadAccountRoster = dispatchReadAccountRosterNative

func dispatchReadAccountRosterNative(root string) ([]dispatchtick.AccountRow, error) {
	registryPath := dispatchAccountRegistryPath(root)
	doc, err := dispatchReadJSONFile(registryPath)
	if err != nil {
		return nil, fmt.Errorf("read account registry %s: %w", registryPath, err)
	}
	rawAccounts, _ := doc["accounts"].([]any)
	if len(rawAccounts) == 0 {
		return nil, fmt.Errorf("account registry %s has no accounts array", registryPath)
	}
	weights := dispatchLoadAccountRouteWeights(root)
	rows := make([]dispatchtick.AccountRow, 0, len(rawAccounts))
	for _, item := range rawAccounts {
		m, ok := item.(map[string]any)
		if !ok {
			continue
		}
		row := dispatchtick.AccountRow{
			Account:        dispatchStringValue(m["account"]),
			Tag:            dispatchStringValue(m["tag"]),
			Product:        dispatchStringValue(m["product"]),
			Dir:            firstString(dispatchStringValue(m["config_dir"]), dispatchStringValue(m["dir"])),
			Model:          dispatchStringValue(m["model"]),
			ModelTier:      dispatchIntValue(m["model_tier"]),
			Available:      dispatchBoolValue(m["available"]),
			BlockReason:    firstString(dispatchStringValue(m["block_reason"]), dispatchStringValue(m["reason"])),
			ActiveSessions: dispatchIntValue(m["active_sessions"]),
			LiveSessions:   dispatchIntValue(m["live_sessions"]),
			RouteWeight:    dispatchIntValue(m["route_weight"]),
			IdentityRole:   dispatchStringValue(m["identity_role"]),
			AccountUUID:    dispatchStringValue(m["account_uuid"]),
			LoginStatus:    dispatchStringValue(m["login_status"]),
		}
		if rawCanServe, ok := m["can_serve"]; ok {
			canServe := dispatchBoolValue(rawCanServe)
			row.CanServe = &canServe
		}
		if row.Account == "" && row.Dir != "" {
			row.Account = dispatchAnyOSBase(row.Dir)
		}
		if row.BlockReason == "" && dispatchBoolValue(m["blocked"]) {
			row.BlockReason = "blocked"
		}
		row = dispatchtick.NormalizeAccountRow(row)
		if row.RouteWeight == 0 {
			row.RouteWeight = dispatchAccountRouteWeight(row, weights)
		}
		dispatchApplyLoginGate(&row)
		rows = append(rows, row)
	}
	if len(rows) == 0 {
		return nil, fmt.Errorf("account registry %s has no readable account rows", registryPath)
	}
	return rows, nil
}

func dispatchApplyLoginGate(row *dispatchtick.AccountRow) {
	if row == nil || row.Product != "claude" {
		return
	}
	blocked := false
	if row.CanServe != nil && !*row.CanServe {
		blocked = true
	}
	if row.LoginStatus != "" && row.LoginStatus != string(configaccounts.LoginReady) {
		blocked = true
	}
	if !blocked {
		return
	}
	row.Available = false
	if row.BlockReason != "" {
		return
	}
	reason, _ := configaccounts.LoginReasonAction(configaccounts.LoginStatus(row.LoginStatus),
		configaccounts.Home{Name: row.Tag, Dir: row.Dir})
	if reason == "" {
		reason = "account login is not ready"
	}
	row.BlockReason = reason
}

func dispatchAccountRegistryPath(root string) string {
	if dir := strings.TrimSpace(os.Getenv("FLEET_REG_DIR")); dir != "" {
		return filepath.Join(dir, "sessions.json")
	}
	return filepath.Join(root, "tools", "_registry", "sessions.json")
}

func dispatchAccountPolicyPath(root string) string {
	if path := strings.TrimSpace(os.Getenv("FLEET_POLICY_PATH")); path != "" {
		return path
	}
	if dir := strings.TrimSpace(os.Getenv("FLEET_POLICY_DIR")); dir != "" {
		return filepath.Join(dir, "accounts_policy.json")
	}
	return filepath.Join(root, "tools", "_registry", "accounts_policy.json")
}

func dispatchLoadAccountRouteWeights(root string) map[string]int {
	doc, err := dispatchReadJSONFile(dispatchAccountPolicyPath(root))
	if err != nil {
		return nil
	}
	raw, _ := doc["route_weights"].(map[string]any)
	if len(raw) == 0 {
		return nil
	}
	weights := make(map[string]int, len(raw))
	for key, val := range raw {
		weights[key] = dispatchIntValue(val)
	}
	return weights
}

func dispatchAccountRouteWeight(row dispatchtick.AccountRow, weights map[string]int) int {
	if len(weights) == 0 {
		return 0
	}
	product := row.Product
	if product == "" {
		product = dispatchtick.ProductFromAccount(row.Account)
	}
	tag := row.Tag
	if tag == "" {
		tag = dispatchtick.TagFromAccount(row.Account)
	}
	for _, key := range []string{row.Account, product + ":" + row.Account, product + ":" + tag, tag, product} {
		if key == "" {
			continue
		}
		if weight, ok := weights[key]; ok {
			return weight
		}
	}
	return 0
}

func dispatchReadJSONFile(path string) (map[string]any, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	dec := json.NewDecoder(f)
	dec.UseNumber()
	var doc map[string]any
	if err := dec.Decode(&doc); err != nil {
		return nil, err
	}
	if doc == nil {
		return nil, errors.New("json document is not an object")
	}
	return doc, nil
}

func dispatchLiveSeatLeases(runsDir string) []dispatchtick.SeatLease {
	st, err := os.Stat(runsDir)
	if err != nil || !st.IsDir() {
		return nil
	}
	matches, _ := filepath.Glob(filepath.Join(runsDir, "resolve-*.pid"))
	sort.Strings(matches)
	leases := make([]dispatchtick.SeatLease, 0, len(matches))
	for _, pidFile := range matches {
		if !dispatchResolvePIDRE.MatchString(filepath.Base(pidFile)) {
			continue
		}
		pid, ok := readPID(pidFile)
		if !ok || !dispatchPIDAlive(pid) {
			continue
		}
		stem := strings.TrimSuffix(pidFile, filepath.Ext(pidFile))
		lease := dispatchtick.SeatLease{Worker: filepath.Base(stem), PID: pid}
		if b, err := os.ReadFile(stem + dispatchtick.AccountSidecarSuffix); err == nil {
			var rec map[string]any
			if json.Unmarshal(b, &rec) == nil {
				lease.Tag = dispatchStringValue(rec["tag"])
				lease.Dir = dispatchStringValue(rec["dir"])
			}
		}
		leases = append(leases, lease)
	}
	return leases
}

func dispatchRunExternalJSONImpl(root string, timeout time.Duration, name string, args ...string) (map[string]any, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, name, args...)
	cmd.Dir = root
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

func dispatchPreflightHostResources() dispatchtick.HostResources {
	cores := runtime.NumCPU()
	freeRAM, threads := dispatchRAMAndThreads()
	return dispatchtick.HostResources{Cores: &cores, FreeRAMMB: freeRAM, TotalThreads: threads}
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
	pids := dispatchLiveResolveWorkerPIDs(filepath.Join(root, dispatchtick.RunsDirName), product)
	if product == "codex" {
		for pid := range dispatchAmbientCodexPIDs() {
			pids[pid] = true
		}
	}
	return len(pids)
}

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
		out[pid] = true
	}
	for pid := range wrappers {
		if !wrappersWithNativeChild[pid] {
			out[pid] = true
		}
	}
	return out
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
	matches, _ := filepath.Glob(filepath.Join(runsDir, "resolve-*.pid"))
	for _, pidFile := range matches {
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
