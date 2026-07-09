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

	"github.com/anthony-chaudhary/fak/internal/accounts"
	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
	"github.com/anthony-chaudhary/fak/internal/fleetaccounts"
	"github.com/anthony-chaudhary/fak/internal/procguard"
	"github.com/anthony-chaudhary/fak/internal/turntaxmeter"
)

const fallbackCodexOAuthSessions = 10

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
		Budgets:       dispatchtick.DefaultHostBudgets(),
		OSWorkerProcs: dispatchProbeWorkerCount(root, product),
	}
	// The fifth cap term (#2221, G3 of epic #2218): fold the MEASURED guard-hook
	// latency rollup UP into admission so a slow kernel earns spawn reluctance. The
	// four in-struct terms only flow caps DOWN; this composes gate health on top and
	// can only lower the effective cap, never raise it.
	res := dispatchtick.EvaluatePreflight(in)
	res = dispatchtick.ApplyGateBackpressure(res, dispatchPreflightGate(root))
	// The rate_budget cap term (docs/safe-to-raise-cap-checklist.md): fold the MEASURED,
	// backend-scoped burst of GENUINE concurrency rate-limit worker exits UP into
	// admission so a fleet storming a throttled seat backs off (and routes to another
	// provider) instead of re-storming it. Fake 429s -- weekly caps, model caps, login
	// walls -- are excluded by the reason=rate_limit taxonomy filter; it only lowers the
	// effective cap, so a zero-signal fold is byte-identical to before.
	res = dispatchtick.ApplyRateLimitBackpressure(res, dispatchPreflightRateLimit(root, product))
	out := res.Map()
	// #3109 self-heal: preflight is otherwise refuse-only on unattributed_live -- it
	// counts orphaned worker PIDs (a botched teardown's `claude` descendant still
	// carrying the dispatch marker but holding NO seat lease) as pool depletion and
	// wedges dispatch until a separately-scheduled janitor clears them. When the pool
	// shows unattributed_live > 0, surface those exact PIDs as a janitor worklist
	// (mirroring how `fak garden tick` surfaces orphan-run worklists) so the tick can
	// tree-reap them via procguard.KillPID and the pool recovers on its own next tick.
	// The predicate is the SAME one preflight already uses to COUNT them -- dispatch
	// marker AND no live lease -- so a leased or unrelated process can never be swept.
	// Observation only here (no side effect); the reap is next-tick / live-gated in the
	// dispatch tick, never in this hot admission path (mis-attributed-kill TOCTOU).
	if res.Seat.UnattributedLive > 0 {
		if worklist := dispatchUnattributedWorklist(dispatchProductWorkerPIDs(root, product), dispatchLeasedWorkerPIDs(root)); len(worklist) > 0 {
			out["janitor_worklist"] = worklist
		}
	}
	return out, nil
}

// dispatchPreflightGate folds the workspace's MEASURED guard-hook latency rollup into
// the gate-health state the fifth preflight cap term consults. It reuses the same
// hook-observation streams `fak hooklat` discovers and folds; a missing/unreadable
// stream or a thin sample simply yields a zero-pressure GateCheck (the fold abstains),
// so preflight never grows an error path for an observability signal. The
// overhead-budget breach input stays false here until a standing-breach ledger exists
// to read -- the honest fence: the signal is the measured rollup, never a self-report.
func dispatchPreflightGate(root string) dispatchtick.GateCheck {
	var obs []turntaxmeter.HookObservation
	for _, p := range discoverHookObservationStreams(root) {
		f, err := os.Open(p)
		if err != nil {
			continue
		}
		rows, _, perr := turntaxmeter.ParseHookObservations(f)
		f.Close()
		if perr != nil {
			continue
		}
		obs = append(obs, rows...)
	}
	// Window the fold to LIVE kernel health before judging the tail. Backpressure is
	// a "hold until the tail recovers" signal, but a fold over the ALL-TIME stream can
	// never recover: a past slow period stays in the denominator forever, so the gate
	// would red permanently on a stream that has accumulated one. FAK_GATE_WINDOW (a
	// duration; default dispatchGateDefaultWindow) scopes it to recent rows the same
	// way `fak hooklat --since` does; "0"/"off" restores the whole-stream fold.
	if window := dispatchGateWindow(); window > 0 {
		obs = turntaxmeter.FilterHookObservationsSince(obs, time.Now().Add(-window))
	}
	rollup := turntaxmeter.FoldHookLatency(obs)
	return dispatchtick.GateCheck{
		Hook:         rollup.Total,
		HookBudgetMS: turntaxmeter.DefaultHookP99BudgetMS,
		MinWorkers:   dispatchGateMinWorkers(),
	}
}

// dispatchGateDefaultWindow scopes the gate's hook-latency fold to recent kernel
// health. Two hours is generous enough to hold a trustworthy tail (n well past
// turntaxmeter.MinHookAlarmSamples on any live fleet) while still letting a resolved
// regression age out so the gate can recover -- the property the all-time fold lacked.
const dispatchGateDefaultWindow = 2 * time.Hour

// dispatchGateWindow resolves the gate's observation lookback from FAK_GATE_WINDOW: a
// Go duration (e.g. "90m") windows the fold; "0" or "off" folds the whole stream; an
// empty or unparseable value falls back to dispatchGateDefaultWindow.
func dispatchGateWindow() time.Duration {
	raw := strings.TrimSpace(os.Getenv("FAK_GATE_WINDOW"))
	switch {
	case raw == "":
		return dispatchGateDefaultWindow
	case raw == "0" || strings.EqualFold(raw, "off"):
		return 0
	}
	if d, err := time.ParseDuration(raw); err == nil && d > 0 {
		return d
	}
	return dispatchGateDefaultWindow
}

// dispatchGateMinWorkers resolves the gate cold-start floor from FAK_GATE_MIN_WORKERS,
// falling back to dispatchtick.DefaultGateMinWorkers. A zero or negative override is
// clamped by GateCheck.floor() back to the default, so the deadlock-at-zero the floor
// forbids cannot be reintroduced through the env.
func dispatchGateMinWorkers() int {
	raw := strings.TrimSpace(os.Getenv("FAK_GATE_MIN_WORKERS"))
	if raw == "" {
		return dispatchtick.DefaultGateMinWorkers
	}
	if n, err := strconv.Atoi(raw); err == nil && n >= 0 {
		return n
	}
	return dispatchtick.DefaultGateMinWorkers
}

// dispatchRateLimitDefaultWindow scopes the rate_budget term's lookback to a RECENT
// burst. A concurrency 429 is transient (a shared seat is momentarily saturated), so the
// window must be short enough that an aged burst stops holding the backend once the
// storm clears -- 15 minutes holds enough recent worker exits to see a genuine cluster
// while letting it age out on its own, the recovery property a whole-stream fold lacks.
const dispatchRateLimitDefaultWindow = 15 * time.Minute

// dispatchRateLimitWindow resolves the rate_budget lookback from FAK_RATELIMIT_WINDOW: a
// Go duration (e.g. "20m") windows the count; "0" or "off" DISABLES the term (zero-value
// fold, a no-op); an empty or unparseable value falls back to the default.
func dispatchRateLimitWindow() time.Duration {
	raw := strings.TrimSpace(os.Getenv("FAK_RATELIMIT_WINDOW"))
	switch {
	case raw == "":
		return dispatchRateLimitDefaultWindow
	case raw == "0" || strings.EqualFold(raw, "off"):
		return 0
	}
	if d, err := time.ParseDuration(raw); err == nil && d > 0 {
		return d
	}
	return dispatchRateLimitDefaultWindow
}

// dispatchRateLimitThreshold resolves the burst arming threshold from
// FAK_RATELIMIT_MIN_429, falling back to dispatchtick.DefaultRateLimitMin429. A zero or
// negative override is ignored (kept at the default) so the term cannot be armed on a
// single stray 429.
func dispatchRateLimitThreshold() int {
	raw := strings.TrimSpace(os.Getenv("FAK_RATELIMIT_MIN_429"))
	if raw == "" {
		return dispatchtick.DefaultRateLimitMin429
	}
	if n, err := strconv.Atoi(raw); err == nil && n > 0 {
		return n
	}
	return dispatchtick.DefaultRateLimitMin429
}

// dispatchRateLimitMinWorkers resolves the cold-start floor from FAK_RATELIMIT_MIN_WORKERS,
// falling back to dispatchtick.DefaultRateLimitMinWorkers. A negative override is ignored;
// the pure fold's floor() re-clamps a zero back to the default, so the one-probe liveness
// carve-out cannot be removed through the env.
func dispatchRateLimitMinWorkers() int {
	raw := strings.TrimSpace(os.Getenv("FAK_RATELIMIT_MIN_WORKERS"))
	if raw == "" {
		return dispatchtick.DefaultRateLimitMinWorkers
	}
	if n, err := strconv.Atoi(raw); err == nil && n >= 0 {
		return n
	}
	return dispatchtick.DefaultRateLimitMinWorkers
}

// dispatchPreflightRateLimit folds the MEASURED, backend-scoped burst of GENUINE
// concurrency rate-limit worker exits into the rate_budget admission term
// (dispatchtick.ApplyRateLimitBackpressure). It counts the finished worker slots whose
// .witness sidecar graded CLAIM_NO_COMMIT with reason=rate_limit (a transient 429/529
// overload the classifier read from the worker log tail -- never a self-report) within a
// recent window on THIS product's backend.
//
// The DISAMBIGUATION is the taxonomy filter (the load-bearing correctness of this term):
// usage_cap (weekly/quota), model_unknown (model cap), and auth_wall (login) are DISTINCT
// classifier reasons and are never counted here, because backing off concurrency does not
// clear any of them -- they are owned by the seat gate, the Layer-2 downgrade ladder, and
// the auth flow. Only reason=rate_limit -- the residual transient-overload class the
// classifier's precedence leaves after skimming those off -- drives concurrency backoff.
//
// Fail-open and byte-identical when idle: a disabled window (FAK_RATELIMIT_WINDOW=0/off),
// a missing runs dir, or zero recent rate_limit exits yields the zero-value check, a no-op
// fold that leaves the preflight untouched.
func dispatchPreflightRateLimit(root, product string) dispatchtick.RateLimitCheck {
	window := dispatchRateLimitWindow()
	if window <= 0 {
		return dispatchtick.RateLimitCheck{}
	}
	runsDir := filepath.Join(root, dispatchtick.RunsDirName)
	if st, err := os.Stat(runsDir); err != nil || !st.IsDir() {
		return dispatchtick.RateLimitCheck{}
	}
	cutoff := time.Now().Add(-window)
	matches := []string{}
	for _, pattern := range []string{"resolve-*" + dispatchtick.WitnessSidecarSuffix, "repair-*" + dispatchtick.WitnessSidecarSuffix} {
		got, _ := filepath.Glob(filepath.Join(runsDir, pattern))
		matches = append(matches, got...)
	}
	count := 0
	for _, wf := range matches {
		info, err := os.Stat(wf)
		if err != nil || info.ModTime().Before(cutoff) {
			continue // aged out of the window -> not part of the current burst
		}
		stem := strings.TrimSuffix(wf, dispatchtick.WitnessSidecarSuffix)
		if product != "" {
			backend := ""
			if b, err := os.ReadFile(stem + ".backend"); err == nil {
				backend = strings.TrimSpace(string(b))
			}
			if !dispatchBackendInProduct(backend, product) {
				continue // a different backend's 429s must not throttle this one
			}
		}
		doc, err := dispatchReadJSONFile(wf)
		if err != nil {
			continue
		}
		if dispatchStringValue(doc["claim"]) != dispatchtick.ClaimNoCommit {
			continue
		}
		// The disambiguation: ONLY reason=rate_limit -- a weekly/usage cap, a model cap,
		// or a login wall is a different reason and is deliberately not counted.
		if dispatchStringValue(doc["reason"]) != dispatchtick.NoCommitRateLimit {
			continue
		}
		count++
	}
	return dispatchtick.RateLimitCheck{
		Recent:     count,
		Window:     window,
		Threshold:  dispatchRateLimitThreshold(),
		MinWorkers: dispatchRateLimitMinWorkers(),
	}
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

func dispatchCodexOAuthSessionCap() int {
	raw := strings.TrimSpace(os.Getenv("FAK_CODEX_OAUTH_SESSIONS"))
	if raw == "" {
		return fallbackCodexOAuthSessions
	}
	n, err := strconv.Atoi(raw)
	if err != nil || n <= 0 {
		return fallbackCodexOAuthSessions
	}
	return n
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func dispatchPreflightSeat(root string, _ io.Writer, product string) dispatchtick.SeatCheck {
	if product == "codex" {
		total := dispatchCodexOAuthSessionCap()
		live := dispatchAmbientCodexProcessCount()
		leased := live
		if leased > total {
			leased = total
		}
		return dispatchtick.SeatCheck{
			Total:    dispatchtick.IntPtr(total),
			Free:     dispatchtick.IntPtr(maxInt(0, total-live)),
			Leased:   dispatchtick.IntPtr(leased),
			Depleted: live >= total,
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
var dispatchProbeWorkerProcessRows = dispatchScanWorkerProcessRowsNative
var dispatchReadAccountRoster = dispatchReadAccountRosterNative

// dispatchReadAccountRosterNative builds the dispatch account roster and then drops
// seats with an ACTIVE account cooldown from the servable pool. Both dispatch pickers
// (dispatchtick.RouteAccount for a bare tick and dispatchtick.AllocateWave for a wave)
// admit only rows whose Available flag survives the login gate; without this overlay
// they route onto a seat the guard already cooled after a usage cap / rehome, which
// immediately 429s and burns the slot. This mirrors the cooldown overlay that
// Registry.LoginReportAt already applies for `fak accounts` and guard rotation
// (internal/accounts/login.go), keyed on the same uuid: bucket the guard writes.
func dispatchReadAccountRosterNative(root string) ([]dispatchtick.AccountRow, error) {
	rows, err := dispatchBuildAccountRoster(root)
	if err != nil {
		return nil, err
	}
	return dispatchApplyAccountCooldown(rows, dispatchLoadAccountCooldownStore(), time.Now()), nil
}

// dispatchLoadAccountCooldownStore loads the shared account-cooldown store, failing
// open (nil) when it is absent or unreadable so a missing store never wedges dispatch.
func dispatchLoadAccountCooldownStore() *accounts.CooldownStore {
	store, err := accounts.LoadCooldownStore(defaultCooldownStorePath())
	if err != nil {
		return nil
	}
	return store
}

// dispatchApplyAccountCooldown marks every roster row whose upstream account holds an
// active cooldown at now as unservable (Available=false, CanServe=false) with a block
// reason, so RouteAccount/AllocateWave skip it. A nil store leaves the roster untouched.
func dispatchApplyAccountCooldown(rows []dispatchtick.AccountRow, store *accounts.CooldownStore, now time.Time) []dispatchtick.AccountRow {
	if store == nil {
		return rows
	}
	for i := range rows {
		uuid := strings.TrimSpace(rows[i].AccountUUID)
		if uuid == "" {
			continue
		}
		entry, cooled := store.CooledDown(accounts.UUIDBucketKey(uuid), now)
		if !cooled {
			continue
		}
		rows[i].Available = false
		canServe := false
		rows[i].CanServe = &canServe
		if strings.TrimSpace(rows[i].BlockReason) == "" {
			rows[i].BlockReason = dispatchCooldownBlockReason(entry, now)
		}
	}
	return rows
}

// dispatchCooldownBlockReason renders a concise, audit-friendly block reason for a
// cooled seat, naming the cooldown kind and the reset instant.
func dispatchCooldownBlockReason(e accounts.CooldownEntry, now time.Time) string {
	remaining := e.ResetAt.Sub(now)
	if remaining < 0 {
		remaining = 0
	}
	return fmt.Sprintf("account in cooldown (%s) until %s (%s remaining)",
		e.Kind, e.ResetAt.UTC().Format(time.RFC3339), remaining.Round(time.Minute))
}

func dispatchBuildAccountRoster(root string) ([]dispatchtick.AccountRow, error) {
	if rows := dispatchAuthoritativeAccountRows(root); len(rows) > 0 {
		return rows, nil
	}
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
		rows = append(rows, row)
	}
	if len(rows) == 0 {
		return nil, fmt.Errorf("account registry %s has no readable account rows", registryPath)
	}
	return rows, nil
}

func dispatchAuthoritativeAccountRows(root string) []dispatchtick.AccountRow {
	toolsDir := filepath.Join(root, "tools")
	paths := fleetaccounts.ResolvePaths(toolsDir)
	pol := fleetaccounts.LoadPolicy(paths)
	reg := fleetaccounts.LoadRegistry(paths.RegistryPath)
	accounts := fleetaccounts.AnnotatedRoster(paths.Home, paths.ConfigHome, pol, reg)
	weights := dispatchLoadAccountRouteWeights(root)
	rows := make([]dispatchtick.AccountRow, 0, len(accounts))
	for _, acct := range accounts {
		row := dispatchAccountRowFromFleetAccount(acct)
		if row.Account == "" && row.Dir != "" {
			row.Account = dispatchAnyOSBase(row.Dir)
		}
		row = dispatchtick.NormalizeAccountRow(row)
		if row.RouteWeight == 0 {
			row.RouteWeight = dispatchAccountRouteWeight(row, weights)
		}
		rows = append(rows, row)
	}
	return rows
}

func dispatchAccountRowFromFleetAccount(acct fleetaccounts.Account) dispatchtick.AccountRow {
	row := dispatchtick.AccountRow{
		Account:      acct.Account,
		Tag:          acct.Tag,
		Product:      acct.Product,
		Dir:          acct.Dir,
		Kind:         string(acct.Kind),
		Available:    dispatchBoolPtrValue(acct.Available),
		BlockReason:  firstString(dispatchStringPtrValue(acct.BlockReason), acct.Reason),
		RouteWeight:  dispatchIntPtrValue(acct.RouteWeight),
		IdentityRole: dispatchStringPtrValue(acct.IdentityRole),
		AccountUUID:  dispatchStringPtrValue(acct.AccountUUID),
		LoginStatus:  dispatchStringPtrValue(acct.LoginStatus),
	}
	if acct.Model != nil {
		row.Model = *acct.Model
	}
	if acct.ModelTier != nil {
		row.ModelTier = *acct.ModelTier
	}
	if acct.ActiveSessions != nil {
		row.ActiveSessions = *acct.ActiveSessions
	}
	if acct.LiveSessions != nil {
		row.LiveSessions = *acct.LiveSessions
	}
	if acct.CanServe != nil {
		canServe := *acct.CanServe
		row.CanServe = &canServe
	}
	return row
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

func dispatchStringPtrValue(p *string) string {
	if p == nil {
		return ""
	}
	return *p
}

func dispatchIntPtrValue(p *int) int {
	if p == nil {
		return 0
	}
	return *p
}

func dispatchBoolPtrValue(p *bool) bool {
	return p != nil && *p
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
	matches := dispatchWorkerPIDFiles(runsDir)
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
	return len(dispatchProductWorkerPIDs(root, product))
}

// dispatchProductWorkerPIDs is the identity behind dispatchProductWorkerCount: the set of
// live worker PIDs for a product -- lease-tracked resolve/repair pidfiles, goal-run
// breadcrumbs, cmdline-marked workers (`resolve GitHub issue #` / `dos-dispatch-loop`),
// plus codex ambient sessions. The count is len() of this set; exposing the set lets the
// #3109 self-heal name the exact orphan PIDs preflight counts as unattributed_live.
func dispatchProductWorkerPIDs(root, product string) map[int]bool {
	pids := dispatchLiveResolveWorkerPIDs(filepath.Join(root, dispatchtick.RunsDirName), product)
	for pid := range dispatchLiveGoalWorkerPIDs(filepath.Join(root, dispatchGoalRunsDirName), product) {
		pids[pid] = true
	}
	for pid := range dispatchCmdlineWorkerPIDs(product) {
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

func dispatchCmdlineWorkerPIDs(product string) map[int]bool {
	out := map[int]bool{}
	rows, err := dispatchProbeWorkerProcessRows()
	if err != nil {
		return out
	}
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

func dispatchLiveGoalWorkerPIDs(goalRunsDir, product string) map[int]bool {
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
	rows, err := dispatchProbeWorkerProcessRows()
	if err != nil {
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
