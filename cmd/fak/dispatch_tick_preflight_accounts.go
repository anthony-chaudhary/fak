package main

// dispatch_tick_preflight_accounts.go — the ACCOUNT-roster seam of dispatch preflight, split
// out of dispatch_tick_preflight.go (#5849) so that file stays under the god-file ceiling.
// Everything here answers "which accounts may this tick route to, and at what weight": the
// native roster build, cooldown subtraction, usage-cap census, and the route-weight policy
// read. The probe wiring and the gate/rate-limit/churn preflights stay in the parent file.

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/accounts"
	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
	"github.com/anthony-chaudhary/fak/internal/fleetaccounts"
)

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
	// This is an ADMISSION path (it drops cooled seats from the routable pool), so an
	// unreadable store warns on os.Stderr like the resolve/launch seams do (#6027) —
	// stdout stays clean for the preflight's JSON.
	return dispatchApplyAccountCooldown(rows, loadCooldownStoreFailOpen("fak dispatch", os.Stderr), time.Now()), nil
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

// dispatchUsageCapAdvisoryThreshold resolves the usage-cap advisory's arming floor from
// FAK_USAGECAP_ADVISORY_MIN (a positive integer count of usage-limit-cooled accounts),
// falling back to dispatchtick.DefaultUsageCapAdvisoryMin on empty/zero/unparseable input.
func dispatchUsageCapAdvisoryThreshold() int {
	raw := strings.TrimSpace(os.Getenv("FAK_USAGECAP_ADVISORY_MIN"))
	if raw == "" {
		return dispatchtick.DefaultUsageCapAdvisoryMin
	}
	if n, err := strconv.Atoi(raw); err == nil && n > 0 {
		return n
	}
	return dispatchtick.DefaultUsageCapAdvisoryMin
}

// dispatchPreflightUsageCap builds the ADVISORY-ONLY usage-cap census the preflight
// surfaces (dispatchtick.UsageCapAdvisory). Unlike the rate_budget term it folds nothing
// into the cap -- it reads the AUTHORITATIVE account-cooldown store (the one signal that
// distinguishes a usage cap from the transient 429 the witness classifier often mislabels
// it as) and reports how many of the backend's routable accounts sit under an active
// usage-limit-kind cooldown, plus the soonest reset. FreeSeats is carried from the seat
// gate's already-computed pool as context only. Fail-open: a codex backend, an unreadable
// roster, or a nil cooldown store yields a zero census (not armed), so the advisory never
// grows an error path and a fleet without the store is byte-identical to before.
func dispatchPreflightUsageCap(root, product string, seat dispatchtick.SeatCheck) dispatchtick.UsageCapAdvisory {
	// Seat-cooldown usage caps are a Claude-seat concept; codex carries no usage-limit
	// cooldown store, so the census abstains rather than counting an unrelated store.
	if product == "codex" {
		return dispatchtick.UsageCapAdvisory{}
	}
	// Silent fold here on purpose: this census is ADVISORY ONLY and runs on the same tick
	// as the roster above, which already warned about the same file — a second copy per
	// tick would be noise, not signal (#6027).
	store := loadCooldownStoreFailOpen("fak dispatch", nil)
	if store == nil {
		return dispatchtick.UsageCapAdvisory{}
	}
	rows, err := dispatchBuildAccountRoster(root)
	if err != nil {
		return dispatchtick.UsageCapAdvisory{}
	}
	free := 0
	if seat.Free != nil {
		free = *seat.Free
	}
	return dispatchUsageCapCensus(rows, store, product, time.Now(), dispatchUsageCapAdvisoryThreshold(), free)
}

// dispatchUsageCapCensus is the pure-ish counting core of the usage-cap advisory: over the
// backend's routable roster it counts unique accounts (deduped by uuid) and, among them,
// how many sit under an ACTIVE usage-limit-kind cooldown at now, tracking the soonest
// reset. Kept separate from the store/roster loading so it is testable in-memory with a
// seeded store (mirroring dispatchApplyAccountCooldown). A nil store yields a zero-capped
// census (nothing cooled), so the caller stays fail-open.
func dispatchUsageCapCensus(rows []dispatchtick.AccountRow, store *accounts.CooldownStore, product string, now time.Time, threshold, freeSeats int) dispatchtick.UsageCapAdvisory {
	total, capped := 0, 0
	var earliest time.Time
	seen := map[string]bool{}
	for _, raw := range rows {
		row := dispatchtick.NormalizeAccountRow(raw)
		// Scope the census to this backend's product, mirroring BuildSeatPool's filter, so
		// another product's usage caps never colour this backend's advisory.
		if product != "" && product != "all" && row.Product != product {
			continue
		}
		uuid := strings.TrimSpace(row.AccountUUID)
		if uuid == "" || seen[uuid] {
			continue // dedup by account: one usage cap removes the account, counted once
		}
		seen[uuid] = true
		total++
		if store == nil {
			continue
		}
		entry, cooled := store.CooledDown(accounts.UUIDBucketKey(uuid), now)
		if !cooled || entry.Kind != accounts.CooldownUsageLimit {
			continue
		}
		capped++
		if earliest.IsZero() || entry.ResetAt.Before(earliest) {
			earliest = entry.ResetAt
		}
	}
	return dispatchtick.UsageCapAdvisory{
		Capped:        capped,
		Accounts:      total,
		FreeSeats:     freeSeats,
		EarliestReset: earliest,
		Threshold:     threshold,
		Now:           now,
	}
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
