package fleetaccounts

import (
	"fmt"
	"sort"
	"strings"
)

const StatusReportSchema = "fleet-account-status/1"

// StatusOptions selects the slice of the roster a status report folds.
type StatusOptions struct {
	Filter  StatusFilter
	GroupBy []string
}

// StatusFilter is intentionally made from operator vocabulary rather than registry-only
// fields: provider is derived from model/account naming so a host can ask for "groq"
// without first extending the account schema.
type StatusFilter struct {
	Product  string `json:"product,omitempty"`
	Provider string `json:"provider,omitempty"`
	Tier     int    `json:"tier,omitempty"`
	State    string `json:"state,omitempty"`
	Account  string `json:"account,omitempty"`
	Model    string `json:"model,omitempty"`
}

// StatusReport is the compact account/switcher rollup behind `fak fleet-accounts status`.
type StatusReport struct {
	Schema      string          `json:"schema"`
	Node        string          `json:"node,omitempty"`
	GeneratedAt string          `json:"generated_at,omitempty"`
	Filters     StatusFilter    `json:"filters"`
	GroupBy     []string        `json:"group_by"`
	Totals      StatusTotals    `json:"totals"`
	Rollups     []StatusRollup  `json:"rollups"`
	Accounts    []StatusAccount `json:"accounts"`
	Warnings    []string        `json:"warnings"`
}

// StatusTotals folds both account counts and slot counts. Slots are counted once per
// distinct rate-limit pool; duplicate/shared dirs remain visible as account rows but do
// not double-count capacity.
type StatusTotals struct {
	Accounts             int `json:"accounts"`
	WorkerAccounts       int `json:"worker_accounts"`
	RoutableAccounts     int `json:"routable_accounts"`
	ReadyAccounts        int `json:"ready_accounts"`
	BlockedAccounts      int `json:"blocked_accounts"`
	UsageBlockedAccounts int `json:"usage_blocked_accounts"`
	AuthBlockedAccounts  int `json:"auth_blocked_accounts"`
	DuplicateAccounts    int `json:"duplicate_accounts"`
	ExcludedAccounts     int `json:"excluded_accounts"`
	NonAccounts          int `json:"non_accounts"`
	Pools                int `json:"pools"`
	TotalSlots           int `json:"total_slots"`
	FreeSlots            int `json:"free_slots"`
	LeasedSlots          int `json:"leased_slots"`
	BlockedSlots         int `json:"blocked_slots"`
}

// StatusAccount is the per-account row in the report. CapacityCounted=false means the row
// is visible but its pool was already counted by a better representative.
type StatusAccount struct {
	Node            string          `json:"node,omitempty"`
	Account         string          `json:"account"`
	Tag             string          `json:"tag"`
	Product         string          `json:"product"`
	Provider        string          `json:"provider"`
	Agent           string          `json:"agent,omitempty"`
	ModelTier       *int            `json:"model_tier,omitempty"`
	Model           string          `json:"model,omitempty"`
	Kind            string          `json:"kind"`
	State           string          `json:"state"`
	Pool            string          `json:"pool,omitempty"`
	CapacityCounted bool            `json:"capacity_counted"`
	SessionCap      int             `json:"session_cap,omitempty"`
	LeasedSlots     int             `json:"leased_slots,omitempty"`
	FreeSlots       int             `json:"free_slots,omitempty"`
	BlockedSlots    int             `json:"blocked_slots,omitempty"`
	ActiveSessions  int             `json:"active_sessions"`
	LiveSessions    int             `json:"live_sessions"`
	Reason          string          `json:"reason,omitempty"`
	DiscoverySource DiscoverySource `json:"discovery_source,omitempty"`
	RootState       RootState       `json:"root_state,omitempty"`
	BlockKind       string          `json:"block_kind,omitempty"`
	Reset           string          `json:"reset,omitempty"`
	Weekly          string          `json:"weekly,omitempty"`
	LoginStatus     *string         `json:"login_status,omitempty"`
	CanServe        *bool           `json:"can_serve,omitempty"`
	StatusSource    string          `json:"status_source,omitempty"`
	RegistryAgeMin  *float64        `json:"registry_age_min,omitempty"`
}

// StatusRollup is one grouped status row.
type StatusRollup struct {
	Key                  string            `json:"key"`
	Label                string            `json:"label"`
	Dimensions           map[string]string `json:"dimensions"`
	Accounts             int               `json:"accounts"`
	WorkerAccounts       int               `json:"worker_accounts"`
	ReadyAccounts        int               `json:"ready_accounts"`
	BlockedAccounts      int               `json:"blocked_accounts"`
	UsageBlockedAccounts int               `json:"usage_blocked_accounts"`
	AuthBlockedAccounts  int               `json:"auth_blocked_accounts"`
	Pools                int               `json:"pools"`
	TotalSlots           int               `json:"total_slots"`
	FreeSlots            int               `json:"free_slots"`
	LeasedSlots          int               `json:"leased_slots"`
	BlockedSlots         int               `json:"blocked_slots"`
	Mixed                bool              `json:"mixed"`
}

// BuildStatusReport folds annotated account rows plus live leases into filterable rollups.
func BuildStatusReport(rows []Account, leases []Lease, opts StatusOptions) StatusReport {
	groupBy := normalizeGroupBy(opts.GroupBy)
	routable := make([]Account, 0)
	for _, row := range rows {
		if RoutableWorker(row) {
			routable = append(routable, row)
		}
	}
	leaseWorkers, _ := leaseWorkersByPool(routable, leases)
	countedRows := map[string]Account{}
	for _, row := range uniquePoolAccounts(routable) {
		countedRows[PoolKey(row)] = row
	}

	rep := StatusReport{
		Schema:  StatusReportSchema,
		Filters: normalizeStatusFilter(opts.Filter),
		GroupBy: groupBy,
	}
	rollupByKey := map[string]*StatusRollup{}
	for _, row := range rows {
		acct := statusAccount(row, leaseWorkers[PoolKey(row)], statusCountsCapacity(row, countedRows))
		if !statusMatches(acct, rep.Filters) {
			continue
		}
		rep.Accounts = append(rep.Accounts, acct)
		addStatusTotals(&rep.Totals, acct)
		ru := statusRollupFor(rollupByKey, acct, groupBy)
		addStatusRollup(ru, acct)
	}
	for _, ru := range rollupByKey {
		ru.Mixed = ru.FreeSlots > 0 && ru.BlockedSlots > 0
		rep.Rollups = append(rep.Rollups, *ru)
	}
	sort.SliceStable(rep.Rollups, func(i, j int) bool {
		return rep.Rollups[i].Key < rep.Rollups[j].Key
	})
	sort.SliceStable(rep.Accounts, func(i, j int) bool {
		return statusAccountLess(rep.Accounts[i], rep.Accounts[j])
	})
	rep.Warnings = statusWarnings(rep)
	if rep.Rollups == nil {
		rep.Rollups = []StatusRollup{}
	}
	if rep.Accounts == nil {
		rep.Accounts = []StatusAccount{}
	}
	if rep.Warnings == nil {
		rep.Warnings = []string{}
	}
	return rep
}

func statusCountsCapacity(row Account, countedRows map[string]Account) bool {
	if !RoutableWorker(row) {
		return false
	}
	rep, ok := countedRows[PoolKey(row)]
	if !ok {
		return false
	}
	return rep.Account == row.Account && rep.Dir == row.Dir
}

func normalizeStatusFilter(f StatusFilter) StatusFilter {
	f.Product = strings.ToLower(strings.TrimSpace(f.Product))
	if f.Product == "all" {
		f.Product = ""
	}
	f.Provider = providerKey(f.Provider)
	if f.Provider == "all" {
		f.Provider = ""
	}
	f.State = normalizeStatusState(f.State)
	f.Account = strings.ToLower(strings.TrimSpace(f.Account))
	f.Model = strings.ToLower(strings.TrimSpace(f.Model))
	return f
}

func normalizeGroupBy(raw []string) []string {
	if len(raw) == 0 {
		return []string{"provider", "tier"}
	}
	seen := map[string]bool{}
	var out []string
	for _, item := range raw {
		for _, p := range strings.Split(item, ",") {
			g := strings.ToLower(strings.TrimSpace(p))
			switch g {
			case "", "none":
				continue
			case "providers":
				g = "provider"
			case "products":
				g = "product"
			case "tiers":
				g = "tier"
			case "models":
				g = "model"
			case "states":
				g = "state"
			case "agents":
				g = "agent"
			case "nodes":
				g = "node"
			}
			if !statusDimensionSupported(g) || seen[g] {
				continue
			}
			seen[g] = true
			out = append(out, g)
		}
	}
	if len(out) == 0 {
		return []string{"provider", "tier"}
	}
	return out
}

func statusDimensionSupported(g string) bool {
	switch g {
	case "provider", "product", "tier", "model", "state", "agent", "node":
		return true
	default:
		return false
	}
}

func statusAccount(row Account, workers []string, countCapacity bool) StatusAccount {
	capacity := 0
	if countCapacity {
		capacity = AccountSessionCap(row)
	}
	leased := fleetMinInt(len(workers), capacity)
	free, blocked := 0, 0
	if countCapacity && capacity > 0 {
		if accountCanBeOffered(row) {
			free = capacity - leased
			if free < 0 {
				free = 0
			}
		} else {
			blocked = capacity - leased
			if blocked < 0 {
				blocked = 0
			}
		}
	}
	state := accountStatusState(row, countCapacity, capacity, leased, free)
	return StatusAccount{
		Account:         row.Account,
		Tag:             row.Tag,
		Product:         productOf(row),
		Provider:        ProviderFamily(row),
		Agent:           derefStr(row.Agent),
		ModelTier:       row.ModelTier,
		Model:           derefStr(row.Model),
		Kind:            string(row.Kind),
		State:           state,
		Pool:            statusPool(row),
		CapacityCounted: countCapacity,
		SessionCap:      capacity,
		LeasedSlots:     leased,
		FreeSlots:       free,
		BlockedSlots:    blocked,
		ActiveSessions:  derefInt(row.ActiveSessions),
		LiveSessions:    derefInt(row.LiveSessions),
		Reason:          statusReason(row),
		DiscoverySource: row.DiscoverySource,
		RootState:       row.RootState,
		BlockKind:       derefStr(row.BlockKind),
		Reset:           derefStr(row.Reset),
		Weekly:          derefStr(row.Weekly),
		LoginStatus:     row.LoginStatus,
		CanServe:        row.CanServe,
		StatusSource:    derefStr(row.StatusSource),
		RegistryAgeMin:  row.RegistryAgeMin,
	}
}

func statusPool(row Account) string {
	if RoutableWorker(row) {
		return PoolKey(row)
	}
	return ""
}

func statusReason(row Account) string {
	if reason := derefStr(row.BlockReason); reason != "" {
		return reason
	}
	if accountLoginBlocked(row) {
		return accountLoginBlockReason(row)
	}
	return row.Reason
}

func accountStatusState(row Account, counted bool, capacity, leased, free int) string {
	if row.Kind != KindWorker {
		return string(row.Kind)
	}
	if IsDuplicateIdentity(row) {
		return "duplicate"
	}
	if !counted {
		return "shared-pool"
	}
	if !accountCanBeOffered(row) {
		kind := strings.ToLower(derefStr(row.BlockKind))
		switch {
		case accountLoginBlocked(row) || kind == "auth":
			return "auth"
		case kind == "usage" || derefBool(row.Throttled):
			return "usage"
		case kind != "":
			return kind
		default:
			return "blocked"
		}
	}
	if capacity > 0 && leased >= capacity {
		return "full"
	}
	if leased > 0 {
		return "leased"
	}
	if free > 0 || capacity == 0 {
		return "ready"
	}
	return "ready"
}

// ProviderFamily derives the provider/reporting family for an account row.
func ProviderFamily(row Account) string {
	text := strings.ToLower(strings.Join([]string{
		row.Account,
		row.Tag,
		productOf(row),
		derefStr(row.Model),
		derefStr(row.SmallModel),
		derefStr(row.ProfileSource),
	}, " "))
	switch {
	case strings.Contains(text, "groq"):
		return "groq"
	case strings.Contains(text, "nvidia") || strings.Contains(text, "nim-") ||
		strings.Contains(text, "default:nvidia-nim"):
		return "nvidia-nim"
	case strings.Contains(text, "gemini") || strings.Contains(text, "google/") ||
		strings.Contains(text, "vertex"):
		return "google"
	case strings.Contains(text, "gpt-") || strings.Contains(text, "openai"):
		return "openai"
	case strings.Contains(text, "anthropic") || strings.Contains(text, "claude") ||
		strings.Contains(text, "opus") || strings.Contains(text, "sonnet") ||
		strings.Contains(text, "haiku"):
		return "anthropic"
	case strings.Contains(text, "deepseek"):
		return "deepseek"
	case strings.Contains(text, "moonshot") || strings.Contains(text, "kimi"):
		return "moonshot"
	case strings.Contains(text, "glm") || strings.Contains(text, "z-ai") ||
		strings.Contains(text, "zai-"):
		return "zai"
	case strings.Contains(text, "local") || strings.Contains(text, "ollama") ||
		strings.Contains(text, "llama"):
		return "local"
	default:
		return productOf(row)
	}
}

func providerKey(raw string) string {
	s := strings.ToLower(strings.TrimSpace(raw))
	s = strings.ReplaceAll(s, "_", "-")
	s = strings.ReplaceAll(s, " ", "-")
	switch s {
	case "claude":
		return "anthropic"
	case "nvidia", "nim", "nvidia-nim-coding":
		return "nvidia-nim"
	case "gemini", "gcp", "vertex":
		return "google"
	case "codex", "gpt":
		return "openai"
	case "glm", "z-ai":
		return "zai"
	case "kimi":
		return "moonshot"
	default:
		return s
	}
}

func statusMatches(a StatusAccount, f StatusFilter) bool {
	if f.Product != "" && strings.ToLower(a.Product) != f.Product {
		return false
	}
	if f.Provider != "" && providerKey(a.Provider) != f.Provider {
		return false
	}
	if f.Tier > 0 && (a.ModelTier == nil || *a.ModelTier != f.Tier) {
		return false
	}
	if f.State != "" && !statusStateMatches(a.State, f.State) {
		return false
	}
	if f.Account != "" {
		hay := strings.ToLower(a.Account + " " + a.Tag)
		if !strings.Contains(hay, f.Account) {
			return false
		}
	}
	if f.Model != "" && !strings.Contains(strings.ToLower(a.Model), f.Model) {
		return false
	}
	return true
}

func normalizeStatusState(raw string) string {
	s := strings.ToLower(strings.TrimSpace(raw))
	s = strings.ReplaceAll(s, "_", "-")
	switch s {
	case "available", "free":
		return "ready"
	case "throttled", "rate-limited", "ratelimited", "limited":
		return "usage"
	case "login", "needs-login":
		return "auth"
	default:
		return s
	}
}

func statusStateMatches(state, filter string) bool {
	state = normalizeStatusState(state)
	filter = normalizeStatusState(filter)
	if state == filter {
		return true
	}
	if filter == "ready" {
		return state == "leased" || state == "full"
	}
	if filter == "blocked" {
		return state == "usage" || state == "auth" || state == "access" || state == "credit"
	}
	return false
}

func addStatusTotals(t *StatusTotals, a StatusAccount) {
	t.Accounts++
	switch a.Kind {
	case string(KindWorker):
		t.WorkerAccounts++
	case string(KindExcluded):
		t.ExcludedAccounts++
	case string(KindNonAccount):
		t.NonAccounts++
	}
	switch a.State {
	case "duplicate":
		t.DuplicateAccounts++
	case "usage":
		t.BlockedAccounts++
		t.UsageBlockedAccounts++
	case "auth":
		t.BlockedAccounts++
		t.AuthBlockedAccounts++
	case "blocked", "access", "credit":
		t.BlockedAccounts++
	case "ready", "leased", "full":
		t.ReadyAccounts++
	}
	if a.Kind == string(KindWorker) && a.State != "duplicate" && a.State != "shared-pool" {
		t.RoutableAccounts++
	}
	if a.CapacityCounted {
		t.Pools++
		t.TotalSlots += a.SessionCap
		t.FreeSlots += a.FreeSlots
		t.LeasedSlots += a.LeasedSlots
		t.BlockedSlots += a.BlockedSlots
	}
}

func statusRollupFor(byKey map[string]*StatusRollup, a StatusAccount, groupBy []string) *StatusRollup {
	dims := map[string]string{}
	parts := make([]string, 0, len(groupBy))
	for _, g := range groupBy {
		v := statusDimension(a, g)
		dims[g] = v
		parts = append(parts, g+"="+v)
	}
	key := strings.Join(parts, " ")
	if existing := byKey[key]; existing != nil {
		return existing
	}
	ru := &StatusRollup{
		Key:        key,
		Label:      key,
		Dimensions: dims,
	}
	byKey[key] = ru
	return ru
}

func statusDimension(a StatusAccount, dim string) string {
	switch dim {
	case "node":
		if a.Node == "" {
			return "local"
		}
		return a.Node
	case "provider":
		return a.Provider
	case "product":
		return a.Product
	case "tier":
		if a.ModelTier == nil {
			return "t?"
		}
		return "t" + itoa(*a.ModelTier)
	case "model":
		if a.Model == "" {
			return "unknown"
		}
		return a.Model
	case "state":
		return a.State
	case "agent":
		if a.Agent == "" {
			return "unknown"
		}
		return a.Agent
	default:
		return "unknown"
	}
}

// StampStatusReport adds portable snapshot metadata after a local status report is built.
// The node label is operator-supplied rather than os.Hostname-derived so a committed or
// shared report does not accidentally leak a private host name.
func StampStatusReport(rep *StatusReport, node, generatedAt string) {
	if rep == nil {
		return
	}
	rep.Node = strings.TrimSpace(node)
	rep.GeneratedAt = strings.TrimSpace(generatedAt)
	for i := range rep.Accounts {
		rep.Accounts[i].Node = rep.Node
	}
}

func addStatusRollup(ru *StatusRollup, a StatusAccount) {
	ru.Accounts++
	if a.Kind == string(KindWorker) {
		ru.WorkerAccounts++
	}
	switch a.State {
	case "ready", "leased", "full":
		ru.ReadyAccounts++
	case "usage":
		ru.BlockedAccounts++
		ru.UsageBlockedAccounts++
	case "auth":
		ru.BlockedAccounts++
		ru.AuthBlockedAccounts++
	case "blocked", "access", "credit":
		ru.BlockedAccounts++
	}
	if a.CapacityCounted {
		ru.Pools++
		ru.TotalSlots += a.SessionCap
		ru.FreeSlots += a.FreeSlots
		ru.LeasedSlots += a.LeasedSlots
		ru.BlockedSlots += a.BlockedSlots
	}
}

func statusWarnings(rep StatusReport) []string {
	var warnings []string
	if len(rep.Accounts) == 0 {
		return []string{"no accounts match the selected filters"}
	}
	for _, ru := range rep.Rollups {
		if ru.Mixed {
			warnings = append(warnings, fmt.Sprintf("mixed limit posture: %s has %d free, %d leased, %d blocked slot(s)",
				ru.Label, ru.FreeSlots, ru.LeasedSlots, ru.BlockedSlots))
		}
	}
	sort.Strings(warnings)
	return warnings
}

func statusAccountLess(a, b StatusAccount) bool {
	if statusRank(a.State) != statusRank(b.State) {
		return statusRank(a.State) < statusRank(b.State)
	}
	if a.Provider != b.Provider {
		return a.Provider < b.Provider
	}
	if a.Product != b.Product {
		return a.Product < b.Product
	}
	at, bt := 99, 99
	if a.ModelTier != nil {
		at = *a.ModelTier
	}
	if b.ModelTier != nil {
		bt = *b.ModelTier
	}
	if at != bt {
		return at < bt
	}
	return a.Tag < b.Tag
}

func statusRank(state string) int {
	switch state {
	case "usage", "auth", "blocked", "access", "credit":
		return 0
	case "ready", "leased", "full":
		return 1
	case "duplicate", "shared-pool":
		return 2
	case "excluded":
		return 3
	default:
		return 4
	}
}

// RenderStatusReport renders the compact human form. includeAccounts is normally false
// for the all-fleet view and true when the caller supplied a filter.
func RenderStatusReport(rep StatusReport, includeAccounts bool) string {
	var b strings.Builder
	fmt.Fprintf(&b, "fleet account status")
	if filterText := renderStatusFilters(rep.Filters); filterText != "" {
		fmt.Fprintf(&b, " (%s)", filterText)
	}
	fmt.Fprintf(&b, "\n")
	fmt.Fprintf(&b, "  slots: %d free / %d total   leased=%d blocked=%d   pools=%d\n",
		rep.Totals.FreeSlots, rep.Totals.TotalSlots, rep.Totals.LeasedSlots,
		rep.Totals.BlockedSlots, rep.Totals.Pools)
	fmt.Fprintf(&b, "  accounts: %d worker=%d routable=%d ready=%d blocked=%d usage=%d auth=%d duplicate=%d excluded=%d non-account=%d\n",
		rep.Totals.Accounts, rep.Totals.WorkerAccounts, rep.Totals.RoutableAccounts,
		rep.Totals.ReadyAccounts, rep.Totals.BlockedAccounts,
		rep.Totals.UsageBlockedAccounts, rep.Totals.AuthBlockedAccounts,
		rep.Totals.DuplicateAccounts, rep.Totals.ExcludedAccounts, rep.Totals.NonAccounts)
	fmt.Fprintf(&b, "\nrollups by %s:\n", strings.Join(rep.GroupBy, "+"))
	if len(rep.Rollups) == 0 {
		b.WriteString("  (none)\n")
	}
	for _, ru := range rep.Rollups {
		mix := ""
		if ru.Mixed {
			mix = " mixed"
		}
		fmt.Fprintf(&b, "  %-44s slots=%d/%d free leased=%d blocked=%d accounts ready=%d blocked=%d%s\n",
			ru.Label, ru.FreeSlots, ru.TotalSlots, ru.LeasedSlots, ru.BlockedSlots,
			ru.ReadyAccounts, ru.BlockedAccounts, mix)
	}
	if len(rep.Warnings) > 0 {
		b.WriteString("\nlimits:\n")
		for _, w := range rep.Warnings {
			fmt.Fprintf(&b, "  %s\n", w)
		}
	}
	if includeAccounts {
		b.WriteString("\naccounts:\n")
		if len(rep.Accounts) == 0 {
			b.WriteString("  (none)\n")
		}
		for _, a := range rep.Accounts {
			tier := "t?"
			if a.ModelTier != nil {
				tier = "t" + itoa(*a.ModelTier)
			}
			slots := "-"
			if a.CapacityCounted {
				slots = fmt.Sprintf("%d/%d free", a.FreeSlots, a.SessionCap)
				if a.LeasedSlots > 0 {
					slots += fmt.Sprintf(", %d leased", a.LeasedSlots)
				}
				if a.BlockedSlots > 0 {
					slots += fmt.Sprintf(", %d blocked", a.BlockedSlots)
				}
			}
			reason := a.Reason
			if len(reason) > 56 {
				reason = reason[:53] + "..."
			}
			fmt.Fprintf(&b, "  [%-10s] %-9s %-18s %-16s %-4s %-10s slots=%-18s %s\n",
				a.Provider, a.Product, a.Tag, a.Account, tier, a.State, slots, reason)
		}
	}
	return b.String()
}

func renderStatusFilters(f StatusFilter) string {
	var parts []string
	if f.Product != "" {
		parts = append(parts, "product="+f.Product)
	}
	if f.Provider != "" {
		parts = append(parts, "provider="+f.Provider)
	}
	if f.Tier > 0 {
		parts = append(parts, "tier=t"+itoa(f.Tier))
	}
	if f.State != "" {
		parts = append(parts, "state="+f.State)
	}
	if f.Account != "" {
		parts = append(parts, "account~"+f.Account)
	}
	if f.Model != "" {
		parts = append(parts, "model~"+f.Model)
	}
	return strings.Join(parts, " ")
}
