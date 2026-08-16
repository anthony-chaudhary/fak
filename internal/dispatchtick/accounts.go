package dispatchtick

import (
	"crypto/sha256"
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"

	configaccounts "github.com/anthony-chaudhary/fak/internal/accounts"
)

const SeatPoolSchema = "fleet-seat-pool/1"

const (
	// DefaultClaudeSessionsPerAccount is the default concurrent session budget
	// for one healthy Claude worker account. It models the account pool as
	// session slots, not a single binary seat.
	DefaultClaudeSessionsPerAccount = 4
	DefaultAccountSessionsPerWorker = 1
)

// SessionsPerAccountEnv retunes the per-account Claude session budget without a
// rebuild. Keep this in lockstep with tools/fleet_accounts.py and
// internal/fleetaccounts so every dispatch front door prices the same capacity.
const SessionsPerAccountEnv = "FAK_SESSIONS_PER_ACCOUNT"

type AccountRow struct {
	Account        string
	Tag            string
	Kind           string
	Product        string
	Dir            string
	Model          string
	ModelTier      int
	Available      bool
	BlockReason    string
	ActiveSessions int
	LiveSessions   int
	RouteWeight    int
	IdentityRole   string
	AccountUUID    string
	LoginStatus    string
	CanServe       *bool
}

type AccountRouteInput struct {
	Rows     []AccountRow
	Product  string
	WorkKind string
}

type SeatSelectionCandidate struct {
	Rank       int    `json:"rank"`
	Tag        string `json:"tag"`
	Tier       int    `json:"tier"`
	Cooldown   bool   `json:"cooldown"`
	SeatFree   bool   `json:"seat_free"`
	CanServe   bool   `json:"can_serve"`
	Score      int    `json:"score"`
	SkipReason string `json:"skip_reason,omitempty"`
}

type SeatSelection struct {
	WinnerTag    string                   `json:"winner_tag,omitempty"`
	WinnerReason string                   `json:"winner_reason"`
	Summary      string                   `json:"summary"`
	Candidates   []SeatSelectionCandidate `json:"candidates"`
}

type AccountRouteResult struct {
	OK                    bool
	Reason                string
	TargetTier            int
	SelectedTier          int
	FallbackUsed          bool
	Account               AccountRow
	BlockedTargetAccounts []AccountRow
	SeatSelection         SeatSelection
}

type AccountWaveInput struct {
	Rows     []AccountRow
	Leases   []SeatLease
	Count    int
	Product  string
	WorkKind string
	WaveID   string
}

type AccountWaveLane struct {
	OK           bool   `json:"ok"`
	Reason       string `json:"reason"`
	Account      string `json:"account"`
	Tag          string `json:"tag"`
	Product      string `json:"product"`
	ConfigDir    string `json:"config_dir"`
	Model        string `json:"model"`
	ModelTier    int    `json:"model_tier"`
	SelectedTier int    `json:"selected_tier"`
	TargetTier   int    `json:"target_tier"`
	FallbackUsed bool   `json:"fallback_used"`
	BlockReason  string `json:"block_reason"`
	LoginStatus  string `json:"login_status,omitempty"`
	CanServe     *bool  `json:"can_serve,omitempty"`
	Pool         string `json:"pool"`
	SessionSlot  int    `json:"session_slot,omitempty"`
	SessionCap   int    `json:"session_cap,omitempty"`
	Rank         int    `json:"rank"`
	WaveID       string `json:"wave_id"`
	Size         int    `json:"size"`
}

type BlockedAccount struct {
	Tag         string `json:"tag"`
	Account     string `json:"account"`
	Product     string `json:"product"`
	ModelTier   int    `json:"model_tier"`
	Model       string `json:"model"`
	Reason      string `json:"reason"`
	LoginStatus string `json:"login_status,omitempty"`
	CanServe    *bool  `json:"can_serve,omitempty"`
}

type AccountWaveResult struct {
	OK                    bool
	Requested             int
	Granted               int
	Shortfall             int
	DistinctPools         int
	Size                  int
	WaveID                string
	TargetTier            int
	Reason                string
	Lanes                 []AccountWaveLane
	BlockedTargetAccounts []BlockedAccount
}

type SeatLease struct {
	Worker string
	PID    int
	Tag    string
	Dir    string
}

type SeatRow struct {
	Seat        string   `json:"seat"`
	Tag         string   `json:"tag"`
	Account     string   `json:"account"`
	Product     string   `json:"product"`
	Model       string   `json:"model"`
	ModelTier   int      `json:"model_tier"`
	Available   bool     `json:"available"`
	State       string   `json:"state"`
	SessionCap  int      `json:"session_cap,omitempty"`
	LeasedSlots int      `json:"leased_slots,omitempty"`
	FreeSlots   int      `json:"free_slots,omitempty"`
	Workers     []string `json:"workers"`
}

type SeatPoolResult struct {
	Schema       string    `json:"schema"`
	Product      string    `json:"product"`
	TotalSeats   int       `json:"total_seats"`
	FreeSeats    int       `json:"free_seats"`
	LeasedSeats  int       `json:"leased_seats"`
	BlockedSeats int       `json:"blocked_seats"`
	Depleted     bool      `json:"depleted"`
	Seats        []SeatRow `json:"seats"`
}

func ProductFromAccount(account string) string {
	if strings.HasPrefix(strings.ToLower(strings.TrimSpace(account)), "opencode") {
		return "opencode"
	}
	return "claude"
}

func TagFromAccount(account string) string {
	a := strings.TrimSpace(account)
	if ProductFromAccount(a) == "opencode" {
		tag := strings.TrimPrefix(a, "opencode-")
		tag = strings.TrimPrefix(tag, "opencode")
		if tag == "" {
			return "default"
		}
		return tag
	}
	tag := strings.TrimPrefix(a, ".claude-")
	tag = strings.TrimPrefix(tag, ".claude")
	tag = strings.TrimSuffix(tag, "-acct")
	if tag == "" {
		return "default"
	}
	return tag
}

func NormalizeAccountRow(row AccountRow) AccountRow {
	row.Account = strings.TrimSpace(row.Account)
	row.Tag = strings.TrimSpace(row.Tag)
	row.Kind = strings.TrimSpace(row.Kind)
	if row.Tag == "" {
		row.Tag = TagFromAccount(row.Account)
	}
	row.Product = strings.ToLower(strings.TrimSpace(row.Product))
	if row.Product == "" {
		row.Product = ProductFromAccount(row.Account)
	}
	if row.ModelTier <= 0 || row.ModelTier > 3 {
		row.ModelTier = inferredModelTier(row)
	}
	if row.Model == "" {
		row.Model = inferredModel(row)
	}
	row = applyAccountLoginGate(row)
	return row
}

func AccountSessionCap(row AccountRow) int {
	row = NormalizeAccountRow(row)
	switch row.Product {
	case "claude":
		return claudeSessionsPerAccount()
	default:
		return DefaultAccountSessionsPerWorker
	}
}

// claudeSessionsPerAccount returns the hard OAuth identity-pool bound. The
// compatibility knob is parsed but cannot widen the one-session safety floor.
func claudeSessionsPerAccount() int {
	if v := strings.TrimSpace(os.Getenv(SessionsPerAccountEnv)); v != "" {
		_, _ = strconv.Atoi(v)
	}
	return DefaultClaudeSessionsPerAccount
}

// underSessionCap reports whether row can take one MORE concurrent session:
// its registry-live session count must sit strictly below the per-account
// session budget. Every single-launch chooser admits through this gate so a
// route-weighted seat that is already full spills to the next candidate
// instead of accreting every new launch until it limit-walls.
func underSessionCap(row AccountRow) bool {
	limit := AccountSessionCap(row)
	return limit <= 0 || row.LiveSessions < limit
}

// atCapBlockReason names the session-cap refusal for blocked-account reporting.
func atCapBlockReason(row AccountRow) string {
	return fmt.Sprintf("at session cap (%d live >= cap %d)", row.LiveSessions, AccountSessionCap(row))
}

// overCapTierAccounts lists the AVAILABLE accounts at tier that were turned away
// only because they are at their session cap, stamped with the cap reason, so a
// route decision can name the seats it refused to overfill.
func overCapTierAccounts(workers []AccountRow, tier int) []AccountRow {
	out := []AccountRow{}
	for _, row := range workers {
		if row.ModelTier == tier && row.Available && !underSessionCap(row) {
			row.BlockReason = atCapBlockReason(row)
			out = append(out, row)
		}
	}
	return out
}

func applyAccountLoginGate(row AccountRow) AccountRow {
	if row.Product != "claude" {
		return row
	}
	blocked := false
	if row.CanServe != nil && !*row.CanServe {
		blocked = true
	}
	if row.LoginStatus != "" && row.LoginStatus != string(configaccounts.LoginReady) {
		blocked = true
	}
	if !blocked {
		return row
	}
	row.Available = false
	if strings.TrimSpace(row.BlockReason) != "" {
		return row
	}
	row.BlockReason = accountLoginBlockReason(row)
	return row
}

func accountLoginBlockReason(row AccountRow) string {
	status := configaccounts.LoginStatus(strings.TrimSpace(row.LoginStatus))
	if status != "" && status != configaccounts.LoginReady {
		reason, _ := configaccounts.LoginReasonAction(status,
			configaccounts.Home{Name: row.Tag, Dir: row.Dir})
		if reason != "" {
			return reason
		}
		return "account login status is " + string(status)
	}
	return "account login cannot serve"
}

func RouteAccount(in AccountRouteInput) AccountRouteResult {
	product, target, workers := normalizeWorkerPool(in.Rows, in.Product, in.WorkKind)
	if len(workers) == 0 {
		reason := "no worker accounts"
		if product != "" {
			reason = "no worker accounts match product filter"
		}
		res := AccountRouteResult{OK: false, Reason: reason, TargetTier: target}
		res.SeatSelection = buildSeatSelection(workers, target, res)
		return res
	}
	tierOrder := []int{target}
	if target == 2 {
		tierOrder = append(tierOrder, 1)
	}
	atCap := 0
	for _, tier := range tierOrder {
		candidates := []AccountRow{}
		for _, row := range availableTierCandidates(workers, tier) {
			if !underSessionCap(row) {
				atCap++
				continue
			}
			candidates = append(candidates, row)
		}
		if len(candidates) == 0 {
			continue
		}
		sort.Slice(candidates, func(i, j int) bool { return accountRouteLess(candidates[i], candidates[j]) })
		res := AccountRouteResult{
			OK:                    true,
			Reason:                chooseString(tier == target, "selected target tier", "selected fallback tier"),
			TargetTier:            target,
			SelectedTier:          tier,
			FallbackUsed:          tier != target,
			Account:               candidates[0],
			BlockedTargetAccounts: append(blockedTierAccounts(workers, target), overCapTierAccounts(workers, target)...),
		}
		res.SeatSelection = buildSeatSelection(workers, target, res)
		return res
	}
	reason := fmt.Sprintf("no available tier %d account", target)
	if atCap > 0 {
		reason += fmt.Sprintf(" under the session cap (%d at cap)", atCap)
	}
	if target == 1 {
		reason += " (tier-1 fallback disabled)"
	}
	res := AccountRouteResult{
		OK:                    false,
		Reason:                reason,
		TargetTier:            target,
		BlockedTargetAccounts: append(blockedTierAccounts(workers, target), overCapTierAccounts(workers, target)...),
	}
	res.SeatSelection = buildSeatSelection(workers, target, res)
	return res
}

func buildSeatSelection(rows []AccountRow, target int, res AccountRouteResult) SeatSelection {
	ranked := append([]AccountRow(nil), rows...)
	sort.SliceStable(ranked, func(i, j int) bool { return accountRouteLess(ranked[i], ranked[j]) })
	out := SeatSelection{WinnerTag: res.Account.Tag, WinnerReason: res.Reason, Candidates: make([]SeatSelectionCandidate, 0, len(ranked))}
	for i, row := range ranked {
		canServe := row.Available
		if row.CanServe != nil {
			canServe = *row.CanServe
		}
		free := underSessionCap(row)
		cooldown := strings.Contains(strings.ToLower(row.BlockReason), "cooldown")
		skip := ""
		switch {
		case row.Tag == res.Account.Tag && res.OK:
		case cooldown:
			skip = "cooldown"
		case !canServe || !row.Available:
			skip = chooseString(strings.TrimSpace(row.LoginStatus) != "", strings.TrimSpace(row.LoginStatus), strings.TrimSpace(row.BlockReason))
			if skip == "" {
				skip = "unavailable"
			}
		case !free:
			skip = "session_cap"
		case row.ModelTier != target && !res.FallbackUsed:
			skip = "non_target_tier"
		default:
			skip = "lower_rank"
		}
		out.Candidates = append(out.Candidates, SeatSelectionCandidate{Rank: i + 1, Tag: row.Tag, Tier: row.ModelTier, Cooldown: cooldown, SeatFree: free, CanServe: canServe, Score: row.RouteWeight, SkipReason: skip})
	}
	if res.OK {
		competitors := len(out.Candidates) - 1
		out.Summary = fmt.Sprintf("picked %s over %d (%s)", res.Account.Tag, maxInt(competitors, 0), res.Reason)
		if !res.Account.Available || (res.Account.CanServe != nil && !*res.Account.CanServe) || strings.EqualFold(strings.TrimSpace(res.Account.LoginStatus), "auth_failed") {
			out.WinnerReason = "chosen despite auth_failed"
			out.Summary = fmt.Sprintf("picked %s over %d (chosen despite auth_failed)", res.Account.Tag, maxInt(competitors, 0))
		}
	} else {
		out.Summary = "picked none (" + res.Reason + ")"
	}
	return out
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func AllocateWave(in AccountWaveInput) AccountWaveResult {
	n := in.Count
	if n < 0 {
		n = 0
	}
	product, target, workers := normalizeWorkerPool(in.Rows, in.Product, in.WorkKind)
	tierOrder := []int{target}
	if target == 2 {
		tierOrder = append(tierOrder, 1)
	}
	lanes := []AccountWaveLane{}
	usedPools := map[string]bool{}
	load := leaseLoadByPool(workers, in.Leases)
	// Floor each pool's load at its registry-live session count: sessions launched
	// outside this dispatcher (interactive resumes, the watchdog) hold no seat
	// lease, and a wave that only counts its own leases would grant a full wave
	// onto a seat already running at capacity. max() rather than sum, because a
	// dispatch-launched session appears in both counts.
	for _, row := range uniquePoolRows(workers) {
		pool := PoolKey(row)
		if row.LiveSessions > load[pool] {
			load[pool] = row.LiveSessions
		}
	}
	for _, tier := range tierOrder {
		if len(lanes) >= n {
			break
		}
		candidates := uniquePoolRows(availableTierCandidates(workers, tier))
		sort.Slice(candidates, func(i, j int) bool { return accountRouteLess(candidates[i], candidates[j]) })
		for len(lanes) < n {
			best := -1
			for i, row := range candidates {
				pool := PoolKey(row)
				if load[pool] >= AccountSessionCap(row) {
					continue
				}
				if best < 0 || accountWaveSlotLess(row, candidates[best], load[pool], load[PoolKey(candidates[best])]) {
					best = i
				}
			}
			if best < 0 {
				break
			}
			row := candidates[best]
			pool := PoolKey(row)
			capacity := AccountSessionCap(row)
			slot := load[pool] + 1
			load[pool] = slot
			usedPools[pool] = true
			lanes = append(lanes, AccountWaveLane{
				OK:           true,
				Reason:       chooseString(tier == target, "wave lane (target tier)", "wave lane (fallback tier)"),
				Account:      row.Account,
				Tag:          row.Tag,
				Product:      row.Product,
				ConfigDir:    row.Dir,
				Model:        row.Model,
				ModelTier:    row.ModelTier,
				SelectedTier: row.ModelTier,
				TargetTier:   target,
				FallbackUsed: tier != target,
				LoginStatus:  row.LoginStatus,
				CanServe:     row.CanServe,
				Pool:         pool,
				SessionSlot:  slot,
				SessionCap:   capacity,
			})
		}
	}
	granted := len(lanes)
	shortfall := n - granted
	if shortfall < 0 {
		shortfall = 0
	}
	waveID := strings.TrimSpace(in.WaveID)
	if waveID == "" {
		pools := make([]string, 0, len(lanes))
		for _, lane := range lanes {
			pools = append(pools, lane.Pool)
		}
		waveID = waveIDForPools(pools)
	}
	for i := range lanes {
		lanes[i].Rank = i
		lanes[i].WaveID = waveID
		lanes[i].Size = granted
	}
	reason := ""
	switch {
	case granted == 0:
		reason = fmt.Sprintf("no available account for a wave (target tier %d", target)
		if product != "" {
			reason += fmt.Sprintf(", product %s", product)
		}
		reason += ")"
	case shortfall > 0:
		reason = fmt.Sprintf("granted %d of %d session slot(s) across %d distinct pool(s); %d short (roster has no more available session slots at the requested tiers)", granted, n, len(usedPools), shortfall)
	default:
		reason = fmt.Sprintf("granted %d session slot(s) across %d distinct pool(s)", granted, len(usedPools))
	}
	return AccountWaveResult{
		OK:                    granted > 0,
		Requested:             n,
		Granted:               granted,
		Shortfall:             shortfall,
		DistinctPools:         len(usedPools),
		Size:                  granted,
		WaveID:                waveID,
		TargetTier:            target,
		Reason:                reason,
		Lanes:                 lanes,
		BlockedTargetAccounts: publicBlockedAccounts(workers, target),
	}
}

func BuildSeatPool(rows []AccountRow, leases []SeatLease, product string) SeatPoolResult {
	wanted := strings.ToLower(strings.TrimSpace(product))
	if wanted == "" {
		wanted = "all"
	}
	pool := SeatPoolResult{Schema: SeatPoolSchema, Product: wanted}
	workers := []AccountRow{}
	for _, raw := range rows {
		row := NormalizeAccountRow(raw)
		if !routableAccount(row) {
			continue
		}
		if wanted != "all" && row.Product != wanted {
			continue
		}
		workers = append(workers, row)
	}
	leaseWorkers := leaseWorkersByPool(workers, leases)
	for _, row := range uniquePoolRows(workers) {
		capacity := AccountSessionCap(row)
		if capacity <= 0 {
			continue
		}
		pool.TotalSeats += capacity
		seatWorkers := append([]string(nil), leaseWorkers[PoolKey(row)]...)
		leasedSlots := len(seatWorkers)
		leasedCapped := minInt(leasedSlots, capacity)
		freeSlots := 0
		state := "blocked"
		switch {
		case leasedSlots > 0:
			state = "leased"
			pool.LeasedSeats += leasedCapped
			if row.Available {
				freeSlots = capacity - leasedCapped
				if freeSlots < 0 {
					freeSlots = 0
				}
				pool.FreeSeats += freeSlots
			} else {
				pool.BlockedSeats += capacity - leasedCapped
			}
		case row.Available:
			state = "free"
			freeSlots = capacity
			pool.FreeSeats += freeSlots
		default:
			pool.BlockedSeats += capacity
		}
		pool.Seats = append(pool.Seats, SeatRow{
			Seat:        PoolKey(row),
			Tag:         row.Tag,
			Account:     row.Account,
			Product:     row.Product,
			Model:       row.Model,
			ModelTier:   row.ModelTier,
			Available:   row.Available,
			State:       state,
			SessionCap:  capacity,
			LeasedSlots: leasedSlots,
			FreeSlots:   freeSlots,
			Workers:     seatWorkers,
		})
	}
	sort.Slice(pool.Seats, func(i, j int) bool {
		return seatSortKey(pool.Seats[i]) < seatSortKey(pool.Seats[j])
	})
	pool.Depleted = pool.FreeSeats == 0
	return pool
}

func (r AccountWaveResult) Map() map[string]any {
	lanes := make([]any, 0, len(r.Lanes))
	for _, lane := range r.Lanes {
		lanes = append(lanes, lane.Map())
	}
	blocked := make([]any, 0, len(r.BlockedTargetAccounts))
	for _, row := range r.BlockedTargetAccounts {
		blocked = append(blocked, row.Map())
	}
	return map[string]any{
		"ok":                      r.OK,
		"requested":               r.Requested,
		"granted":                 r.Granted,
		"shortfall":               r.Shortfall,
		"distinct_pools":          r.DistinctPools,
		"size":                    r.Size,
		"wave_id":                 r.WaveID,
		"target_tier":             r.TargetTier,
		"reason":                  r.Reason,
		"lanes":                   lanes,
		"blocked_target_accounts": blocked,
	}
}

func (l AccountWaveLane) Map() map[string]any {
	out := map[string]any{
		"ok":            l.OK,
		"reason":        l.Reason,
		"account":       l.Account,
		"tag":           l.Tag,
		"product":       l.Product,
		"config_dir":    l.ConfigDir,
		"model":         l.Model,
		"model_tier":    l.ModelTier,
		"selected_tier": l.SelectedTier,
		"target_tier":   l.TargetTier,
		"fallback_used": l.FallbackUsed,
		"block_reason":  l.BlockReason,
		"pool":          l.Pool,
		"rank":          l.Rank,
		"wave_id":       l.WaveID,
		"size":          l.Size,
	}
	if l.LoginStatus != "" {
		out["login_status"] = l.LoginStatus
	}
	if l.CanServe != nil {
		out["can_serve"] = *l.CanServe
	}
	if l.SessionSlot > 0 {
		out["session_slot"] = l.SessionSlot
	}
	if l.SessionCap > 0 {
		out["session_cap"] = l.SessionCap
	}
	return out
}

func (b BlockedAccount) Map() map[string]any {
	out := map[string]any{
		"tag":        b.Tag,
		"account":    b.Account,
		"product":    b.Product,
		"model_tier": b.ModelTier,
		"model":      b.Model,
		"reason":     b.Reason,
	}
	if b.LoginStatus != "" {
		out["login_status"] = b.LoginStatus
	}
	if b.CanServe != nil {
		out["can_serve"] = *b.CanServe
	}
	return out
}

func PoolKey(row AccountRow) string {
	row = NormalizeAccountRow(row)
	if strings.TrimSpace(row.AccountUUID) != "" {
		return "uuid:" + strings.TrimSpace(row.AccountUUID)
	}
	if row.Account != "" {
		return "dir:" + row.Account
	}
	return "dir:" + row.Dir
}

// normalizeWorkerPool normalizes rawRows, drops the non-routable ones, and (when
// a product filter is set) keeps only rows matching it. It returns the normalized
// lower-cased product filter, the target model tier for workKind, and the routable
// worker rows — the shared front matter of RouteAccount and AllocateWave.
func normalizeWorkerPool(rawRows []AccountRow, product, workKind string) (string, int, []AccountRow) {
	normProduct := strings.ToLower(strings.TrimSpace(product))
	target := targetTierForWorkKind(workKind)
	workers := []AccountRow{}
	for _, raw := range rawRows {
		row := NormalizeAccountRow(raw)
		if !routableAccount(row) {
			continue
		}
		if normProduct != "" && row.Product != normProduct {
			continue
		}
		workers = append(workers, row)
	}
	return normProduct, target, workers
}

// availableTierCandidates returns the available worker rows whose ModelTier equals
// tier, preserving input order (callers sort with accountRouteLess afterward).
func availableTierCandidates(workers []AccountRow, tier int) []AccountRow {
	candidates := []AccountRow{}
	for _, row := range workers {
		if row.Available && row.ModelTier == tier {
			candidates = append(candidates, row)
		}
	}
	return candidates
}

func uniquePoolRows(rows []AccountRow) []AccountRow {
	byPool := map[string]AccountRow{}
	order := []string{}
	for _, row := range rows {
		pool := PoolKey(row)
		if _, ok := byPool[pool]; !ok {
			order = append(order, pool)
			byPool[pool] = row
			continue
		}
		if accountPoolRowLess(row, byPool[pool]) {
			byPool[pool] = row
		}
	}
	out := make([]AccountRow, 0, len(order))
	for _, pool := range order {
		out = append(out, byPool[pool])
	}
	return out
}

func accountPoolRowLess(a, b AccountRow) bool {
	if a.Available != b.Available {
		return a.Available
	}
	return accountRouteLess(a, b)
}

func accountWaveSlotLess(a, b AccountRow, loadA, loadB int) bool {
	if loadA != loadB {
		return loadA < loadB
	}
	return accountRouteLess(a, b)
}

// LaneUnattributed is the bucket LaneSeatShare files a leased seat under when its
// lane cannot be resolved (laneOf is nil or returns an empty/whitespace lane).
const LaneUnattributed = "unattributed"

// LaneShare is the per-lane rollup LaneSeatShare returns: how many leased seats
// resolved to the lane, and that lane's fraction of all leases in the input.
type LaneShare struct {
	Count int     `json:"count"`
	Share float64 `json:"share"`
}

// LaneSeatShare groups leased seats by lane and returns each lane's leased-seat
// Count and its Share of the total leases. laneOf resolves a lease to its lane; a
// lease whose resolved lane is empty (unresolved, or laneOf is nil) is bucketed
// under LaneUnattributed so no lease is silently dropped and the counts always sum
// to len(leases). Share is Count/total, so over a non-empty input the shares sum to
// 1. A pure fold -- no I/O, deterministic in its inputs; an empty input returns an
// empty (non-nil) map with no divide-by-zero.
func LaneSeatShare(leases []SeatLease, laneOf func(SeatLease) string) map[string]LaneShare {
	counts := map[string]int{}
	total := 0
	for _, lease := range leases {
		lane := ""
		if laneOf != nil {
			lane = strings.TrimSpace(laneOf(lease))
		}
		if lane == "" {
			lane = LaneUnattributed
		}
		counts[lane]++
		total++
	}
	out := make(map[string]LaneShare, len(counts))
	for lane, count := range counts {
		share := 0.0
		if total > 0 {
			share = float64(count) / float64(total)
		}
		out[lane] = LaneShare{Count: count, Share: share}
	}
	return out
}

func leaseLoadByPool(rows []AccountRow, leases []SeatLease) map[string]int {
	out := map[string]int{}
	for pool, workers := range leaseWorkersByPool(rows, leases) {
		out[pool] = len(workers)
	}
	return out
}

func leaseWorkersByPool(rows []AccountRow, leases []SeatLease) map[string][]string {
	out := map[string][]string{}
	for _, lease := range leases {
		for _, row := range rows {
			if !leaseMatchesSeat(lease, row) {
				continue
			}
			worker := strings.TrimSpace(lease.Worker)
			if worker == "" && lease.PID > 0 {
				worker = fmt.Sprintf("%d", lease.PID)
			}
			if worker == "" {
				worker = "?"
			}
			pool := PoolKey(row)
			out[pool] = append(out[pool], worker)
			break
		}
	}
	return out
}

func routableAccount(row AccountRow) bool {
	if strings.EqualFold(row.IdentityRole, "duplicate") {
		return false
	}
	kind := strings.ToLower(strings.TrimSpace(row.Kind))
	if kind != "" && kind != "worker" {
		return false
	}
	return strings.TrimSpace(row.Account) != "" || strings.TrimSpace(row.Dir) != ""
}

func targetTierForWorkKind(workKind string) int {
	switch strings.ToLower(strings.TrimSpace(workKind)) {
	case "gardening", "garden", "maintenance", "maint", "cleanup", "chore", "triage":
		return 2
	default:
		return 1
	}
}

func accountRouteLess(a, b AccountRow) bool {
	if a.RouteWeight != b.RouteWeight {
		return a.RouteWeight > b.RouteWeight
	}
	if a.LiveSessions != b.LiveSessions {
		return a.LiveSessions < b.LiveSessions
	}
	if a.ActiveSessions != b.ActiveSessions {
		return a.ActiveSessions < b.ActiveSessions
	}
	if a.Product != b.Product {
		return a.Product < b.Product
	}
	return a.Tag < b.Tag
}

func blockedTierAccounts(rows []AccountRow, tier int) []AccountRow {
	out := []AccountRow{}
	for _, row := range rows {
		if row.ModelTier == tier && !row.Available {
			out = append(out, row)
		}
	}
	sort.Slice(out, func(i, j int) bool { return accountRouteLess(out[i], out[j]) })
	return out
}

// BlockedAccountFromRow projects a blocked AccountRow into the public BlockedAccount
// shape (defaulting an empty block reason to "blocked"), so the dispatch preflight can
// carry the SAME structured per-account block reason the wave allocator already
// surfaces via publicBlockedAccounts -- instead of collapsing the pool to a bare list
// of tags that hides WHY each seat was refused (throttled / needs-login / at-cap).
func BlockedAccountFromRow(row AccountRow) BlockedAccount {
	reason := strings.TrimSpace(row.BlockReason)
	if reason == "" {
		reason = "blocked"
	}
	return BlockedAccount{
		Tag:         row.Tag,
		Account:     row.Account,
		Product:     row.Product,
		ModelTier:   row.ModelTier,
		Model:       row.Model,
		Reason:      reason,
		LoginStatus: row.LoginStatus,
		CanServe:    row.CanServe,
	}
}

func publicBlockedAccounts(rows []AccountRow, tier int) []BlockedAccount {
	out := []BlockedAccount{}
	for _, row := range rows {
		if row.ModelTier != tier || row.Available {
			continue
		}
		out = append(out, BlockedAccountFromRow(row))
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Product != out[j].Product {
			return out[i].Product < out[j].Product
		}
		return out[i].Tag < out[j].Tag
	})
	return out
}

func waveIDForPools(pools []string) string {
	if len(pools) == 0 {
		return ""
	}
	sort.Strings(pools)
	sum := sha256.Sum256([]byte(strings.Join(pools, ",")))
	return "wave-" + fmt.Sprintf("%x", sum[:6])
}

func leaseMatchesSeat(lease SeatLease, row AccountRow) bool {
	ldir := strings.TrimSpace(lease.Dir)
	if ldir != "" {
		rdir := strings.TrimSpace(row.Dir)
		if rdir != "" && samePathish(ldir, rdir) {
			return true
		}
		if basePathish(ldir) == row.Account {
			return true
		}
	}
	ltag := strings.ToLower(strings.TrimSpace(lease.Tag))
	return ltag != "" && ltag == strings.ToLower(strings.TrimSpace(row.Tag))
}

func inferredModelTier(row AccountRow) int {
	lower := strings.ToLower(row.Model + " " + row.Tag + " " + row.Account)
	switch {
	case strings.Contains(lower, "local") || strings.Contains(lower, "faklocal"):
		return 3
	case strings.Contains(lower, "glm") || strings.Contains(lower, "zai"):
		return 2
	case row.Product == "claude":
		return 1
	default:
		return 3
	}
}

func inferredModel(row AccountRow) string {
	if row.Product == "claude" {
		if row.ModelTier == 3 {
			return "local"
		}
		return "opus"
	}
	lower := strings.ToLower(row.Tag + " " + row.Account)
	if strings.Contains(lower, "glm") || strings.Contains(lower, "zai") {
		return "zai-coding-plan/glm-5.2"
	}
	return ""
}

func samePathish(a, b string) bool {
	return strings.EqualFold(strings.ReplaceAll(a, "\\", "/"), strings.ReplaceAll(b, "\\", "/"))
}

func basePathish(path string) string {
	path = strings.TrimRight(strings.ReplaceAll(path, "\\", "/"), "/")
	if i := strings.LastIndex(path, "/"); i >= 0 {
		return path[i+1:]
	}
	return path
}

func chooseString(cond bool, yes, no string) string {
	if cond {
		return yes
	}
	return no
}

func seatSortKey(s SeatRow) string {
	stateRank := "2"
	switch s.State {
	case "leased":
		stateRank = "0"
	case "free":
		stateRank = "1"
	}
	return stateRank + "\x00" + s.Product + "\x00" + s.Tag
}
