package dispatchtick

import (
	"crypto/sha256"
	"fmt"
	"sort"
	"strings"

	configaccounts "github.com/anthony-chaudhary/fak/internal/accounts"
)

const SeatPoolSchema = "fleet-seat-pool/1"

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

type AccountRouteResult struct {
	OK                    bool
	Reason                string
	TargetTier            int
	SelectedTier          int
	FallbackUsed          bool
	Account               AccountRow
	BlockedTargetAccounts []AccountRow
}

type AccountWaveInput struct {
	Rows     []AccountRow
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
	Seat      string   `json:"seat"`
	Tag       string   `json:"tag"`
	Account   string   `json:"account"`
	Product   string   `json:"product"`
	Model     string   `json:"model"`
	ModelTier int      `json:"model_tier"`
	Available bool     `json:"available"`
	State     string   `json:"state"`
	Workers   []string `json:"workers"`
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
	product := strings.ToLower(strings.TrimSpace(in.Product))
	target := targetTierForWorkKind(in.WorkKind)
	workers := []AccountRow{}
	for _, raw := range in.Rows {
		row := NormalizeAccountRow(raw)
		if !routableAccount(row) {
			continue
		}
		if product != "" && row.Product != product {
			continue
		}
		workers = append(workers, row)
	}
	if len(workers) == 0 {
		reason := "no worker accounts"
		if product != "" {
			reason = "no worker accounts match product filter"
		}
		return AccountRouteResult{OK: false, Reason: reason, TargetTier: target}
	}
	tierOrder := []int{target}
	if target == 2 {
		tierOrder = append(tierOrder, 1)
	}
	for _, tier := range tierOrder {
		candidates := []AccountRow{}
		for _, row := range workers {
			if row.Available && row.ModelTier == tier {
				candidates = append(candidates, row)
			}
		}
		if len(candidates) == 0 {
			continue
		}
		sort.Slice(candidates, func(i, j int) bool { return accountRouteLess(candidates[i], candidates[j]) })
		return AccountRouteResult{
			OK:                    true,
			Reason:                chooseString(tier == target, "selected target tier", "selected fallback tier"),
			TargetTier:            target,
			SelectedTier:          tier,
			FallbackUsed:          tier != target,
			Account:               candidates[0],
			BlockedTargetAccounts: blockedTierAccounts(workers, target),
		}
	}
	reason := fmt.Sprintf("no available tier %d account", target)
	if target == 1 {
		reason += " (tier-1 fallback disabled)"
	}
	return AccountRouteResult{
		OK:                    false,
		Reason:                reason,
		TargetTier:            target,
		BlockedTargetAccounts: blockedTierAccounts(workers, target),
	}
}

func AllocateWave(in AccountWaveInput) AccountWaveResult {
	n := in.Count
	if n < 0 {
		n = 0
	}
	product := strings.ToLower(strings.TrimSpace(in.Product))
	target := targetTierForWorkKind(in.WorkKind)
	workers := []AccountRow{}
	for _, raw := range in.Rows {
		row := NormalizeAccountRow(raw)
		if !routableAccount(row) {
			continue
		}
		if product != "" && row.Product != product {
			continue
		}
		workers = append(workers, row)
	}
	tierOrder := []int{target}
	if target == 2 {
		tierOrder = append(tierOrder, 1)
	}
	lanes := []AccountWaveLane{}
	seenPools := map[string]bool{}
	for _, tier := range tierOrder {
		if len(lanes) >= n {
			break
		}
		candidates := []AccountRow{}
		for _, row := range workers {
			if row.Available && row.ModelTier == tier {
				candidates = append(candidates, row)
			}
		}
		sort.Slice(candidates, func(i, j int) bool { return accountRouteLess(candidates[i], candidates[j]) })
		for _, row := range candidates {
			if len(lanes) >= n {
				break
			}
			pool := PoolKey(row)
			if seenPools[pool] {
				continue
			}
			seenPools[pool] = true
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
		reason = fmt.Sprintf("granted %d of %d distinct pools; %d short (roster has no more distinct available pools at the requested tiers)", granted, n, shortfall)
	default:
		reason = fmt.Sprintf("granted %d distinct pools", granted)
	}
	return AccountWaveResult{
		OK:                    granted > 0,
		Requested:             n,
		Granted:               granted,
		Shortfall:             shortfall,
		DistinctPools:         granted,
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
	for _, raw := range rows {
		row := NormalizeAccountRow(raw)
		if !routableAccount(row) {
			continue
		}
		if wanted != "all" && row.Product != wanted {
			continue
		}
		workers := []string{}
		for _, lease := range leases {
			if leaseMatchesSeat(lease, row) {
				worker := strings.TrimSpace(lease.Worker)
				if worker == "" && lease.PID > 0 {
					worker = fmt.Sprintf("%d", lease.PID)
				}
				if worker == "" {
					worker = "?"
				}
				workers = append(workers, worker)
			}
		}
		state := "blocked"
		switch {
		case len(workers) > 0:
			state = "leased"
			pool.LeasedSeats++
		case row.Available:
			state = "free"
			pool.FreeSeats++
		default:
			pool.BlockedSeats++
		}
		pool.TotalSeats++
		pool.Seats = append(pool.Seats, SeatRow{
			Seat:      PoolKey(row),
			Tag:       row.Tag,
			Account:   row.Account,
			Product:   row.Product,
			Model:     row.Model,
			ModelTier: row.ModelTier,
			Available: row.Available,
			State:     state,
			Workers:   workers,
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

func publicBlockedAccounts(rows []AccountRow, tier int) []BlockedAccount {
	out := []BlockedAccount{}
	for _, row := range rows {
		if row.ModelTier != tier || row.Available {
			continue
		}
		reason := strings.TrimSpace(row.BlockReason)
		if reason == "" {
			reason = "blocked"
		}
		out = append(out, BlockedAccount{
			Tag:         row.Tag,
			Account:     row.Account,
			Product:     row.Product,
			ModelTier:   row.ModelTier,
			Model:       row.Model,
			Reason:      reason,
			LoginStatus: row.LoginStatus,
			CanServe:    row.CanServe,
		})
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
