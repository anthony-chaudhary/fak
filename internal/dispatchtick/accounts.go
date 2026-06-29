package dispatchtick

import (
	"fmt"
	"sort"
	"strings"
)

const SeatPoolSchema = "fleet-seat-pool/1"

type AccountRow struct {
	Account        string
	Tag            string
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
	return row
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
