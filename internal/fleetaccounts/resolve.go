package fleetaccounts

import (
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/strmatch"
)

func itoa(i int) string { return strconv.Itoa(i) }

// FaklocalTag is the dogfood isolated-account tag.
const FaklocalTag = "faklocal"

// ReadOAuthToken returns the account's long-lived setup token, or "" (and ok=false).
// Pure read — never sets an env var. Reads only the .oauth-token file.
func ReadOAuthToken(accountDir string) (string, bool) {
	if accountDir == "" {
		return "", false
	}
	data, err := os.ReadFile(filepath.Join(accountDir, ".oauth-token"))
	if err != nil {
		return "", false
	}
	tok := strings.TrimSpace(string(data))
	if tok == "" {
		return "", false
	}
	return tok, true
}

// Resolved is the FLAT resolve_account return shape — the single record a dispatch front
// door reads (config_dir to pin as CLAUDE_CONFIG_DIR + the long-lived oauth_token + tier).
type Resolved struct {
	OK           bool    `json:"ok"`
	Reason       string  `json:"reason"`
	Account      string  `json:"account"`
	Tag          string  `json:"tag"`
	Product      string  `json:"product"`
	ConfigDir    string  `json:"config_dir"`
	OAuthToken   *string `json:"oauth_token"`
	Model        string  `json:"model"`
	ModelTier    *int    `json:"model_tier"`
	SelectedTier *int    `json:"selected_tier"`
	TargetTier   *int    `json:"target_tier"`
	FallbackUsed bool    `json:"fallback_used"`
	BlockReason  string  `json:"block_reason"`
	LoginStatus  *string `json:"login_status,omitempty"`
	CanServe     *bool   `json:"can_serve,omitempty"`

	BlockedTargetAccounts []BlockedAccount `json:"blocked_target_accounts,omitempty"`
}

func flattenResolved(acct Account, ok bool, reason string, selectedTier, targetTier *int,
	fallbackUsed bool, blockReason string) Resolved {
	configDir := acct.Dir
	var tok *string
	if t, found := ReadOAuthToken(configDir); found {
		tok = &t
	}
	st := selectedTier
	if st == nil {
		st = acct.ModelTier
	}
	model := derefStr(acct.Model)
	return Resolved{
		OK: ok, Reason: reason, Account: acct.Account, Tag: acct.Tag, Product: acct.Product,
		ConfigDir: configDir, OAuthToken: tok, Model: model, ModelTier: acct.ModelTier,
		SelectedTier: st, TargetTier: targetTier, FallbackUsed: fallbackUsed,
		BlockReason: blockReason, LoginStatus: acct.LoginStatus, CanServe: acct.CanServe,
	}
}

// ResolveRequest carries the resolve inputs (mirrors resolve_account kwargs).
type ResolveRequest struct {
	Pin               string
	TaskText          string
	TaskClass         string
	WorkKind          string
	Product           string
	AllowTierFallback bool
	StrictTier        bool
	FaklocalOK        bool
}

// Resolve resolves ONE fully-specified account for a dispatch — the canonical front-door
// call. Pin path validates availability; otherwise it delegates to RouteAccount. The rows
// must already be the AnnotatedRoster (so availability is folded). home is needed only for
// the faklocal synthesis path.
func Resolve(rows []Account, home string, req ResolveRequest, pol Policy) Resolved {
	// The dogfood isolated account: synthesized on demand.
	if req.FaklocalOK && req.Pin != "" && AccountTag(req.Pin) == FaklocalTag {
		d := filepath.Join(home, ".claude-faklocal")
		_ = os.MkdirAll(filepath.Join(d, "projects"), 0o755)
		three := 3
		acct := Account{Account: ".claude-faklocal", Tag: FaklocalTag, Product: "claude",
			Dir: d, Model: strp("local"), ModelTier: &three}
		return flattenResolved(acct, true, "isolated dogfood faklocal account", &three, &three, false, "")
	}

	if req.Pin != "" {
		needle := strings.ToLower(strings.TrimSpace(req.Pin))
		var match *Account
		for i := range rows {
			r := rows[i]
			if r.Kind == KindWorker &&
				(strings.ToLower(r.Tag) == needle || strings.ToLower(r.Account) == needle) {
				match = &rows[i]
				break
			}
		}
		if match == nil {
			return flattenResolved(Account{}, false,
				"account '"+req.Pin+"' is not an offered worker", nil, nil, false, "")
		}
		if !accountCanBeOffered(*match) && !req.AllowTierFallback {
			why := derefStr(match.BlockReason)
			if why == "" && accountLoginBlocked(*match) {
				why = accountLoginBlockReason(*match)
			}
			if why == "" {
				why = "blocked"
			}
			return flattenResolved(*match, false,
				"account '"+req.Pin+"' is blocked: "+why, nil, nil, false, why)
		}
		return flattenResolved(*match, true, "pinned account",
			match.ModelTier, match.ModelTier, false, "")
	}

	cls := req.TaskClass
	strict := req.StrictTier
	wk := strings.ToLower(strings.TrimSpace(req.WorkKind))
	if gardeningWorkKinds[wk] || engineeringWorkKinds[wk] {
		cls, strict = wk, false
	}
	route := RouteAccount(rows, req.TaskText, cls, req.AllowTierFallback, strict, req.Product, pol)
	if !route.OK {
		reason := route.Reason
		if reason == "" {
			reason = "no available account"
		}
		tt := route.TargetTier
		return Resolved{OK: false, Reason: reason, TargetTier: &tt,
			BlockedTargetAccounts: route.BlockedTargetAccounts}
	}
	tt := route.TargetTier
	return flattenResolved(*route.Account, true,
		strmatch.FirstNonEmpty(route.Reason, "routed"), route.SelectedTier, &tt, route.FallbackUsed, "")
}

// PoolKey returns the rate-limit pool a worker dir draws on — the unit a wave must hand
// out distinctly. Two Claude dirs on one Anthropic account share ONE pool (their
// accountUuid); a dir with no login is its own pool, keyed by its basename.
func PoolKey(r Account) string {
	uuid := derefStr(r.AccountUUID)
	if uuid != "" {
		return "uuid:" + uuid
	}
	name := r.Account
	if name == "" {
		name = r.Dir
	}
	return "dir:" + name
}

// SeatPoolSchema is the seat-pool envelope schema tag.
const SeatPoolSchema = "fleet-seat-pool/1"

const (
	// DefaultClaudeSessionsPerAccount is the default concurrent session budget
	// for one healthy Claude worker account. It models account capacity as
	// session slots rather than a single binary seat.
	DefaultClaudeSessionsPerAccount = 1
	DefaultAccountSessionsPerWorker = 1
)

// SessionsPerAccountEnv retunes the per-account Claude session budget without a
// rebuild — the same "one knob, one way" pattern as FAK_MAX_WORKERS (read by both
// tools/dispatch_preflight.py and internal/dispatchtick). Python reads the same
// variable in fleet_accounts._session_cap, so the two languages stay in lockstep.
const SessionsPerAccountEnv = "FAK_SESSIONS_PER_ACCOUNT"

func AccountSessionCap(row Account) int {
	switch strings.ToLower(productOf(row)) {
	case "claude":
		return claudeSessionsPerAccount()
	default:
		return DefaultAccountSessionsPerWorker
	}
}

// claudeSessionsPerAccount returns the hard identity-pool bound. Historical
// builds accepted FAK_SESSIONS_PER_ACCOUNT=6 and launched six concurrent workers
// through one OAuth identity; the witnessed #6572 wave yielded 0/10 commits and
// progressively shorter sessions. Keep parsing the compatibility knob only so a
// future diagnostics surface can name it; never let it widen the safety bound.
func claudeSessionsPerAccount() int {
	if v := strings.TrimSpace(os.Getenv(SessionsPerAccountEnv)); v != "" {
		_, _ = strconv.Atoi(v)
	}
	return DefaultClaudeSessionsPerAccount
}

// Lease is a live-worker lease record parsed from a `.account` sidecar.
type Lease struct {
	Worker string `json:"worker"`
	PID    *int   `json:"pid"`
	Tag    string `json:"tag"`
	Dir    string `json:"dir"`
}

// Seat is one seat-pool row.
type Seat struct {
	Seat        string   `json:"seat"`
	Tag         string   `json:"tag"`
	Account     string   `json:"account"`
	Product     string   `json:"product"`
	Model       *string  `json:"model"`
	ModelTier   *int     `json:"model_tier"`
	Available   bool     `json:"available"`
	State       string   `json:"state"`
	SessionCap  int      `json:"session_cap,omitempty"`
	LeasedSlots int      `json:"leased_slots,omitempty"`
	FreeSlots   int      `json:"free_slots,omitempty"`
	Workers     []string `json:"workers"`
}

// DoubleBooked names a session pool held beyond its configured cap (an invariant violation).
type DoubleBooked struct {
	Seat    string   `json:"seat"`
	Tag     string   `json:"tag"`
	Workers []string `json:"workers"`
}

// UnboundLease names a live worker on an account not in the pool.
type UnboundLease struct {
	Worker string `json:"worker"`
	Tag    string `json:"tag"`
	Dir    string `json:"dir"`
}

// SeatPool is the explicit multi-seat account pool envelope.
type SeatPool struct {
	Schema        string         `json:"schema"`
	Product       string         `json:"product"`
	TotalSeats    int            `json:"total_seats"`
	FreeSeats     int            `json:"free_seats"`
	LeasedSeats   int            `json:"leased_seats"`
	BlockedSeats  int            `json:"blocked_seats"`
	Depleted      bool           `json:"depleted"`
	DoubleBooked  []DoubleBooked `json:"double_booked"`
	UnboundLeases []UnboundLease `json:"unbound_leases"`
	Seats         []Seat         `json:"seats"`
}

// leaseWorkerLabel resolves the human-readable worker id for a lease: the recorded worker
// name, else the PID, else "?" when neither is known.
func leaseWorkerLabel(ls Lease) string {
	w := ls.Worker
	if w == "" && ls.PID != nil {
		w = itoa(*ls.PID)
	}
	if w == "" {
		w = "?"
	}
	return w
}

func leaseMatchesSeat(lease Lease, row Account) bool {
	ldir, rdir := lease.Dir, row.Dir
	if ldir != "" {
		if rdir != "" && ldir == rdir {
			return true
		}
		if filepath.Base(strings.TrimRight(ldir, "/\\")) == row.Account {
			return true
		}
	}
	ltag := strings.ToLower(lease.Tag)
	return ltag != "" && ltag == strings.ToLower(row.Tag)
}

// BuildSeatPool builds the explicit session-slot pool (M distinct routable worker pools
// times each product's per-account session cap) with live worker bindings.
func BuildSeatPool(rows []Account, leases []Lease, product string) SeatPool {
	wanted := strings.ToLower(product)
	var workers []Account
	for _, row := range rows {
		if !RoutableWorker(row) {
			continue
		}
		if wanted != "" && strings.ToLower(row.Product) != wanted {
			continue
		}
		workers = append(workers, row)
	}
	leaseWorkers, matched := leaseWorkersByPool(workers, leases)
	seats := []Seat{}
	doubleBooked := []DoubleBooked{}
	total, free, leased, blocked := 0, 0, 0, 0
	for _, row := range uniquePoolAccounts(workers) {
		capacity := AccountSessionCap(row)
		if capacity <= 0 {
			continue
		}
		poolKey := PoolKey(row)
		seatWorkers := append([]string(nil), leaseWorkers[poolKey]...)
		leasedSlots := len(seatWorkers)
		leasedCapped := fleetMinInt(leasedSlots, capacity)
		freeSlots := 0
		available := accountCanBeOffered(row)
		state := "blocked"
		total += capacity
		switch {
		case leasedSlots > 0:
			state = "leased"
			leased += leasedCapped
			if available {
				freeSlots = capacity - leasedCapped
				if freeSlots < 0 {
					freeSlots = 0
				}
				free += freeSlots
			} else {
				blocked += capacity - leasedCapped
			}
		case available:
			state = "free"
			freeSlots = capacity
			free += freeSlots
		default:
			blocked += capacity
		}
		seat := Seat{
			Seat: poolKey, Tag: row.Tag, Account: row.Account, Product: row.Product,
			Model: row.Model, ModelTier: row.ModelTier, Available: available,
			State: state, SessionCap: capacity, LeasedSlots: leasedSlots,
			FreeSlots: freeSlots, Workers: seatWorkers,
		}
		seats = append(seats, seat)
		if leasedSlots > capacity {
			doubleBooked = append(doubleBooked, DoubleBooked{Seat: seat.Seat, Tag: seat.Tag, Workers: seatWorkers})
		}
	}
	var unbound []UnboundLease
	for i, ls := range leases {
		if !matched[i] {
			unbound = append(unbound, UnboundLease{Worker: leaseWorkerLabel(ls), Tag: ls.Tag, Dir: ls.Dir})
		}
	}
	sort.SliceStable(seats, func(i, j int) bool {
		li := seats[i].State != "leased"
		lj := seats[j].State != "leased"
		if li != lj {
			return !li && lj
		}
		fi := seats[i].State != "free"
		fj := seats[j].State != "free"
		if fi != fj {
			return !fi && fj
		}
		if seats[i].Product != seats[j].Product {
			return seats[i].Product < seats[j].Product
		}
		return seats[i].Tag < seats[j].Tag
	})
	if doubleBooked == nil {
		doubleBooked = []DoubleBooked{}
	}
	if unbound == nil {
		unbound = []UnboundLease{}
	}
	if seats == nil {
		seats = []Seat{}
	}
	prod := wanted
	if prod == "" {
		prod = "all"
	}
	return SeatPool{
		Schema: SeatPoolSchema, Product: prod, TotalSeats: total, FreeSeats: free,
		LeasedSeats: leased, BlockedSeats: blocked, Depleted: free == 0,
		DoubleBooked: doubleBooked, UnboundLeases: unbound, Seats: seats,
	}
}

func uniquePoolAccounts(rows []Account) []Account {
	byPool := map[string]Account{}
	order := []string{}
	for _, row := range rows {
		pool := PoolKey(row)
		if _, ok := byPool[pool]; !ok {
			order = append(order, pool)
			byPool[pool] = row
			continue
		}
		if accountPoolLess(row, byPool[pool]) {
			byPool[pool] = row
		}
	}
	out := make([]Account, 0, len(order))
	for _, pool := range order {
		out = append(out, byPool[pool])
	}
	return out
}

func accountPoolLess(a, b Account) bool {
	aa, ba := accountCanBeOffered(a), accountCanBeOffered(b)
	if aa != ba {
		return aa
	}
	return rankLess(a, b)
}

func leaseWorkersByPool(rows []Account, leases []Lease) (map[string][]string, map[int]bool) {
	out := map[string][]string{}
	matched := map[int]bool{}
	for i, ls := range leases {
		for _, row := range rows {
			if !leaseMatchesSeat(ls, row) {
				continue
			}
			pool := PoolKey(row)
			out[pool] = append(out[pool], leaseWorkerLabel(ls))
			matched[i] = true
			break
		}
	}
	return out, matched
}

func fleetMinInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}
