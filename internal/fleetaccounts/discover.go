package fleetaccounts

import (
	"encoding/json"
	"os"
	"path/filepath"
	"sort"
	"strings"

	configaccounts "github.com/anthony-chaudhary/fak/internal/accounts"
)

// Account is one discovered config dir's roster row. The JSON field names and order
// mirror the dicts emitted by fleet_accounts.py so the `json` shape is byte-compatible.
//
// The base classification fields (dir/product/account/tag/kind/reason/notes) are always
// present. Worker rows additionally carry the model Profile (flattened) + route_weight,
// and Claude worker rows carry the logged-in identity + the reconciliation verdict.
// Runtime status fields are attached by Annotate.
type Account struct {
	Dir     string `json:"dir"`
	Product string `json:"product"`
	Account string `json:"account"`
	Tag     string `json:"tag"`
	Kind    Kind   `json:"kind"`
	Reason  string `json:"reason"`
	Notes   string `json:"notes"`

	// Worker profile (omitted for non-worker rows, matching the Python row which only
	// stamps these on worker rows). Pointers so an unset profile serializes as absent.
	ModelTier     *int    `json:"model_tier,omitempty"`
	Model         *string `json:"model,omitempty"`
	SmallModel    *string `json:"small_model,omitempty"`
	ModelEffort   *string `json:"model_effort,omitempty"`
	Agent         *string `json:"agent,omitempty"`
	ProfileSource *string `json:"profile_source,omitempty"`
	RouteWeight   *int    `json:"route_weight,omitempty"`

	// Claude worker identity (stamped at classify time, then reconciled).
	AccountUUID *string `json:"account_uuid,omitempty"`
	LoginEmail  *string `json:"login_email,omitempty"`
	OrgUUID     *string `json:"org_uuid,omitempty"`
	OrgType     *string `json:"org_type,omitempty"`
	Plan        *string `json:"plan,omitempty"`

	IdentityRole  *string  `json:"identity_role,omitempty"`
	IdentityPeers []string `json:"identity_peers,omitempty"`
	TagLoginMatch *bool    `json:"tag_login_match,omitempty"`

	LoginStatus *string `json:"login_status,omitempty"`
	CanServe    *bool   `json:"can_serve,omitempty"`

	// Runtime status (attached by Annotate).
	Available           *bool    `json:"available,omitempty"`
	Blocked             *bool    `json:"blocked,omitempty"`
	BlockKind           *string  `json:"block_kind"`
	BlockReason         *string  `json:"block_reason,omitempty"`
	Reset               *string  `json:"reset"`
	Weekly              *string  `json:"weekly"`
	Throttled           *bool    `json:"throttled,omitempty"`
	ActiveSessions      *int     `json:"active_sessions,omitempty"`
	LiveSessions        *int     `json:"live_sessions,omitempty"`
	AuthBlockedSessions *int     `json:"auth_blocked_sessions,omitempty"`
	StatusSource        *string  `json:"status_source,omitempty"`
	RegistryAgeMin      *float64 `json:"registry_age_min"`
}

func strp(s string) *string { return &s }
func intp(i int) *int       { return &i }
func boolp(b bool) *bool    { return &b }

// readJSONObject reads a JSON object from a path. Never raises: a missing/malformed/
// non-object file yields (nil, false). Used for opencode config + identity reads.
func readJSONObject(path string) (map[string]any, bool) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, false
	}
	var doc any
	if err := json.Unmarshal(data, &doc); err != nil {
		return nil, false
	}
	m, ok := doc.(map[string]any)
	return m, ok
}

func asString(v any) string {
	if s, ok := v.(string); ok {
		return s
	}
	return ""
}

// Identity is the logged-in Anthropic identity read from a Claude config dir's .claude.json.
type Identity struct {
	AccountUUID string
	LoginEmail  string
	OrgUUID     string
	OrgType     string
	Plan        string
}

// ReadAccountIdentity reads the logged-in identity from a Claude config dir's .claude.json.
// The single source of truth for WHO a dir is actually logged in as. Never raises — reads
// only the small oauthAccount identity fields; credentials/tokens are never touched.
func ReadAccountIdentity(acctDir string) Identity {
	out := Identity{}
	doc, ok := readJSONObject(filepath.Join(acctDir, ".claude.json"))
	if !ok {
		return out
	}
	oa, ok := doc["oauthAccount"].(map[string]any)
	if !ok {
		return out
	}
	out.AccountUUID = asString(oa["accountUuid"])
	out.LoginEmail = asString(oa["emailAddress"])
	out.OrgUUID = asString(oa["organizationUuid"])
	out.OrgType = asString(oa["organizationType"])
	out.Plan = asString(oa["organizationType"])
	if out.Plan == "" {
		out.Plan = asString(oa["seatTier"])
	}
	return out
}

// classifyRow applies policy + structure checks to one discovered dir. The caller has
// already confirmed acctDir is an account dir (projects/ for Claude, opencode.json for
// opencode) before invoking this.
func classifyRow(acctDir, product, account string, pol Policy) Account {
	tag := AccountTag(account)
	note := pol.Notes[tag]
	base := Account{Dir: acctDir, Product: product, Account: account, Tag: tag, Notes: note}

	st, err := os.Stat(acctDir)
	if err != nil || !st.IsDir() {
		base.Kind = KindNonAccount
		base.Reason = "not a directory"
		return base
	}
	// Intrinsic tombstone: a `.DELETED` marker decommissions the dir regardless of policy.
	if strings.Contains(strings.ToLower(account), ".deleted") {
		base.Kind = KindExcluded
		base.Reason = "tombstoned (.DELETED marker)"
		return base
	}
	id := Identity{}
	if product == "claude" {
		id = ReadAccountIdentity(acctDir)
	}
	if hit := excludedMatch(tag, account, pol.Exclude, id.LoginEmail); hit != "" {
		base.Kind = KindExcluded
		if note != "" {
			base.Reason = note
		} else if hitNote := pol.Notes[hit]; hitNote != "" {
			base.Reason = hitNote
		} else {
			base.Reason = "excluded by policy (matches '" + hit + "')"
		}
		return base
	}
	var includeOnly []string
	for _, t := range pol.IncludeOnly {
		if t != "" {
			includeOnly = append(includeOnly, t)
		}
	}
	if len(includeOnly) > 0 {
		matched := false
		for _, t := range includeOnly {
			if strings.Contains(strings.ToLower(tag), strings.ToLower(t)) {
				matched = true
				break
			}
		}
		if !matched {
			base.Kind = KindExcluded
			base.Reason = "not in include_only allowlist"
			return base
		}
	}
	label := "real offered account"
	if product == "opencode" {
		label = "real offered opencode account"
	}
	row := base
	row.Kind = KindWorker
	row.Reason = label
	prof := accountProfile(row, pol)
	row.ModelTier = intp(prof.ModelTier)
	row.Model = strp(prof.Model)
	row.SmallModel = strp(prof.SmallModel)
	row.ModelEffort = strp(prof.ModelEffort)
	row.Agent = strp(prof.Agent)
	row.ProfileSource = strp(prof.ProfileSource)
	row.RouteWeight = intp(accountRouteWeight(row, pol))
	if product == "claude" {
		row.AccountUUID = strp(id.AccountUUID)
		row.LoginEmail = strp(id.LoginEmail)
		row.OrgUUID = strp(id.OrgUUID)
		row.OrgType = strp(id.OrgType)
		row.Plan = strp(id.Plan)
		st, can := claudeLoginStatus(acctDir, tag)
		row.LoginStatus = strp(string(st))
		row.CanServe = boolp(can)
	}
	return row
}

func claudeLoginStatus(acctDir, tag string) (configaccounts.LoginStatus, bool) {
	h := configaccounts.Home{
		Name:     tag,
		Dir:      acctDir,
		Identity: configaccounts.DeriveIdentity(acctDir),
	}
	st := h.LoginStatus()
	return st, st == configaccounts.LoginReady
}

func discoverClaude(home string, pol Policy) []Account {
	var rows []Account
	matches, _ := filepath.Glob(filepath.Join(home, ".claude*"))
	for _, acctDir := range matches {
		account := filepath.Base(acctDir)
		tag := AccountTag(account)
		note := pol.Notes[tag]
		st, err := os.Stat(acctDir)
		if err != nil || !st.IsDir() {
			rows = append(rows, Account{Dir: acctDir, Product: "claude", Account: account,
				Tag: tag, Kind: KindNonAccount, Reason: "not a directory", Notes: note})
			continue
		}
		pst, perr := os.Stat(filepath.Join(acctDir, "projects"))
		if perr != nil || !pst.IsDir() {
			rows = append(rows, Account{Dir: acctDir, Product: "claude", Account: account,
				Tag: tag, Kind: KindNonAccount, Reason: "no projects/ subdir", Notes: note})
			continue
		}
		rows = append(rows, classifyRow(acctDir, "claude", account, pol))
	}
	return rows
}

func discoverOpencode(configHome string, pol Policy) []Account {
	var rows []Account
	matches, _ := filepath.Glob(filepath.Join(configHome, "opencode*"))
	for _, acctDir := range matches {
		account := filepath.Base(acctDir)
		tag := AccountTag(account)
		note := pol.Notes[tag]
		st, err := os.Stat(acctDir)
		if err != nil || !st.IsDir() {
			rows = append(rows, Account{Dir: acctDir, Product: "opencode", Account: account,
				Tag: tag, Kind: KindNonAccount, Reason: "not a directory", Notes: note})
			continue
		}
		hasMarker := false
		for _, m := range OpencodeMarkerFiles {
			if mst, merr := os.Stat(filepath.Join(acctDir, m)); merr == nil && !mst.IsDir() {
				hasMarker = true
				break
			}
		}
		if !hasMarker {
			rows = append(rows, Account{Dir: acctDir, Product: "opencode", Account: account,
				Tag: tag, Kind: KindNonAccount, Reason: "no opencode.json config", Notes: note})
			continue
		}
		rows = append(rows, classifyRow(acctDir, "opencode", account, pol))
	}
	return rows
}

// dirRecency returns the newest .jsonl mtime under the dir's projects/ — a cheap "last
// actually used" proxy used to pick the canonical dir among same-identity dirs.
func dirRecency(acctDir string) float64 {
	proj := filepath.Join(acctDir, "projects")
	var newest float64
	_ = filepath.Walk(proj, func(path string, info os.FileInfo, err error) error {
		if err != nil || info.IsDir() {
			return nil
		}
		if strings.HasSuffix(path, ".jsonl") {
			if m := float64(info.ModTime().UnixNano()); m > newest {
				newest = m
			}
		}
		return nil
	})
	return newest
}

// reconcileIdentities detects Claude worker dirs sharing ONE logged-in Anthropic account,
// stamping identity_role (unique|canonical|duplicate|no-login), identity_peers, and
// tag_login_match. Duplicates stay visible but callers exclude them from routing counts.
func reconcileIdentities(rows []Account) {
	// gather pointers to Claude workers
	var workers []*Account
	for i := range rows {
		if rows[i].Kind == KindWorker && rows[i].Product == "claude" {
			workers = append(workers, &rows[i])
		}
	}
	byUUID := map[string][]*Account{}
	for _, r := range workers {
		uuid := derefStr(r.AccountUUID)
		if uuid != "" {
			byUUID[uuid] = append(byUUID[uuid], r)
		}
	}
	// pass 1: tag<->login agreement for EVERY worker first.
	for _, r := range workers {
		email := derefStr(r.LoginEmail)
		tag := r.Tag
		match := false
		if email != "" {
			tl := strings.ToLower(tag)
			if strings.Contains(tl, strings.ToLower(tag)) && strings.Contains(strings.ToLower(email), tl) {
				match = true
			}
			if !match {
				local := strings.ToLower(strings.SplitN(email, "@", 2)[0])
				for _, part := range strings.Split(local, ".") {
					if part != "" && strings.Contains(tl, part) {
						match = true
						break
					}
				}
			}
		}
		r.TagLoginMatch = boolp(match)
	}
	// pass 2: role per worker.
	for _, r := range workers {
		uuid := derefStr(r.AccountUUID)
		email := derefStr(r.LoginEmail)
		if uuid == "" {
			r.IdentityRole = strp("no-login")
			r.IdentityPeers = []string{}
			continue
		}
		group := byUUID[uuid]
		if len(group) == 0 {
			group = []*Account{r}
		}
		peers := []string{}
		for _, g := range group {
			if g != r {
				peers = append(peers, g.Tag)
			}
		}
		sort.Strings(peers)
		r.IdentityPeers = peers
		if len(group) == 1 {
			r.IdentityRole = strp("unique")
			continue
		}
		canonical := canonicalDir(group)
		if r == canonical {
			r.IdentityRole = strp("canonical")
		} else {
			r.IdentityRole = strp("duplicate")
			name := email
			if name == "" && len(uuid) >= 8 {
				name = uuid[:8]
			} else if name == "" {
				name = uuid
			}
			r.Reason = "duplicate identity: same Anthropic account as " +
				derefStr(canonical.TagPtr()) + " (" + name + ")"
		}
	}
}

// canonicalDir picks the canonical dir among same-identity dirs: a tag-matched dir wins,
// then a non-"default" name, then the most-recently-active dir.
func canonicalDir(group []*Account) *Account {
	best := group[0]
	bestKey := canonKey(best)
	for _, g := range group[1:] {
		k := canonKey(g)
		if canonKeyLess(bestKey, k) {
			best, bestKey = g, k
		}
	}
	return best
}

type canonKeyT struct {
	tagMatch int
	notDef   int
	recency  float64
}

func canonKey(g *Account) canonKeyT {
	tm := 0
	if derefBool(g.TagLoginMatch) {
		tm = 1
	}
	nd := 1
	if g.Tag == "default" {
		nd = 0
	}
	return canonKeyT{tm, nd, dirRecency(g.Dir)}
}

// canonKeyLess reports whether a < b (so max picks the largest, like Python's max()).
func canonKeyLess(a, b canonKeyT) bool {
	if a.tagMatch != b.tagMatch {
		return a.tagMatch < b.tagMatch
	}
	if a.notDef != b.notDef {
		return a.notDef < b.notDef
	}
	return a.recency < b.recency
}

// TagPtr returns a pointer to the account's Tag (helper for reconcile reason text).
func (a *Account) TagPtr() *string { return &a.Tag }

func derefStr(p *string) string {
	if p == nil {
		return ""
	}
	return *p
}
func derefBool(p *bool) bool {
	if p == nil {
		return false
	}
	return *p
}
func derefInt(p *int) int {
	if p == nil {
		return 0
	}
	return *p
}

// Discover classifies every account config dir across both products, then reconciles
// shared Claude identities. Rows are sorted by (product, kind != worker, tag) to match
// fleet_accounts.discover_accounts.
func Discover(home, configHome string, pol Policy) []Account {
	rows := append(discoverClaude(home, pol), discoverOpencode(configHome, pol)...)
	reconcileIdentities(rows)
	sort.SliceStable(rows, func(i, j int) bool {
		if rows[i].Product != rows[j].Product {
			return rows[i].Product < rows[j].Product
		}
		wi, wj := rows[i].Kind != KindWorker, rows[j].Kind != KindWorker
		if wi != wj {
			return !wi && wj
		}
		return rows[i].Tag < rows[j].Tag
	})
	return rows
}

// IsDuplicateIdentity reports whether a worker dir is a non-canonical copy of another
// dir's account (routing to it would double-count one account's capacity).
func IsDuplicateIdentity(a Account) bool {
	return derefStr(a.IdentityRole) == "duplicate"
}

// RoutableWorker reports a worker the switcher may offer: a real worker that is not a
// duplicate of another dir's identity.
func RoutableWorker(a Account) bool {
	return a.Kind == KindWorker && !IsDuplicateIdentity(a)
}
