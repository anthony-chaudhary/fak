package fleetaccounts

import (
	"encoding/json"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	configaccounts "github.com/anthony-chaudhary/fak/internal/accounts"
	"github.com/anthony-chaudhary/fak/internal/harnessprofile"
)

// Account is one discovered config dir's roster row. The JSON field names and order
// mirror the dicts emitted by fleet_accounts.py so the `json` shape is byte-compatible.
//
// The base classification fields (dir/product/account/tag/kind/reason/notes) are always
// present. Worker rows additionally carry the model Profile (flattened) + route_weight,
// and config-home worker rows (Claude/Codex) carry the logged-in identity + the
// reconciliation verdict.
// Runtime status fields are attached by Annotate.
type Account struct {
	Dir     string `json:"dir"`
	Product string `json:"product"`
	Account string `json:"account"`
	Tag     string `json:"tag"`
	Kind    Kind   `json:"kind"`
	Reason  string `json:"reason"`
	Notes   string `json:"notes"`

	// CredKind is the credential KIND this seat's config-home registry entry declares
	// (#5331). It is EMPTY for the historical subscription-OAuth seat — the zero value the
	// registry itself reads as oauth — so an OAuth row's published JSON keeps the exact
	// legacy key set the cross-surface parity gate compares; only an api-key seat stamps it.
	// APIKeyEnv is the NAME of the environment variable holding that seat's Anthropic API
	// key: a REFERENCE, never the secret, matching the registry's own posture.
	CredKind  configaccounts.CredKind `json:"cred_kind,omitempty"`
	APIKeyEnv string                  `json:"api_key_env,omitempty"`

	// Worker profile (omitted for non-worker rows, matching the Python row which only
	// stamps these on worker rows). Pointers so an unset profile serializes as absent.
	ModelTier          *int     `json:"model_tier,omitempty"`
	Model              *string  `json:"model,omitempty"`
	SmallModel         *string  `json:"small_model,omitempty"`
	ModelEffort        *string  `json:"model_effort,omitempty"`
	Agent              *string  `json:"agent,omitempty"`
	ProfileSource      *string  `json:"profile_source,omitempty"`
	RouteWeight        *int     `json:"route_weight,omitempty"`
	RoutingCostPerMTok *float64 `json:"routing_cost_per_mtok,omitempty"`
	BilledCostPerMTok  *float64 `json:"billed_cost_per_mtok,omitempty"`

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
	// UsageSoonReset carries a still-active DAILY usage cap that a fresh OK probe reopened
	// the seat over (see markUsageSoon). Advisory only: the seat stays Available; this lets
	// the roster show "serving, cap resets HH:MM" instead of a blank near-cap row. Nil on
	// every row not in that reopen-over-a-live-cap state, and MarshalJSON emits the key only
	// when set, so a normal row's byte-parity JSON is unchanged.
	UsageSoonReset *string `json:"usage_soon_reset,omitempty"`
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

// accountsRegistryIndex folds the config-home registry into the lookups discovery needs: the
// ACTIVE seats by name and by cleaned dir path, and the TOMBSTONED seats by name, dir, and
// account identity. The active maps carry the whole Home rather than a mere presence bit so a
// discovered dir can INHERIT its seat's credential kind (#5331) instead of only learning that
// the dir is enrolled.
type accountsRegistryIndex struct {
	activeNames          map[string]configaccounts.Home
	activeDirs           map[string]configaccounts.Home
	tombstonedNames      map[string]configaccounts.Home
	tombstonedDirs       map[string]configaccounts.Home
	tombstonedIdentities map[string]configaccounts.Home
}

func loadAccountsRegistry(home string) configaccounts.Registry {
	path := strings.TrimSpace(os.Getenv("FAK_ACCOUNTS_REGISTRY"))
	if path == "" && home != "" {
		path = filepath.Join(home, ".claude-accounts", "registry.json")
	}
	if path == "" {
		return configaccounts.Registry{}
	}
	reg, err := configaccounts.LoadRegistry(path)
	if err != nil {
		return configaccounts.Registry{}
	}
	return reg.Refresh()
}

func indexAccountsRegistry(reg configaccounts.Registry) accountsRegistryIndex {
	idx := accountsRegistryIndex{
		activeNames:          map[string]configaccounts.Home{},
		activeDirs:           map[string]configaccounts.Home{},
		tombstonedNames:      map[string]configaccounts.Home{},
		tombstonedDirs:       map[string]configaccounts.Home{},
		tombstonedIdentities: map[string]configaccounts.Home{},
	}
	for _, h := range reg.Homes {
		name := strings.ToLower(strings.TrimSpace(h.Name))
		dir := pathKey(h.Dir)
		ident := h.Identity.AccountKey()
		if h.Active() {
			if name != "" {
				idx.activeNames[name] = h
			}
			if dir != "" {
				idx.activeDirs[dir] = h
			}
			continue
		}
		if name != "" {
			idx.tombstonedNames[name] = h
		}
		if dir != "" {
			idx.tombstonedDirs[dir] = h
		}
		if ident != "" {
			idx.tombstonedIdentities[ident] = h
		}
	}
	return idx
}

func pathKey(path string) string {
	path = strings.TrimSpace(path)
	if path == "" {
		return ""
	}
	if abs, err := filepath.Abs(path); err == nil {
		path = abs
	}
	return strings.ToLower(filepath.Clean(path))
}

func registryLookupKeys(account, tag, product string) map[string]bool {
	out := map[string]bool{}
	for _, k := range []string{account, tag} {
		k = strings.ToLower(strings.TrimSpace(k))
		if k != "" {
			out[k] = true
		}
	}
	if product == "claude" {
		switch {
		case account == ".claude":
			out["default"] = true
		case strings.HasPrefix(account, ".claude-"):
			out[strings.ToLower(strings.TrimPrefix(account, ".claude-"))] = true
		}
	}
	return out
}

func accountsRegistryReason(h configaccounts.Home, fallback string) string {
	reason := strings.TrimSpace(h.TombstoneReason)
	if reason == "" {
		reason = fallback
	}
	if h.RehomeTo != "" && !strings.Contains(strings.ToLower(reason), "rehome") {
		reason += "; rehome -> " + h.RehomeTo
	}
	return reason
}

func accountsRegistryExclusion(acctDir, product, account, tag string, id Identity, idx accountsRegistryIndex) string {
	keys := registryLookupKeys(account, tag, product)
	dkey := pathKey(acctDir)
	for k := range keys {
		if h, ok := idx.tombstonedNames[k]; ok {
			return accountsRegistryReason(h, "tombstoned in fak accounts registry")
		}
	}
	if dkey != "" {
		if h, ok := idx.tombstonedDirs[dkey]; ok {
			return accountsRegistryReason(h, "tombstoned in fak accounts registry")
		}
	}

	for k := range keys {
		if _, ok := idx.activeNames[k]; ok {
			return ""
		}
	}
	if dkey != "" {
		if _, ok := idx.activeDirs[dkey]; ok {
			return ""
		}
	}

	identityKey := ""
	if id.AccountUUID != "" {
		identityKey = "uuid:" + id.AccountUUID
	}
	if identityKey != "" {
		if h, ok := idx.tombstonedIdentities[identityKey]; ok {
			return accountsRegistryReason(h, "same account identity as tombstoned fak accounts registry seat")
		}
	}
	return ""
}

// activeSeat returns the ACTIVE config-home registry seat backing a discovered dir. It
// matches on the cleaned dir path first — the strongest binding, since one dir is one seat —
// and then on the seat-name aliases the roster tag can carry, walked in sorted order so an
// ambiguous fixture resolves deterministically. It is how a discovered dir learns the
// credential KIND its registry entry declares (#5331); an unenrolled dir returns the zero
// Home and false, which reads as the historical OAuth seat everywhere downstream.
func (idx accountsRegistryIndex) activeSeat(acctDir, product, account, tag string) (configaccounts.Home, bool) {
	if dkey := pathKey(acctDir); dkey != "" {
		if h, ok := idx.activeDirs[dkey]; ok {
			return h, true
		}
	}
	keys := make([]string, 0, 4)
	for k := range registryLookupKeys(account, tag, product) {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		if h, ok := idx.activeNames[k]; ok {
			return h, true
		}
	}
	return configaccounts.Home{}, false
}

// intrinsicExclusion returns the reason an account is kept off the roster by its NAME alone —
// the `.DELETED` tombstone marker and the fak-kernel dogfood homes — or "" when the name
// carries no intrinsic exclusion. Shared by the dir-driven classifyRow and the
// registry-driven apiKeySeatRow so both roster paths agree on which names are never offered.
func intrinsicExclusion(account string) string {
	if strings.Contains(strings.ToLower(account), ".deleted") {
		return "tombstoned (.DELETED marker)"
	}
	// The `.claude-faklocal*` homes are the fak-kernel dogfood dirs — synthesized on demand by
	// Resolve(--faklocal-ok), never an enrolled account. They carry no login and only exist to
	// point Claude Code at a locally served model, so a credential-less `needs_login` row for
	// one just clutters the switcher roster. Keep them off it the way a .DELETED marker does;
	// the dogfood resolve path bypasses discovery entirely, so nothing that depends on it
	// regresses. Re-offer as a first-class servable seat once native `fak serve` is stable
	// (see #3558).
	if strings.HasPrefix(strings.ToLower(account), ".claude-faklocal") {
		return "fak-kernel dogfood home (synthesized on demand; not a roster account)"
	}
	return ""
}

// policyExclusion returns the reason operator POLICY keeps this account off the roster — an
// exclude-list hit (reported with the operator's note when one exists) or an include_only
// allowlist miss — or "" when policy offers it. loginEmail is folded into the exclude match so
// a rule can name the account by its login as well as by its tag/dir; pass "" when no login
// identity is available (an api-key seat derives none offline).
func policyExclusion(account, tag string, pol Policy, loginEmail string) string {
	if hit := excludedMatch(tag, account, pol.Exclude, loginEmail); hit != "" {
		if note := pol.Notes[tag]; note != "" {
			return note
		}
		if hitNote := pol.Notes[hit]; hitNote != "" {
			return hitNote
		}
		return "excluded by policy (matches '" + hit + "')"
	}
	var includeOnly []string
	for _, t := range pol.IncludeOnly {
		if t != "" {
			includeOnly = append(includeOnly, t)
		}
	}
	if len(includeOnly) == 0 {
		return ""
	}
	for _, t := range includeOnly {
		if strings.Contains(strings.ToLower(tag), strings.ToLower(t)) {
			return ""
		}
	}
	return "not in include_only allowlist"
}

// stampProfile fills a worker row's model-routing profile and route weight from policy. Both
// roster paths (dir-driven and registry-driven) stamp the identical block, so it lives here
// rather than being restated per path where the two could silently drift apart.
func stampProfile(row *Account, pol Policy) {
	prof := accountProfile(*row, pol)
	row.ModelTier = intp(prof.ModelTier)
	row.Model = strp(prof.Model)
	row.SmallModel = strp(prof.SmallModel)
	row.ModelEffort = strp(prof.ModelEffort)
	row.Agent = strp(prof.Agent)
	row.ProfileSource = strp(prof.ProfileSource)
	row.RouteWeight = intp(accountRouteWeight(*row, pol))
}

// classifyRow applies policy + structure checks to one discovered dir. The caller has
// already confirmed acctDir is an account dir (projects/ for Claude, opencode.json for
// opencode) before invoking this.
func classifyRow(acctDir, product, account string, pol Policy, acctIdx accountsRegistryIndex) Account {
	tag := AccountTag(account)
	note := pol.Notes[tag]
	base := Account{Dir: acctDir, Product: product, Account: account, Tag: tag, Notes: note}

	st, err := os.Stat(acctDir)
	if err != nil || !st.IsDir() {
		base.Kind = KindNonAccount
		base.Reason = "not a directory"
		return base
	}
	// Intrinsic tombstones: a `.DELETED` marker or a dogfood home is off the roster
	// regardless of policy.
	if reason := intrinsicExclusion(account); reason != "" {
		base.Kind = KindExcluded
		base.Reason = reason
		return base
	}
	id := Identity{}
	if product == "claude" {
		id = ReadAccountIdentity(acctDir)
	} else if product == "codex" {
		id, _, _ = codexLoginStatus(acctDir, tag)
	}
	// seat is this dir's ACTIVE config-home registry entry, when it has one; the zero Home
	// otherwise, which reads as the historical OAuth seat everywhere downstream.
	//
	// Both folds below are Claude-only on purpose: the ~/.claude-accounts registry is a Claude
	// config-home registry, and its role names (especially "default") are not globally unique
	// product ids, so applying it to a `.codex` row would falsely inherit a Claude
	// tombstone/rehome with the same short tag.
	seat := configaccounts.Home{}
	if product == "claude" {
		if reason := accountsRegistryExclusion(acctDir, product, account, tag, id, acctIdx); reason != "" {
			base.Kind = KindExcluded
			base.Reason = reason
			return base
		}
		// An enrolled api-key dir INHERITS its seat's credential reference (the kind + the
		// env-var NAME, never the secret) so its readiness is read by KIND below instead of by
		// the OAuth disk probe, which would mis-report it as credential-less (#5331).
		if h, ok := acctIdx.activeSeat(acctDir, product, account, tag); ok &&
			h.CredentialKind() == configaccounts.CredKindAPIKey {
			seat = h
			base.CredKind = configaccounts.CredKindAPIKey
			base.APIKeyEnv = h.APIKeyEnv
		}
	}
	if reason := policyExclusion(account, tag, pol, id.LoginEmail); reason != "" {
		base.Kind = KindExcluded
		base.Reason = reason
		return base
	}
	label := "real offered account"
	switch product {
	case "opencode":
		label = "configured opencode account; serving requires active inference probe"
	case "codex":
		label = "real offered Codex account"
	}
	if base.CredKind == configaccounts.CredKindAPIKey {
		label = apiKeySeatReason(base.APIKeyEnv)
	}
	row := base
	row.Kind = KindWorker
	row.Reason = label
	stampProfile(&row, pol)
	if product == "claude" {
		row.AccountUUID = strp(id.AccountUUID)
		row.LoginEmail = strp(id.LoginEmail)
		row.OrgUUID = strp(id.OrgUUID)
		row.OrgType = strp(id.OrgType)
		row.Plan = strp(id.Plan)
		st, can := claudeLoginStatus(acctDir, tag, seat)
		row.LoginStatus = strp(string(st))
		row.CanServe = boolp(can)
	} else if product == "codex" {
		codexID, st, can := codexLoginStatus(acctDir, tag)
		row.AccountUUID = strp(codexID.AccountUUID)
		row.LoginStatus = strp(string(st))
		row.CanServe = boolp(can)
	} else if product == "opencode" {
		// OpenCode account homes are admitted only after opencode.json is parsed by
		// stampProfile above. Project that concrete configured-home witness into the
		// same readiness contract dispatch preflight requires from every backend.
		status := string(configaccounts.LoginReady)
		row.LoginStatus = &status
		row.CanServe = boolp(true)
	}
	return row
}

// codexLoginStatus projects the generic config-home identity reader into the roster's
// credential-safe identity/login fields. The reader consumes only auth.json account metadata
// and credential presence; it never returns token bytes. A home with config.toml but no live
// auth remains visible in the picker as needs_login and is not offerable.
func codexLoginStatus(acctDir, tag string) (Identity, configaccounts.LoginStatus, bool) {
	profile, ok := harnessprofile.Lookup("codex")
	if !ok {
		return Identity{}, configaccounts.LoginNeedsLogin, false
	}
	derived := configaccounts.DeriveIdentityForProfile(acctDir, profile)
	home := configaccounts.Home{Name: tag, Dir: acctDir, Identity: derived}
	st := home.LoginStatus()
	return Identity{AccountUUID: derived.AccountUUID}, st, st == configaccounts.LoginReady
}

// claudeLoginStatus folds a Claude config dir into the shared login verdict. seat is the dir's
// ACTIVE config-home registry entry (the zero Home when the dir is not enrolled): only its
// credential REFERENCE is consumed — the KIND and, for an api-key seat, the env-var NAME — so
// the derivation dispatches on kind (DerivedIdentity) rather than always probing the OAuth
// disk credential. Without that, an api-key seat read as HasCreds=false and reported
// needs_login/can_serve=false, which made it unroutable no matter how the registry was
// authored (#5331). An OAuth seat carries an empty kind and so keeps the exact prior path.
func claudeLoginStatus(acctDir, tag string, seat configaccounts.Home) (configaccounts.LoginStatus, bool) {
	h := configaccounts.Home{
		Name:      tag,
		Dir:       acctDir,
		CredKind:  seat.CredKind,
		APIKeyEnv: seat.APIKeyEnv,
	}
	h.Identity = h.DerivedIdentity()
	st := h.LoginStatus()
	// The .credentials.json expiry reprieve below is an OAUTH-disk concept: an api-key seat is
	// ready exactly when its named env var holds a key, so a stale OAuth blob left in its dir
	// must never reopen it. Gating on kind is what keeps the fail-open reprieve from becoming
	// a false "serving" verdict for a seat whose key is simply not set.
	if st == configaccounts.LoginNeedsLogin && h.CredentialKind() != configaccounts.CredKindAPIKey {
		exp := ReadCredExpiry(acctDir)
		if exp.HasExpiry && !exp.NeedsLogin(time.Now().UTC()) {
			return configaccounts.LoginReady, true
		}
	}
	return st, st == configaccounts.LoginReady
}

// discoverAPIKeySeats surfaces the ACTIVE api-key seats (#5331) that directory discovery
// structurally cannot reach. Discovery globs config DIRS, but an api-key seat's credential is
// an env-var reference rather than a directory — accounts.Validate deliberately exempts an
// active api-key home from the "must have a dir" rule — so a dir-less seat would never appear
// on the roster and could never be routed to a dispatch worker. This pass folds each such seat
// in straight from the registry. A seat the glob ALREADY produced a row for is skipped (by
// cleaned dir first, then by name), so the directory-driven path stays the single owner of
// every dir it can see and existing rows are untouched.
func discoverAPIKeySeats(reg configaccounts.Registry, pol Policy, found []Account) []Account {
	seenDirs := map[string]bool{}
	seenTags := map[string]bool{}
	for _, r := range found {
		if r.Product != "claude" {
			continue
		}
		if k := pathKey(r.Dir); k != "" {
			seenDirs[k] = true
		}
		if t := strings.ToLower(strings.TrimSpace(r.Tag)); t != "" {
			seenTags[t] = true
		}
	}
	var rows []Account
	for _, h := range reg.Homes {
		if !h.Active() || h.CredentialKind() != configaccounts.CredKindAPIKey {
			continue
		}
		if k := pathKey(h.Dir); k != "" && seenDirs[k] {
			continue
		}
		name := strings.ToLower(strings.TrimSpace(h.Name))
		if name != "" && seenTags[name] {
			continue
		}
		row := apiKeySeatRow(h, pol)
		if name != "" {
			seenTags[name] = true
		}
		if t := strings.ToLower(strings.TrimSpace(row.Tag)); t != "" {
			seenTags[t] = true
		}
		rows = append(rows, row)
	}
	return rows
}

// apiKeySeatReason is the roster reason text for an offered api-key seat. It names the env
// var by NAME only — the reference the registry stores — so the roster never carries, and can
// never print, the key itself. Both roster paths use it, so a seat reads the same whether it
// happened to be reached through its config dir or straight from the registry.
func apiKeySeatReason(env string) string {
	return "real offered API-key account (credential from $" + env + ")"
}

// apiKeySeatRow builds the roster row for one ACTIVE api-key registry seat. It is the
// registry-driven twin of classifyRow: the same intrinsic and policy gates decide whether the
// seat is offered, and an offered seat gets the same worker profile stamp — but its readiness
// comes from the seat's credential KIND (is the named env var set?) instead of from probing a
// config dir, which a dir-less api-key seat does not have. The row names the env VAR only; the
// key itself is never read here, let alone reported.
func apiKeySeatRow(h configaccounts.Home, pol Policy) Account {
	account := strings.TrimSpace(h.Name)
	if strings.TrimSpace(h.Dir) != "" {
		account = filepath.Base(h.Dir)
	}
	tag := AccountTag(account)
	base := Account{
		Dir:       h.Dir,
		Product:   "claude",
		Account:   account,
		Tag:       tag,
		Notes:     pol.Notes[tag],
		CredKind:  configaccounts.CredKindAPIKey,
		APIKeyEnv: h.APIKeyEnv,
	}
	if reason := intrinsicExclusion(account); reason != "" {
		base.Kind = KindExcluded
		base.Reason = reason
		return base
	}
	// No login email to match on: an api-key seat's identity derivation is offline and yields
	// no org email (deriving one needs a live Console probe of the key).
	if reason := policyExclusion(account, tag, pol, ""); reason != "" {
		base.Kind = KindExcluded
		base.Reason = reason
		return base
	}
	row := base
	row.Kind = KindWorker
	row.Reason = apiKeySeatReason(h.APIKeyEnv)
	stampProfile(&row, pol)
	// Re-derive by kind off the registry seat itself, so Status/Enabled still gate the verdict
	// (a disabled api-key seat reports disabled, not ready) exactly as they do for an OAuth one.
	seat := h
	seat.Identity = h.DerivedIdentity()
	st := seat.LoginStatus()
	row.AccountUUID = strp(seat.Identity.AccountUUID)
	row.LoginEmail = strp(seat.Identity.Email)
	row.OrgUUID = strp("")
	row.OrgType = strp("")
	row.Plan = strp("")
	row.LoginStatus = strp(string(st))
	row.CanServe = boolp(st == configaccounts.LoginReady)
	return row
}

// discoverProduct globs pattern under root and folds each match into an Account
// row: the common "not a directory" guard first (every product agrees on that),
// then a product-specific second gate (extraCheck) that returns a non-empty
// Reason to short-circuit as a KindNonAccount row, or "" to fall through to
// classifyRow. This is the glob-then-classify idiom discoverClaude, discoverCodex, and
// discoverOpencode share; only the glob pattern, the product tag, and the second gate differ.
func discoverProduct(root, pattern, product string, pol Policy, acctIdx accountsRegistryIndex, extraCheck func(acctDir string) (reason string)) []Account {
	var rows []Account
	matches, _ := filepath.Glob(filepath.Join(root, pattern))
	for _, acctDir := range matches {
		account := filepath.Base(acctDir)
		tag := AccountTag(account)
		note := pol.Notes[tag]
		st, err := os.Stat(acctDir)
		if err != nil || !st.IsDir() {
			rows = append(rows, Account{Dir: acctDir, Product: product, Account: account,
				Tag: tag, Kind: KindNonAccount, Reason: "not a directory", Notes: note})
			continue
		}
		if reason := extraCheck(acctDir); reason != "" {
			rows = append(rows, Account{Dir: acctDir, Product: product, Account: account,
				Tag: tag, Kind: KindNonAccount, Reason: reason, Notes: note})
			continue
		}
		rows = append(rows, classifyRow(acctDir, product, account, pol, acctIdx))
	}
	return rows
}

func discoverClaude(home string, pol Policy, acctIdx accountsRegistryIndex) []Account {
	return discoverProduct(home, ".claude*", "claude", pol, acctIdx, func(acctDir string) string {
		pst, perr := os.Stat(filepath.Join(acctDir, "projects"))
		if perr != nil || !pst.IsDir() {
			return "no projects/ subdir"
		}
		return ""
	})
}

func discoverCodex(home string, pol Policy, acctIdx accountsRegistryIndex) []Account {
	return discoverProduct(home, ".codex*", "codex", pol, acctIdx, func(acctDir string) string {
		for _, marker := range []string{"auth.json", "config.toml"} {
			if mst, merr := os.Stat(filepath.Join(acctDir, marker)); merr == nil && !mst.IsDir() {
				return ""
			}
		}
		return "no auth.json or config.toml"
	})
}

func discoverOpencode(configHome string, pol Policy, acctIdx accountsRegistryIndex) []Account {
	return discoverProduct(configHome, "opencode*", "opencode", pol, acctIdx, func(acctDir string) string {
		for _, m := range OpencodeMarkerFiles {
			if mst, merr := os.Stat(filepath.Join(acctDir, m)); merr == nil && !mst.IsDir() {
				return ""
			}
		}
		return "no opencode.json config"
	})
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

// reconcileIdentities detects config-home worker dirs sharing ONE logged-in provider account,
// stamping identity_role (unique|canonical|duplicate|no-login), identity_peers, and
// tag_login_match. Duplicates stay visible but callers exclude them from routing counts.
func reconcileIdentities(rows []Account) {
	// Gather config-home workers. OpenCode is env/config based and has no per-home identity
	// reader, so it remains outside this dedup fold.
	var workers []*Account
	for i := range rows {
		if rows[i].Kind == KindWorker && (rows[i].Product == "claude" || rows[i].Product == "codex") {
			workers = append(workers, &rows[i])
		}
	}
	byUUID := map[string][]*Account{}
	for _, r := range workers {
		uuid := derefStr(r.AccountUUID)
		if uuid != "" {
			byUUID[r.Product+"\x00"+uuid] = append(byUUID[r.Product+"\x00"+uuid], r)
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
		group := byUUID[r.Product+"\x00"+uuid]
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
			provider := "provider"
			if r.Product == "claude" {
				provider = "Anthropic"
			} else if r.Product == "codex" {
				provider = "Codex"
			}
			r.Reason = "duplicate identity: same " + provider + " account as " +
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
func derefFloat64(p *float64) float64 {
	if p == nil {
		return 0
	}
	return *p
}

func derefInt(p *int) int {
	if p == nil {
		return 0
	}
	return *p
}

// Discover classifies every account config dir across all products, folds in the config-home
// registry's api-key seats that own no discoverable dir, then reconciles shared Claude/Codex
// identities. Rows are sorted by (product, kind != worker, tag) to match
// fleet_accounts.discover_accounts.
func Discover(home, configHome string, pol Policy) []Account {
	reg := loadAccountsRegistry(home)
	acctIdx := indexAccountsRegistry(reg)
	rows := append(discoverClaude(home, pol, acctIdx), discoverCodex(home, pol, acctIdx)...)
	rows = append(rows, discoverOpencode(configHome, pol, acctIdx)...)
	// Fold in the api-key seats the directory glob structurally cannot reach — a seat whose
	// credential is an env var need not own a config dir at all (#5331).
	rows = append(rows, discoverAPIKeySeats(reg, pol, rows)...)
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
