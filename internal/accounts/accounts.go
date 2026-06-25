// Package accounts is the durable registry of Claude config HOMES — the
// CLAUDE_CONFIG_DIR "seats" a host switches between (~/.claude, ~/.claude-gem8-seat,
// …) — with one job no other surface does cleanly: resolve a seat name to the home
// that actually serves it, FOLLOWING a tombstone to its rehome target.
//
// WHY THIS EXISTS. A config home's directory NAME is set once and never re-checked
// against the account it is actually logged into. So ~/.claude-q-seat can be logged
// in as gem8, and `switch to q` silently lands on gem8 — a guess dressed as a fact.
// And when an account is retired (its subscription disabled, its dir renamed
// .DELETED-…), anything still pinned to it breaks instead of falling forward to a
// live seat. This package makes both first-class and typed:
//
//   - IDENTITY IS DISK-DERIVED, NEVER AUTHORED. Discover/DeriveIdentity read each
//     home's .claude.json (oauthAccount) + .credentials.json, so the logged-in email
//     is ground truth and a name that disagrees with it is FLAGGED (Home.NameLie),
//     never silently trusted.
//   - A TOMBSTONE CARRIES A REHOME. A retired seat is Status=tombstoned with a
//     RehomeTo naming a live seat; Resolve follows the chain transitively, so a
//     session/launcher pinned to a dead account auto-rehomes to a better one instead
//     of failing. A tombstone with no rehome, a dangling rehome, or a cycle is a
//     fail-loud Validate error — never a silent fallback to an arbitrary seat.
//
// This is the in-product, typed sibling of the private fleet's policy file
// (tools/_registry/accounts_policy.json, where `exclude` == tombstoned) and the
// dos roster (~/.claude/accounts.yaml, name → config_dir). It is provider-neutral
// and credential-safe: it records WHICH directory and WHO it is logged in as, never
// a secret — the registry is safe to read, diff, and back up.
//
// The package is pure stdlib. Resolve/Validate/Default are pure functions over a
// Registry VALUE; Discover/DeriveIdentity/LoadRegistry do the read-only disk I/O.
// It is DISTINCT from internal/modelroute's Account/Roster, which switches PROVIDER
// CREDENTIALS for a routed model id — a different concern at a different layer.
package accounts

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// RegistryVersion is this registry's on-disk schema tag. It is DISTINCT from
// modelroute.RosterVersion (a separate file, a different concern): a config-home
// registry vs a provider-credential roster. A registry MAY omit it (treated as
// current); one naming a different major is refused.
const RegistryVersion = "fak-config-homes/v1"

// Status is the lifecycle of a config-home seat. A CLOSED set: an empty status reads
// as Active, so a hand-written registry need only mark the tombstones.
type Status string

const (
	// StatusActive is a live seat that can serve sessions directly.
	StatusActive Status = "active"
	// StatusTombstoned is a retired seat (account disabled, dir decommissioned). It
	// MUST carry a RehomeTo so anything pinned to it falls forward to a live seat.
	StatusTombstoned Status = "tombstoned"
)

// Identity is the DISK-DERIVED truth about which account a config home is logged into.
// It is filled by DeriveIdentity from the home's own files, never authored by hand, so
// a home's NAME can never silently disagree with its login. It holds an email + the
// account UUID + a non-secret fingerprint of the setup token, never a credential value.
type Identity struct {
	Email       string `json:"email,omitempty"`
	AccountUUID string `json:"account_uuid,omitempty"`
	HasCreds    bool   `json:"has_creds"`
	Exists      bool   `json:"exists"`
	// TokenFP is a short, one-way fingerprint (SHA-256 prefix) of the home's setup
	// token (<dir>/.oauth-token), or "" when absent. It is NOT the token: the secret is
	// read, hashed, and discarded. It exists because the interactive login
	// (.claude.json / .credentials.json) and the setup token can name DIFFERENT
	// accounts in one dir — a headless `claude -p` that honors CLAUDE_CODE_OAUTH_TOKEN
	// then burns the TOKEN's account, not the dir-name's. Two homes with the same
	// TokenFP share one rate-limit bucket regardless of what their .claude.json says, so
	// Reconcile uses it to collapse phantom duplicates the email/UUID alone would miss.
	TokenFP string `json:"token_fp,omitempty"`
}

// AccountKey is the identity a seat collapses onto for dedup: two seats with the same
// AccountKey are the SAME account (one rate-limit bucket) and must never be counted as
// independent capacity. It prefers the interactive login's AccountUUID (ground truth for
// WHO the dir is logged in as); only when that is absent does it fall back to the
// setup-token fingerprint, so a half-set-up token-only dir still groups. Empty when the
// home has neither a login nor a token (nothing to collapse on).
func (id Identity) AccountKey() string {
	if id.AccountUUID != "" {
		return "uuid:" + id.AccountUUID
	}
	if id.TokenFP != "" {
		return "tok:" + id.TokenFP
	}
	return ""
}

// Home is one Claude config home (a CLAUDE_CONFIG_DIR seat). Name is the roster handle
// the launcher references ("gem8-seat"); Dir is the config-home path. Status empty or
// "active" is a live seat; "tombstoned" REQUIRES RehomeTo. Default marks the preferred
// single-session seat. Identity is disk-derived (filled by Discover/refresh), advisory
// for display + the NameLie check — never the secret.
type Home struct {
	Name     string   `json:"name"`
	Dir      string   `json:"dir,omitempty"`
	Status   Status   `json:"status,omitempty"`
	Default  bool     `json:"default,omitempty"`
	RehomeTo string   `json:"rehome_to,omitempty"`
	Role     string   `json:"role,omitempty"`
	Note     string   `json:"note,omitempty"`
	Identity Identity `json:"identity,omitempty"`
	// HistoryAt names this seat's history BUNDLE in the registry's shared-history store
	// (a path relative to Registry.SharedHistory, defaulting to Name). A tombstoned seat
	// keeps its sessions/projects in the SHARED store — not trapped in a home that may be
	// renamed away — so a rehome can pull them on demand. Empty for a live seat that
	// hasn't deposited.
	HistoryAt string `json:"history_at,omitempty"`
}

// bundle returns this seat's absolute history-bundle path under store (HistoryAt, or
// the seat Name when HistoryAt is empty).
func (h Home) bundle(store string) string {
	rel := h.HistoryAt
	if rel == "" {
		rel = h.Name
	}
	return filepath.Join(store, rel)
}

// Active reports whether the seat serves sessions directly (empty status == active).
func (h Home) Active() bool { return h.Status == "" || h.Status == StatusActive }

// Registry is the on-disk set of config-home seats. Homes is the whole roster; a seat's
// status decides whether Resolve serves it or follows its rehome. SharedHistory is the
// root of the controlled store where retired seats deposit their session/project history
// (so it is not trapped in a home that may be renamed away); a rehome PULLS the relevant
// bundle from there on demand (see PullPlan).
type Registry struct {
	Version       string `json:"version,omitempty"`
	SharedHistory string `json:"shared_history,omitempty"`
	Homes         []Home `json:"homes"`
}

// PullPlan is the recipe for serving a (possibly tombstoned) name: the live config dir
// to launch under, and the history bundles a rehome should pull into it from the shared
// store — one per tombstoned hop on the way to the live seat, nearest-first. With no
// tombstone hop, From is empty (nothing to pull). It is what `fak accounts pull` and any
// auto-rehome execute.
type PullPlan struct {
	Name string   `json:"name"`           // the requested seat
	Into Home     `json:"into"`           // the resolved LIVE seat (launch CLAUDE_CONFIG_DIR=Into.Dir)
	From []string `json:"from,omitempty"` // shared-store bundle paths to pull in, nearest tombstone first
}

// Plan resolves name to a PullPlan: the live seat it rehomes to, plus the shared-store
// history bundles to pull (the tombstoned hops' bundles). It reuses Resolve's
// fail-loud chain walk, and requires SharedHistory to be set when there is anything to
// pull.
func (r Registry) Plan(name string) (PullPlan, error) {
	live, chain, err := r.Resolve(name)
	if err != nil {
		return PullPlan{}, err
	}
	plan := PullPlan{Name: name, Into: live}
	for _, hop := range chain {
		h, _ := r.home(hop) // present: Resolve walked it
		if r.SharedHistory == "" {
			return PullPlan{}, fmt.Errorf("accounts: %q rehomes through tombstone %q but the registry has no shared_history store to pull from", name, hop)
		}
		plan.From = append(plan.From, h.bundle(r.SharedHistory))
	}
	return plan, nil
}

// home returns the seat with the given name.
func (r Registry) home(name string) (Home, bool) {
	for _, h := range r.Homes {
		if h.Name == name {
			return h, true
		}
	}
	return Home{}, false
}

// Default returns the seat marked default (the preferred single-session identity).
func (r Registry) Default() (Home, bool) {
	for _, h := range r.Homes {
		if h.Default {
			return h, true
		}
	}
	return Home{}, false
}

// Resolve returns the LIVE seat that serves name, following a tombstone's RehomeTo
// transitively. chain is the ordered list of tombstoned names hopped through (empty
// when name was already active), so a caller can warn "q is tombstoned → rehomed to
// gem8". It is fail-loud: an unknown name, a tombstone with no RehomeTo, a dangling
// rehome target, or a rehome cycle is an error — never a silent fallback to an
// arbitrary seat.
func (r Registry) Resolve(name string) (Home, []string, error) {
	var chain []string
	seen := make(map[string]bool)
	cur := name
	for {
		h, ok := r.home(cur)
		if !ok {
			if len(chain) == 0 {
				return Home{}, nil, fmt.Errorf("accounts: no home named %q", name)
			}
			return Home{}, chain, fmt.Errorf("accounts: %q rehomes to %q, which is not in the registry", chain[len(chain)-1], cur)
		}
		if h.Active() {
			return h, chain, nil
		}
		if seen[cur] {
			return Home{}, chain, fmt.Errorf("accounts: rehome cycle through %q", cur)
		}
		seen[cur] = true
		chain = append(chain, cur)
		if h.RehomeTo == "" {
			return Home{}, chain, fmt.Errorf("accounts: home %q is tombstoned with no rehome_to", cur)
		}
		cur = h.RehomeTo
	}
}

// serveable reports whether a seat can run a session right now: it is active (not
// tombstoned) and its config home exists on disk with live credentials. It reads the
// disk-derived Identity, so callers should Refresh the registry first.
func (r Registry) serveable(h Home) bool {
	return h.Active() && h.Identity.Exists && h.Identity.HasCreds
}

// Serve resolves name to the seat that should actually run it, REHOMING BY DEFAULT.
// This is the non-aggressive default: pinning to one exact account is brittle (the
// account gets retired, throttled, or logged out), so unless a seat can serve right now,
// resolution falls FORWARD — a tombstoned seat follows its rehome_to (as Resolve does),
// and a live-but-unserveable seat (missing dir / no credentials) likewise falls forward,
// to its rehome_to if set, else the registry's Default seat. chain reports the hops
// (requested -> … -> served) so a caller can explain the redirect. An already-serveable
// requested seat is returned as-is (rehome only kicks in when needed). Use Resolve when
// you truly need to PIN to an exact seat. Serve reads disk-derived Identity, so Refresh
// first; an unknown name is still fail-loud (a typo must not silently rehome).
func (r Registry) Serve(name string) (Home, []string, error) {
	var chain []string
	seen := make(map[string]bool)
	cur := name
	for {
		h, ok := r.home(cur)
		if !ok {
			if len(chain) == 0 {
				return Home{}, nil, fmt.Errorf("accounts: no home named %q", name)
			}
			return Home{}, chain, fmt.Errorf("accounts: %q rehomes to %q, which is not in the registry", chain[len(chain)-1], cur)
		}
		if r.serveable(h) {
			return h, chain, nil
		}
		if seen[cur] {
			return Home{}, chain, fmt.Errorf("accounts: rehome cycle through %q", cur)
		}
		seen[cur] = true
		chain = append(chain, cur)
		next := h.RehomeTo
		if next == "" {
			d, ok := r.Default()
			if !ok || d.Name == cur {
				return Home{}, chain, fmt.Errorf("accounts: %q cannot serve and has no rehome_to or default to fall forward to", cur)
			}
			next = d.Name
		}
		cur = next
	}
}

// Validate checks the registry is well-formed and that every tombstone resolves. The
// invariants, each a fail-loud boundary:
//   - a known major version; at least one home;
//   - each home: a non-empty, unique name; a status in the closed set; an active home
//     carries a dir; a tombstoned home carries a rehome_to that is not itself;
//   - at most one default, and the default is not tombstoned;
//   - every tombstone Resolves to a live seat (no dangling rehome, no cycle).
//
// A misconfigured registry must fail here, never fall through to an arbitrary seat.
func (r Registry) Validate() error {
	if r.Version != "" && !strings.HasPrefix(r.Version, RegistryVersion) {
		return fmt.Errorf("accounts: registry version %q is not %s.x", r.Version, RegistryVersion)
	}
	if len(r.Homes) == 0 {
		return fmt.Errorf("accounts: registry has no homes")
	}
	seen := make(map[string]bool, len(r.Homes))
	defaults := 0
	for i, h := range r.Homes {
		if h.Name == "" {
			return fmt.Errorf("accounts: home %d has an empty name", i)
		}
		if seen[h.Name] {
			return fmt.Errorf("accounts: duplicate home name %q", h.Name)
		}
		seen[h.Name] = true
		switch h.Status {
		case "", StatusActive, StatusTombstoned:
		default:
			return fmt.Errorf("accounts: home %q has unknown status %q", h.Name, h.Status)
		}
		if h.Default {
			defaults++
			if !h.Active() {
				return fmt.Errorf("accounts: default home %q is tombstoned", h.Name)
			}
		}
		if h.Active() {
			if h.Dir == "" {
				return fmt.Errorf("accounts: active home %q has no dir", h.Name)
			}
		} else {
			if h.RehomeTo == "" {
				return fmt.Errorf("accounts: tombstoned home %q needs a rehome_to", h.Name)
			}
			if h.RehomeTo == h.Name {
				return fmt.Errorf("accounts: home %q rehomes to itself", h.Name)
			}
		}
	}
	if defaults > 1 {
		return fmt.Errorf("accounts: %d homes marked default (at most one allowed)", defaults)
	}
	// Referential: every tombstone must reach a live seat (catches dangling rehome
	// targets and cycles, transitively).
	for _, h := range r.Homes {
		if !h.Active() {
			if _, _, err := r.Resolve(h.Name); err != nil {
				return err
			}
		}
	}
	return nil
}

// ---------------------------------------------------------------------------
// NAME-vs-IDENTITY — surface a seat whose dir name disagrees with its login.
// ---------------------------------------------------------------------------

// NameLie reports whether this seat's NAME disagrees with the account actually logged
// into it (per the disk-derived Identity). It is the q-seat-is-really-gem8 detector:
// true when a meaningful token of the name stem is absent from the login email's local
// part. Advisory (a display flag), and only meaningful once Identity is filled; a home
// with no derived email is never a lie.
func (h Home) NameLie() bool {
	if h.Identity.Email == "" {
		return false
	}
	// "default" is the bare-`claude` seat (~/.claude) — a ROLE name, not a claim about
	// which account it is — so it is never a name-lie regardless of who's logged in.
	if strings.EqualFold(h.Name, "default") {
		return false
	}
	local := h.Identity.Email
	if at := strings.IndexByte(local, '@'); at >= 0 {
		local = local[:at]
	}
	localNorm := normAlnum(local)
	for _, tok := range nameTokens(h.Name) {
		if !strings.Contains(localNorm, normAlnum(tok)) {
			return true
		}
	}
	return false
}

// nameTokens splits a seat name into identity-bearing tokens, dropping role/product
// suffixes (-seat, -claude) that say nothing about WHO the account is.
func nameTokens(name string) []string {
	s := strings.ToLower(name)
	s = strings.TrimSuffix(s, "-seat")
	s = strings.TrimSuffix(s, "-claude")
	var out []string
	for _, t := range strings.FieldsFunc(s, func(r rune) bool { return r == '-' || r == '_' || r == '.' }) {
		switch t {
		case "", "seat", "claude":
		default:
			out = append(out, t)
		}
	}
	return out
}

// normAlnum lowercases and keeps only [a-z0-9], so "jack.barker" and "jack-barker"
// compare equal and separators don't manufacture a false mismatch.
func normAlnum(s string) string {
	var b strings.Builder
	for _, r := range strings.ToLower(s) {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') {
			b.WriteRune(r)
		}
	}
	return b.String()
}

// ---------------------------------------------------------------------------
// RECONCILIATION — collapse N config homes that are really ONE account.
// ---------------------------------------------------------------------------
//
// A host accretes config homes faster than it retires them: ~/.claude,
// ~/.claude-gem8-netra, ~/.claude-day24-netra, a leftover ~/.claude-q-…DELETED — and
// several end up logged into the SAME account (one rate-limit bucket) by an honest
// mistake (a copied .oauth-token, a re-login that landed on the wrong account). Keyed
// purely on the dir NAME, the roster then shows one account as several independent
// "serving" seats, so a spread fans onto what is really one window and they all wall
// together. Reconcile is the auto-handler: it groups seats by the account each truly
// resolves to and elects ONE canonical per account, so a roster can collapse the rest
// instead of double-counting them — and it flags the subtler split where a dir's setup
// TOKEN belongs to a different account than its interactive login.

// IdentityRole classifies an active seat within its resolved-account group.
type IdentityRole string

const (
	// RoleUnique is the only seat on its account — nothing to collapse.
	RoleUnique IdentityRole = "unique"
	// RoleCanonical is the kept seat when several share one account.
	RoleCanonical IdentityRole = "canonical"
	// RoleDuplicate is a seat that shares another's account; collapse it onto Canonical.
	RoleDuplicate IdentityRole = "duplicate"
	// RoleNoLogin is a seat with no derivable identity (no login, no token) — ungroupable.
	RoleNoLogin IdentityRole = "no-login"
)

// SeatIdentity is the derived dedup verdict for one active seat. It is computed on
// demand from disk-derived Identity (never persisted) — the reconciliation sibling of
// the advisory NameLie flag.
type SeatIdentity struct {
	Name string `json:"name"`
	// Role places this seat in its account group (unique/canonical/duplicate/no-login).
	Role IdentityRole `json:"role"`
	// Account is the grouping key (uuid:… or tok:…); "" when the seat has no identity.
	Account string `json:"account,omitempty"`
	// Canonical is the seat this one collapses onto (== Name for unique/canonical).
	Canonical string `json:"canonical,omitempty"`
	// Peers are the other seat names on the SAME account (sorted), empty when unique.
	Peers []string `json:"peers,omitempty"`
	// TokenTwin lists seats sharing this seat's setup-token but a DIFFERENT interactive
	// login — the split-identity warning: a headless launch here may burn THEIR account's
	// bucket, not the one this dir's name/login implies.
	TokenTwin []string `json:"token_twin,omitempty"`
}

// Reconcile groups the registry's ACTIVE seats by the account each resolves to and
// returns a per-seat verdict keyed by seat name. Tombstoned seats are excluded (a
// tombstone already collapses via RehomeTo). It is pure over the homes' disk-derived
// Identity, so Refresh first for a live answer. The election is deterministic and reads
// no disk: a seat whose NAME matches its login beats a name-lie; a named seat beats the
// generic "default"; a seat with live creds beats one without; ties break on the
// lexically smaller name.
func (r Registry) Reconcile() map[string]SeatIdentity {
	var active []Home
	for _, h := range r.Homes {
		if h.Active() {
			active = append(active, h)
		}
	}
	byAccount := map[string][]Home{}
	byToken := map[string][]Home{}
	for _, h := range active {
		if k := h.Identity.AccountKey(); k != "" {
			byAccount[k] = append(byAccount[k], h)
		}
		if fp := h.Identity.TokenFP; fp != "" {
			byToken[fp] = append(byToken[fp], h)
		}
	}
	out := make(map[string]SeatIdentity, len(active))
	for _, h := range active {
		key := h.Identity.AccountKey()
		si := SeatIdentity{Name: h.Name, Account: key, Canonical: h.Name}
		if key == "" {
			si.Role = RoleNoLogin
		} else {
			group := byAccount[key]
			for _, g := range group {
				if g.Name != h.Name {
					si.Peers = append(si.Peers, g.Name)
				}
			}
			sort.Strings(si.Peers)
			canon := canonicalSeat(group)
			si.Canonical = canon.Name
			switch {
			case len(group) == 1:
				si.Role = RoleUnique
			case h.Name == canon.Name:
				si.Role = RoleCanonical
			default:
				si.Role = RoleDuplicate
			}
		}
		// Split identity: same setup token, different interactive login.
		if fp := h.Identity.TokenFP; fp != "" {
			for _, g := range byToken[fp] {
				if g.Name != h.Name && g.Identity.AccountUUID != h.Identity.AccountUUID {
					si.TokenTwin = append(si.TokenTwin, g.Name)
				}
			}
			sort.Strings(si.TokenTwin)
		}
		out[h.Name] = si
	}
	return out
}

// canonicalSeat picks the seat a same-account group collapses onto (see Reconcile for
// the ordering). group is non-empty.
func canonicalSeat(group []Home) Home {
	best := group[0]
	for _, h := range group[1:] {
		if canonRank(h) > canonRank(best) ||
			(canonRank(h) == canonRank(best) && h.Name < best.Name) {
			best = h
		}
	}
	return best
}

// canonRank scores a seat's fitness to be its account's canonical home: a truthful name
// dominates, then a named (non-"default") seat, then live credentials.
func canonRank(h Home) int {
	rank := 0
	if !h.NameLie() {
		rank += 4
	}
	if !strings.EqualFold(h.Name, "default") {
		rank += 2
	}
	if h.Identity.HasCreds {
		rank++
	}
	return rank
}

// ---------------------------------------------------------------------------
// DISCOVERY — disk truth: which ~/.claude* dirs exist + who they're logged in as.
// ---------------------------------------------------------------------------

// claudeConfig is the slice of a home's .claude.json this package reads: just the
// logged-in account. Unknown fields are ignored (the file carries hundreds).
type claudeConfig struct {
	OAuthAccount struct {
		EmailAddress string `json:"emailAddress"`
		AccountUUID  string `json:"accountUuid"`
	} `json:"oauthAccount"`
}

// DeriveIdentity reads the disk truth for one config-home dir: whether it exists, who
// it is logged in as (.claude.json oauthAccount), and whether it holds live credentials
// (.credentials.json). It never returns an error — a missing/unreadable file just
// leaves the corresponding field zero, so a half-set-up home reads as "exists, no creds".
func DeriveIdentity(dir string) Identity {
	var id Identity
	if dir == "" {
		return id
	}
	fi, err := os.Stat(dir)
	if err != nil || !fi.IsDir() {
		return id
	}
	id.Exists = true
	if b, err := os.ReadFile(filepath.Join(dir, ".claude.json")); err == nil {
		var c claudeConfig
		if json.Unmarshal(b, &c) == nil {
			id.Email = c.OAuthAccount.EmailAddress
			id.AccountUUID = c.OAuthAccount.AccountUUID
		}
	}
	if fi, err := os.Stat(filepath.Join(dir, ".credentials.json")); err == nil && !fi.IsDir() {
		id.HasCreds = true
	}
	id.TokenFP = tokenFingerprint(dir)
	return id
}

// tokenFingerprint returns a short, non-secret fingerprint of a config home's setup
// token (<dir>/.oauth-token), or "" when the file is absent/empty. It is a SHA-256
// prefix of the token bytes — one-way, so two seats sharing one token fingerprint share
// one rate-limit bucket without the registry ever storing or exposing the secret. The
// token is read, hashed, and discarded; only the 12-hex-char fingerprint is retained.
func tokenFingerprint(dir string) string {
	b, err := os.ReadFile(filepath.Join(dir, ".oauth-token"))
	if err != nil {
		return ""
	}
	tok := bytes.TrimSpace(b)
	if len(tok) == 0 {
		return ""
	}
	sum := sha256.Sum256(tok)
	return hex.EncodeToString(sum[:6])
}

// isConfigHome reports whether dir looks like a Claude config home rather than an
// adjacent ~/.claude-* directory (backups, a monitor cache). The marker mirrors the
// fleet's: a .claude.json or a projects/ subdir.
func isConfigHome(dir string) bool {
	if fi, err := os.Stat(filepath.Join(dir, ".claude.json")); err == nil && !fi.IsDir() {
		return true
	}
	if fi, err := os.Stat(filepath.Join(dir, "projects")); err == nil && fi.IsDir() {
		return true
	}
	return false
}

// Discover globs home/.claude* and returns a Home per config-home directory with its
// disk-derived Identity, sorted by name for determinism. The seat name is the dir
// basename with the ".claude-" prefix stripped (".claude" itself → "default"). Adjacent
// non-home dirs (…-account-backups, …-monitor) are skipped via isConfigHome. The
// returned homes have no Status/RehomeTo — Discover reports what EXISTS; lifecycle is
// overlaid from a registry or policy by the caller.
func Discover(home string) ([]Home, error) {
	matches, err := filepath.Glob(filepath.Join(home, ".claude*"))
	if err != nil {
		return nil, fmt.Errorf("accounts: glob %s: %w", home, err)
	}
	var homes []Home
	for _, m := range matches {
		fi, err := os.Stat(m)
		if err != nil || !fi.IsDir() || !isConfigHome(m) {
			continue
		}
		base := filepath.Base(m)
		name := "default"
		if base != ".claude" {
			name = strings.TrimPrefix(base, ".claude-")
		}
		homes = append(homes, Home{Name: name, Dir: m, Identity: DeriveIdentity(m)})
	}
	sort.Slice(homes, func(i, j int) bool { return homes[i].Name < homes[j].Name })
	return homes, nil
}

// Refresh re-derives every home's Identity from disk in place, so a loaded registry's
// cached identities reflect the current logins (a home re-/logged-in since it was
// written). It mutates the receiver's Homes and returns it for chaining.
func (r Registry) Refresh() Registry {
	for i := range r.Homes {
		r.Homes[i].Identity = DeriveIdentity(r.Homes[i].Dir)
	}
	return r
}

// ---------------------------------------------------------------------------
// LOAD / DUMP — the JSON registry round-trip (mirrors modelroute's Roster).
// ---------------------------------------------------------------------------

// JSON renders the registry as canonical indented JSON, stamping the current
// RegistryVersion when absent and newline-terminating it for clean redirection.
func (r Registry) JSON() []byte {
	out := r
	if out.Version == "" {
		out.Version = RegistryVersion
	}
	b, _ := json.MarshalIndent(out, "", "  ")
	return append(b, '\n')
}

// ParseRegistry decodes and validates a registry. Unknown JSON fields are REJECTED
// (DisallowUnknownFields) so a typo fails loudly instead of silently changing which
// seat serves a name.
func ParseRegistry(b []byte) (Registry, error) {
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.DisallowUnknownFields()
	var r Registry
	if err := dec.Decode(&r); err != nil {
		return Registry{}, fmt.Errorf("accounts: parse registry: %w", err)
	}
	if err := r.Validate(); err != nil {
		return Registry{}, err
	}
	return r, nil
}

// LoadRegistry reads and validates a registry from a file path.
func LoadRegistry(path string) (Registry, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return Registry{}, fmt.Errorf("accounts: read registry %s: %w", path, err)
	}
	return ParseRegistry(b)
}
