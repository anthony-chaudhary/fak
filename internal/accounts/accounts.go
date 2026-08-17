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
	"time"

	"github.com/anthony-chaudhary/fak/internal/canon"
	"github.com/anthony-chaudhary/fak/internal/maputil"
)

// RegistryVersion is this registry's on-disk schema tag, and RegistryFamily is the
// prefix every version of THIS file shares. It is DISTINCT from modelroute.RosterVersion
// (a separate file, a different concern): a config-home registry vs a provider-credential
// roster. A registry MAY omit the version (treated as current); one in the same family
// (fak-config-homes/*) is accepted so additive, omitempty schema growth (e.g. the policy
// attributes on Home) never strands an existing file; one from a FOREIGN family is refused.
// RegistryFamily is exported so the `fak accounts version` surface can report the family a
// binary supports — the line that makes a stale binary visible instead of silently lacking a verb.
const (
	RegistryVersion = "fak-config-homes/v1"
	RegistryFamily  = "fak-config-homes/"
)

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
	// APIKeyEnv, when non-empty, marks this as an API-KEY seat's identity (CredKindAPIKey,
	// #5331): the credential is an Anthropic API key held in the environment variable of
	// this NAME (never the secret — this is the reference, mirroring modelroute's CredEnv).
	// It is the disk-independent credential handle AccountKey folds into an "apikey:" bucket
	// when no org/workspace UUID is known, so two seats on the same key env collapse.
	// DeriveAPIKeyIdentity fills it; an OAuth seat leaves it empty.
	APIKeyEnv string `json:"api_key_env,omitempty"`
}

// AccountKey is the identity a seat collapses onto for dedup: two seats with the same
// AccountKey are the SAME account (one rate-limit bucket) and must never be counted as
// independent capacity. It prefers the interactive login's AccountUUID (ground truth for
// WHO the dir is logged in as); only when that is absent does it fall back to the
// setup-token fingerprint, so a half-set-up token-only dir still groups. Empty when the
// home has neither a login nor a token (nothing to collapse on).
func (id Identity) AccountKey() string {
	if id.AccountUUID != "" {
		return UUIDBucketKey(id.AccountUUID)
	}
	if id.TokenFP != "" {
		return "tok:" + id.TokenFP
	}
	// An API-key seat (#5331) has no OAuth login/UUID and no setup token; offline it
	// buckets on its env-var reference so two seats on the same key env still collapse.
	// TODO(#5331): once a live probe resolves the key's org UUID, AccountUUID above wins.
	if id.APIKeyEnv != "" {
		return APIKeyBucketKey(id.APIKeyEnv)
	}
	return ""
}

// UUIDBucketKey builds the account-bucket key for an interactive-login AccountUUID — the
// "uuid:"-prefixed form AccountKey() elects when a login UUID is present. It is exported so a
// caller that already holds a UUID (e.g. the runtime headroom fold in
// cmd/fak/accounts_headroom.go, which keys its scored map by the same bucket the cooldown
// overlay walls) builds the SAME string AccountKey() would, instead of hand-restating the
// prefix and risking a silent drift that would break bucket alignment across the seam.
func UUIDBucketKey(uuid string) string {
	return "uuid:" + uuid
}

// Home is one Claude config home (a CLAUDE_CONFIG_DIR seat). Name is the roster handle
// the launcher references ("gem8-seat"); Dir is the config-home path. Status empty or
// "active" is a live seat; "tombstoned" REQUIRES RehomeTo. Identity is disk-derived (filled
// by Discover/refresh), advisory for display + the NameLie check — never the secret.
//
// Which seat is the launch default vs the rehome anchor now lives in Registry.Roles, not on
// the Home — see the Default field note below.
type Home struct {
	Name   string `json:"name"`
	Dir    string `json:"dir,omitempty"`
	Status Status `json:"status,omitempty"`
	// Default is the LEGACY per-home anchor flag, retained only so a pre-roles registry still
	// decodes (DisallowUnknownFields would otherwise reject the field). On load, migrate folds
	// a true value into Roles[RoleAnchor] and clears it, so a freshly-saved registry never
	// carries it. Do NOT read it — use Registry.Role / Registry.Default. It will be dropped in
	// a future schema version.
	Default  bool     `json:"default,omitempty"`
	RehomeTo string   `json:"rehome_to,omitempty"`
	Terminal bool     `json:"terminal,omitempty"`
	Role     string   `json:"role,omitempty"`
	Note     string   `json:"note,omitempty"`
	Identity Identity `json:"identity,omitempty"`
	// Policy ATTRIBUTES of the one account — the fields that used to be restated by hand in
	// the dos roster (~/.claude/accounts.yaml) and the job roster
	// (job/config/claude_accounts.yaml). Carrying them here makes the registry the single
	// source of truth and those two files GENERATED views (see SyncViews). They are
	// AUTHORED (not disk-derived like Identity), so Discover preserves them across a rescan.
	//
	// Enabled is a pointer so "absent" (nil) is distinguishable from an explicit false: a
	// nil Enabled reads as TRUE (EnabledOrDefault), matching the rosters' "default true"
	// semantics, so an old v1 registry with no enabled field stays fully enrolled.
	Enabled       *bool  `json:"enabled,omitempty"`
	Reserved      bool   `json:"reserved,omitempty"`
	ChromeProfile string `json:"chrome_profile,omitempty"`
	// CredKind names the credential KIND this seat's identity is derived from (#5331). Empty
	// (the default, and every pre-#5331 registry) reads as CredKindOAuth — the historical
	// subscription-OAuth seat. CredKindAPIKey marks an Anthropic API-key seat whose credential
	// is an API key held in the env var APIKeyEnv names. It is AUTHORED (like the policy
	// attributes above), so Discover/MergeDiscovered preserve it across a rescan.
	CredKind CredKind `json:"cred_kind,omitempty"`
	// APIKeyEnv is the NAME of the environment variable holding this seat's Anthropic API key
	// (e.g. "ANTHROPIC_API_KEY"). It is a REFERENCE, never the secret — the same posture as
	// modelroute.Account.CredEnv. Set only on a CredKindAPIKey seat; `fak accounts launch`
	// fronts the guard with `--api-key-env <APIKeyEnv>` so the child reads the key from env.
	APIKeyEnv string `json:"api_key_env,omitempty"`
	// BaseURL points this seat at a THIRD-PARTY Anthropic-compatible endpoint (a vendor
	// gateway that speaks the Messages API under its own tenant credential) instead of
	// first-party api.anthropic.com. Empty — every registry written before this field — means
	// first-party, so the historical launch is byte-for-byte unchanged.
	//
	// It is AUTHORED, like the policy attributes above, so a rescan preserves it. It is NOT a
	// secret (it is a URL), but it IS load-bearing for correctness: `fak guard` fronts the
	// child with its OWN ANTHROPIC_BASE_URL at guard's loopback gateway and proxies using the
	// credential guard holds, so a guarded launch would silently ignore this endpoint and bill
	// the wrong upstream. `fak accounts launch` therefore REFUSES a guarded launch of a seat
	// that carries a BaseURL rather than quietly routing elsewhere (see accounts_launch.go).
	BaseURL string `json:"base_url,omitempty"`
	// ExtraEnv is extra environment handed to the agent this seat launches. It exists because
	// a third-party endpoint is not describable by CLAUDE_CONFIG_DIR alone: the vendor's model
	// id, its required custom headers, and the client-side bootstrap toggles it needs
	// (ANTHROPIC_MODEL, ANTHROPIC_CUSTOM_HEADERS, CLAUDE_CODE_USE_GATEWAY, …) are read from the
	// ENVIRONMENT by the agent binary at startup, not from any file fak controls. Before this
	// field the launch env was exactly `os.Environ() + CLAUDE_CONFIG_DIR`, with no seam for
	// them, so such a seat could not be expressed at all.
	//
	// It holds NON-SECRET values only, and that is ENFORCED, not merely asked for
	// (ValidateExtraEnv): the registry is a plaintext file that `fak accounts list --json`
	// prints, so a credential parked here would leak into ordinary operator output and into
	// any log capturing it. A seat's credential keeps the existing reference posture — fak
	// stores the env var's NAME (APIKeyEnv) and the operator supplies the value in the
	// launching environment, so the secret never enters the registry.
	ExtraEnv map[string]string `json:"extra_env,omitempty"`
	// Tombstone audit trail, canonical here so the generated views need not strand it: when
	// this seat was retired and why. RehomeTo (above) is the third audit field. Empty for a
	// live seat. These retain retirement evidence in the canonical registry without exposing it in operator rosters.
	TombstonedAt    string `json:"tombstoned_at,omitempty"`
	TombstoneReason string `json:"tombstone_reason,omitempty"`
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

// EnabledOrDefault reads the Enabled attribute with its default-true semantics: a nil
// pointer (the field omitted, e.g. an older registry) reads as enabled, so an account is
// only OUT of the rosters when it carries an explicit `"enabled": false`. This mirrors the
// dos/job rosters, where a row with no `enabled:` key is a rotation candidate.
func (h Home) EnabledOrDefault() bool { return h.Enabled == nil || *h.Enabled }

// Registry is the on-disk set of config-home seats. Homes is the whole roster; a seat's
// status decides whether Resolve serves it or follows its rehome. SharedHistory is the
// root of the controlled store where retired seats deposit their session/project history
// (so it is not trapped in a home that may be renamed away); a rehome PULLS the relevant
// bundle from there on demand (see PullPlan).
type Registry struct {
	Version       string `json:"version,omitempty"`
	SharedHistory string `json:"shared_history,omitempty"`
	Homes         []Home `json:"homes"`
	// Roles maps a well-known ROLE to the home name that currently fills it. It replaces the
	// per-Home `default: true` boolean, which conflated two genuinely different jobs onto one
	// flag: the seat a bare/interactive launch should run as (RoleActive — rotates as you
	// switch working accounts) and the seat everything falls FORWARD onto when its own seat
	// can't serve (RoleAnchor — a stable, always-available anchor you rarely change). One
	// boolean could not hold both, so setting the active account silently moved the rehome
	// anchor. As a map it also extends to a new role (a reserved fallback, a billing owner)
	// without a schema change, and the "which seat is X?" answer lives in ONE place instead of
	// scattered across every Home. A role value MUST name an active, serveable home (Validate
	// enforces it). Empty/absent is legal — a registry need not fill every role. A legacy
	// `default: true` migrates to RoleAnchor on load (see migrate), preserving the old
	// Serve() fall-forward behavior, where Default WAS the anchor.
	Roles map[string]string `json:"roles,omitempty"`
	// Views holds the per-consumer config blocks (the dos roster's defaults; the job
	// roster's defaults/rotation/launch) that used to live ONLY in the generated files. It
	// is keyed by view name ("dos", "job"), and each value is an opaque config tree carried
	// verbatim so a consumer can add a config key without a schema change here. SyncViews
	// projects each view's blocks back out. Empty when no view config has been adopted yet.
	Views map[string]ViewConfig `json:"views,omitempty"`
}

// Well-known roles a home can fill (see Registry.Roles). The set is open — a consumer may
// define its own role key — but these two are load-bearing in this package:
const (
	// RoleActive is the seat a bare / interactive `claude` launch and the watchdog run as.
	// It is the "default active account" an operator picks; it ROTATES as you switch the
	// account you are working as. Surfaced as active_default in the dos view.
	RoleActive = "active"
	// RoleAnchor is the rehome fall-forward target: when a seat can't serve and has no
	// rehome_to of its own, Serve collapses onto the anchor. It should be a stable,
	// always-available account; you rarely change it. (This is what `default: true` meant
	// inside Serve before roles existed.)
	RoleAnchor = "anchor"
)

// Role returns the home filling the named role, following the same lookup as home(): the role
// must be set AND name a present home. ok is false when the role is unset or dangles (Validate
// rejects a dangling role, so a loaded registry never dangles).
func (r Registry) Role(role string) (Home, bool) {
	name, ok := r.Roles[role]
	if !ok || name == "" {
		return Home{}, false
	}
	return r.home(name)
}

// ViewConfig is one generated view's non-account config: the named top-level YAML blocks it
// carries below `accounts:` (e.g. "defaults", "rotation", "launch"),
// each an arbitrary nested map emitted as YAML by the projector. Order is fixed by BlockOrder
// so the generated file is byte-stable across runs.
type ViewConfig struct {
	// Blocks maps a top-level YAML key to its (arbitrary, nested) value.
	Blocks map[string]any `json:"blocks,omitempty"`
	// BlockOrder fixes the emission order of Blocks' keys (a JSON object is unordered).
	// Keys present in Blocks but absent here are emitted after, sorted, for determinism.
	BlockOrder []string `json:"block_order,omitempty"`
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

// Default returns the seat that fills the rehome ANCHOR role — the seat everything falls
// forward onto. It is a compatibility shim over Role(RoleAnchor): before roles, the
// per-Home `default: true` boolean named exactly this seat (Serve's fall-forward target), so
// callers that asked for the "default" wanted the anchor. New code should call Role directly
// (RoleActive for the launch seat, RoleAnchor for the fall-forward seat) — the two the old
// single Default conflated.
func (r Registry) Default() (Home, bool) { return r.Role(RoleAnchor) }

// ActiveMemoryDir resolves the ACTIVE persona's agent-memory directory for workspace —
// the store a recall (e.g. dos_recall) should read from instead of the hardcoded
// ~/.claude default. It is the affordance that lets a caller aim recall at the seat the
// operator is actually working as (RoleActive — the rotating launch seat), not whichever
// home happens to be ~/.claude.
//
// It returns <activeSeat.Dir>/projects/<key>/memory, where <key> is the Claude Code
// session-store slug for workspace: every non-alphanumeric rune of the cleaned absolute
// path collapsed to '-' (so "C:\\work\\fak" -> "C--work-fak"). This is the SAME slug
// Claude Code derives for projects/<cwd> on disk (see projectSlug), so the path lands on
// the real per-workspace store rather than a guessed one.
//
// Fail-closed: with no active seat (RoleActive unset or dangling) it returns ("", false) —
// never a guessed path. It deliberately does NOT fall forward to the anchor: "the active
// store" is a precise question and RoleActive is its only honest answer; the anchor
// fall-forward in Serve/fallbackSeat is a different concern (a seat that can't SERVE),
// not "whose memory is active". A caller that wants the anchor's store can call
// Role(RoleAnchor) and join the same suffix.
func (r Registry) ActiveMemoryDir(workspace string) (string, bool) {
	h, ok := r.Role(RoleActive)
	if !ok || h.Dir == "" {
		return "", false
	}
	return filepath.Join(h.Dir, "projects", projectSlug(workspace), "memory"), true
}

// projectSlug is the on-disk session-store key Claude Code derives from a working
// directory: it cleans the path, then collapses every NON-alphanumeric rune (separators,
// the drive colon, '.', spaces) to a single '-' each — so "C:\\work\\fak" keys as
// "C--work-fak" and "/home/u/p" as "-home-u-p". It mirrors the slug the fleet's Python
// resume/checkpoint tools compute (re.sub(r"[^A-Za-z0-9]", "-", normpath(path))) so the
// Go and Python views agree on which projects/<key> dir a workspace owns.
func projectSlug(path string) string {
	clean := filepath.Clean(path)
	var b strings.Builder
	b.Grow(len(clean))
	for _, r := range clean {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9':
			b.WriteRune(r)
		default:
			b.WriteByte('-')
		}
	}
	return b.String()
}

// migrate folds a legacy registry forward in place: a pre-roles registry carried the rehome
// anchor as a per-Home `default: true`. If no anchor role is set but some home carries the
// legacy flag, adopt it as RoleAnchor (preserving Serve's old fall-forward target), then clear
// the boolean so the registry has ONE representation of the anchor. It is idempotent and runs
// before Validate, so both LoadRegistry and the SaveRegistry round-trip self-check see the
// same migrated shape.
func (r *Registry) migrate() {
	legacyIdx := -1
	for i := range r.Homes {
		if r.Homes[i].Default {
			legacyIdx = i
			break
		}
	}
	if legacyIdx >= 0 {
		if r.Roles == nil {
			r.Roles = map[string]string{}
		}
		if _, ok := r.Roles[RoleAnchor]; !ok {
			r.Roles[RoleAnchor] = r.Homes[legacyIdx].Name
		}
		// Clear every legacy flag — the anchor now lives in Roles, the single source.
		for i := range r.Homes {
			r.Homes[i].Default = false
		}
	}
}

// Resolve returns the LIVE seat that serves name, following a tombstone's RehomeTo
// transitively. chain is the ordered list of tombstoned names hopped through (empty
// when name was already active), so a caller can warn "q is tombstoned → rehomed to
// gem8". It is fail-loud: an unknown name, a tombstone with no RehomeTo, a dangling
// rehome target, or a rehome cycle is an error — never a silent fallback to an
// arbitrary seat.
func (r Registry) Resolve(name string) (Home, []string, error) {
	return r.walkRehome(name, Home.Active, func(h Home) (string, error) {
		if h.RehomeTo == "" {
			return "", fmt.Errorf("accounts: home %q is tombstoned with no rehome_to", h.Name)
		}
		return h.RehomeTo, nil
	})
}

// walkRehome is the shared rehome-chain walk behind Resolve and Serve. Starting at
// name, it follows the chain until stop(h) is true (the seat that serves), returning that
// seat and the ordered list of tombstoned names hopped through. It is fail-loud the same
// way for both callers: an unknown name, an unknown rehome target, or a cycle is an error
// — never a silent fallback. The callers differ only in stop (Active vs serveableAt)
// and in next, which yields the next name to walk to (or an error when there is nowhere
// to fall forward); next is consulted only after the current seat is recorded in chain.
func (r Registry) walkRehome(name string, stop func(Home) bool, next func(Home) (string, error)) (Home, []string, error) {
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
		if stop(h) {
			return h, chain, nil
		}
		if seen[cur] {
			return Home{}, chain, fmt.Errorf("accounts: rehome cycle through %q", cur)
		}
		seen[cur] = true
		chain = append(chain, cur)
		nxt, err := next(h)
		if err != nil {
			return Home{}, chain, err
		}
		cur = nxt
	}
}

// serveableAt reports whether a seat can run a session right now, with an optional
// usage-limit cooldown overlay. The static half is CanServe — the closed login status
// vocabulary owns that definition so launch, rotation, and human status surfaces do not
// drift: ready means active, enabled, present on disk, and carrying live credentials.
// The overlay half reads the CooldownStore the same way LoginReportAt does: an
// otherwise-serveable seat whose upstream ACCOUNT has an active cooldown at now is NOT
// serveable — rehoming onto it would just bounce off the same usage wall (#4673). A nil
// store or zero now applies no overlay, so serveableAt(h, nil, time.Time{}) equals
// h.CanServe() exactly. It reads disk-derived Identity, so callers should Refresh the
// registry first.
func serveableAt(h Home, cd *CooldownStore, now time.Time) bool {
	if !h.CanServe() {
		return false
	}
	_, cooled := cooldownFor(h, cd, now)
	return !cooled
}

// cooldownFor returns the active cooldown entry for h's upstream account at now, and
// whether one holds. A nil store or zero now means no cooldown overlay (never cooled),
// mirroring LoginReportAt's gate, so zero-value callers keep the pure, cooldown-blind
// fold.
func cooldownFor(h Home, cd *CooldownStore, now time.Time) (CooldownEntry, bool) {
	if cd == nil || now.IsZero() {
		return CooldownEntry{}, false
	}
	return cd.CooledDown(h.Identity.AccountKey(), now)
}

// Serve resolves name to the seat that should actually run it, REHOMING BY DEFAULT.
// This is the non-aggressive default: pinning to one exact account is brittle (the
// account gets retired, throttled, or logged out), so unless a seat can serve right now,
// resolution falls FORWARD — a tombstoned seat follows its rehome_to (as Resolve does),
// and a live-but-unserveable seat (disabled, missing dir, or no credentials) likewise
// falls forward, to its rehome_to if set, else the registry's Default seat. chain reports the hops
// (requested -> … -> served) so a caller can explain the redirect. An already-serveable
// requested seat is returned as-is (rehome only kicks in when needed). Use Resolve when
// you truly need to PIN to an exact seat. Serve reads disk-derived Identity, so Refresh
// first; an unknown name is still fail-loud (a typo must not silently rehome).
//
// Serve is ServeAt with no cooldown overlay: it cannot see usage-limit throttling, so a
// throttled-but-logged-in seat still reads as serving. A caller that holds the
// CooldownStore should call ServeAt so the fall-forward walks PAST throttled seats.
func (r Registry) Serve(name string) (Home, []string, error) {
	h, chain, _, err := r.ServeAt(name, nil, time.Time{})
	return h, chain, err
}

// ServeAt is Serve with the usage-limit cooldown overlay the rehome walk was blind to
// (#4673): the fall-forward stop predicate is serveableAt, so an otherwise-Ready seat
// whose account has an active cooldown at now is walked PAST — recorded in chain like
// any other non-serving hop — instead of stopped on. A nil cd or zero now applies no
// overlay, making ServeAt(name, nil, time.Time{}) behave exactly like Serve always has.
//
// The returned *CooldownEntry is the explicit all-cooled signal for the terminal case
// where the chain reaches ONLY cooled-down seats: rather than fail loud (a throttled
// seat that resets in 20 minutes still beats a hard resolution error) or silently land
// on a throttled seat (the bug this fixes), ServeAt returns the reachable
// otherwise-serveable seat with the SOONEST reset, its active cooldown entry (non-nil
// ⇔ degraded, ResetAt says when it returns), and the chain of hops that reached it.
// When any seat truly serves, the entry is nil. Structural failures — unknown name,
// dangling rehome, cycle among unserveable seats, nowhere to fall forward — stay
// fail-loud exactly as in Serve.
func (r Registry) ServeAt(name string, cd *CooldownStore, now time.Time) (Home, []string, *CooldownEntry, error) {
	// cooled collects the seats the walk refused ONLY because of an active cooldown —
	// otherwise-serveable pool members — with each one's active entry and the length of
	// the chain that reached it, so the all-cooled terminal case below can still resolve.
	type cooledSeat struct {
		home     Home
		entry    CooldownEntry
		chainLen int
	}
	var cooled []cooledSeat
	walked := 0
	stop := func(h Home) bool {
		if serveableAt(h, cd, now) {
			return true
		}
		if h.CanServe() {
			// Refused purely by the cooldown overlay: remember it as a degraded-mode
			// candidate. walkRehome appends h to chain right after a false stop, so the
			// chain that reaches h is exactly the walked hops so far.
			if e, ok := cooldownFor(h, cd, now); ok {
				cooled = append(cooled, cooledSeat{home: h, entry: e, chainLen: walked})
			}
		}
		walked++
		return false
	}
	// refused names every seat this walk has already turned down, so the fall-forward never
	// re-offers one. Without it, two unserveable ROLE seats (a dead active plus a dead
	// anchor) bounce off each other — anchor→active→anchor — and the walk dies on its own
	// cycle guard while a perfectly ready seat sits unexamined in the pool.
	refused := map[string]bool{}
	next := func(h Home) (string, error) {
		refused[h.Name] = true
		nxt := h.RehomeTo
		if nxt == "" {
			// No explicit rehome target: fall forward to a ROLE seat, then to the pool. Prefer
			// the anchor (its whole job is to be the always-available fall-forward); if no
			// anchor is set, the active seat is the next-best stable target. A role pointing
			// back at a seat this walk already refused can't help, so skip it.
			fb, ok := r.fallbackSeat(h.Name, refused, cd, now)
			if !ok {
				return "", fmt.Errorf("accounts: %q cannot serve and has no rehome_to, anchor, or active seat to fall forward to", h.Name)
			}
			nxt = fb
		}
		return nxt, nil
	}
	h, chain, err := r.walkRehome(name, stop, next)
	if err == nil {
		return h, chain, nil, nil
	}
	if len(cooled) == 0 {
		// No seat was refused for cooldown alone, so the walk failed for a structural
		// reason — identical to Serve's fail-loud behavior.
		return h, chain, nil, err
	}
	// Every reachable, otherwise-serveable seat is cooled down (the walk only ran out of
	// road because the overlay refused them all). Resolve to the soonest-reset one and
	// say so via its entry; ties keep the first-walked seat for determinism.
	best := cooled[0]
	for _, c := range cooled[1:] {
		if c.entry.ResetAt.Before(best.entry.ResetAt) {
			best = c
		}
	}
	entry := best.entry
	return best.home, chain[:best.chainLen], &entry, nil
}

// fallbackSeat returns the seat to fall forward onto when the seat named avoid can't serve
// and carries no rehome_to. It offers, in order:
//
//  1. the anchor role (its whole purpose is to be the always-available fall-forward);
//  2. the active role, the next-best stable target when no anchor is set;
//  3. any OTHER seat that can serve right now, in registry order — the pool rung;
//  4. any other seat refused ONLY by the cooldown overlay, so the walk still reaches it and
//     ServeAt's all-cooled terminal can degrade onto the soonest-reset seat.
//
// It never returns avoid, nor any seat in refused — the ones this walk already turned down.
// That skip is what makes rungs 3 and 4 reachable at all: with both role seats dead they
// point at each other, and re-offering a walked seat used to end the walk on ServeAt's
// cycle guard ("rehome cycle through …") while ready seats sat unexamined. A cycle among
// UNSERVEABLE seats is still fail-loud — the walk simply exhausts every rung first, so the
// error now means "nowhere left to go", not "the two role seats point at each other".
//
// ok is false when no rung offers a different seat.
func (r Registry) fallbackSeat(avoid string, refused map[string]bool, cd *CooldownStore, now time.Time) (string, bool) {
	skip := func(name string) bool { return name == avoid || refused[name] }
	for _, role := range []string{RoleAnchor, RoleActive} {
		if h, ok := r.Role(role); ok && !skip(h.Name) {
			return h.Name, true
		}
	}
	// Both roles are exhausted (unset, or already refused by this walk). Rather than dead-end,
	// sweep the pool: a seat that serves right now first, then one held back only by an active
	// cooldown — a seat that resets in 20 minutes still beats refusing to resolve at all.
	for _, want := range []func(Home) bool{
		func(h Home) bool { return serveableAt(h, cd, now) },
		Home.CanServe,
	} {
		for _, h := range r.Homes {
			if !skip(h.Name) && want(h) {
				return h.Name, true
			}
		}
	}
	return "", false
}

// Validate checks the registry is well-formed and that every tombstone resolves. The
// invariants, each a fail-loud boundary:
//   - a known major version; at least one home;
//   - each home: a non-empty, unique name; a status in the closed set; an active home
//     carries a dir; a tombstoned home may omit rehome_to only as an explicit terminal
//     tombstone (nothing in this harness remains to fall forward to);
//   - every role (active, anchor, …) names a present, ACTIVE home — never a typo or a
//     tombstone (a tombstone can't serve, so it can't fill a role);
//   - every tombstone Resolves to a live seat (no dangling rehome, no cycle).
//
// The legacy per-Home `default: true` is NOT validated here: migrate (run before Validate
// in ParseRegistry) has already folded it into RoleAnchor and cleared the flag, so by the
// time Validate runs the anchor lives only in Roles.
//
// A misconfigured registry must fail here, never fall through to an arbitrary seat.
func (r Registry) Validate() error {
	if r.Version != "" && !strings.HasPrefix(r.Version, RegistryFamily) {
		return fmt.Errorf("accounts: registry version %q is not in the %s* family", r.Version, RegistryFamily)
	}
	if len(r.Homes) == 0 {
		return fmt.Errorf("accounts: registry has no homes")
	}
	seen := make(map[string]bool, len(r.Homes))
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
		switch h.CredKind {
		case "", CredKindOAuth, CredKindAPIKey:
		default:
			return fmt.Errorf("accounts: home %q has unknown cred_kind %q", h.Name, h.CredKind)
		}
		// An API-key seat (#5331) MUST carry a valid env-var NAME as its credential reference
		// — never a pasted secret — and its identity is that key, not a dir, so it is exempt
		// from the active-home dir requirement below.
		if h.CredentialKind() == CredKindAPIKey {
			if !ValidAPIKeyEnvName(h.APIKeyEnv) {
				return fmt.Errorf("accounts: api-key home %q needs a valid api_key_env env-var name (never the secret), got %q", h.Name, h.APIKeyEnv)
			}
		} else if h.APIKeyEnv != "" {
			return fmt.Errorf("accounts: home %q sets api_key_env %q but is not an api_key seat", h.Name, h.APIKeyEnv)
		}
		if h.Active() {
			if h.Dir == "" && h.CredentialKind() != CredKindAPIKey {
				return fmt.Errorf("accounts: active home %q has no dir", h.Name)
			}
		} else {
			if h.RehomeTo == "" && !h.Terminal {
				return fmt.Errorf("accounts: tombstoned home %q needs a rehome_to or terminal=true", h.Name)
			}
			if h.RehomeTo != "" && h.Terminal {
				return fmt.Errorf("accounts: terminal tombstone %q must not carry rehome_to", h.Name)
			}
			if h.RehomeTo == h.Name {
				return fmt.Errorf("accounts: home %q rehomes to itself", h.Name)
			}
		}
	}
	// Every role must name a present, active home (in role-name order for a stable error).
	for _, role := range maputil.SortedKeys(r.Roles) {
		name := r.Roles[role]
		h, ok := r.home(name)
		if !ok {
			return fmt.Errorf("accounts: role %q names %q, which is not in the registry", role, name)
		}
		if !h.Active() {
			return fmt.Errorf("accounts: role %q names tombstoned seat %q (a role must be an active seat)", role, name)
		}
	}
	// Referential: every tombstone must reach a live seat (catches dangling rehome
	// targets and cycles, transitively).
	for _, h := range r.Homes {
		if !h.Active() && !h.Terminal {
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
// into it (per the disk-derived Identity). It is the q-seat-is-really-gem8 detector: a
// non-default seat with a known login whose name shares NOTHING with that login
// (nameMatch == 0). Flagging on NO match rather than ANY-token-absent is deliberate: an
// org-suffixed but truthful name (gem8-netra logged into gem8@netra…) still names the
// right account through its "gem8" token even though the "-netra" org suffix never
// appears in the email local part — so it must NOT read as a lie. Advisory (a display
// flag), meaningful only once Identity is filled; a home with no derived email, or a
// name that makes no identity claim, is never a lie.
func (h Home) NameLie() bool {
	if h.Identity.Email == "" {
		return false
	}
	// "default" is the bare-`claude` seat (~/.claude) — a ROLE name, not a claim about
	// which account it is — so it is never a name-lie regardless of who's logged in.
	if strings.EqualFold(h.Name, "default") {
		return false
	}
	if len(nameTokens(h.Name)) == 0 {
		return false // a name that makes no identity claim cannot lie
	}
	return h.nameMatch() == 0
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
			// Restored/recreated account dirs sometimes carry an operational suffix glued
			// to the identity token (gem8NEW-netra, day26NEW-netra). That suffix is not
			// an account claim; keep the identity-bearing base so a valid restore does
			// not look like a name lie, while gem8NEW logged into day26 still fails.
			if strings.HasSuffix(t, "new") && len(t) > len("new") {
				t = strings.TrimSuffix(t, "new")
			}
			out = append(out, t)
		}
	}
	return out
}

// normAlnum lowercases and keeps only [a-z0-9], so "jack.barker" and "jack-barker"
// compare equal and separators don't manufacture a false mismatch.
func normAlnum(s string) string { return canon.SqueezeAlnum(s) }

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

// canonRank scores a seat's fitness to be its account's canonical home: a name that
// names the login dominates (the more tokens it matches, the better), then a named
// (non-"default") seat beats the generic ~/.claude, then live credentials. Using the
// MATCH COUNT rather than the NameLie boolean keeps an org-suffixed truthful name
// (gem8-netra logged into gem8@…) ahead of the role-named "default", which NameLie —
// special-cased never-a-lie — would otherwise tie with or beat.
func canonRank(h Home) int {
	rank := h.nameMatch() * 8
	if !strings.EqualFold(h.Name, "default") {
		rank += 2
	}
	if h.Identity.HasCreds {
		rank++
	}
	return rank
}

// nameMatch counts how many of a seat's identity-bearing name tokens appear in its login
// email's local part — a positive signal that the NAME tells the truth about WHO the dir
// is logged into. 0 for "default" (a role, not an identity claim) and for a name that
// shares nothing with the login.
func (h Home) nameMatch() int {
	if h.Identity.Email == "" || strings.EqualFold(h.Name, "default") {
		return 0
	}
	local := h.Identity.Email
	if at := strings.IndexByte(local, '@'); at >= 0 {
		local = local[:at]
	}
	localNorm := normAlnum(local)
	n := 0
	for _, tok := range nameTokens(h.Name) {
		if t := normAlnum(tok); t != "" && strings.Contains(localNorm, t) {
			n++
		}
	}
	return n
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

// statConfigHome is the shared front of every per-home identity reader (Claude's
// DeriveIdentity and codex's deriveCodexIdentity): it seeds a zero Identity and reports
// whether dir is a usable config home to read further. When dir is empty or not an existing
// directory it returns the zero Identity with ok=false (the caller returns it as-is — a
// missing home reads as "does not exist"); otherwise it returns Identity{Exists: true} with
// ok=true so the caller can layer on the harness-specific credential/account fields.
func statConfigHome(dir string) (Identity, bool) {
	var id Identity
	if dir == "" {
		return id, false
	}
	fi, err := os.Stat(dir)
	if err != nil || !fi.IsDir() {
		return id, false
	}
	id.Exists = true
	return id, true
}

// DeriveIdentity reads the disk truth for one config-home dir: whether it exists, who
// it is logged in as (.claude.json oauthAccount), and whether it holds live credentials
// (.credentials.json). It never returns an error — a missing/unreadable file just
// leaves the corresponding field zero, so a half-set-up home reads as "exists, no creds".
func DeriveIdentity(dir string) Identity {
	id, ok := statConfigHome(dir)
	if !ok {
		return id
	}
	id.Email, id.AccountUUID = stateIdentity(dir)
	id.HasCreds = hasClaudeCredentials(dir)
	id.TokenFP = tokenFingerprint(dir)
	return id
}

// hasClaudeCredentials reports whether a config home has a credential Claude can
// actually use. A placeholder .credentials.json with a claudeAiOauth object but no
// access/refresh token is not a login; that was the July-4 drift where fak said
// "ready" while Claude and the job switcher correctly saw a missing session/login.
// Legacy/minimal fixtures that only prove the file exists still read as credentials
// when they do not carry a claudeAiOauth object.
//
// When the disk says no, the macOS login Keychain gets the last word (#5363): Claude
// Code on darwin stores the live credential there and writes NO .credentials.json, so
// without this fallback every healthy Mac seat read as needs_login. Non-darwin builds
// have no keychain seam and keep the disk-only answer byte-for-byte.
func hasClaudeCredentials(dir string) bool {
	if hasClaudeFileCredentials(dir) {
		return true
	}
	return ClaudeKeychainHasCreds(dir)
}

func hasClaudeFileCredentials(dir string) bool {
	if tok, err := os.ReadFile(filepath.Join(dir, ".oauth-token")); err == nil {
		trimmed := string(bytes.TrimSpace(tok))
		if strings.HasPrefix(trimmed, "sk-ant-oat") {
			return true
		}
	}
	path := filepath.Join(dir, ".credentials.json")
	b, err := os.ReadFile(path)
	if err != nil {
		return false
	}
	if fi, err := os.Stat(path); err != nil || fi.IsDir() {
		return false
	}
	var doc map[string]any
	if json.Unmarshal(b, &doc) != nil {
		return true
	}
	raw, ok := doc["claudeAiOauth"]
	if !ok {
		return true
	}
	oauth, ok := raw.(map[string]any)
	if !ok {
		return false
	}
	for _, key := range []string{"accessToken", "refreshToken"} {
		if v, ok := oauth[key].(string); ok && strings.TrimSpace(v) != "" {
			return true
		}
	}
	return false
}

// stateIdentity resolves who a config home is logged in as. A named home (launched with
// CLAUDE_CONFIG_DIR=<dir>) has exactly one state file, <dir>/.claude.json. The DEFAULT home
// (~/.claude) has TWO writers: an explicit CLAUDE_CONFIG_DIR=~/.claude launch writes
// <dir>/.claude.json, while a bare `claude` writes the profile-root ~/.claude.json — so after
// a bare /login the in-dir copy still carries the PREVIOUS account and the registry row lies
// about the seat's live identity (the stale-default-row regression: creds bill the new
// account while every roster surface still names the old one). For the default home the
// fresher of the two files wins; a file with no oauthAccount never outranks one that has it.
func stateIdentity(dir string) (email, uuid string) {
	inDir := filepath.Join(dir, ".claude.json")
	email, uuid = readOAuthAccount(inDir)
	if filepath.Base(dir) != ".claude" {
		return email, uuid
	}
	root := filepath.Join(filepath.Dir(dir), ".claude.json")
	rEmail, rUUID := readOAuthAccount(root)
	if rEmail == "" && rUUID == "" {
		return email, uuid
	}
	if (email == "" && uuid == "") || newerFile(root, inDir) {
		return rEmail, rUUID
	}
	return email, uuid
}

// readOAuthAccount reads the oauthAccount identity out of one .claude.json state file;
// zero values mean absent, unreadable, or not logged in.
func readOAuthAccount(path string) (email, uuid string) {
	b, err := os.ReadFile(path)
	if err != nil {
		return "", ""
	}
	var c claudeConfig
	if json.Unmarshal(b, &c) != nil {
		return "", ""
	}
	return c.OAuthAccount.EmailAddress, c.OAuthAccount.AccountUUID
}

// newerFile reports whether a was modified after b; a missing/unstat-able file sorts oldest.
func newerFile(a, b string) bool {
	afi, err := os.Stat(a)
	if err != nil {
		return false
	}
	bfi, err := os.Stat(b)
	if err != nil {
		return true
	}
	return afi.ModTime().After(bfi.ModTime())
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

// isTombstonedDir reports whether a config-dir basename carries the house tombstone
// marker (the `.DELETED-<date>` suffix `fak accounts remove --archive` renames a retired
// dir to). An archived dir keeps its .claude.json + projects/ intact, so the config-home
// markers below would otherwise re-admit it as a LIVE seat — the exact way a deleted seat
// resurfaced in the switcher when Discover ran without a registry to overlay the tombstone.
// This makes Discover honor the same `.DELETED` rule the policy-aware scanners already do
// (fleetaccounts.classifyRow / fleet_accounts._classify_row), so a decommissioned dir is
// never offered as a seat and MergeDiscovered never resurrects it into the registry.
func isTombstonedDir(dir string) bool {
	return strings.Contains(strings.ToLower(filepath.Base(dir)), ".deleted")
}

// isConfigHome reports whether dir looks like a Claude config home rather than an
// adjacent ~/.claude-* directory (backups, a monitor cache). The marker mirrors the
// fleet's: a .claude.json or a projects/ subdir. A tombstoned (`.DELETED`) dir is never a
// config home regardless of its contents — a retired seat must not be re-offered.
func isConfigHome(dir string) bool {
	if isTombstonedDir(dir) {
		return false
	}
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

// MergeDiscovered folds a fresh disk scan of home/.claude* INTO an existing canonical
// registry, returning the merged registry. It is the non-destructive regenerator the
// single-source model needs: identity is disk-derived truth, but the policy ATTRIBUTES
// (Status, Default, RehomeTo, Role, Note, Enabled, Reserved, ChromeProfile, BaseURL,
// ExtraEnv, HistoryAt) are
// AUTHORED and must survive a rescan. So for every home already in the registry it refreshes
// ONLY Identity (and self-heals Dir to the scan's canonical path form) and keeps every
// authored field; every config dir on disk that the registry does not yet know becomes a NEW
// active home (identity-only, policy defaults). A registry home whose dir has vanished from
// the scan is kept verbatim — pruning/tombstoning is an explicit operator decision (Stage
// 4/5), not a side effect of discovery.
//
// Matching is by NAME — the stable roster handle, which Discover derives from the same dir
// basename convention the roster uses (".claude-<name>" -> "<name>"). Name is preferred over
// Dir because the SAME directory can surface under different path representations across a
// scan (e.g. a /tmp vs C:/…/Temp form under MSYS), which would otherwise fork one seat into a
// spurious duplicate. A cleaned-path index is kept only as a secondary tie-break for a
// dir-less placeholder entry whose name differs.
func (r Registry) MergeDiscovered(home string) (Registry, error) {
	discovered, err := Discover(home)
	if err != nil {
		return r, err
	}
	byName := map[string]int{}
	byDir := map[string]int{}
	// tombstonedKeys indexes the ACCOUNT IDENTITY (uuid:/tok:) of every tombstoned home, so a
	// brand-new dir logged into a retired account is refused even when its dir/registry name
	// differs. A tombstone is keyed to a MUTABLE handle (the seat name / dir); re-logging the
	// same account into a fresh `.claude-<name>NEW` dir would otherwise present a new name the
	// name-match below treats as a stranger and admits as active — resurrecting the retired
	// account under an alias. The `.DELETED`-basename filter in isConfigHome only catches the
	// archived-dir case; this catches the re-login-into-a-new-dir case it cannot see.
	tombstonedKeys := map[string]bool{}
	for i, h := range r.Homes {
		byName[h.Name] = i
		if h.Dir != "" {
			byDir[filepath.Clean(h.Dir)] = i
		}
		if h.Status == StatusTombstoned {
			if k := h.Identity.AccountKey(); k != "" {
				tombstonedKeys[k] = true
			}
		}
	}
	out := r // copy header (Version, SharedHistory)
	out.Homes = append([]Home(nil), r.Homes...)
	for _, d := range discovered {
		idx, ok := byName[d.Name]
		if !ok {
			if j, okd := byDir[filepath.Clean(d.Dir)]; okd && out.Homes[j].Name == "" {
				idx, ok = j, true
			}
		}
		if ok {
			// Known seat: adopt the scan's canonical Dir + fresh Identity, preserve all
			// authored policy fields already in out.Homes[idx]. An API-key seat's dir carries
			// the same config-home markers the scan keys on, so the disk-shaped discovery would
			// mis-derive it as a credential-less OAuth home; re-derive it by its stored KIND
			// (from the env-var reference, #5331) instead of overwriting with the scan identity.
			out.Homes[idx].Dir = d.Dir
			if out.Homes[idx].CredentialKind() == CredKindAPIKey {
				out.Homes[idx].Identity = out.Homes[idx].DerivedIdentity()
			} else {
				out.Homes[idx].Identity = d.Identity
			}
			continue
		}
		// A dir the registry does not know BY NAME but whose account identity matches a
		// tombstoned seat is the same retired account under a new dir — never re-admit it as
		// active. Skipping it (rather than folding it onto the tombstone) keeps the tombstone's
		// authored dir/rehome intact; the shadow dir simply stays invisible to the roster.
		if k := d.Identity.AccountKey(); k != "" && tombstonedKeys[k] {
			continue
		}
		// Brand-new config dir the registry never knew: add as an active seat with policy
		// defaults (Enabled nil => default-true).
		out.Homes = append(out.Homes, d)
	}
	// Refresh identity for any KNOWN home the scan did NOT cover (its dir is outside `home`
	// or vanished) so a loaded registry's cached identities still reflect disk reality.
	covered := map[string]bool{}
	for _, d := range discovered {
		covered[d.Name] = true
	}
	for i := range out.Homes {
		// An API-key seat carries its credential as an env-var reference, not a dir, so it
		// must be refreshed even when it has no Dir the scan could cover (#5331).
		if covered[out.Homes[i].Name] {
			continue
		}
		if out.Homes[i].Dir != "" || out.Homes[i].CredentialKind() == CredKindAPIKey {
			out.Homes[i].Identity = out.Homes[i].DerivedIdentity()
		}
	}
	return out, nil
}

// Refresh re-derives every home's Identity from disk in place, so a loaded registry's
// cached identities reflect the current logins (a home re-/logged-in since it was
// written). It mutates the receiver's Homes and returns it for chaining.
func (r Registry) Refresh() Registry {
	for i := range r.Homes {
		// DerivedIdentity dispatches on credential KIND so an API-key seat re-derives from
		// its env-var reference (#5331) instead of being mis-read as a credential-less home.
		r.Homes[i].Identity = r.Homes[i].DerivedIdentity()
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
	r.migrate() // fold a legacy `default: true` into RoleAnchor before validating
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

// SaveRegistry validates r and writes it to path atomically: it marshals via JSON() (which
// stamps the version), parses the bytes back through ParseRegistry as a self-check so a
// registry that would not round-trip is NEVER persisted, then writes to a sibling temp file
// and renames it over path. The rename is the atomic step — a reader sees either the old
// file or the fully-written new one, never a half-written registry. This is the single
// writer the canonical-source model needs; before it, JSON() was the only serializer and
// nothing persisted the registry back.
func SaveRegistry(path string, r Registry) error {
	b := r.JSON()
	if _, err := ParseRegistry(b); err != nil {
		return fmt.Errorf("accounts: refusing to write invalid registry %s: %w", path, err)
	}
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("accounts: mkdir %s: %w", dir, err)
	}
	tmp, err := os.CreateTemp(dir, ".registry-*.tmp")
	if err != nil {
		return fmt.Errorf("accounts: temp registry in %s: %w", dir, err)
	}
	tmpName := tmp.Name()
	// Best-effort cleanup if we bail before the rename; after a successful rename the temp
	// no longer exists, so the Remove is a harmless no-op.
	defer os.Remove(tmpName)
	if _, err := tmp.Write(b); err != nil {
		tmp.Close()
		return fmt.Errorf("accounts: write temp registry %s: %w", tmpName, err)
	}
	if err := tmp.Close(); err != nil {
		return fmt.Errorf("accounts: close temp registry %s: %w", tmpName, err)
	}
	if err := os.Rename(tmpName, path); err != nil {
		return fmt.Errorf("accounts: rename %s -> %s: %w", tmpName, path, err)
	}
	return nil
}
