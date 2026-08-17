package accounts

import (
	"fmt"
	"sort"
	"time"
)

// login.go is the observable account-login layer for config-home seats. The
// registry already knows lifecycle, policy, and disk-derived identity; this file
// folds those facts into one closed status vocabulary that launchers, switchers,
// and humans can read without re-deriving "does this seat actually work?" from a
// handful of booleans.

const LoginReportSchema = "fak.accounts.login.v1"

// LoginStatus is the primary login/readiness state for a config-home seat.
// Exactly one status is assigned per seat.
type LoginStatus string

const (
	LoginReady      LoginStatus = "ready"
	LoginTombstoned LoginStatus = "tombstoned"
	LoginDisabled   LoginStatus = "disabled"
	LoginMissingDir LoginStatus = "missing_dir"
	LoginNeedsLogin LoginStatus = "needs_login"
	// LoginIdentityMismatch means the config home has credentials, but the disk-derived
	// login identity does not match the named seat. Treat it like a login wall for launch
	// purposes: serving would burn the wrong account's quota.
	LoginIdentityMismatch LoginStatus = "identity_mismatch"
	// LoginCooledDown means the seat is otherwise Ready — creds present, enabled — but its
	// upstream account recently hit a usage/weekly cap or a transient 429, so serving it now
	// would just bounce off the same wall. It is a time-boxed overlay, not a registry fact:
	// LoginStatus() never returns it (that fold stays pure); only LoginReportAt applies it
	// from the cooldown store, and it auto-clears when the window elapses. CanServe is false.
	LoginCooledDown LoginStatus = "cooled_down"
)

// LoginWarning is an auxiliary condition that does not necessarily stop a
// launch, but changes how a switcher/operator should interpret the seat.
type LoginWarning string

const (
	LoginWarningReserved          LoginWarning = "reserved"
	LoginWarningUnverifiedAccount LoginWarning = "unverified_account"
	LoginWarningNameLie           LoginWarning = "name_identity_mismatch"
	LoginWarningDuplicateBucket   LoginWarning = "duplicate_account_bucket"
	LoginWarningTokenTwin         LoginWarning = "split_setup_token"
)

// LoginObservation is the complete, credential-safe status record for one seat.
// It contains only directory names, non-secret identity, and token fingerprints
// already present in Identity; never a credential value.
type LoginObservation struct {
	Name         string         `json:"name"`
	Dir          string         `json:"dir,omitempty"`
	Status       LoginStatus    `json:"status"`
	CanServe     bool           `json:"can_serve"`
	Reason       string         `json:"reason,omitempty"`
	NextAction   string         `json:"next_action,omitempty"`
	Lifecycle    string         `json:"lifecycle"`
	RehomeTo     string         `json:"rehome_to,omitempty"`
	Roles        []string       `json:"roles,omitempty"`
	Account      string         `json:"account,omitempty"`
	Email        string         `json:"email,omitempty"`
	HasCreds     bool           `json:"has_creds"`
	Exists       bool           `json:"exists"`
	Enabled      bool           `json:"enabled"`
	Reserved     bool           `json:"reserved,omitempty"`
	IdentityRole IdentityRole   `json:"identity_role,omitempty"`
	Canonical    string         `json:"canonical,omitempty"`
	Peers        []string       `json:"peers,omitempty"`
	TokenTwin    []string       `json:"token_twin,omitempty"`
	Warnings     []LoginWarning `json:"warnings,omitempty"`
	// CredKind names the seat's credential kind (#5331): empty/omitted for the historical
	// subscription-OAuth seat, "api_key" for an Anthropic API-key seat. APIKeyEnv names the
	// env var holding that key (a REFERENCE, never the secret) and is set only on an api_key
	// seat — so a consumer of the --json surface can tell an org/API seat from an OAuth one.
	CredKind  CredKind `json:"cred_kind,omitempty"`
	APIKeyEnv string   `json:"api_key_env,omitempty"`
}

// LoginSummary is the rollup over a LoginReport.
type LoginSummary struct {
	Total int `json:"total"`
	// ActiveStyleSeats counts seats in the live, serviceable roster — Total minus
	// terminal/unusable seats (tombstoned, disabled, missing_dir). It is the honest
	// denominator for CanServe: the fraction of the working pool that can serve now.
	ActiveStyleSeats int            `json:"active_style_seats"`
	CanServe         int            `json:"can_serve"`
	DistinctAccounts int            `json:"distinct_accounts"`
	WarningSeats     int            `json:"warning_seats"`
	ByStatus         map[string]int `json:"by_status"`
}

// LoginReport is the machine-readable account-login status surface.
type LoginReport struct {
	Schema  string             `json:"schema"`
	Summary LoginSummary       `json:"summary"`
	Seats   []LoginObservation `json:"seats"`
}

// LoginStatus classifies this home using only facts already carried by the
// registry. Ready means the seat can be launched without dropping into /login:
// it is active, enabled, the config dir exists, and credentials are present.
func (h Home) LoginStatus() LoginStatus {
	switch {
	case !h.Active():
		return LoginTombstoned
	case !h.EnabledOrDefault():
		return LoginDisabled
	case !h.Identity.Exists:
		return LoginMissingDir
	case !h.Identity.HasCreds:
		return LoginNeedsLogin
	case h.NameLie():
		return LoginIdentityMismatch
	default:
		return LoginReady
	}
}

// CanServe reports whether this seat is ready to launch directly.
func (h Home) CanServe() bool { return h.LoginStatus() == LoginReady }

// ActiveStyle reports whether a seat in this login status belongs to the live,
// serviceable roster — the pool rotation draws from now or after a cooldown/login
// reset — as opposed to a terminal or unusable seat. Tombstoned (retired),
// disabled (administratively off), and missing_dir (no config home on disk) seats
// are not part of the working pool, so they are excluded. Everything else — ready,
// cooled_down, needs_login, identity_mismatch — is a real seat that is serving,
// will serve after a reset, or is recoverable by an operator login/rehome.
//
// This is the honest denominator for "how many of my seats can serve": a fleet of
// 36 where 22 are tombstoned is really a 14-seat pool, and reporting 5/36 badly
// understates the servable fraction.
func (s LoginStatus) ActiveStyle() bool {
	switch s {
	case LoginTombstoned, LoginDisabled, LoginMissingDir:
		return false
	default:
		return true
	}
}

// LoginReport folds every home into an observable status record plus a rollup.
// Call Refresh first when current disk state matters. It applies no cooldown
// overlay; use LoginReportAt to drop usage-limited seats from the servable pool.
func (r Registry) LoginReport() LoginReport {
	return r.LoginReportAt(nil, time.Time{})
}

// LoginReportAt is LoginReport with an optional usage-limit cooldown overlay. When
// cd is non-nil, an otherwise-Ready seat whose upstream account has an active
// cooldown at now is downgraded to LoginCooledDown (CanServe false), so the pool
// stops dispatching into a wall the seat cannot pass until the window elapses. A
// nil store or zero now means no overlay — behaviorally identical to LoginReport.
func (r Registry) LoginReportAt(cd *CooldownStore, now time.Time) LoginReport {
	rec := r.Reconcile()
	report := LoginReport{Schema: LoginReportSchema}
	for _, h := range r.Homes {
		report.Seats = append(report.Seats, r.loginObservation(h, rec[h.Name], cd, now))
	}
	report.Summary = summarizeLoginObservations(report.Seats)
	return report
}

// WithoutTombstoned returns the operator-facing roster. Retired seats stay in the
// canonical registry for audit and restore, but do not enter default selectors or TUI payloads.
func (r LoginReport) WithoutTombstoned() LoginReport {
	filtered := r
	filtered.Seats = make([]LoginObservation, 0, len(r.Seats))
	for _, obs := range r.Seats {
		if obs.Status != LoginTombstoned {
			filtered.Seats = append(filtered.Seats, obs)
		}
	}
	filtered.Summary = summarizeLoginObservations(filtered.Seats)
	return filtered
}

func summarizeLoginObservations(seats []LoginObservation) LoginSummary {
	summary := LoginSummary{ByStatus: map[string]int{}}
	accounts := map[string]bool{}
	for _, obs := range seats {
		summary.Total++
		summary.ByStatus[string(obs.Status)]++
		if obs.Status.ActiveStyle() {
			summary.ActiveStyleSeats++
		}
		if obs.CanServe {
			summary.CanServe++
		}
		if obs.Account != "" {
			accounts[obs.Account] = true
		}
		if len(obs.Warnings) > 0 {
			summary.WarningSeats++
		}
	}
	summary.DistinctAccounts = len(accounts)
	return summary
}

func (r Registry) loginObservation(h Home, si SeatIdentity, cd *CooldownStore, now time.Time) LoginObservation {
	status := h.LoginStatus()
	reason, action := LoginReasonAction(status, h)
	// Cooldown overlay: only an otherwise-Ready seat can be cooled. A seat that
	// already fails to serve for a static reason (needs-login, tombstoned, …)
	// keeps that more actionable status.
	if status == LoginReady && cd != nil && !now.IsZero() {
		if e, ok := cd.CooledDown(h.Identity.AccountKey(), now); ok {
			status = LoginCooledDown
			reason, action = cooldownReasonAction(e)
		}
	}
	// Org-auth-wall overlay (#4998): a witnessed upstream organization wall outranks
	// both the generic cooldown rendering and a later needs_login degradation — once
	// Claude blanks the unusable tokens, `/login` is exactly the repair the original
	// terminal 403 already proved futile, so the weaker cause must not displace the
	// stronger witnessed one. Only the ready/needs_login pair (plus the cooldown
	// rendering derived from ready) may be overridden: tombstoned, disabled, and
	// missing-dir are deliberate operator states whose repair is more actionable
	// than the wall's.
	if cd != nil && !now.IsZero() &&
		(status == LoginReady || status == LoginNeedsLogin || status == LoginCooledDown) {
		if e, ok := cd.OrgAuthWall(h.Identity.AccountKey(), now); ok {
			status = LoginOrgAuthWall
			reason, action = orgAuthWallReasonAction(e, now)
		}
	}
	obs := LoginObservation{
		Name:         h.Name,
		Dir:          h.Dir,
		Status:       status,
		CanServe:     status == LoginReady,
		Reason:       reason,
		NextAction:   action,
		Lifecycle:    lifecycleString(h),
		RehomeTo:     h.RehomeTo,
		Roles:        r.rolesFor(h.Name),
		Account:      h.Identity.AccountKey(),
		Email:        h.Identity.Email,
		HasCreds:     h.Identity.HasCreds,
		Exists:       h.Identity.Exists,
		Enabled:      h.EnabledOrDefault(),
		Reserved:     h.Reserved,
		IdentityRole: si.Role,
		Canonical:    si.Canonical,
		Peers:        append([]string(nil), si.Peers...),
		TokenTwin:    append([]string(nil), si.TokenTwin...),
		CredKind:     h.CredKind,
		APIKeyEnv:    h.APIKeyEnv,
	}
	if obs.IdentityRole == "" && obs.CanServe && obs.Account == "" {
		obs.IdentityRole = RoleNoLogin
	}
	if h.Reserved {
		obs.Warnings = append(obs.Warnings, LoginWarningReserved)
	}
	if obs.CanServe && obs.Account == "" {
		obs.Warnings = append(obs.Warnings, LoginWarningUnverifiedAccount)
	}
	if h.NameLie() {
		obs.Warnings = append(obs.Warnings, LoginWarningNameLie)
	}
	if si.Role == RoleDuplicate {
		obs.Warnings = append(obs.Warnings, LoginWarningDuplicateBucket)
	}
	if len(si.TokenTwin) > 0 {
		obs.Warnings = append(obs.Warnings, LoginWarningTokenTwin)
	}
	return obs
}

// LoginReasonAction returns the human reason and next action for a primary
// login status.
func LoginReasonAction(status LoginStatus, h Home) (string, string) {
	switch status {
	case LoginReady:
		if h.CredentialKind() == CredKindAPIKey {
			return "API key present in $" + h.APIKeyEnv, ""
		}
		return "config home has live credentials", ""
	case LoginTombstoned:
		if h.RehomeTo != "" {
			return "seat is tombstoned", "launch through its rehome target or restore it deliberately"
		}
		return "seat is tombstoned without a rehome target", "set rehome_to or remove the broken tombstone"
	case LoginDisabled:
		return "seat is explicitly disabled", "enable it or choose another seat"
	case LoginMissingDir:
		return "config directory is missing", "restore the directory or tombstone/rehome the seat"
	case LoginNeedsLogin:
		if h.CredentialKind() == CredKindAPIKey {
			return "API key env var $" + h.APIKeyEnv + " is unset or empty",
				"export $" + h.APIKeyEnv + " with this account's Anthropic API key (the registry keeps only the reference, never the secret)"
		}
		return "config directory exists but has no live credentials", "run /login for this CLAUDE_CONFIG_DIR or rehome the seat"
	case LoginIdentityMismatch:
		action := "log out and re-login this CLAUDE_CONFIG_DIR with the browser profile that belongs to this seat"
		if h.ChromeProfile != "" {
			action = "log out and re-login this CLAUDE_CONFIG_DIR with Chrome " + h.ChromeProfile
		}
		return "config directory has credentials for a different account than its seat name", action
	case LoginCooledDown:
		return "account recently hit a usage/rate limit", "wait for the cooldown window to elapse, or clear it once the account is free"
	default:
		return "", ""
	}
}

// cooldownReasonAction renders the seat-facing reason and next action for an
// active cooldown, naming the reset time so an operator knows when the seat
// returns on its own. A probation entry (#3389: window elapsed, canary exit
// gate armed) says so instead of naming an already-past reset as if it were
// still pending.
func cooldownReasonAction(e CooldownEntry) (string, string) {
	kind := "usage limit"
	if e.Kind == CooldownRateLimit {
		kind = "rate limit"
	}
	if e.Probation {
		reason := fmt.Sprintf("%s — window elapsed, in probation awaiting a successful canary round-trip", kind)
		if e.Reason != "" {
			reason = fmt.Sprintf("%s (%s) — window elapsed, in probation awaiting a successful canary round-trip", kind, e.Reason)
		}
		return reason, "leave it; it re-enters the pool once a canary round-trip succeeds, or run `fak accounts cooldown --clear <account>` if the account is already free"
	}
	reason := fmt.Sprintf("%s — resets at %s", kind, e.ResetAt.UTC().Format(time.RFC3339))
	if e.Reason != "" {
		reason = fmt.Sprintf("%s (%s) — resets at %s", kind, e.Reason, e.ResetAt.UTC().Format(time.RFC3339))
	}
	return reason, "leave it; it re-enters the pool at reset, or run `fak accounts cooldown --clear <account>` if the account is already free"
}

func lifecycleString(h Home) string {
	if h.Active() {
		return string(StatusActive)
	}
	return string(h.Status)
}

func (r Registry) rolesFor(name string) []string {
	var roles []string
	for role, seat := range r.Roles {
		if seat == name {
			roles = append(roles, role)
		}
	}
	sort.Strings(roles)
	return roles
}
