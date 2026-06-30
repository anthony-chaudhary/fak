package accounts

import "sort"

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
}

// LoginSummary is the rollup over a LoginReport.
type LoginSummary struct {
	Total            int            `json:"total"`
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
	default:
		return LoginReady
	}
}

// CanServe reports whether this seat is ready to launch directly.
func (h Home) CanServe() bool { return h.LoginStatus() == LoginReady }

// LoginReport folds every home into an observable status record plus a rollup.
// Call Refresh first when current disk state matters.
func (r Registry) LoginReport() LoginReport {
	rec := r.Reconcile()
	report := LoginReport{
		Schema: LoginReportSchema,
		Summary: LoginSummary{
			ByStatus: map[string]int{},
		},
	}
	accounts := map[string]bool{}
	for _, h := range r.Homes {
		obs := r.loginObservation(h, rec[h.Name])
		report.Seats = append(report.Seats, obs)
		report.Summary.Total++
		report.Summary.ByStatus[string(obs.Status)]++
		if obs.CanServe {
			report.Summary.CanServe++
		}
		if obs.Account != "" {
			accounts[obs.Account] = true
		}
		if len(obs.Warnings) > 0 {
			report.Summary.WarningSeats++
		}
	}
	report.Summary.DistinctAccounts = len(accounts)
	return report
}

func (r Registry) loginObservation(h Home, si SeatIdentity) LoginObservation {
	status := h.LoginStatus()
	reason, action := LoginReasonAction(status, h)
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
		return "config directory exists but has no live credentials", "run /login for this CLAUDE_CONFIG_DIR or rehome the seat"
	default:
		return "", ""
	}
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
