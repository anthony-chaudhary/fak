package modelroute

import (
	"os"
	"sort"
	"strings"
)

// account_readiness.go is the observable, secret-safe readiness layer for provider
// account rosters. Validate proves the manifest is well formed; this report proves
// the operator's current environment has the credential references the manifest names.
// It never reads a network endpoint and never returns a secret value.

const AccountReadinessSchema = "fak.modelroute.accounts.v1"

// EnvLookup is the injectable environment read used by Roster.Readiness.
type EnvLookup func(string) (string, bool)

// AccountReadinessStatus is the primary readiness state for one provider account.
type AccountReadinessStatus string

const (
	AccountReady           AccountReadinessStatus = "ready"
	AccountNeedsCredential AccountReadinessStatus = "needs_credential"
)

// CredentialState is the credential-env observation for one account.
type CredentialState string

const (
	CredentialPresent     CredentialState = "present"
	CredentialMissing     CredentialState = "missing"
	CredentialNotRequired CredentialState = "not_required"
)

// AccountReadinessObservation is the credential-safe status record for one provider account.
type AccountReadinessObservation struct {
	ID          string                 `json:"id"`
	Kind        ProviderKind           `json:"kind"`
	Local       bool                   `json:"local"`
	BaseURL     string                 `json:"base_url"`
	CredEnv     string                 `json:"cred_env,omitempty"`
	Credential  CredentialState        `json:"credential"`
	Status      AccountReadinessStatus `json:"status"`
	Default     bool                   `json:"default,omitempty"`
	BoundModels []string               `json:"bound_models,omitempty"`
	Reason      string                 `json:"reason,omitempty"`
	NextAction  string                 `json:"next_action,omitempty"`
}

// AccountReadinessSummary is the rollup over a provider account roster.
type AccountReadinessSummary struct {
	Total           int            `json:"total"`
	Ready           int            `json:"ready"`
	NeedsCredential int            `json:"needs_credential"`
	Local           int            `json:"local"`
	Remote          int            `json:"remote"`
	ByStatus        map[string]int `json:"by_status"`
}

// AccountReadinessReport is the machine-readable provider-account readiness surface.
type AccountReadinessReport struct {
	Schema  string                        `json:"schema"`
	Summary AccountReadinessSummary       `json:"summary"`
	Rows    []AccountReadinessObservation `json:"accounts"`
}

// Readiness reports whether each provider account's credential reference is usable in the
// current environment. Remote accounts need a non-empty env var; local accounts need none.
func (r Roster) Readiness(lookup EnvLookup) AccountReadinessReport {
	if lookup == nil {
		lookup = os.LookupEnv
	}
	byAccount := r.boundModelsByAccount()
	report := AccountReadinessReport{
		Schema: AccountReadinessSchema,
		Summary: AccountReadinessSummary{
			ByStatus: map[string]int{},
		},
	}
	for _, a := range r.Accounts {
		obs := accountReadinessObservation(a, r.Default == a.ID, byAccount[a.ID], lookup)
		report.Rows = append(report.Rows, obs)
		report.Summary.Total++
		report.Summary.ByStatus[string(obs.Status)]++
		if obs.Status == AccountReady {
			report.Summary.Ready++
		}
		if obs.Status == AccountNeedsCredential {
			report.Summary.NeedsCredential++
		}
		if obs.Local {
			report.Summary.Local++
		} else {
			report.Summary.Remote++
		}
	}
	return report
}

func accountReadinessObservation(a Account, isDefault bool, bound []string, lookup EnvLookup) AccountReadinessObservation {
	base := a.BaseURL
	if base == "" {
		base = KindBaseURL(a.Kind)
	}
	cred := credentialState(a, lookup)
	status := AccountReady
	reason, action := "account is configured", ""
	if cred == CredentialMissing {
		status = AccountNeedsCredential
		reason = "remote account credential env var is unset or empty"
		if a.CredEnv == "" {
			action = "add a cred_env name for this remote account"
		} else {
			action = "set " + a.CredEnv + " before dispatching this account"
		}
	}
	return AccountReadinessObservation{
		ID:          a.ID,
		Kind:        a.Kind,
		Local:       !remoteKind(a.Kind),
		BaseURL:     base,
		CredEnv:     a.CredEnv,
		Credential:  cred,
		Status:      status,
		Default:     isDefault,
		BoundModels: append([]string(nil), bound...),
		Reason:      reason,
		NextAction:  action,
	}
}

func credentialState(a Account, lookup EnvLookup) CredentialState {
	if !remoteKind(a.Kind) {
		return CredentialNotRequired
	}
	if a.CredEnv == "" {
		return CredentialMissing
	}
	v, ok := lookup(a.CredEnv)
	if !ok || strings.TrimSpace(v) == "" {
		return CredentialMissing
	}
	return CredentialPresent
}

func (r Roster) boundModelsByAccount() map[string][]string {
	out := map[string][]string{}
	for _, b := range r.Bindings {
		out[b.Account] = append(out[b.Account], b.Model)
	}
	for k := range out {
		sort.Strings(out[k])
	}
	return out
}
