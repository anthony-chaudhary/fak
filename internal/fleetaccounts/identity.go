package fleetaccounts

import (
	"sort"
	"strings"
)

// identityAlias binds a canonical identity key to the field-name spellings it can appear
// under — snake_case registry/throttle rows and camelCase oauthAccount blobs alike.
type identityAlias struct {
	key     string
	aliases []string
}

// identityAliases mirrors fleet_accounts._IDENTITY_ALIASES. token_fp is extracted but,
// like the Python, is NOT one of the keys a match verdict is decided on (identityKeys) —
// it collapses phantom duplicates elsewhere, it does not decide WHO a throttle belongs to.
var identityAliases = []identityAlias{
	{"account_uuid", []string{"account_uuid", "accountUuid"}},
	{"login_email", []string{"login_email", "emailAddress", "email"}},
	{"org_uuid", []string{"org_uuid", "organizationUuid"}},
	{"token_fp", []string{"token_fp", "tokenFP"}},
}

// identityKeys mirrors fleet_accounts._IDENTITY_KEYS: the identity fields a match verdict
// is decided on, in priority order (account_uuid is the re-login discriminator, so it
// leads). token_fp is deliberately excluded — see identityAliases.
var identityKeys = []string{"account_uuid", "login_email", "org_uuid"}

// identityNestedKeys mirrors the nested sources _account_identity_from folds in beyond the
// top-level dict, so an identity carried under an "oauthAccount"/"identity" blob is found.
var identityNestedKeys = []string{"identity", "account_identity", "oauthAccount"}

// ageSentinel mirrors the Python's 10**9 sort key for a session row with no age_min, so an
// ageless row sorts last (least-fresh) exactly as in _throttle_matches_current_identity.
const ageSentinel = 1e9

// accountIdentityFrom is the Go port of fleet_accounts._account_identity_from: pull the
// identity fields out of a throttle / registry / session / probe record, checking the
// top-level dict and its nested identity blobs. First non-empty alias wins; values are
// trimmed and lower-cased so a case/whitespace difference never reads as a mismatch. An
// info carrying no identity field yields nil (Python's empty dict).
func accountIdentityFrom(info map[string]any) map[string]string {
	if info == nil {
		return nil
	}
	sources := []map[string]any{info}
	for _, nk := range identityNestedKeys {
		if nested, ok := info[nk].(map[string]any); ok {
			sources = append(sources, nested)
		}
	}
	out := map[string]string{}
	for _, ka := range identityAliases {
		for _, src := range sources {
			hit := false
			for _, alias := range ka.aliases {
				if v := strings.TrimSpace(asString(src[alias])); v != "" {
					out[ka.key] = strings.ToLower(v)
					hit = true
					break
				}
			}
			if hit {
				break
			}
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

// identityMatch is the Go port of fleet_accounts._identity_match: decide equality on the
// first identity key BOTH sides carry. The second return is the "decided" bit — false
// mirrors Python's None (no shared key), telling the caller to fall through to the next
// candidate rather than treat an undecidable pair as a mismatch.
func identityMatch(left, right map[string]string) (match bool, decided bool) {
	for _, key := range identityKeys {
		l, r := left[key], right[key]
		if l != "" && r != "" {
			return l == r, true
		}
	}
	return false, false
}

// currentConfigIdentity is the Go port of fleet_accounts._current_config_identity: read
// WHO the seat's config dir is actually logged in as (oauthAccount in .claude.json) and
// shape it through the same alias normalizer, so it compares cleanly against a throttle
// stamp. An empty dir or a dir with no login yields nil, exactly like Python's empty-dict
// fallback — nothing to match on, so it never decides a verdict.
func currentConfigIdentity(dir string) map[string]string {
	if strings.TrimSpace(dir) == "" {
		return nil
	}
	id := ReadAccountIdentity(dir)
	return accountIdentityFrom(map[string]any{
		"account_uuid": id.AccountUUID,
		"login_email":  id.LoginEmail,
		"org_uuid":     id.OrgUUID,
	})
}

// registryAccountIdentity is the Go port of fleet_accounts._registry_account_identity: the
// identity stamped on the matching row of the registry's `accounts` list (nil when the
// registry has no such row or the row carries no identity).
func registryAccountIdentity(reg Registry, account string) map[string]string {
	for _, row := range reg.Accounts {
		if asString(row["account"]) == account {
			return accountIdentityFrom(row)
		}
	}
	return nil
}

// sessionIdentity extracts the identity a session row carries. Rows built by LoadRegistry
// keep their raw map for this; a Session constructed directly (tests) has no raw and yields
// nil — the same "no identity here" answer as a row that simply omits the fields.
func sessionIdentity(s Session) map[string]string {
	return accountIdentityFrom(s.raw)
}

// throttleMatchesCurrentIdentity is the Go port of
// fleet_accounts._throttle_matches_current_identity. A carried throttle only holds a seat
// closed while it belongs to the identity the dir is CURRENTLY logged in as: if the
// throttle stamps an identity that provably differs from the seat's current identity — the
// "usage limit that belonged to a DIFFERENT account the dir was logged into before a
// re-login" case — the hold is cleared and the fresh OK probe reopens the seat.
//
// Verdict order matches the Python: probe identity -> current-config identity -> registry
// row -> freshest session rows (ascending age). The FIRST candidate that shares an
// identity key with the throttle stamp decides. A throttle with no stamped identity, or no
// candidate sharing a key, returns true (hold) — fail-closed, so an unknown identity never
// reopens a capped seat. Today's recorders stamp no throttle identity, so this returns
// true and preserves the conservative weekly-cap hold; it is the forward-compatible seam
// that activates the moment a throttle carries WHO it capped.
func throttleMatchesCurrentIdentity(account, dir string, thr map[string]any, reg Registry, acctSessions []Session, probeIdentity map[string]string) bool {
	throttleIdentity := accountIdentityFrom(thr)
	if len(throttleIdentity) == 0 {
		return true
	}
	candidates := []map[string]string{
		probeIdentity,
		currentConfigIdentity(dir),
		registryAccountIdentity(reg, account),
	}
	sorted := make([]Session, len(acctSessions))
	copy(sorted, acctSessions)
	sort.SliceStable(sorted, func(i, j int) bool {
		return sessionAgeSort(sorted[i]) < sessionAgeSort(sorted[j])
	})
	for i := range sorted {
		candidates = append(candidates, sessionIdentity(sorted[i]))
	}
	for _, cand := range candidates {
		if len(cand) == 0 {
			continue
		}
		if verdict, decided := identityMatch(throttleIdentity, cand); decided {
			return verdict
		}
	}
	return true
}

// sessionAgeSort is the age sort key: the row's age_min, or the ageless sentinel so a row
// with no age sorts last (least-fresh), mirroring the Python key function.
func sessionAgeSort(s Session) float64 {
	if age, ok := sessionAge(s); ok {
		return age
	}
	return ageSentinel
}
