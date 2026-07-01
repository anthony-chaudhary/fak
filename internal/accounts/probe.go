package accounts

// OAuth identity probe — resolve an opaque setup/OAuth token to the account it authenticates
// as, by asking Anthropic's OAuth profile endpoint. This is the GROUND-TRUTH identity step
// for enrolling a brand-new account: a sk-ant-oat… token is opaque (you cannot read the email
// out of it), and a freshly-created config dir has no .claude.json yet, so disk derivation
// returns nothing. The probe both (a) yields the email + account UUID to record in the
// registry and (b) proves the credential actually works before we enroll it.

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// DefaultProfileURL is Anthropic's OAuth profile endpoint. Overridable in ProbeToken for tests.
const DefaultProfileURL = "https://api.anthropic.com/api/oauth/profile"

// oauthBeta is the beta header the OAuth profile endpoint expects.
const oauthBeta = "oauth-2025-04-20"

// ProbedIdentity is the subset of the profile response the registry records.
type ProbedIdentity struct {
	Email       string
	AccountUUID string
	FullName    string
}

// IsScopeError reports whether err is the SPECIFIC 403 the OAuth profile endpoint returns
// for a `claude setup-token` credential — a valid, serveable token that simply lacks the
// `user:profile`/`user:office` scope the profile read requires. This is NOT a bad token:
// it serves fine, it just cannot answer "who am I" at the profile endpoint. Enrollment
// treats it as "identity pending" (binds on first interactive login) rather than a failure,
// so the add flow stops emitting a scary warning for the expected case.
func IsScopeError(err error) bool {
	if err == nil {
		return false
	}
	s := err.Error()
	return strings.Contains(s, "status 403") &&
		(strings.Contains(s, "scope requirement") || strings.Contains(s, "user:profile") || strings.Contains(s, "permission_error"))
}

// ProbeToken asks the OAuth profile endpoint who `token` authenticates as. A non-2xx or a
// transport error is returned as an error (the token does not work / is not a real account),
// so a caller can refuse to enroll a credential that cannot prove itself. url defaults to
// DefaultProfileURL when empty; client defaults to a short-timeout client when nil.
func ProbeToken(client *http.Client, url, token string) (ProbedIdentity, error) {
	token = strings.TrimSpace(token)
	if token == "" {
		return ProbedIdentity{}, fmt.Errorf("accounts: probe: empty token")
	}
	if url == "" {
		url = DefaultProfileURL
	}
	if client == nil {
		client = &http.Client{Timeout: 20 * time.Second}
	}
	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return ProbedIdentity{}, fmt.Errorf("accounts: probe: %w", err)
	}
	req.Header.Set("Authorization", "Bearer "+token)
	req.Header.Set("anthropic-beta", oauthBeta)
	resp, err := client.Do(req)
	if err != nil {
		return ProbedIdentity{}, fmt.Errorf("accounts: probe %s: %w", url, err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return ProbedIdentity{}, fmt.Errorf("accounts: probe %s: status %d: %s", url, resp.StatusCode, strings.TrimSpace(string(body)))
	}
	return parseProfile(body)
}

// profileResponse mirrors the OAuth profile JSON we consume (account block only).
type profileResponse struct {
	Account struct {
		UUID         string `json:"uuid"`
		Email        string `json:"email"`
		EmailAddress string `json:"email_address"`
		FullName     string `json:"full_name"`
	} `json:"account"`
}

// parseProfile extracts the identity from a profile response body.
func parseProfile(body []byte) (ProbedIdentity, error) {
	var pr profileResponse
	if err := json.Unmarshal(body, &pr); err != nil {
		return ProbedIdentity{}, fmt.Errorf("accounts: probe: decode profile: %w", err)
	}
	email := pr.Account.Email
	if email == "" {
		email = pr.Account.EmailAddress
	}
	if email == "" && pr.Account.UUID == "" {
		return ProbedIdentity{}, fmt.Errorf("accounts: probe: profile carried no account identity")
	}
	return ProbedIdentity{Email: email, AccountUUID: pr.Account.UUID, FullName: pr.Account.FullName}, nil
}
