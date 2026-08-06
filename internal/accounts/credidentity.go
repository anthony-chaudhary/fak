package accounts

// Credential-identity reconcile — the fix for the provenance bug where a seat's recorded
// identity is derived from on-disk METADATA (.claude.json oauthAccount) instead of from the
// account the seat's live CREDENTIAL actually serves. After a `/login` into a shared/default
// config dir rewrites only .credentials.json, the two disagree: the metadata still names the
// PREVIOUS account, so DeriveIdentity mislabels the seat and the roster silently attributes one
// account's seat to another (and burns the wrong rate-limit bucket).
//
// DeriveIdentity stays pure and disk-only (it runs on every launcher/roster tick, so it must
// never touch the network). This file adds the OPT-IN reconcile: given an IdentityProber, read
// the credential's true identity from the OAuth profile endpoint and prefer it over stale disk
// metadata. Enrollment (`enroll-current`) uses it to write a CORRECTED .claude.json so all later
// disk reads are right; `status --probe` uses it to FLAG a live disagreement.

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
)

// CredentialAccessToken returns the OAuth access token a dir's live session credential
// (.credentials.json → claudeAiOauth.accessToken) currently serves, or "" when absent/empty.
// This is the token whose account is GROUND TRUTH for the seat — the identity .claude.json's
// oauthAccount only *claims*, and gets wrong after a re-login rewrote the credential but not
// the metadata. It is deliberately narrow to the auto-refreshing session credential: a bare
// .oauth-token setup credential is handled separately by the enroll flow's token probe.
func CredentialAccessToken(dir string) string {
	access, _ := credentialTokens(dir)
	return access
}

// credentialTokens decodes a dir's .credentials.json claudeAiOauth block ONCE and returns both
// tokens, whitespace-trimmed; either is "" when absent, and both are "" when the dir carries no
// readable credential. The two tokens answer different questions — the ACCESS token identifies the
// account (CredentialAccessToken, for the identity probe) while the REFRESH token identifies the
// token FAMILY (RefreshFamilyID, for the sharing hazard in credfamily.go) — but they are one read
// of one file, so the decode lives here once rather than being cloned per caller.
func credentialTokens(dir string) (accessToken, refreshToken string) {
	if dir == "" {
		return "", ""
	}
	b, err := os.ReadFile(filepath.Join(dir, ".credentials.json"))
	if err != nil {
		return "", ""
	}
	var doc struct {
		ClaudeAiOauth struct {
			AccessToken  string `json:"accessToken"`
			RefreshToken string `json:"refreshToken"`
		} `json:"claudeAiOauth"`
	}
	if json.Unmarshal(b, &doc) != nil {
		return "", ""
	}
	return strings.TrimSpace(doc.ClaudeAiOauth.AccessToken), strings.TrimSpace(doc.ClaudeAiOauth.RefreshToken)
}

// LoginWarningIdentityStale is the status/list warning for a seat whose on-disk .claude.json
// identity metadata names a DIFFERENT account than the one its live credential actually serves —
// a silent mislabel that would attribute the seat to (and burn the quota of) the wrong account.
// It is surfaced by `status --probe` (which needs a live oauth/profile probe to know the
// credential's true account); the durable fix is `enroll-current`, which rewrites the metadata to
// the probed truth. It lives here beside the reconcile primitive that produces it — the natural
// home for the credential-identity feature — rather than in the base login vocabulary.
const LoginWarningIdentityStale LoginWarning = "identity_metadata_stale"

// IdentityProber resolves an opaque OAuth/access token to the account it authenticates as.
// It is the injection seam over ProbeToken so callers bind the endpoint + client (real use)
// and tests supply an httptest stand-in. A nil prober disables the live probe (disk-only).
type IdentityProber func(token string) (ProbedIdentity, error)

// CredentialResolution reconciles a dir's on-disk identity metadata (.claude.json oauthAccount,
// via DeriveIdentity) against the account its live credential actually serves (probed).
type CredentialResolution struct {
	// Disk is the identity derived purely from on-disk metadata (DeriveIdentity).
	Disk Identity
	// Credential is the identity the live credential probes as; zero when not probed.
	Credential ProbedIdentity
	// Resolved is the identity to trust/record: the credential's identity wins over disk when
	// a live credential was probed; otherwise it is Disk unchanged.
	Resolved Identity
	// Probed is true when a live credential access token was found and the prober answered.
	Probed bool
	// Stale is true when disk metadata carried a NON-EMPTY identity that DISAGREES with the
	// probed credential — the exact mislabel this reconcile exists to catch. Filling in an
	// empty disk identity from the credential is not "stale".
	Stale bool
	// ProbeErr carries a probe transport/endpoint error (the credential could not prove itself);
	// Resolved falls back to Disk in that case and Probed is false.
	ProbeErr error
}

// ResolveCredentialIdentity derives a dir's disk identity and, when the dir carries a live
// session credential AND a prober is supplied, reconciles it against the credential's true
// identity — with the CREDENTIAL winning any disagreement. It never panics and treats a probe
// failure as non-fatal (Resolved falls back to disk, ProbeErr is set), so a callable path that
// wants disk-only truth simply passes a nil prober. This is the single primitive both the
// enroll-current write path and the status --probe read path consume.
func ResolveCredentialIdentity(dir string, probe IdentityProber) CredentialResolution {
	disk := DeriveIdentity(dir)
	res := CredentialResolution{Disk: disk, Resolved: disk}
	if probe == nil {
		return res
	}
	token := CredentialAccessToken(dir)
	if token == "" {
		return res // no live session credential to probe — disk is all we have
	}
	probed, err := probe(token)
	if err != nil {
		res.ProbeErr = err
		return res
	}
	if probed.Email == "" && probed.AccountUUID == "" {
		return res // probe answered but carried no identity — keep disk
	}
	res.Credential = probed
	res.Probed = true
	res.Stale = identitiesDisagree(disk, probed)
	// The credential is ground truth: overwrite the identity fields, keep the disk-derived
	// existence/creds/token-fingerprint facts (those are about the dir, not the account).
	resolved := disk
	resolved.Email = probed.Email
	resolved.AccountUUID = probed.AccountUUID
	res.Resolved = resolved
	return res
}

// identitiesDisagree reports whether on-disk metadata names a DIFFERENT account than the probed
// credential. It compares on AccountUUID first (the AccountKey ground truth), falling back to a
// case-insensitive email compare when a uuid is absent on either side. An empty disk identity
// never "disagrees" — there is nothing to be stale, only a gap to fill.
func identitiesDisagree(disk Identity, cred ProbedIdentity) bool {
	if disk.Email == "" && disk.AccountUUID == "" {
		return false
	}
	if disk.AccountUUID != "" && cred.AccountUUID != "" {
		return disk.AccountUUID != cred.AccountUUID
	}
	if disk.Email != "" && cred.Email != "" {
		return !strings.EqualFold(strings.TrimSpace(disk.Email), strings.TrimSpace(cred.Email))
	}
	// One side has a uuid and the other only an email (or vice versa): can't prove agreement,
	// but can't prove disagreement either — treat as agreeing to avoid a false stale flag.
	return false
}
