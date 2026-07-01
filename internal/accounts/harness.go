package accounts

import (
	"encoding/base64"
	"encoding/json"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/harnessprofile"
)

// harness.go generalizes the account model so a NON-Claude harness's config homes — the
// ~/.codex* homes today — enter the SAME rotation pool the ~/.claude* homes use (C4,
// #1955). The rotation engine (RotationPlan / NextInRotation, rotation.go) already keys
// purely on Identity.AccountKey() + Home.LoginStatus(); it never mentions Claude. So the
// ONLY harness-specific work is DERIVING the Identity: which glob finds the homes
// (HarnessProfile.ConfigHomeGlob), which file proves a live credential, and where the
// rate-limit bucket key comes from (HarnessProfile.Identity). This file adds the codex
// reader beside the existing Claude one (DeriveIdentity) and a profile-dispatched
// discovery, leaving DeriveIdentity / Discover — the Claude path — byte-identical.
//
// Honest coverage (the #1955 fence): claude and codex have a REAL identity reader (a per-
// home file that names the rate-limit account). opencode/aider/hermes are declared-but-thin
// — env-key only, no per-account config home — so they collapse to a single implicit bucket
// and do not rotate by home. HasIdentityReader reports which is which.

// DeriveIdentityForProfile reads the disk truth for one config-home dir using the reader
// the profile declares (profile.Identity). IdentityClaude reproduces DeriveIdentity exactly
// (so the Claude path is unchanged); IdentityCodex reads <dir>/auth.json for the ChatGPT
// account id + credential presence; IdentityEnvKey / IdentityNone have no per-home identity
// file, so they report existence only (no AccountKey bucket → a single implicit bucket,
// thin rotation). It never errors — a missing/unreadable file leaves the field zero, so a
// half-set-up home reads as "exists, no creds".
func DeriveIdentityForProfile(dir string, profile harnessprofile.HarnessProfile) Identity {
	switch profile.Identity {
	case harnessprofile.IdentityClaude:
		return DeriveIdentity(dir)
	case harnessprofile.IdentityCodex:
		return deriveCodexIdentity(dir)
	default:
		return deriveExistsOnlyIdentity(dir)
	}
}

// HasIdentityReader reports whether a profile has a REAL per-home identity reader (claude,
// codex) as opposed to being declared-but-thin (env-key / none). A profile without a reader
// still detects + repoints, but its homes cannot dedup or rotate by account — they share one
// implicit bucket. This is the honest-coverage witness #1955 asks for.
func HasIdentityReader(profile harnessprofile.HarnessProfile) bool {
	switch profile.Identity {
	case harnessprofile.IdentityClaude, harnessprofile.IdentityCodex:
		return true
	default:
		return false
	}
}

// DiscoverProfile globs the profile's config-home convention (ConfigHomeGlob, relative to
// home) and returns a Home per config-home directory with its profile-derived Identity,
// sorted by name for determinism — the harness-agnostic twin of Discover. A profile with no
// config-home glob (env-key-only harnesses) has no per-account homes, so it returns nil. The
// seat name strips the glob's fixed prefix (".codex-" → the seat, ".codex" itself →
// "default"), mirroring Discover's ".claude-" handling, so a codex roster reads like a
// Claude one. Like Discover, it reports what EXISTS; lifecycle is overlaid by the caller.
func DiscoverProfile(home string, profile harnessprofile.HarnessProfile) ([]Home, error) {
	glob := strings.TrimSpace(profile.ConfigHomeGlob)
	if glob == "" {
		return nil, nil // env-key harness: no per-account config homes to rotate.
	}
	// Claude keeps its exact original path (name derivation, isConfigHome marker) so the
	// Claude rotation stays byte-identical whether reached via Discover or DiscoverProfile.
	if profile.Identity == harnessprofile.IdentityClaude {
		return Discover(home)
	}
	matches, err := filepath.Glob(filepath.Join(home, glob))
	if err != nil {
		return nil, err
	}
	prefix := strings.TrimSuffix(glob, "*") // ".codex"
	var homes []Home
	for _, m := range matches {
		fi, err := os.Stat(m)
		if err != nil || !fi.IsDir() || !isConfigHomeForProfile(m, profile) {
			continue
		}
		base := filepath.Base(m)
		name := "default"
		if base != prefix {
			name = strings.TrimPrefix(base, prefix+"-")
		}
		homes = append(homes, Home{Name: name, Dir: m, Identity: DeriveIdentityForProfile(m, profile)})
	}
	sort.Slice(homes, func(i, j int) bool { return homes[i].Name < homes[j].Name })
	return homes, nil
}

// deriveExistsOnlyIdentity reports only whether the dir exists — the identity for a
// declared-but-thin profile (env-key / none) that has no per-home account file. With no
// AccountUUID/TokenFP its AccountKey() is "", so RotationPlan treats it as an unidentifiable
// singleton bucket rather than deduping it against anything.
func deriveExistsOnlyIdentity(dir string) Identity {
	var id Identity
	if dir == "" {
		return id
	}
	if fi, err := os.Stat(dir); err == nil && fi.IsDir() {
		id.Exists = true
	}
	return id
}

// codexAuthFile is the credential file `codex login` writes under a Codex home
// (CODEX_HOME / ~/.codex). It is the codex twin of Claude's .credentials.json — its
// presence with a live token is what makes a codex home LoginReady, and the ChatGPT account
// id inside it is the rate-limit bucket key.
const codexAuthFile = "auth.json"

// deriveCodexIdentity reads <dir>/auth.json for a codex home: the ChatGPT account id becomes
// the AccountUUID (so AccountKey() yields "uuid:<account-id>" and two homes on the same
// ChatGPT account collapse to one bucket, mirroring Claude's uuid dedup), and a present
// credential (a subscription access token or an OPENAI_API_KEY) sets HasCreds so
// LoginStatus() reads Ready. A missing/partial file leaves the home "exists, no creds"
// (LoginNeedsLogin), exactly as a half-set-up Claude home reads.
func deriveCodexIdentity(dir string) Identity {
	id, ok := statConfigHome(dir)
	if !ok {
		return id
	}
	b, err := os.ReadFile(filepath.Join(dir, codexAuthFile))
	if err != nil {
		return id // exists, no creds
	}
	acct, hasCred := parseCodexAuthIdentity(b)
	id.AccountUUID = acct
	id.HasCreds = hasCred
	return id
}

// codexAuthIdentityDoc mirrors the subset of a Codex auth.json needed to bucket + gate a
// home: the account id (which has moved across Codex versions — tokens.account_id, then the
// top-level account_id, then the id_token JWT) and whether any credential is present. It is
// a read-only, deliberately narrow copy of cmd/fak's codexAuthDoc — internal/accounts is a
// foundation leaf and may not import cmd/fak, so the shape is restated here.
type codexAuthIdentityDoc struct {
	OpenAIAPIKey *string `json:"OPENAI_API_KEY"`
	AccountID    string  `json:"account_id"`
	Tokens       struct {
		IDToken     string `json:"id_token"`
		AccessToken string `json:"access_token"`
		AccountID   string `json:"account_id"`
	} `json:"tokens"`
}

// parseCodexAuthIdentity decodes a codex auth.json body and returns the ChatGPT account id
// (the rate-limit bucket) and whether the home holds a live credential. account is resolved
// in the same precedence cmd/fak uses (tokens.account_id → account_id → id_token JWT).
// hasCred is true when either a subscription access token or an OPENAI_API_KEY is present —
// either is enough for the home to serve, so either makes it rotation-ready.
func parseCodexAuthIdentity(raw []byte) (account string, hasCred bool) {
	var doc codexAuthIdentityDoc
	if json.Unmarshal(raw, &doc) != nil {
		return "", false
	}
	account = strings.TrimSpace(doc.Tokens.AccountID)
	if account == "" {
		account = strings.TrimSpace(doc.AccountID)
	}
	if account == "" {
		account = codexAccountIDFromIDToken(doc.Tokens.IDToken)
	}
	hasCred = strings.TrimSpace(doc.Tokens.AccessToken) != "" ||
		(doc.OpenAIAPIKey != nil && strings.TrimSpace(*doc.OpenAIAPIKey) != "")
	return account, hasCred
}

// codexAccountIDFromIDToken extracts chatgpt_account_id from the OIDC id_token JWT payload
// claim `https://api.openai.com/auth`, WITHOUT verifying the signature: the account id is
// used only as a bucket key here, never as a trust decision. Returns "" on any malformed
// token, so a hand-edited or truncated file degrades to "no account id" rather than
// crashing discovery. It mirrors cmd/fak's codexAccountIDFromIDToken.
func codexAccountIDFromIDToken(idToken string) string {
	idToken = strings.TrimSpace(idToken)
	if idToken == "" {
		return ""
	}
	parts := strings.Split(idToken, ".")
	if len(parts) != 3 {
		return ""
	}
	payload, err := base64.RawURLEncoding.DecodeString(parts[1])
	if err != nil {
		p, err2 := base64.URLEncoding.DecodeString(parts[1])
		if err2 != nil {
			return ""
		}
		payload = p
	}
	var claims struct {
		Auth struct {
			ChatGPTAccountID string `json:"chatgpt_account_id"`
		} `json:"https://api.openai.com/auth"`
	}
	if json.Unmarshal(payload, &claims) != nil {
		return ""
	}
	return strings.TrimSpace(claims.Auth.ChatGPTAccountID)
}

// isConfigHomeForProfile reports whether dir is a config home for the given profile,
// generalizing isConfigHome (the Claude .claude.json / projects marker). A codex home is
// marked by its auth.json or config.toml; any other profile falls back to the Claude marker.
func isConfigHomeForProfile(dir string, profile harnessprofile.HarnessProfile) bool {
	if profile.Identity == harnessprofile.IdentityCodex {
		if fi, err := os.Stat(filepath.Join(dir, codexAuthFile)); err == nil && !fi.IsDir() {
			return true
		}
		if fi, err := os.Stat(filepath.Join(dir, "config.toml")); err == nil && !fi.IsDir() {
			return true
		}
		return false
	}
	return isConfigHome(dir)
}
