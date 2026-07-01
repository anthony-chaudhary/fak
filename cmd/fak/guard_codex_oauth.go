package main

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// guard_codex_oauth.go — the Codex ChatGPT-SUBSCRIPTION credential resolver: the OpenAI
// twin of resolveAnthropicOAuthToken (guard.go). It reads the OAuth access token AND the
// ChatGPT account id that `codex login` writes to <codex-home>/auth.json, so a future
// `fak guard -- codex` can hold the SUBSCRIPTION credential upstream the same way the
// Claude path holds the Pro/Max OAuth token — instead of requiring an OPENAI_API_KEY
// (API billing, the only Codex auth `fak guard` wires today).
//
// HONEST FENCE — why this file is the RESOLVER only, not the whole subscription wiring.
// The Codex subscription path differs from Claude in two ways the resolver alone does
// not close, so they are named follow-ons (see the plan doc,
// docs/notes/CODEX-GUARD-SUBSCRIPTION-OAUTH-2026-06-30.md):
//
//  1. DIFFERENT UPSTREAM. Claude's OAuth token goes to the SAME host (api.anthropic.com)
//     with only a different auth scheme (Bearer + the oauth beta), which the anthropic
//     adapter already picks by token shape (IsAnthropicOAuthToken). Codex's subscription
//     token does NOT go to api.openai.com — it goes to the ChatGPT backend
//     (https://chatgpt.com/backend-api/codex) on the Responses wire. So the guard's
//     resolved base URL must repoint there when this credential is held.
//  2. EXTRA REQUIRED HEADER. The ChatGPT backend needs `ChatGPT-Account-Id: <id>`
//     alongside the bearer; drop it and it 401/403s. The current Responses adapter
//     (internal/agent/adapters.go openAIResponsesAdapter.Headers) emits only
//     `Authorization: Bearer` and has no seam for a per-request extra header — that seam
//     is a separate, reviewed increment.
//
// This resolver is the FOUNDATION both follow-ons build on: it extracts the token AND the
// account id as a MATCHED PAIR from the one file that holds both, so a later wiring change
// can never send a mismatched account id (a documented 401 cause). It is pure + read-only
// and NOTHING calls it yet, so shipping it changes no live path — exactly how
// codexMemoryBackend shipped ahead of its one-line dispatch wiring (memq_codex.go).

// codexAuthFileName is the credential file `codex login` writes under the Codex home
// (CODEX_HOME, else ~/.codex). Matches the file `codex login` maintains and refreshes.
const codexAuthFileName = "auth.json"

// codexSubscriptionCredential is the resolved Codex ChatGPT-subscription credential: the
// OAuth access token `codex login` holds and the ChatGPT account id the backend requires
// beside it. Both are read from the SAME auth.json so they are guaranteed a matched pair —
// splitting them across sources risks a token/account-id mismatch the backend rejects.
type codexSubscriptionCredential struct {
	AccessToken string
	AccountID   string
	AuthMode    string // "chatgpt" for a subscription login; "apikey"/"" otherwise
	Source      string // the auth.json path the credential was read from
}

// codexAuthDoc mirrors the subset of <codex-home>/auth.json fak reads. The account id has
// moved between the top level and the tokens object across Codex versions, and is always
// also encoded in the id_token JWT, so all three locations are decoded (codexAccountID).
// OPENAI_API_KEY is a POINTER so a JSON null (the ChatGPT-mode value) is distinguishable
// from an empty string, letting codexAuthMode tell subscription mode from API-key mode.
type codexAuthDoc struct {
	OpenAIAPIKey *string `json:"OPENAI_API_KEY"`
	AuthMode     string  `json:"auth_mode"`
	AccountID    string  `json:"account_id"` // older/top-level placement
	Tokens       struct {
		IDToken      string `json:"id_token"`
		AccessToken  string `json:"access_token"`
		RefreshToken string `json:"refresh_token"`
		AccountID    string `json:"account_id"` // codex-rs placement
	} `json:"tokens"`
	LastRefresh string `json:"last_refresh"`
}

// resolveCodexSubscriptionCredential resolves the Codex home (flag > CODEX_HOME > ~/.codex,
// via resolveCodexHome with optIn=true — guard wrapping codex IS the opt-in) and reads the
// subscription credential from <home>/auth.json. It returns an error that names where it
// looked and the fix (`codex login` / OPENAI_API_KEY) so the caller can fail loud or fall
// back to the API-billing path, mirroring resolveAnthropicOAuthToken's error contract.
func resolveCodexSubscriptionCredential(codexHomeFlag string) (codexSubscriptionCredential, error) {
	home, _ := resolveCodexHome(codexHomeFlag, true)
	if strings.TrimSpace(home) == "" {
		return codexSubscriptionCredential{}, fmt.Errorf(
			"could not resolve a Codex home (no --codex-home, no CODEX_HOME, no user home dir) to read %s from — run `codex login`, or export OPENAI_API_KEY for API billing",
			codexAuthFileName)
	}
	return readCodexSubscriptionCredential(filepath.Join(home, codexAuthFileName))
}

// readCodexSubscriptionCredential reads and parses a Codex auth.json at an explicit path.
// It is split from resolveCodexSubscriptionCredential so the file I/O is the only impure
// step and parseCodexSubscriptionCredential stays a pure, unit-testable core.
func readCodexSubscriptionCredential(path string) (codexSubscriptionCredential, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return codexSubscriptionCredential{}, fmt.Errorf(
				"no Codex login at %s — run `codex login` for a ChatGPT subscription, or export OPENAI_API_KEY for API billing", path)
		}
		return codexSubscriptionCredential{}, fmt.Errorf("read %s: %w", path, err)
	}
	return parseCodexSubscriptionCredential(b, path)
}

// parseCodexSubscriptionCredential is the pure core: it decodes an auth.json body and
// returns the matched token+account-id pair, or an error when no OAuth access token is
// present (an API-key login, or an empty/partial file) — in which case AuthMode is still
// reported so the caller can log why the subscription path did not apply.
func parseCodexSubscriptionCredential(raw []byte, source string) (codexSubscriptionCredential, error) {
	var doc codexAuthDoc
	if err := json.Unmarshal(raw, &doc); err != nil {
		return codexSubscriptionCredential{}, fmt.Errorf("parse %s: %w", source, err)
	}
	mode := codexAuthMode(doc)
	tok := strings.TrimSpace(doc.Tokens.AccessToken)
	if tok == "" {
		// No OAuth access token: an API-key login (auth_mode=apikey) or an empty/partial
		// file. The subscription path does not apply; surface the mode so the caller can
		// fall back to API billing with an accurate reason rather than an opaque miss.
		return codexSubscriptionCredential{AuthMode: mode, Source: source}, fmt.Errorf(
			"no Codex ChatGPT-subscription token in %s (auth_mode=%q) — run `codex login`, or export OPENAI_API_KEY for API billing", source, mode)
	}
	return codexSubscriptionCredential{
		AccessToken: tok,
		AccountID:   codexAccountID(doc),
		AuthMode:    mode,
		Source:      source,
	}, nil
}

// codexAuthMode reports the login mode. It honors an explicit auth_mode, else infers:
// a present access token means a ChatGPT subscription login (older auth.json had no
// auth_mode field), a present OPENAI_API_KEY means API-key mode, otherwise unknown ("").
func codexAuthMode(doc codexAuthDoc) string {
	if m := strings.TrimSpace(doc.AuthMode); m != "" {
		return m
	}
	if strings.TrimSpace(doc.Tokens.AccessToken) != "" {
		return "chatgpt"
	}
	if doc.OpenAIAPIKey != nil && strings.TrimSpace(*doc.OpenAIAPIKey) != "" {
		return "apikey"
	}
	return ""
}

// codexAccountID resolves the ChatGPT account id the backend requires beside the bearer,
// from — in precedence order — tokens.account_id (the codex-rs placement), the top-level
// account_id (older placement), then the id_token JWT claim. The multi-source lookup is
// deliberate: the id placement has moved across Codex versions, but it is always derivable
// from the id_token, so this stays correct without pinning a single layout.
func codexAccountID(doc codexAuthDoc) string {
	if a := strings.TrimSpace(doc.Tokens.AccountID); a != "" {
		return a
	}
	if a := strings.TrimSpace(doc.AccountID); a != "" {
		return a
	}
	return codexAccountIDFromIDToken(doc.Tokens.IDToken)
}

// codexAccountIDFromIDToken extracts the ChatGPT account id from the OIDC id_token's JWT
// payload claim `https://api.openai.com/auth` -> `chatgpt_account_id`. It decodes the
// payload segment WITHOUT verifying the signature: fak reads the account id only as a
// routing header, never as a trust decision — the access_token is the real credential, and
// the backend re-validates both. Returns "" on any malformed token so a hand-edited or
// truncated file degrades to "no account id" rather than crashing the resolver.
func codexAccountIDFromIDToken(idToken string) string {
	idToken = strings.TrimSpace(idToken)
	if idToken == "" {
		return ""
	}
	parts := strings.Split(idToken, ".")
	if len(parts) != 3 {
		return "" // not a well-formed header.payload.signature JWT
	}
	payload, err := base64.RawURLEncoding.DecodeString(parts[1])
	if err != nil {
		// Tolerate an encoder that emitted padding (non-standard for JWT, but seen).
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
