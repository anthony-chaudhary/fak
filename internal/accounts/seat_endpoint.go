package accounts

import (
	"fmt"
	"sort"
	"strings"
)

// Third-party endpoint seats — the BaseURL/ExtraEnv half of Home.
//
// A seat used to be describable by exactly one variable: CLAUDE_CONFIG_DIR. That is enough
// for a first-party subscription or API-key seat, where the endpoint is implied and the
// credential is the only thing that varies. It is NOT enough for a seat that fronts a
// third-party Anthropic-compatible gateway: the agent binary reads that vendor's endpoint,
// model id, required headers, and client bootstrap toggles from its ENVIRONMENT at startup,
// and none of those have anywhere to live in the registry. This file is that seam, plus the
// enforcement that keeps a credential from riding along in it.

// envOverlayReservedKeys are variables an ExtraEnv may not set because fak itself owns
// them on the launch env. Letting a seat override one would not customize the launch, it
// would break the mechanism that makes the launch a SEAT launch at all.
var envOverlayReservedKeys = map[string]string{
	// CLAUDE_CONFIG_DIR *is* the seat. A seat that could rewrite it would run under another
	// seat's credentials and history while every log line still named this one.
	"CLAUDE_CONFIG_DIR": "the seat's own config dir; fak sets it from Home.Dir",
	// ANTHROPIC_BASE_URL has a dedicated field so the guarded-launch refusal can SEE it.
	// Hidden in ExtraEnv it would sail past that check and be silently overridden by guard's
	// own loopback base URL, which is the exact failure the field exists to make loud.
	"ANTHROPIC_BASE_URL": "the seat's endpoint; use the base_url field so a guarded launch can refuse",
}

// credentialWords are the substrings that make an env var NAME credential-shaped. The
// registry is plaintext and `fak accounts list --json` prints it, so a value under one of
// these names would leak into ordinary operator output. Deliberately broad: a false refusal
// costs an operator one rename, while a false accept publishes a secret.
var credentialWords = []string{
	"TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL", "PRIVATE",
	"APIKEY", "API_KEY", "AUTH", "COOKIE", "SESSION_ID", "SIGNING",
}

// looksLikeCredentialName reports whether an env var name is one a secret would plausibly
// be stored under. Matching is on the UPPER-CASED name so a lowercase spelling cannot slip
// through, and on substrings so ANTHROPIC_AUTH_TOKEN and VENDOR_PAT_SECRET both hit.
func looksLikeCredentialName(key string) bool {
	up := strings.ToUpper(key)
	for _, w := range credentialWords {
		if strings.Contains(up, w) {
			return true
		}
	}
	// "KEY" alone is too broad to blanket-ban (a KEYRING_PATH or KEYMAP is not a secret), so
	// it counts only as a trailing component — the shape an actual key variable takes.
	return strings.HasSuffix(up, "_KEY") || up == "KEY"
}

// ValidateExtraEnv checks a seat's ExtraEnv against the two invariants that make the field
// safe to have added: it may not hold a credential, and it may not seize a variable fak
// owns. It returns the FIRST violation as an error naming the key and the reason, so a bad
// registry fails loud at launch instead of producing a subtly mis-routed agent.
//
// A nil/empty map is valid — that is every seat written before the field existed.
//
// Known limit, stated rather than papered over: this checks the NAME, not the value. A
// secret pasted under an innocuous name (most plausibly inside ANTHROPIC_CUSTOM_HEADERS,
// which legitimately carries arbitrary header text) still lands in the registry. Enforcing
// that would need a value-shape heuristic that either misses real secrets or refuses real
// header values, so the contract is: names are enforced, values are the operator's
// responsibility, and the credential itself belongs in the launching environment under the
// name APIKeyEnv references.
func ValidateExtraEnv(extra map[string]string) error {
	for _, key := range sortedKeys(extra) {
		if key == "" {
			return fmt.Errorf("accounts: extra_env has an empty variable name")
		}
		if strings.ContainsAny(key, "=\x00") {
			return fmt.Errorf("accounts: extra_env name %q contains %q or a NUL; it cannot be passed to a process", key, "=")
		}
		if why, bad := envOverlayReservedKeys[key]; bad {
			return fmt.Errorf("accounts: extra_env may not set %s — %s", key, why)
		}
		if looksLikeCredentialName(key) {
			return fmt.Errorf("accounts: extra_env name %s is credential-shaped; the registry is plaintext and `fak accounts list --json` prints it. "+
				"Export the value in the launching environment instead and name the variable in api_key_env", key)
		}
	}
	return nil
}

// EnvOverlay renders the seat's endpoint and extra environment as KEY=VALUE entries to
// append to a launch environment, sorted by key so a launch plan and any broker audit of it
// are byte-stable across runs. It returns nil for a first-party seat that carries neither,
// which keeps the historical launch env exactly as it was.
//
// The caller appends this AFTER the ambient environment (so a seat overrides the shell it
// was launched from — the point of naming a seat) and BEFORE CLAUDE_CONFIG_DIR (so the
// fak-owned variable stays authoritative even if validation is ever bypassed).
func (h Home) EnvOverlay() []string {
	if h.BaseURL == "" && len(h.ExtraEnv) == 0 {
		return nil
	}
	out := make([]string, 0, len(h.ExtraEnv)+1)
	if h.BaseURL != "" {
		out = append(out, "ANTHROPIC_BASE_URL="+h.BaseURL)
	}
	for _, k := range sortedKeys(h.ExtraEnv) {
		out = append(out, k+"="+h.ExtraEnv[k])
	}
	return out
}

// EnvOverlayKeys lists just the variable NAMES this seat overlays, sorted. The launch plan
// prints these so an operator can see what the seat changed without the values — which is
// what makes the plan safe to paste into an issue even when a value is sensitive.
func (h Home) EnvOverlayKeys() []string {
	names := make([]string, 0, len(h.ExtraEnv)+1)
	if h.BaseURL != "" {
		names = append(names, "ANTHROPIC_BASE_URL")
	}
	names = append(names, sortedKeys(h.ExtraEnv)...)
	sort.Strings(names)
	return names
}

// ThirdParty reports whether this seat fronts a third-party Anthropic-compatible endpoint
// rather than first-party api.anthropic.com. It is the single predicate the launch path
// keys its guard refusal off, so "what makes a seat third-party" has one definition.
func (h Home) ThirdParty() bool { return h.BaseURL != "" }

func sortedKeys(m map[string]string) []string {
	if len(m) == 0 {
		return nil
	}
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}
