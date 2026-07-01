package harnessprofile

import (
	"path/filepath"
	"strings"
)

// harnessprofile is the one place "which coding harness is this, and how is it
// wired / credentialed / rotated" lives as DATA instead of a spray of switch arms
// across cmd/fak/guard_*.go. Today `fak guard` decides the upstream wire in
// guardDetectProvider, the repoint mechanism in `if guardIsCodex` / `case "claude"`,
// the credential in per-harness resolvers, and the rotation home in a Claude-only
// account model. Each of those is a different switch keyed on the same fact — the
// harness's identity. A HarnessProfile carries that fact ONCE, so guard reads a
// descriptor instead of re-deciding per subsystem, and a new harness becomes a
// registry entry (C6: a config entry) rather than an edit to every switch.
//
// This leaf is PURE: data + a pure Lookup, no I/O and no import of cmd/ or
// internal/accounts. It REFERENCES the shipped wires (internal/agent's Provider
// adapters) by name via Wire — it does not reimplement a wire — and it DECLARES the
// rotation-identity reader by kind (Identity*), leaving the file-reading to the
// account model that consumes this descriptor (C4). See the design note
// docs/notes/UNIVERSAL-HARNESS-PROFILES-2026-07-01.md for the two-plane split (wire —
// already abstracted; harness-as-process — this package) and the honest fences.

// Wire is the upstream provider a harness's traffic is proxied on. It is the same
// closed vocabulary `fak guard --provider` and the gateway already accept; a profile
// REFERENCES a shipped internal/agent adapter by this string, it never defines a new
// wire. The values match guardDetectProvider's returns exactly, so C2 can delegate.
type Wire string

const (
	// WireAnthropic is the Anthropic Messages wire (Claude Code, the Anthropic SDKs).
	WireAnthropic Wire = "anthropic"
	// WireOpenAI is the OpenAI chat-completions wire (OpenCode, Aider, the OpenAI SDK).
	WireOpenAI Wire = "openai"
	// WireOpenAIResponses is the OpenAI Responses wire the current Codex CLI prefers
	// (Chat Completions is deprecated for future removal, so Codex autodetects here).
	WireOpenAIResponses Wire = "openai-responses"
)

// RepointMechanism names HOW guard points a wrapped child at the in-process gateway.
// It is a CLOSED set of the three mechanisms that exist in guard today; a profile
// SELECTS from it by data instead of guard choosing by `if guardIsCodex` /
// `case "claude"`. The implementations are untouched — the profile only picks them.
type RepointMechanism string

const (
	// RepointEnv injects the wire's conventional base-URL env var(s): ANTHROPIC_BASE_URL
	// for the Anthropic wire, OPENAI_BASE_URL + OPENAI_API_BASE for the OpenAI wires
	// (guardInjectedEnv). It fires for every profile — the always-on repoint.
	RepointEnv RepointMechanism = "env"
	// RepointCLIConfig prepends Codex's `-c model_providers.fak.*` provider overrides
	// (installGuardCodexConfig). Codex does not reliably honor OPENAI_BASE_URL for a
	// custom upstream, so this is its real repoint.
	RepointCLIConfig RepointMechanism = "cli-config"
	// RepointSettingsFile writes Claude's `--settings` hooks (PreCompact/Stop) and the
	// `--mcp-config` self-query registration (guard_precompact.go / guard_stophook.go /
	// guard_mcp.go).
	RepointSettingsFile RepointMechanism = "settings-file"
)

// Valid reports whether m is one of the closed RepointMechanism values. C6 (config
// overrides) uses this to reject an unknown mechanism at load rather than silently
// dropping it.
func (m RepointMechanism) Valid() bool {
	switch m {
	case RepointEnv, RepointCLIConfig, RepointSettingsFile:
		return true
	default:
		return false
	}
}

// Valid reports whether w is one of the closed Wire values. C6 uses this to reject an
// unknown wire at config-load time — a user may reference a shipped adapter, never
// declare a brand-new upstream protocol in config.
func (w Wire) Valid() bool {
	switch w {
	case WireAnthropic, WireOpenAI, WireOpenAIResponses:
		return true
	default:
		return false
	}
}

// CredentialKind names the SHAPE of a harness account's live credential, so the
// account model (C4) knows which reader to run. It generalizes the two resolvers guard
// ships today: resolveAnthropicOAuthToken (an OAuth JSON file) and the OPENAI_API_KEY
// env-key path used by the plain OpenAI-wire agents.
type CredentialKind string

const (
	// CredentialEnvKey is a bearer token read from an environment variable (the OpenAI
	// SDK convention, OPENAI_API_KEY). No file, a single implicit account bucket.
	CredentialEnvKey CredentialKind = "env-key"
	// CredentialOAuthFile is an OAuth credential JSON living under the config home
	// (~/.claude/.credentials.json, ~/.codex/auth.json). Its token — and, for Codex, its
	// account id — is what the rotation identity reader extracts.
	CredentialOAuthFile CredentialKind = "oauth-file"
)

// CredentialSource declares where a harness account's live credential lives and its
// shape. For CredentialEnvKey only EnvKey is meaningful; for CredentialOAuthFile only
// File (a path relative to the resolved config home) is.
type CredentialSource struct {
	Kind   CredentialKind
	EnvKey string // CredentialEnvKey: the env var holding the bearer (e.g. OPENAI_API_KEY).
	File   string // CredentialOAuthFile: the credential file, relative to the config home.
}

// IdentityKind names the rotation-identity READER a profile uses to derive its
// rate-limit bucket key for dedup/rotation. It is a DECLARATION, not a function: the
// leaf stays pure and the account model (C4) switches on this kind to run the actual
// file read. It generalizes accounts.Identity.AccountKey() — today only the Claude
// reader exists.
type IdentityKind string

const (
	// IdentityNone is a profile with no rotation-identity reader (declared-but-thin):
	// detection + repoint work, but it does not yet enter the rotation pool.
	IdentityNone IdentityKind = ""
	// IdentityClaude reads ~/.claude/.claude.json + .credentials.json → a uuid:/tok:
	// bucket (the existing accounts.DeriveIdentity path).
	IdentityClaude IdentityKind = "claude"
	// IdentityCodex reads ~/.codex/auth.json → its chatgpt_account_id bucket (C4).
	IdentityCodex IdentityKind = "codex"
	// IdentityEnvKey buckets by the env-key value — a single implicit account for an
	// OpenAI-compatible harness that has no per-account config home.
	IdentityEnvKey IdentityKind = "env-key"
)

// HarnessProfile is the declarative descriptor for one coding harness: the single fact
// (its identity) projected onto every axis guard needs — detect, wire, repoint,
// credential, config home, rotation identity. Built-ins are shipped in the registry;
// C6 lets a user declare more in config. It is pure data.
type HarnessProfile struct {
	// Name is the canonical profile id, used in banners and as a rotation identity label.
	Name string
	// Names are the executable base names (lowercased, launcher-suffix stripped) that
	// select this profile — the detect axis, generalizing guardDetectProvider's cases.
	Names []string
	// Wire is the upstream provider the harness's traffic is proxied on (the referenced
	// internal/agent adapter). DefaultBaseURL is that wire's public API base.
	Wire           Wire
	DefaultBaseURL string
	// Repoint is the ordered set of mechanisms guard applies to point the child at the
	// gateway. A profile may declare more than one (Claude = env + settings-file); every
	// profile declares env, the always-on repoint.
	Repoint []RepointMechanism
	// Credential declares where this harness's account credential lives + its shape.
	Credential CredentialSource
	// ConfigHomeGlob is the config-home directory convention, a glob RELATIVE TO the
	// user's home dir (".claude*", ".codex*"). Empty means the harness has no per-account
	// config home (env-key-only), so it is not rotatable by home.
	ConfigHomeGlob string
	// Identity names the rotation-identity reader kind (see IdentityKind). IdentityNone
	// marks a declared-but-thin profile that does not yet enter the rotation pool.
	Identity IdentityKind
}

// Recognized reports whether the profile was matched from a known name (as opposed to
// the anthropic default the caller substitutes for an unrecognized agent).
func (p HarnessProfile) Recognized() bool { return p.Name != "" }

// HasRepoint reports whether the profile declares mechanism m, so a repoint dispatcher
// (C3) can ask "does this harness take settings-file?" by data rather than by
// `case "claude"`.
func (p HarnessProfile) HasRepoint(m RepointMechanism) bool {
	for _, r := range p.Repoint {
		if r == m {
			return true
		}
	}
	return false
}

// builtins is the registry that ENCODES today's guard switch tables as data. The order
// is stable so Profiles (C6's dump) and any registry walk are deterministic. Adding a
// harness here (or, in C6, in config) is the whole change — no new switch arm.
var builtins = []HarnessProfile{
	{
		Name:           "claude",
		Names:          []string{"claude", "claude-code"},
		Wire:           WireAnthropic,
		DefaultBaseURL: "https://api.anthropic.com",
		// Claude reads ANTHROPIC_BASE_URL (env) AND takes the --settings hooks +
		// --mcp-config (settings-file). Both fire today; both are declared here.
		Repoint:        []RepointMechanism{RepointEnv, RepointSettingsFile},
		Credential:     CredentialSource{Kind: CredentialOAuthFile, File: ".credentials.json"},
		ConfigHomeGlob: ".claude*",
		Identity:       IdentityClaude,
	},
	{
		Name:           "codex",
		Names:          []string{"codex"},
		Wire:           WireOpenAIResponses,
		DefaultBaseURL: "https://api.openai.com/v1",
		// guardInjectedEnv fires for every provider (env), but Codex ignores
		// OPENAI_BASE_URL for a custom upstream, so its real repoint is the `-c`
		// provider overrides (cli-config). Both are declared to match today's argv.
		Repoint:        []RepointMechanism{RepointEnv, RepointCLIConfig},
		Credential:     CredentialSource{Kind: CredentialOAuthFile, File: "auth.json"},
		ConfigHomeGlob: ".codex*",
		Identity:       IdentityCodex,
	},
	{
		Name:           "openai-generic",
		Names:          []string{"opencode", "aider", "hermes"},
		Wire:           WireOpenAI,
		DefaultBaseURL: "https://api.openai.com/v1",
		// These read OPENAI_BASE_URL / OPENAI_API_BASE and nothing else — env-only
		// repoint, an env-key credential, no per-account config home (thin rotation).
		Repoint:        []RepointMechanism{RepointEnv},
		Credential:     CredentialSource{Kind: CredentialEnvKey, EnvKey: "OPENAI_API_KEY"},
		ConfigHomeGlob: "",
		Identity:       IdentityEnvKey,
	},
}

// Lookup resolves a wrapped-agent command to its HarnessProfile, matching on the
// executable's normalized base name. It generalizes guardAgentBaseName + the
// guardDetectProvider switch: the SAME name normalization (directory stripped handling
// both path separators regardless of host OS, plus a Windows launcher suffix removed),
// then a registry scan. The bool is false for an unrecognized agent — the caller keeps
// the historical anthropic fallback (resolveGuardProvider owns that), so guard behavior
// is unchanged. The zero HarnessProfile is returned when not found; its Recognized()
// reports false.
func Lookup(agentCommand string) (HarnessProfile, bool) {
	name := baseName(agentCommand)
	if name == "" {
		return HarnessProfile{}, false
	}
	for _, p := range builtins {
		for _, n := range p.Names {
			if n == name {
				return p, true
			}
		}
	}
	return HarnessProfile{}, false
}

// BaseURLForWire maps a Wire to its public API base URL — the wire→URL half of the old
// guardDefaultBaseURL switch, so C2 can delegate to a single source of truth. An
// unknown wire returns "" so the caller can require an explicit --base-url instead of
// guessing. The Anthropic host carries NO /v1 (its client appends the Messages path);
// both OpenAI wires share the /v1 root (chat appends /chat/completions, Responses
// appends /responses).
func BaseURLForWire(w Wire) string {
	switch w {
	case WireAnthropic:
		return "https://api.anthropic.com"
	case WireOpenAI, WireOpenAIResponses:
		return "https://api.openai.com/v1"
	default:
		return ""
	}
}

// Profiles returns a copy of the built-in registry in stable order, for the
// dump/inspect surface (C6's --dump-harness-profiles) and any read-only walk. The
// returned slice is a fresh copy so a caller cannot mutate the registry.
func Profiles() []HarnessProfile {
	out := make([]HarnessProfile, len(builtins))
	copy(out, builtins)
	return out
}

// baseName normalizes a wrapped-agent command to its lowercased executable base name:
// any directory component is stripped handling BOTH path separators regardless of the
// host OS (filepath.Base is host-specific — on a Linux runner it will not split a
// Windows backslash path, so LastIndexAny over `/` and `\` keeps the match
// cross-platform), and a Windows .exe/.cmd/.bat/.ps1/.com launcher suffix is removed.
// It is a port of cmd/fak's guardAgentBaseName so Lookup normalizes identically.
func baseName(command string) string {
	base := strings.ToLower(strings.TrimSpace(command))
	if i := strings.LastIndexAny(base, `/\`); i >= 0 {
		base = base[i+1:]
	}
	switch filepath.Ext(base) {
	case ".exe", ".cmd", ".bat", ".ps1", ".com":
		base = strings.TrimSuffix(base, filepath.Ext(base))
	}
	return base
}
