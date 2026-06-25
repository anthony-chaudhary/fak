package dropin

import (
	"path/filepath"
	"strings"
)

// Ready reports that the generated drop-in wiring skeleton is linked into the build.
func Ready() bool { return true }

// dropin is the canonical, extracted form of the wire-resolution that powers the
// "drop fak in front of the agent you already run" path. The rules here are exactly
// the ones cmd/fak's `fak guard -- <agent>` resolves with (guardDetectProvider /
// resolveGuardProvider / guardEnvVar / guardEnvValue / guardInjectedEnv in
// cmd/fak/guard.go); the two are kept in lockstep by ONE shared test truth table —
// dropin_test.go here and guard_test.go in cmd/fak assert the SAME inputs map to the
// SAME wire, so neither can drift without a red test. cmd/dropindemo reads its
// entry-point gallery from this package, so what the demo SHOWS is what `fak guard`
// does, by construction rather than by a hand-copied table.
//
// The package is deliberately pure: stdlib only, no side effects, no I/O. Every
// function is a total mapping from (agent name, provider flag, gateway URL) to a
// resolved wire, so it is trivially testable and identical on every OS.

// DetectProvider infers the upstream wire from the wrapped agent's command when the
// operator passes no --provider, so naming a known agent (`fak guard -- codex`) Just
// Works without also having to say `--provider openai`. The table lists agents that
// read a base-URL variable guard injects: ANTHROPIC_BASE_URL for the Anthropic wire,
// and OPENAI_BASE_URL plus OPENAI_API_BASE for the OpenAI wire (guard sets both, so
// Aider, which reads OPENAI_API_BASE rather than OPENAI_BASE_URL, connects too). An
// agent that reads neither is left to an explicit --provider/--env on purpose, rather
// than autodetected into a base URL it ignores. Matching is on the executable's base
// name, lowercased, with any directory and a Windows .exe/.cmd/.bat/.ps1/.com launcher
// suffix stripped, so an absolute path or a wrapped launcher still matches.
func DetectProvider(command string) (provider string, recognized bool) {
	base := strings.ToLower(strings.TrimSpace(command))
	// Strip any directory component handling BOTH path separators regardless of the
	// host OS. filepath.Base is host-specific — on a Linux CI runner it does not
	// split a Windows backslash path, so a launcher like `C:\…\claude.exe` would
	// fail to match there even though it works on a Windows dev box. LastIndexAny
	// over `/` and `\` makes the match cross-platform.
	if i := strings.LastIndexAny(base, `/\`); i >= 0 {
		base = base[i+1:]
	}
	switch filepath.Ext(base) {
	case ".exe", ".cmd", ".bat", ".ps1", ".com":
		base = strings.TrimSuffix(base, filepath.Ext(base))
	}
	switch base {
	case "claude", "claude-code":
		return "anthropic", true
	case "codex", "opencode", "aider":
		return "openai", true
	default:
		return "", false
	}
}

// ResolveProvider picks the upstream wire for a guard session. An explicit
// --provider value (normalized) always wins. An empty value is inferred from the
// wrapped agent's name via DetectProvider, and an unrecognized agent falls back to
// anthropic (Claude Code, the historical default) — so an existing `fak guard --
// claude` is unchanged while `fak guard -- codex` picks the OpenAI wire on its own.
// The bool reports whether the wire was inferred (for the banner / the demo card).
func ResolveProvider(flagValue, command string) (provider string, autodetected bool) {
	if v := strings.ToLower(strings.TrimSpace(flagValue)); v != "" {
		return v, false
	}
	if detected, ok := DetectProvider(command); ok {
		return detected, true
	}
	return "anthropic", false
}

// DefaultBaseURL maps a provider to its public API base URL. The anthropic host is
// given WITHOUT a /v1 suffix (the gateway's Anthropic client appends the Messages
// path), matching the witnessed `fak serve --provider anthropic --base-url
// https://api.anthropic.com`. An unknown provider returns "" so the caller can
// require an explicit --base-url instead of guessing.
func DefaultBaseURL(provider string) string {
	switch provider {
	case "anthropic":
		return "https://api.anthropic.com"
	case "openai":
		return "https://api.openai.com/v1"
	default:
		return ""
	}
}

// EnvVar picks the env var that points the child agent at the gateway. An explicit
// override always wins; otherwise it is the provider's conventional base-URL variable:
// Anthropic clients (Claude Code, the Anthropic SDKs) read ANTHROPIC_BASE_URL;
// OpenAI-compatible clients read OPENAI_BASE_URL (gemini/xai are proxied on the
// OpenAI-compatible surface here).
func EnvVar(provider, override string) string {
	if v := strings.TrimSpace(override); v != "" {
		return v
	}
	switch provider {
	case "anthropic":
		return "ANTHROPIC_BASE_URL"
	default:
		return "OPENAI_BASE_URL"
	}
}

// EnvValue is the base-URL VALUE injected into the child — and the two wires disagree
// on the /v1 suffix, which is the difference between a working session and a 404.
// Anthropic clients (Claude Code) append "/v1/messages" to ANTHROPIC_BASE_URL, so it
// must be the bare host. OpenAI-compatible clients (OpenCode, Codex, the OpenAI SDK,
// the Vercel AI SDK) treat OPENAI_BASE_URL as ending in "/v1" and append
// "/chat/completions" — so the value MUST carry the /v1 the gateway serves its OpenAI
// routes under. Without it the client calls "<host>/chat/completions" and the gateway
// (which exposes "/v1/chat/completions") 404s.
func EnvValue(provider, gwURL string) string {
	if provider == "anthropic" {
		return gwURL
	}
	return strings.TrimRight(gwURL, "/") + "/v1"
}

// InjectedEnv lists the environment variables guard sets in the child to point it at
// the gateway. An explicit override yields exactly that one var (value follows the
// wire's /v1 convention). The Anthropic wire is ANTHROPIC_BASE_URL. The
// OpenAI-compatible wire gets BOTH conventional base-URL variables a client might
// read: OPENAI_BASE_URL (the OpenAI SDK, Codex, OpenCode, the Vercel AI SDK) and
// OPENAI_API_BASE (LiteLLM-backed clients and Aider). Setting both is harmless to a
// client that reads only one, and it means more agents work under `fak guard` with no
// extra flag. Both pairs share one value (EnvValue), so the gateway URL is injected
// once under two names.
func InjectedEnv(provider, override, gwURL string) [][2]string {
	val := EnvValue(provider, gwURL)
	primary := EnvVar(provider, override)
	pairs := [][2]string{{primary, val}}
	if strings.TrimSpace(override) == "" && primary == "OPENAI_BASE_URL" {
		pairs = append(pairs, [2]string{"OPENAI_API_BASE", val})
	}
	return pairs
}

// Plan is the fully resolved drop-in wiring for one wrapped agent: which provider
// wire `fak guard` proxies to, whether that wire was inferred from the agent name,
// the upstream public base URL, and the env var(s) injected into the child process
// (and nowhere else). It is what `fak guard --dry-run` prints and what each card in
// the entry-point gallery renders — the same value, two surfaces.
type Plan struct {
	Agent        string      `json:"agent"`        // the wrapped command, verbatim (e.g. "claude", "/usr/bin/codex")
	Provider     string      `json:"provider"`     // resolved upstream wire: anthropic | openai | gemini | xai | …
	Autodetected bool        `json:"autodetected"` // the wire was inferred from the agent name (no --provider given)
	Recognized   bool        `json:"recognized"`   // the agent name is in the known-agent table
	BaseURL      string      `json:"base_url"`     // upstream public API base for this provider ("" if none — needs --base-url)
	EnvVars      [][2]string `json:"env_vars"`     // env var(s) injected into the CHILD only ([name,value] pairs)
	GatewayURL   string      `json:"gateway_url"`  // the loopback gateway URL the child is pointed at
}

// PlanFor resolves the complete drop-in wiring for `fak guard [--provider P]
// [--env E] -- <agentCmd>`, pointed at the loopback gateway gwURL. It is the single
// call that composes the resolution steps the way guard does at startup, so a Plan is
// a faithful preview of what a real wrapped session would wire — used by both
// `fak guard --dry-run` and the demo gallery.
func PlanFor(agentCmd, providerFlag, envOverride, gwURL string) Plan {
	provider, autodetected := ResolveProvider(providerFlag, agentCmd)
	_, recognized := DetectProvider(agentCmd)
	return Plan{
		Agent:        agentCmd,
		Provider:     provider,
		Autodetected: autodetected,
		Recognized:   recognized,
		BaseURL:      DefaultBaseURL(provider),
		EnvVars:      InjectedEnv(provider, envOverride, gwURL),
		GatewayURL:   gwURL,
	}
}

// Agent is one known entry point in the drop-in gallery: a coding agent `fak guard`
// recognizes by name, so `fak guard -- <Command>` wires it with NO --provider flag.
// The metadata here is for display; the WIRING for a card is always recomputed through
// PlanFor(Command, …) so the gallery shows exactly what guard resolves, never a
// hand-maintained copy that could drift.
type Agent struct {
	Command string `json:"command"` // the executable name guard matches (also the `fak guard -- <Command>` you type)
	Display string `json:"display"` // human label, e.g. "Claude Code"
	Note    string `json:"note"`    // one line: what it is / how it authenticates
	Home    string `json:"home"`    // docs/home URL (display only)
}

// KnownAgents returns the coding agents `fak guard` autodetects — the ones where the
// drop-in is literally `fak guard -- <name>` with no other flag. This list is the
// display companion to DetectProvider's recognized set: every Command here MUST be
// recognized by DetectProvider (TestKnownAgentsAreRecognized pins that), so the
// gallery never advertises an autodetect that does not exist. The broader long tail —
// any tool that lets you set a base URL — is reached with an explicit `--provider` or
// via `fak serve`, and is documented in docs/integrations/compatibility-matrix.md
// rather than enumerated here.
func KnownAgents() []Agent {
	return []Agent{
		{
			Command: "claude",
			Display: "Claude Code",
			Note:    "Anthropic's coding CLI. Runs on your Claude Pro/Max subscription by default — no API key needed.",
			Home:    "https://docs.claude.com/en/docs/claude-code",
		},
		{
			Command: "codex",
			Display: "OpenAI Codex",
			Note:    "OpenAI's coding CLI on the OpenAI-compatible wire. Bring OPENAI_API_KEY, or point --base-url at a local model.",
			Home:    "https://github.com/openai/codex",
		},
		{
			Command: "opencode",
			Display: "OpenCode",
			Note:    "Open-source coding agent on the OpenAI-compatible wire; its lowercase tool set crosses the same floor.",
			Home:    "https://opencode.ai",
		},
		{
			Command: "aider",
			Display: "Aider",
			Note:    "Pair-programming CLI. Reads OPENAI_API_BASE, which guard injects alongside OPENAI_BASE_URL.",
			Home:    "https://aider.chat",
		},
	}
}
