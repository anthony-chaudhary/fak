package main

import (
	"fmt"
	"io"
	"strings"
)

// guard_codex.go — the first-class `fak guard -- codex` wiring. It is the OpenAI-Codex
// twin of the Claude `--settings` hook install (guard_stophook.go / guard_precompact.go):
// the piece that makes the wrapped agent actually route through the in-process kernel
// gateway, on the wire that agent natively speaks.
//
// Why Codex needs its own install path. The other OpenAI-wire agents (OpenCode, Aider)
// read OPENAI_BASE_URL / OPENAI_API_BASE, so guardInjectedEnv repoints them with an env
// var alone. The modern Codex CLI does NOT: a custom upstream is defined by a
// `[model_providers.<id>]` table in ~/.codex/config.toml (an injected OPENAI_BASE_URL is
// not a reliable repoint), and the current Codex docs prefer the Responses API while
// deprecating Chat Completions for future removal. So to put the kernel in front of Codex
// we (1) autodetect the `openai-responses` UPSTREAM wire (guardDetectProvider) so current
// Codex models round-trip on the recommended wire, and (2) inject the provider definition
// Codex honors via its highest-precedence mechanism: per-invocation `-c key=value`
// overrides on the child command, defining a `fak` provider whose base_url is the
// gateway's `/v1`. Codex then POSTs `/v1/responses` at the gateway, every proposed tool
// call crosses the same capability floor as the Claude path, and the gateway proxies
// upstream on the Responses wire.
//
// Credential posture (the honest fence). The injected provider authenticates with an API
// key read from env_key (default OPENAI_API_KEY); the child sends it to the gateway and
// the gateway forwards it upstream (or substitutes the --api-key-env key). This is the
// API-billing path. A Codex ChatGPT subscription login (`codex login`) is NOT yet wired
// through guard the way the Claude Pro/Max subscription is — that is a named follow-on, not
// a claim this path makes.

// guardCodexProviderID is the model-provider id `fak guard` defines in Codex's config for
// the gateway. It must avoid Codex's reserved built-in ids (openai, ollama, lmstudio), so
// a plain "fak" is both safe and self-describing in Codex's `/status` provider line.
const guardCodexProviderID = "fak"

// guardCodexDefaultEnvKey is the env var Codex reads the upstream bearer token from when
// the operator names no --api-key-env. It matches the OpenAI SDK convention, so a box that
// already exports OPENAI_API_KEY for Codex keeps working with the kernel in front.
const guardCodexDefaultEnvKey = "OPENAI_API_KEY"

// guardIsCodex reports whether the wrapped agent is the OpenAI Codex CLI, matched on the
// executable's normalized base name (so an absolute path or a Windows launcher suffix still
// matches). It is the gate for installGuardCodexConfig: the `-c` override syntax is
// Codex-specific, so it must never be appended to any other agent's argv — `fak guard
// --provider openai-responses -- some-other-agent` gets the wire but not the Codex flags.
func guardIsCodex(command string) bool {
	return guardAgentBaseName(command) == "codex"
}

// guardCodexInstall records what the Codex config injection did, for the banner and tests.
type guardCodexInstall struct {
	Applied    bool
	ProviderID string
	EnvKey     string
	BaseURL    string
}

// guardCodexConfigArgs builds the ordered `-c key=value` override arguments that point
// Codex at the gateway. Each value is a TOML literal, so strings carry their double quotes
// verbatim (guard execs the child directly, with no shell to strip them — Codex's own TOML
// parser consumes the quotes). base_url is the gateway origin plus the `/v1` Codex appends
// `/responses` to, so the request lands on the gateway's `/v1/responses` route.
func guardCodexConfigArgs(gwURL, apiKeyEnv string) []string {
	base := guardCodexBaseURL(gwURL)
	envKey := guardCodexEnvKey(apiKeyEnv)
	id := guardCodexProviderID
	q := func(s string) string { return `"` + s + `"` }
	return []string{
		"-c", "model_provider=" + id,
		"-c", "model_providers." + id + ".name=" + q("fak (kernel-adjudicated)"),
		"-c", "model_providers." + id + ".base_url=" + q(base),
		"-c", "model_providers." + id + ".wire_api=" + q("responses"),
		"-c", "model_providers." + id + ".env_key=" + q(envKey),
	}
}

// guardCodexBaseURL is the gateway origin with the single `/v1` suffix Codex's Responses
// client appends `/responses` to. Idempotent on a gwURL that already carries `/v1`, and a
// trailing slash is trimmed first so it never doubles up.
func guardCodexBaseURL(gwURL string) string {
	b := strings.TrimRight(strings.TrimSpace(gwURL), "/")
	if b == "" || strings.HasSuffix(b, "/v1") {
		return b
	}
	return b + "/v1"
}

// guardCodexEnvKey resolves the env var Codex reads the upstream bearer from: the operator's
// --api-key-env when set, else the OPENAI_API_KEY convention.
func guardCodexEnvKey(apiKeyEnv string) string {
	if v := strings.TrimSpace(apiKeyEnv); v != "" {
		return v
	}
	return guardCodexDefaultEnvKey
}

// installGuardCodexConfig rewrites a Codex child command to route through the gateway by
// prepending the `-c` provider overrides immediately after the codex executable — before
// any subcommand (`exec`) or user args, since Codex's global `-c` flag precedes the
// subcommand. A non-Codex agent, or enabled=false, is returned unchanged (no install), so
// the path is inert for every other wrapped agent. An empty command is a no-op.
func installGuardCodexConfig(command []string, enabled bool, gwURL, apiKeyEnv string) ([]string, guardCodexInstall) {
	if !enabled || len(command) == 0 || !guardIsCodex(command[0]) {
		return command, guardCodexInstall{}
	}
	args := guardCodexConfigArgs(gwURL, apiKeyEnv)
	out := make([]string, 0, len(command)+len(args))
	out = append(out, command[0])
	out = append(out, args...)
	out = append(out, command[1:]...)
	return out, guardCodexInstall{
		Applied:    true,
		ProviderID: guardCodexProviderID,
		EnvKey:     guardCodexEnvKey(apiKeyEnv),
		BaseURL:    guardCodexBaseURL(gwURL),
	}
}

// printGuardCodexNote explains the Codex repoint on the banner: the gateway provider that
// was injected, the wire, and the credential env var the child must hold. It also names the
// honest fence (API-key billing, not the ChatGPT subscription) so a subscription-only user
// is told why a missing OPENAI_API_KEY would fail rather than hitting an opaque Codex error.
func printGuardCodexNote(w io.Writer, in guardCodexInstall) {
	if !in.Applied {
		return
	}
	fmt.Fprintf(w, "fak guard: Codex wired via -c model_provider=%s (wire_api=responses, base_url=%s) — every tool call crosses the kernel floor\n", in.ProviderID, in.BaseURL)
	fmt.Fprintf(w, "fak guard: Codex upstream auth — API key from $%s (forwarded upstream). A `codex login` ChatGPT subscription is not yet wired through guard; export %s for first-class `fak guard -- codex`.\n", in.EnvKey, in.EnvKey)
}
