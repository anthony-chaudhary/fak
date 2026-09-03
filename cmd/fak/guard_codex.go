package main

import (
	"errors"
	"fmt"
	"io"
	"os"
	"strings"

	configaccounts "github.com/anthony-chaudhary/fak/internal/accounts"
	"github.com/anthony-chaudhary/fak/internal/harnessprofile"
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
// Credential posture. The injected provider authenticates to the local gateway with an
// env_key (default OPENAI_API_KEY). On the API-key path that key is the upstream secret
// selected by --api-key-env or the client. On the ChatGPT-subscription path, guard holds
// the real `codex login` OAuth token upstream and gives the child only a placeholder key.

// guardCodexProviderID is the model-provider id `fak guard` defines in Codex's config for
// the gateway. It must avoid Codex's reserved built-in ids (openai, ollama, lmstudio), so
// a plain "fak" is both safe and self-describing in Codex's `/status` provider line.
const guardCodexProviderID = "fak"

// guardCodexDefaultEnvKey is the env var Codex reads the upstream bearer token from when
// the operator names no --api-key-env. It matches the OpenAI SDK convention, so a box that
// already exports OPENAI_API_KEY for Codex keeps working with the kernel in front.
const guardCodexDefaultEnvKey = "OPENAI_API_KEY"

// guardIsCodex reports whether the wrapped agent takes the Codex `cli-config` repoint — the
// `-c model_providers.fak.*` overrides installGuardCodexConfig prepends. It now delegates to
// the profile registry (C3, #1954): a harness gets cli-config iff its HarnessProfile declares
// RepointCLIConfig, which today is exactly the codex profile. So the `-c` override syntax is
// still never appended to any other agent's argv, but the SELECTION is data (profile.Repoint),
// not a hardcoded base-name check — matching on the same normalization as before
// (harnessprofile.Lookup ports guardAgentBaseName).
func guardIsCodex(command string) bool {
	return guardProfileHasRepoint(command, harnessprofile.RepointCLIConfig)
}

// guardCodexInstall records what the Codex config injection did, for the banner and tests.
type guardCodexInstall struct {
	Applied    bool
	ProviderID string
	EnvKey     string
	BaseURL    string
	Model      string
	Reasoning  string
	// ReasoningOptIn records that Reasoning came from the operator's explicit
	// FAK_GUARD_CODEX_REASONING_EFFORT opt-in rather than the configured default, so an
	// escalated effort is attributable to a decision, not a silent guard posture (#10669).
	ReasoningOptIn bool
	AuthMode       string
	AuthSource     string
}

const (
	guardCodexLocalPlaceholderAPIKey = "fak-local-codex-placeholder"
	guardCodexOAuthPlaceholderAPIKey = "fak-guard-oauth-placeholder"
	guardCodexDefaultModelID         = configaccounts.CodexDefaultModel
	guardCodexDefaultReasoningEffort = configaccounts.CodexDefaultReasoningEffort
)

func guardCodexGatewayModel(command []string, model, provider string) string {
	var profile harnessprofile.HarnessProfile
	if len(command) > 0 {
		profile, _ = harnessprofile.Lookup(command[0])
	}
	return guardCodexGatewayModelForProfile(profile, model, provider)
}

func guardCodexGatewayModelForProfile(profile harnessprofile.HarnessProfile, model, provider string) string {
	if strings.TrimSpace(model) != "" {
		return strings.TrimSpace(model)
	}
	if profile.HasRepoint(harnessprofile.RepointCLIConfig) && strings.TrimSpace(provider) == "openai-responses" {
		return guardCodexDefaultModelID
	}
	return model
}

func guardCodexLoopGateConfig(command []string, threshold, codexHome string, sinceHours float64, limit int, quiet bool) (codexLoopGateConfig, bool) {
	var profile harnessprofile.HarnessProfile
	if len(command) > 0 {
		profile, _ = harnessprofile.Lookup(command[0])
	}
	return guardCodexLoopGateConfigForProfile(profile, command, threshold, codexHome, sinceHours, limit, quiet)
}

func guardCodexLoopGateConfigForProfile(profile harnessprofile.HarnessProfile, command []string, threshold, codexHome string, sinceHours float64, limit int, quiet bool) (codexLoopGateConfig, bool) {
	if len(command) == 0 || !profile.HasRepoint(harnessprofile.RepointCLIConfig) {
		return codexLoopGateConfig{}, false
	}
	return codexLoopGateConfig{
		Threshold:     threshold,
		CodexHome:     codexHome,
		SinceHours:    sinceHours,
		Limit:         limit,
		Quiet:         quiet,
		BypassCommand: "fak m --codex-loop-gate off -- " + strings.Join(command, " "),
	}, true
}

// guardCodexConfigArgs builds the ordered `-c key=value` override arguments that point
// Codex at the gateway. Each value is a TOML literal, so strings carry their double quotes
// verbatim (guard execs the child directly, with no shell to strip them — Codex's own TOML
// parser consumes the quotes). base_url is the gateway origin plus the `/v1` Codex appends
// `/responses` to, so the request lands on the gateway's `/v1/responses` route. The same
// session receives the gateway MCP endpoint additively, allowing Codex's native tool router to
// execute the FAK substrate tools that guard exposes to the model.
func guardCodexConfigArgs(gwURL, apiKeyEnv, model string) []string {
	base := guardCodexBaseURL(gwURL)
	envKey := guardCodexEnvKey(apiKeyEnv)
	model = guardCodexConfiguredModel(model)
	effort := guardCodexResolveReasoningEffort(model, os.Getenv).Effort
	id := guardCodexProviderID
	q := func(s string) string { return `"` + s + `"` }
	args := []string{
		"-c", "model_provider=" + id,
		"-c", "model=" + q(model),
		"-c", "model_providers." + id + ".name=" + q("fak (kernel-adjudicated)"),
		"-c", "model_providers." + id + ".base_url=" + q(base),
		"-c", "model_providers." + id + ".wire_api=" + q("responses"),
		"-c", "model_providers." + id + ".env_key=" + q(envKey),
		"-c", "mcp_servers.fak_guard.url=" + q(guardCodexMCPURL(gwURL)),
	}
	if effort != "" {
		args = append(args, "-c", "model_reasoning_effort="+q(effort))
	}
	return args
}

func guardCodexConfiguredModel(model string) string {
	if model = strings.TrimSpace(model); model != "" {
		return model
	}
	return guardCodexDefaultModelID
}

// guardCodexReasoningEffortEnv is the explicit operator opt-in for a guarded Codex session's
// reasoning effort (#10669). The guard never escalates silently: unset/empty defers to the
// configured default, and only a non-empty value here can pin xhigh (which roughly doubles
// the post-tool wait vs high). The value is passed through verbatim (whitespace-trimmed, not
// case-munged) — an explicit operator value is Codex's to accept or reject loudly.
const guardCodexReasoningEffortEnv = "FAK_GUARD_CODEX_REASONING_EFFORT"

// guardCodexEffortResolution is the resolved reasoning-effort decision for one guarded Codex
// install: the effort the session config pins ("" = no pin) and whether it came from the
// explicit opt-in rather than the configured default.
type guardCodexEffortResolution struct {
	Effort string
	OptIn  bool
}

// guardCodexResolveReasoningEffort resolves the guarded Codex session's reasoning effort in
// strict order (#10669): an explicit $FAK_GUARD_CODEX_REASONING_EFFORT opt-in wins over
// everything; otherwise the managed GPT-5.6 default gets the configured effort — never the
// silent xhigh escalation the guard used to force onto every goal-continuation turn; an
// explicit custom/local model keeps its own supported-effort contract instead of receiving a
// config value it may reject. A later user-supplied `-c model_reasoning_effort=...` still
// overrides this earlier default in Codex's argv.
func guardCodexResolveReasoningEffort(model string, getenv func(string) string) guardCodexEffortResolution {
	if getenv != nil {
		if v := strings.TrimSpace(getenv(guardCodexReasoningEffortEnv)); v != "" {
			return guardCodexEffortResolution{Effort: v, OptIn: true}
		}
	}
	return guardCodexEffortResolution{Effort: guardCodexReasoningEffort(model)}
}

// guardCodexReasoningEffort is the no-opt-in effort for the model Codex is being pointed at:
// the configured default for the managed GPT-5.6 models, and no pin at all for a custom or
// local model whose supported-effort set the guard cannot know.
func guardCodexReasoningEffort(model string) string {
	switch strings.ToLower(strings.TrimSpace(model)) {
	case "gpt-5.6", "gpt-5.6-sol":
		return guardCodexDefaultReasoningEffort
	default:
		return ""
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

// guardCodexMCPURL points Codex's native MCP client at the same in-process gateway as the
// model route. Accept either the gateway origin or its /v1 model base without producing
// /v1/mcp, which is not an MCP route.
func guardCodexMCPURL(gwURL string) string {
	b := strings.TrimRight(strings.TrimSpace(gwURL), "/")
	b = strings.TrimSuffix(b, "/v1")
	if b == "" {
		return ""
	}
	return b + "/mcp"
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
	var profile harnessprofile.HarnessProfile
	if len(command) > 0 {
		profile, _ = harnessprofile.Lookup(command[0])
	}
	return installGuardCodexConfigForProfile(command, profile, enabled, gwURL, apiKeyEnv)
}

func guardCodexAuthManagementCommand(command []string) bool {
	if len(command) == 2 {
		return command[0] == "codex" &&
			(command[1] == "login" || command[1] == "logout")
	}
	return len(command) == 3 &&
		command[0] == "codex" &&
		command[1] == "login" &&
		command[2] == "status"
}
func installGuardCodexConfigForProfile(command []string, profile harnessprofile.HarnessProfile, enabled bool, gwURL, apiKeyEnv string) ([]string, guardCodexInstall) {
	if !enabled || len(command) == 0 || guardCodexAuthManagementCommand(command) || !profile.HasRepoint(harnessprofile.RepointCLIConfig) {
		return command, guardCodexInstall{}
	}
	model := guardCodexDefaultModelID
	resolved := guardCodexResolveReasoningEffort(model, os.Getenv)
	args := guardCodexConfigArgs(gwURL, apiKeyEnv, model)
	out := make([]string, 0, len(command)+len(args))
	out = append(out, command[0])
	out = append(out, args...)
	out = append(out, command[1:]...)
	return out, guardCodexInstall{
		Applied:        true,
		ProviderID:     guardCodexProviderID,
		EnvKey:         guardCodexEnvKey(apiKeyEnv),
		BaseURL:        guardCodexBaseURL(gwURL),
		Model:          model,
		Reasoning:      resolved.Effort,
		ReasoningOptIn: resolved.OptIn,
	}
}

// guardCodexAuthEnv returns any explicit env grant the Codex child needs for the injected
// provider. Codex validates model_providers.fak.env_key before the first turn, so an absent
// key otherwise surfaces as an opaque Codex startup error after plugin loading. For API
// billing, hand the resolved key to the child explicitly; for pure local in-kernel mode, a
// placeholder is enough because the gateway never proxies to an upstream provider.
func guardCodexAuthEnv(in guardCodexInstall, upstreamAPIKey string, localOnly bool, getenv func(string) string) ([][2]string, error) {
	if !in.Applied {
		return nil, nil
	}
	envKey := strings.TrimSpace(in.EnvKey)
	if envKey == "" {
		return nil, errors.New("Codex provider env_key is empty")
	}
	if in.AuthMode == "chatgpt" {
		return [][2]string{{envKey, guardCodexOAuthPlaceholderAPIKey}}, nil
	}
	if strings.TrimSpace(upstreamAPIKey) != "" {
		return [][2]string{{envKey, upstreamAPIKey}}, nil
	}
	if getenv == nil {
		getenv = func(string) string { return "" }
	}
	if strings.TrimSpace(getenv(envKey)) != "" {
		return nil, nil
	}
	if localOnly {
		return [][2]string{{envKey, guardCodexLocalPlaceholderAPIKey}}, nil
	}
	return nil, fmt.Errorf("Codex provider env $%s is empty and no ChatGPT subscription auth.json was resolved. Run `codex login`, export %s, or pass --api-key-env VAR. For a local or no-auth OpenAI-compatible upstream, set %s to any non-empty placeholder.", envKey, envKey, envKey)
}

// printGuardCodexNote explains the Codex repoint on the banner: the gateway provider that
// was injected, the wire, and the credential posture the child sees.
func printGuardCodexNote(w io.Writer, in guardCodexInstall) {
	if !in.Applied {
		return
	}
	fmt.Fprintf(w, "fak guard: Codex wired via -c model_provider=%s (wire_api=responses, base_url=%s) — every tool call crosses the kernel floor\n", in.ProviderID, in.BaseURL)
	if in.Reasoning != "" {
		if in.ReasoningOptIn {
			fmt.Fprintf(w, "fak guard: Codex session config — model=%s model_reasoning_effort=%s (explicit opt-in via $%s)\n", in.Model, in.Reasoning, guardCodexReasoningEffortEnv)
		} else {
			fmt.Fprintf(w, "fak guard: Codex session config — model=%s model_reasoning_effort=%s\n", in.Model, in.Reasoning)
		}
	} else {
		fmt.Fprintf(w, "fak guard: Codex session config — model=%s (custom model keeps its own reasoning-effort default)\n", in.Model)
	}
	if in.AuthMode == "chatgpt" {
		fmt.Fprintf(w, "fak guard: Codex upstream auth — ChatGPT subscription from %s (token held by guard; child sees $%s placeholder)\n", in.AuthSource, in.EnvKey)
		return
	}
	fmt.Fprintf(w, "fak guard: Codex upstream auth — API key from $%s when API billing is selected; `codex login` is used automatically when a ChatGPT subscription auth.json is present.\n", in.EnvKey)
}
