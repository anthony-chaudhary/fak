package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"sync"

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
	ReasoningOptIn     bool
	AuthMode           string
	AuthSource         string
	SandboxContainment bool
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
		return configaccounts.NormalizeCodexModelSlug(strings.TrimSpace(model))
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
//
// When the effective Codex config.toml declares [mcp_servers.fak], guard also injects
// `-c mcp_servers.fak.enabled=false` along with `-c mcp_servers.fak_guard.url=...` so that
// Codex does not start a second conflicting stdio instance of fak alongside fak_guard (#10295).
func guardCodexConfigArgs(gwURL, apiKeyEnv, model string, codexHome ...string) []string {
	base := guardCodexBaseURL(gwURL)
	envKey := guardCodexEnvKey(apiKeyEnv)
	model = guardCodexConfiguredModel(model)
	effort := guardCodexResolveReasoningEffort(model, os.Getenv).Effort
	id := guardCodexProviderID
	q := func(s string) string { return `"` + s + `"` }
	home := ""
	if len(codexHome) > 0 {
		home = codexHome[0]
	}
	args := []string{
		"-c", "model_provider=" + id,
		"-c", "model=" + q(model),
		"-c", "model_providers." + id + ".name=" + q("fak (kernel-adjudicated)"),
		"-c", "model_providers." + id + ".base_url=" + q(base),
		"-c", "model_providers." + id + ".wire_api=" + q("responses"),
		"-c", "model_providers." + id + ".env_key=" + q(envKey),
		"-c", "mcp_servers.fak_guard.url=" + q(guardCodexMCPURL(gwURL)),
	}
	if codexConfigHasMCPServerFak(effectiveCodexConfigFile(home)) {
		args = append(args, "-c", "mcp_servers.fak.enabled=false")
	}
	if effort != "" {
		args = append(args, "-c", "model_reasoning_effort="+q(effort))
	}
	return args
}

// effectiveCodexConfigFile locates the Codex config.toml to inspect:
// explicit codexHome > CODEX_HOME environment variable > ~/.codex/config.toml.
func effectiveCodexConfigFile(codexHome string) string {
	home := strings.TrimSpace(codexHome)
	if home == "" {
		home = strings.TrimSpace(os.Getenv("CODEX_HOME"))
	}
	if home != "" {
		home = expandHome(home)
		if strings.HasSuffix(strings.ToLower(home), ".toml") {
			if _, err := os.Stat(home); err == nil {
				return home
			}
		}
		p := filepath.Join(home, "config.toml")
		if _, err := os.Stat(p); err == nil {
			return p
		}
		pSub := filepath.Join(home, ".codex", "config.toml")
		if _, err := os.Stat(pSub); err == nil {
			return pSub
		}
		return p
	}
	if uHome, err := os.UserHomeDir(); err == nil && uHome != "" {
		return filepath.Join(uHome, ".codex", "config.toml")
	}
	return ""
}

// codexConfigHasMCPServerFak checks whether the Codex config.toml declares [mcp_servers.fak].
func codexConfigHasMCPServerFak(path string) bool {
	if strings.TrimSpace(path) == "" {
		return false
	}
	b, err := os.ReadFile(path)
	if err != nil {
		return false
	}
	lines := strings.Split(string(b), "\n")
	if start, _ := findTOMLSection(lines, "mcp_servers.fak"); start >= 0 {
		return true
	}
	for _, raw := range lines {
		line := strings.TrimSpace(strings.SplitN(raw, "#", 2)[0])
		if !strings.HasPrefix(line, "[") {
			continue
		}
		inner := strings.Trim(line, "[]")
		inner = strings.TrimSpace(inner)
		inner = strings.ReplaceAll(inner, `"`, "")
		inner = strings.ReplaceAll(inner, `'`, "")
		inner = strings.ReplaceAll(inner, ` `, "")
		if inner == "mcp_servers.fak" {
			return true
		}
	}
	return false
}

func guardCodexConfiguredModel(model string) string {
	if model = strings.TrimSpace(model); model != "" {
		return configaccounts.NormalizeCodexModelSlug(model)
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

var codexReasoningModelRE = regexp.MustCompile(`(^|/)(gpt-5\.6|gpt-[6-9]|o[1-9]|astra)`)

// guardCodexReasoningEffort is the no-opt-in effort for the model Codex is being pointed at:
// the configured default for detected reasoning model families (e.g. GPT-5.6*, GPT-6.*,
// GPT-7.*, o1.*-o5.*, Astra models), and no pin at all for a custom or local model
// whose supported-effort set the guard cannot know.
func guardCodexReasoningEffort(model string) string {
	m := strings.ToLower(configaccounts.NormalizeCodexModelSlug(model))
	if codexReasoningModelRE.MatchString(m) || strings.Contains(m, "astra") {
		return guardCodexDefaultReasoningEffort
	}
	return ""
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
func installGuardCodexConfig(command []string, enabled bool, gwURL, apiKeyEnv string, codexHome ...string) ([]string, guardCodexInstall) {
	var profile harnessprofile.HarnessProfile
	if len(command) > 0 {
		profile, _ = harnessprofile.Lookup(command[0])
	}
	return installGuardCodexConfigForProfile(command, profile, enabled, gwURL, apiKeyEnv, codexHome...)
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

const (
	guardCodexCompactTimeoutSeconds = 10
	guardCodexPreCompactCommand     = "fak sessions codex-compact-hook --pre"
	guardCodexPostCompactCommand    = "fak sessions codex-compact-hook --post"
)

func guardCodexHookKeyForOS(eventName, goos string) string {
	if goos == "windows" {
		return `C:\<session-flags>\config.toml:` + eventName + `:0:0`
	}
	return `/<session-flags>/config.toml:` + eventName + `:0:0`
}

func guardCodexPreCompactHookKey() string {
	return guardCodexHookKeyForOS("pre_compact", runtime.GOOS)
}

func guardCodexPostCompactHookKey() string {
	return guardCodexHookKeyForOS("post_compact", runtime.GOOS)
}

func guardCodexCompactTrustedHash(eventName, command string, timeout int) string {
	identity := map[string]any{
		"event_name": eventName,
		"hooks": []any{map[string]any{
			"async":   false,
			"command": command,
			"timeout": timeout,
			"type":    "command",
		}},
	}
	data, _ := json.Marshal(identity)
	sum := sha256.Sum256(data)
	return fmt.Sprintf("sha256:%x", sum)
}

func guardCodexCompactConfigArgs() []string {
	preKey := guardCodexPreCompactHookKey()
	preHash := guardCodexCompactTrustedHash("pre_compact", guardCodexPreCompactCommand, guardCodexCompactTimeoutSeconds)
	postKey := guardCodexPostCompactHookKey()
	postHash := guardCodexCompactTrustedHash("post_compact", guardCodexPostCompactCommand, guardCodexCompactTimeoutSeconds)

	preVal := `[{hooks=[{type="command",command="` + guardCodexPreCompactCommand + `",timeout=` + strconv.Itoa(guardCodexCompactTimeoutSeconds) + `}]}]`
	postVal := `[{hooks=[{type="command",command="` + guardCodexPostCompactCommand + `",timeout=` + strconv.Itoa(guardCodexCompactTimeoutSeconds) + `}]}]`
	stateVal := "{" + strconv.Quote(preKey) + "={trusted_hash=" + strconv.Quote(preHash) + "}," + strconv.Quote(postKey) + "={trusted_hash=" + strconv.Quote(postHash) + "}}"

	return []string{
		"-c", "hooks.PreCompact=" + preVal,
		"-c", "hooks.PostCompact=" + postVal,
		"-c", "hooks.state=" + stateVal,
	}
}

func installGuardCodexConfigForProfile(command []string, profile harnessprofile.HarnessProfile, enabled bool, gwURL, apiKeyEnv string, codexHome ...string) ([]string, guardCodexInstall) {
	if !enabled || len(command) == 0 || guardCodexAuthManagementCommand(command) || !profile.HasRepoint(harnessprofile.RepointCLIConfig) {
		return command, guardCodexInstall{}
	}
	model := guardCodexDefaultModelID
	resolved := guardCodexResolveReasoningEffort(model, os.Getenv)
	args := guardCodexConfigArgs(gwURL, apiKeyEnv, model, codexHome...)
	compactArgs := guardCodexCompactConfigArgs()

	stateIdx := -1
	for i, arg := range command {
		if strings.HasPrefix(arg, "hooks.state=") {
			stateIdx = i
			break
		}
	}

	out := make([]string, 0, len(command)+len(args)+len(compactArgs))
	out = append(out, command[0])
	out = append(out, args...)
	if stateIdx >= 0 {
		preKey := guardCodexPreCompactHookKey()
		preHash := guardCodexCompactTrustedHash("pre_compact", guardCodexPreCompactCommand, guardCodexCompactTimeoutSeconds)
		postKey := guardCodexPostCompactHookKey()
		postHash := guardCodexCompactTrustedHash("post_compact", guardCodexPostCompactCommand, guardCodexCompactTimeoutSeconds)

		mergedState := strings.TrimSuffix(strings.TrimSpace(command[stateIdx]), "}") +
			"," + strconv.Quote(preKey) + "={trusted_hash=" + strconv.Quote(preHash) + "}," +
			strconv.Quote(postKey) + "={trusted_hash=" + strconv.Quote(postHash) + "}}"
		command[stateIdx] = mergedState

		out = append(out, compactArgs[:4]...)
	} else {
		out = append(out, compactArgs...)
	}
	out = append(out, command[1:]...)
	return out, guardCodexInstall{
		Applied:            true,
		ProviderID:         guardCodexProviderID,
		EnvKey:             guardCodexEnvKey(apiKeyEnv),
		BaseURL:            guardCodexBaseURL(gwURL),
		Model:              model,
		Reasoning:          resolved.Effort,
		ReasoningOptIn:     resolved.OptIn,
		SandboxContainment: true,
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

// ---------------------------------------------------------------------------
// PROTECTED PATH CONTAINMENT & VIRTUALIZATION PROXY
// ---------------------------------------------------------------------------

// GuardCodexProtectedDirs names the internal git and agent directories protected
// under Codex sandboxed execution paths (#11523).
var GuardCodexProtectedDirs = []string{".git", ".agents", ".codex"}

// isGuardCodexProtectedPath reports whether target path refers to an internal
// protected directory (.git, .agents, .codex) or any file within them.
func isGuardCodexProtectedPath(p string) bool {
	norm := filepath.ToSlash(filepath.Clean(p))
	parts := strings.Split(norm, "/")
	for _, part := range parts {
		for _, dir := range GuardCodexProtectedDirs {
			if strings.EqualFold(part, dir) {
				return true
			}
		}
	}
	return false
}

// GuardCodexSandboxViolationError records a typed sandbox access violation on
// a protected path.
type GuardCodexSandboxViolationError struct {
	Path    string
	Op      string
	Message string
}

func (e *GuardCodexSandboxViolationError) Error() string {
	return fmt.Sprintf("sandbox access violation on protected path %q (%s): %s", e.Path, e.Op, e.Message)
}

func isSandboxViolation(err error) bool {
	if err == nil {
		return false
	}
	msg := strings.ToLower(err.Error())
	return strings.Contains(msg, "sandbox access violation") ||
		strings.Contains(msg, "sandbox violation") ||
		strings.Contains(msg, "protected path") ||
		strings.Contains(msg, "permission denied") ||
		strings.Contains(msg, "operation not permitted") ||
		strings.Contains(msg, "access is denied")
}

// GuardCodexContainmentProxy provides path virtualization and containment for
// protected directories (.git, .agents, .codex) so that sandboxed Codex worker
// processes can inspect repository and agent state safely without breaching
// sandbox invariants or crashing on access violations (#11523).
type GuardCodexContainmentProxy struct {
	WorkspaceRoot    string
	ContainmentRoot  string
	VirtualGitDir    string
	VirtualAgentsDir string
	VirtualCodexDir  string
	mu               sync.RWMutex
}

// NewGuardCodexContainmentProxy constructs a containment proxy initialized for
// the workspace root.
func NewGuardCodexContainmentProxy(workspaceRoot string) (*GuardCodexContainmentProxy, error) {
	ws, err := filepath.Abs(filepath.Clean(workspaceRoot))
	if err != nil {
		return nil, fmt.Errorf("resolve workspace root: %w", err)
	}

	containmentRoot := filepath.Join(ws, ".codex-containment")
	if err := os.MkdirAll(containmentRoot, 0o755); err != nil {
		tmpDir, tmpErr := os.MkdirTemp("", "codex-containment-*")
		if tmpErr != nil {
			return nil, fmt.Errorf("create containment directory: %w (fallback: %v)", err, tmpErr)
		}
		containmentRoot = tmpDir
	}

	proxy := &GuardCodexContainmentProxy{
		WorkspaceRoot:    ws,
		ContainmentRoot:  containmentRoot,
		VirtualGitDir:    filepath.Join(containmentRoot, ".git"),
		VirtualAgentsDir: filepath.Join(containmentRoot, ".agents"),
		VirtualCodexDir:  filepath.Join(containmentRoot, ".codex"),
	}

	if err := proxy.initVirtualGit(); err != nil {
		_ = os.RemoveAll(containmentRoot)
		return nil, fmt.Errorf("initialize virtual git containment: %w", err)
	}
	proxy.initVirtualAgents()
	proxy.initVirtualCodex()

	return proxy, nil
}

func (p *GuardCodexContainmentProxy) Close() error {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.ContainmentRoot != "" && strings.Contains(p.ContainmentRoot, "codex-containment") {
		return os.RemoveAll(p.ContainmentRoot)
	}
	return nil
}

func (p *GuardCodexContainmentProxy) IsProtected(path string) bool {
	return isGuardCodexProtectedPath(path)
}

func (p *GuardCodexContainmentProxy) VirtualizePath(targetPath string) string {
	p.mu.RLock()
	defer p.mu.RUnlock()

	norm := filepath.ToSlash(filepath.Clean(targetPath))
	wsNorm := filepath.ToSlash(filepath.Clean(p.WorkspaceRoot))

	var rel string
	if strings.HasPrefix(norm, wsNorm+"/") {
		rel = strings.TrimPrefix(norm, wsNorm+"/")
	} else if norm == wsNorm {
		rel = "."
	} else if !filepath.IsAbs(targetPath) {
		rel = norm
	} else {
		return targetPath
	}

	rel = strings.TrimPrefix(rel, "./")

	if rel == ".git" || strings.HasPrefix(rel, ".git/") {
		sub := strings.TrimPrefix(rel, ".git")
		sub = strings.TrimPrefix(sub, "/")
		return filepath.Join(p.VirtualGitDir, filepath.FromSlash(sub))
	}
	if rel == ".agents" || strings.HasPrefix(rel, ".agents/") {
		sub := strings.TrimPrefix(rel, ".agents")
		sub = strings.TrimPrefix(sub, "/")
		return filepath.Join(p.VirtualAgentsDir, filepath.FromSlash(sub))
	}
	if rel == ".codex" || strings.HasPrefix(rel, ".codex/") {
		sub := strings.TrimPrefix(rel, ".codex")
		sub = strings.TrimPrefix(sub, "/")
		return filepath.Join(p.VirtualCodexDir, filepath.FromSlash(sub))
	}

	return targetPath
}

func (p *GuardCodexContainmentProxy) ReadProtected(relOrAbsPath string) ([]byte, error) {
	p.mu.RLock()
	defer p.mu.RUnlock()

	virtualPath := p.VirtualizePath(relOrAbsPath)
	if data, err := os.ReadFile(virtualPath); err == nil {
		return data, nil
	}

	target := relOrAbsPath
	if !filepath.IsAbs(target) {
		target = filepath.Join(p.WorkspaceRoot, relOrAbsPath)
	}
	data, err := os.ReadFile(target)
	if err != nil {
		return nil, err
	}
	_ = os.MkdirAll(filepath.Dir(virtualPath), 0o755)
	_ = os.WriteFile(virtualPath, data, 0o644)
	return data, nil
}

func (p *GuardCodexContainmentProxy) initVirtualGit() error {
	dotGit := filepath.Join(p.WorkspaceRoot, ".git")
	fi, err := os.Stat(dotGit)
	if err != nil {
		return os.MkdirAll(p.VirtualGitDir, 0o755)
	}

	targetDir := dotGit
	commonGitDir := dotGit
	if !fi.IsDir() {
		content, readErr := os.ReadFile(dotGit)
		if readErr == nil {
			line := strings.TrimSpace(string(content))
			if rest, ok := strings.CutPrefix(line, "gitdir:"); ok {
				ptr := strings.TrimSpace(rest)
				if !filepath.IsAbs(ptr) {
					ptr = filepath.Join(p.WorkspaceRoot, ptr)
				}
				targetDir = filepath.Clean(ptr)
			}
		}
		if commonBytes, readErr := os.ReadFile(filepath.Join(targetDir, "commondir")); readErr == nil {
			c := strings.TrimSpace(string(commonBytes))
			if filepath.IsAbs(c) {
				commonGitDir = filepath.Clean(c)
			} else {
				commonGitDir = filepath.Clean(filepath.Join(targetDir, c))
			}
		} else {
			commonGitDir = targetDir
		}
	}

	if err := os.MkdirAll(p.VirtualGitDir, 0o755); err != nil {
		return err
	}

	srcHEAD := filepath.Join(targetDir, "HEAD")
	if _, err := os.Stat(srcHEAD); err == nil {
		_ = copyGuardCodexFile(srcHEAD, filepath.Join(p.VirtualGitDir, "HEAD"))
	}

	srcIndex := filepath.Join(targetDir, "index")
	if _, err := os.Stat(srcIndex); err == nil {
		_ = copyGuardCodexFile(srcIndex, filepath.Join(p.VirtualGitDir, "index"))
	}

	srcConfig := filepath.Join(commonGitDir, "config")
	if _, err := os.Stat(srcConfig); err == nil {
		_ = copyGuardCodexFile(srcConfig, filepath.Join(p.VirtualGitDir, "config"))
	}

	srcRefs := filepath.Join(commonGitDir, "refs")
	if sfi, err := os.Stat(srcRefs); err == nil && sfi.IsDir() {
		_ = copyGuardCodexDir(srcRefs, filepath.Join(p.VirtualGitDir, "refs"), false)
	}

	srcPackedRefs := filepath.Join(commonGitDir, "packed-refs")
	if _, err := os.Stat(srcPackedRefs); err == nil {
		_ = copyGuardCodexFile(srcPackedRefs, filepath.Join(p.VirtualGitDir, "packed-refs"))
	}

	srcObjects := filepath.Join(commonGitDir, "objects")
	dstObjects := filepath.Join(p.VirtualGitDir, "objects")
	if sfi, err := os.Stat(srcObjects); err == nil && sfi.IsDir() {
		_ = copyGuardCodexDir(srcObjects, dstObjects, true)
		infoDir := filepath.Join(dstObjects, "info")
		_ = os.MkdirAll(infoDir, 0o755)
		_ = os.WriteFile(filepath.Join(infoDir, "alternates"), []byte(srcObjects+"\n"), 0o644)
	}

	infoDir := filepath.Join(p.VirtualGitDir, "info")
	_ = os.MkdirAll(infoDir, 0o755)

	return nil
}

func (p *GuardCodexContainmentProxy) initVirtualAgents() {
	src := filepath.Join(p.WorkspaceRoot, ".agents")
	if fi, err := os.Stat(src); err == nil && fi.IsDir() {
		_ = copyGuardCodexDir(src, p.VirtualAgentsDir, false)
	}
}

func (p *GuardCodexContainmentProxy) initVirtualCodex() {
	src := filepath.Join(p.WorkspaceRoot, ".codex")
	if fi, err := os.Stat(src); err == nil && fi.IsDir() {
		_ = copyGuardCodexDir(src, p.VirtualCodexDir, false)
	}
}

func (p *GuardCodexContainmentProxy) syncGitIndexLocked() {
	srcIndex := filepath.Join(p.WorkspaceRoot, ".git", "index")
	dstIndex := filepath.Join(p.VirtualGitDir, "index")
	srcFi, err := os.Stat(srcIndex)
	if err != nil {
		return
	}
	dstFi, err := os.Stat(dstIndex)
	if err == nil && dstFi.ModTime().After(srcFi.ModTime()) {
		return
	}
	_ = copyGuardCodexFile(srcIndex, dstIndex)
}

func (p *GuardCodexContainmentProxy) WrapGitCommand(command []string) ([]string, []string, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if len(command) == 0 {
		return command, nil, nil
	}

	p.syncGitIndexLocked()

	hasGitDir := false
	for _, arg := range command {
		if arg == "--git-dir" || strings.HasPrefix(arg, "--git-dir=") {
			hasGitDir = true
			break
		}
	}

	wrapped := make([]string, 0, len(command)+4)
	wrapped = append(wrapped, command[0])
	if !hasGitDir {
		wrapped = append(wrapped, "--git-dir="+p.VirtualGitDir, "--work-tree="+p.WorkspaceRoot)
	}
	wrapped = append(wrapped, command[1:]...)

	env := []string{
		"GIT_DIR=" + p.VirtualGitDir,
		"GIT_WORK_TREE=" + p.WorkspaceRoot,
	}

	return wrapped, env, nil
}

func copyGuardCodexFile(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()

	fi, err := in.Stat()
	if err != nil {
		return err
	}

	_ = os.MkdirAll(filepath.Dir(dst), 0o755)
	out, err := os.OpenFile(dst, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, fi.Mode().Perm())
	if err != nil {
		return err
	}
	defer out.Close()

	if _, err := io.Copy(out, in); err != nil {
		return err
	}
	return out.Close()
}

func linkOrCopyGuardCodexFile(src, dst string) error {
	_ = os.MkdirAll(filepath.Dir(dst), 0o755)
	_ = os.Remove(dst)
	if err := os.Link(src, dst); err == nil {
		return nil
	}
	return copyGuardCodexFile(src, dst)
}

func copyGuardCodexDir(src, dst string, linkFiles bool) error {
	entries, err := os.ReadDir(src)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(dst, 0o755); err != nil {
		return err
	}
	for _, entry := range entries {
		srcPath := filepath.Join(src, entry.Name())
		dstPath := filepath.Join(dst, entry.Name())
		if entry.IsDir() {
			if err := copyGuardCodexDir(srcPath, dstPath, linkFiles); err != nil {
				return err
			}
		} else {
			if linkFiles {
				if err := linkOrCopyGuardCodexFile(srcPath, dstPath); err != nil {
					return err
				}
			} else {
				if err := copyGuardCodexFile(srcPath, dstPath); err != nil {
					return err
				}
			}
		}
	}
	return nil
}

func isGitInspectionSubcommand(subcmd string) bool {
	switch strings.ToLower(strings.TrimSpace(subcmd)) {
	case "status", "diff", "log", "show", "rev-parse", "branch", "describe", "tag", "cat-file":
		return true
	default:
		return false
	}
}

// GuardCodexExecutionResult captures the result of a command run within a sandboxed session.
type GuardCodexExecutionResult struct {
	ExitCode int
	Stdout   string
	Stderr   string
	Proxied  bool
}

// GuardCodexSandboxSession models a sandboxed Codex worker execution session with
// protected path containment support (#11523).
type GuardCodexSandboxSession struct {
	WorkspaceRoot string
	SandboxActive bool
	Containment   *GuardCodexContainmentProxy
}

// NewGuardCodexSandboxSession instantiates a sandboxed Codex execution session with
// path virtualization and containment initialized.
func NewGuardCodexSandboxSession(workspaceRoot string, sandboxActive bool) (*GuardCodexSandboxSession, error) {
	ws, err := filepath.Abs(filepath.Clean(workspaceRoot))
	if err != nil {
		return nil, fmt.Errorf("invalid workspace root: %w", err)
	}
	proxy, err := NewGuardCodexContainmentProxy(ws)
	if err != nil {
		return nil, fmt.Errorf("initialize containment proxy: %w", err)
	}
	return &GuardCodexSandboxSession{
		WorkspaceRoot: ws,
		SandboxActive: sandboxActive,
		Containment:   proxy,
	}, nil
}

func (s *GuardCodexSandboxSession) Close() error {
	if s.Containment != nil {
		return s.Containment.Close()
	}
	return nil
}

func (s *GuardCodexSandboxSession) ReadProtected(relOrAbsPath string) ([]byte, error) {
	if s.Containment != nil {
		return s.Containment.ReadProtected(relOrAbsPath)
	}
	target := relOrAbsPath
	if !filepath.IsAbs(target) {
		target = filepath.Join(s.WorkspaceRoot, relOrAbsPath)
	}
	return os.ReadFile(target)
}

func (s *GuardCodexSandboxSession) SafeSandboxAccess(path string, op func() error) (err error) {
	defer func() {
		if r := recover(); r != nil {
			if isGuardCodexProtectedPath(path) {
				err = nil
				return
			}
			panic(r)
		}
	}()
	err = op()
	if err != nil && isGuardCodexProtectedPath(path) && isSandboxViolation(err) {
		return nil
	}
	return err
}

func (s *GuardCodexSandboxSession) ExecuteCommand(ctx context.Context, command []string) (res GuardCodexExecutionResult, err error) {
	if len(command) == 0 {
		return GuardCodexExecutionResult{}, nil
	}

	defer func() {
		if r := recover(); r != nil {
			res = GuardCodexExecutionResult{
				ExitCode: 0,
				Stdout:   fmt.Sprintf("[containment proxy active: recovered from %v]", r),
				Proxied:  true,
			}
			err = nil
		}
	}()

	isGit := strings.EqualFold(filepath.Base(command[0]), "git") || strings.EqualFold(filepath.Base(command[0]), "git.exe")

	if s.SandboxActive && isGit && len(command) > 1 && isGitInspectionSubcommand(command[1]) {
		wrappedCmd, env, wrapErr := s.Containment.WrapGitCommand(command)
		if wrapErr != nil {
			return GuardCodexExecutionResult{ExitCode: 1, Stderr: wrapErr.Error()}, wrapErr
		}

		cmd := exec.CommandContext(ctx, wrappedCmd[0], wrappedCmd[1:]...)
		cmd.Dir = s.WorkspaceRoot
		cmd.Env = append(os.Environ(), env...)

		var stdoutBuf, stderrBuf bytes.Buffer
		cmd.Stdout = &stdoutBuf
		cmd.Stderr = &stderrBuf

		runErr := cmd.Run()
		exitCode := 0
		if runErr != nil {
			var ee *exec.ExitError
			if errors.As(runErr, &ee) {
				exitCode = ee.ExitCode()
			} else {
				exitCode = 1
			}
		}

		return GuardCodexExecutionResult{
			ExitCode: exitCode,
			Stdout:   stdoutBuf.String(),
			Stderr:   stderrBuf.String(),
			Proxied:  true,
		}, runErr
	}

	var cmdArgs []string
	if s.SandboxActive {
		cmdArgs = make([]string, len(command))
		for i, arg := range command {
			if isGuardCodexProtectedPath(arg) {
				cmdArgs[i] = s.Containment.VirtualizePath(arg)
			} else {
				cmdArgs[i] = arg
			}
		}
	} else {
		cmdArgs = command
	}

	cmd := exec.CommandContext(ctx, cmdArgs[0], cmdArgs[1:]...)
	cmd.Dir = s.WorkspaceRoot
	var stdoutBuf, stderrBuf bytes.Buffer
	cmd.Stdout = &stdoutBuf
	cmd.Stderr = &stderrBuf

	runErr := cmd.Run()
	exitCode := 0
	if runErr != nil {
		var ee *exec.ExitError
		if errors.As(runErr, &ee) {
			exitCode = ee.ExitCode()
		} else {
			exitCode = 1
		}
	}

	return GuardCodexExecutionResult{
		ExitCode: exitCode,
		Stdout:   stdoutBuf.String(),
		Stderr:   stderrBuf.String(),
		Proxied:  s.SandboxActive,
	}, runErr
}
