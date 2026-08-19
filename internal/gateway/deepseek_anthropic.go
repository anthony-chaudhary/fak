package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// deepseek_anthropic.go — the DeepSeek ANTHROPIC-COMPATIBLE route profile and its
// compatibility fences (#3010, under the DeepSeek V4 support program #3006).
//
// DeepSeek documents an Anthropic-compatible API at
// https://api.deepseek.com/anthropic (source: https://api-docs.deepseek.com/guides/anthropic_api)
// so a Claude Code-style client can front DeepSeek through the SAME `/anthropic`
// base + `/v1/messages` composition fak already speaks to real Anthropic. This file
// is the load-bearing wire that decides whether that fronting is SAFE — it does NOT
// change the default Anthropic provider behavior for real Claude (that stays exactly
// as internal/agent's anthropicAdapter composes it); it adds a DeepSeek profile on
// top of the same composition and fences the documented compatibility GAPS so fak
// fails loud instead of silently shipping a block the route drops.
//
// The three gaps this route MUST fence (all from the DeepSeek compatibility table,
// https://api-docs.deepseek.com/guides/anthropic_api):
//  1. Content blocks: images / documents / redacted-thinking / code-execution / MCP
//     blocks are NOT all supported. Sending one silently would lose data. The fence
//     refuses with a CLOSED reason (fail loud), never drops silently.
//  2. `cache_control` is IGNORED by DeepSeek. So a cache_control breakpoint on this
//     route is a provider no-op, NOT a fak-managed prompt-cache saving — booking it
//     as one would misprice value accounting. IsFakManagedCacheSaving reports false.
//  3. `thinking` is supported but `budget_tokens` is IGNORED. The request builder
//     enables thinking WITHOUT a budget so no caller is misled into thinking the
//     budget bound the response.
//
// PROVENANCE: this profile OBSERVES a third-party route. Everything it reports about
// DeepSeek's behavior (ignored cache_control, model fallback) is PROVIDER-OBSERVED,
// never fak-authored — the same OBSERVED-vs-WITNESSED discipline deepseek_pricing.go
// applies to the billing counters.

// DeepSeekAnthropicMessagesPath is the endpoint SUFFIX the Anthropic Messages wire
// appends to a base URL. It MUST match internal/agent's anthropicAdapter.Endpoint
// (`joinEndpoint(baseURL, "/v1/messages")`) so the DeepSeek route composes byte-for-byte
// the way the real-Anthropic route does; deepseek_anthropic_test.go pins the parity.
const DeepSeekAnthropicMessagesPath = "/v1/messages"

// DeepSeekAnthropicVersion is the anthropic-version header the DeepSeek route sends,
// matching the value the real-Anthropic adapter uses. DeepSeek's Anthropic-compatible
// endpoint accepts the same versioned Messages wire.
const DeepSeekAnthropicVersion = "2023-06-01"

// closed-vocabulary reasons for a REFUSED unsupported content block. A refusal names
// exactly one of these — never free text — so a caller (or a test) can branch on the
// reason and an operator sees a stable, greppable token instead of prose.
const (
	DeepSeekAnthropicUnsupportedImage            = "DEEPSEEK_ANTHROPIC_UNSUPPORTED_IMAGE"
	DeepSeekAnthropicUnsupportedDocument         = "DEEPSEEK_ANTHROPIC_UNSUPPORTED_DOCUMENT"
	DeepSeekAnthropicUnsupportedRedactedThinking = "DEEPSEEK_ANTHROPIC_UNSUPPORTED_REDACTED_THINKING"
	DeepSeekAnthropicUnsupportedCodeExecution    = "DEEPSEEK_ANTHROPIC_UNSUPPORTED_CODE_EXECUTION"
	DeepSeekAnthropicUnsupportedMCP              = "DEEPSEEK_ANTHROPIC_UNSUPPORTED_MCP"
)

// deepSeekAnthropicUnsupportedBlocks maps each Anthropic-wire content-block `type`
// DeepSeek does NOT support to its closed refusal reason. The keys are the documented
// unsupported set (image, document, redacted thinking, code execution, MCP); a
// server_tool_use block is DeepSeek's code-execution surface, and mcp_tool_use /
// mcp_tool_result are the MCP content blocks. Everything NOT in this map is treated
// as supported (text / tool_use / tool_result / thinking), so a genuinely new block
// type fails OPEN as supported rather than being wrongly refused — the fence only
// speaks about blocks the compatibility table names.
var deepSeekAnthropicUnsupportedBlocks = map[string]string{
	"image":             DeepSeekAnthropicUnsupportedImage,
	"document":          DeepSeekAnthropicUnsupportedDocument,
	"redacted_thinking": DeepSeekAnthropicUnsupportedRedactedThinking,
	"code_execution":    DeepSeekAnthropicUnsupportedCodeExecution,
	"server_tool_use":   DeepSeekAnthropicUnsupportedCodeExecution,
	"mcp_tool_use":      DeepSeekAnthropicUnsupportedMCP,
	"mcp_tool_result":   DeepSeekAnthropicUnsupportedMCP,
}

// DeepSeekAnthropicUnsupportedBlockReason reports whether a content-block `type` is
// unsupported on the DeepSeek Anthropic route, and if so its closed refusal reason.
// The type is normalized (trim + lower) so wire casing does not slip a block past the
// fence.
func DeepSeekAnthropicUnsupportedBlockReason(blockType string) (reason string, unsupported bool) {
	r, ok := deepSeekAnthropicUnsupportedBlocks[strings.ToLower(strings.TrimSpace(blockType))]
	return r, ok
}

// DeepSeekAnthropicModelSource labels HOW a model id was resolved, so a ledger can tell
// an operator-chosen direct id apart from a Claude-name mapping fak applied.
type DeepSeekAnthropicModelSource string

const (
	// DeepSeekAnthropicModelDirect: the caller already named a real DeepSeek V4 model
	// id (deepseek-v4-pro / deepseek-v4-flash). This is the PREFERRED form.
	DeepSeekAnthropicModelDirect DeepSeekAnthropicModelSource = "direct"
	// DeepSeekAnthropicModelClaudeMapped: the caller named a Claude family (opus /
	// sonnet / haiku) and fak applied its OWN explicit mapping. This is fak's policy,
	// NOT reliance on DeepSeek's server-side "unknown model → Flash" fallback.
	DeepSeekAnthropicModelClaudeMapped DeepSeekAnthropicModelSource = "claude-name-mapped:deepseek-coding-agents-guide"
)

// ResolveDeepSeekAnthropicModel turns the model id a Claude Code-style client sent into
// the concrete DeepSeek V4 id fak will put on the wire. It PREFERS a direct
// deepseek-v4-* id; for a Claude family name it applies fak's OWN explicit mapping
// (Opus/Sonnet → Pro, Haiku → Flash) taken from DeepSeek's coding-agents guide
// (https://api-docs.deepseek.com/guides/coding_agents), so the id on the wire is one
// fak chose rather than leaving DeepSeek to silently coerce an unknown name to Flash.
//
// The DeepSeek Claude Code guide sets ANTHROPIC_MODEL=deepseek-v4-pro[1m]; the trailing
// `[1m]` selects the 1M-context variant. It is stripped for the routing decision (both
// variants map to the same base id) but reported via has1M so the caller can preserve
// the marker on the wire if it wants the extended window.
//
// It FAILS LOUD (err != nil) on any other name rather than defaulting to Flash — the
// issue is explicit that DeepSeek's unsupported-model fallback is NOT fak's routing
// policy.
func ResolveDeepSeekAnthropicModel(requested string) (model string, source DeepSeekAnthropicModelSource, has1M bool, err error) {
	norm := strings.ToLower(strings.TrimSpace(requested))
	if strings.HasSuffix(norm, "[1m]") {
		has1M = true
		norm = strings.TrimSpace(strings.TrimSuffix(norm, "[1m]"))
	}
	switch norm {
	case "":
		return "", "", has1M, fmt.Errorf("deepseek-anthropic: empty model id; name deepseek-v4-pro or deepseek-v4-flash explicitly")
	case modelroute.DeepSeekV4ProModel:
		return modelroute.DeepSeekV4ProModel, DeepSeekAnthropicModelDirect, has1M, nil
	case modelroute.DeepSeekV4FlashModel:
		return modelroute.DeepSeekV4FlashModel, DeepSeekAnthropicModelDirect, has1M, nil
	}
	switch {
	case strings.Contains(norm, "opus"), strings.Contains(norm, "sonnet"):
		return modelroute.DeepSeekV4ProModel, DeepSeekAnthropicModelClaudeMapped, has1M, nil
	case strings.Contains(norm, "haiku"):
		return modelroute.DeepSeekV4FlashModel, DeepSeekAnthropicModelClaudeMapped, has1M, nil
	}
	return "", "", has1M, fmt.Errorf("deepseek-anthropic: unroutable model %q; name deepseek-v4-pro/deepseek-v4-flash or a Claude opus/sonnet/haiku family (fak will not rely on DeepSeek's silent unknown-model→Flash fallback)", requested)
}

// DeepSeek's Anthropic-compatible effort control (`output_config.effort`) accepts these
// two levels for a coding agent; DeepSeek's guide recommends "max" for Claude Code.
const (
	DeepSeekAnthropicEffortHigh = "high"
	DeepSeekAnthropicEffortMax  = "max"
)

// deepSeekAnthropicWireEfforts is the effort vocabulary this route can put on the wire —
// the clamp target set for ResolveDeepSeekAnthropicEffort.
var deepSeekAnthropicWireEfforts = []string{DeepSeekAnthropicEffortHigh, DeepSeekAnthropicEffortMax}

// ResolveDeepSeekAnthropicEffort maps a CANONICAL fak effort tier onto the two levels this
// route speaks, SATURATING to the nearest supported tier rather than erroring a request the
// provider can serve one rung away (#4069). fak's canonical ladder carries rungs DeepSeek
// simply does not name (none/minimal/low/medium/xhigh), and refusing them made a caller's
// perfectly ordinary `--effort low` fail the whole request; it now runs at high, and xhigh
// runs at max (the tie breaks upward — see modelroute.SaturateEffort).
//
// The TYPO FENCE this replaced is preserved exactly where it earns its keep. Saturation is
// only ever applied to a tier fak can PLACE on its ladder; a token off the ladder entirely
// ("extreem") is still refused, because it carries no ordering to clamp with and is far
// likelier a mistake than an intent. So the route degrades gracefully across a known
// vocabulary gap while still failing loud on a genuinely unknown one.
//
// Returns ("", nil) for a blank effort: the control stays unset, which is not an error.
func ResolveDeepSeekAnthropicEffort(effort string) (string, error) {
	if strings.TrimSpace(effort) == "" {
		return "", nil
	}
	if tier := modelroute.SaturateEffort(effort, deepSeekAnthropicWireEfforts); tier != "" {
		return tier, nil
	}
	return "", fmt.Errorf("deepseek-anthropic: unknown output_config.effort %q; want a canonical effort tier (none/minimal/low/medium/high/xhigh/max), which is saturated to %q or %q", effort, DeepSeekAnthropicEffortHigh, DeepSeekAnthropicEffortMax)
}

// ValidateDeepSeekAnthropicEffort reports whether an effort can reach the wire at all: nil
// for a blank control and for any canonical tier (which ResolveDeepSeekAnthropicEffort
// saturates into range), an error only for a tier off fak's ladder. Callers that need the
// tier actually sent should use ResolveDeepSeekAnthropicEffort instead.
func ValidateDeepSeekAnthropicEffort(effort string) error {
	_, err := ResolveDeepSeekAnthropicEffort(effort)
	return err
}

// DeepSeekAnthropicCacheControlProvenance is the provenance label for a cache_control
// breakpoint sent to the DeepSeek Anthropic route: DeepSeek IGNORES cache_control, so
// it is a provider no-op, never a fak-authored saving.
const DeepSeekAnthropicCacheControlProvenance = "provider-ignored"

// IsFakManagedCacheSaving reports whether a cache_control breakpoint on the DeepSeek
// Anthropic route should be booked as a fak-managed prompt-cache SAVING. It is always
// false: DeepSeek ignores cache_control, so any implied discount is provider-observed
// economics (DeepSeek's own default context caching, priced by deepseek_pricing.go),
// not something a fak mechanism shaped. This is the fence against reporting an ignored
// cache_control as a fak cache effect.
func IsFakManagedCacheSaving() bool { return false }

// DeepSeekAnthropicProfile is a launch preset that fronts DeepSeek through the
// Anthropic-compatible route. The zero value is not usable; use
// NewDeepSeekAnthropicProfile so BaseURL defaults correctly.
type DeepSeekAnthropicProfile struct {
	// BaseURL is the Anthropic-compatible base; defaults to
	// modelroute.DeepSeekAnthropicBaseURL. The Messages path is appended by
	// MessagesURL, exactly as the real-Anthropic adapter composes it.
	BaseURL string
	// APIKey is the DeepSeek API key (the operator-owned DEEPSEEK_API_KEY secret). It
	// rides as x-api-key and is NEVER placed in the URL, model id, or body.
	APIKey string
	// Model is the DeepSeek V4 id to put on the wire (already resolved via
	// ResolveDeepSeekAnthropicModel).
	Model string
	// Effort, if set, is emitted as output_config.effort. It holds a tier DeepSeek names
	// ("high"/"max"); NewDeepSeekAnthropicProfile has already saturated any other
	// canonical tier onto that set (#4069).
	Effort string
}

// NewDeepSeekAnthropicProfile builds a profile, defaulting BaseURL to DeepSeek's
// documented Anthropic-compatible root and resolving the model id (failing loud on an
// unroutable name). The [1m] extended-window marker, if the caller sent one, is
// preserved on the wire model id.
func NewDeepSeekAnthropicProfile(apiKey, requestedModel, effort string) (DeepSeekAnthropicProfile, error) {
	model, _, has1M, err := ResolveDeepSeekAnthropicModel(requestedModel)
	if err != nil {
		return DeepSeekAnthropicProfile{}, err
	}
	wireEffort, err := ResolveDeepSeekAnthropicEffort(effort)
	if err != nil {
		return DeepSeekAnthropicProfile{}, err
	}
	wireModel := model
	if has1M {
		wireModel += "[1m]"
	}
	return DeepSeekAnthropicProfile{
		BaseURL: modelroute.DeepSeekAnthropicBaseURL,
		APIKey:  apiKey,
		Model:   wireModel,
		Effort:  wireEffort,
	}, nil
}

// MessagesURL composes the concrete POST target the same way the real-Anthropic route
// does: the base URL with a trailing slash trimmed, plus the Messages path. With the
// default base that is https://api.deepseek.com/anthropic/v1/messages.
func (p DeepSeekAnthropicProfile) MessagesURL() string {
	base := p.BaseURL
	if base == "" {
		base = modelroute.DeepSeekAnthropicBaseURL
	}
	return strings.TrimRight(base, "/") + DeepSeekAnthropicMessagesPath
}

// headers builds the request headers. The API key rides ONLY as x-api-key (DeepSeek's
// documented scheme for the Anthropic-compatible route); it never lands in the URL or
// body, so a request/response log cannot leak it from those surfaces.
func (p DeepSeekAnthropicProfile) headers() map[string]string {
	h := map[string]string{
		"Content-Type":      "application/json",
		"anthropic-version": DeepSeekAnthropicVersion,
	}
	if p.APIKey != "" {
		h["x-api-key"] = p.APIKey
	}
	return h
}

// DeepSeekAnthropicBlock is one Anthropic-wire content block on the REQUEST side. Only
// the `type` is load-bearing for the fence; Text is carried for text blocks.
type DeepSeekAnthropicBlock struct {
	Type string `json:"type"`
	Text string `json:"text,omitempty"`
}

// DeepSeekAnthropicMessage is one role-tagged message with its content blocks.
type DeepSeekAnthropicMessage struct {
	Role    string                   `json:"role"`
	Content []DeepSeekAnthropicBlock `json:"content"`
}

// FenceContentBlocks scans every request block and returns an error naming EVERY
// unsupported block (deduped, sorted, each with its closed reason) so fak fails loud
// rather than shipping an image/document/redacted-thinking/code-exec/MCP block to a
// route that would silently drop it. Returns nil when every block is supported.
func FenceContentBlocks(messages []DeepSeekAnthropicMessage) error {
	seen := map[string]bool{}
	var refusals []string
	for _, m := range messages {
		for _, b := range m.Content {
			if reason, bad := DeepSeekAnthropicUnsupportedBlockReason(b.Type); bad {
				key := strings.ToLower(strings.TrimSpace(b.Type)) + " " + reason
				if !seen[key] {
					seen[key] = true
					refusals = append(refusals, fmt.Sprintf("%s(%s)", strings.ToLower(strings.TrimSpace(b.Type)), reason))
				}
			}
		}
	}
	if len(refusals) == 0 {
		return nil
	}
	sort.Strings(refusals)
	return fmt.Errorf("deepseek-anthropic: unsupported content block(s) refused: %s", strings.Join(refusals, ", "))
}

// deepSeekAnthropicWireRequest is the JSON body posted to the Messages endpoint. It
// enables thinking WITHOUT budget_tokens (DeepSeek ignores it) and emits
// output_config.effort only when set.
type deepSeekAnthropicWireRequest struct {
	Model     string                     `json:"model"`
	MaxTokens int                        `json:"max_tokens"`
	System    string                     `json:"system,omitempty"`
	Messages  []DeepSeekAnthropicMessage `json:"messages"`
	Thinking  *struct {
		Type string `json:"type"`
	} `json:"thinking,omitempty"`
	OutputConfig *struct {
		Effort string `json:"effort"`
	} `json:"output_config,omitempty"`
}

// DeepSeekAnthropicResponseBlock is one block of the assistant response, covering the
// text and thinking blocks DeepSeek returns (thinking is supported on this route).
type DeepSeekAnthropicResponseBlock struct {
	Type      string `json:"type"`
	Text      string `json:"text,omitempty"`
	Thinking  string `json:"thinking,omitempty"`
	Signature string `json:"signature,omitempty"`
}

// DeepSeekAnthropicResponse is the decoded Messages response. Usage carries DeepSeek's
// own top-level cache hit/miss counters (the OBSERVED numbers deepseek_pricing.go
// prices), NOT a fak-managed cache saving.
type DeepSeekAnthropicResponse struct {
	ID         string                           `json:"id"`
	Model      string                           `json:"model"`
	Role       string                           `json:"role"`
	Content    []DeepSeekAnthropicResponseBlock `json:"content"`
	StopReason string                           `json:"stop_reason"`
	Usage      struct {
		InputTokens           int `json:"input_tokens"`
		OutputTokens          int `json:"output_tokens"`
		PromptCacheHitTokens  int `json:"prompt_cache_hit_tokens"`
		PromptCacheMissTokens int `json:"prompt_cache_miss_tokens"`
	} `json:"usage"`
}

// PostMessages builds the request (fencing unsupported blocks and validating effort),
// posts it to MessagesURL with the x-api-key credential, and decodes the response. It
// is the offline-testable seam: pass an *http.Client aimed at an httptest server to
// exercise the full round trip without a network. thinking enables the thinking block;
// budget_tokens is intentionally never sent (DeepSeek ignores it).
func (p DeepSeekAnthropicProfile) PostMessages(ctx context.Context, client *http.Client, system string, messages []DeepSeekAnthropicMessage, maxTokens int, thinking bool) (*DeepSeekAnthropicResponse, error) {
	if strings.TrimSpace(p.Model) == "" {
		return nil, fmt.Errorf("deepseek-anthropic: profile has no model; build it with NewDeepSeekAnthropicProfile")
	}
	// Saturate here too, not just in the constructor: a profile can be hand-built as a
	// struct literal, and the tier that reaches the wire must be one DeepSeek names.
	wireEffort, err := ResolveDeepSeekAnthropicEffort(p.Effort)
	if err != nil {
		return nil, err
	}
	if err := FenceContentBlocks(messages); err != nil {
		return nil, err
	}
	body := deepSeekAnthropicWireRequest{
		Model:     p.Model,
		MaxTokens: maxTokens,
		System:    system,
		Messages:  messages,
	}
	if thinking {
		body.Thinking = &struct {
			Type string `json:"type"`
		}{Type: "enabled"}
	}
	if wireEffort != "" {
		body.OutputConfig = &struct {
			Effort string `json:"effort"`
		}{Effort: wireEffort}
	}
	raw, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("deepseek-anthropic: marshal request: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, p.MessagesURL(), bytes.NewReader(raw))
	if err != nil {
		return nil, fmt.Errorf("deepseek-anthropic: build request: %w", err)
	}
	for k, v := range p.headers() {
		req.Header.Set(k, v)
	}
	agent.ApplyTraceContext(req)
	if client == nil {
		client = &http.Client{Timeout: durEnv("FAK_DEEPSEEK_ANTHROPIC_TIMEOUT_S", 120*time.Second)}
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("deepseek-anthropic: post: %w", err)
	}
	defer resp.Body.Close()
	payload, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("deepseek-anthropic: read response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("deepseek-anthropic: upstream status %d: %s", resp.StatusCode, strings.TrimSpace(string(payload)))
	}
	var out DeepSeekAnthropicResponse
	if err := json.Unmarshal(payload, &out); err != nil {
		return nil, fmt.Errorf("deepseek-anthropic: decode response: %w", err)
	}
	return &out, nil
}

// ClaudeCodeDeepSeekEnv is the machine-readable source for the Claude Code runbook
// (rendered by the human runbook, #3012): the environment a Claude Code-style client
// sets to route through fak into DeepSeek. The API-key entry is the NAME of the
// operator-owned secret env var (DEEPSEEK_API_KEY), never a literal secret, so the
// preset can be printed or committed without leaking a credential. requestedModel is
// resolved (and validated) so an unroutable name is caught before it reaches a runbook.
func ClaudeCodeDeepSeekEnv(requestedModel string) (map[string]string, error) {
	prof, err := NewDeepSeekAnthropicProfile("", requestedModel, "")
	if err != nil {
		return nil, err
	}
	return map[string]string{
		"ANTHROPIC_BASE_URL": modelroute.DeepSeekAnthropicBaseURL,
		"ANTHROPIC_MODEL":    prof.Model,
		"ANTHROPIC_API_KEY":  "$" + modelroute.DeepSeekAPIKeyEnv,
		"ANTHROPIC_AUTH_KEY": modelroute.DeepSeekAPIKeyEnv,
	}, nil
}

// ClaudeCodeDeepSeekRunbook renders the env preset as copy-pasteable shell lines for
// one platform: "unix" (bash/zsh `export`) or "windows" (PowerShell `$env:`). Both
// reference the operator-owned DEEPSEEK_API_KEY by name and never inline a secret.
func ClaudeCodeDeepSeekRunbook(platform, requestedModel string) ([]string, error) {
	env, err := ClaudeCodeDeepSeekEnv(requestedModel)
	if err != nil {
		return nil, err
	}
	keys := make([]string, 0, len(env))
	for k := range env {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	var lines []string
	switch strings.ToLower(strings.TrimSpace(platform)) {
	case "unix", "linux", "macos", "darwin":
		for _, k := range keys {
			lines = append(lines, fmt.Sprintf("export %s=%s", k, env[k]))
		}
	case "windows", "powershell":
		for _, k := range keys {
			// PowerShell reads a shell var with $env:NAME; a $-prefixed value from the
			// preset (the secret reference) is rewritten to PowerShell's own form.
			val := env[k]
			if strings.HasPrefix(val, "$") {
				val = "$env:" + strings.TrimPrefix(val, "$")
			}
			lines = append(lines, fmt.Sprintf("$env:%s = \"%s\"", k, val))
		}
	default:
		return nil, fmt.Errorf("deepseek-anthropic: unknown runbook platform %q; want \"unix\" or \"windows\"", platform)
	}
	return lines, nil
}
