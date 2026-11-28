package main

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/deploymanifest"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/ratelimit"
)

// debugStatsSink returns the per-turn debug sink for `--debug-stats` (#793): a stderr
// line-writer when on, nil (the no-op default) when off. The gateway emits one compact,
// payload-free line per served turn through it.
func debugStatsSink(on bool) func(string, ...any) {
	if !on {
		return nil
	}
	return func(format string, args ...any) {
		fmt.Fprintf(os.Stderr, format+"\n", args...)
	}
}

// streamProgressTimeoutOff is the gateway.Config/HTTPPlanner encoding for "no
// content-progress deadline at all": a NEGATIVE duration, which
// agent.(*HTTPPlanner).streamProgressWindow resolves to a disabled window. It is deliberately
// NOT the zero value — on that field zero means "unconfigured, take
// agent.DefaultStreamProgressTimeout", so the operator's off switch cannot be passed through
// as-is and has to be translated here.
const streamProgressTimeoutOff = -1 * time.Second

// serveStreamProgressTimeout maps --stream-progress-timeout onto that encoding.
//
// The flag spells "off" as 0 because that is what every other serve knob with an off switch
// spells it as (--ctx-view-budget 0, --compact-history-budget 0, --elide-result-bytes 0,
// --assume-session-turns 0, --metrics-snapshot 0): an operator who wants no deadline types a
// zero, never a negative duration. The flag DEFAULTS to agent.DefaultStreamProgressTimeout
// rather than 0, so a zero here is always something the operator typed — the ambiguity the
// raw config field has (where 0 is the unconfigured state) does not exist at the front door.
// Every other value rides through verbatim for streamProgressWindow to band-check.
func serveStreamProgressTimeout(d time.Duration) time.Duration {
	if d <= 0 {
		return streamProgressTimeoutOff
	}
	return d
}

func configureServeToolEngines() {
	// Serve exposes fak_read over MCP even when it is not running the demo agent loop.
	// Register only the confined read miss engine; agent.Configure would also install
	// the demo tool policy and is intentionally not part of serve startup.
	agent.RegisterReadEngine("")
}

// resolveTenantAdmission reads the --tenant-* flags into the three per-tenant gate
// policies (#14453). All-zero input yields all-disabled policies — the historical inert
// path. A negative value fails loud: a typo must not silently disable a fairness gate.
func resolveTenantAdmission(sf *serveFlags) (rate gateway.RateBucketPolicy, inflight gateway.InflightPolicy, tokens gateway.TokenRatePolicy, err error) {
	if *sf.tenantRateRPS < 0 || *sf.tenantRateBurst < 0 {
		return rate, inflight, tokens, errors.New("--tenant-rate-rps and --tenant-rate-burst accept 0 (off) or positive values")
	}
	if *sf.tenantMaxInflight < 0 {
		return rate, inflight, tokens, errors.New("--tenant-max-inflight accepts 0 (off) or a positive value")
	}
	if *sf.tenantTokenWindowTokens < 0 {
		return rate, inflight, tokens, errors.New("--tenant-token-window-tokens accepts 0 (off) or a positive value")
	}
	if *sf.tenantTokenWindow <= 0 {
		return rate, inflight, tokens, errors.New("--tenant-token-window must be a positive duration")
	}
	rate = gateway.RateBucketPolicy{
		RequestsPerSecond: int64(*sf.tenantRateRPS),
		RequestBurst:      int64(*sf.tenantRateBurst),
	}
	inflight = gateway.InflightPolicy{PerPrincipal: *sf.tenantMaxInflight}
	tokens = gateway.TokenRatePolicy{
		Caps:   ratelimit.TokenCaps{MaxTotalTokens: int64(*sf.tenantTokenWindowTokens)},
		Window: *sf.tenantTokenWindow,
	}
	return rate, inflight, tokens, nil
}

// installTenantAdmissionGates wires the armed per-tenant admission gates onto the Server
// (#14453): the request-rate bucket, the in-flight caps, and the token allotment book —
// the same order beginServedAdmission consults them. All-off leaves every gate nil and
// the served request path byte-for-byte historical; each armed gate gets one startup
// message naming its budget.
func installTenantAdmissionGates(sf *serveFlags, srv *gateway.Server) ([]gateway.StartupMessage, error) {
	rate, inflight, tokens, err := resolveTenantAdmission(sf)
	if err != nil {
		return nil, err
	}
	var msgs []gateway.StartupMessage
	if rate.RequestsPerSecond > 0 {
		srv.SetRequestRateGate(gateway.NewRequestRateGate(rate))
		burst := rate.RequestBurst
		if burst <= 0 {
			burst = rate.RequestsPerSecond
		}
		msgs = append(msgs, gateway.StartupMessage{Source: "serve", Kind: "tenant-admission", Level: "info", Text: fmt.Sprintf("tenant admission: request-rate gate armed (%d req/s per tenant, burst %d)", rate.RequestsPerSecond, burst)})
	}
	if inflight.PerPrincipal > 0 {
		srv.SetInflightGate(gateway.NewInflightGate(inflight))
		msgs = append(msgs, gateway.StartupMessage{Source: "serve", Kind: "tenant-admission", Level: "info", Text: fmt.Sprintf("tenant admission: in-flight gate armed (%d concurrent requests per tenant)", inflight.PerPrincipal)})
	}
	if tokens.Caps.MaxTotalTokens > 0 {
		srv.SetPrincipalTokenRates(gateway.NewPrincipalTokenRates(tokens, 0))
		msgs = append(msgs, gateway.StartupMessage{Source: "serve", Kind: "tenant-admission", Level: "info", Text: fmt.Sprintf("tenant admission: token window gate armed (%d tokens per tenant per %s)", tokens.Caps.MaxTotalTokens, tokens.Window)})
	}
	return msgs, nil
}

// ErrSpeculativeModeUnknown marks a --speculative selector (or its FAK_SPECULATIVE /
// SPECULATIVE override) that names a mode outside the closed admission set. It is the
// startup refusal for a misspelled structural selector: such a value must never boot a
// server whose fast decode route is silently unselected.
var ErrSpeculativeModeUnknown = errors.New("unknown speculative mode")

// The closed structural speculative-decode selector values admitted by `fak serve`:
// the historical mtp/ngram routes plus the DFlash2 sidecar route; "" disables.
const (
	serveSpeculativeMTP     = "mtp"
	serveSpeculativeNGram   = "ngram"
	serveSpeculativeDFlash2 = "dflash2"
)

// serveSpeculativeOptOutTokens are the explicit opt-out spellings. They normalize to
// the disabled selector "" so every downstream route sees the same off state.
var serveSpeculativeOptOutTokens = map[string]bool{
	"off": true, "none": true, "false": true, "disabled": true, "0": true,
}

// serveRetainMTPHead reports whether the in-kernel load must keep the MTP head tensors.
// Only the resolved mtp route on a backend that consumes the head (Metal, or resident
// Vulkan/CUDA) keeps it; other routes load the head-free layout.
func serveRetainMTPHead(resolvedMode string, metal bool, backendName string) bool {
	if resolvedMode != serveSpeculativeMTP {
		return false
	}
	name := strings.ToLower(strings.TrimSpace(backendName))
	return metal || name == "vulkan" || name == "cuda"
}

// serveMTPBackendReplayStatus checks whether a resident non-Metal backend selected
// under --speculative mtp advertises Qwen35SequencePrefixReplayPath. When it does not
// (for example, a CUDA backend without prefix replay), callers emit an honest warning
// so MTP is never silently dead.
func serveMTPBackendReplayStatus(resolvedMode string, metal bool, be compute.Backend) (bool, string) {
	if resolvedMode != serveSpeculativeMTP || metal || be == nil {
		return true, ""
	}
	if !serveRetainMTPHead(resolvedMode, false, be.Name()) {
		return true, ""
	}
	if replay, ok := be.(compute.Qwen35SequencePrefixReplayBackend); ok && replay.Qwen35SequencePrefixReplayPath() == compute.Qwen35SequencePrefixReplayPath {
		return true, ""
	}
	return false, fmt.Sprintf("backend %q lacks %s; MTP speculative decoding is inactive and will fall back to ordinary decode", be.Name(), compute.Qwen35SequencePrefixReplayPath)
}

// resolveSpeculativeMode resolves the effective structural speculative-decode selector
// for `fak serve`. The FAK_SPECULATIVE / SPECULATIVE operator overrides win over the
// --speculative flag (FAK_SPECULATIVE first), mirroring the gateway's historical env-OR
// precedence; normalization (lowercase, trim) matches the gateway's own comparison so an
// admitted value reaches every downstream route byte-for-byte. An opt-out token resolves
// to "" (disabled). The winner must land in the closed admission set
// {mtp, ngram, dflash2, empty}; anything else fails startup with
// ErrSpeculativeModeUnknown before any weight load. An admitted value is returned
// verbatim for buildGateway to thread into gateway.Config.SpeculativeMode — including
// "dflash2". The CLI separately refuses that route before acquisition until its
// authenticated target owner is wired; gateway/library owner support is unchanged.
func resolveSpeculativeMode(flagValue string, getenv func(string) string) (string, error) {
	if getenv == nil {
		getenv = os.Getenv
	}
	effective := strings.ToLower(strings.TrimSpace(flagValue))
	for _, name := range []string{"FAK_SPECULATIVE", "SPECULATIVE"} {
		if v := strings.ToLower(strings.TrimSpace(getenv(name))); v != "" {
			effective = v
			break
		}
	}
	if serveSpeculativeOptOutTokens[effective] {
		return "", nil
	}
	switch effective {
	case "", serveSpeculativeMTP, serveSpeculativeNGram, serveSpeculativeDFlash2:
		return effective, nil
	default:
		return "", fmt.Errorf("%w: %q (admitted: mtp, ngram, dflash2, empty; opt-out: off)", ErrSpeculativeModeUnknown, effective)
	}
}

// resolveServeToolGuard maps the --guard flag + FAK_GUARD env onto the gateway
// ToolGuard posture. The DEFAULT is raw pass-through (off): the router is an
// inference pipe with zero interception processing unless the operator arms the
// guard. An explicitly passed --guard wins over the env; an explicit
// FAK_GUARD wins over the flag default; an unparsable FAK_GUARD fails loud
// (a misconfigured security boundary never silently resolves).
func resolveServeToolGuard(guard bool, explicit map[string]bool, getenv func(string) string) (string, error) {
	if getenv == nil {
		getenv = os.Getenv
	}
	if explicit != nil && explicit["guard"] {
		if guard {
			return gateway.ToolGuardOn, nil
		}
		return gateway.ToolGuardOff, nil
	}
	if raw := strings.TrimSpace(getenv("FAK_GUARD")); raw != "" {
		mode, err := gateway.ParseToolGuardMode(raw)
		if err != nil {
			return "", fmt.Errorf("FAK_GUARD: %v", err)
		}
		return mode, nil
	}
	return gateway.ToolGuardOff, nil
}

// resolveServeCtxPaging maps the --ctx-paging flag + FAK_CTX_PAGING env onto the
// gateway CtxPaging posture. The DEFAULT is off: the serving router does not page
// tool results into {"_paged":true} restore stubs (the 2026-11-10 hosting-
// simplification flip). An explicitly passed --ctx-paging wins over the env; an
// explicit FAK_CTX_PAGING wins over the flag default; an unparsable value fails
// loud (a context rewriter never silently resolves).
func resolveServeCtxPaging(paging string, explicit map[string]bool, getenv func(string) string) (string, error) {
	if getenv == nil {
		getenv = os.Getenv
	}
	if explicit != nil && explicit["ctx-paging"] {
		return gateway.ParseCtxPagingMode(paging)
	}
	if raw := strings.TrimSpace(getenv("FAK_CTX_PAGING")); raw != "" {
		mode, err := gateway.ParseCtxPagingMode(raw)
		if err != nil {
			return "", fmt.Errorf("FAK_CTX_PAGING: %v", err)
		}
		return mode, nil
	}
	return gateway.DefaultServeCtxPaging, nil
}

// resolveServeChatToolResultElide maps the --chat-elide flag + FAK_CHAT_ELIDE env
// onto the gateway ChatToolResultElide posture. The DEFAULT is off: the serving
// router does not rewrite old tool results and does not auto-advertise the
// restore tool (the same hosting-simplification flip). Precedence matches the
// tool guard: explicit flag > explicit env > the off default; unparsable fails loud.
func resolveServeChatToolResultElide(elide bool, explicit map[string]bool, getenv func(string) string) (string, error) {
	if getenv == nil {
		getenv = os.Getenv
	}
	if explicit != nil && explicit["chat-elide"] {
		if elide {
			return gateway.ChatToolResultElideOn, nil
		}
		return gateway.ChatToolResultElideOff, nil
	}
	if raw := strings.TrimSpace(getenv("FAK_CHAT_ELIDE")); raw != "" {
		mode, err := gateway.ParseChatToolResultElideMode(raw)
		if err != nil {
			return "", fmt.Errorf("FAK_CHAT_ELIDE: %v", err)
		}
		return mode, nil
	}
	return gateway.DefaultServeChatToolResultElide, nil
}

// serveDFlash2SidecarDecision resolves the per-load DFlash2 sidecar retention
// for a structural selector: only a dflash2-resolved load with a resident
// in-kernel model AND an operator-pinned FAK_DFLASH2_SIDECAR path retains the
// artifact. Every other combination leaves the route unretained — the
// gateway's fail-loud target-owner validation keeps governing admission, and
// no sidecar is opened for a load that will not draft.
func serveDFlash2SidecarDecision(resolvedMode string, inKernelLoaded bool, getenv func(string) string) (string, bool) {
	if resolvedMode != serveSpeculativeDFlash2 || !inKernelLoaded {
		return "", false
	}
	path := strings.TrimSpace(getenv("FAK_DFLASH2_SIDECAR"))
	if path == "" {
		return "", false
	}
	return filepath.Clean(path), true
}

func serveNativePlannerConfigWithContext(sf *serveFlags, resolvedContextTokens int) agent.InKernelPlannerConfig {
	cfg := serveNativePlannerConfig(sf)
	cfg.ContextTokens = resolvedContextTokens
	// Spend part of the resolved window on a compaction shed-line. Without this the
	// native path never shrinks: ApplyPromptShrink short-circuits on a zero budget
	// and an over-window transcript is refused with HTTP 400 context_length_exceeded
	// instead of being compacted and served. An explicit operator budget still wins.
	if cfg.CompactHistoryBudget <= 0 {
		cfg.CompactHistoryBudget = agent.DeriveCompactHistoryBudget(resolvedContextTokens, 0)
	}
	return cfg
}

func validateServeMaxTotalTokens(sf *serveFlags, admissionTokenBudget int) error {
	maxTokens := sf.effectiveMaxTotalTokens()
	if maxTokens > 0 && admissionTokenBudget > 0 && maxTokens > admissionTokenBudget {
		return fmt.Errorf("max_total_tokens (%d) exceeds admission token budget (%d); decrease --max-batch-prefill-tokens or --max-total-tokens", maxTokens, admissionTokenBudget)
	}
	return nil
}

func effectiveServeConfigWithQwen38Runtime(sf *serveFlags, m deploymanifest.Manifest, hasManifest bool, explicit map[string]bool) effectiveServeConfigReport {
	report := effectiveServeConfig(sf, m, hasManifest, explicit)
	source := "built-in"
	if explicit["qwen38-runtime"] {
		source = "flag"
	}
	report.Values["qwen38_runtime"] = effectiveConfigValue{Value: *sf.qwen38Runtime, Source: source}
	report.Values["qwen38_runtime_identity"] = effectiveConfigValue{Value: qwen38RuntimeIdentity(*sf.qwen38Runtime), Source: source}
	return report
}

// isServeVulkan reports whether the shared native selector resolved a usable
// Vulkan backend. Proxy modes remain lease-free because they delegate model
// residency. Keeping admission on the same resolver prevents automatic or
// FAK_BACKEND-selected Vulkan from loading local weights without its GPU lease.
func isServeVulkan(sf *serveFlags) bool {
	if sf == nil || sf.backendName == nil {
		return false
	}
	if sf.baseURL != nil && *sf.baseURL != "" {
		return false
	}
	backend, err := resolveServeChatBackend(*sf.backendName)
	return err == nil && backend != nil && backend.Name() == "vulkan"
}
