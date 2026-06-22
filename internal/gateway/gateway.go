// Package gateway is the kernel-adjudicated wire: it fronts the fak kernel over
// MCP (newline-delimited JSON-RPC) and an OpenAI-compatible HTTP surface so an
// agent written in ANY language can route its tool calls through the in-process
// syscall boundary WITHOUT writing Go.
//
// Direction (DIRECTION.md). The gateway is Go and sits ON the request path — it
// adjudicates every call through the existing abi.Kernel. That is in-direction:
// the typed core does the deciding. It adds NO non-Go surface in-tree; the non-Go
// CLIENT lives in the adopter's repo. Everything crossing the wire is untrusted
// input that typed Go re-validates before it reaches the kernel — the same
// posture the policy loader takes toward a manifest and the kernel takes toward a
// tool result. A wire client NEVER supplies an abi.Ref (a content-addressed CAS
// handle); it supplies raw argument bytes and the gateway mints a tainted,
// agent-scoped Ref itself, so the IFC/secret/self-modify rungs stay armed.
//
// The three operations, all funnelling into one adjudication helper:
//
//	fak_adjudicate  — k.Decide only (no dispatch, no pending state): the pre-exec
//	                  verdict a client-side executor asks for BEFORE running a tool.
//	fak_syscall     — k.Syscall (adjudicate + dispatch to the registered engine +
//	                  context-MMU admit): the self-contained / CI / demo path.
//	/v1/chat/completions — an adjudication PROXY: it forwards the chat to an
//	                  upstream model, then runs each PROPOSED tool_call through
//	                  k.Decide, dropping denied calls and rewriting grammar-repaired
//	                  ones before the caller ever sees them. It does NOT execute the
//	                  client's tools (the client does).
package gateway

import (
	"context"
	"errors"
	"fmt"
	"log"
	"net"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/cachemeta"
	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/enginecache"
	"github.com/anthony-chaudhary/fak/internal/kernel"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/recall"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
	"github.com/anthony-chaudhary/fak/internal/vdso"
)

// Config configures a gateway Server. The zero value is not valid — use New,
// which fills defaults and validates against the registered ABI.
type Config struct {
	// EngineID selects the registered engine fak_syscall dispatches an ALLOWED
	// call to (default "inkernel": the model fused into the kernel — a real
	// in-kernel decode, synthetic checkpoint unless FAK_MODEL_DIR names an export).
	// Validated against abi.EngineIDs().
	EngineID string
	// Model is advertised by GET /v1/models and used as the upstream model id.
	Model string
	// BaseURL, if non-empty, makes /v1/chat/completions a live proxy in front of
	// that provider endpoint. Empty => the deterministic offline mock planner
	// (CI / drop-in testing).
	BaseURL string
	// Provider selects the upstream transcript wire when BaseURL is set
	// (openai, anthropic, gemini, xai). Empty keeps the OpenAI-compatible default.
	Provider string
	// APIKey is the bearer token sent to the upstream model (proxy mode only).
	APIKey string
	// EngineCacheEngine optionally selects a self-hosted serving-engine cache reset
	// endpoint to call when inbound tool-result admission quarantines bytes before
	// an upstream proxy turn. Empty disables remote cache reset.
	EngineCacheEngine string
	// EngineCacheBaseURL is the serving engine's control/base URL. Empty defaults
	// to BaseURL when EngineCacheEngine is set.
	EngineCacheBaseURL string
	// EngineCacheAdminKey is sent as a bearer token to the serving-engine reset
	// endpoint. Empty sends no Authorization header.
	EngineCacheAdminKey string
	// EngineCacheIdleTimeout is SGLang's optional /flush_cache idle timeout.
	EngineCacheIdleTimeout time.Duration
	// EngineCacheRequireExactSpan refuses a quarantined proxy turn when the
	// selected engine exposes only whole-prefix-cache reset.
	EngineCacheRequireExactSpan bool
	// InKernelModel, when non-nil along with Tokenizer and an empty BaseURL, makes
	// /v1/chat/completions AND /v1/messages serve the in-kernel model directly (real
	// ChatML chat via internal/tokenizer) instead of the offline MockPlanner. Set by
	// `fak serve --gguf …` (no --base-url); Tokenizer is the explicit --tokenizer or the
	// GGUF's embedded tokenizer. Proxy mode (BaseURL set) wins.
	InKernelModel *model.Model
	// Tokenizer is the BPE tokenizer the in-kernel chat planner encodes ChatML with.
	Tokenizer *tokenizer.Tokenizer
	// InKernelQ4K flags the preloaded model as resident-Q4_K so the chat decode runs
	// Session.Q4K (the SDOT int8 GEMV path, FAK_Q4K at boot).
	InKernelQ4K bool
	// Backend, when non-nil, makes the in-kernel chat planner decode through the
	// compute HAL device backend (e.g. CUDA) instead of the CPU session. Set by
	// `fak serve --backend <name>`. Ignored unless InKernelModel+Tokenizer are set
	// (proxy mode and the mock planner do not touch a device).
	Backend compute.Backend
	// RequireKey, if non-empty, is the bearer token the gateway REQUIRES on every
	// request (except /healthz). Empty => no auth (drop-in compatible, loopback).
	RequireKey string
	// VDSO toggles the kernel's dedup fast path for fak_syscall.
	VDSO bool
	// Invalidation selects the process-global vDSO tier-2 invalidation granularity for
	// the live fleet sharing this gateway: "global" (v0.1 full-flush, the default),
	// "namespace" (a write strands only its resource class), or "resource" (only the
	// named entity). Parsed by vdso.ParseGranularity; an unknown value fails New().
	Invalidation string
	// Version is surfaced in the MCP serverInfo handshake (default "dev").
	Version string
	// ReloadPolicy reloads the process policy floor in-place. Nil disables the
	// /v1/fak/policy/reload route.
	ReloadPolicy PolicyReloadFunc
	// ResetTrace clears one trace's process-local taint ledger mark. Nil disables
	// the /v1/fak/trace/reset route.
	ResetTrace TraceResetFunc
	// ObserveTrace reports one trace's current IFC taint high-water mark (the
	// read-only complement of ResetTrace). Nil disables the GET /v1/fak/trace/{id}
	// observe route.
	ObserveTrace TraceObserveFunc
	// Logf is the structured log sink (default: stderr). MCP-over-stdio sets this
	// to stderr so protocol bytes on stdout are never corrupted.
	Logf func(format string, args ...any)
	// StartTime is the process-start instant the boot timeline is measured from. The
	// zero value defaults to time.Now() at New — set it from the host CLI's first
	// statement so phases timed BEFORE New (policy load, flag parse) are accounted
	// for in fak_gateway_time_to_ready_seconds.
	StartTime time.Time
	// StartupPhases are boot phases the host timed before calling New (e.g.
	// "policy-load"). The gateway appends the phases it can time itself
	// ("planner-init", "vdso-config", "kernel-init") and exposes the union as
	// fak_gateway_startup_phase_duration_seconds.
	StartupPhases []StartupPhase
}

// PolicyReloadFunc is injected by the host CLI so the gateway can expose a reload
// route without importing policy/adjudicator/ifc internals.
type PolicyReloadFunc func(context.Context) (PolicyReloadResponse, error)

// PolicyReloadResponse is the wire result of POST /v1/fak/policy/reload.
type PolicyReloadResponse struct {
	Reloaded bool   `json:"reloaded"`
	Source   string `json:"source,omitempty"`
	Summary  string `json:"summary,omitempty"`
}

// TraceResetFunc is injected by the host CLI so the gateway can reset live IFC
// trace state without importing IFC internals.
type TraceResetFunc func(context.Context, string) error

// TraceObserveFunc is injected by the host CLI so the gateway can read one trace's
// live IFC taint high-water mark without importing IFC internals. It returns the
// taint level name ("trusted"|"tainted"|"quarantined") and whether that level is
// dangerous to feed a sensitive sink (Tainted or worse). An unseen trace reads
// "trusted" — the ledger's own clean default.
type TraceObserveFunc func(context.Context, string) (level string, dangerous bool)

// TraceObserveResponse is the wire result of GET /v1/fak/trace/{trace_id}: the
// current IFC taint high-water mark for a live/recent served session.
type TraceObserveResponse struct {
	TraceID   string `json:"trace_id"`
	Taint     string `json:"taint"`
	Dangerous bool   `json:"dangerous"`
}

// TraceResetRequest is the body of POST /v1/fak/trace/reset.
type TraceResetRequest struct {
	TraceID string `json:"trace_id"`
}

// TraceResetResponse is the wire result of POST /v1/fak/trace/reset.
type TraceResetResponse struct {
	Reset   bool   `json:"reset"`
	TraceID string `json:"trace_id"`
}

// Server is a configured, ready-to-serve gateway. Construct with New; serve with
// Handler()/ListenAndServe (HTTP) or ServeStdio (MCP over stdin/stdout).
type Server struct {
	k            *kernel.Kernel
	engineID     string
	model        string
	requireKey   string
	version      string
	logf         func(format string, args ...any)
	feed         *coherenceFeed // the cross-agent "what changed" feed (vdso coherence bus)
	metrics      *gatewayMetrics
	traceSeq     uint64 // mints a non-empty TraceID when the wire omits one (atomic)
	reloadPolicy PolicyReloadFunc
	resetTrace   TraceResetFunc
	observeTrace TraceObserveFunc

	// startup is the one-time boot timeline (start -> ready, per-phase costs),
	// exposed as fak_gateway_startup_* gauges. See startup.go.
	startup *startupProfile
	// modelLoad is the optional boot-time weight-load breakdown set by the host via
	// SetModelLoadProfile when it eagerly loads a model (fak serve --gguf). nil
	// suppresses every fak_model_load_* metric. Guarded by modelLoadMu.
	modelLoadMu sync.Mutex
	modelLoad   *ModelLoadProfile

	// planner generates the assistant turn for the /v1/chat/completions proxy. A
	// live HTTPPlanner when BaseURL is set, else the offline MockPlanner. Settable
	// in-package for tests.
	planner     agent.Planner
	engineCache *enginecache.Client

	// cacheStream is the unified cachemeta.Entry observability fold (fak_cache_*).
	// New subscribes it to the process-global vDSO's live tier-2 cache-event sink so
	// every fill/hit/evict/revoke on the strongest local cache is rendered on
	// /metrics; Close detaches the sink. nil suppresses the family. See metrics.go.
	cacheStream *cachemeta.StreamMetrics
}

// New builds a Server. It validates that the ABI is wired (a resolver is
// registered — i.e. internal/registrations was imported) and that EngineID names
// a registered engine. It fails loud rather than degrade to a permissive default.
func New(cfg Config) (*Server, error) {
	if abi.ActiveResolver() == nil {
		return nil, errors.New("gateway: no Ref resolver registered (blank-import internal/registrations before New)")
	}
	engineID := cfg.EngineID
	if engineID == "" {
		engineID = "inkernel"
	}
	if !engineRegistered(engineID) {
		return nil, fmt.Errorf("gateway: engine %q is not registered (have: %s)", engineID, strings.Join(abi.EngineIDs(), ", "))
	}
	model := cfg.Model
	if model == "" {
		model = engineID
	}
	logf := cfg.Logf
	if logf == nil {
		logf = func(format string, args ...any) { log.Printf(format, args...) }
	}
	version := cfg.Version
	if version == "" {
		version = "dev"
	}

	// Boot timeline: start from the host-supplied process-start instant (so phases
	// the CLI timed before New are on the same clock) and append each New-internal
	// phase as we complete it.
	startup := newStartupProfile(cfg.StartTime)
	for _, ph := range cfg.StartupPhases {
		startup.phase(ph.Name, ph.Dur)
	}

	var planner agent.Planner
	t := time.Now()
	switch {
	case cfg.BaseURL != "":
		p, err := agent.NewProviderHTTPPlanner(cfg.Provider, cfg.BaseURL, model, cfg.APIKey)
		if err != nil {
			return nil, err
		}
		planner = p
	case cfg.InKernelModel != nil && cfg.Tokenizer != nil:
		// Serve the model fused into the kernel as the chat backend on BOTH
		// /v1/chat/completions and /v1/messages (they share s.planner.Complete):
		// real ChatML chat via internal/tokenizer, the cmd/fakchat recipe factored
		// into a Planner. Falls through to MockPlanner if the host didn't preload.
		planner = agent.NewInKernelPlanner(cfg.InKernelModel, cfg.Tokenizer, model, cfg.InKernelQ4K, cfg.Backend)
	default:
		// No upstream (--base-url) and no in-kernel model (--gguf/FAK_MODEL_DIR): the
		// chat surface silently fell back to the deterministic offline mock. Warn
		// LOUDLY so an operator never mistakes scripted demo text for real model
		// output — the /healthz planner:"mock" field carries the same signal to a
		// liveness probe.
		logf("gateway: WARNING — POST /v1/chat/completions is served by the DETERMINISTIC MOCK planner: responses are SCRIPTED, not model output. Pass --base-url (proxy a real provider) or --gguf/FAK_MODEL_DIR (serve the in-kernel model) to disable the mock.")
		planner = agent.NewMockPlanner(model)
	}
	startup.phase("planner-init", time.Since(t))

	remoteCache, err := newEngineCacheClient(cfg)
	if err != nil {
		return nil, err
	}

	// Select the live fleet's tier-2 invalidation granularity (process-global vDSO).
	// Fail loud on an unknown name rather than silently degrading to a full flush.
	t = time.Now()
	if g, ok := vdso.ParseGranularity(cfg.Invalidation); ok {
		vdso.Default.SetGranularity(g)
	} else {
		return nil, fmt.Errorf("gateway: unknown invalidation granularity %q (want global|namespace|resource)", cfg.Invalidation)
	}
	startup.phase("vdso-config", time.Since(t))

	t = time.Now()
	k := kernel.New(engineID)
	k.SetVDSO(cfg.VDSO)
	startup.phase("kernel-init", time.Since(t))

	// Unified cache-stream observability: subscribe the live tier-2 cache-event sink
	// of the process-global vDSO (the SAME instance writeVDSOMetrics reads Stats from)
	// so every fill/hit/evict/revoke folds into the fak_cache_* family. The sink fires
	// OUTSIDE the vDSO lock and Observe only takes its own cheap lock, so it never
	// blocks the hot path. Close detaches it. This is the gateway's single production
	// consumer of the sink (only tests set it otherwise), so owning it is safe.
	cacheStream := cachemeta.NewStreamMetrics()
	vdso.Default.SetCacheEventSink(func(ev vdso.CacheEvent) {
		cacheStream.Observe(string(ev.Kind), ev.Entry)
	})

	return &Server{
		k:            k,
		engineID:     engineID,
		model:        model,
		requireKey:   cfg.RequireKey,
		version:      version,
		logf:         logf,
		reloadPolicy: cfg.ReloadPolicy,
		resetTrace:   cfg.ResetTrace,
		observeTrace: cfg.ObserveTrace,
		startup:      startup,
		planner:      planner,
		engineCache:  remoteCache,
		cacheStream:  cacheStream,
		feed:         newCoherenceFeed(0),
		metrics:      newGatewayMetrics(time.Now()),
	}, nil
}

func newEngineCacheClient(cfg Config) (*enginecache.Client, error) {
	engineName := strings.ToLower(strings.TrimSpace(cfg.EngineCacheEngine))
	baseURL := strings.TrimSpace(cfg.EngineCacheBaseURL)
	if engineName == "" && baseURL == "" && strings.TrimSpace(cfg.EngineCacheAdminKey) == "" && cfg.EngineCacheIdleTimeout == 0 && !cfg.EngineCacheRequireExactSpan {
		return nil, nil
	}
	if engineName == "" {
		return nil, errors.New("gateway: engine cache reset requires EngineCacheEngine (sglang|vllm)")
	}
	if baseURL == "" {
		baseURL = strings.TrimSpace(cfg.BaseURL)
	}
	if baseURL == "" {
		return nil, errors.New("gateway: engine cache reset requires EngineCacheBaseURL or BaseURL")
	}
	engine := enginecache.Engine(engineName)
	switch engine {
	case enginecache.EngineSGLang, enginecache.EngineVLLM:
	default:
		return nil, fmt.Errorf("gateway: unsupported engine cache engine %q (want sglang|vllm)", cfg.EngineCacheEngine)
	}
	requiredScope := ""
	if cfg.EngineCacheRequireExactSpan {
		requiredScope = enginecache.ScopeExactSpan
	}
	return &enginecache.Client{
		Engine:        engine,
		BaseURL:       baseURL,
		AdminAPIKey:   cfg.EngineCacheAdminKey,
		IdleTimeout:   cfg.EngineCacheIdleTimeout,
		RequiredScope: requiredScope,
	}, nil
}

// MarkReady stamps the instant the gateway became able to serve requests, closing
// the boot timeline (fak_gateway_time_to_ready_seconds / fak_gateway_ready_time_
// seconds). Idempotent and safe on a nil-startup Server; the first call wins.
func (s *Server) MarkReady() {
	if s == nil {
		return
	}
	s.startup.markReady(time.Now())
}

// SetModelLoadProfile records the boot-time weight-load breakdown the host captured
// while eagerly loading a model (fak serve --gguf), exposing it as the
// fak_model_load_* metric family. Passing nil clears it. Safe for concurrent use
// and on a nil Server.
func (s *Server) SetModelLoadProfile(p *ModelLoadProfile) {
	if s == nil {
		return
	}
	s.modelLoadMu.Lock()
	s.modelLoad = p
	s.modelLoadMu.Unlock()
}

func (s *Server) modelLoadProfile() *ModelLoadProfile {
	s.modelLoadMu.Lock()
	defer s.modelLoadMu.Unlock()
	return s.modelLoad
}

// complete runs the configured planner for one turn and records the inference
// metrics that make real model work visible at /metrics — the token counts the
// planner reports plus the wall-clock spent generating. Both /v1/chat/completions
// and /v1/messages route through it so the fak_gateway_inference_* family reflects
// every served turn on either wire. On a planner error nothing is recorded (a turn
// that produced no tokens is not a generation); the error is returned untouched so
// the caller's existing error handling is unchanged.
func (s *Server) complete(ctx context.Context, messages []agent.Message, tools []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error) {
	start := time.Now()
	comp, err := s.planner.Complete(ctx, messages, tools, opts...)
	dur := time.Since(start)
	if err != nil {
		return nil, err
	}
	s.metrics.observeInference(comp.Usage.PromptTokens, comp.Usage.CompletionTokens, comp.Usage.CachedPromptTokens(), comp.FinishReason, dur)
	return comp, nil
}

// plannerKind classifies the /v1/chat/completions backend for the /healthz
// "planner" field, so an operator (or a liveness probe) can tell at a glance
// whether a served chat is a real model or the deterministic offline mock:
//
//   - "mock"     the scripted offline fallback (no --base-url, no --gguf) — the
//     same condition New warns about loudly at boot.
//   - "proxy"    a live upstream provider (fak serve --base-url).
//   - "inkernel" the model fused into the kernel (fak serve --gguf).
//
// A nil or unrecognized planner reports "unknown" rather than masquerading as a
// real backend.
func plannerKind(p agent.Planner) string {
	switch p.(type) {
	case *agent.MockPlanner:
		return "mock"
	case *agent.HTTPPlanner:
		return "proxy"
	case *agent.InKernelPlanner:
		return "inkernel"
	default:
		return "unknown"
	}
}

func engineRegistered(id string) bool {
	for _, e := range abi.EngineIDs() {
		if e == id {
			return true
		}
	}
	return false
}

// adjudicate runs ONLY the adjudicator chain (k.Decide) over a (tool, rawArgs)
// pair and returns the pre-execution verdict — no engine dispatch, no pending
// submission state, nothing to leak. This is what a client-side executor asks for
// before it runs a tool. On a TRANSFORM (grammar repair) it resolves the rewritten
// args so the client can run the canonical form; that repaired-args string is the
// second return.
func (s *Server) adjudicate(ctx context.Context, tool, rawArgs string, readOnly bool, witness, traceID string) (wv WireVerdict, repaired string, err error) {
	start := time.Now()
	opTrace, opTool := traceID, tool
	defer func() {
		dur := time.Since(start)
		s.metrics.observeOperation("adjudicate", wv, err, dur)
		s.logGatewayOperation("adjudicate", opTrace, opTool, wv, err, dur)
	}()
	tc, err := s.buildCall(ctx, tool, rawArgs, readOnly, witness, traceID)
	if err != nil {
		return WireVerdict{}, "", err
	}
	opTrace, opTool = tc.TraceID, tc.Tool
	v := s.k.Decide(ctx, tc)
	wv = renderVerdict(v, nil)
	if v.Kind == abi.VerdictTransform {
		if tp, ok := v.Payload.(abi.TransformPayload); ok {
			repaired = string(resolveBytes(ctx, tp.NewArgs))
		}
	}
	return wv, repaired, nil
}

// syscall runs a (tool, rawArgs) pair through the FULL syscall boundary
// (k.Syscall: adjudicate -> vDSO -> dispatch to the registered engine ->
// context-MMU admit) and returns the rendered verdict plus the admitted result.
// This is the self-contained path: fak's registered engine produces the result,
// and a quarantined/poisoned result is already paged out before the gateway sees
// it.
func (s *Server) syscall(ctx context.Context, tool, rawArgs string, readOnly bool, witness, traceID string) (wv WireVerdict, env *ResultEnvelope, err error) {
	start := time.Now()
	opTrace, opTool := traceID, tool
	defer func() {
		dur := time.Since(start)
		s.metrics.observeOperation("syscall", wv, err, dur)
		s.logGatewayOperation("syscall", opTrace, opTool, wv, err, dur)
	}()
	tc, err := s.buildCall(ctx, tool, rawArgs, readOnly, witness, traceID)
	if err != nil {
		return WireVerdict{}, nil, err
	}
	opTrace, opTool = tc.TraceID, tc.Tool
	r, v := s.k.Syscall(ctx, tc)
	wv = renderVerdict(v, resultMeta(r))
	if r != nil {
		env = &ResultEnvelope{
			Status:  statusName(r.Status),
			Content: string(resolveBytes(ctx, r.Payload)),
			Meta:    r.Meta,
		}
	}
	return wv, env, nil
}

// buildCall converts untrusted wire input into an abi.ToolCall. The raw argument
// bytes are Put into a tainted, agent-scoped Ref (the fail-closed default the IFC
// sink-gate relies on) — the wire NEVER carries a Ref. Empty args normalize to
// "{}" so a zero Ref is never submitted.
func (s *Server) buildCall(ctx context.Context, tool, rawArgs string, readOnly bool, witness, traceID string) (*abi.ToolCall, error) {
	if strings.TrimSpace(tool) == "" {
		return nil, errors.New("missing tool name")
	}
	args := []byte(rawArgs)
	if len(args) == 0 {
		args = []byte("{}")
	}
	ref, err := abi.ActiveResolver().Put(ctx, args)
	if err != nil {
		return nil, fmt.Errorf("resolver: %w", err)
	}
	meta := metaFor(tool, readOnly)
	// The external world-state witness (a git commit / blob / lease epoch the caller is
	// reading at) keys this read for cross-agent dedup AND for causal revocation: a later
	// fak_revoke of this witness evicts every pooled entry admitted under it.
	if witness != "" {
		meta["witness"] = witness
	}
	// Thread a TraceID end-to-end: the IFC ledger + plan-CFI key their per-session
	// state on it, so a served call MUST carry one. The wire supplies it for
	// cross-call correlation; absent, we mint a fresh non-empty id rather than fall
	// back to the empty shared-default trace (which would pool every served session
	// onto one taint high-water mark).
	return &abi.ToolCall{Tool: tool, Args: ref, TraceID: s.traceFor(traceID), Meta: meta}, nil
}

// traceFor returns the caller's TraceID, or mints a fresh, process-unique non-empty
// one so the result-side IFC ledger + plan-CFI never collapse distinct served
// sessions onto the empty-string default trace.
func (s *Server) traceFor(traceID string) string {
	if traceID != "" {
		return traceID
	}
	return "gw-" + itoa(atomic.AddUint64(&s.traceSeq, 1))
}

// admit runs a CLIENT-PRODUCED tool result through the kernel's result-side stack
// (k.AdmitResult: the context-MMU quarantine + the IFC source-stamp that raises the
// per-trace taint ledger). This is what arms the exfil floor on the served path: a
// client that executes its own tool sends the RESULT back here, and a poisoned
// result is quarantined (paged out) + the session's taint high-water mark is raised
// before the result is admitted to context. The verdict + the admitted (possibly
// paged-out) result are rendered for the wire. It is the explicit fak_admit verb;
// admitOp is the shared core the auto-proxy also drives under its own op label.
func (s *Server) admit(ctx context.Context, tool, rawResult, witness, traceID string) (wv WireVerdict, env *ResultEnvelope, err error) {
	wv, env, err = s.admitOp(ctx, "admit", tool, rawResult, witness, traceID)
	if err != nil {
		return wv, env, err
	}
	// Native-path parity with the proxy (#411). The proxy fires the remote
	// engine-cache reset from admitInboundResults; the native admit routes
	// (POST /v1/fak/admit, the fak_admit MCP tool) quarantined locally but never
	// reset the upstream serving-engine cache, so a poisoned token-sequence could
	// survive in the provider KV/prefix cache when an agent drives fak natively
	// instead of through /v1/chat/completions. resetEngineCacheAfterQuarantine is
	// the SAME reset the proxy fires (a no-op when no engine cache is configured);
	// a remote-reset failure is surfaced fail-closed, wrapped so the HTTP handler
	// maps it to a 502 rather than a client 400.
	if wv.Kind == "QUARANTINE" {
		if rerr := s.resetEngineCacheAfterQuarantine(ctx, []ResultAdmission{{Verdict: wv}}); rerr != nil {
			return wv, env, fmt.Errorf("%w: %v", errEngineCacheReset, rerr)
		}
	}
	return wv, env, nil
}

// errEngineCacheReset marks an admit failure that originated in the REMOTE
// engine-cache reset (not the local admission). handleFakAdmit maps it to a 502 —
// the same fail-closed signal the proxy returns on a reset failure — while a local
// build/resolver error stays a 400 client error.
var errEngineCacheReset = errors.New("engine cache reset failed")

// admitOp is the shared result-side admission core: it builds an agent-scoped call,
// puts the result bytes into a tainted Ref, and folds the kernel's ResultAdmitter
// chain over them (context-MMU quarantine + IFC source-stamp/taint ledger), tagged
// with the caller's op label for metrics/logs. Both the explicit fak_admit verb
// (op "admit") and the auto /v1/chat/completions proxy (op "proxy_admit") route
// through it, so the result-side floor is identical on every served topology.
func (s *Server) admitOp(ctx context.Context, operation, tool, rawResult, witness, traceID string) (wv WireVerdict, env *ResultEnvelope, err error) {
	start := time.Now()
	opTrace, opTool := traceID, tool
	defer func() {
		dur := time.Since(start)
		s.metrics.observeOperation(operation, wv, err, dur)
		s.logGatewayOperation(operation, opTrace, opTool, wv, err, dur)
	}()
	tc, err := s.buildCall(ctx, tool, "", false, witness, traceID)
	if err != nil {
		return WireVerdict{}, nil, err
	}
	opTrace, opTool = tc.TraceID, tc.Tool
	body := []byte(rawResult)
	if len(body) == 0 {
		body = []byte("{}")
	}
	ref, err := abi.ActiveResolver().Put(ctx, body)
	if err != nil {
		return WireVerdict{}, nil, fmt.Errorf("resolver: %w", err)
	}
	r := &abi.Result{Call: tc, Payload: ref, Status: abi.StatusOK, Meta: map[string]string{}}
	v := s.k.AdmitResult(ctx, tc, r)
	env = &ResultEnvelope{
		Status:  statusName(r.Status),
		Content: string(resolveBytes(ctx, r.Payload)),
		Meta:    r.Meta,
	}
	wv = renderVerdict(v, r.Meta)
	return wv, env, nil
}

// admitInboundResults arms the RESULT-side floor on the auto /v1/chat/completions
// proxy (#7). In the OpenAI tool protocol a tool RESULT the client executed comes
// back on the NEXT turn as a role="tool" message; before this, those results flowed
// straight to the upstream model, so the result-side containment (context-MMU
// quarantine + IFC source-stamp/taint ledger + eviction) was inert on the proxy —
// armed only on the in-process Syscall/Reap path and the explicit fak_admit verb.
//
// Each inbound tool result is routed through k.AdmitResult keyed on the per-session
// traceID BEFORE it reaches the model: a poisoned/secret-bearing result is PAGED
// OUT (its forwarded content replaced with the quarantine stub, so the upstream
// model's KV never ingests the poison), and an untrusted-source result RAISES the
// trace's IFC taint high-water mark. That high-water mark is exactly what the
// already-wired sink-gate (k.Decide, keyed on the SAME traceID) reads when it
// adjudicates the calls the model then proposes — so an exfil call on a tainted
// session is refused. messages is mutated in place (request-local). The per-result
// admissions are returned for the fak response extension.
func (s *Server) admitInboundResults(ctx context.Context, messages []agent.Message, traceID string) ([]ResultAdmission, error) {
	var admissions []ResultAdmission
	for i := range messages {
		if messages[i].Role != agent.RoleTool {
			continue
		}
		tool := messages[i].Name
		if tool == "" {
			// A nameless tool result is still untrusted cross-boundary input. Admit it
			// under a placeholder so the content screen + fail-closed taint still fire
			// (provenance treats an unregistered tool as Untrusted).
			tool = "tool_result"
		}
		wv, envlp, aerr := s.admitOp(ctx, "proxy_admit", tool, messages[i].Content, "", traceID)
		if aerr != nil {
			// A result we cannot even admit is held out fail-closed rather than
			// forwarded raw to the model.
			messages[i].Content = `{"_quarantined":true,"boundary":"proxy","reason":"ADMIT_ERROR"}`
			admissions = append(admissions, ResultAdmission{
				ToolCallID: messages[i].ToolCallID, Tool: messages[i].Name,
				Verdict: WireVerdict{Kind: "QUARANTINE", Reason: "ADMIT_ERROR", Disposition: "TERMINAL"}})
			continue
		}
		// On a quarantine/transform the kernel paged the bytes out and rewrote the
		// payload in place; forward the paged-out form so the poison never reaches
		// the model. A plain Allow leaves the content untouched.
		if envlp != nil && (wv.Kind == "QUARANTINE" || wv.Kind == "TRANSFORM") {
			messages[i].Content = envlp.Content
		}
		admissions = append(admissions, ResultAdmission{
			ToolCallID: messages[i].ToolCallID,
			Tool:       messages[i].Name,
			Verdict:    wv,
		})
	}
	if err := s.resetEngineCacheAfterQuarantine(ctx, admissions); err != nil {
		return admissions, err
	}
	return admissions, nil
}

func (s *Server) resetEngineCacheAfterQuarantine(ctx context.Context, admissions []ResultAdmission) error {
	if s.engineCache == nil {
		return nil
	}
	for _, a := range admissions {
		if a.Verdict.Kind != "QUARANTINE" {
			continue
		}
		dirs := []cachemeta.ExternalInvalidationDirective{{
			Kind:      cachemeta.ExternalInvalidateKVSpan,
			Plane:     cachemeta.PlaneKVPrefix,
			Residency: cachemeta.Residency{Tier: cachemeta.TierRemote, Owner: string(s.engineCache.Engine)},
			Provider:  string(s.engineCache.Engine),
			Engine:    string(s.engineCache.Engine),
			Reason:    "proxy_tool_result_quarantine",
		}}
		res, err := s.engineCache.Invalidate(ctx, dirs)
		if err != nil {
			s.logf("gateway: engine cache reset failed after quarantined tool result: %v", err)
			return err
		}
		s.logf("gateway: engine cache reset engine=%s scope=%s directives=%d endpoint=%s",
			res.Engine, res.Scope, res.Directives, res.Endpoint)
		return nil
	}
	return nil
}

// contextChange applies a requester-initiated context-control mutation to a
// persisted recall core image. This is intentionally narrower than general file
// mutation: the only shipped operation is a tombstone, which makes a page absent
// from future model-visible recall while leaving the original page row and CAS
// bytes available for audit.
func (s *Server) contextChange(ctx context.Context, req ContextChangeRequest) (ContextChangeResponse, error) {
	select {
	case <-ctx.Done():
		return ContextChangeResponse{}, ctx.Err()
	default:
	}
	imageDir := strings.TrimSpace(req.ImageDir)
	if imageDir == "" {
		return ContextChangeResponse{}, errors.New("context change requires image_dir")
	}
	sess, err := recall.Load(imageDir)
	if err != nil {
		return ContextChangeResponse{}, fmt.Errorf("load core image: %w", err)
	}
	ch, err := sess.RequestContextChange(recall.ContextChangeRequest{
		Action:      contextAction(req.Action),
		Step:        req.Step,
		Digest:      strings.TrimSpace(req.Digest),
		Reason:      req.Reason,
		RequestedBy: req.RequestedBy,
		Witness:     req.Witness,
	})
	if err != nil {
		return ContextChangeResponse{}, err
	}
	if err := sess.Persist(imageDir); err != nil {
		return ContextChangeResponse{}, fmt.Errorf("persist core image: %w", err)
	}
	s.logf("gateway: context change %s step=%d image=%s requested_by=%s", ch.Action, ch.Step, imageDir, ch.RequestedBy)
	return contextChangeResponse(imageDir, ch, sess.Tombstoned(ch.Step)), nil
}

func contextAction(action string) recall.ContextAction {
	switch strings.TrimSpace(action) {
	case "", "tombstone", string(recall.ContextActionTombstone):
		return recall.ContextActionTombstone
	default:
		return recall.ContextAction(strings.TrimSpace(action))
	}
}

func contextChangeResponse(imageDir string, ch recall.ContextChange, tombstoned bool) ContextChangeResponse {
	return ContextChangeResponse{
		ImageDir:    imageDir,
		ID:          ch.ID,
		Action:      string(ch.Action),
		Step:        ch.Step,
		Digest:      ch.Digest,
		Reason:      ch.Reason,
		RequestedBy: ch.RequestedBy,
		Witness:     ch.Witness,
		TrustEpoch:  ch.TrustEpoch,
		Applied:     ch.Applied,
		Tombstoned:  tombstoned,
	}
}

// metaFor derives the kernel call hints. A caller may explicitly mark a call
// read-only (enabling vDSO dedup of duplicate reads); otherwise the gateway uses
// the read-only NAME prefix family (the same convention as DefaultPolicy's
// AllowPrefix and agent.metaFor) and FAILS CLOSED to destructive for anything
// else, so the vDSO never serves a stale write.
func metaFor(tool string, readOnly bool) map[string]string {
	if readOnly || readOnlyPrefix(tool) {
		return map[string]string{"readOnlyHint": "true", "idempotentHint": "true"}
	}
	return map[string]string{"readOnlyHint": "false", "idempotentHint": "false", "destructive": "true"}
}

func readOnlyPrefix(tool string) bool {
	for _, p := range []string{"get_", "read_", "search_", "list_", "lookup_", "find_", "calc"} {
		if strings.HasPrefix(tool, p) {
			return true
		}
	}
	return false
}

// resolveBytes materializes a Ref's bytes through the active resolver (mirrors
// agent.refBytes). An inline Ref carries its own bytes; a blob/region Ref is
// resolved on demand.
func resolveBytes(ctx context.Context, r abi.Ref) []byte {
	if r.Kind == abi.RefInline {
		return r.Inline
	}
	if res := abi.ActiveResolver(); res != nil {
		if b, err := res.Resolve(ctx, r); err == nil {
			return b
		}
	}
	return nil
}

func resultMeta(r *abi.Result) map[string]string {
	if r == nil {
		return nil
	}
	return r.Meta
}

// loopbackOnly reports whether addr binds ONLY the loopback interface — used to
// warn loudly when a no-auth gateway is exposed beyond localhost. It classifies by
// IP VALUE (net.ParseIP + IsLoopback), not by string prefix: "127.0.0.1.evil.com"
// is NOT loopback, and an empty host (the ":port" wildcard, which net.Listen binds
// to ALL interfaces) is NOT loopback either. A non-IP host (a DNS name) cannot be
// proven loopback at bind time, so it is treated as exposed.
func loopbackOnly(addr string) bool {
	host, _, err := net.SplitHostPort(addr)
	if err != nil {
		host = addr // no port present
	}
	host = strings.Trim(host, "[]")
	if host == "" {
		return false // ":port" => all interfaces
	}
	if host == "localhost" {
		return true
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}
