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
	"github.com/anthony-chaudhary/fak/internal/modelroute"
	"github.com/anthony-chaudhary/fak/internal/recall"
	"github.com/anthony-chaudhary/fak/internal/rungobs"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
	"github.com/anthony-chaudhary/fak/internal/vdso"
)

// DefaultCompactHistoryBudget is the resident-token line the cache-prefix-preserving
// history compaction trims the kept window to BY DEFAULT on the Anthropic passthrough.
// It is the operator's "reset once a conversation sprawls" trigger, expressed as a
// budget: once the compactible (uncached) suffix grows past it, the cut fires and drops
// the un-cacheable middle the provider re-bills every turn — while the cached_control
// prefix stays byte-identical, so a still-warm cache hit survives. ~48k keeps a typical
// short session untouched and only acts on genuinely long ones. Default-on is safe by
// construction: the cut only ever sheds UNCACHED bytes (it proves prefix-byte-identity
// before returning, agent.CompactAnthropicHistory), so it can never net-charge more by
// discarding a cached prefix. An explicit --compact-history-budget wins; 0 means OFF.
const DefaultCompactHistoryBudget = 48000

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
	// ReplicaBaseURLs adds static upstream replicas to the live proxy. When BaseURL
	// plus ReplicaBaseURLs names two or more endpoints, New wraps the per-endpoint
	// HTTP planners in a ReplicaRouter and dispatches turns round-robin. Empty keeps
	// the historical single-upstream behavior.
	ReplicaBaseURLs []string
	// Provider selects the upstream transcript wire when BaseURL is set
	// (openai, anthropic, gemini, xai). Empty keeps the OpenAI-compatible default.
	Provider string
	// APIKey is the credential sent to the upstream model (proxy mode only). On the
	// Anthropic wire its SCHEME is chosen by the token itself: an OAuth subscription
	// token ("sk-ant-oat…", agent.IsAnthropicOAuthToken) goes as Authorization:
	// Bearer + the oauth beta; a plain API key goes as x-api-key.
	APIKey string
	// PinUpstreamCredential makes the gateway authenticate the upstream with its OWN
	// configured APIKey and IGNORE the inbound client's credential — the subscription
	// path, where fak holds the real OAuth token and the wrapped client only sends a
	// placeholder key to satisfy its own "do I have credentials" check. Default false
	// keeps the transparent-hop passthrough (forward the client's own key upstream).
	PinUpstreamCredential bool
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
	// CPUOffloadExperts, when true with a device Backend, keeps the MoE expert GEMMs on
	// host RAM while the dense projections + router + attention run on the device — the
	// `--n-cpu-moe` hybrid that lets a model whose experts dwarf VRAM (e.g. GLM-5.2 Q4)
	// serve at all. Set by `fak serve --cpu-offload-experts`; ignored without a Backend.
	CPUOffloadExperts bool
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
	// ObserveSession reports one served session's current DRIVE state (run-state,
	// budget, priority, pace) — the read side of the session-control surface. Nil
	// disables the GET /v1/fak/session/{id} route. Injected by cmd/fak so this
	// package stays session-internals-blind, mirroring ObserveTrace.
	ObserveSession SessionObserveFunc
	// ControlSession applies one control verb (run/budget/pace/priority) to a served
	// session's DRIVE state — the write side of the session-control surface. Nil
	// disables the POST /v1/fak/session/{id}/{verb} route. Injected by cmd/fak.
	ControlSession SessionControlFunc
	// SteerSession enqueues an operator steer onto the host's a2achan bus so a RUNNING
	// detached session receives the input at its next turn boundary (#760). Nil disables
	// the POST /v1/fak/session/{id}/steer route (404). Injected by cmd/fak so this package
	// stays a2achan-blind, mirroring ControlSession.
	SteerSession SteerSessionFunc
	// ListSessions returns a snapshot of EVERY live session's DRIVE state — the
	// multi-session read behind GET /v1/fak/sessions (the table's Snapshot turned
	// into a live operator surface). Nil disables the route. Injected by cmd/fak so
	// this package stays session-internals-blind, mirroring ObserveSession.
	ListSessions SessionListFunc
	// DecideSession gates one served request at its session boundary. It is the
	// mutating hot-path twin of ObserveSession: the host calls session.Table.Decide,
	// so run-state refusal, TurnsLeft debit, budget exhaustion, and per-turn pace are
	// applied before the model turn is served. Nil keeps the historical observe-only
	// admission path.
	DecideSession SessionDecideFunc
	// DebitSession reports the just-served turn's token usage after the planner
	// returns, so TokensLeft and the long-context budget can be debited from the
	// live session table. Nil is a no-op for embedders that have not wired the
	// session table.
	DebitSession SessionDebitFunc
	// ResetOnBudget is the OPT-IN "human-like reset": when a served session crosses
	// its (context/output) token budget, instead of refusing the next request with a
	// 409 + a passive reset directive, the host distills the transcript into a compact
	// carryover seed, re-arms a fresh session, and the gateway splices the seed ahead
	// of the live messages so the CLIENT'S next request just continues transparently.
	// It is given the canonical transcript and returns the fresh trace id + the seed
	// messages to prepend. ok=false (or a nil hook) falls back to the historical 409 +
	// SessionResetDirective path verbatim — so the reset is strictly additive and the
	// default behavior is unchanged. Injected by cmd/fak (fak serve --reset-on-budget).
	ResetOnBudget ResetOnBudgetFunc
	// OnBudgetExhausted is the host/supervisor notification fired after a served turn's
	// reported usage drains a resettable budget. Unlike ResetOnBudget, it fires with
	// the just-served transcript still available, so a process supervisor can build a
	// carryover seed and restart a wrapped child before the child sends another giant
	// request. Nil is inert.
	OnBudgetExhausted BudgetExhaustedFunc
	// DefaultTraceID is used when a proxied HTTP/MCP caller omits X-Trace-Id /
	// trace_id. Empty preserves the historical process-unique gw-N mint. A stable
	// value lets wrapped CLIs that do not expose trace headers still share one
	// operator-addressable session budget.
	DefaultTraceID string
	// Logf is the structured log sink (default: stderr). MCP-over-stdio sets this
	// to stderr so protocol bytes on stdout are never corrupted.
	Logf func(format string, args ...any)
	// DebugStatsf is the OPTIONAL per-turn human debug sink (#793). When non-nil, every
	// served turn renders ONE compact, payload-free line — request/cache_read/cache_creation
	// tokens, the compaction action, and the resetScore SHADOW health (one of the five
	// healthy_cache/cache_decay/stale_prefix/cooldown/unknown_provider states) — so an
	// operator can watch turn-by-turn cache & compaction behavior live. nil (the default)
	// emits nothing; it is independent of Logf (the JSON --log stream), so --debug-stats
	// works with a clean --log-off terminal. `fak guard --debug-stats` / `fak serve
	// --debug-stats` wire it to stderr.
	DebugStatsf func(format string, args ...any)
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
	// CtxViewBudget, when > 0, wires the ctxplan context PLANNER into the live
	// serve/guard loop: each turn, the forwarded message history is lowered into a
	// lossless ctxplan store and re-materialized as an O(1) planned VIEW under this
	// resident-token budget, replacing the append-the-whole-transcript path with a
	// planned view (issue #555). On the buffered/OpenAI wire it re-plans the decoded
	// []Message; on the flagship Anthropic PASSTHROUGH it materializes the view onto
	// req.Raw by stubbing each elided middle turn in place while the cache_control prefix
	// stays byte-identical (#927 — the deferred #555 req.Raw transform). 0 (the default)
	// leaves the existing path byte-for-byte unchanged — the guard a production deploy
	// needs before an in-flight rewrite of turn history ships (the same posture as the
	// agent seam's FAK_CTXPLAN_SEAM).
	CtxViewBudget int
	// CompactHistoryBudget, when > 0, wires the cache-prefix-preserving history rewrite
	// into the flagship `fak guard -- claude` Anthropic PASSTHROUGH. Each turn the OUTBOUND
	// body is compacted so OLD whole turns beyond the cache_control prefix are dropped to
	// this resident-token budget, while the cached prefix bytes are copied VERBATIM so the
	// upstream cache hit survives (see agent.CompactAnthropicHistory). 0 means OFF (body
	// forwarded byte-for-byte). The CLI flag defaults this to DefaultCompactHistoryBudget
	// (a non-zero default-on trigger that trims sprawl while a typical short session stays
	// untouched), so the byte-for-byte path is now the explicit --compact-history-budget=0
	// opt-out, not the default. Anthropic passthrough only; it is an inert no-op on every
	// other wire. Sibling of CtxViewBudget: compaction drops a contiguous suffix of old
	// turns, ctxview stubs the planner's non-contiguous resident-set misses (#927).
	CompactHistoryBudget int
	// ToolFloorDenies, when non-nil, is the INBOUND twin of CompactHistoryBudget: the
	// host's pure predicate "would the capability floor DEFAULT_DENY this tool name for
	// every possible argument?" — true ONLY for a name the policy admits under no args
	// (absent from Allow and matching no AllowPrefix), never an arg-conditional tool.
	// When set on the Anthropic PASSTHROUGH, each turn the gateway prunes those provably-
	// unreachable tool DEFINITIONS from the OUTBOUND tools[] (promptmmu.CompactInboundTools),
	// splicing on the original bytes so the cache_control prefix stays byte-identical and
	// the upstream prompt-cache hit survives. The kernel still default-denies the call if the
	// model somehow names a pruned tool, so dropping the definition is behavior-preserving by
	// construction. nil (the default) leaves tools[] byte-for-byte unchanged. The gateway
	// imports no policy internals — the host (cmd/fak) supplies the floor predicate, mirroring
	// ReloadPolicy / DecideSession. Anthropic passthrough only; inert on every other wire.
	ToolFloorDenies func(toolName string) bool
	// RouteManifest, when non-nil, makes the gateway classify each fak_syscall tool
	// call into a modelroute.Subject and route it: for a single-model (PICK) plan the
	// chosen model id is written to abi.ToolCall.Engine BEFORE Submit, so the kernel
	// dispatches to it AND the residency PDP adjudicates the real route (a tenant /
	// sensitive call bound for a remote model is denied at the boundary, never
	// fail-open). nil (the default) leaves Engine unset -> the kernel default engine,
	// byte-for-byte the pre-routing behavior. An ensemble (multi-member) plan is NOT
	// fanned out here — that is issue #597; the gateway leaves Engine unset and defers
	// to the kernel default until ensemble dispatch lands. New() validates a non-nil
	// manifest and fails loud on a malformed one (a mis-routed model is a security
	// boundary, never a silent default). Set by `fak serve --route-manifest` (#601).
	RouteManifest *modelroute.Manifest
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

// SessionState is the wire form of a served session's DRIVE state — the value
// GET /v1/fak/session/{id} returns and every control verb echoes back. Run is the
// lowercase run-state TOKEN ("running"|"throttled"|"paused"|"draining"|"stopped"),
// never the enum: the gateway is session-internals-blind the same way it is
// IFC-internals-blind for TraceObserveFunc, so it carries wire names only. Rev is
// the monotonic revision the table bumps on every write; a client may round-trip it
// as if_rev to reject a stale clobber (optimistic concurrency).
type SessionState struct {
	TraceID        string        `json:"trace_id"`
	Run            string        `json:"run"`
	Budget         SessionBudget `json:"budget"`
	Priority       int           `json:"priority"`
	Pace           SessionPace   `json:"pace"`
	Reason         string        `json:"reason,omitempty"`
	ContinuationID string        `json:"continuation_id,omitempty"`
	ParentTrace    string        `json:"parent_trace,omitempty"`
	Generation     int           `json:"generation,omitempty"`
	Rev            uint64        `json:"rev"`
}

// SessionBudget is the wire form of internal/session.Budget. TurnsLeft/TokensLeft
// at -1 (session.Unbounded) mean no cap; ContextTokensLeft uses 0 as off and a
// positive value as the long-window reset budget.
type SessionBudget struct {
	TurnsLeft         int `json:"turns_left"`
	TokensLeft        int `json:"tokens_left"`
	ContextTokensLeft int `json:"context_tokens_left,omitempty"`
}

// SessionPace is the wire form of internal/session.Pace. Zero on either axis means
// "no opinion" (the planner's own default stands).
type SessionPace struct {
	MaxTokensPerTurn int `json:"max_tokens_per_turn"`
	MinTurnGapMs     int `json:"min_turn_gap_ms"`
}

// SessionControlRequest is the gateway-parsed body of POST
// /v1/fak/session/{trace_id}/{verb}. Exactly the field named by the verb is read;
// the others are ignored. if_rev, when non-zero, is the optimistic-concurrency
// guard: the write is taken only if the session's current Rev matches, else the
// route returns 409 (the client re-reads and retries).
type SessionControlRequest struct {
	Run      string         `json:"run,omitempty"`      // verb "run": target run-state token
	Reason   string         `json:"reason,omitempty"`   // verb "run": reason token (closed vocabulary)
	Budget   *SessionBudget `json:"budget,omitempty"`   // verb "budget"
	Pace     *SessionPace   `json:"pace,omitempty"`     // verb "pace"
	Priority *int           `json:"priority,omitempty"` // verb "priority"
	IfRev    uint64         `json:"if_rev,omitempty"`   // optional CAS guard
}

// SessionObserveFunc is injected by the host CLI so the gateway can read one
// session's live DRIVE state without importing internal/session. An unseen trace
// reads its default (Running, unbounded) — the table's own safe default, never a
// phantom Stopped.
type SessionObserveFunc func(context.Context, string) SessionState

// SessionControlFunc is injected by the host CLI so the gateway can apply one
// control verb to a session's DRIVE state without importing internal/session. It
// returns the NEW state, an ok flag (false ⇒ the session is terminal, or an if_rev
// CAS guard lost the race; the route returns 409), and an error (non-nil ⇒ the
// verb or body was malformed — unknown run-state token, missing field, unknown
// verb; the route returns 400). ok==false with err==nil is a concurrent/terminal
// refusal, not a client error.
type SessionControlFunc func(ctx context.Context, traceID, verb string, req SessionControlRequest) (SessionState, bool, error)

// SteerRequest is the body of POST /v1/fak/session/{trace_id}/steer (#760): operator
// input sent to a RUNNING detached session, delivered at its next turn boundary. Text is
// the message the running agent receives — the "send input without stopping it" affordance
// of Claude Code #21419.
type SteerRequest struct {
	Text string `json:"text"`
}

// SteerSessionFunc is injected by the host CLI so the gateway can enqueue an operator
// steer onto the host's a2achan bus (Session locale, keyed by traceID) without importing
// internal/a2achan. A non-nil error is the adjudication floor's deny-as-value surfaced
// (tainted / over-scoped / uncapped body), which the route maps to 422 — the same
// default-deny floor that gates a tool call, here gating operator input. nil hook ⇒ the
// steer route is not configured (404).
type SteerSessionFunc func(ctx context.Context, traceID, text string) error

// SessionVerdict is the gateway wire-neutral projection of session.Verdict. The
// gateway intentionally carries only primitive fields so it stays decoupled from
// internal/session while still applying the table's mutating Decide semantics on the
// served request path.
type SessionVerdict struct {
	Proceed   bool
	MaxTokens int
	MinGapMs  int
	State     SessionState
	Stop      bool
	Reason    string
}

// SessionDecideFunc is injected by the host CLI to run session.Table.Decide for one
// served request boundary. It returns a SessionVerdict instead of importing
// internal/session into gateway.
type SessionDecideFunc func(ctx context.Context, traceID string) SessionVerdict

// SessionUsage is the gateway's session-table-neutral token accounting for one
// served request. CompletionTokens debits the historical output budget; ContextTokens
// is the provider-normalized prompt/context window for the long-session reset budget.
type SessionUsage struct {
	PromptTokens             int `json:"prompt_tokens,omitempty"`
	CompletionTokens         int `json:"completion_tokens,omitempty"`
	ContextTokens            int `json:"context_tokens,omitempty"`
	CacheReadInputTokens     int `json:"cache_read_input_tokens,omitempty"`
	CacheCreationInputTokens int `json:"cache_creation_input_tokens,omitempty"`
}

// SessionDebitFunc is injected by the host CLI to run session.Table.DebitUsage with
// the token usage reported after a served request finishes.
type SessionDebitFunc func(ctx context.Context, traceID string, usage SessionUsage) SessionState

// ResetOnBudgetFunc is the host's budget-reset action (Config.ResetOnBudget). Given
// the budget-drained session's trace and its canonical transcript, the host builds
// the carryover seed (durable facts + task recap + warm-prefix + verbatim tail), calls
// session.Table.Recontinue to re-arm a fresh session, and returns the fresh trace id
// plus the seed messages the gateway prepends to the live request. ok=false means the
// host declined to reset (not a budget-reset reason, or no carryover) — the gateway
// then falls back to the historical refusal. The gateway stays session-internals-blind:
// it never imports internal/session or internal/sessionreset; the host owns both.
type ResetOnBudgetFunc func(ctx context.Context, trace string, messages []agent.Message) (newTrace string, seed []agent.Message, ok bool)

// BudgetExhaustedFunc is injected by hosts that supervise a real child process.
// It fires after a served turn's post-response usage debit drains a resettable
// budget, while the transcript for that turn is still available.
type BudgetExhaustedFunc func(ctx context.Context, st SessionState, messages []agent.Message)

// Server is a configured, ready-to-serve gateway. Construct with New; serve with
// Handler()/ListenAndServe (HTTP) or ServeStdio (MCP over stdin/stdout).
type Server struct {
	k              *kernel.Kernel
	engineID       string
	model          string
	requireKey     string
	version        string
	logf           func(format string, args ...any)
	debugStatsf    func(format string, args ...any) // optional per-turn human debug sink (#793); nil = off
	feed           *coherenceFeed                   // the cross-agent "what changed" feed (vdso coherence bus)
	metrics        *gatewayMetrics
	traceSeq       uint64 // mints a non-empty TraceID when the wire omits one (atomic)
	reloadPolicy   PolicyReloadFunc
	resetTrace     TraceResetFunc
	observeTrace   TraceObserveFunc
	observeSession SessionObserveFunc
	controlSession SessionControlFunc
	steerSession   SteerSessionFunc
	listSessions   SessionListFunc
	decideSession  SessionDecideFunc
	debitSession   SessionDebitFunc
	resetOnBudget  ResetOnBudgetFunc
	budgetDrained  BudgetExhaustedFunc
	defaultTraceMu sync.RWMutex
	defaultTraceID string

	// startup is the one-time boot timeline (start -> ready, per-phase costs),
	// exposed as fak_gateway_startup_* gauges. See startup.go.
	startup *startupProfile
	// modelLoad is the optional boot-time weight-load breakdown set by the host via
	// SetModelLoadProfile when it eagerly loads a model (fak serve --gguf). nil
	// suppresses every fak_model_load_* metric. Guarded by modelLoadMu.
	modelLoadMu sync.Mutex
	modelLoad   *ModelLoadProfile

	// planner generates the assistant turn for the /v1/chat/completions proxy. A
	// live HTTPPlanner/ReplicaRouter when BaseURL/ReplicaBaseURLs are set, else the
	// offline MockPlanner. Settable in-package for tests.
	planner     agent.Planner
	engineCache *enginecache.Client

	// ctxView, when non-nil, is the guarded ctxplan seam that re-plans each buffered
	// turn's history into an O(1) resident view (issue #555). nil (CtxViewBudget == 0)
	// leaves the forwarded history untouched; maybePlanMessages is an inert identity then.
	ctxView *agent.CtxViewPlanner

	// sessionPlanners holds ONE persistent agent.SessionPlanner per session trace id, so
	// the live ctxplan path maintains an incremental index across a conversation's turns
	// (O(c·N) cumulative) instead of rebuilding the lossless store and full-scanning every
	// turn (the stateless CtxViewPlanner.RenderTurn path, O(N²)). nil/empty until ctxView
	// is enabled; minted lazily by sessionPlannerFor and bounded so it cannot grow without
	// limit. Guarded by sessionPlannerMu. The two paths are output-equivalent (proven by
	// agent.TestSessionPlannerBoundedMatchesStatelessFullScan), so this only changes COST.
	sessionPlannerMu sync.Mutex
	sessionPlanners  map[string]*agent.SessionPlanner

	// resetHealth holds ONE rolling compaction-health record per session trace id, fed the
	// provider's OBSERVED cache counters on every compacted turn so the per-session resetScore
	// shadow surface (#792, reset_shadow.go) can recommend cut-vs-reset without re-deriving the
	// session's cache health from a global counter. nil/empty until the first compacted turn;
	// minted lazily by resetHealthForLocked and bounded by maxResetHealthSessions. Guarded by
	// resetHealthMu. SHADOW-only: nothing here ever resets a session.
	resetHealthMu sync.Mutex
	resetHealth   map[string]*sessionResetHealth

	// compactHistoryBudget mirrors Config.CompactHistoryBudget: when > 0 the flagship
	// Anthropic passthrough compacts OLD turns in the OUTBOUND body to this resident-token
	// budget while preserving the cached-prefix bytes (agent.CompactAnthropicHistory). 0
	// (the default) leaves the body byte-for-byte unchanged.
	compactHistoryBudget int

	// toolFloorDenies mirrors Config.ToolFloorDenies: the INBOUND-half predicate over a
	// tool name (true ⇔ the floor DEFAULT_DENYs it for every arg). When non-nil the
	// Anthropic passthrough prunes those provably-unreachable tool DEFINITIONS from the
	// outbound tools[] while keeping the cache_control prefix byte-identical. nil leaves
	// tools[] unchanged.
	toolFloorDenies func(toolName string) bool

	// pinUpstreamCredential, when set, makes the Anthropic passthrough authenticate
	// upstream with the planner's OWN configured credential and ignore the inbound
	// client's key (the subscription path — see Config.PinUpstreamCredential).
	pinUpstreamCredential bool

	// cacheStream is the unified cachemeta.Entry observability fold (fak_cache_*).
	// New subscribes it to the process-global vDSO's live tier-2 cache-event sink so
	// every fill/hit/evict/revoke on the strongest local cache is rendered on
	// /metrics; Close detaches the sink. nil suppresses the family. See metrics.go.
	cacheStream *cachemeta.StreamMetrics

	// rungObs is the passive rung-decision distribution counter (fak_kernel_decisions_total).
	// New registers it as a global abi.Emitter subscribed to EvDecide/EvDeny/EvVDSOHit;
	// it re-folds each decided call off the hot path to bucket it by winning rung. nil
	// (older/non-gateway construction paths) suppresses the metric family. It is passive:
	// it never touches the verdict or Counters, so the decide/deny hot path is unchanged.
	rungObs *rungobs.Observer

	// route, when non-nil, is the per-call model-routing policy buildCall consults to
	// set abi.ToolCall.Engine PRE-submit (the load-bearing residency contract — see
	// Config.RouteManifest and buildCall). nil leaves Engine unset (kernel default).
	// It is a *modelroute.Live (an atomic holder), not a bare *Manifest, so a host
	// watcher can hot-swap the policy on a file edit without a torn read (#842): a
	// classification sees either the whole old manifest or the whole new one.
	route *modelroute.Live
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
	// A misconfigured routing policy is a security boundary (it decides which model
	// — local or remote — a tenant payload reaches), so validate it at New and fail
	// loud rather than fall through to a silent default model at dispatch time.
	if cfg.RouteManifest != nil {
		if err := cfg.RouteManifest.Validate(); err != nil {
			return nil, fmt.Errorf("gateway: route manifest: %w", err)
		}
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

	proxyURLs, err := proxyBaseURLs(cfg)
	if err != nil {
		return nil, err
	}

	var planner agent.Planner
	t := time.Now()
	switch {
	case len(proxyURLs) != 0:
		planner, err = newProxyPlanner(cfg, model, proxyURLs)
		if err != nil {
			return nil, err
		}
	case cfg.InKernelModel != nil && cfg.Tokenizer != nil:
		// Serve the model fused into the kernel as the chat backend on BOTH
		// /v1/chat/completions and /v1/messages (they share s.planner.Complete):
		// real ChatML chat via internal/tokenizer, the cmd/fakchat recipe factored
		// into a Planner. Falls through to MockPlanner if the host didn't preload.
		planner = agent.NewInKernelPlanner(cfg.InKernelModel, cfg.Tokenizer, model, cfg.InKernelQ4K, cfg.Backend, cfg.CPUOffloadExperts)
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

	// Passive rung-decision observability (issue #693): register a rungobs Emitter that
	// folds the kernel's verdict stream into a per-(rung,kind,reason) histogram,
	// exposed on /metrics as fak_kernel_decisions_total. It subscribes to ONLY
	// EvDecide/EvDeny/EvVDSOHit, so it adds zero work to the every-syscall event path,
	// and it is passive (re-folds off the hot path; never mutates verdict or Counters).
	rungObs := rungobs.New()
	abi.RegisterEmitter(rungObs)

	// The ctxplan view planner is OFF unless the host set a resident-token budget. nil
	// leaves maybePlanMessages an inert identity (the byte-for-byte-unchanged guard).
	var ctxView *agent.CtxViewPlanner
	if cfg.CtxViewBudget > 0 {
		ctxView = &agent.CtxViewPlanner{Enabled: true, Budget: cfg.CtxViewBudget}
	}

	return &Server{
		k:                    k,
		engineID:             engineID,
		model:                model,
		requireKey:           cfg.RequireKey,
		version:              version,
		logf:                 logf,
		debugStatsf:          cfg.DebugStatsf,
		reloadPolicy:         cfg.ReloadPolicy,
		resetTrace:           cfg.ResetTrace,
		observeTrace:         cfg.ObserveTrace,
		observeSession:       cfg.ObserveSession,
		controlSession:       cfg.ControlSession,
		steerSession:         cfg.SteerSession,
		listSessions:         cfg.ListSessions,
		decideSession:        cfg.DecideSession,
		debitSession:         cfg.DebitSession,
		resetOnBudget:        cfg.ResetOnBudget,
		budgetDrained:        cfg.OnBudgetExhausted,
		defaultTraceID:       strings.TrimSpace(cfg.DefaultTraceID),
		startup:              startup,
		planner:              planner,
		engineCache:          remoteCache,
		ctxView:              ctxView,
		compactHistoryBudget: cfg.CompactHistoryBudget,
		toolFloorDenies:      cfg.ToolFloorDenies,
		cacheStream:          cacheStream,
		rungObs:              rungObs,
		feed:                 newCoherenceFeed(0),
		metrics:              newGatewayMetrics(time.Now()),
		route:                newRouteLive(cfg.RouteManifest),

		pinUpstreamCredential: cfg.PinUpstreamCredential,
	}, nil
}

// newRouteLive wraps the validated config manifest in an atomic Live holder, or
// returns nil when routing is off (no --route-manifest). A nil Live leaves
// routeDecision on the kernel-default path, byte-for-byte the pre-routing behavior.
func newRouteLive(m *modelroute.Manifest) *modelroute.Live {
	if m == nil {
		return nil
	}
	return modelroute.NewLive(m)
}

// RouteLive returns the atomic holder of the live routing policy, or nil when no
// --route-manifest is installed. The host (cmd/fak serve) hands this to a
// modelroute.Watcher so a manifest edit hot-swaps the policy this server reads —
// the same Live, so the swap is visible on the hot path with no restart (#842).
func (s *Server) RouteLive() *modelroute.Live { return s.route }

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
		urls, err := proxyBaseURLs(cfg)
		if err != nil {
			return nil, err
		}
		if len(urls) > 1 {
			return nil, errors.New("gateway: engine cache reset with replica base URLs requires EngineCacheBaseURL")
		}
		if len(urls) == 1 {
			baseURL = urls[0]
		}
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

func proxyBaseURLs(cfg Config) ([]string, error) {
	urls := make([]string, 0, 1+len(cfg.ReplicaBaseURLs))
	if base := strings.TrimSpace(cfg.BaseURL); base != "" {
		urls = append(urls, base)
	}
	for i, base := range cfg.ReplicaBaseURLs {
		base = strings.TrimSpace(base)
		if base == "" {
			return nil, fmt.Errorf("gateway: replica base URL %d is empty", i+1)
		}
		urls = append(urls, base)
	}
	return urls, nil
}

func newProxyPlanner(cfg Config, model string, baseURLs []string) (agent.Planner, error) {
	if len(baseURLs) == 1 {
		return agent.NewProviderHTTPPlanner(cfg.Provider, baseURLs[0], model, cfg.APIKey)
	}
	replicas := make([]PlannerReplica, 0, len(baseURLs))
	for i, base := range baseURLs {
		p, err := agent.NewProviderHTTPPlanner(cfg.Provider, base, model, cfg.APIKey)
		if err != nil {
			return nil, err
		}
		replicas = append(replicas, PlannerReplica{
			Name:    fmt.Sprintf("replica-%d", i+1),
			Planner: p,
		})
	}
	return NewReplicaRouter(model, replicas)
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

// AdjudicationSummary returns a verdict roll-up over every kernel decision this
// gateway has made so far — proposed-call adjudication, direct syscalls, and inbound
// result admission. It is the live tally `fak guard` prints on exit (what the kernel
// allowed vs denied / repaired / quarantined), read straight from the same operation
// counters /metrics exposes. Safe on a nil Server (returns the zero summary).
func (s *Server) AdjudicationSummary() AdjudicationSummary {
	if s == nil {
		return AdjudicationSummary{ByReason: map[string]uint64{}}
	}
	return s.metrics.adjudicationSummary()
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
	s.modelLoad = p.clone()
	s.modelLoadMu.Unlock()
}

func (s *Server) modelLoadProfile() *ModelLoadProfile {
	s.modelLoadMu.Lock()
	defer s.modelLoadMu.Unlock()
	return s.modelLoad.clone()
}

// maybePlanMessages is the live-loop integration point for the ctxplan context PLANNER
// (issue #555): when the view planner is enabled, each buffered turn's history is lowered
// into a lossless store and re-materialized as an O(1) resident view under the configured
// budget — a planned view in place of appending the whole transcript. When the planner is
// off (the default) it returns the input UNCHANGED, so a deploy that leaves the flag off is
// byte-for-byte identical to the pre-seam path. It is FAIL-SAFE: any planner error or empty
// render falls back to the full lossless history, so an experimental rewrite can never
// break or empty a turn — the planner only ever SHORTENS, and on doubt it shortens nothing.
func (s *Server) maybePlanMessages(ctx context.Context, trace string, messages []agent.Message) []agent.Message {
	if s.ctxView == nil || !s.ctxView.Enabled {
		return messages
	}
	// With a stable session trace, plan through the PERSISTENT per-session index — the
	// incremental O(c·N) path, output-equivalent to the stateless full-scan but without
	// rebuilding the lossless store every turn. Without a trace (a one-shot caller), fall
	// back to the stateless shared planner so behavior is unchanged for an unkeyed request.
	if sp := s.sessionPlannerFor(trace); sp != nil {
		planned := sp.RenderTurn(ctx, messages)
		if len(planned) == 0 {
			return messages // fail-safe: never empty a turn
		}
		return planned
	}
	planned, err := s.ctxView.RenderTurn(ctx, messages)
	if err != nil || len(planned) == 0 {
		s.logf("gateway: ctxplan view planning fell back to full history: %v", err)
		return messages
	}
	return planned
}

// maxSessionPlanners bounds the per-session planner cache so a long-lived gateway serving
// many distinct traces cannot grow it without limit. When the cache is full a new trace
// evicts the whole map (a cheap generational reset) rather than tracking per-entry LRU —
// the planners are reconstructible from the next turn's full history, so eviction only
// costs that session one O(N) rebuild, never correctness.
const maxSessionPlanners = 8192

// sessionPlannerFor returns the persistent SessionPlanner for a trace, minting one lazily
// from the shared ctxView config (CtxViewPlanner.NewSession). It returns nil when the
// planner is disabled or the trace is empty, so the caller falls back to the stateless
// path. Concurrency-safe: the per-session planner is mutated only under sessionPlannerMu
// by the single in-flight turn for that trace (turns of one session are serial).
func (s *Server) sessionPlannerFor(trace string) *agent.SessionPlanner {
	if s.ctxView == nil || !s.ctxView.Enabled || trace == "" {
		return nil
	}
	s.sessionPlannerMu.Lock()
	defer s.sessionPlannerMu.Unlock()
	if s.sessionPlanners == nil {
		s.sessionPlanners = make(map[string]*agent.SessionPlanner)
	}
	if sp, ok := s.sessionPlanners[trace]; ok {
		return sp
	}
	if len(s.sessionPlanners) >= maxSessionPlanners {
		s.sessionPlanners = make(map[string]*agent.SessionPlanner) // generational reset
	}
	sp := s.ctxView.NewSession()
	s.sessionPlanners[trace] = sp
	return sp
}

// complete runs the configured planner for one turn and records the inference
// metrics that make real model work visible at /metrics — the token counts the
// planner reports plus the wall-clock spent generating. Both /v1/chat/completions
// and /v1/messages route through it so the fak_gateway_inference_* family reflects
// every served turn on either wire. On a planner error nothing is recorded (a turn
// that produced no tokens is not a generation); the error is returned untouched so
// the caller's existing error handling is unchanged.
func (s *Server) complete(ctx context.Context, trace string, messages []agent.Message, tools []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error) {
	// Re-plan the turn history into an O(1) resident view before the model sees it —
	// the "replace append+compact with a planned view" rung (issue #555). Inert (an
	// identity) unless CtxViewBudget > 0, so the default path is unchanged. The trace keys
	// the persistent per-session planner so the rewrite is O(c·N), not O(N²), across turns.
	messages = s.maybePlanMessages(ctx, trace, messages)
	start := time.Now()
	comp, err := s.planner.Complete(ctx, messages, tools, opts...)
	dur := time.Since(start)
	if err != nil {
		if _, _, _, ok := inKernelOOMObservation(err); ok {
			s.observePlannerRequestMemory()
		}
		return nil, err
	}
	s.metrics.observeInference(comp.Usage.PromptTokens, comp.Usage.CompletionTokens, comp.Usage.CachedPromptTokens(), comp.FinishReason, dur)
	s.observePlannerRequestMemory()
	return comp, nil
}

// completeServed is complete plus the served-session usage debit. The request
// boundary has already called beginServedSessionTurn (and therefore Decide); after a
// successful planner response the provider usage is finally known, so debit the
// output/context budgets here. Planner errors keep the old behavior: no usage was
// reported, so there is nothing to debit beyond the turn admission already taken.
func (s *Server) completeServed(ctx context.Context, turn servedSessionTurn, messages []agent.Message, tools []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error) {
	comp, err := s.complete(ctx, turn.traceID, messages, tools, opts...)
	if err != nil {
		return nil, err
	}
	s.debitServedSessionTurn(ctx, turn, comp.Usage, messages)
	return comp, nil
}

// plannerKind classifies the /v1/chat/completions backend for the /healthz
// "planner" field, so an operator (or a liveness probe) can tell at a glance
// whether a served chat is a real model or the deterministic offline mock:
//
//   - "mock"     the scripted offline fallback (no --base-url, no --gguf) — the
//     same condition New warns about loudly at boot.
//   - "proxy"    one live upstream provider (fak serve --base-url).
//   - "replica"  a static round-robin live upstream fleet.
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
	case *ReplicaRouter:
		return "replica"
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
	// Ensemble fan-out (issue #597): a multi-member routing Plan runs each member as
	// its OWN adjudicated kernel call and folds the outputs. buildCall left tc.Engine
	// unset for an ensemble (routeEngine returns "" rather than collapse to one member);
	// dispatchEnsemble re-reads the same routing decision and submits N independent
	// calls. The single-model PICK below is byte-for-byte the pre-#597 path.
	if plan, ok := s.ensemblePlan(tc.Tool, readOnly, tc.Meta); ok {
		wv, env, err = s.dispatchEnsemble(ctx, tc, plan)
		return wv, env, err
	}
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

// dispatchEnsemble executes a multi-member routing Plan (issue #597): it runs each
// member as its OWN independently-adjudicated kernel call — carrying THAT member's
// model in abi.ToolCall.Engine (the same pre-submit residency contract a single-model
// route obeys) — gathers the ALLOWED members' outputs in Plan.Members order, and folds
// them with modelroute.Combine(plan.Reduce, votes). The contract this honors, point by
// point (see the internal/modelroute package doc):
//
//   - N INDEPENDENTLY-ADJUDICATED CALLS, never one fan-out that bypasses the floor.
//     Each member is a full k.Syscall, so a vote member bound for a REMOTE model still
//     crosses the residency/policy floor and is DENIED for a tenant/sensitive payload.
//   - MEMBER ORDER PRESERVED INTO THE FOLD. votes are appended in Plan.Members order,
//     so ReduceConcat / ReduceVote tie-breaks stay deterministic. A member the kernel
//     refused (or that errored at dispatch) contributes NO vote; the survivors keep
//     their relative order.
//   - FAIL CLOSED on a wipeout. If EVERY member was refused, there is no silent empty
//     success — the last member's refusal verdict is surfaced (so a residency/policy
//     reason reaches the wire) and the result Status is ERROR.
//
// vDSO interaction: a member contributes a vote iff its result Status is OK (a refused
// member's Reap yields a Status=Error deny-as-value). For the canonical write-shaped
// ensemble (a guard quorum over a destructive tool) the vDSO never dedups, so every
// member's engine actually runs. A read-only idempotent ensemble may have later members
// served from an earlier member's tier-2 fill — consistent with fak's engine-independence
// model for idempotent reads (the same bytes regardless of which engine), where an
// ensemble adds nothing anyway.
func (s *Server) dispatchEnsemble(ctx context.Context, base *abi.ToolCall, plan modelroute.Plan) (WireVerdict, *ResultEnvelope, error) {
	votes := make([]modelroute.Vote, 0, len(plan.Members))
	var lastRefused abi.Verdict
	refused := 0
	for _, mem := range plan.Members {
		r, v := s.k.Syscall(ctx, memberCall(base, mem.Model))
		if r == nil || r.Status != abi.StatusOK {
			lastRefused = v
			refused++
			continue
		}
		votes = append(votes, modelroute.Vote{Member: mem, Output: string(resolveBytes(ctx, r.Payload))})
	}
	if len(votes) == 0 {
		// Every member was refused or errored at dispatch — fail closed (never a silent
		// empty success). Surface the last refusal verdict so the residency/policy reason
		// reaches the wire; default to a plain deny if the verdict was somehow non-refusing.
		wv := renderVerdict(lastRefused, nil)
		if wv.Kind == "ALLOW" {
			wv = WireVerdict{Kind: "DENY", Reason: abi.ReasonName(abi.ReasonPolicyBlock), By: "modelroute-ensemble", Disposition: "TERMINAL"}
		}
		env := &ResultEnvelope{Status: "ERROR", Content: "", Meta: map[string]string{
			"served_by":        "modelroute-ensemble",
			"ensemble_refused": itoa(uint64(refused)),
		}}
		return wv, env, nil
	}
	folded, ferr := modelroute.Combine(plan.Reduce, votes)
	if ferr != nil {
		// A misconfigured reduce over incompatible outputs (e.g. all_reduce over
		// non-numeric tool results) is a fail-loud error, never a silent guess.
		return WireVerdict{}, nil, fmt.Errorf("gateway: ensemble combine: %w", ferr)
	}
	meta := map[string]string{
		"served_by":        "modelroute-ensemble",
		"reduce":           string(folded.Reduce),
		"ensemble_members": itoa(uint64(folded.Members)),
	}
	if refused > 0 {
		meta["ensemble_refused"] = itoa(uint64(refused))
	}
	if folded.Winner != "" {
		meta["winner"] = folded.Winner
	}
	return WireVerdict{Kind: "ALLOW", By: "modelroute-ensemble"},
		&ResultEnvelope{Status: "OK", Content: folded.Output, Meta: meta}, nil
}

// memberCall clones a base ToolCall for one ensemble member, binding THAT member's
// model to Engine before submission (the pre-submit residency contract) and giving the
// call a fresh identity (SeqNo unset, an independent Meta copy) so the kernel
// adjudicates and dispatches each member on its own. The content-addressed Args Ref is
// shared — every member sees the same input — while the Meta map is copied so a
// per-call kernel annotation can never leak across members.
func memberCall(base *abi.ToolCall, model string) *abi.ToolCall {
	meta := make(map[string]string, len(base.Meta))
	for k, v := range base.Meta {
		meta[k] = v
	}
	return &abi.ToolCall{
		Tool:    base.Tool,
		Args:    base.Args,
		TraceID: base.TraceID,
		Meta:    meta,
		Engine:  model,
	}
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
	// Lower the request's isolation principal (a tenant / user / auth subject, carried
	// request-scoped on ctx from the X-Fak-Principal header or the request's principal
	// field) onto the call so the vDSO scopes its tier-2 cache entry PER PRINCIPAL: a
	// different principal can neither be served nor fill the same (tool,args) entry,
	// closing the cross-tenant cache leak + the hit/miss timing oracle. Empty =>
	// single-tenant (every caller shares, the v0.1 behavior).
	if p := principalFromContext(ctx); p != "" {
		meta[vdso.MetaPrincipal] = p
	}
	// Thread a TraceID end-to-end: the IFC ledger + plan-CFI key their per-session
	// state on it, so a served call MUST carry one. The wire supplies it for
	// cross-call correlation; absent, we mint a fresh non-empty id rather than fall
	// back to the empty shared-default trace (which would pool every served session
	// onto one taint high-water mark).
	tc := &abi.ToolCall{Tool: tool, Args: ref, TraceID: s.traceFor(traceID), Meta: meta}
	// Per-call model routing (opt-in): classify this tool call into a routing Subject
	// and, for a single-model PICK, bind the chosen model to Engine HERE — before the
	// caller hands tc to k.Syscall. That is the load-bearing residency contract: the
	// residency PDP reads c.Engine INSIDE the adjudication fold, so a route written
	// any later (at Reap/dispatch) would adjudicate an empty Engine and fail open on a
	// tenant payload bound for a remote model. nil manifest => Engine "" => kernel
	// default (byte-for-byte the pre-routing path).
	tc.Engine = s.routeEngine(tool, readOnly, meta)
	return tc, nil
}

// routeDecision classifies a tool call into a modelroute.Subject (aspect=tool_call,
// the tool name, and the read-only / sensitivity / tenant signals the gateway already
// attests) and returns the manifest's routing Decision. The second return is false
// when no manifest is configured (the kernel-default path). routeEngine and
// ensemblePlan share this single classification so the single-model and ensemble
// paths can never diverge on what a call routes to.
func (s *Server) routeDecision(tool string, readOnly bool, meta map[string]string) (modelroute.Decision, bool) {
	if s.route == nil {
		return modelroute.Decision{}, false
	}
	return s.route.Route(modelroute.Subject{
		Aspect: modelroute.AspectToolCall,
		Tool:   tool,
		Labels: routeLabels(readOnly, meta),
	}), true
}

// routeEngine consults the optional per-call routing policy and returns the engine
// route to bind to abi.ToolCall.Engine, or "" for the kernel default. It returns
// Decision.Plan.Primary() for a single-model PICK. An ENSEMBLE plan is left to the
// kernel default here (route ""): the N-submit fan-out happens at dispatch time in
// dispatchEnsemble (the syscall path), and collapsing an ensemble to one member here
// would be a silent wrong route. The returned route is the model id verbatim
// (Plan.Primary()'s documented destination), NOT collapsed to a registered engine id —
// the string must keep the model's remote-ness so the residency gate can deny a
// tenant/sensitive payload bound for a remote model. A route to a model with no
// registered engine driver fails LOUD at dispatch ("no engine registered for route"),
// never silently runs elsewhere.
func (s *Server) routeEngine(tool string, readOnly bool, meta map[string]string) string {
	d, ok := s.routeDecision(tool, readOnly, meta)
	if !ok || d.Plan.IsEnsemble() {
		return ""
	}
	return d.Plan.Primary()
}

// ensemblePlan returns the routing Plan for this call WHEN it is a multi-member
// ensemble, so the syscall path can fan it out (issue #597). A single-model PICK, or
// no manifest, returns ok=false (the call dispatches once on the route routeEngine
// already bound to Engine). The classification is identical to routeEngine's — same
// Subject, same routeDecision — so the two never disagree on whether a call is an
// ensemble.
func (s *Server) ensemblePlan(tool string, readOnly bool, meta map[string]string) (modelroute.Plan, bool) {
	d, ok := s.routeDecision(tool, readOnly, meta)
	if !ok || !d.Plan.IsEnsemble() {
		return modelroute.Plan{}, false
	}
	return d.Plan, true
}

// routeLabels lowers the call signals the gateway honestly knows into the OPEN
// Subject.Labels a manifest Match can route on: read_only (read- vs write-shaped),
// and the sensitivity / tenant tags the residency floor also reads. Per-call prompt
// token estimation and richer classification are a later signal-enrichment child
// (#599 scout classification); the gateway routes on what it can attest today.
func routeLabels(readOnly bool, meta map[string]string) map[string]string {
	labels := map[string]string{"read_only": boolLabel(readOnly)}
	if meta != nil {
		sens := meta["sensitivity"]
		if sens == "" {
			sens = meta["data_sensitivity"]
		}
		if sens != "" {
			labels["sensitivity"] = sens
		}
		if p := meta[vdso.MetaPrincipal]; p != "" {
			labels["tenant"] = p
		}
	}
	return labels
}

// boolLabel renders a bool as a routing-label string ("true"/"false") without
// pulling strconv into this file (it formats ints via the local itoa).
func boolLabel(b bool) string {
	if b {
		return "true"
	}
	return "false"
}

// traceFor returns the caller's TraceID, or mints a fresh, process-unique non-empty
// one so the result-side IFC ledger + plan-CFI never collapse distinct served
// sessions onto the empty-string default trace.
func (s *Server) traceFor(traceID string) string {
	traceID = strings.TrimSpace(traceID)
	if traceID != "" {
		return traceID
	}
	s.defaultTraceMu.RLock()
	defaultTraceID := s.defaultTraceID
	s.defaultTraceMu.RUnlock()
	if defaultTraceID != "" {
		return defaultTraceID
	}
	return "gw-" + itoa(atomic.AddUint64(&s.traceSeq, 1))
}

// SetDefaultTraceID changes the trace used for callers that omit X-Trace-Id /
// trace_id. Guard's budget-restart supervisor uses this when it relaunches a child
// under a continuation id; a blank value restores the historical minted gw-N default.
func (s *Server) SetDefaultTraceID(traceID string) {
	if s == nil {
		return
	}
	s.defaultTraceMu.Lock()
	s.defaultTraceID = strings.TrimSpace(traceID)
	s.defaultTraceMu.Unlock()
}

// principalCtxKey is the context key carrying a request's isolation principal.
type principalCtxKey struct{}

// WithPrincipal returns a context carrying the caller's isolation principal (a tenant /
// user / auth subject). buildCall lowers it onto ToolCall.Meta[vdso.MetaPrincipal] so
// the vDSO scopes tier-2 cache entries per principal — a different principal can neither
// read nor fill the same (tool,args) entry, closing the cross-tenant cache leak + the
// hit/miss timing oracle. An empty principal returns ctx unchanged (single-tenant: every
// caller shares, the v0.1 behavior). Exported so a host embedding the gateway can set the
// principal from its own auth context before calling Syscall.
func WithPrincipal(ctx context.Context, principal string) context.Context {
	principal = strings.TrimSpace(principal)
	if principal == "" {
		return ctx
	}
	return context.WithValue(ctx, principalCtxKey{}, principal)
}

// principalFromContext returns the request-scoped isolation principal, or "" if none.
func principalFromContext(ctx context.Context) string {
	if ctx == nil {
		return ""
	}
	p, _ := ctx.Value(principalCtxKey{}).(string)
	return p
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
	// Snapshot each message's ORIGINAL content before admission rewrites any quarantined
	// payload in place. The in-kernel poison-eviction hook needs the original (poisoned)
	// bytes to render the token path that was actually cached, not the paged-out form.
	origContent := make([]string, len(messages))
	for i := range messages {
		origContent[i] = messages[i].Content
	}
	var admissions []ResultAdmission
	var quarantinedIdx []int
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
			quarantinedIdx = append(quarantinedIdx, i)
			continue
		}
		// On a quarantine/transform the kernel paged the bytes out and rewrote the
		// payload in place; forward the paged-out form so the poison never reaches
		// the model. A plain Allow leaves the content untouched.
		if envlp != nil && (wv.Kind == "QUARANTINE" || wv.Kind == "TRANSFORM") {
			messages[i].Content = envlp.Content
		}
		if wv.Kind == "QUARANTINE" {
			quarantinedIdx = append(quarantinedIdx, i)
		}
		admissions = append(admissions, ResultAdmission{
			ToolCallID: messages[i].ToolCallID,
			Tool:       messages[i].Name,
			Verdict:    wv,
		})
	}
	// Defense in depth (candidate #14): a result the kernel just quarantined may have been
	// admitted as benign on an EARLIER turn and prefilled into the in-kernel KV cache. Drop
	// any cached prefix that attended to it so a later turn re-prefills instead of replaying
	// the poisoned KV. Fires on the SAME quarantine event as the external engine-cache reset.
	s.evictInKernelPoison(messages, origContent, quarantinedIdx)
	if err := s.resetEngineCacheAfterQuarantine(ctx, admissions); err != nil {
		return admissions, err
	}
	return admissions, nil
}

// evictInKernelPoison drives the in-kernel poison eviction when the chat backend is the
// in-kernel planner. It drives TWO complementary seams on the SAME quarantine event, each a
// no-op on a planner that does not implement it (proxy/mock, or the seam left off):
//
//   - agent.PoisonEvictor — drops the reusable RadixAttention PREFIX node along the poisoned
//     path so a later turn re-prefills instead of replaying the poisoned KV (candidate #14).
//   - agent.KVSpanEvictor — the model-side KV-quarantine eviction BRIDGE (issue #579): it
//     rebuilds the transcript's per-message K/V SPANS on a fresh model.Session over the loaded
//     model and evicts the quarantined result's span via the proven model.KVCache.Evict
//     (re-RoPE + renumber), so the live session's attention state is bit-identical to a run
//     that never saw the poison — the flagship guarantee, now fired by a LIVE request and not
//     only the synthetic-model unit witness. DEFAULT OFF (FAK_INKERNEL_KVMMU opts in).
//
// The transcript is rendered with each message's ORIGINAL content so the evicted token path
// matches what the cache actually prefilled before the verdict paged the bytes out.
func (s *Server) evictInKernelPoison(messages []agent.Message, origContent []string, quarantinedIdx []int) {
	if len(quarantinedIdx) == 0 {
		return
	}
	prefixEv, hasPrefix := s.planner.(agent.PoisonEvictor)
	spanEv, hasSpan := s.planner.(agent.KVSpanEvictor)
	if !hasPrefix && !hasSpan {
		return
	}
	restored := make([]agent.Message, len(messages))
	copy(restored, messages)
	for i := range restored {
		if i < len(origContent) {
			restored[i].Content = origContent[i]
		}
	}
	for _, idx := range quarantinedIdx {
		if hasPrefix {
			if freed := prefixEv.EvictPoisoned(restored, idx); freed > 0 {
				s.logf("gateway: in-kernel KV prefix evicted on tool-result quarantine msg=%d freed=%dtok", idx, freed)
			}
		}
		if hasSpan {
			// Default-off bridge: a no-op (0,false) unless FAK_INKERNEL_KVMMU opted it in, so the
			// served path is unchanged by default. When on, a non-zero freed span proves the live
			// KVCache.Evict fired; reposition_exact records the bit-exact never-saw invariant.
			if freed, exact := spanEv.EvictKVSpan(restored, idx); freed > 0 {
				s.logf("gateway: in-kernel KV span evicted on tool-result quarantine msg=%d freed=%dpos reposition_exact=%v", idx, freed, exact)
			}
		}
	}
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
