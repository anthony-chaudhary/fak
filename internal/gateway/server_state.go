package gateway

import (
	"context"
	"sync"
	"sync/atomic"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/auditreceipt"
	"github.com/anthony-chaudhary/fak/internal/bgloop"
	"github.com/anthony-chaudhary/fak/internal/cachemeta"
	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
	"github.com/anthony-chaudhary/fak/internal/enginecache"
	"github.com/anthony-chaudhary/fak/internal/guardrsi"
	"github.com/anthony-chaudhary/fak/internal/kernel"
	"github.com/anthony-chaudhary/fak/internal/modelroute"
	"github.com/anthony-chaudhary/fak/internal/nativeperf"
	"github.com/anthony-chaudhary/fak/internal/rungobs"
	"github.com/anthony-chaudhary/fak/internal/toolplugin"
)

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
	TraceID          string                  `json:"trace_id"`
	Run              string                  `json:"run"`
	Budget           SessionBudget           `json:"budget"`
	Priority         int                     `json:"priority"`
	Pace             SessionPace             `json:"pace"`
	Reason           string                  `json:"reason,omitempty"`
	ContinuationID   string                  `json:"continuation_id,omitempty"`
	ParentTrace      string                  `json:"parent_trace,omitempty"`
	Generation       int                     `json:"generation,omitempty"`
	CacheAffinity    SessionCacheAffinity    `json:"cache_affinity,omitempty,omitzero"`
	ResetTransaction SessionResetTransaction `json:"reset_transaction,omitempty,omitzero"`
	ProviderBoundary SessionProviderBoundary `json:"provider_boundary,omitempty,omitzero"`
	Assumptions      []SessionAssumption     `json:"assumptions,omitempty"`
	// Time is the session's WALL-CLOCK budget projection (issue #1584): elapsed,
	// remaining, limit, and whether the envelope is exceeded. It is the observability
	// twin of Budget's token axes — the field that makes `--max-duration` (and the
	// managed-context wall axis) legible in `fak session status`, which the flag's own
	// help promises but a bare Budget/Pace projection could never deliver. Advisory /
	// read-only: nothing gates on it, and the zero value (no wall-clock envelope, never
	// started) marshals away via omitzero so a session with no time budget keeps the
	// pre-#1584 wire shape byte-for-byte.
	Time SessionTime `json:"time,omitempty,omitzero"`
	// Throughput is the throughput envelope's read projection (#2762): the
	// configured expected/min rates plus the observed sustained rate. Advisory /
	// read-only, like Time; the zero value (axis unconfigured, nothing observed)
	// marshals away via omitzero so the pre-#2762 wire shape is unchanged.
	Throughput SessionThroughput `json:"throughput,omitempty,omitzero"`
	Activity   ActionCounts      `json:"activity,omitempty,omitzero"`
	Rev        uint64            `json:"rev"`
}

// SessionTime is the gateway wire projection of internal/session.TimeBudget's read-only
// Query verdict: the wall-clock allotment an operator sees in `fak session status`.
// Durations are carried as whole seconds (a wall-clock budget is minutes-to-hours scale,
// so seconds is both lossless-enough and legible in the raw `--json` form). Bounded is
// false when no envelope is configured, in which case Remaining/Limit are 0 and only
// Elapsed (if the clock ever started) is meaningful.
type SessionTime struct {
	Bounded          bool  `json:"bounded,omitempty"`
	Exceeded         bool  `json:"exceeded,omitempty"`
	ElapsedSeconds   int64 `json:"elapsed_seconds,omitempty"`
	RemainingSeconds int64 `json:"remaining_seconds,omitempty"`
	LimitSeconds     int64 `json:"limit_seconds,omitempty"`
}

// IsZero reports whether no wall-clock budget is attached (nothing configured and no
// time ticked), for json omitzero — so a pre-#1584 session's wire form is unchanged.
func (t SessionTime) IsZero() bool { return t == SessionTime{} }

// SessionAssumption is the gateway's wire-neutral projection of an active
// session assumption. The host owns how rows are minted; gateway only carries
// provenance, confidence, and expiry to operator/debug surfaces.
type SessionAssumption struct {
	TraceID    string  `json:"trace_id,omitempty"`
	Key        string  `json:"key"`
	Statement  string  `json:"statement,omitempty"`
	Source     string  `json:"source"` // user_stated, inferred, queried, witnessed, stale, unknown
	Confidence float64 `json:"confidence,omitempty"`
	Expiry     string  `json:"expiry,omitempty"`
	SourceRef  string  `json:"source_ref,omitempty"`
}

// SessionCacheAffinity is the gateway wire form of session.CacheAffinityDecision.
// It is provider-neutral and audit-only: hosts can derive provider-specific routing
// hints from AffinityKey, but correctness must not depend on that hint landing.
type SessionCacheAffinity struct {
	Action      string `json:"action,omitempty"`
	AffinityKey string `json:"affinity_key,omitempty"`
	FromTraceID string `json:"from_trace_id,omitempty"`
	ToTraceID   string `json:"to_trace_id,omitempty"`
	Reason      string `json:"reason,omitempty"`
}

// IsZero reports whether the decision is absent, for json omitzero.
func (d SessionCacheAffinity) IsZero() bool {
	return d.Action == "" && d.AffinityKey == "" && d.FromTraceID == "" && d.ToTraceID == "" && d.Reason == ""
}

// SessionResetTransaction is the gateway wire form of session.ResetTransaction.
type SessionResetTransaction struct {
	Schema           string                    `json:"schema,omitempty"`
	OldTrace         string                    `json:"old_trace,omitempty"`
	NewTrace         string                    `json:"new_trace,omitempty"`
	SeedDigest       string                    `json:"seed_digest,omitempty"`
	Contributors     []string                  `json:"contributors,omitempty"`
	OmittedSpans     []SessionResetOmittedSpan `json:"omitted_spans,omitempty"`
	BudgetRearm      SessionResetBudgetRearm   `json:"budget_rearm,omitempty,omitzero"`
	WarmPrefixDigest string                    `json:"warm_prefix_digest,omitempty"`
}

// SessionProviderBoundary is the gateway wire form of an explicit provider-side
// conversation reset such as /clear.
type SessionProviderBoundary struct {
	Schema            string `json:"schema,omitempty"`
	Provider          string `json:"provider,omitempty"`
	Source            string `json:"source,omitempty"`
	PreviousTrace     string `json:"previous_trace,omitempty"`
	ProviderSessionID string `json:"provider_session_id,omitempty"`
}

// IsZero reports whether no provider boundary is attached.
func (b SessionProviderBoundary) IsZero() bool { return b == (SessionProviderBoundary{}) }

// IsZero reports whether no reset transaction was attached.
func (tx SessionResetTransaction) IsZero() bool {
	return tx.Schema == "" && tx.OldTrace == "" && tx.NewTrace == "" &&
		tx.SeedDigest == "" && len(tx.Contributors) == 0 && len(tx.OmittedSpans) == 0 &&
		tx.BudgetRearm == (SessionResetBudgetRearm{}) && tx.WarmPrefixDigest == ""
}

// SessionResetBudgetRearm records the fresh budget armed by a reset.
type SessionResetBudgetRearm struct {
	TurnsLeft         int `json:"turns_left"`
	TokensLeft        int `json:"tokens_left"`
	ContextTokensLeft int `json:"context_tokens_left,omitempty"`
	ContextTokensCap  int `json:"context_tokens_cap,omitempty"`
}

// SessionResetOmittedSpan is a payload-free pointer to an omitted source span.
type SessionResetOmittedSpan struct {
	Index  int    `json:"index"`
	Role   string `json:"role,omitempty"`
	Digest string `json:"digest"`
	Reason string `json:"reason,omitempty"`
}

// SessionBudget is the wire form of internal/session.Budget. TurnsLeft/TokensLeft
// at -1 (session.Unbounded) mean no cap; ContextTokensLeft uses 0 as off and a
// positive value as the long-window reset budget.
//
// ContextTokensCap and ResidentContextTokens carry the two extra signals the outbound
// history-compaction burst gate reads on a context-budgeted-but-turn-unbounded session (the
// common headless `fak guard -- claude` shape, where TurnsLeft stays Unbounded): the configured
// context ceiling and the last debited turn's resident window. Both are 0 (omitempty) on a
// session with no context budget or no debited turn yet, so an un-budgeted session marshals
// byte-identically to the pre-field wire form and the gate derives NO horizon from them.
type SessionBudget struct {
	TurnsLeft             int `json:"turns_left"`
	TokensLeft            int `json:"tokens_left"`
	ContextTokensLeft     int `json:"context_tokens_left,omitempty"`
	ContextTokensCap      int `json:"context_tokens_cap,omitempty"`
	ResidentContextTokens int `json:"resident_context_tokens,omitempty"`
	// SpendMicroCentsLeft/Cap mirror internal/session.Budget's priced spend axis
	// (#2762): the remaining/configured dollar ceiling in micro-cents (1e-8 USD).
	// 0 = no spend budget. Carried on the wire so the envelope control route can
	// SET a spend ceiling and a budget read-modify-write can PRESERVE one instead
	// of silently clearing it.
	SpendMicroCentsLeft int64 `json:"spend_micro_cents_left,omitempty"`
	SpendMicroCentsCap  int64 `json:"spend_micro_cents_cap,omitempty"`
}

// SessionPace is the wire form of internal/session.Pace. Zero on either axis means
// "no opinion" (the planner's own default stands).
type SessionPace struct {
	MaxTokensPerTurn int `json:"max_tokens_per_turn"`
	MinTurnGapMs     int `json:"min_turn_gap_ms"`
}

// SessionWall is the wire form of the wall-clock LIMIT the control route applies
// (#2762): the total envelope in nanoseconds, mirroring
// internal/session.TimeBudget.LimitNanos. <=0 clears the envelope.
type SessionWall struct {
	LimitNanos int64 `json:"limit_nanos"`
}

// SessionThroughput is the wire form of internal/session.ThroughputBudget's
// configured rates (#2762): the soft expected pace-shaping rate and the enforced
// minimum sustained-rate floor. ObservedTokensPerSec is a read-only projection on
// SessionState (the measured sustained rate the floor is judged against); it is
// ignored on control writes.
type SessionThroughput struct {
	ExpectedTokensPerSec float64 `json:"expected_tokens_per_sec,omitempty"`
	MinTokensPerSec      float64 `json:"min_tokens_per_sec,omitempty"`
	ObservedTokensPerSec float64 `json:"observed_tokens_per_sec,omitempty"`
}

// ActionCounts records tool execution and effect activity metrics.
type ActionCounts struct {
	Tools   int `json:"tools,omitempty"`
	Execs   int `json:"execs,omitempty"`
	Fail    int `json:"fail,omitempty"`
	Effects int `json:"effects,omitempty"`
}

// IsZero reports whether no actions have been counted, for json omitzero.
func (a ActionCounts) IsZero() bool { return a == (ActionCounts{}) }

// SessionControlRequest is the gateway-parsed body of POST
// /v1/fak/session/{trace_id}/{verb}. Exactly the field named by the verb is read;
// the others are ignored. if_rev, when non-zero, is the optimistic-concurrency
// guard: the write is taken only if the session's current Rev matches, else the
// route returns 409 (the client re-reads and retries).
type SessionControlRequest struct {
	Run        string             `json:"run,omitempty"`        // verb "run": target run-state token
	Reason     string             `json:"reason,omitempty"`     // verb "run": reason token (closed vocabulary)
	Budget     *SessionBudget     `json:"budget,omitempty"`     // verb "budget", and the mid-flight "set-budget" (#2403)
	CallID     string             `json:"call_id,omitempty"`    // verb "drop-pending-call" (#2403): the one queued call to skip
	Pace       *SessionPace       `json:"pace,omitempty"`       // verb "pace"
	Priority   *int               `json:"priority,omitempty"`   // verb "priority"
	Wall       *SessionWall       `json:"wall,omitempty"`       // verb "wall" (#2762): wall-clock limit
	Throughput *SessionThroughput `json:"throughput,omitempty"` // verb "throughput" (#2762): expected rate + min floor
	IfRev      uint64             `json:"if_rev,omitempty"`     // optional CAS guard
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
	// Principal OPTIONALLY names a non-operator steer source (e.g. the machine guard
	// "doomloop-guard" delivering a re-anchor nudge, #3529). Empty ⇒ the human "operator"
	// default. It is an ATTRIBUTION label only and cannot elevate trust: the a2achan floor
	// gates a steer on caps+taint+scope, never on the `from` string, and the steer body is
	// always Tainted/ScopeFleet — so a client-supplied principal only truthfully names a
	// machine source, it never buys the input more authority than an operator steer has.
	Principal string `json:"principal,omitempty"`
	// Class OPTIONALLY carries the scheduler class of the append (#2402): "now" (interrupt
	// at the next safe boundary — it lands before the next tool dispatch), "next" (fold into
	// the next querying turn — the default and the legacy behavior), or "later" (hold until
	// the loop would otherwise idle). Empty ⇒ "next"; an unrecognized value is a 400. See
	// SteerClass / steer_class.go.
	Class string `json:"class,omitempty"`
	// Query is the query bit (#2402): a nil pointer or true means the append forces a model
	// turn (legacy steer semantics); an explicit false means "context arrived" — the append
	// is screened, taint-stamped, and staged for the next querying turn WITHOUT scheduling a
	// planner call of its own. This decoupling of context arrival from a spent model turn is
	// what makes cheap continuous observation of a running loop affordable.
	Query *bool `json:"query,omitempty"`
}

// SteerSessionFunc is injected by the host CLI so the gateway can enqueue an operator
// steer onto the host's a2achan bus (Session locale, keyed by traceID) without importing
// internal/a2achan. A non-nil error is the adjudication floor's deny-as-value surfaced
// (tainted / over-scoped / uncapped body), which the route maps to 422 — the same
// default-deny floor that gates a tool call, here gating operator input. nil hook ⇒ the
// steer route is not configured (404).
type SteerSessionFunc func(ctx context.Context, traceID, principal, text string) error

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

// StopGateResult is the evidence check at an owned-loop completion boundary.
// Witness names the declaration in audit and model feedback when Satisfied is false.
type StopGateResult struct {
	Satisfied bool
	Witness   string
}

// StopGateFunc evaluates declared completion evidence. It must read an external
// effect, not trust the model's completion prose.
type StopGateFunc func(ctx context.Context, traceID string) StopGateResult

// SessionUsage is the gateway's session-table-neutral token accounting for one
// served request. CompletionTokens debits the historical output budget; ContextTokens
// is the provider-normalized prompt/context window for the long-session reset budget.
type SessionUsage struct {
	PromptTokens             int `json:"prompt_tokens,omitempty"`
	CompletionTokens         int `json:"completion_tokens,omitempty"`
	ContextTokens            int `json:"context_tokens,omitempty"`
	CacheReadInputTokens     int `json:"cache_read_input_tokens,omitempty"`
	CacheCreationInputTokens int `json:"cache_creation_input_tokens,omitempty"`
	// DurationNanos is the served turn's real wall-clock duration, feeding the
	// session throughput axis's sustained-rate observation (#2762). 0 = unknown.
	DurationNanos int64 `json:"duration_nanos,omitempty"`
}

// SessionDebitFunc is injected by the host CLI to run session.Table.DebitUsage with
// the token usage reported after a served request finishes.
type SessionDebitFunc func(ctx context.Context, traceID string, usage SessionUsage) SessionState

// ResetOnBudgetFunc is the host's reset action (Config.ResetOnBudget). Given a session's
// trace and its canonical transcript, the host builds the carryover seed (durable facts +
// task recap + warm-prefix + verbatim tail), calls session.Table.Recontinue to re-arm a
// fresh session, and returns the fresh trace id plus the seed messages the gateway prepends
// to the live request. ok=false means the host declined to reset (no carryover) — the gateway
// then falls back to the historical refusal (budget path) or proceeds unchanged (coherence
// path). It is invoked from TWO triggers: a budget drain (maybeResetOnBudget, the refusal
// path) and a #3159 compaction-coherence thrash (maybeResetOnCoherence, the admitted path) —
// so the host should not assume a budget exhaustion. The gateway stays session-internals-blind:
// it never imports internal/session or internal/sessionreset; the host owns both.
type ResetOnBudgetFunc func(ctx context.Context, trace string, messages []agent.Message) (newTrace string, seed []agent.Message, ok bool)

// BudgetExhaustedFunc is injected by hosts that supervise a real child process.
// It fires after a served turn's post-response usage debit drains a resettable
// budget, while the transcript for that turn is still available.
type BudgetExhaustedFunc func(ctx context.Context, st SessionState, messages []agent.Message)

// Server is a configured, ready-to-serve gateway. Construct with New; serve with
// Handler()/ListenAndServe (HTTP) or ServeStdio (MCP over stdin/stdout).
type Server struct {
	nativeProgress  nativeProgressReplay
	k               *kernel.Kernel
	toolPlugins     []toolplugin.Plugin
	toolPreferences toolplugin.PreferenceLayers
	engineID        string
	model           string
	richDashboards  *richDashboardManager
	requireKey      string
	// readBearer is the read-scoped observability bearer (Config.ReadBearer): accepted
	// ONLY on the diagnostic reads (/debug/vars, /metrics, /v1/fak/observation),
	// never on a mutating route.
	readBearer string
	// keyset binds additional api keys to org/project isolation principals
	// (Config.KeyPrincipals, #5332), matched in withAuth by a constant-time digest
	// compare. nil => RequireKey-only auth, unchanged. Holds only key DIGESTS — the raw
	// keys are hashed and dropped in New, never retained here.
	keyset *keyset
	// exposeUpstreamErrorDetail folds a scrubbed, bounded snapshot of the upstream's
	// own 400 body into the client-facing message. TRUE only on the trusted local
	// path (fak guard, loopback-bound); FALSE (the default) keeps the no-leak
	// #82/#346 boundary the externally-exposed serve path relies on. See Config.
	exposeUpstreamErrorDetail bool
	// denialRecoveryOff mirrors Config.DenialRecoveryOff: recovery is armed unless the
	// host stood it down. Negative sense so the zero-value Server recovers, matching the
	// field it copies.
	denialRecoveryOff        bool
	upstreamBadRequestNotify func(detail string)
	version                  string
	logf                     func(format string, args ...any)
	debugStatsf              func(format string, args ...any) // optional per-turn human debug sink (#793); nil = off
	turnCacheStatsMu         sync.Mutex
	turnCacheStats           turnCacheHistory // successful-turn cache economics rendered by fak-turn
	feed                     *coherenceFeed   // the cross-agent "what changed" feed (vdso coherence bus)
	sessionFeed              *sessionFeed     // the drive-state revision feed (#630; host-pushed via PublishSessionRevision)
	metrics                  *gatewayMetrics
	otlp                     *otlpExporter
	orgAudit                 *auditreceipt.Exporter
	traceparentInvalid       uint64
	// toolPages is the tool catalog's home (#2440): each advertised tool schema is a
	// content-hashed read-only page owned by the ctxmmu, registered at the
	// maybeCompactInboundTools seam. The page table — not the transcript — is the
	// source of truth, so compaction can only evict a schema re-faultably, never lose
	// it; identical schemas dedupe across turns/sessions by content hash. Its
	// ResidentBytes/DedupHits back the tool_schema_resident_bytes and
	// tool_page_dedup_hits_total /metrics rows. Built in New; nil-safe for a bare Server.
	toolPages               *ctxmmu.ToolPageTable
	servedFailure           servedFailure // recent served-turn panic behind /healthz honesty (#2336); see served_failure.go
	traceSeq                uint64        // mints a non-empty TraceID when the wire omits one (atomic)
	reloadPolicy            PolicyReloadFunc
	policyCanaryTurns       int
	policyCanaryMu          sync.Mutex
	policyCanaryRemaining   int
	policyCanaryConsecutive int
	policyCanaryRollback    func()
	policyCanaryRolledBack  atomic.Bool
	observePolicy           PolicyObserveFunc
	resetTrace              TraceResetFunc
	observeTrace            TraceObserveFunc
	observeSession          SessionObserveFunc
	controlSession          SessionControlFunc
	steerSession            SteerSessionFunc
	listSessions            SessionListFunc
	trajctlMetrics          TrajctlMetricsFunc
	decideSession           SessionDecideFunc
	stopGate                StopGateFunc
	debitSession            SessionDebitFunc
	resetOnBudget           ResetOnBudgetFunc
	budgetDrained           BudgetExhaustedFunc
	defaultTraceMu          sync.RWMutex
	defaultTraceID          string

	// warmup is the #3051 boot warmup-inference readiness gate behind /healthz:
	// when the host arms it, /healthz reports ok:false (warmup_pending) until a
	// synthetic warmup inference returns its first token (MarkWarmupComplete), so
	// the operator's first real turn is warm-path, not the ~500s cold tax. Exposes
	// time_to_ready_ms once complete. Default-silent for a serve that never arms it.
	// See readiness_warmup.go.
	warmup warmupGate

	// startupDecode is the #3051-sibling boot fixed-prompt decode-coherence gate
	// behind /healthz: SetStartupDecodeProbe records the host's deterministic probe
	// output and classifyDecode folds a degenerate decode (empty/punctuation/repeated
	// token) into an ok:false reason until the process restarts and re-probes. Zero
	// value == not probed, so a proxy/mock serve that never calls it is unaffected.
	// See readiness_decode.go.
	startupDecode startupDecodeProbe

	// routeWatcher is the model-routing manifest hot-reload seam behind POST
	// /v1/fak/route/reload (#4003) — the SIGHUP-style manual twin of the background
	// Watcher.Run poll loop. It is an atomic pointer because the host installs the
	// watcher AFTER New (the watcher needs RouteLive), while a request may hit the
	// route concurrently; a nil load means routing hot-reload is not configured and
	// the route answers 404, mirroring a nil reloadPolicy. Set via SetRouteWatcher.
	routeWatcher atomic.Pointer[modelroute.Watcher]

	// policyReloads counts the SUCCESSFUL POST /v1/fak/policy/reload swaps this
	// process has served; it backs PolicyObservation.ReloadCount so a GET
	// /v1/fak/policy answer distinguishes a floor that has been hot-swapped under the
	// operator from one that has stood since launch (#3960). Gateway-owned on
	// purpose: the injected host func never sets it, so the count cannot be spoofed
	// by the policy loader. Atomic because a reload POST and an observe GET race.
	policyReloads atomic.Int64

	// boundAddr is the address this process is actually listening on, recorded by
	// Serve once the listener exists (#5642). Before it, nothing in the Server knew
	// its own dialable address, so the A2A agent card advertised a literal
	// `fleet.example.com` — a served descriptor pointing at a host that is not fak.
	// It is an atomic pointer because Serve writes it while a handler may read it
	// concurrently; a nil load means "not serving on a listener we own" (a bare
	// Handler under httptest), in which case the request's own Host is the only
	// answer. See a2aSelfBaseURL.
	boundAddr atomic.Pointer[string]

	// loops is the in-kernel background-loop supervisor (internal/bgloop): the
	// runtime that keeps registered recurring loops progressing while the gateway is
	// up, observable via /v1/fak/loops and the fak_bgloop_* metrics. Started on the
	// serve lifecycle context in Serve, joined on shutdown. Built in New (never nil
	// for a New'd Server; nil only in a bare zero value).
	loops *bgloop.Supervisor

	// activity is the bounded per-trace activity registry behind the agents pane's
	// live-status cell (#2627): the last admitted tool, the subagent-spawn count, and
	// the in-flight/idle age of each live trace, projected onto debugSessionVars.
	// Payload-free (tool NAME only). Built in New; nil-safe for a bare Server. See
	// session_activity.go.
	activity *sessionActivity

	// startup is the one-time boot timeline (start -> ready, per-phase costs),
	// exposed as fak_gateway_startup_* gauges. See startup.go.
	startup *startupProfile
	// modelLoad is the optional boot-time weight-load breakdown set by the host via
	// SetModelLoadProfile when it eagerly loads a model (fak serve --gguf). nil
	// suppresses every fak_model_load_* metric. Guarded by modelLoadMu.
	modelLoadMu sync.Mutex
	modelLoad   *ModelLoadProfile

	// endpointsProvider is the optional pull source for the live accounts+nodes block
	// on /debug/vars (the "endpoints" block). fak guard wires it to a closure over the
	// on-box account roster + the resolved serving nodes (see cmd/fak/guard_endpoints.go);
	// nil on the default serve path. Guarded by endpointsMu. See session_endpoints.go.
	endpointsMu       sync.Mutex
	endpointsProvider func() SessionEndpoints

	// harnessSnapshotProvider is the optional pull source for the live harness-resource
	// block on /debug/vars and /v1/fak/observation (kernel/agent CPU/RSS/IO/net/GPU).
	// The typed observation form can report empty/stale/unavailable without collapsing
	// those states into omission; the legacy debug block remains observed-data only.
	// nil on the default serve path. Guarded by harnessSnapshotMu. See session_endpoints.go.
	harnessSnapshotMu       sync.Mutex
	harnessSnapshotProvider func() SessionHarnessObservation

	// sessionFleetProvider is the optional pull source for the live cross-machine fleet
	// block on /debug/vars (the "fleet" block). fak guard wires it to its TTL-cached
	// snapshot fold (see cmd/fak/guard_fleet.go); nil on the default serve path. Unlike the
	// endpoints provider it reports its own ok. Distinct from the worker-membership fleet
	// above: this is the fleet-of-MACHINES display aggregate, not the router's live view.
	// Guarded by sessionFleetMu. See session_fleet.go.
	sessionFleetMu       sync.Mutex
	sessionFleetProvider func() (SessionFleet, bool)

	// accountRehomeFn is the optional operator seat-switch function behind
	// POST /v1/fak/account/rehome. fak guard wires it to its account-failover state
	// (see cmd/fak/guard_account_failover.go); nil everywhere else keeps the route
	// inert. Guarded by accountRehomeMu. See account_rehome.go.
	accountRehomeMu sync.Mutex
	accountRehomeFn func(reason string) (AccountRehome, error)

	// planner generates the assistant turn for the /v1/chat/completions proxy. A
	// live HTTPPlanner/ReplicaRouter when BaseURL/ReplicaBaseURLs are set, else the
	// offline MockPlanner. Settable in-package for tests.
	planner agent.Planner
	// servedSide is the deployment-constant serving locality selectChatPlanner
	// resolved for the deployments that do NOT proxy: self-hosted for the in-kernel
	// model, unknown for the mock. servedLocality reads it. The zero value is the
	// honest unknown, so a test that sets planner directly leaves its turns
	// UNCLASSIFIED rather than silently attributed.
	servedSide servingLocality
	// upstream is what the configured base URL(s) establish about who owns the other
	// end of the proxy — and, by being nil, whether this deployment proxies at all.
	// The two facts ride in one field so the invalid state (a non-proxying server
	// that nonetheless has an upstream side) cannot be built. A proxying deployment
	// whose upstream cannot be placed carries a non-nil localityUnknown: it proxies,
	// and we decline to say to whom. See upstreamAttribution.
	upstream *servingLocality
	// inKernelModelButChatIsMock tracks when kernel has real weights loaded (for
	// fak_syscalls) but chat falls back to mock due to missing tokenizer (#1115).
	// Set at New when InKernelModel != nil && Tokenizer == nil. The /healthz
	// endpoint exposes this field as in_kernel_model_but_chat_is_mock to expose the
	// mismatch for witness fidelity.
	inKernelModelButChatIsMock bool
	engineCache                *enginecache.Client

	// kvReclaimer turns "a session slot freed" into "a KV block freed" for a waiting
	// sequence (#915, the drain/stop↔evict edge of #912): when a Scheduler SlotEvent with a
	// TERMINAL cause (draining/stopped) fires, ReclaimKVOnSlotFreed drives this reclaimer's
	// real KV free (kvmmu.Context.EvictColdest / model.KVCache.Evict). nil (the default)
	// leaves the edge a no-op; the host injects one backed by the live served residency via
	// SetKVResidencyReclaimer. Guarded by kvReclaimMu — the slot-freed observer runs on the
	// table's observer goroutine, so the read must be race-safe against a late install.
	kvReclaimMu sync.RWMutex
	kvReclaimer KVResidencyReclaimer

	// kvPressure{Provider,Sweeper} are the post-decode KV pressure-relief seams (#1073, the
	// keystone of epic #1072): after a served turn mutates the KV cache, maybeRelieveKVPressure
	// drives the provider for the live pressured spans and the sweeper for the real demote (the
	// engine.CapacityAdapter executing abi.KVBackend.StageSpan+Evict). nil (the default) leaves
	// the edge inert; the host injects both via SetKVPressureRelief once a device backend +
	// served residency are loaded — so "wired" IS the "there is a device to relieve" signal,
	// keeping the gateway free of the engine/compute imports. Guarded by kvPressureMu — the read
	// runs on a request goroutine, so it must be race-safe against a late install.
	kvPressureMu       sync.RWMutex
	kvPressureProvider KVPressureCandidateProvider
	kvPressureSweeper  KVPressureSweeper

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

	// ctxViewPending records decoded-message ctxplan rewrites that fired before the
	// provider usage is known. logInferenceTurn consumes one pending event for the same
	// trace so ctxvalue can count the served turn as a context event without threading a
	// new return value through every complete() caller. Anthropic raw passthrough uses an
	// explicit contextEvent bit and does not consume this map.
	ctxViewPendingMu sync.Mutex
	ctxViewPending   map[string]int

	// resetHealth holds ONE rolling compaction-health record per session trace id, fed the
	// provider's OBSERVED cache counters on every compacted turn so the per-session resetScore
	// shadow surface (#792, reset_shadow.go) can recommend cut-vs-reset without re-deriving the
	// session's cache health from a global counter. nil/empty until the first compacted turn;
	// minted lazily by resetHealthForLocked and bounded by maxResetHealthSessions. Guarded by
	// resetHealthMu. SHADOW-only: nothing here ever resets a session.
	resetHealthMu sync.Mutex
	resetHealth   map[string]*sessionResetHealth
	// coherenceResetArmed holds trace ids the #3159 non-holding-rewrite escalation has armed for
	// a hard reset on their NEXT admitted turn (armCoherenceReset). Unlike resetHealth this is
	// an ACTUATION latch, consumed once by maybeResetOnCoherence (which reuses the host's opt-in
	// ResetOnBudget callback). Guarded by the same resetHealthMu; minted lazily and bounded
	// generationally by maxResetHealthSessions.
	coherenceResetArmed map[string]bool

	// ctxValue holds ONE rolling managed-context record per session trace id, fed by
	// logInferenceTurn on EVERY served turn (all wires) so the multi-level long-session
	// context report (ctxvalue.go: tokens / turns / session + step advice) is answerable
	// live. Minted lazily by ctxValueForLocked and bounded by maxCtxValueSessions with
	// the same generational reset as resetHealth. Guarded by ctxValueMu. Advice-only:
	// nothing here feeds the request path.
	ctxValueMu sync.Mutex
	ctxValue   map[string]*sessionCtxValue

	// ctxExpenseWarn / ctxExpenseBlock are the effective per-turn as-sent-volume lines the
	// context-expense verdict (ctxexpense.go) grades on, in ESTIMATED tokens (0 = that tier
	// off). Derived from Config.CtxExpense{Warn,Block}Tokens via ctxExpenseThresholdOr, so
	// detection is on by default. ctxExpenseGate is the soak switch (FAK_CTX_EXPENSE_GATE):
	// when true, a block-tier verdict emits ONE in-band [fak] advisory per session; when
	// false (the default) the block tier is view-only. ctxExpenseNoted dedups that advisory
	// to once per session (a per-session once-note pattern), guarded by ctxExpenseNotedMu
	// and bounded by the same reaper. All read-only relative to the request path — the gate's
	// only actuation is the in-band note; the passthrough body is never mutated.
	ctxExpenseWarn    int
	ctxExpenseBlock   int
	ctxExpenseGate    bool
	ctxExpenseNotedMu sync.Mutex
	ctxExpenseNoted   map[string]struct{}

	// compactionContract holds, per session trace, the continuation contract the LAST
	// compaction boundary emitted (#2422, compaction_contract.go), pending the next completed
	// turn that reports it. Unlike ctxExpenseNoted this is a TAKE-ONCE latch rather than a
	// once-per-session seen-set: every boundary is a distinct loss the model has to be told
	// about, so the reporting turn consumes the record instead of suppressing later ones.
	// Minted lazily by noteCompactionContract, drained by takeCompactionContract, and bounded
	// by the same maxResetHealthSessions reaper. Report-only: nothing here mutates the body.
	compactionContractMu sync.Mutex
	compactionContract   map[string]*CompactionContract

	// ctxRestore holds, per session trace, the content-addressed stash of ORIGINATING tasks the
	// Anthropic-passthrough compaction dropped (ctxrestore.go). A fired tombstone hands the gateway
	// the dropped turn's bytes + its sha256-hex handle (agent.CompactOutcome.RestoreID/RestoreBytes),
	// which this table records so fak_context_restore(id) can page the full task back in for a model
	// resuming past the compaction. Minted lazily by stashRestore, bounded by maxCtxRestoreSessions
	// with the same generational reset as ctxValue. Guarded by ctxRestoreMu. READ-ONLY recovery:
	// restore returns bytes fak already dropped from the wire; it never re-enters the request path.
	ctxRestoreMu sync.Mutex
	ctxRestore   map[string]*sessionCtxRestore

	// traceOwner binds a session trace to the principal that owns it — the C1 read-scope
	// floor (#4192). The first served turn on a trace records its principal here
	// (first-writer-wins, via bindTraceOwner from handleAnthropicMessages), so a later
	// read-self op (fak_context_restore / fak_context_spans) can be scoped: a caller may
	// page a trace's dropped originating task back in ONLY when its principal matches the
	// owner. On the common no-RequireKey loopback both sides are "" (single-tenant: every
	// caller shares), which reads as a self-read; a caller naming a DIFFERENT principal is
	// refused READ_SCOPE_DENIED. Guarded by traceOwnerMu; bounded by the same
	// maxCtxRestoreSessions generational reset as ctxRestore so it cannot grow unbounded.
	traceOwnerMu sync.RWMutex
	traceOwner   map[string]string

	// tracePrincipal binds a session trace to the AUTHORITY principal its CURRENT turn
	// is attributed to (human / self-model / peer-agent / timer / network-tool / unknown) —
	// the #2412 inbound-principal floor. Stamped at the served-request boundary
	// (handleAnthropicMessages) from the inbound principal-class label, last-writer-wins
	// because authority is a PER-TURN fact (a session may relay a peer turn mid-stream),
	// and read at the adjudication seam (adjudicateProposedServed) to type-check
	// authority-consuming acts. Distinct from traceOwner (the tenant ISOLATION principal,
	// a string identity for read-scope). Guarded by tracePrincipalMu; bounded by the same
	// generational reset as traceOwner so it cannot grow unbounded.
	tracePrincipalMu sync.RWMutex
	tracePrincipal   map[string]Principal

	// sessionLeases holds the kernel-MINTED lease identities cross-agent sends address
	// (#2439): id -> (name, expiry). Addressing a lease id rather than a name is what
	// makes a send to a dead session refuse instead of misrouting to whoever holds that
	// name now. Guarded by leaseMu; bounded by the same generational reset as traceOwner.
	leaseMu       sync.RWMutex
	sessionLeases map[string]sessionLease

	// controlPlaneLog is the control-plane principal journal (#2439): one row per
	// /v1/fak/session/{id}/{verb} event with the principal the KERNEL assigned it, refused
	// or not, so a relayed authority attempt leaves a countable witness. Guarded by
	// controlPlaneMu; bounded to maxControlPlaneEvents, oldest dropped first.
	controlPlaneMu  sync.Mutex
	controlPlaneLog []ControlPlaneEvent

	// turnSafetyMu guards turnSafety, the per-trace stash of the LAST turn's adjudication
	// SAFETY delta (calls blocked / repaired this turn, results quarantined this turn). The
	// per-turn fak-turn debug line (debug_stats.go) already shows the turn's cache/token
	// VALUE; this carries its SAFETY half so a blocked rm -rf or a quarantined secret is
	// VISIBLE the moment it happens — not only in the exit summary. Written where the turn
	// adjudicates (recordTurnSafety, on both the buffered and streaming proxy paths) and
	// read-and-cleared by the render (takeTurnSafety), so each line reports THIS turn's
	// delta, never a running cumulative. Bounded by maxResetHealthSessions (same reaper as
	// resetHealth). SHADOW-only: an observability surface, never on the request path.
	turnSafetyMu sync.Mutex
	turnSafety   map[string]turnSafetyDelta

	// pastCompactRuns is the bounded per-trace consecutive-nudge ladder (#2638).
	// A compact/checkpoint turn or any return below the threshold clears the run.
	pastCompactMu   sync.Mutex
	pastCompactRuns map[string]int

	// placementFired records, per trace, whether fak's OFFENSIVE cache-breakpoint placement
	// (agent.PlaceAnthropicCacheBreakpoint) actually PLACED a breakpoint on THIS turn — i.e. the
	// caller sent no cache_control of its own and fak spliced one onto the stable head. That is the
	// one slice where the provider's cache_read this turn is unambiguously fak-UNLOCKED (without the
	// placement a no-breakpoint caller earns 0 provider cache), so the per-turn debug line can credit
	// it as fak-authored (fak=<tok> placement-unlocked) instead of the conservative fak=0. Written at
	// the placement site (recordPlacement) and read-and-cleared by the render (takePlacement), so each
	// line reports THIS turn's placement, never a running cumulative. Same reaper/bound as turnSafety.
	// SHADOW-only: an observability signal, never on the request path.
	placementMu    sync.Mutex
	placementFired map[string]bool

	// livelock tracks consecutive identical tool-call outcomes per trace so the third
	// repeat can be surfaced as a structured LIVELOCK_DETECTED envelope inside the same
	// turn stream. It records only tool names and args digests, never raw arguments.
	livelockMu sync.Mutex
	livelock   *guardrsi.LivelockDetector

	// resultLivelock is the result-side sibling for replayed inbound tool_result
	// quarantines. It is separate from livelock so a normal proposed tool call in the
	// same trace does not reset the "same quarantined result replayed again" run.
	resultLivelockMu sync.Mutex
	resultLivelock   *guardrsi.LivelockDetector
	// resultLivelockObserved is the REPLAY GATE for the result-side detector, guarded by
	// resultLivelockMu. The client replays the full transcript every turn, so
	// admitInboundResults re-quarantines the SAME held result on every subsequent turn;
	// without this gate the detector would re-count an unchanged held result each turn and
	// trip a livelock the agent never caused (a false positive). Keyed by trace -> set of
	// stable per-result keys (resultNoteKey: tool_call_id, or Tool|Reason when idless): a
	// key is observed at most once, so passive replay is filtered while a genuinely
	// RE-ISSUED call — which the client stamps with a NEW tool_call_id — is a new key and
	// still climbs toward a real trip. Bounded by maxResetHealthSessions (same reaper as
	// admitLedger/turnSafety).
	resultLivelockObserved map[string]map[string]struct{}
	// resultLivelockRecorded dedups the DURABLE side effects of a real trip — the
	// AppendLivelock journal row and the fleet-observation JSONL line — to at most once per
	// (trace, failure_hash), guarded by resultLivelockMu. A persistent loop re-fires the
	// envelope on each new re-issue; recording every fire would spam the journal and the
	// fleet-correlation feed, so the shared cause is recorded once per session and the
	// cross-trace breadth is what the correlator counts. Bounded like resultLivelockObserved.
	resultLivelockRecorded map[string]map[string]struct{}
	// fleetObsPathOverride and fleetObsMu drive the cross-trace fleet-observation feed. On a
	// real result-side trip the gateway appends one guardrsi.FleetObservation JSONL line
	// (deduped per (trace, failure_hash) by resultLivelockRecorded) that `fak knownbad
	// correlate` folds across traces into a shared-cause candidate. The sink path comes from
	// FAK_FLEET_OBS_PATH (set on the guarded session by the launcher), or this override in
	// tests; empty ⇒ the feed is off and only the durable LIVELOCK journal row is written.
	// fleetObsMu serializes appends from concurrent traces.
	fleetObsPathOverride string
	fleetObsMu           sync.Mutex

	// prunedToolDefs remembers, per served trace, tool definitions fak removed from the
	// advertised Anthropic tools[] because the capability floor could never admit them. If
	// the model later proposes one of those names anyway, adjudicateProposed logs that drift
	// once per trace/tool as a wire witness (tool name only; never raw arguments).
	prunedToolDefsMu         sync.Mutex
	prunedToolDefs           map[string]map[string]struct{}
	notedPrunedToolProposals map[string]map[string]struct{}

	// contextQueryAudit is the managed-context clarification journal (#1622): every
	// context question minted by the gateway records the answer source/default and
	// the assumption source ref it produced, so a replay can attribute a later
	// assumption to the question/default that created it. Runtime-only and bounded;
	// scrubbed corpus rows live in internal/sessionobs.
	contextQueryAuditMu  sync.Mutex
	contextQueryAuditSeq uint64
	contextQueryAudit    []ContextQueryAuditRecord

	// admitLedger keys result admission to content, per trace (#2417): a tool result is
	// screened EXACTLY ONCE, at first arrival, and its verdict is recorded on the entry;
	// a later replay of the same content consults the record instead of re-running the
	// result-side stack over the whole client-replayed transcript. This replaces the old
	// notedResults / notedToolFailures dedup maps — which only suppressed the repeated
	// human-facing banner while the re-screening still happened every turn — so "was this
	// result screened?" is a ledger query, the held-out banner and the exit-143 recovery
	// note are surfaced once, and /metrics counts unique results, not N×turns. The
	// machine-readable verdict still rides the `fak` extension every turn (the record is
	// consulted, so no signal is lost). See admission_ledger.go; carries its own mutex.
	admitLedger admissionLedger

	// originSeq maps an admitted origin call to the sequence stamped on its DECIDE row,
	// so a later client-produced tool_result can journal its QUARANTINE against the
	// originating call. Native fak_syscall records the kernel submission SeqNo by
	// trace/tool/args; proxy adjudication records a gateway-reserved sequence by
	// trace/tool_call_id because the client, not fak, executes the tool.
	originSeqMu   sync.Mutex
	originSeq     map[string]uint64
	originSeqByID map[string]uint64
	originSeqNext uint64

	// resumeProj holds the resume PROJECTED-vs-OBSERVED RESIDUAL accumulators (#941), a
	// self-contained metric family (resume_projection.go) the host's opt-in resume hook folds one
	// boundary into via observeResumeProjection. SHADOW / observe-only: nothing here resumes, cuts,
	// or resets a session. The projection is WITNESSED (fak's resume.Plan); the first-turn cache
	// bill it is differenced against is OBSERVED (provider-relayed). Its own mutex; zero-value ready.
	resumeProj resumeProjMetrics

	// observers is the stratified ASYNC observer rung on the result-admission chain (#2434):
	// non-blocking ResultObservers handed a READ-ONLY copy of each admitted result AFTER the
	// blocking chain settles, delivered off the turn path under a per-rung latency budget and
	// sample rate, with N-failures-in-window auto-disable + a HOOK_UNHEALTHY journal row. It is
	// the observability dual of the blocking abi.ResultAdmitter — it cannot block or mutate. Its
	// own metric family (fak_gateway_observer_*) renders self-contained via writeMetrics, like
	// resumeProj. Built in New; nil-safe for a bare Server (every method no-ops on nil).
	observers *observerStratum

	// guardRecoveryPrompt is the one-shot model-visible hint the guard host supplies
	// after a prior run ended with guard refusals. It is popped on the first served
	// Anthropic Messages request so a resume sees the recovery context once, without
	// turning it into persistent conversation noise.
	guardRecoveryMu     sync.Mutex
	guardRecoveryPrompt string

	// compactHistoryBudget mirrors Config.CompactHistoryBudget: when > 0 the flagship
	// Anthropic passthrough compacts OLD turns in the OUTBOUND body to this resident-token
	// budget while preserving the cached-prefix bytes (agent.CompactAnthropicHistory). 0
	// (the default) leaves the body byte-for-byte unchanged.
	compactHistoryBudget int
	// positiveResidualSubstitution mirrors the default-off config gate.
	positiveResidualSubstitution bool
	autoCheckpoint               func(session, reason string)

	// compactAnchorHead mirrors Config.CompactAnchorHead: false (the default) keeps the
	// warm-cache-safe agent.CompactAnchorFirstBP anchor; true re-anchors compaction on
	// agent.CompactAnchorHead, the opt-in #1407/#1408 fix for anchor-starved sessions.
	compactAnchorHead bool

	// assumeSessionTurns mirrors Config.AssumeSessionTurns: the head-anchored burst gate's
	// assumed session length when no bounded turn horizon is wired. Positive fires the shed
	// early on a warm continuously-active long session; 0 keeps the conservative behavior.
	assumeSessionTurns int

	// compactSolvencyFloorTokens mirrors Config.CompactSolvencyFloorTokens: the observed peak
	// resident occupancy at or above which context solvency overrides the head-anchored burst
	// gate's cache economics. 0 (the default) leaves the gate on pure economics.
	compactSolvencyFloorTokens int

	// exposeProfile mirrors Config.ExposeProfile: the descriptive surface label
	// ("headless"|"interactive") a usage-provenance stamp reads back via ExposeProfile().
	exposeProfile string

	// elideResultBytes mirrors Config.ElideResultBytes: when > 0 the flagship Anthropic
	// passthrough shrinks oversized tool_result bodies in the un-cached, non-recent middle to a
	// bounded head+tail form (agent.ElideAnthropicResults), keeping the cached-prefix bytes
	// verbatim and never touching a cache_control-bearing message. 0 leaves the body
	// byte-for-byte unchanged. The bounded-loss sibling of compactHistoryBudget.
	elideResultBytes int

	// elideStaleReads mirrors Config.ElideStaleReads: when true the flagship Anthropic passthrough
	// replaces a Read tool_result superseded by a later same-file edit with a restorable marker
	// (agent.ElideStaleReads), in the SAME cache-safe working-set band as elideResultBytes and
	// stashing the pre-edit snapshot behind a fak_context_restore handle. false leaves the body
	// unchanged. Size-independent; the read-lifecycle sibling of elideResultBytes.
	elideStaleReads bool

	// cacheTTL1H mirrors Config.CacheTTL1H or FAK_ABLATE_TTL_1H. When true, the Anthropic
	// passthrough upgrades stable-head cache_control breakpoints to ttl:"1h" before forwarding.
	cacheTTL1H atomic.Bool

	// provider mirrors Config.Provider — the resolved upstream wire ("anthropic",
	// "openai-responses", ...). The managed-cache posture surfaces read it to stay wire-aware:
	// the 1h-TTL upgrade lever cacheTTL1H drives is Anthropic-only, so on the OpenAI Responses
	// (codex) wire an ACTIVE session with zero upgrades is expected (the real lever is the pinned
	// prompt_cache_key), NOT the #2190 ACTIVE-but-inert misconfiguration. Empty preserves the
	// historical Anthropic reading.
	provider string

	// prefixGuard mirrors Config.PrefixGuard or FAK_ABLATE_PREFIX_GUARD (#2182): when true,
	// each served passthrough turn folds the inbound protected-prefix digest into the
	// fak_prefix_guard_* determinism witness (observePrefixGuard). Lossless — wire bytes
	// are never changed; off keeps the family at its emit-at-0 zeros.
	prefixGuard bool

	// vcacheAnchor mirrors Config.VCacheAnchor: when true the Anthropic passthrough runs the M2
	// star-anchor pre-flight rewrite (maybeAnchorAnthropicRaw) by DEFAULT — hoisting volatile
	// system blocks behind a byte-stable cacheable anchor and placing a breakpoint the caller did
	// not send — DECOUPLED from compactHistoryBudget (#1493). false leaves the pre-flight gate off.
	vcacheAnchor      bool
	vcacheCalibration *VCacheRuntimeCalibration

	// toolFloorDenies mirrors Config.ToolFloorDenies: the INBOUND-half predicate over a
	// tool name (true ⇔ the floor DEFAULT_DENYs it for every arg). When non-nil the
	// Anthropic passthrough prunes those provably-unreachable tool DEFINITIONS from the
	// outbound tools[] while keeping the cache_control prefix byte-identical. nil leaves
	// tools[] unchanged.
	toolFloorDenies func(toolName string) bool

	// exposeAllow mirrors Config.ExposeTools compiled to a predicate: non-nil ⇔ an
	// allowlist is in force, and it reports whether a tool NAME is exposed. nil (the
	// default) exposes every tool. exposedToolDescriptors (the single filter used by
	// every discovery view) and the tools/call guard are its only readers; both fall
	// open to the full surface when it is nil.
	exposeAllow func(toolName string) bool

	// deferMCPTools mirrors Config.DeferMCPTools (|| FAK_DEFER_MCP_TOOLS): true means
	// tools/list returns only the bootstrap view (#3231). Read only by
	// toolsListDescriptors; every other surface (fak_tools_search, tools/call) sees
	// the full registry regardless.
	deferMCPTools bool

	// capabilitiesReuse is the bounded, success-only reuse entry for the MCP
	// fak_capabilities discovery operation. Its zero value is ready for use.
	capabilitiesReuse capabilitiesReuseCache

	// deferColdTools mirrors Config.DeferColdTools: true means the outbound Anthropic
	// body defers its cold tool tail via defer_loading + an injected tool_search_tool
	// (#3232). Read only by maybeDeferColdTools; fail-safe identity when off.
	deferColdTools bool

	// systemBlockDrop is the same inbound-prune seam for typed Anthropic system[]
	// blocks: true means this named block element may be removed after the cached
	// system breakpoint. nil leaves system[] byte-for-byte unchanged.
	systemBlockDrop func(block, name string) bool

	// auditLog is the optional A2A audit logging function. When non-nil, all A2A task
	// state transitions are logged for tamper-evident tracking. nil disables A2A audit logging.
	// Set by cmd/fak to wire in the DECISION JOURNAL-backed audit system.
	auditLog func(log a2aAuditLog)

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

	// roster, when non-nil, is the model-ACCOUNT switcher the routed model id is bound
	// through before dispatch (#2528, Config.RouteAccounts). resolveRoute maps the
	// abstract plan member ("guard-a") to the account-resolved Target.EngineRoute()
	// ("openai:acct/gpt-5.5") the residency PDP and the kernel dispatch read; nil leaves
	// the plan member string as the route (the pre-#2528 path). It is a validated value
	// (New() calls Validate), so Resolve's dangling-ref/locality invariants hold.
	roster *modelroute.Roster

	// native, when true, routes a non-streaming /v1/messages turn through fak's OWN agent
	// loop (agent.RunArm) — the native-harness keystone (#1316). nativeMaxTurns bounds the
	// loop's model round-trips per request. See Config.Native / native_serve.go.
	native            bool
	nativeMaxTurns    int
	nativeCodeCatalog []agent.ToolDef
	nativeSpeculate   bool
	// midflight is the live per-trace mid-flight verb mailbox registry (#2403): the
	// lookup POST /v1/fak/session/{trace}/{interrupt|drop-pending-call|set-budget}
	// crosses to reach the owned run it names. See native_midflight.go.
	midflight midflightRuns
	// vdsoProxyFill opts the proxy path into warming the vDSO from admitted inbound
	// tool_result blocks (Config.VDSOProxyFill). Default false. See admitInboundResults.
	vdsoProxyFill bool

	// maxAgeByTool mirrors Config.ToolMaxAge: the per-tool served-read freshness ceiling
	// (#1349) adjudicateProposedServed enforces after a tier-2 hit. Set at New from the
	// config and mutable at wiring time via SetToolMaxAge; read on request goroutines
	// without a lock, so it must not be mutated once Serve is accepting turns. nil/empty
	// ⇒ no tool has a ceiling ⇒ the served path is byte-identical to today.
	maxAgeByTool map[string]time.Duration

	// fleet is the host-injected live worker membership/health/drain/failover loop
	// (fleet_membership.go) — the live fleet view the router reads. The metrics surface
	// DRAINS its transition log onto /metrics with a per-worker label (#42) via the
	// fleetMetrics bridge, whose cumulative per-(worker,kind) counter stays monotonic
	// across scrapes even after a worker is removed. nil (the default) emits no fleet
	// family — a host attaches a loop via SetFleetMembership once it has built the fleet
	// view, the same inject-after-New posture as the KV seams. fleetMu guards both fields:
	// SetFleetMembership may install the loop concurrently with a scrape that publishes it.
	fleetMu      sync.Mutex
	fleet        *FleetMembership
	fleetMetrics *FleetMembershipMetrics

	// admissionCtl is the optional native-serving ADMISSION / PRIORITY / FAIRNESS gate
	// (#35, admission.go) — the policy layer above modelengine.NativeScheduler's
	// continuous-batching loop. nil (the default) leaves the /metrics surface free of the
	// fak_sched_* family (no phantom zero series), the same inject-after-New posture as the
	// fleet / KV seams. A host attaches a live controller via SetAdmissionController once the
	// native scheduler is on the serve loop, at which point renderMetrics folds its running/
	// waiting/admitted/rejected counts into the shared L2 serving-metrics schema so a fleet
	// router / autoscaler can read per-worker load. admissionMu guards it: the install may
	// race a /metrics scrape that reads it.
	admissionMu  sync.RWMutex
	admissionCtl *AdmissionController

	// tokenRateGate is the optional HOST-LEVEL provider-token admission gate (#2019,
	// token_admission.go): a rolling-window TPM/ITPM/OTPM + concurrency budget the served
	// path reserves against before the planner runs and settles with the provider's real
	// normalized usage after. nil (the default) leaves the request path byte-for-byte
	// historical; a host attaches one per provider budget (account/seat) via
	// SetTokenRateGate. Guarded by admissionMu alongside admissionCtl.
	tokenRateGate *TokenRateGate

	// principalTokenRates is the optional PER-PRINCIPAL token allotment book (#5379,
	// token_admission_principal.go) that composes ABOVE tokenRateGate on the IDENTITY
	// axis: the authenticated keyset principal selects its own rolling-window budget,
	// consulted BEFORE the shared provider gate, so one noisy tenant can no longer drink
	// the whole provider window. nil (the default) leaves the single shared budget exactly
	// as it was. Guarded by admissionMu alongside tokenRateGate; installed via
	// SetPrincipalTokenRates.
	principalTokenRates *PrincipalTokenRates

	// spendGovernor is the optional control-plane SPEND CAP (#3273, epic #3256):
	// per-scope (tenant/team/agent/session) token/dollar budgets folded from the usage
	// the gateway already accumulates and evaluated at the served boundary, so a scope
	// that crosses its budget is hard-paused/killed by the kernel — not by asking the
	// model. spendScopeOf resolves a request trace to its scope hierarchy; nil ⇒
	// session-only (Session=trace). A nil governor (the default) leaves the request path
	// byte-for-byte historical. Guarded by admissionMu alongside tokenRateGate; see
	// spend_governor.go and the served-boundary wiring in session_admit.go.
	spendGovernor *SpendGovernor
	spendScopeOf  func(trace string) ScopeKey

	// preemptionMetrics is the optional native-serving KV preemption / swap / recompute
	// metric writer (#31). nil leaves fak_sched_preempt_* absent; a host attaches the live
	// native scheduler only after a positive paged-KV block budget arms preemption.
	preemptionMu      sync.RWMutex
	preemptionMetrics KVPreemptionMetricWriter

	// nativePDMetrics is the optional native prefill/decode role-split metrics writer (#28).
	// nil leaves fak_native_pd_* absent; a host attaches the live NativePDCluster once the
	// split prefill/decode pool is on the serving path.
	nativePDMu      sync.RWMutex
	nativePDMetrics NativePDMetricsProvider

	// nativeReceiptMetrics projects authoritative per-request fak-native receipts
	// into the shared /metrics surface. It never admits fallback-active receipts.
	nativeReceiptMetrics *nativeperf.ReceiptMetrics
}
