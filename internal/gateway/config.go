package gateway

import (
	"context"
	"net/http"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/auditreceipt"
	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/harnessversion"
	"github.com/anthony-chaudhary/fak/internal/kv"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/modelroute"
	"github.com/anthony-chaudhary/fak/internal/policy"
	"github.com/anthony-chaudhary/fak/internal/session"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
	"github.com/anthony-chaudhary/fak/internal/toolplugin"
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

// HeadlessCompactHistoryBudget is the floor-aware compaction budget a headless dispatch
// worker (`fak guard --expose-profile headless`) uses in place of the interactive
// DefaultCompactHistoryBudget. The interactive default assumes a LEAN prompt, but a
// Claude-Code dispatch worker carries a large fixed system+tools floor (~48-64k resident
// observed — at or ABOVE the 48k default). Because the inversion nudge and the fleet's
// compact-runaway spawn-hold both measure TOTAL resident (floor + conversation) against
// the budget (debug_stats.formatCompactionBudgetNudge, issue_resolve_dispatch's
// ACTIVE_COMPACT_RUNAWAY_MIN_PAST_K), while the compaction cut itself only sheds the
// post-floor SUFFIX, a 48k budget leaves every worker structurally past-compact from
// turn one: the nudge fires every turn and the fleet freezes new spawns, yet compaction
// correctly bails under_budget (the floor is immutable, the sheddable suffix is within
// budget). Right-sizing the budget to the worker's real resident shape (floor + a genuine
// ~46k conversation window) clears both false signals while keeping the cut a true
// runaway bound (resident stays under floor+budget, safely below the model's context
// ceiling). An explicit --compact-history-budget still wins.
const HeadlessCompactHistoryBudget = 96000

// DefaultVCacheAnchor is the default-on posture of the M2 star-anchor pre-flight gate
// (#1493): on the flagship Anthropic passthrough, APPLY cachemeta.RecommendLayout before
// send (maybeAnchorAnthropicRaw) rather than merely reporting it — hoist volatile system
// blocks behind a byte-stable cacheable anchor and splice a cache_control breakpoint onto
// the stable head a no-breakpoint caller did NOT send, so the first natural request warms
// provider prefix caching and later siblings read it. It is DECOUPLED from
// DefaultCompactHistoryBudget: the compaction path only placed that anchor while its own
// budget was > 0, so --compact-history-budget=0 silently took anchoring down with it; this
// gate fires independently. Default-on is safe by construction — the placement is fail-safe
// identity on any ambiguity (a hoist that would change the model-visible prefix is REFUSED,
// not applied) and idempotent with the compaction/TTL placements (a body already carrying a
// breakpoint bails already_set). An explicit --vcache-anchor=false wins (byte-for-byte OFF).
const DefaultVCacheAnchor = true

// DefaultAssumedSessionTurns is the session length the head-anchored compaction burst gate
// ASSUMES when no bounded turn horizon is wired (the default un-budgeted `fak guard -- claude`
// path, where Budget.TurnsLeft is Unbounded). Without a horizon the gate can only fire on an
// observed-cold trace (idle past the message-span TTL), so a warm, continuously-active long
// session — exactly the kind that benefits most — never sheds. Assuming a plausible length lets
// the SAME break-even economics (agent.CacheBurstPaysBack) fire EARLY in a presumed-long session,
// when the per-turn shed saving still has many turns to compound, and correctly REFUSE late
// (near the assumed end). The trade is sound because the burst penalty is ONE-TIME and bounded
// (a cold re-write of the invalidated suffix) while the shed saving is per-turn: an early burst
// on a session that runs long is a clear win, and a large-suffix shed still refuses regardless.
// This value is CALIBRATED from the observed session-length distribution in the TRACKED corpus
// docs/nightrun/gateway-usage.jsonl (gatewayusageledger.PublishedLedgerRel) — NOT the gitignored
// per-box .fak/nightrun sibling, whose population is night-dispatch fleet traffic (median 6 turns)
// rather than the interactive sessions this prior is about (#5406). Over that corpus at calibration
// time (n=1893 real guard/serve exits, `cached_turns`): median 7
// turns, p75 33, p90 52, p95 70 — 89% of sessions finish by 50 turns and only ~2% exceed 100. The
// same corpus re-measured 2026-07-28 on the same basis (n=803 rows carrying `cached_turns`) reads
// median 26, p75 39, p90 54, p95 73: the value 50 has NOT been retuned and still sits between p75
// and p90, which is what calibration_corpus_drift_test.go asserts every run. The
// earlier prior of 100 presumed a repaying tail ~14x the median, so warm early bursts on SHORT
// sessions (compaction fired on a median-6-turn session in that corpus) were justified by turns
// that never arrived — the low-end over-shed. 50 (~p90) sizes the presumed payback to a genuinely
// long real session: the common case still fires early (its whole length sits under the horizon)
// while a burst that would need more than ~p90 repaying turns is now correctly refused. A session
// that outruns the horizon still sheds via the penalty-free observed-cold path (idle past the
// message-span TTL), so the long tail is never stranded. The firing seam reads only the resolved
// int, so tuning changes the value, never the gate. An explicit --assume-session-turns wins;
// 0 disables the prior and restores the conservative "no horizon ⇒ no fire" behavior exactly.
const DefaultAssumedSessionTurns = 50

// Volume-aware horizon (the head-anchored burst gate's held-volume signal). DefaultAssumedSessionTurns
// alone is a TURN count, but the break-even it feeds is a TOKEN ratio — breakEven ≈ 11.5 ×
// (invalidatedSuffix / droppedMiddle) (agent.CacheBurstBreakEvenTurns). A MEDIAN real session (~65k
// resident, ~17k droppable middle) needs break-even ~6–33 cleared, so a flat turn horizon fires it
// only while it is YOUNG and starves it once served depth eats the remaining turns — the exact seam
// the 100→50 recalibration tightened. The fix keys the horizon on the trace's OBSERVED peak resident
// window (the coordinator's ResidentTokens, input+cached): a trace that has demonstrably held a large
// context is an empirically long/heavy session, so it keeps a repaying-turn FLOOR regardless of how
// deep it already is, instead of winding down to zero against a short-session constant.
//
// headHorizonHeavyResidentFloor is the peak resident-token watermark above which a trace is treated as
// heavy for the horizon prior. Sized just under the observed real-traffic floor (cached_prompt_tokens
// per turn is ≥~51k at p10 in the TRACKED docs/nightrun/gateway-usage.jsonl — same population as
// DefaultAssumedSessionTurns above, and for the same reason (#5406) — resident adds this turn's input on
// top), so a genuine working session qualifies while a SHORT chat (the token-light majority — median 7
// turns) stays on the conservative base horizon. headHorizonHeavyHeadroom is the number of repaying
// turns a heavy trace is guaranteed to retain: the effective TotalTurns becomes
// max(DefaultAssumedSessionTurns, servedTurns+headroom), so a heavy trace deep in its life still has
// ~headroom turns of break-even room. Both are INERT for the short/thin majority — the max() keeps the
// base horizon while servedTurns < base-headroom, and the whole branch is skipped below the resident
// floor — so this only lengthens the horizon for the demonstrably heavy-and-deep long tail, exactly
// where a flat turn count refused a burst that would in fact repay. The economics still gate every
// fire: a break-even that exceeds the granted headroom still refuses (a thin-middle / huge-suffix shed
// never fires no matter how heavy the session). A wired Budget.TurnsLeft horizon still wins over the
// whole prior, and the prior-disabled (assumeSessionTurns<=0) path is byte-for-byte unchanged.
//
// Cross-ref: this is the RESIDENT-VOLUME axis of the head-anchored burst gate — it lengthens the
// presumed horizon for a heavy trace. Its turn-indexed sibling is the early-firing budget ramp
// (earlyFireRampTurns, below), which instead moves the budget LINE by served-turn depth. Both feed
// the SAME CacheBurstPaysBack break-even from different axes, so retuning one shifts where the other bites.
const (
	headHorizonHeavyResidentFloor int64 = 60000
	headHorizonHeavyHeadroom            = 30
)

// Early-firing budget ramp (the "fak fires by ~step 5-10" seam). The head-anchored compaction
// against the FULL DefaultCompactHistoryBudget (~48k) only crosses its threshold once a session
// has sprawled past ~48k resident tokens — step ~35+ on a heavy coding session, and never on a
// normal one — so fak's OWN witnessed cache value stays $0 for the whole early session (the
// fak_share≈0 the guard-cache-value goal chased). But the burst economics (CacheBurstPaysBack)
// are MOST favorable early: the earlier a fire sheds the un-cacheable middle, the more future
// turns the per-turn read saving compounds over, AND a lower budget drops MORE of the middle
// relative to the fixed one-time recent-breakpoint burst — flipping the same break-even from
// "unprofitable" to a clean fire. So the effective budget the head-anchored fire targets RAMPS
// from a floor fraction of the configured budget at the start of a session up to the full
// configured ceiling by earlyFireRampTurns, keyed on the trace's real served-turn depth. The
// CacheBurstPaysBack gate still runs on the real horizon, so the ramp only moves WHERE the budget
// line sits — it can never force an unprofitable burst (the economics remain the safety net), and
// the effective budget is always <= the configured budget (the ceiling and its byte-safety
// guarantees are untouched). Inert unless head-anchored compaction is engaged WITH a resolved
// horizon (assumed or wired): the conservative first-breakpoint default and the cold-only path
// are byte-for-byte unchanged. Disabled by the same kill switches that disable early firing at
// all (--compact-anchor-head=false or --assume-session-turns 0 with no wired horizon).
//
// Cross-ref: this is the TURN-DEPTH axis of the head-anchored burst gate — it moves the budget line
// by served-turn depth. Its resident-volume sibling is the volume-aware horizon
// (headHorizonHeavyResidentFloor, above), which instead lengthens the presumed session. Both feed the
// SAME CacheBurstPaysBack break-even from different axes, so retuning one shifts where the other bites.
const (
	// earlyFireBudgetFloorFrac is the fraction of the configured compaction budget the
	// head-anchored fire targets at served-turn depth 1, ramping linearly to the full budget by
	// earlyFireRampTurns. 0.33 keeps a ~16k resident recent window at the floor (of the 48k
	// default) — aggressive enough to fire within the first several turns of a heavy session,
	// conservative enough to keep the model's live working set whole.
	earlyFireBudgetFloorFrac = 0.33
	// earlyFireRampTurns is the served-turn depth by which the effective budget reaches the full
	// configured ceiling. Past it the steady-state resident window is exactly the configured
	// budget, so a mature session is unchanged; the ramp only front-loads firing into the early
	// turns where the compounding and the burst headroom are greatest.
	earlyFireRampTurns = 25
)

const (
	// DocumentedElideResultBytes is the reviewed threshold for oversized tool-result
	// elision: a tool_result whose text payload exceeds this many bytes is a candidate
	// for head+tail shrinking when it is old, un-cached, and outside the working set.
	DocumentedElideResultBytes = 16384
	// DefaultElideResultBytes arms oversized-result elision ON by default at the documented
	// threshold. The lever is default-on because it is bounded-loss and fail-safe: it only
	// shrinks an OLD tool_result (after the cache head, outside the recent working-set window,
	// in a message with no cache_control the shrinker can reach), keeps the cached head prefix
	// byte-identical, never drops a result entirely (head+tail survive), and returns identity
	// on any ambiguity. Justified by adversarial verification (two rounds, four bugs closed),
	// a synthetic dogfood (~56% shed on a large coding session), and a real-corpus prevalence
	// scan (oversized tool_results in ~31% of 600 sampled real Claude Code sessions, ~2.9M
	// estimated tokens of scrolled-past content; experiments/agent-live/). Pass
	// --elide-result-bytes 0 to opt out; a larger value raises the threshold.
	DefaultElideResultBytes = DocumentedElideResultBytes
)

// DefaultElideStaleReads arms read-lifecycle STALE elision ON by default. It is the
// size-INDEPENDENT sibling of oversized-result elision: where DefaultElideResultBytes shrinks a
// tool_result because it is BIG, this replaces a Read tool_result because it is SUPERSEDED — a
// later in-session Edit/Write/MultiEdit/NotebookEdit of the same file has already changed those
// bytes, so the pre-edit snapshot no longer reflects disk and (unlike a scrolled-past command
// output) is actively misleading. The default-on posture rests on three legs, none of them a
// claim this const invents:
//
//  1. SAFETY — it shares the oversized shrinker's fail-safe machinery VERBATIM (splices on the
//     original bytes, re-proves the protected cache prefix byte-identical via verifySplicedBody,
//     never touches a cache_control-bearing message, returns identity on ANY ambiguity) but with a
//     STRICTLY MORE CONSERVATIVE predicate (a superseded read, not merely a large one) and full
//     RESTORABILITY (the pre-edit body is stashed behind a content-addressed fak_context_restore
//     handle, not discarded). Witnessed by internal/agent/anthropic_elide_stale_test.go
//     (TestReadLifecycleElidesStaleKeepsFreshAndPrefix pins the byte-identical prefix) and
//     internal/gateway/messages_elide_stale_test.go (TestMaybeElideStaleReadsRoundTrip pins the
//     fire + restore round-trip).
//  2. VALUE — a real-corpus prevalence scan (experiments/agent-live/
//     elide-stale-read-prevalence-2026-07-09.json): ~11% of 600 sampled real Claude Code sessions
//     carry at least one stale read; 565 stale reads totaling ~3.4 MB (~854K estimated tokens,
//     bytes/4 proxy) of pre-edit snapshot content the marker replaces. A sound lower-bound
//     motivation, the same status as the oversized-elision prevalence artifact.
//  3. NOT gated — unlike --defer-cold-tools (whose default-off was pinned to the #3537 A/B until
//     those gates cleared and DefaultDeferColdTools flipped it on), stale
//     elision's prior off-by-default was purely the initial conservative posture for a lossy-but-
//     restorable transform; there is no pending validation gate blocking the flip.
//
// Pass --elide-stale-reads=false to opt out.
const DefaultElideStaleReads = true

// DefaultDeferColdTools arms the #3232 cold-tool deferral lever ON by default at both
// front doors (fak guard / fak serve) — the #3537 payoff flip of epic #3229. The
// mechanism (messages_tooldefer.go) marks every allowed-but-COLD custom tool
// defer_loading:true and injects one tool_search_tool on the OUTBOUND Anthropic body, so
// the provider loads only the hot core into context and faults a cold schema in on
// demand. FAIL-SAFE ON FIRST REAL USE: every deferred def STAYS in tools[] byte-complete
// (name/description/input_schema untouched — the transform only ADDS the defer_loading
// key), so a deferred tool is discoverable via the injected search tool and its first
// call still resolves — nothing goes silently missing; and a floor-denied tool stays
// denied deferred or not (tooldefer_no_bypass_test.go). Deterministic + cache-safe
// (byte-stable tools[] turn-over-turn) and fail-safe identity on any ambiguity.
//
// Blocker verdicts behind the flip (#3537 gate): #3530 flag wiring CLOSED, #3532
// token-delta A/B CLOSED, #3533 held-accuracy CLOSED, #3534 poison/quarantine no-bypass
// CLOSED; #3536 live-dogfood soak still open — the opt-outs below are its rollback lever.
//
// Opt out per-launch with --defer-cold-tools=false, or stand the live seam down with
// FAK_ABLATE_DEFER_TOOLS=1 (the A/B ablation arm). This const is the FRONT-DOOR default;
// the embedded Config zero value stays OFF, so an SDK embedder opts in explicitly.
const DefaultDeferColdTools = true

// Config configures a gateway Server. The zero value is not valid — use New,
// which fills defaults and validates against the registered ABI.
type Config struct {
	// RichDashboards controls the lazy Grafana integration. New wires the manager
	// to the Server's actual bound listener so owned bundled Prometheus instances
	// scrape the live port, including an ephemeral or non-default loopback port.
	RichDashboards RichDashboardConfig
	// OTLPEndpoint enables bounded asynchronous OTLP/HTTP JSON trace export. Empty disables it.
	OTLPEndpoint      string
	OTLPQueueCapacity int
	OTLPTimeout       time.Duration
	// OrgAudit enables enrolled, privacy-screened adjudication receipts. Zero disables it.
	OrgAudit auditreceipt.Config
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
	// HedgePolicy explicitly enables bounded delayed hedging for eligible buffered
	// calls. Nil preserves the historical one-physical-call default.
	HedgePolicy *HedgePolicy
	// KVStore configures the direct I/O block KV cache backend for the gateway.
	// When non-nil, Server.KVStore() returns this store; otherwise New initializes DefaultStore().
	KVStore kv.Store
	// Provider selects the upstream transcript wire when BaseURL is set
	// (openai, anthropic, gemini, xai). Empty keeps the OpenAI-compatible default.
	Provider string
	// APIKey is the credential sent to the upstream model (proxy mode only). On the
	// Anthropic wire its SCHEME is chosen by the token itself: an OAuth subscription
	// token ("sk-ant-oat…", agent.IsAnthropicOAuthToken) goes as Authorization:
	// Bearer + the oauth beta; a plain API key goes as x-api-key.
	APIKey string
	// APIKeyFunc, when non-nil, supplies the upstream credential FRESH on every proxied
	// request instead of the frozen APIKey. It is how `fak guard` keeps a long pinned
	// subscription session alive: a Claude Pro/Max OAuth access token is short-lived and
	// the client rotates it on disk roughly hourly, so a planner that pinned the boot-time
	// token would start 401ing mid-session (even after a re-login, whose refreshed token
	// the frozen string never re-reads). With APIKeyFunc set, the proxy re-resolves the
	// token per request. Empty result falls back to APIKey. nil keeps the static-key path
	// unchanged.
	APIKeyFunc func() string
	// AccountFailoverFunc, when non-nil, supplies a REPLACEMENT upstream credential from a
	// permitted SIBLING account when the current one hits an ACCOUNT-SCOPED wall — a 403/402
	// whose body says this credential's organization (or region/billing) is denied even though
	// the credential is valid (the canonical case: the org has OAuth/subscription inference
	// disabled upstream, so every re-login is futile). It flows onto the planner's hook of the
	// same name; `fak guard` builds it to walk the sibling config homes, pick one on a different,
	// permitted, non-demoted account, and hand back its live token — also stickily redirecting
	// APIKeyFunc to the adopted account so the swap persists across turns. reason is a classified
	// remedy label, never a raw upstream body. nil keeps the terminal-on-account-403 behavior
	// unchanged.
	AccountFailoverFunc func(reason string) (newCred string, ok bool)
	// TransientTargetFunc supplies a distinct replacement credential after a temporary
	// upstream 5xx/529 survives one same-target probe. It must not permanently wall the
	// current account; nil preserves same-target retry behavior.
	TransientTargetFunc func(status int) (newCred string, ok bool)
	// ExtraHeaders are trusted host-supplied headers added to every upstream provider
	// request in proxy mode. They carry account-routing metadata that is not a generic
	// provider credential, such as ChatGPT-Account-Id for Codex ChatGPT subscription
	// sessions. Empty/nil preserves the historical adapter headers exactly.
	ExtraHeaders map[string]string
	// ExtraHeadersFunc supplies fresh host headers per proxied request. It is paired with
	// APIKeyFunc for rotating subscription credentials whose header metadata is resolved
	// from the same source as the bearer token. nil preserves the static/no-extra-header
	// path.
	ExtraHeadersFunc func() map[string]string
	// ForceResponsesStream asks an OpenAI Responses upstream for SSE while preserving the
	// gateway's buffered/adjudicated client response. Used for Codex ChatGPT subscription
	// upstreams, which reject non-streaming Responses requests.
	ForceResponsesStream bool
	// StreamProgressTimeout is the streaming CONTENT-progress deadline (#5486): how long a
	// proxied stream may stay WARM — keepalives arriving, so the inter-byte deadline never
	// fires — without one frame that advances the turn. It is carried verbatim onto every
	// proxy planner (newConfiguredHTTPPlanner) and resolved there by
	// agent.(*HTTPPlanner).streamProgressWindow, so this field uses that resolver's encoding
	// exactly: ZERO (the unconfigured default every caller who never sets it gets) means
	// agent.DefaultStreamProgressTimeout, a NEGATIVE value DISABLES the deadline — the escape
	// hatch for a provider whose prefill legitimately outlasts the window — and a positive
	// value outside [5s, 600s] falls back to the default rather than being clamped, so a
	// typo'd window never silently becomes a different real deadline.
	//
	// `fak serve` feeds this from --stream-progress-timeout, which spells the off switch the
	// way every other serve knob does (0), and translates that 0 into the negative encoding
	// above; see cmd/fak/serve.go:serveStreamProgressTimeout.
	StreamProgressTimeout time.Duration
	// PinUpstreamCredential makes the gateway authenticate the upstream with its OWN
	// configured APIKey and IGNORE the inbound client's credential — the subscription
	// path, where fak holds the real OAuth token and the wrapped client only sends a
	// placeholder key to satisfy its own "do I have credentials" check. Default false
	// keeps the transparent-hop passthrough (forward the client's own key upstream).
	PinUpstreamCredential bool
	// UpstreamResponseObserver, when non-nil, is called with the status + headers of
	// EVERY upstream provider response the proxy planners receive (buffered,
	// streaming, and each retry attempt) — the host's read-only window onto the
	// provider's OBSERVED account-usage headers (anthropic-ratelimit-* /
	// x-ratelimit-*). `fak guard` points this at its account tracker
	// (internal/accountobs) so the exit summary can report how loaded the upstream
	// account is. Read-only by contract: it must not mutate the response. nil (the
	// default) leaves the planners' transports byte-for-byte unchanged.
	UpstreamResponseObserver func(status int, header http.Header)
	// UpstreamTransportErrorObserver reports only transient dial/read/EOF/reset failures.
	UpstreamTransportErrorObserver func(error)
	// UpstreamFailureObserver receives bounded origin-attributed failure receipts.
	UpstreamFailureObserver func(UpstreamFailureReceipt)
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
	// InKernelPlanner carries native execution settings from the operator-facing serve
	// flags into the planner. Zero values preserve the native defaults.
	InKernelPlanner agent.InKernelPlannerConfig
	// LocalModelID names the model id a client asks for to reach the in-kernel model
	// when it is served ALONGSIDE a live upstream proxy — BaseURL set AND
	// InKernelModel+Tokenizer loaded, the dual planner (dual_planner.go). Empty
	// defaults to "local", and the literal id "local" always routes to the in-kernel
	// side too. Ignored outside dual mode (the single-planner paths keep Model).
	LocalModelID string
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
	// Metal, when true, runs the in-kernel chat through the Apple-Silicon metalgemm GPU
	// forward (GPU prefill + GPU-resident Q8 decode) on the CPU session. Set by
	// `fak serve --metal` (or FAK_METAL). It is the CPU-session seam (the session keeps
	// s.Backend nil and gets s.Metal=true), so it is MUTUALLY EXCLUSIVE with Backend —
	// serve rejects --metal together with --backend. A no-op on non-Metal builds
	// (the metalgemm stub makes the decode/prefill dispatch fall back to CPU), and the
	// resident decode self-declines anything but a dense Qwen-class Q8 model.
	Metal bool
	// ExpertParallelRanks is the expert-parallel rank count for the in-kernel MoE forward:
	// the number of expert shards the routed glm_moe_dsa MoE delta is reduced across
	// (model.SetExpertParallelRanks; the EP twin glmMoeEPFFN). 0/1 leave the forward on the
	// monolith glmMoeFFN (the no-op default — an existing serve is unchanged); >1 dispatches
	// routed layers through the EP path. Set by `fak serve --expert-parallel N`. At ranks=1
	// the EP path is bit-exact vs the monolith and needs no device; ranks>1 reduce through the
	// Collective the build wires (LocalCollective today), so a real multi-GPU resident-expert
	// serve is gated until the device NCCL CollectiveBackend lands — serve rejects N>1 until then.
	ExpertParallelRanks int
	// RequireKey, if non-empty, is the bearer token the gateway REQUIRES on every
	// request (except /healthz). Empty => no auth (drop-in compatible, loopback).
	RequireKey string
	// ReadBearer, if non-empty, is an ADDITIONAL bearer accepted ONLY on the read-only
	// observability endpoints (/healthz, /debug/vars, /metrics) as an alternative to
	// the loopback exemption / RequireKey (#3461). It never authorizes any other
	// route — a mutating call presenting it still 401s under RequireKey. `fak guard`
	// publishes it in the guard-session index so a second process can read this
	// session's status with no prior port (or key) knowledge.
	ReadBearer string
	// KeyPrincipals binds ADDITIONAL api keys to org/project ISOLATION principals
	// (#5332). Each entry maps a raw api key (as the caller presents it in x-api-key or
	// Authorization: Bearer) to the tenant principal it authenticates as; a matching
	// inbound key both AUTHENTICATES the caller and ATTRIBUTES the session to that
	// principal (access log + /v1/fak/events, joinable by X-Trace-Id). Additive to
	// RequireKey — the single RequireKey bearer still authenticates the anonymous
	// single-tenant caller, and an empty map leaves that path byte-for-byte unchanged.
	// The raw keys are consumed at construction: New hashes each to a SHA-256 digest and
	// drops the plaintext, so no raw key is retained in the Server or written anywhere —
	// the operator sources them from the environment at the front door, never from a
	// secret at rest.
	KeyPrincipals map[string]string
	// ExposeUpstreamErrorDetail, when true, lets the proxy fold a SCRUBBED, bounded
	// snapshot of the upstream provider's own 400 error body into the client-facing
	// message (see upstreamErrorStatus / plannerErrorStatus). It exists ONLY for the
	// trusted local path: `fak guard` binds an in-process gateway on a private
	// loopback port and injects its URL into the WRAPPED CHILD alone, so the "caller"
	// is that trusted agent and the upstream's real "which field was malformed"
	// detail is exactly what it needs to self-correct — with no meaningful leak.
	// Default FALSE preserves the #82/#346 no-leak invariant EXACTLY (the generic
	// "HTTP 400 — check the model name, roles, and parameter ranges" message), so the
	// externally-exposed `fak serve` path never forwards an upstream body. guard sets
	// it true ONLY when its listener is actually loopback-bound (guardLoopbackOnly);
	// a guard pushed off-host with --addr keeps it false. Scoped to 400 today (the
	// reported case); 401/403 stay generic — see the field's use site.
	ExposeUpstreamErrorDetail bool
	// UpstreamBadRequestNotify receives the same scrubbed, bounded provider 400 detail
	// exposed to a trusted local child, for persistence in an operator-side audit journal.
	UpstreamBadRequestNotify func(detail string)
	// DenialRecoveryOff stands down the #5212 denial-recovery sample: a turn whose every
	// proposed call the capability floor refused goes straight to the typed blocked state
	// instead of spending one bounded re-sample looking for an allowed alternative.
	//
	// Stated in the NEGATIVE on purpose. Recovery is the shipped default — a denial-only
	// turn dressed as a completed answer is the defect the feature exists to fix — so the
	// zero value has to mean ARMED. Every Config literal that predates this field (and
	// every embedder that never heard of it) therefore keeps the recovering behavior;
	// standing it down is the thing a deploy must say out loud.
	DenialRecoveryOff bool
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
	// PolicyRuntime is the active policy runtime used to validate and resolve canonical tool names
	// when normalising harness-specific tool namespace prefixes (e.g. "functions.", "mcp__<server>__").
	PolicyRuntime *policy.Runtime
	// PolicyCanaryTurns arms a default-off post-reload window. A deny-all streak
	// spanning the configured number of served turns rolls the floor back.
	PolicyCanaryTurns int
	// ObservePolicy reports which capability floor governs this process RIGHT NOW —
	// source + effective (post-overlay) digest — WITHOUT reloading anything (#3960).
	// It is the read-only complement of ReloadPolicy, mirroring ObserveTrace/ResetTrace:
	// before it, an operator had to POST a reload (re-reading the file, possibly CHANGING
	// the floor) merely to look. Nil disables the GET /v1/fak/policy route.
	ObservePolicy PolicyObserveFunc
	// ReloadRoute is the model-routing manifest hot-reload watcher (#4003) — the
	// route-plane twin of ReloadPolicy. When non-nil it seeds the atomic seam that
	// backs POST /v1/fak/route/reload, a manual SIGHUP-style forced reload of the
	// installed --route-manifest. The watcher is normally constructed AFTER New (it
	// needs the server's live routing holder, RouteLive), so the host installs it
	// post-hoc via SetRouteWatcher; this config field is the pre-New seam for a
	// host/test that already holds the watcher. A nil watcher — both here and via
	// SetRouteWatcher — disables the route (404), mirroring a nil ReloadPolicy.
	ReloadRoute *modelroute.Watcher
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
	// TrajctlMetrics projects bounded objective health onto /metrics. Nil is inert.
	TrajctlMetrics TrajctlMetricsFunc
	// DecideSession gates one served request at its session boundary. It is the
	// mutating hot-path twin of ObserveSession: the host calls session.Table.Decide,
	// so run-state refusal, TurnsLeft debit, budget exhaustion, and per-turn pace are
	// applied before the model turn is served. Nil keeps the historical observe-only
	// admission path.
	DecideSession SessionDecideFunc

	// Table is the optional shared drive-state table for served sessions.
	Table *session.Table
	// Scheduler is the optional session scheduler.
	Scheduler *session.Scheduler
	// Pool is the optional fleet-wide token budget pool.
	Pool *session.Pool
	// HarnessRouter manages sub-harness runtime multi-versioning, sticky session pinning,
	// and canary traffic distribution. When nil, New initializes a default router with v1 active.
	HarnessRouter *harnessversion.StickySessionRouter

	// StopGate checks declared completion evidence at a model-final boundary. Nil disables it.
	StopGate StopGateFunc
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
	// GuardRecoveryPrompt is a one-time, model-visible recovery note supplied by
	// `fak guard` when the previous guarded run recorded capability-floor refusals.
	// The gateway injects it into the first Anthropic Messages request for this
	// server, then clears it. Empty leaves requests byte-for-byte unchanged.
	GuardRecoveryPrompt string
	// Logf is the structured log sink (default: stderr). MCP-over-stdio sets this
	// to stderr so protocol bytes on stdout are never corrupted.
	Logf func(format string, args ...any)
	// DebugStatsf is the OPTIONAL per-turn human debug sink (#793). When non-nil, every
	// served turn renders ONE compact, payload-free line — request/cache_read/cache_creation
	// tokens plus current/previous/average/median/high/low session cache savings, the
	// compaction action, and the resetScore SHADOW health (one of the five
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
	// PositiveResidualSubstitution enables conservative positive-state extraction for the
	// history span compaction drops. It is default-off; original bytes remain restorable.
	PositiveResidualSubstitution bool
	// AutoCheckpoint is a best-effort risky-boundary callback. It runs asynchronously
	// when context step advice reaches checkpoint/rebuild; gateway service never waits.
	AutoCheckpoint func(session, reason string)
	// CtxExpenseWarnTokens / CtxExpenseBlockTokens set the per-turn as-sent-volume lines the
	// context-expense arm (ctxexpense.go) warns and blocks on — the total volume of context
	// (system + tools + history + tail) that would be re-shipped on ONE turn, in ESTIMATED
	// tokens. Detection is on by default: the struct zero value takes the built-in
	// ctxExpense{Warn,Block}TokensDefault, a positive value overrides, and a NEGATIVE value
	// disables that tier (the explicit off, like --compact-history-budget=0). The warn tier
	// is pure observability (the report + the --debug-stats line); the block tier only
	// ACTUATES when the FAK_CTX_EXPENSE_GATE soak switch is on, and then only as one in-band
	// [fak] advisory per session — the passthrough body is never refused.
	CtxExpenseWarnTokens  int
	CtxExpenseBlockTokens int
	// CtxExpenseGate is the soak switch for the block-tier ACTUATION: false (the default)
	// keeps the expense arm view-only (report + --debug-stats line); true arms the once-per-
	// session in-band [fak] advisory a block-tier verdict emits. The ablation/soak harness
	// also arms it with FAK_CTX_EXPENSE_GATE=1. Detection (warn + block verdicts) is
	// unaffected by this — only whether the block tier writes an in-band note.
	CtxExpenseGate bool
	// CompactAnchorHead, when true, re-anchors CompactHistoryBudget's protected prefix on
	// the stable provider head (agent.CompactAnchorHead) instead of the default first-
	// breakpoint anchor (agent.CompactAnchorFirstBP). This is the #1407 anchor-starved fix:
	// real Claude Code traffic marks its cache_control breakpoint on a RECENT message, so
	// the default anchor protects almost the whole conversation and the budget can never
	// shed anything (CompactionAnchorStarved). Re-anchoring on the head makes the WHOLE
	// message array compactible, but it bursts the recent breakpoint's cached suffix once,
	// so agent.CompactAnthropicHistoryWithOptions only fires it when the burst repays
	// (agent.CacheBurstPaysBack, #1408) — without a wired session-turn horizon this gateway
	// leaves TotalTurns/CurrentTurn unset, so it only fires on a zero-penalty burst. False
	// (the default) reproduces the pre-#1407 CompactAnchorFirstBP behavior byte-for-byte;
	// this is an explicit opt-in, not a default-on lever, because it can burst a warm cache.
	CompactAnchorHead bool
	// AssumeSessionTurns is the session length the head-anchored burst gate ASSUMES when no
	// bounded turn horizon is wired (Budget.TurnsLeft Unbounded), so a warm continuously-active
	// long session can shed instead of waiting to idle past the message-span TTL. Positive ⇒ the
	// gate fires early in a presumed-long session (CurrentTurn from the trace's served-turn depth,
	// TotalTurns from this value) and refuses near the presumed end; 0 ⇒ the conservative
	// "no horizon ⇒ no fire unless zero-penalty" behavior, byte-for-byte. A wired positive
	// Budget.TurnsLeft always wins over this prior. The command surfaces default this to
	// DefaultAssumedSessionTurns. Consulted only when CompactAnchorHead is on and the head
	// re-anchor engages; inert on every other path.
	AssumeSessionTurns int
	// CompactSolvencyFloorTokens arms the context-solvency override on the head-anchored burst
	// gate (agent.CompactOptions.SolvencyFloorTokens): once a trace's OBSERVED peak resident
	// window reaches this many tokens, a head-anchored compaction fires even when the burst does
	// not repay in cache dollars. It answers the measured pathology that the pure-economics gate
	// refuses HARDEST exactly where refusing is most expensive — over 3191 real served turns the
	// fire rate inverted with occupancy (33% at 96-125k → 3.4% at 155-170k → 0% above 170k) and
	// 100% of traces that ever fired never fired again, drifting a median +33.8k further into the
	// window before the run ended.
	//
	// The caller supplies it because only the caller knows the model's context window: the right
	// floor is a fraction of (window − output reserve), which the gateway cannot derive (it never
	// sees a window size). `fak guard --compact-solvency-floor` surfaces it, and the dispatch
	// fleet derives it from its own window/reserve constants.
	//
	// 0 (the default) leaves the burst gate byte-for-byte on pure economics, so every existing
	// caller, ablation row and test is unchanged. Consulted only when CompactAnchorHead is on and
	// the head re-anchor engages; inert on every other path, and it can only ever turn a
	// burst_unprofitable bail INTO a fire — never the reverse.
	CompactSolvencyFloorTokens int
	// ElideResultBytes is the oversized tool-result elision threshold.
	// 0 keeps the transform inert; a positive value arms the documented head+tail
	// shrinker for results outside the active working set. The command surfaces default this
	// to DefaultElideResultBytes.
	ElideResultBytes int
	// ElideStaleReads arms the read-lifecycle STALE elision: a Read tool_result whose file was
	// Edited/Written in a LATER in-session turn is replaced (in the same cache-safe working-set band
	// as ElideResultBytes) by a restorable marker, the pre-edit snapshot stashed behind a
	// fak_context_restore handle. Size-independent; lossy but restorable. The command surfaces
	// default this ON via DefaultElideStaleReads (the safer, restorable sibling of oversized
	// elision); pass --elide-stale-reads=false to opt out.
	ElideStaleReads bool
	// CacheTTL1H upgrades an existing stable-head Anthropic cache_control breakpoint to the
	// 1-hour tier. It is gate-only for now; the ablation harness also enables it with
	// FAK_ABLATE_TTL_1H=1.
	CacheTTL1H bool
	// PrefixGuard arms the prefix-determinism guard (#2182, the runtime form of
	// #1602/#1604): each served Anthropic passthrough turn, witness whether the inbound
	// protected prefix (the content-free digest the harness-coherence seam already takes
	// BEFORE any fak transform) stayed byte-deterministic turn-over-turn, folded into the
	// fak_prefix_guard_* metric family — the "is the cacheable prefix actually stable?"
	// check an operator (or an ablation row) reads BEFORE relying on provider-cache
	// economics. LOSSLESS: it never changes outbound wire bytes. Default off; it must earn
	// default-on through its ablation row. The ablation harness also enables it with
	// FAK_ABLATE_PREFIX_GUARD=1.
	PrefixGuard bool
	// VCacheAnchor, when true, arms the M2 star-anchor canonicalization as a DEFAULT-ON
	// pre-flight gate on the flagship Anthropic PASSTHROUGH (#1493): before any other body
	// transform, apply cachemeta.RecommendLayout (agent.PlaceAnthropicCacheBreakpointWithOutcome)
	// so a caller that sent NO cache_control gets its volatile system blocks hoisted behind a
	// byte-stable cacheable anchor and a breakpoint spliced onto the stable head — earning
	// provider prefix caching by default, DECOUPLED from CompactHistoryBudget (compaction only
	// placed the anchor while its own budget was > 0, so --compact-history-budget=0 silently took
	// anchoring down with it). The CLI flag defaults this to true (--vcache-anchor default-on for
	// the Anthropic path); false is the explicit opt-out. Fail-safe identity on any ambiguity — a
	// hoist that would change model-visible semantics (a volatile-only head, no stable span) is
	// REFUSED, not silently applied — and idempotent with the compaction/TTL placements (a body
	// that already carries a breakpoint bails already_set). Anthropic passthrough only; a zero
	// value leaves the pre-flight gate OFF (the compaction/TTL placements are unaffected either way).
	VCacheAnchor bool
	// VCacheCalibration is an optional fresh, measured provider calibration. A
	// measured minimum prefix gates cache_control authoring below the provider's
	// observed cacheability floor; nil preserves the static default-on posture.
	VCacheCalibration *VCacheRuntimeCalibration
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

	// ExposeTools, when non-empty, is an ALLOWLIST of tool-name glob patterns that
	// restricts BOTH the advertised MCP surface (tools/list, fak_tools_search,
	// fak_feature_query, fak_capabilities, the capabilities resource) and what
	// tools/call will invoke: a tool whose name matches no pattern is neither
	// listed nor callable, and an attempt to call it answers "unknown tool"
	// exactly as a non-existent tool would — hiding a tool never leaks that it
	// exists. Patterns are path.Match globs over the bare tool name (e.g.
	// "fak_index_*", "fak_capabilities"). A single entry may be comma-separated
	// ("fak_index_*,fak_capabilities") and the flag may repeat; New splits/trims.
	// nil/empty (the default) exposes the full registry, byte-for-byte the
	// pre-allowlist surface. New fails LOUD on a pattern that is a malformed glob
	// or matches ZERO known tools, so a typo aborts startup rather than silently
	// shrinking the surface to nothing. Set by `fak serve --expose`.
	ExposeTools []string
	// ExposeProfile is the operative expose-profile label ("headless" | "interactive")
	// this session launched under, recorded verbatim so a gateway-usage provenance stamp
	// can scope calibration to a known surface (a headless dispatch worker runs a pruned
	// fak_* registry; an interactive child the full one). Purely descriptive — the actual
	// pruning is driven by ExposeTools; empty is read back as "interactive" by the caller.
	ExposeProfile string
	// DeferMCPTools, when true, makes the fak MCP server's tools/list return only a
	// small BOOTSTRAP set (the hot syscall/read/adjudicate core + the fak_tools_search
	// entry) instead of the full ~24-tool registry, so the cold fak_* schemas are
	// absent from the always-sent floor and faulted in on demand by fak_tools_search
	// (epic #3229, #3231). tools/call still routes EVERY registered tool, deferred or
	// not — deferral hides the schema, never the route or the guard. Composes UNDER
	// ExposeTools (it filters the already-exposed set). Default false: the reduction
	// depends on the client re-finding a searched tool and on the pin/quarantine guard
	// (#3200), so the default flip is gated on that validation. Also enabled by
	// FAK_DEFER_MCP_TOOLS=1.
	DeferMCPTools bool
	// DisableMCPDefer, when true, disables schema-light tools/list bootstrap deferral
	// and advertises the full tool catalog on tools/list (equivalent to FAK_ABLATE_MCP_TOOL_FILTER=1).
	// Used by clients like OpenCode that query tools/list once at startup and lack dynamic tool fault-in.
	DisableMCPDefer bool
	// MCPToolCeiling, when positive, clamps the number of advertised tools on tools/list
	// to a curated active set when deferral is disabled (e.g. DisableMCPDefer or ablation).
	// 0 preserves unbounded tool advertisement.
	MCPToolCeiling int
	// ModelVision declares whether the active model supports multimodal vision/image inputs (#10779).
	// When false (or unset for text-only models), image tools like view_image are omitted from advertised schemas.
	ModelVision bool
	// DeferColdTools, when true, defers the COLD tool tail on the OUTBOUND Anthropic
	// Messages body (#3232, the 10x floor lever): every allowed-but-cold custom tool is
	// marked `defer_loading:true` and one standard `tool_search_tool` is injected, so the
	// provider loads only the hot core into context and faults a cold schema in on demand.
	// This reaches the SYSTEMIC built-in slice (Read/Write/Edit/Bash/… — harness-owned, but
	// just req.Tools to fak's gateway), not just fak's own MCP tools. Deterministic and
	// cache-safe (byte-stable tools[] turn-over-turn), fail-safe identity on any ambiguity.
	// Zero value false (an SDK embedder opts in explicitly), but the FRONT DOORS (fak
	// guard / fak serve) now default their --defer-cold-tools flag ON via
	// DefaultDeferColdTools — the #3537 flip, its A/B (token-delta × held-accuracy ×
	// poison) gates reported PASS. Opt out with --defer-cold-tools=false; ablate the live
	// seam with FAK_ABLATE_DEFER_TOOLS=1. The fault-in leans on #3200's pin/quarantine.
	DeferColdTools bool
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
	// RouteAccounts, when non-nil, is the model-ACCOUNT roster the gateway consults at
	// live dispatch time (#2528): after RouteManifest PICKs an abstract model id
	// ("guard-a"), the roster BINDS it to a concrete Target (which provider kind, which
	// of the user's accounts, and the upstream wire model), and the account-resolved
	// Target.EngineRoute() ("openai:acct/gpt-5.5") — NOT the bare plan-member string — is
	// what buildCall writes to abi.ToolCall.Engine before Submit. So the residency PDP
	// adjudicates the ACCOUNT-resolved remote/local route, an ensemble member each binds
	// independently, and a route to a provider with no registered adapter fails LOUD at
	// dispatch ("no engine registered for route") rather than silently running through the
	// default engine. nil (the default) leaves the pre-#2528 behavior byte-for-byte: the
	// plan member string IS the route. New() validates a non-nil roster and fails loud on a
	// misconfigured one (a mis-bound account is a security boundary). Set by `fak serve
	// --route-accounts FILE`; the pure resolver is internal/modelroute (reused verbatim, per
	// the ticket's "the pure resolver is the source of truth" non-goal).
	RouteAccounts *modelroute.Roster
	// Native, when true, makes /v1/messages drive fak's OWN agent loop (agent.RunArm /
	// RunArmStream) instead of the single-shot proxy turn: fak owns dispatch, the in-kernel
	// syscall boundary is the sole tool path, and no external harness owns the turn loop.
	// This is the native-harness keystone (#1316/#1837) — it gives the owned loop its first
	// live, non-test serve-path caller and wires the WithSessionGate / WithRouteManifest /
	// steer options that otherwise have zero live callers. The loop is seeded with the
	// request's last user message and drives the kernel-owned tool catalog to a final answer;
	// the per-turn agent.ArmMetrics ride back on the response `fak.native_arm` extension.
	// nil/false (the default) leaves /v1/messages on the byte-for-byte proxy path. Set by
	// `fak serve --native`.
	Native bool
	// NativeMaxTurns caps the owned loop's model round-trips per served request when Native
	// is set. <= 0 falls back to DefaultNativeMaxTurns. Inert when Native is false.
	NativeMaxTurns      int
	NativeCodeWorkspace string // optional root for the six kernel coding engines
	NativeSpeculate     bool   // opt-in effect-free coding speculation on native turns
	// VDSOProxyFill, when true, warms the vDSO tier-2 cache from ADMITTED inbound
	// tool_result blocks on the proxy path: an ALLOWED, read-only-shaped result the
	// client sends back fills (tool,args)->result so a LATER re-proposed identical read
	// is served inline (adjudicateProposedServed) with no client re-execution. Default
	// OFF — it is sound only when the principal is named and writes that touch the same
	// resource reach fak (proxy-closed world), so it is an explicit operator opt-in.
	// Set by `fak serve --vdso-proxy-fill`. Inert (zero behavior change) when false.
	VDSOProxyFill bool
	// ToolPlugins and ToolPreferences arm the optional monotone extension host on
	// live fak_syscall calls. Empty plugins + zero preferences preserve the legacy
	// syscall path byte-for-byte.
	ToolPlugins     []toolplugin.Plugin
	ToolPreferences toolplugin.PreferenceLayers

	// ToolMaxAge is the per-tool served-read freshness CEILING (#1349): the operator's
	// deterministic counterpart to the model-driven `_fak_fresh` opt-out. A positive
	// duration for a tool name means a vDSO tier-2 hit OLDER than it is NOT served inline
	// — the read declines the cache and passes through to the client to run fresh (the
	// same already-tested fall-through as a cache miss / `_fak_fresh`). This is the first
	// ENFORCED (not forensic-only) staleness bound behind abi.ConsistencyBoundedStale:
	// BOUNDED_STALE declares a call ACCEPTS a bounded-age read; this map supplies the
	// bound. An absent tool (or a zero/negative duration) has NO ceiling — behavior is
	// byte-identical to today. Only tier-2 hits carry an age (age_ms), so tier-1 pure /
	// tier-3 static hits are never gated. Set at wiring time (before Serve) directly or
	// via SetToolMaxAge.
	ToolMaxAge map[string]time.Duration
}

// DefaultNativeMaxTurns bounds the native serve loop's model round-trips per request
// when Config.NativeMaxTurns is unset — enough headroom for a multi-step tool flow
// while still terminating a runaway loop.
const DefaultNativeMaxTurns = 16

// PolicyReloadFunc is injected by the host CLI so the gateway can expose a reload
// route without importing policy/adjudicator/ifc internals.
type PolicyReloadFunc func(context.Context) (PolicyReloadResponse, error)

// PolicyReloadResponse is the wire result of POST /v1/fak/policy/reload.
type PolicyReloadResponse struct {
	Reloaded        bool   `json:"reloaded"`
	Source          string `json:"source,omitempty"`
	Summary         string `json:"summary,omitempty"`
	EffectiveDigest string `json:"effective_digest,omitempty"`
	RolledBack      bool   `json:"rolled_back,omitempty"`
	Rollback        func() `json:"-"`
}

// PolicyObserveFunc is injected by the host CLI so the gateway can ATTEST the
// installed capability floor without importing policy/adjudicator internals — the
// read-only complement of PolicyReloadFunc (#3960), mirroring TraceObserveFunc.
type PolicyObserveFunc func(context.Context) (PolicyObservation, error)

// PolicyObservation is the wire result of GET /v1/fak/policy: which capability floor
// governs this process right now, attested by digest, with NO reload side effect.
//
// EffectiveDigest is deliberately the same field (and the same host-side computation)
// as PolicyReloadResponse.EffectiveDigest, so a GET answer and a POST-reload answer are
// directly comparable — that equality is what lets an operator prove a reload was a
// no-op. It covers the EFFECTIVE floor (base manifest folded with the operator
// allow/deny overlays), not the file bytes, because the overlay is re-applied on every
// reload and so the enforced floor differs from what is on disk.
type PolicyObservation struct {
	Source          string `json:"source,omitempty"`
	EffectiveDigest string `json:"effective_digest,omitempty"`
	Summary         string `json:"summary,omitempty"`
	// ReloadCount is the number of successful POST /v1/fak/policy/reload swaps this
	// process has served. The gateway owns this counter (the host func never sets it),
	// so an operator can tell a floor that has been hot-swapped under it from one that
	// has stood since launch. Always emitted, including the 0 of a never-reloaded floor.
	ReloadCount int64 `json:"reload_count"`
}

// RouteReloadResponse is the wire result of POST /v1/fak/route/reload — the outcome
// of a forced model-routing manifest reload (the route-plane twin of
// PolicyReloadResponse). Reloaded is false with no error when the on-disk manifest
// was byte-identical to the installed policy (a no-op reload); a malformed edit is
// reported as a 400, never a silent success.
type RouteReloadResponse struct {
	Reloaded bool   `json:"reloaded"`
	Source   string `json:"source,omitempty"`  // the watched route-manifest path
	Changed  bool   `json:"changed,omitempty"` // file content differed from the installed policy
	Reloads  int64  `json:"reloads,omitempty"` // cumulative successful hot-swaps (after this reload)
	Rejects  int64  `json:"rejects,omitempty"` // cumulative rejected (malformed) reload attempts
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
