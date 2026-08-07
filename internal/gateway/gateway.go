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
	"net/http"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/bgloop"
	"github.com/anthony-chaudhary/fak/internal/cachemeta"
	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
	"github.com/anthony-chaudhary/fak/internal/enginecache"
	"github.com/anthony-chaudhary/fak/internal/guardrsi"
	"github.com/anthony-chaudhary/fak/internal/kernel"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/modelroute"
	"github.com/anthony-chaudhary/fak/internal/rungobs"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
	"github.com/anthony-chaudhary/fak/internal/vcacheobserve"
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
	// DecideSession gates one served request at its session boundary. It is the
	// mutating hot-path twin of ObserveSession: the host calls session.Table.Decide,
	// so run-state refusal, TurnsLeft debit, budget exhaustion, and per-turn pace are
	// applied before the model turn is served. Nil keeps the historical observe-only
	// admission path.
	DecideSession SessionDecideFunc

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
	NativeMaxTurns int
	// VDSOProxyFill, when true, warms the vDSO tier-2 cache from ADMITTED inbound
	// tool_result blocks on the proxy path: an ALLOWED, read-only-shaped result the
	// client sends back fills (tool,args)->result so a LATER re-proposed identical read
	// is served inline (adjudicateProposedServed) with no client re-execution. Default
	// OFF — it is sound only when the principal is named and writes that touch the same
	// resource reach fak (proxy-closed world), so it is an explicit operator opt-in.
	// Set by `fak serve --vdso-proxy-fill`. Inert (zero behavior change) when false.
	VDSOProxyFill bool

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
	k          *kernel.Kernel
	engineID   string
	model      string
	requireKey string
	// readBearer is the read-scoped observability bearer (Config.ReadBearer): accepted
	// ONLY on /healthz, /debug/vars, /metrics, never on a mutating route.
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
	feed                     *coherenceFeed                   // the cross-agent "what changed" feed (vdso coherence bus)
	sessionFeed              *sessionFeed                     // the drive-state revision feed (#630; host-pushed via PublishSessionRevision)
	metrics                  *gatewayMetrics
	// toolPages is the tool catalog's home (#2440): each advertised tool schema is a
	// content-hashed read-only page owned by the ctxmmu, registered at the
	// maybeCompactInboundTools seam. The page table — not the transcript — is the
	// source of truth, so compaction can only evict a schema re-faultably, never lose
	// it; identical schemas dedupe across turns/sessions by content hash. Its
	// ResidentBytes/DedupHits back the tool_schema_resident_bytes and
	// tool_page_dedup_hits_total /metrics rows. Built in New; nil-safe for a bare Server.
	toolPages      *ctxmmu.ToolPageTable
	servedFailure  servedFailure // recent served-turn panic behind /healthz honesty (#2336); see served_failure.go
	traceSeq       uint64        // mints a non-empty TraceID when the wire omits one (atomic)
	reloadPolicy   PolicyReloadFunc
	resetTrace     TraceResetFunc
	observeTrace   TraceObserveFunc
	observeSession SessionObserveFunc
	controlSession SessionControlFunc
	steerSession   SteerSessionFunc
	listSessions   SessionListFunc
	decideSession  SessionDecideFunc
	stopGate       StopGateFunc
	debitSession   SessionDebitFunc
	resetOnBudget  ResetOnBudgetFunc
	budgetDrained  BudgetExhaustedFunc
	defaultTraceMu sync.RWMutex
	defaultTraceID string

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
	// block on /debug/vars (kernel/agent CPU/RSS/IO/net/GPU) — a structured twin of the
	// /metrics-only SetHarnessMetricsProvider. nil on the default serve path. Guarded by
	// harnessSnapshotMu. See session_endpoints.go.
	harnessSnapshotMu       sync.Mutex
	harnessSnapshotProvider func() SessionHarness

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
	cacheTTL1H bool

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
	vcacheAnchor bool

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
	native         bool
	nativeMaxTurns int
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
	// A misconfigured account roster is the SAME class of security boundary: it decides
	// which provider/account (local or remote) a routed model — and thus a tenant payload
	// — reaches. Validate at New and fail loud rather than fall through to a silent
	// mis-bind or a residency-floor bypass at dispatch time (#2528).
	if cfg.RouteAccounts != nil {
		if err := cfg.RouteAccounts.Validate(); err != nil {
			return nil, fmt.Errorf("gateway: route accounts: %w", err)
		}
	}
	// The MCP tool-exposure allowlist (--expose) narrows which tools are advertised
	// AND callable. It is the same class of boundary as the route manifest — it
	// decides what a client can reach — so compile-and-validate it here and fail
	// loud on a malformed or zero-match pattern rather than let a typo silently
	// shrink the surface to nothing at first connect. Empty ⇒ nil ⇒ full surface.
	exposeAllow, err := compileToolExposeAllow(cfg.ExposeTools)
	if err != nil {
		return nil, err
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
		startup.phaseWithProvenance(ph.Name, ph.Dur, ph.Provenance)
	}

	proxyURLs, err := proxyBaseURLs(cfg)
	if err != nil {
		return nil, err
	}

	planner, servedSide, upstreamSide, inKernelModelButChatIsMock, err := selectChatPlanner(cfg, model, proxyURLs, logf, startup)
	if err != nil {
		return nil, err
	}

	remoteCache, err := newEngineCacheClient(cfg)
	if err != nil {
		return nil, err
	}

	// Select the live fleet's tier-2 invalidation granularity (process-global vDSO).
	// Fail loud on an unknown name rather than silently degrading to a full flush.
	t := time.Now()
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
	var admissionCtl *AdmissionController
	if cfg.InKernelModel != nil && cfg.Tokenizer != nil && len(proxyURLs) == 0 {
		admissionCtl = NewAdmissionController(DefaultAdmissionPolicy())
	}

	s := &Server{
		k:                            k,
		engineID:                     engineID,
		model:                        model,
		servedSide:                   servedSide,
		upstream:                     upstreamSide,
		requireKey:                   cfg.RequireKey,
		readBearer:                   cfg.ReadBearer,
		keyset:                       newKeyset(cfg.KeyPrincipals),
		exposeUpstreamErrorDetail:    cfg.ExposeUpstreamErrorDetail,
		denialRecoveryOff:            cfg.DenialRecoveryOff,
		upstreamBadRequestNotify:     cfg.UpstreamBadRequestNotify,
		version:                      version,
		logf:                         logf,
		debugStatsf:                  cfg.DebugStatsf,
		reloadPolicy:                 cfg.ReloadPolicy,
		resetTrace:                   cfg.ResetTrace,
		observeTrace:                 cfg.ObserveTrace,
		observeSession:               cfg.ObserveSession,
		controlSession:               cfg.ControlSession,
		steerSession:                 cfg.SteerSession,
		listSessions:                 cfg.ListSessions,
		decideSession:                cfg.DecideSession,
		stopGate:                     cfg.StopGate,
		debitSession:                 cfg.DebitSession,
		resetOnBudget:                cfg.ResetOnBudget,
		budgetDrained:                cfg.OnBudgetExhausted,
		defaultTraceID:               strings.TrimSpace(cfg.DefaultTraceID),
		guardRecoveryPrompt:          strings.TrimSpace(cfg.GuardRecoveryPrompt),
		startup:                      startup,
		planner:                      planner,
		inKernelModelButChatIsMock:   inKernelModelButChatIsMock,
		engineCache:                  remoteCache,
		admissionCtl:                 admissionCtl,
		ctxView:                      ctxView,
		compactHistoryBudget:         cfg.CompactHistoryBudget,
		positiveResidualSubstitution: cfg.PositiveResidualSubstitution,
		autoCheckpoint:               cfg.AutoCheckpoint,
		ctxExpenseWarn:               ctxExpenseThresholdOr(cfg.CtxExpenseWarnTokens, ctxExpenseWarnTokensDefault),
		ctxExpenseBlock:              ctxExpenseThresholdOr(cfg.CtxExpenseBlockTokens, ctxExpenseBlockTokensDefault),
		ctxExpenseGate:               cfg.CtxExpenseGate || envEnabled("FAK_CTX_EXPENSE_GATE"),
		compactAnchorHead:            cfg.CompactAnchorHead,
		assumeSessionTurns:           cfg.AssumeSessionTurns,
		compactSolvencyFloorTokens:   cfg.CompactSolvencyFloorTokens,
		exposeProfile:                cfg.ExposeProfile,
		elideResultBytes:             ablateUncachedTrimBytes(cfg.ElideResultBytes),
		elideStaleReads:              cfg.ElideStaleReads,
		cacheTTL1H:                   cfg.CacheTTL1H || envEnabled("FAK_ABLATE_TTL_1H"),
		provider:                     strings.TrimSpace(cfg.Provider),
		prefixGuard:                  cfg.PrefixGuard || envEnabled("FAK_ABLATE_PREFIX_GUARD"),
		vcacheAnchor:                 cfg.VCacheAnchor || envEnabled("FAK_ABLATE_BP_PLAN"),
		toolFloorDenies:              cfg.ToolFloorDenies,
		exposeAllow:                  exposeAllow,
		deferMCPTools:                cfg.DeferMCPTools || envEnabled("FAK_DEFER_MCP_TOOLS"),
		deferColdTools:               cfg.DeferColdTools || envEnabled("FAK_DEFER_COLD_TOOLS"),
		cacheStream:                  cacheStream,
		rungObs:                      rungObs,
		observers:                    newObserverStratum(observerJournalPath(), logf),
		feed:                         newCoherenceFeed(0),
		sessionFeed:                  newSessionFeed(0),
		activity:                     newSessionActivity(),
		toolPages:                    ctxmmu.NewToolPageTable(nil), // nil ⇒ the process-global MMU pager (#2440)
		metrics:                      newGatewayMetrics(time.Now()),
		route:                        newRouteLive(cfg.RouteManifest),
		roster:                       cfg.RouteAccounts,
		native:                       cfg.Native,
		nativeMaxTurns:               nativeMaxTurnsOr(cfg.NativeMaxTurns),
		vdsoProxyFill:                cfg.VDSOProxyFill,
		maxAgeByTool:                 cfg.ToolMaxAge,

		pinUpstreamCredential: cfg.PinUpstreamCredential,
	}

	// #4003: seed the model-routing hot-reload seam behind POST /v1/fak/route/reload.
	// The watcher is normally installed AFTER New via SetRouteWatcher (it needs the
	// server's live routing holder), so this is a no-op unless a host/test supplies a
	// pre-built watcher in the config. A nil watcher leaves the route disabled (404).
	if cfg.ReloadRoute != nil {
		s.routeWatcher.Store(cfg.ReloadRoute)
	}

	// Wire retry observability onto the proxy planner (#793 follow-on): Complete's 429/5xx
	// backoff is otherwise invisible — up to ~8s of silent waiting. The hook bumps a retry
	// counter and prints a glanceable `fak-turn … retry` line to the default --debug-stats
	// sink, so an operator sees the backoff happening instead of a frozen terminal. Only the
	// direct HTTPPlanner carries the loop — unwrapped from a dual planner's proxy side, which
	// fronts the same upstream; the mock/in-kernel/replica planners don't, so this is a
	// no-op for them.
	if hp := unwrapHTTPPlanner(planner); hp != nil {
		hp.RetryNotify = s.onUpstreamRetry
		hp.AuthRefreshNotify = s.onAuthRefresh
		hp.ForbiddenRetryNotify = s.onForbiddenRetry
		hp.AccountFailoverNotify = s.onAccountFailover
	}

	// Build the in-kernel background-loop supervisor and register the built-in loops
	// (a liveness heartbeat). It is not running yet — Serve starts it on the lifecycle
	// context and joins it on shutdown, so the loops progress exactly while the kernel
	// is up.
	s.loops = newBgloopSupervisor(s)

	return s, nil
}

// selectChatPlanner picks the chat backend for the /v1/chat/completions and
// /v1/messages surfaces from the wired configuration — a dual (local-alongside-API)
// planner, a proxy planner, the in-kernel chat planner, or the deterministic mock —
// records the planner-init startup phase, and reports whether real weights are loaded
// for syscalls but chat still falls back to the mock (missing tokenizer, #1115).
func selectChatPlanner(cfg Config, model string, proxyURLs []string, logf func(string, ...any), startup *startupProfile) (agent.Planner, servingLocality, *servingLocality, bool, error) {
	var planner agent.Planner
	var err error
	t := time.Now()
	inKernelModelButChatIsMock := false
	// Which SIDE serves a turn under this deployment — the attribution the usage
	// ledger's self-hosted split needs and could not previously get. It is resolved
	// here, and only here, because this switch is the one place the four modes are
	// exhaustive by construction: a fifth branch added later must state its own side
	// or inherit the honest unknown, whereas a type-switch somewhere downstream would
	// quietly read a new planner as unclassified and shrink coverage without saying so.
	side := localityUnknown
	switch {
	case len(proxyURLs) != 0 && cfg.InKernelModel != nil && cfg.Tokenizer != nil:
		// DUAL (small local model ALONGSIDE the API upstream, dual_planner.go): a live
		// proxy AND a loaded in-kernel model in ONE gateway. Requests addressed to the
		// local model id (Config.LocalModelID, default "local") decode in-kernel with no
		// upstream call; every other request — including the wrapped agent's default
		// turns — proxies upstream byte-for-byte as the proxy-only path does.
		// Historically the proxy silently WON this combination and the loaded weights
		// were dead; now the combination is the alongside deployment.
		proxy, perr := newProxyPlanner(cfg, model, proxyURLs)
		if perr != nil {
			return nil, localityUnknown, nil, false, perr
		}
		localID := localModelIDOr(cfg.LocalModelID)
		planner, err = NewDualPlanner(proxy, newInKernelChatPlanner(cfg, localID, logf), localID)
		if err != nil {
			return nil, localityUnknown, nil, false, err
		}
		logf("gateway: dual planner — model id %q (and \"local\") decodes in-kernel; every other model id proxies upstream", localID)
		// Deliberately left unknown: dual is the one mode with no deployment-constant
		// side, because the side is a property of each REQUEST's model id. servedLocality
		// asks the planner itself (RoutesLocal) rather than guessing from a default here.
	case len(proxyURLs) != 0:
		planner, err = newProxyPlanner(cfg, model, proxyURLs)
		if err != nil {
			return nil, localityUnknown, nil, false, err
		}
		// Deliberately left unknown, for the same reason dual is: a proxying
		// deployment has no deployment-CONSTANT side. `--base-url` is how both
		// self-hosting rungs are reached in the common case — an on-box server and a
		// company-operated one — so reading the flag itself as "a third-party API"
		// told an operator who bought nothing that they bought everything. What the
		// upstream URL does establish is resolved by upstreamAttribution and rides on
		// Server.upstream, where a per-model roster binding can still override it;
		// servedLocality composes the two.
	case cfg.InKernelModel != nil && cfg.Tokenizer != nil:
		// Serve the model fused into the kernel as the chat backend on BOTH
		// /v1/chat/completions and /v1/messages (they share s.planner.Complete):
		// real ChatML chat via internal/tokenizer, the cmd/fakchat recipe factored
		// into a Planner. Falls through to MockPlanner if the host didn't preload.
		planner = newInKernelChatPlanner(cfg, model, logf)
		// Every turn decodes on this box against weights we host.
		side = localitySelfHosted
	default:
		// No upstream (--base-url) and no in-kernel model (--gguf/FAK_MODEL_DIR): the
		// chat surface silently fell back to the deterministic offline mock. Warn
		// LOUDLY so an operator never mistakes scripted demo text for real model
		// output — the /healthz planner:"mock" field carries the same signal to a
		// liveness probe.
		if cfg.InKernelModel != nil && cfg.Tokenizer == nil {
			// #1115: kernel has real weights loaded (for fak_syscalls) but chat
			// falls back to mock due to missing tokenizer. Flag for witness fidelity.
			inKernelModelButChatIsMock = true
			logf("gateway: WARNING — POST /v1/chat/completions is served by the DETERMINISTIC MOCK planner: responses are SCRIPTED, not model output. --gguf was passed but no BPE tokenizer was found (GGUF has no embedded BPE tokenizer and no --tokenizer was provided). Pass --tokenizer <dir|file> to enable real chat, or --base-url to proxy a real provider.")
		} else {
			logf("gateway: WARNING — POST /v1/chat/completions is served by the DETERMINISTIC MOCK planner: responses are SCRIPTED, not model output. Pass --base-url (proxy a real provider) or --gguf/FAK_MODEL_DIR (serve the in-kernel model) to disable the mock.")
		}
		planner = agent.NewMockPlanner(model)
		// Scripted text is not inference, from here or from a vendor. It stays
		// UNCLASSIFIED so a mock run can never pad the self-hosted share with turns
		// no hardware ever served.
	}
	startup.phase("planner-init", time.Since(t))
	return planner, side, upstreamAttribution(proxyURLs), inKernelModelButChatIsMock, nil
}

// ablateUncachedTrimBytes composes Config.ElideResultBytes with the FAK_ABLATE_UNCACHED_TRIM
// ablation arm (#2182): a configured positive threshold always wins verbatim, and an arm
// sweeping the uncached-tail trim on a construction that left the lever inert (<= 0) arms it
// at the documented default. This is the int-shaped form of the `cfg.X || envEnabled(...)`
// compose the boolean levers (CacheTTL1H, VCacheAnchor, PrefixGuard) use, so guard flags and
// ablation env compose instead of replacing each other.
func ablateUncachedTrimBytes(configured int) int {
	if configured <= 0 && envEnabled("FAK_ABLATE_UNCACHED_TRIM") {
		return DefaultElideResultBytes
	}
	return configured
}

func envEnabled(name string) bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv(name))) {
	case "1", "true", "yes", "on":
		return true
	default:
		return false
	}
}

// onUpstreamRetry is the planner's RetryNotify hook: count the retry and surface it on the
// default debug-stats line so the otherwise-silent 429/5xx backoff window is visible. status is
// the upstream HTTP status that triggered the retry (0 for a transient transport error).
func (s *Server) onUpstreamRetry(attempt, status int, wait time.Duration) {
	if s == nil {
		return
	}
	if s.metrics != nil {
		s.metrics.observeUpstreamRetry(wait)
	}
	if s.debugStatsf != nil {
		s.debugStatsf("fak-turn retry attempt=%d status=%d wait=%s", attempt, status, wait.Round(100*time.Millisecond))
	}
}

// onAuthRefresh is the planner's AuthRefreshNotify hook: surface a 401 token-rotation self-heal
// on the rotating-subscription path. It is SEPARATE from onUpstreamRetry so a credential expiry
// is never conflated with a 429/5xx backoff. outcome is "recovered" (a fresh token was adopted
// mid-session and the call re-sent in place — the live guarded session healed across a re-login)
// or "exhausted" (no fresher token landed within the grace window, so the 401 is about to surface
// and the wrapped agent will drop into its own /login). This is the otherwise-INVISIBLE event the
// "fak guard gets stuck on login sometimes" class hinges on: with this line an operator sees the
// self-heal happen — or sees it give up — instead of a silent session loss.
func (s *Server) onAuthRefresh(outcome string, attempt int) {
	if s == nil {
		return
	}
	if s.metrics != nil {
		s.metrics.observeUpstreamAuthRefresh(outcome)
	}
	if s.debugStatsf != nil {
		s.debugStatsf("fak-turn auth-refresh outcome=%s attempt=%d", outcome, attempt)
	}
}

// onForbiddenRetry is the planner's ForbiddenRetryNotify hook: surface a 403 transient-recovery
// outcome. It is SEPARATE from onAuthRefresh (a 401 credential rotation) and onUpstreamRetry (a
// 429/5xx backoff) so a transient-permission flap is its own signal, not conflated with the
// other two. outcome is "recovered" (a retry within the short window returned 200 — a transient
// abuse/capacity gate cleared and the live session healed in place instead of dropping into a
// spurious /login) or "exhausted" (the bounded window/attempts elapsed still 403ing, so the
// denial is the permanent entitlement kind now surfacing with the actionable answer). This is the
// otherwise-INVISIBLE event the 2026-07-03 gem8 storm exposed: five sessions 403'd for ~9 minutes
// then recovered on the same token — with this line an operator sees the flap self-heal, or sees
// it give up on a real permission denial, instead of a silent session loss.
func (s *Server) onForbiddenRetry(outcome string, attempt int) {
	if s == nil {
		return
	}
	if s.metrics != nil {
		s.metrics.observeUpstreamForbiddenRetry(outcome)
	}
	if s.debugStatsf != nil {
		s.debugStatsf("fak-turn forbidden-retry outcome=%s attempt=%d", outcome, attempt)
	}
}

// onAccountFailover is the planner's AccountFailoverNotify hook: surface an ACCOUNT-SCOPED failover
// outcome — the response to a 403/402 whose body says this credential's organization (or region or
// billing) is walled, even though the credential is valid. It is SEPARATE from the other three
// notify hooks (a 429/5xx backoff, a 401 token rotation, a transient-403 flap) because the cause is
// distinct — a permanent PER-CREDENTIAL wall that no retry or re-login clears — and so is the fix:
// swap to a permitted sibling account. outcome is "recovered" (a permitted sibling credential was
// adopted and the walled turn completed in place, so the session healed onto a working account
// instead of dropping into a futile /login) or "exhausted" (no failover target existed, so the
// account-scoped 403 now surfaces). The SAME hook also reports the 429-account-cap seat rehome that
// rides this swap mechanism: "rehomed_seat" (a session/weekly/usage cap — whose reset can be hours
// away — was answered by moving to a free sibling seat that served the turn now, instead of sleeping
// toward the reset) or "rehome_seat_unavailable" (every sibling seat was capped/walled, so the
// cap-aware backoff rode it out). This is the otherwise-INVISIBLE event behind the org-OAuth-disabled
// failure AND the "429 that was longer than it looked": with this line an operator sees the session
// auto-switch accounts/seats — or sees it give up because every one is walled — instead of a silent
// session loss.
func (s *Server) onAccountFailover(outcome string, attempt int) {
	if s == nil {
		return
	}
	if s.metrics != nil {
		s.metrics.observeUpstreamAccountFailover(outcome)
	}
	if s.debugStatsf != nil {
		s.debugStatsf("fak-turn account-failover outcome=%s attempt=%d", outcome, attempt)
	}
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

// SetRouteWatcher installs (or clears, with nil) the model-routing manifest
// hot-reload watcher that backs POST /v1/fak/route/reload (#4003). The host calls
// this from the serve lifecycle right after it constructs the watcher over the
// server's RouteLive holder, so the manual reload route drives the SAME watcher as
// the background poll loop. The store is atomic, so a concurrent request that hits
// the route observes either the old or the new watcher, never a torn pointer. A nil
// watcher leaves the route disabled (404), mirroring an unset reloadPolicy.
func (s *Server) SetRouteWatcher(w *modelroute.Watcher) { s.routeWatcher.Store(w) }

// currentRouteWatcher loads the installed route-reload watcher, or nil when routing
// hot-reload is not configured for this deployment (the 404 predicate for the route
// handler).
func (s *Server) currentRouteWatcher() *modelroute.Watcher { return s.routeWatcher.Load() }

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
			// urls[0] may carry a name=URL identity; the cache-control target is the URL.
			_, baseURL = parseReplicaEntry(urls[0])
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

// newInKernelChatPlanner builds the in-kernel chat planner (the model fused into the
// kernel) advertising modelID, wiring expert parallelism onto the model exactly as the
// single-planner path always has. Shared by the pure in-kernel case and the dual
// (local-alongside-API) case so the EP semantics cannot drift between them.
// Expert parallelism is model state, set on the in-kernel Model here (the EP rank
// lives on the Model, consumed by ffnForLayer); 0/1 is the no-op default.
func newInKernelChatPlanner(cfg Config, modelID string, logf func(string, ...any)) agent.Planner {
	if cfg.ExpertParallelRanks > 1 && !cfg.InKernelModel.IsExpertParallelRankLocal() {
		cfg.InKernelModel.SetExpertParallelRanks(cfg.ExpertParallelRanks)
		// Reduce the routed-expert partials through the DEVICE collective the serve
		// initialized — serve.go gates ranks>1 on a backend advertising Caps().Collective
		// (the NCCL CollectiveBackend), so the decode AllReduceSum must cross those GPUs,
		// not the hardcoded single-box LocalCollective glmMoeEPFFN reduced through before.
		// On cpu-ref the bridge is byte-identical to LocalCollective (collective_bridge_test.go),
		// so this changes no host-tested bytes; on the NCCL backend the SAME call all-reduces
		// across the rank fleet. Fail-soft: a backend without the seam leaves the bit-exact
		// LocalCollective default (the EP output stays correct, just reduced host-side).
		if cfg.Backend != nil {
			if err := cfg.InKernelModel.SetExpertParallelDeviceCollective(cfg.Backend); err == nil {
				logf("gateway: expert-parallel ranks=%d → routed-expert AllReduceSum reduces through device collective %q (Caps().Collective=%v)", cfg.ExpertParallelRanks, cfg.Backend.Name(), cfg.Backend.Caps().Collective)
			} else {
				logf("gateway: expert-parallel ranks=%d: backend %q exposes no device collective (%v) — reducing host-side via LocalCollective (correct, single-box)", cfg.ExpertParallelRanks, cfg.Backend.Name(), err)
			}
		}
	} else if cfg.ExpertParallelRanks > 1 && cfg.InKernelModel.IsExpertParallelRankLocal() {
		// A SHARDED EP rank: the serve already set the rank, the world size, and the DistComm
		// process-group collective (each rank holds only its band, reduces cross-process). Do
		// NOT re-wire a single-process device/Local collective here — it would clobber the
		// cross-process reduce and break the sharded serve (#971).
		logf("gateway: expert-parallel ranks=%d rank-local (sharded serve) — reducing through the serve's DistComm process group, device-collective wiring skipped", cfg.ExpertParallelRanks)
	}
	return agent.NewInKernelPlanner(cfg.InKernelModel, cfg.Tokenizer, modelID, cfg.InKernelQ4K, cfg.Backend, cfg.Metal, cfg.CPUOffloadExperts)
}

func newProxyPlanner(cfg Config, model string, baseURLs []string) (agent.Planner, error) {
	if len(baseURLs) == 1 {
		// A lone upstream needs no replica identity, but still honor a name=URL form
		// (an operator may pin one) by dialing only the URL part.
		_, dialURL := parseReplicaEntry(baseURLs[0])
		return newConfiguredHTTPPlanner(cfg, model, dialURL)
	}
	replicas := make([]PlannerReplica, 0, len(baseURLs))
	for _, base := range baseURLs {
		// Stable, order-independent identity (#3968): use the operator-chosen id from a
		// name=URL entry, else derive replica-<digest> from the endpoint so the same
		// upstream keeps one identity — and one set of metric/residency labels — no
		// matter its flag position or a membership change that drops a peer.
		name, dialURL := parseReplicaEntry(base)
		if name == "" {
			name = deriveReplicaName(dialURL)
		}
		p, err := newConfiguredHTTPPlanner(cfg, model, dialURL)
		if err != nil {
			return nil, err
		}
		replicas = append(replicas, PlannerReplica{
			Name:    name,
			Planner: p,
		})
	}
	return NewReplicaRouter(model, replicas)
}

// newConfiguredHTTPPlanner dials one upstream and applies every Config-derived knob to
// the resulting planner — the wiring newProxyPlanner performs once for a lone upstream
// and once per replica, so a new Config field is threaded through in exactly one place.
func newConfiguredHTTPPlanner(cfg Config, model, dialURL string) (*agent.HTTPPlanner, error) {
	p, err := agent.NewProviderHTTPPlanner(cfg.Provider, dialURL, model, cfg.APIKey)
	if err != nil {
		return nil, err
	}
	p.APIKeyFunc = cfg.APIKeyFunc
	p.AccountFailoverFunc = cfg.AccountFailoverFunc
	p.ExtraHeaders = cloneConfigHeaders(cfg.ExtraHeaders)
	p.ExtraHeadersFunc = cfg.ExtraHeadersFunc
	p.ForceResponsesStream = cfg.ForceResponsesStream
	// Passed through verbatim: Config.StreamProgressTimeout carries the planner field's own
	// encoding (0 = the agent default, negative = disabled, out-of-band = the default), so a
	// Config nobody configures leaves the planner at the 300s default byte-for-byte.
	p.StreamProgressTimeout = cfg.StreamProgressTimeout
	wrapUpstreamObserver(p.Client, cfg.UpstreamResponseObserver, cfg.UpstreamTransportErrorObserver)
	return p, nil
}

func cloneConfigHeaders(in map[string]string) map[string]string {
	if len(in) == 0 {
		return nil
	}
	out := make(map[string]string, len(in))
	for k, v := range in {
		if strings.TrimSpace(k) != "" {
			out[k] = v
		}
	}
	return out
}

// MarkReady stamps the instant the gateway became able to serve requests, closing
// the boot timeline (fak_gateway_time_to_ready_seconds / fak_gateway_ready_time_
// seconds). Idempotent and safe on a nil-startup Server; the first call wins.

// RecordStartupPhase appends a host-timed startup stage, including work after MarkReady.
func (s *Server) RecordStartupPhase(name string, dur time.Duration, provenance string) {
	s.startup.phaseWithProvenance(name, dur, provenance)
}

// BeginChildStartup starts the external child cold-start observation window.
func (s *Server) BeginChildStartup(at time.Time) { s.startup.beginChildStartup(at) }

// MarkChildUsable closes the child window at the first authenticated API request.
func (s *Server) MarkChildUsable(at time.Time) { s.startup.markChildUsable(at) }

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
	sum := s.metrics.adjudicationSummary()
	// The compaction budget lives on the Server, not the metrics ledger; attach it here so
	// the exit line can distinguish "enabled but idle" from "disabled" (0).
	sum.CompactionBudget = s.compactHistoryBudget
	return sum
}

// KernelCounters returns a snapshot of the kernel's call-path tallies (engine
// dispatches, vDSO hits, in-syscall repairs, fast-reject denies) — the raw counts a
// tier-4 caller folds through internal/callavoid to render the avoided-call
// amplification headline for the `fak guard` exit summary. The verdict roll-up
// (allowed/denied/…) is AdjudicationSummary; this is the orthogonal call-path axis
// (was the call avoided, and how much further did the agent get per real dispatch?),
// read straight from the same kernel.Counters the fak_kernel_* metrics expose. Safe
// on a nil Server (returns the zero Counters).
func (s *Server) KernelCounters() kernel.Counters {
	if s == nil {
		return kernel.Counters{}
	}
	return s.k.Counters()
}

// HarnessCoherenceSummary returns the operator-line roll-up of the harness-coherence
// family (the same numbers the fak_harness_coherence_* scrape reports). It is exposed
// next to AdjudicationSummary/KernelCounters so a host can persist the session's
// ObservedTurns — the REAL count of served passthrough turns — at exit. That count is
// the honest session-length signal the gateway-usage ledger records: Submits is 0 on the
// guard proxy path (the kernel submit boundary is not on the pass-through wire), so the
// durable turn-distribution corpus would otherwise have only CachedTurns as a proxy.
// Safe on a nil Server (returns the zero summary — a nil Server served no turns).
func (s *Server) HarnessCoherenceSummary() HarnessCoherenceSummary {
	if s == nil {
		return HarnessCoherenceSummary{}
	}
	return s.metrics.harnessCoherenceSummary()
}

// AssumeSessionTurns returns the resolved session-length prior (Config.AssumeSessionTurns;
// 0 = the head-anchored burst gate's prior is disabled) this Server booted with. Exposed
// so a host can stamp it into a persisted row's Provenance — the corpus that CALIBRATES
// DefaultAssumedSessionTurns must be able to exclude override sessions from the fit. Safe
// on a nil Server (returns 0).
func (s *Server) AssumeSessionTurns() int {
	if s == nil {
		return 0
	}
	return s.assumeSessionTurns
}

// CompactHistoryBudget returns the resident-token budget compaction fires against
// (Config.CompactHistoryBudget; 0 = compaction OFF), the sibling provenance knob to
// AssumeSessionTurns. Safe on a nil Server (returns 0). AdjudicationSummary also surfaces
// this as CompactionBudget; this dedicated accessor lets a provenance stamp read it without
// folding the whole counter roll-up.
func (s *Server) CompactHistoryBudget() int {
	if s == nil {
		return 0
	}
	return s.compactHistoryBudget
}

// ExposeProfile returns the descriptive expose-profile label this session launched under
// (Config.ExposeProfile; "" when unset), the sibling provenance knob to AssumeSessionTurns
// and CompactHistoryBudget. Safe on a nil Server (returns ""); the caller normalizes an
// empty value to "interactive".
func (s *Server) ExposeProfile() string {
	if s == nil {
		return ""
	}
	return s.exposeProfile
}

// VCacheTurnsSnapshot returns a copy of the per-turn provider-cache window this session
// observed (input/cache_read/cache_creation tokens per turn, the OBSERVED axis fed by
// observeVCacheTurn on every streamed passthrough turn), plus whether the bounded window
// has dropped older turns. It is the live source `fak vcache score` reads to report the
// REALIZED cache multiplier instead of the synthetic-Zipf forecast — exposed here, next to
// AdjudicationSummary, so a host can persist it at session exit. Safe on a nil Server.
func (s *Server) VCacheTurnsSnapshot() ([]vcacheobserve.Turn, bool) {
	if s == nil {
		return nil, false
	}
	turns, capped := s.metrics.vcacheTurnsSnapshot()
	attachVCacheContextEconomics(turns, s.compactHistoryBudget)
	return turns, capped
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
	if hasStructuredToolContinuation(messages) {
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

func hasStructuredToolContinuation(messages []agent.Message) bool {
	for _, m := range messages {
		if len(m.ToolCalls) > 0 || m.FunctionCall != nil || m.ToolCallID != "" {
			return true
		}
	}
	return false
}

func (s *Server) observeDecodedCtxViewRewrite(trace string, full, planned []agent.Message) {
	if s == nil || len(planned) >= len(full) {
		return
	}
	fullTokens := estimateMessageContentTokens(full)
	plannedTokens := estimateMessageContentTokens(planned)
	shed := 0
	if fullTokens > plannedTokens {
		shed = fullTokens - plannedTokens
	}
	s.metrics.observeCtxViewRewrite(agent.CompactOutcome{
		Reason:     agent.CompactReasonNone,
		Dropped:    len(full) - len(planned),
		ShedTokens: shed,
	})
	trace = strings.TrimSpace(trace)
	if trace == "" {
		return
	}
	s.ctxViewPendingMu.Lock()
	if s.ctxViewPending == nil {
		s.ctxViewPending = map[string]int{}
	}
	s.ctxViewPending[trace]++
	s.ctxViewPendingMu.Unlock()
}

func (s *Server) consumeDecodedCtxViewEvent(trace string) bool {
	if s == nil {
		return false
	}
	trace = strings.TrimSpace(trace)
	if trace == "" {
		return false
	}
	s.ctxViewPendingMu.Lock()
	defer s.ctxViewPendingMu.Unlock()
	n := s.ctxViewPending[trace]
	if n <= 0 {
		return false
	}
	if n == 1 {
		delete(s.ctxViewPending, trace)
	} else {
		s.ctxViewPending[trace] = n - 1
	}
	return true
}

func estimateMessageContentTokens(messages []agent.Message) int {
	total := 0
	for _, m := range messages {
		if m.Content == "" {
			continue
		}
		total += (len(m.Content) + 3) / 4
	}
	return total
}

type suppressDecodedCtxViewKey struct{}

func withDecodedCtxViewSuppressed(ctx context.Context) context.Context {
	return context.WithValue(ctx, suppressDecodedCtxViewKey{}, true)
}

func decodedCtxViewSuppressed(ctx context.Context) bool {
	v, _ := ctx.Value(suppressDecodedCtxViewKey{}).(bool)
	return v
}

// maybeElideKVResidency drives the model-side PLANNED-ELISION residency bridge (issue #579, the
// kvmmu-planned-eviction half) when the context planner shrank the turn history. It is the
// capacity-plan twin of evictInKernelPoison's KVSpanEvictor (which enforces a trust quarantine):
// the planner's O(1) text view becomes a real O(1) KV RESIDENCY, with the elided spans' K/V
// actually evicted from the kernel-owned cache instead of physically held behind an O(1) view.
//
// Honest posture (issue #579, "bit-exact provable direction only"): the bridge engages on the
// plan the context planner produced, and ElideKVSpans evicts every elided span — but it asserts
// the bit-exact O(1)-residency invariant ONLY when the elided spans are the positional SUFFIX (the
// over-budget direction the kvmmu witness proves: a re-RoPE renumbers survivors with no surviving
// earlier token having attended to an evicted later one). Eliding an old prefix the recent
// resident tail already attended to still shrinks residency but is NOT reported bit-exact, rather
// than overclaiming an invariant a re-RoPE cannot satisfy.
//
// It is a no-op (returns silently) unless the planner implements KVSpanElider (the in-kernel
// engine with FAK_INKERNEL_KVMMU on) AND the planned view is a clean sub-sequence of the history
// (a reorder/rewrite is left untouched — the bridge fails OPEN rather than evict the wrong span).
// Default posture is therefore unchanged unless an operator opts the in-kernel bridge in.
func (s *Server) maybeElideKVResidency(fullHistory, planned []agent.Message) {
	elider, ok := s.planner.(agent.KVSpanElider)
	if !ok {
		return
	}
	// Recover which fullHistory messages the planner elided. The planned view must be a clean
	// trailing SUFFIX of fullHistory (planning dropped a leading prefix); a reorder/rewrite is
	// not a shape the residency bridge can map safely, so it is skipped (fail-open).
	elided, ok := elidedPrefixMask(fullHistory, planned)
	if !ok {
		return
	}
	plan := agent.SegElisionPlan(fullHistory, elided)
	if len(plan.Elided) == 0 {
		return // planning kept everything resident — nothing to evict
	}
	if freed, exact := elider.ElideKVSpans(fullHistory, plan); freed > 0 {
		s.logf("gateway: in-kernel KV residency shrank to planned view elided=%d freed=%dpos reposition_exact=%v", len(plan.Elided), freed, exact)
	}
}

// elidedPrefixMask recovers which fullHistory messages the planner elided, for the case the
// planned view is a trailing SUFFIX of fullHistory (planning dropped a leading prefix). It returns
// a mask where the leading prefix is elided (true) and the resident suffix is kept (false), and
// ok=false when the planned view is not a clean trailing suffix (a reorder or rewrite). Compares
// role+content, the fields renderTranscript lowers into the spans the bridge evicts.
//
// NOTE the bit-exactness direction is decided downstream: a trailing-suffix RESIDENT view means
// the ELIDED spans are the leading prefix — the non-bit-exact direction — so ElideKVSpans will
// shrink residency but report reposition_exact=false here. The provable (suffix-elided) direction
// is exercised by the unit witness driving SegElisionPlan directly. This gate is the conservative
// pre-filter; ElideKVSpans re-checks positional order and is the load-bearing proof.
func elidedPrefixMask(fullHistory, planned []agent.Message) (mask []bool, ok bool) {
	if len(planned) == 0 || len(planned) >= len(fullHistory) {
		return nil, false
	}
	off := len(fullHistory) - len(planned)
	for i := range planned {
		if planned[i].Role != fullHistory[off+i].Role || planned[i].Content != fullHistory[off+i].Content {
			return nil, false
		}
	}
	mask = make([]bool, len(fullHistory))
	for i := 0; i < off; i++ {
		mask[i] = true
	}
	return mask, true
}

// maybeElideMessages shrinks oversized OLD tool-role message Content to a bounded head+tail on
// the DECODED []Message path — the OpenAI / in-kernel wire a LOCAL model served by fak takes
// (GLM-5.2 / Qwen-3.6-27B via an OpenAI backend or the in-kernel engine), where the byte-splice
// ElideAnthropicResults — which only fires on the real-Anthropic passthrough — never runs. It is
// the decoded-path twin of maybeElideAnthropicRaw, so oversized-result elision is enabled by
// default on BOTH wires. Guarded OFF on the passthrough (handled there on req.Raw) and when
// --elide-result-bytes is 0. agent.ElideMessages is copy-on-write and fail-safe (it only ever
// SHORTENS an old tool message, never empties a turn, recent working set protected), so this can
// never break a turn or mutate the caller's slice.
func (s *Server) maybeElideMessages(messages []agent.Message) []agent.Message {
	if s.elideResultBytes <= 0 || s.anthropicPassthrough() {
		return messages
	}
	out, outcome := agent.ElideMessages(messages, s.elideResultBytes)
	s.metrics.observeUncachedTrim(outcome)
	return out
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

// existingSessionPlanner returns the retained per-trace SessionPlanner WITHOUT minting one — the
// read-only counterpart of sessionPlannerFor. The restore path (fak_context_restore, #3062) uses it
// so a restore call for a trace that never planned a turn is a plain miss that falls through to the
// next source, rather than a freshly-minted empty planner (which would allocate, pollute the map,
// and still miss). It does not gate on ctxView.Enabled: if a planner was retained for the trace its
// lossless store is a valid ctxview-elision source regardless of the current seam flag. The returned
// planner is safe to call Spans/Materialize on concurrently — those methods take the planner's own
// lock — so the read holds sessionPlannerMu only for the map lookup.
func (s *Server) existingSessionPlanner(trace string) *agent.SessionPlanner {
	if trace == "" {
		return nil
	}
	s.sessionPlannerMu.Lock()
	defer s.sessionPlannerMu.Unlock()
	return s.sessionPlanners[trace]
}

// complete runs the configured planner for one turn and records the inference
// metrics that make real model work visible at /metrics — the token counts the
// planner reports plus the wall-clock spent generating. Both /v1/chat/completions
// and /v1/messages route through it so the fak_gateway_inference_* family reflects
// every served turn on either wire. On a planner error nothing is recorded (a turn
// that produced no tokens is not a generation); the error is returned untouched so
// the caller's existing error handling is unchanged.
func (s *Server) complete(ctx context.Context, trace string, messages []agent.Message, tools []agent.ToolDef, opts ...agent.SampleOpt) (comp *agent.Completion, err error) {
	// Bind authenticated request identity to in-kernel prefix-cache visibility.
	// Empty principal preserves legacy single-user reuse; authenticated prefixes
	// enter tenant-private storage and cannot shape another tenant's hit timing.
	if principal := principalFromContext(ctx); principal != "" {
		ctx = agent.WithPrefixCacheIdentity(ctx, principal, "")
	}
	defer func() {
		if r := recover(); r != nil {
			if evictErr, ok := recoverRecurrentEvictUnsupported(r); ok {
				comp, err = nil, evictErr
				return
			}
			panic(r)
		}
	}()
	// Re-plan the turn history into an O(1) resident view before the model sees it —
	// the "replace append+compact with a planned view" rung (issue #555). Inert (an
	// identity) unless CtxViewBudget > 0, so the default path is unchanged. The trace keys
	// the persistent per-session planner so the rewrite is O(c·N), not O(N²), across turns.
	fullHistory := messages
	messages = s.maybePlanMessages(ctx, trace, messages)
	if !decodedCtxViewSuppressed(ctx) {
		s.observeDecodedCtxViewRewrite(trace, fullHistory, messages)
	}
	// Shrink the kernel-owned KV residency to match the planned O(1) view (issue #579, the
	// kvmmu-planned-eviction half): when planning ELIDED older history (the view is a strict
	// trailing window of the full transcript), drive the model-side residency bridge so the
	// elided spans' K/V is actually evicted via model.KVCache.Evict — making the "O(1) view" a
	// real O(1) KV RESIDENCY instead of an O(1) text view over an O(N) cache. Default OFF /
	// fail-open: a no-op unless FAK_INKERNEL_KVMMU opted the in-kernel bridge in.
	s.maybeElideKVResidency(fullHistory, messages)
	// Oversized tool_result elision on the DECODED path — the OpenAI / in-kernel wire a LOCAL
	// model served by fak takes (GLM-5.2 / Qwen-3.6-27B), where the byte-splice passthrough
	// elision never fires. Shrinks old oversized tool-role content to head+tail; default-on,
	// fail-safe, recent working set protected. No-op on the Anthropic passthrough (handled on
	// req.Raw there).
	messages = s.maybeElideMessages(messages)
	start := time.Now()
	comp, err = s.planner.Complete(ctx, messages, tools, opts...)
	dur := time.Since(start)
	if err != nil {
		if _, _, _, ok := inKernelOOMObservation(err); ok {
			s.observePlannerRequestMemory()
		}
		return nil, err
	}
	s.metrics.observeInferenceServed(s.servedLocalityOf(opts), comp.Usage.PromptTokens, comp.Usage.CompletionTokens, comp.Usage.CachedPromptTokens(), comp.Usage.CacheCreationInputTokens, comp.FinishReason, dur)
	s.observePlannerRequestMemory()
	// The served turn has mutated the KV cache; relieve HBM pressure by demoting a hot span to
	// the colder tier instead of dropping it (#1073, the live serve-path call site for the
	// capacity executor). Fail-open + gated: a no-op unless FAK_INKERNEL_KVMMU armed the bridge
	// AND the host injected a device-backed provider+sweeper via SetKVPressureRelief.
	s.maybeRelieveKVPressure(ctx)
	return comp, nil
}

func recurrentEvictUnsupported(err error) bool {
	var evictErr *model.RecurrentEvictUnsupportedError
	return errors.As(err, &evictErr)
}

func recoverRecurrentEvictUnsupported(r any) (error, bool) {
	err, ok := r.(error)
	if !ok || !recurrentEvictUnsupported(err) {
		return nil, false
	}
	return err, true
}

// completeServed is complete plus the served-session usage debit. The request
// boundary has already called beginServedSessionTurn (and therefore Decide); after a
// successful planner response the provider usage is finally known, so debit the
// output/context budgets here. Planner errors keep the old behavior: no usage was
// reported, so there is nothing to debit beyond the turn admission already taken.
func (s *Server) completeServed(ctx context.Context, turn servedSessionTurn, messages []agent.Message, tools []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error) {
	messages = applyNegframeRequestPass(messages, turn.traceID)
	// Mark the trace as holding an open model request so the agents pane can show its
	// in-flight age (#2627). The window closes when this call returns; adjudication then
	// stamps last_tool/idle from the served turn. Cleared on every exit path via defer.
	began := time.Now()
	s.activity.beginTurn(turn.traceID, began)
	defer s.activity.endTurn(turn.traceID)
	lease, err := s.beginServedAdmission(ctx, turn, messages, tools, sampleMaxTokens(opts))
	if err != nil {
		return nil, err
	}
	defer lease.Release()
	comp, err := s.complete(ctx, turn.traceID, messages, tools, opts...)
	if err != nil {
		return nil, err
	}
	// The provider's real usage is now known — settle the token-rate window with it
	// (#2019), replacing the admission-time estimate.
	lease.SettleUsage(comp.Usage)
	s.debitServedSessionTurn(ctx, turn, comp.Usage, time.Since(began), messages)
	return comp, nil
}
