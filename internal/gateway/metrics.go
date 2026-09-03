package gateway

import (
	"encoding/json"
	"errors"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/cachemeta"
	"github.com/anthony-chaudhary/fak/internal/cacheobs"
	"github.com/anthony-chaudhary/fak/internal/compactcohere"
	"github.com/anthony-chaudhary/fak/internal/metrics"
	"github.com/anthony-chaudhary/fak/internal/vcacheobserve"
)

var gatewayLatencyBuckets = []float64{
	0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1,
	0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300, 600, 900, 1800,
}

type gatewayMetrics struct {
	start    time.Time
	inflight int64

	mu         sync.Mutex
	http       map[httpMetricKey]*latencyCounter
	operations map[operationMetricKey]*latencyCounter

	// inflightMu guards the live in-flight request registry. It is kept separate
	// from mu (which guards the completion-time histograms) because begin/end run
	// on the hot path for EVERY request, and renderMetrics walks the registry at
	// scrape time to derive signals the completion-time histograms structurally
	// cannot show: a request that is still running has not been observed into any
	// histogram yet, so a slow or wedged in-flight request is otherwise invisible
	// until it finishes (or never, if it hangs).
	inflightMu  sync.Mutex
	inflightReq map[uint64]inflightEntry
	inflightSeq uint64

	// inferenceMu guards the model-inference accumulators. These count the REAL
	// generation work every served chat/messages turn does (token counts, finish
	// reason, decode wall-clock) — work that otherwise reaches only a log line and
	// never /metrics, leaving the kernel/vDSO counters (which a pure chat turn does
	// not exercise) reading 0 and the dashboard looking dead on a busy box.
	inferenceMu       sync.Mutex
	inferReqs         map[string]uint64 // finish_reason -> turns served
	inferPromptTokens uint64
	inferComplTokens  uint64
	inferCachedTokens uint64
	inferCachedHits   uint64 // served turns whose prompt got a provider cache READ (>0 cached tokens)
	// The self-hosted split of the SAME volume the unsplit totals above accumulate,
	// attributed by servedLocality at the observation (epic #5416). Both groups are
	// strict subsets: a turn whose side could not be resolved lands in neither, so
	// local+vendor <= the unsplit total and the shortfall is the honest measure of
	// how much volume went unclassified. Never derive one group by subtracting the
	// other from the total — that would silently book every unclassified turn as
	// self-hosted, which is the one direction of error that flatters us.
	inferSelfHostedTurns        uint64
	inferSelfHostedPromptTokens uint64
	inferSelfHostedComplTokens  uint64
	inferVendorTurns            uint64
	inferVendorPromptTokens     uint64
	inferVendorComplTokens      uint64
	// inferCacheCreationTokens is the cumulative provider cache_creation_input_tokens —
	// the WRITE axis the read-only ProviderCacheSavingsUSD never retained. With it and
	// the read total, the session can report NET realized vcache economics (read saving
	// minus write premium) via the same engine `fak vcache observe` uses offline.
	inferCacheCreationTokens uint64
	// inferCacheCreationTokensUpgraded is the SUBSET of inferCacheCreationTokens whose
	// write happened on a turn where the managed-cache 1h TTL-upgrade rung
	// (maybeUpgradeAnthropicCacheTTL1H) was already active for the session — GATEWAY-
	// ATTRIBUTED (fak's own per-turn upgrade witness), never provider-reported: the
	// Anthropic usage block does not split 5m vs 1h creation tokens (#2179). A turn's
	// write only lands here when Server.ttl1hActiveFor reports true for its trace.
	inferCacheCreationTokensUpgraded     uint64
	inferCacheCreationTokensTierObserved uint64
	// Split the upgraded write arm into the original head-only baseline and the
	// message-prefix extension (#2186). Provider usage cannot subdivide a single
	// write further, so each served turn is attributed by its admitted layout.
	inferCacheCreationTokensHeadOnly      uint64
	inferCacheCreationTokensMessagePrefix uint64
	inferDecodeSecs                       float64
	// Prefill (time-to-first-token) is split from decode ONLY on a path that can
	// observe the first content delta — the streaming Anthropic passthrough. On a
	// buffered turn the planner returns one all-up duration with no observable
	// first-token boundary, so it contributes to inferDecodeSecs (the total) and
	// leaves these two untouched. inferTTFTTurns is the denominator that keeps the
	// prefill rate honest: it counts ONLY turns whose TTFT was actually measured, so
	// a mixed buffered+streaming workload never divides streamed prefill tokens by
	// buffered turns. inferPrefillPromptTokens is the prompt-token sum over just
	// those measured turns (the prefill-rate numerator).
	inferPrefillSecs         float64
	inferTTFTTurns           uint64
	inferPrefillPromptTokens uint64
	// Decode side of the measured turns, kept as its own pair so the decode rate
	// never mixes denominators: inferMeasuredDecodeSecs sums (dur-ttft) and
	// inferMeasuredComplTokens sums completion tokens over the SAME ttftTurns. On a
	// mixed buffered+streaming workload this divides measured completion tokens by
	// measured decode time only — subtracting the all-workload decode total would
	// skew it.
	inferMeasuredDecodeSecs  float64
	inferMeasuredComplTokens uint64
	// Latency DISTRIBUTIONS, not just the cumulative MEANS above. The prefill/decode
	// rates are session-cumulative gauges that structurally hide the P95/P99 tail an
	// operator watches under load. These three Prometheus histograms record per-turn
	// time-to-first-token, per-output-token (inter-token) latency, and whole-turn
	// wall-clock, fed on the SAME turn-completion path under inferenceMu, so
	// P50/P95/P99 become queryable. They stay empty (count 0) on an idle gateway — no
	// phantom distribution — and are the fak analogues of vLLM's
	// time_to_first_token / inter_token_latency / e2e_request_latency histograms.
	inferTTFTHist *latencyCounter
	inferTPOTHist *latencyCounter
	inferE2EHist  *latencyCounter

	// reqMemoryMu guards cumulative in-kernel request-memory pressure observed after
	// planner turns. The planner already exposes the most recent admission plan; these
	// accumulators keep the per-class totals/high-water marks so pressure is visible
	// over time instead of only at scrape-time last value.
	reqMemoryMu       sync.Mutex
	reqMemoryObserved map[string]uint64 // backend -> turns with an observed request plan
	reqMemoryPlan     map[requestMemoryMetricKey]*requestMemoryMetricStats
	reqMemoryTokens   map[requestMemoryTokenKey]*requestMemoryTokenStats
	reqMemoryFit      map[requestMemoryFitKey]*requestMemoryFitStats

	// compactMu guards the history-compaction accumulators. Compaction (the
	// --compact-history-budget lever on the Anthropic passthrough) is otherwise INVISIBLE: it
	// returns identity on any bail with no signal, so a silent failure reads like success.
	// These split cleanly into what fak CONTROLS and what it only OBSERVES — a distinction the
	// surface must keep, or a provider-side miss reads as a fak bug:
	//   - WITNESSED (fak authored): attempts{fired|bailed|off}, the bail reason, dropped, shed.
	//     A turn only counts `fired` if the protected-prefix bytes were byte-identical to the
	//     input (else it bails to `prefix_mismatch`), so these are facts about what fak SENT.
	//   - OBSERVED (provider-reported, relayed): compactCacheReads / compactLastCacheRd are the
	//     upstream's cache_read_input_tokens. fak attributes nothing to itself from them.
	// Kept off inferenceMu — different hot path, no lock coupling.
	compactMu            sync.Mutex
	compactAttempts      map[string]uint64 // WITNESSED: outcome -> count: fired | bailed | off
	compactBailReasons   map[string]uint64 // WITNESSED: CompactReason* -> count (why a bail happened)
	compactDropped       uint64            // WITNESSED: whole messages stubbed out across all fires
	compactShed          uint64            // WITNESSED: estimated tokens fak removed from the body across all fires
	compactCacheReads    uint64            // OBSERVED: sum of provider-reported cache_read on compacted turns
	compactLastCacheRd   float64           // OBSERVED: provider-reported cache_read on the MOST RECENT compacted turn
	compactAnchorStarved uint64            // WITNESSED: under_budget bails whose protected prefix ALREADY exceeded the budget — the cache_control anchor swallowed the conversation so the lever structurally cannot fire (#1407). A subset of bailReasons[under_budget]; the signal that idle is NOT a benign short session.
	// WITNESSED headroom: how far the MOST RECENT under_budget bail sat from firing
	// (agent.CompactOutcome.SuffixTokens, the compactible messages[] span — the system/tools floor
	// is a separate top-level block and is NOT counted here). The bail reason alone cannot tell a
	// session that is one turn from the cut apart from one structurally incapable of ever reaching
	// it, because the compactor computes this split on every bail and then discards it. Recording
	// the last and peak values turns "under_budget x872" into a distance, which is the difference
	// between "the budget is fine, this session is short" and "the line sits above the band this
	// traffic ever occupies". Zero until the first under_budget bail carries a span.
	compactLastSuffixTokens uint64
	compactPeakSuffixTokens uint64
	// WITNESSED: sessions whose window refilled to the compaction limit ctxThrashConsecutiveRefills
	// turns RUNNING (#2424, ctxadvice.go). Counted ONCE per thrashing stretch. Deliberately its
	// own counter rather than a compactAttempts["bailed"] + compactBailReasons row: compaction
	// FIRED on every one of those turns, so booking it as a bail would both inflate the bailed
	// lump and (via compactBailPartition) skew the alertable candidate-bail rate with a reason the
	// compactor never emitted. It joins the AdjudicationSummary.CompactionBailReasons map at read
	// time instead, where an operator asking "why is compaction not holding this session" reads it.
	compactThrashSessions uint64
	compactSolvencyForced uint64            // WITNESSED: fires the burst economics REFUSED that the context-solvency floor forced anyway (agent.CompactOutcome.SolvencyForced). A subset of attempts[fired]; these are deliberately unprofitable bursts bought to keep the session inside its window, so they must never be read as cache wins.
	uncachedTrimResults   uint64            // WITNESSED: oversized old tool_result bodies shrunk by the uncached-tail trim.
	uncachedTrimShed      uint64            // WITNESSED: estimated tokens removed by uncached-tail trim, folded into compactShed for fak attribution.
	ttlUpgrades           map[string]uint64 // WITNESSED: managed-cache 1h TTL upgrade attempts by outcome ("upgraded" | agent.TTLUpgradeReason*). Recorded only while the lever (--managed-cache / CacheTTL1H) is on, so a zero panel with the lever active means every head was ineligible — visible, not silent.
	placementAttempts     map[string]uint64 // WITNESSED: offensive cache-breakpoint placement attempts by outcome ("placed" | agent.BreakpointReason*). "placed" is the fak-authored slice — a breakpoint spliced onto a caller that sent none, so the provider cache_read it earns is fak-unlocked; "already_set" is the Claude-Code shape fak leaves alone (the client's cache, not fak's).
	// anchorMon is the LIVE loop over that same placement stream (#3622): the counter above is a
	// cumulative tally nobody watches, so a session whose head turns volatile mid-conversation just
	// stops incrementing "placed" and nothing fires. The monitor folds each outcome into a rolling
	// refused fraction over DECISIVE turns only and raises ANCHOR_REFUSED_RISING on the crossing.
	// Guarded by compactMu — the same lock observePlacement already holds, so the fold costs no new
	// synchronization and can never race the counter it mirrors.
	anchorMon *metrics.AnchorRefusalMonitor

	// ctxViewMu guards ctxplan planned-view rewrites. This is a CONTEXT-plane witness,
	// but not a history-compaction attempt: the planner materialized an O(1) resident
	// view onto req.Raw and proved the cached prefix survived, so vcache scoring should
	// credit the context plane without inflating compaction_fired/debug lines.
	ctxViewMu      sync.Mutex
	ctxViewEvents  uint64
	ctxViewDropped uint64
	ctxViewShed    uint64

	// toolPruneMu guards the INBOUND tool-definition prune accumulators (the twin of
	// the compaction family above, for the tools[] axis). maybeCompactInboundTools drops
	// tool DEFINITIONS the floor can NEVER admit from the outbound tools[] — but only the
	// ones strictly AFTER the cache_control breakpoint, so the cached prefix stays
	// byte-identical. Like compaction it is otherwise INVISIBLE: it returns identity with no
	// signal, so an operator cannot tell a lever that delivered tool-def cache savings from
	// one that fired zero times. These are WITNESSED (fak authored — fak chose what to drop):
	//   - toolPruneTurns: turns on which at least one tool def was pruned.
	//   - toolPruneCount: cumulative tool defs removed across all those turns.
	// Kept off compactMu — a different transform with its own (rare) hot path.
	toolPruneMu       sync.Mutex
	toolPruneTurns    uint64 // WITNESSED: turns where >=1 unreachable tool def was pruned from tools[]
	toolPruneCount    uint64 // WITNESSED: total tool defs removed across all prune turns
	toolPrunedPropose uint64 // WITNESSED: pruned tool names later proposed, deduped once per trace/tool

	// deferMu guards the tool-DEFERRAL accumulators (#3232, the 10x floor lever): turns
	// where the cold tool tail was marked defer_loading + a tool_search_tool injected, and
	// the cumulative cold-def count. WITNESSED: the transform proved the non-tools body
	// bytes stayed byte-identical and re-decoded before Changed, so a counted defer never
	// bursts the upstream cache.
	deferMu         sync.Mutex
	deferFiredTurns uint64 // WITNESSED: turns where the cold tool tail was deferred
	deferColdCount  uint64 // WITNESSED: total cold defs marked defer_loading across those turns
	// deferColdNames is the DISTINCT set of custom tool names fak marked defer_loading across
	// those turns (#3647) — the "which tools were deferred" the operator pane names. Held as a
	// set so the deterministic per-turn defer of the same cold tail does not multiply-list a
	// name; nil until the lever first fires.
	deferColdNames map[string]struct{}
	// deferStandDownTurns / deferStandDownReasons are the DENOMINATOR the #3621
	// DEFER_ENABLED_BUT_INERT watchdog needs: turns on which the transform actually RAN — the
	// lever on, the wire the Anthropic passthrough, the ablation arm off — and still stood down
	// to byte-identity, keyed by the deferResult reason. Without them a zero-defer session is
	// indistinguishable from a lever-off one, which is exactly the silent-identity blind spot.
	// They accrue only PAST maybeDeferColdTools' eligibility gate, so a lever-off, non-Anthropic,
	// or ablated session can never trip the finding. Nil map until the first stand-down.
	deferStandDownTurns   uint64
	deferStandDownReasons map[string]uint64

	// Aggregate-only native tool-filter receipts; never descriptors, names, or schemas.
	toolFilterMu     sync.Mutex
	toolFilterEvents uint64
	toolFilterTools  uint64
	toolFilterBytes  uint64
	toolFilterTokens uint64

	// Aggregate-only stale-read elision receipts; never paths or content.
	staleElideMu     sync.Mutex
	staleElideTurns  uint64
	staleElideReads  uint64
	staleElideBytes  uint64
	staleElideTokens uint64
	staleElideBails  map[string]uint64

	// toolRefMu guards the tool_reference SANITIZE accumulators (a correctness transform, not a
	// cache saving): the client's INTERNAL `tool_reference` blocks — emitted inside a ToolSearch
	// tool_result — are not a valid Anthropic tool_result.content type, so a body carrying one is
	// 400'd upstream as malformed. sanitizeAnthropicToolReferences rewrites each into a wire-valid
	// text block before relay. WITNESSED (fak authored the rewrite):
	//   - toolRefTurns:     turns on which >=1 tool_reference block was converted.
	//   - toolRefConverted: cumulative tool_reference blocks converted across all those turns.
	// A body with no tool_reference records nothing (the common turn), exactly like a clean prune.
	toolRefMu        sync.Mutex
	toolRefTurns     uint64 // WITNESSED: turns where >=1 tool_reference block was converted to text
	toolRefConverted uint64 // WITNESSED: total tool_reference blocks converted across all sanitize turns

	// emptyContentMu guards the general-form OUTBOUND EMPTY-CONTENT GATE accumulators (#3118):
	// the residual backstop to the tool_reference sanitizer above. Where that converts one known
	// client-internal block type, this catches any tool_result whose content array ended up EMPTY
	// for ANY reason (a future client-internal type, or a genuinely empty source result) and
	// backfills a placeholder text block, since an empty content array is itself a 400. WITNESSED:
	//   - emptyContentTurns:    turns on which >=1 empty tool_result.content array was repaired.
	//   - emptyContentRepaired: cumulative empty content arrays backfilled across all those turns.
	// A body with no empty content array records nothing (the common turn), like a clean sanitize.
	emptyContentMu       sync.Mutex
	emptyContentTurns    uint64 // WITNESSED: turns where >=1 empty tool_result.content array was repaired
	emptyContentRepaired uint64 // WITNESSED: total empty content arrays backfilled across all repair turns

	// resetShadowMu guards the per-session resetScore SHADOW accumulators (#792). The reset
	// policy (reset_score.go) recommends cut-vs-reset; this folds the recommend-only verdict
	// stream into /metrics so an operator sees the cut-vs-reset pressure WITHOUT the policy ever
	// acting. The verdict is WITNESSED (fak's own policy); the cache ratios it folds are OBSERVED.
	resetShadowMu        sync.Mutex
	resetShadowReasons   map[string]uint64 // ResetReason -> compacted turns scored that way
	resetShadowRecommend uint64            // compacted turns whose SHADOW verdict was ShouldReset (acted on: none)
	resetShadowLastScore float64           // the most recent turn's 0..1 reset-pressure score

	// harnessCoherence is the #1132 gateway seam onto the shipped compactcohere decision surface:
	// per-trace coordinators + the cross-session fak_harness_coherence_* accumulators. It is the
	// SINGLE source both the /metrics scrape (writeHarnessCoherenceMetrics) and the operator line
	// (#1135, summary) fold, so the two views can never disagree. Its own internal lock guards the
	// per-trace state — kept off the locks above (a different, content-free hot path). Never nil for
	// a newGatewayMetrics'd value.
	harnessCoherence *harnessCoherenceMetrics

	// routing is the #603 (epic #595) gateway seam onto modelroute's per-aspect decision
	// surface: an append-only DecisionJournal of every routing decision the gateway takes on
	// the served path, plus the fak_gateway_routing_* accumulators its Counts() projects into.
	// It is the SINGLE source both the /metrics scrape (writeRoutingMetrics) and the operator
	// roll-up (routingSummary) fold, so the two views can never disagree. Its own lock guards
	// the journal — kept off the locks above (a distinct, one-fold-per-routed-call hot path).
	// Never nil for a newGatewayMetrics'd value; the journal stays empty until a RouteManifest
	// is configured and a tool call routes.
	routing *routingMetrics

	// servingEmitters are host-injected serving-metric producers that already speak
	// the normalized fak_serving_* schema rows: a scrape relabeler for ridden
	// vLLM/SGLang workers and native step-loop emitters as they come online. The
	// gateway renders HELP/TYPE once for the merged row set, so many workers do not
	// duplicate Prometheus family headers.
	servingMu       sync.Mutex
	servingEmitters []ServingMetricsEmitter
	// harnessProvider is a host-injected pull source for the fak_harness_* family
	// (epic #2044): fak guard sets it to its live harness resource sampler's
	// PrometheusText so a running session's CPU/mem/IO is scrapeable, not only printed
	// at exit. Guarded by servingMu; nil renders nothing (the default serve path).
	harnessProvider func() string
	// logvaultProvider is a host-injected pull source for the fak_logvault_* family
	// (#2455): fak guard sets it to the box's capture vault observability
	// (logvault.Vault.MetricsText) so last-capture age, footprint, and verify
	// mismatches are scrapeable. Guarded by servingMu; nil renders nothing.
	logvaultProvider func() string

	// oomMu guards the in-kernel device-OOM visibility family. These are LOCAL resource
	// exhaustion faults: either recovered compute.DeviceAllocError allocations or a request
	// capacity precheck that refused a known-too-large plan before allocation, never provider
	// errors. The Prometheus labels stay class-only to avoid allocator-site cardinality;
	// /debug/vars keeps the most recent site for operator drilldown.
	oomMu       sync.Mutex
	inKernelOOM map[string]*inKernelOOMClassStats

	// upstreamErrMu guards the upstream-error visibility family: a count of proxy/planner
	// turn FAILURES keyed by a KIND (stalled / unreachable / oom / rate_limited / auth /
	// forbidden / transport / status_4xx / status_5xx / other), so an operator can scrape WHY turns are
	// failing — including telling a rate-limit storm apart from an auth-failure storm — not
	// just that the route returned a 502/504. This is the metric twin of the per-turn `fak-turn … FAILED` debug
	// line: the line is glanceable-per-turn, this is cumulative-per-session. Observational
	// only; nothing in the request path reads it.
	upstreamErrMu  sync.Mutex
	upstreamErrors map[string]uint64

	// upstreamRetries counts upstream retry ATTEMPTS (the planner's 429/5xx backoff loop) this
	// session — the otherwise-invisible backoff window. Bumped atomically from the planner's
	// RetryNotify hook, off the request path. The metric twin of the `fak-turn … retry` line.
	upstreamRetries uint64

	// upstreamRetryWaitNS accumulates the wall-clock the backoff loop SLEPT between those
	// attempts, in nanoseconds — the TIME twin of upstreamRetries. The count says how often
	// fak absorbed provider pushback on the session's behalf; this says how much of the
	// session's wall-clock that absorption cost. Before this accumulator each computed wait
	// reached only the per-occurrence debug line and was dropped in aggregate, so the
	// dominant "session felt slow" cause (a long 429 window) was unmeasurable after the fact.
	upstreamRetryWaitNS uint64

	// upstreamAuthRefreshes counts 401 token-rotation self-heals on the rotating-subscription
	// path, keyed by outcome ("recovered" = a fresh OAuth token was adopted mid-session and the
	// call re-sent in place; "exhausted" = no fresher token landed within the grace window, so
	// the 401 surfaced and the agent dropped into its own /login). Guarded by upstreamErrMu (the
	// same low-frequency off-path family). The metric twin of the `fak-turn auth-refresh` line —
	// the observability for the "fak guard gets stuck on login sometimes" event class.
	upstreamAuthRefreshes map[string]uint64

	// upstreamForbiddenRetries counts 403 transient-recovery outcomes, keyed by outcome
	// ("recovered" = a retry within the short bounded window returned 200, so a transient
	// abuse/capacity gate cleared and the live session healed in place; "exhausted" = the
	// window/attempts elapsed still 403ing, so the denial is the permanent entitlement kind that
	// surfaced with the actionable answer). SEPARATE from upstreamAuthRefreshes (a 401 credential
	// rotation is a different cause) and from the forbidden bucket in upstreamErrors (which counts
	// only the terminal 403 that surfaced, not the transient ones this arm absorbed). Guarded by
	// upstreamErrMu. The metric twin of the `fak-turn forbidden-retry` line — the observability the
	// 2026-07-03 gem8 transient-403 storm showed was missing.
	upstreamForbiddenRetries map[string]uint64

	// upstreamAccountFailovers counts ACCOUNT-SCOPED failover outcomes, keyed by outcome
	// ("recovered" = a 403/402 named this credential's org/region/billing as walled and a permitted
	// sibling account was adopted so the turn completed in place; "exhausted" = no failover target
	// existed and the account-scoped denial surfaced). Written by observeUpstreamAccountFailover off
	// the request path under upstreamErrMu. The metric twin of the `fak-turn account-failover` line —
	// the observability behind the org-OAuth-disabled failure, where re-login is futile and only an
	// account swap heals the session.
	upstreamAccountFailovers map[string]uint64

	// lastForbiddenDetail is a SCRUBBED, bounded snapshot of the most recent PERSISTENT 403's
	// upstream body — the operator-only drilldown (loopback /debug/vars) that tells org-disabled
	// apart from model-not-permitted apart from an abuse gate. Set through scrubForbiddenDetail
	// (secrets removed, bounded) so a credential fragment an upstream echoes into an error body
	// never persists. Guarded by upstreamErrMu; empty until a persistent 403 surfaces.
	lastForbiddenDetail string

	// servedInline counts read-only tool calls the vDSO served LOCALLY on a served turn
	// (vDSO live in the hot path): a re-proposed read whose fresh cached answer fak folded
	// into the assistant turn and dropped from the kept set, so the client never re-ran it —
	// the engine round-trip it would otherwise cost is SAVED. Bumped atomically from the
	// adjudication seam on every wire. DISTINCT from the kernel's VDSOHits counter (only the
	// k.Syscall path bumps that); naming a gateway-seam Lookup a kernel-submission hit would
	// conflate provenances. The vDSO's own hits/lookups still update, so fak_vdso_hit_rate
	// reflects these probes too.
	servedInline uint64

	// cacheBreakMu guards the per-session cache-break witness sink (#2916): the
	// CONSUMER half of the cache-integrity invariant. Each entry is one witnessed
	// cache-break — a warm prompt prefix that broke mid-conversation — carrying the
	// closed cause (toolset_change/altered_turn/rebuilt_prompt/provider_quirk/unknown)
	// and the cold-rebuild token cost it caused, folded through internal/metrics so the
	// guard exit summary and the Prometheus surface read the SAME numbers. The SOURCE of
	// the events (the mid-conversation prefix-mutation detector) is sibling #2915;
	// recordCacheBreak is the seam that detector calls. Until #2915 wires a live producer
	// this sink stays empty and the family renders a clean zero (its HELP/TYPE declared,
	// no samples), the same dogfooded-at-zero posture the deny-all/auth-refresh families
	// keep so a panel exists from the first scrape. Kept off inferenceMu — folded only at
	// scrape / exit-summary time, never on the hot path.
	cacheBreakMu     sync.Mutex
	cacheBreakEvents []metrics.CacheBreakEvent

	// vcacheMu guards the per-family live-observe accumulator (#935). The cumulative
	// fak_vcache_* family above is one aggregate row; this retains the per-turn,
	// family-tagged provider-cache telemetry so the live gateway can expose the SAME
	// per-family / governor / warmth / concentration view `fak vcache observe` gives
	// offline — fed through the SAME vcacheobserve.Observe engine, so it reconciles
	// with the offline verb on the same traffic by construction. It is purely
	// observational: nothing in the request path reads it, so correctness never
	// depends on a cache hit (Law A2). Bounded to vcacheTurnCap turns (drop-oldest) so
	// a 24/7 gateway stays flat in memory; the view is over that rolling window, and
	// vcacheTurnsDropped records whether the window has been trimmed. Kept off
	// inferenceMu — folded only at turn-log / scrape time, never on the hot path.
	vcacheMu           sync.Mutex
	vcacheTurns        []vcacheobserve.Turn
	vcacheTurnsDropped bool
	vcacheGovernor     *vcacheGovernorDecisionJournal
	vcacheWarmth       *vcacheWarmthDemotionJournal

	// usageMu guards the per-request usage-record window (#10670): the ONE
	// stable row per completed request (ordinal + provider token axes + cache
	// alignment), retained drop-oldest to usageRecordCap so the
	// /v1/fak/usage/cache-alignment read answers "last N" live instead of via
	// offline journal forensics. Purely observational (Law A2) — nothing in the
	// request path reads it — and kept off inferenceMu like the vcache families:
	// one fold per served turn at the same chokepoint.
	usageMu             sync.Mutex
	usageRecords        []UsageRecord
	usageOrdinals       map[string]uint64
	usageRecordsDropped bool

	// denyAllMu guards the deny-all stop family: a served turn whose EVERY proposed tool
	// call the capability floor refused (kept==0). The wire MUST report such a turn as
	// end_turn (else the client hangs hunting for the dropped tool_use block — the v0.15.0
	// fix), but end_turn halts the agent loop though the model wanted to ACT — a STOP the
	// agent did not choose. These two numbers make that otherwise-invisible "fak ended the
	// turn" legible, and the consecutive gauge is the bounded signal the guard
	// `--deny-all-continue` Stop-hook polls to auto-continue the agent past it. Kept off the
	// locks above — a one-fold-per-served-turn path with no coupling to the cache families.
	denyAllMu               sync.Mutex
	denyAllStops            uint64 // cumulative deny-all turns this session
	policyCanaryRollbacks   atomic.Uint64
	denyAllConsecutive      uint64 // consecutive deny-all turns ending the most recent served turn (reset by any non-deny-all turn)
	denyAllSameConsecutive  uint64 // consecutive deny-all turns proposing the IDENTICAL refused action (same tool+reason); re-seeded to 1 whenever the fingerprint changes, reset to 0 by any non-deny-all turn. The same-issue signal the guard Stop hook keys its give-up on — a varied session pins this at 1 and is never stopped.
	denyAllFingerprint      string // the fingerprint (denyAllFingerprint) of the most recent deny-all turn; the equality test that advances vs re-seeds denyAllSameConsecutive
	toolFeedbackTurns       uint64 // cumulative retryable tool-feedback turns (all proposed calls rejected as model-fixable feedback)
	toolFeedbackConsecutive uint64 // consecutive retryable tool-feedback turns ending the most recent served turn

	// fakVerbCalls counts admitted MCP fak-verb (tools/call) invocations since process
	// start — the signal that fak was actually USED as a substrate, not merely present as a
	// passive guard. The guard-stophook polls it to warn when a long run ends having called
	// ZERO fak verbs (the #3093 unused-substrate pathology). Atomic, off the deny-all lock —
	// a one-increment-per-tools/call path with no coupling to the stop families.
	fakVerbCalls uint64
}

type inflightEntry struct {
	route string
	start time.Time
}

type httpMetricKey struct {
	route  string
	method string
	status string
}

type operationMetricKey struct {
	operation   string
	verdict     string
	reason      string
	disposition string
	by          string // which adjudicator decided (forensics) — answers WHO refused, not just that it was refused
}

type requestMemoryMetricKey struct {
	backend string
	class   string
	scope   string
	dtype   string
}

type requestMemoryTokenKey struct {
	backend string
	kind    string
}

type requestMemoryFitKey struct {
	backend string
	scope   string
}

type inKernelOOMClassStats struct {
	count           uint64
	failedBytes     uint64
	lastFailedBytes uint64
	lastSite        string
}

type requestMemoryMetricStats struct {
	observations   uint64
	totalBytes     uint64
	highWaterBytes int64
}

type requestMemoryTokenStats struct {
	observations uint64
	total        uint64
	highWater    int
}

type requestMemoryFitStats struct {
	observations   uint64
	wantHighWater  int64
	marginLowWater int64
	marginKnown    bool
}

type latencyCounter struct {
	count   uint64
	sum     float64
	buckets []uint64
}

func newGatewayMetrics(now time.Time) *gatewayMetrics {
	return &gatewayMetrics{
		start:                    now,
		http:                     map[httpMetricKey]*latencyCounter{},
		operations:               map[operationMetricKey]*latencyCounter{},
		inflightReq:              map[uint64]inflightEntry{},
		compactAttempts:          map[string]uint64{},
		compactBailReasons:       map[string]uint64{},
		ttlUpgrades:              map[string]uint64{},
		placementAttempts:        map[string]uint64{},
		anchorMon:                metrics.NewAnchorRefusalMonitor(metrics.AnchorRefusalThresholds{}),
		reqMemoryObserved:        map[string]uint64{},
		reqMemoryPlan:            map[requestMemoryMetricKey]*requestMemoryMetricStats{},
		reqMemoryTokens:          map[requestMemoryTokenKey]*requestMemoryTokenStats{},
		reqMemoryFit:             map[requestMemoryFitKey]*requestMemoryFitStats{},
		inKernelOOM:              map[string]*inKernelOOMClassStats{},
		upstreamErrors:           map[string]uint64{},
		upstreamAuthRefreshes:    map[string]uint64{},
		upstreamForbiddenRetries: map[string]uint64{},
		upstreamAccountFailovers: map[string]uint64{},
		harnessCoherence:         newHarnessCoherenceMetrics(compactcohere.DefaultProviderCacheTTL),
		routing:                  newRoutingMetrics(),
		vcacheGovernor:           newVCacheGovernorDecisionJournal(),
		vcacheWarmth:             newVCacheWarmthDemotionJournal(),
		inferTTFTHist:            newLatencyCounter(),
		inferTPOTHist:            newLatencyCounter(),
		inferE2EHist:             newLatencyCounter(),
	}
}

// upstreamErrorKind classifies a planner/proxy error into the coarse KIND label the
// upstream-error counter and the `fak-turn … FAILED` debug line both use. The ladder mirrors
// upstreamErrorStatus's error.As order so the metric and the client-facing status never
// disagree about what KIND of failure a turn hit. A nil error returns "" (not counted).
func upstreamErrorKind(err error) string {
	if err == nil {
		return ""
	}
	var stalled *agent.UpstreamStalledError
	if errors.As(err, &stalled) {
		return "stalled"
	}
	var oom *agent.InKernelOOMError
	var capErr *agent.InKernelCapacityError
	if errors.As(err, &oom) || errors.As(err, &capErr) {
		return "oom"
	}
	var ue *agent.UpstreamUnreachableError
	if errors.As(err, &ue) {
		return "unreachable"
	}
	var se *agent.UpstreamStatusError
	if errors.As(err, &se) {
		// Split the operationally-distinct 4xx conditions into their own kinds so a
		// /metrics scrape (and the FAILED debug line) can tell a RATE-LIMIT storm apart
		// from an AUTH-failure storm apart from a LOGIN/permission denial — the same
		// distinction upstreamErrorStatus now draws for the client. This stays in lockstep
		// with that ladder (the cross-ladder test pins the pairing): 429 -> rate_limited,
		// 401 -> auth, 403 -> forbidden, every other 4xx -> the coarse status_4xx bucket.
		switch se.Status {
		case http.StatusTooManyRequests:
			return "rate_limited"
		case http.StatusUnauthorized:
			return "auth"
		case http.StatusForbidden:
			return "forbidden"
		case 529: // Anthropic "Overloaded" — a PROVIDER-capacity storm, distinct on /metrics
			// from a generic 5xx so an operator can tell upstream-over-capacity apart from
			// upstream-crashing. Stays in lockstep with upstreamErrorStatus's overloaded arm.
			return "overloaded"
		}
		if se.Status >= 400 && se.Status < 500 {
			return "status_4xx"
		}
		return "status_5xx"
	}
	// A TRANSIENT transport failure (mid-flight reset, truncated read, I/O timeout) that
	// exhausted the planner's in-handler retry and surfaced to the wrapped agent. Split from
	// the coarse "other" bucket so `fak guard`'s supervisor has a specific signal for a wire
	// crash worth one bounded relaunch (#3514), distinct from the deterministic "unreachable"
	// already handled above. Checked last so a typed status/stall/oom error always wins.
	if transientTransportError(err) {
		return "transport"
	}
	return "other"
}

func inKernelOOMObservation(err error) (class string, bytes uint64, site string, ok bool) {
	var oom *agent.InKernelOOMError
	if errors.As(err, &oom) {
		if oom.Bytes > 0 {
			bytes = uint64(oom.Bytes)
		}
		return oomClassLabel(string(oom.Class)), bytes, strings.TrimSpace(oom.Site), true
	}
	var capErr *agent.InKernelCapacityError
	if errors.As(err, &capErr) {
		if capErr.Want > 0 {
			bytes = uint64(capErr.Want)
		}
		return oomClassLabel(string(capErr.Class)), bytes, strings.TrimSpace(capErr.Site), true
	}
	return "", 0, "", false
}

func oomClassLabel(class string) string {
	class = strings.TrimSpace(class)
	if class == "" {
		return "unknown"
	}
	return class
}

// cacheTTLUpgradePlacedAndUpgraded is the composed-transform outcome (#2175): the caller sent
// zero cache_control, so maybeUpgradeAnthropicCacheTTL1H placed a stable-head breakpoint AND
// upgraded it to the 1h tier in one turn. Distinct from "upgraded" (an existing breakpoint was
// upgraded) and placement's own "placed" row (compaction-path placement, no TTL upgrade), so a
// sweep can attribute placement-only vs upgrade-only vs composed.
const cacheTTLUpgradePlacedAndUpgraded = "placed_and_upgraded"

func estimatedTokensFromBytes(n int) uint64 {
	if n <= 0 {
		return 0
	}
	return uint64(n / 4)
}

type adjudicationOutcomeSignal int

const (
	adjudicationOutcomeReset adjudicationOutcomeSignal = iota
	adjudicationOutcomeDenyAll
	adjudicationOutcomeToolFeedback
)

// sessionRunStates is the closed vocabulary of session DRIVE-state tokens the
// fak_sessions{state} gauge carries. These are the wire forms of SessionState.Run
// (lowercase), so the scrape surface stays consistent with GET /v1/fak/sessions and
// /debug/vars. Editing this set is a metrics-series change.
var sessionRunStates = []string{"running", "throttled", "paused", "draining", "stopped"}

var sessionEnvelopeRefusalReasonOrder = []string{
	"TIME_BUDGET_EXHAUSTED",
	"BUDGET_SPEND_EXHAUSTED",
	"THROUGHPUT_BELOW_FLOOR",
}

var sessionEnvelopeRefusalReasons = map[string]bool{
	"TIME_BUDGET_EXHAUSTED":  true,
	"BUDGET_SPEND_EXHAUSTED": true,
	"THROUGHPUT_BELOW_FLOOR": true,
}

// AdjudicationSummary is a verdict roll-up over every kernel decision a gateway has
// made — the tally `fak guard` prints when the wrapped agent exits, so an operator
// sees what the kernel allowed vs blocked without scraping /metrics. It folds the
// per-(operation, verdict, reason) operation counters into one record; every count is
// the SAME number the fak_gateway_operations_total scrape would report, so the exit
// line can never disagree with the metrics.
type AdjudicationSummary struct {
	Total       uint64 `json:"total"`
	Allowed     uint64 `json:"allowed"`
	Denied      uint64 `json:"denied"`
	Transformed uint64 `json:"transformed"`
	Quarantined uint64 `json:"quarantined"`
	// Deferred counts DEFER verdicts: a non-blocking admit (e.g. an inbound tool
	// result the kernel let through while raising the session's taint watermark).
	// It is NOT an error — the old default-bucket fold reported it under Errored,
	// which made a perfectly healthy proxy_admit read as a failure in the exit
	// summary `fak guard` prints (a tool-bearing turn always admits its result).
	Deferred uint64 `json:"deferred"`
	// Escalated counts REQUIRE_WITNESS verdicts: a call HELD pending a witness /
	// human approval rather than allowed or denied outright. Also not an error.
	Escalated uint64 `json:"escalated"`
	// Errored counts genuine ERROR verdicts (and any unknown future kind) — a real
	// adjudication failure, never silently dropped.
	Errored uint64 `json:"errored"`
	// ByReason maps a deny/quarantine reason code to its count (the forensic "why").
	ByReason map[string]uint64 `json:"by_reason,omitempty"`
	// CachedPromptTokens is the cumulative count of prompt (input) tokens the upstream
	// PROVIDER served from its OWN prompt cache (cache_read) across this session's
	// turns — the provider-side reuse `fak guard` preserves byte-for-byte by forwarding
	// the client's cache_control prefix unchanged through the kernel hop. CachedTurns
	// counts the turns that got such a hit. Surfaced in the guard exit summary so the
	// operator SEES the cache reuse rather than having to scrape /metrics.
	CachedPromptTokens uint64 `json:"cached_prompt_tokens"`
	CachedTurns        uint64 `json:"cached_turns"`
	// InputTokens (the uncached input remainder), OutputTokens, and CacheCreationTokens
	// (the cache WRITE axis) are retained alongside CachedPromptTokens (the READ axis) so
	// the summary can price the NET realized provider-cache saving — read rebate MINUS
	// write premium — via ProviderCacheNetSavings and the Track-2 savings ledger. All are
	// OBSERVED (provider-relayed).
	InputTokens         uint64 `json:"input_tokens"`
	OutputTokens        uint64 `json:"output_tokens"`
	CacheCreationTokens uint64 `json:"cache_creation_tokens"`
	// SelfHosted*/Vendor* split InputTokens and OutputTokens by WHO SERVED the turn
	// (epic #5416): tokens generated in-kernel on hardware we host, versus tokens
	// bought from a third-party API. Both are strict subsets of the unsplit totals —
	// a turn whose side servedLocality could not resolve is counted in the totals and
	// in neither group — so the shortfall between them is the unclassified volume,
	// and a reader can tell a measured self-hosted zero (Vendor turns present, no
	// local ones) from an unmeasured one (no classified turn at all). All omitempty:
	// a summary from a build or deployment that classified nothing must not serialize
	// six zeros that read as "we self-host nothing".
	SelfHostedTurns        uint64 `json:"self_hosted_turns,omitempty"`
	SelfHostedInputTokens  uint64 `json:"self_hosted_input_tokens,omitempty"`
	SelfHostedOutputTokens uint64 `json:"self_hosted_output_tokens,omitempty"`
	VendorTurns            uint64 `json:"vendor_turns,omitempty"`
	VendorInputTokens      uint64 `json:"vendor_input_tokens,omitempty"`
	VendorOutputTokens     uint64 `json:"vendor_output_tokens,omitempty"`
	// CacheCreationTokensUpgraded is the subset of CacheCreationTokens attributed to the
	// managed-cache 1h TTL tier (GATEWAY_ATTRIBUTED — fak's own per-turn upgrade witness,
	// since the provider wire never splits 5m vs 1h creation tokens; see
	// inferCacheCreationTokensUpgraded). 0 means either the lever was off or every write
	// stayed on the 5m tier; MechanismSavings/ProviderCacheNetSavings price the remainder
	// (CacheCreationTokens - CacheCreationTokensUpgraded) at the 5m tier as before (#2179).
	CacheCreationTokensUpgraded      uint64 `json:"cache_creation_tokens_upgraded,omitempty"`
	CacheCreationTokensTierObserved  uint64 `json:"cache_creation_tokens_tier_observed,omitempty"`
	CacheCreationTokensHeadOnly      uint64 `json:"cache_creation_tokens_head_only,omitempty"`
	CacheCreationTokensMessagePrefix uint64 `json:"cache_creation_tokens_message_prefix,omitempty"`

	// Compaction* folds the Anthropic history-compaction visibility into the same guard exit
	// summary, split WITNESSED (what fak authored) vs OBSERVED (what the provider reported):
	// Fired/Bailed/Off and DroppedTurns/ShedTokens are WITNESSED — what fak attempted and what
	// it removed (a turn only counts Fired when the prefix it shipped was byte-identical).
	// CompactionCacheReadTokens / LastCompactionCacheRead are OBSERVED — the provider's
	// cache_read_input_tokens, relayed verbatim. They are NOT proof fak preserved the cache (the
	// byte-identity is); a low value with no prefix_mismatch is a provider-side miss fak does not
	// control (TTL/eviction/client breakpoint move).
	CompactionFired           uint64  `json:"compaction_fired"`
	CompactionBailed          uint64  `json:"compaction_bailed"`
	CompactionOff             uint64  `json:"compaction_off"`
	CompactionDroppedTurns    uint64  `json:"compaction_dropped_turns"`
	CompactionShedTokens      uint64  `json:"compaction_shed_tokens"`
	CompactionCacheReadTokens uint64  `json:"compaction_cache_read_tokens"`
	LastCompactionCacheRead   float64 `json:"last_compaction_cache_read"`
	// CompactionBailReasons is the WITNESSED per-reason breakdown of the CompactionBailed
	// lump (CompactReason* -> count). Without it, "N bailed" is uninterpretable: a session
	// that bailed N times to under_budget (the compactible suffix already fit — benign,
	// working-as-designed) is indistinguishable from one that bailed to prefix_mismatch (the
	// only fak-fault cache signal, must stay 0) or no_breakpoint (no anchor — can't fire).
	// Already surfaced on /metrics + /debug/vars; folded here so the guard exit summary can
	// render it the same way ByReason renders deny reasons.
	CompactionBailReasons map[string]uint64 `json:"compaction_bail_reasons,omitempty"`
	// CompactionAnchorStarved is the count of under_budget bails whose protected prefix ALREADY
	// exceeded the budget — the cache_control anchor swallowed the conversation, so compaction
	// structurally cannot fire no matter how long the session grows. It is a SUBSET of
	// CompactionBailReasons["under_budget"], broken out because the two are operationally opposite:
	// a plain under_budget is a benign short session (nothing to shed), an anchor-starved one is the
	// dormant-on-real-Claude-Code-traffic pathology (#1407) that no budget tightening can fix.
	CompactionAnchorStarved uint64 `json:"compaction_anchor_starved"`
	// CompactionSolvencyForced is the count of fires the head-anchored burst economics REFUSED
	// and the context-solvency floor forced anyway (Config.CompactSolvencyFloorTokens). It is a
	// SUBSET of CompactionFired, broken out because the two are economically opposite: an
	// ordinary fire is a burst that repays in cache dollars, a forced one is a burst knowingly
	// taken at a loss to keep the session inside its context window. A nonzero count is not a
	// fault — it is the override doing its job — but the cache-value ledger must not book these
	// as savings, and a count that dominates CompactionFired means the window, not the cache, is
	// the binding constraint on this workload.
	CompactionSolvencyForced uint64 `json:"compaction_solvency_forced"`
	// CompactionBudget is the resident-token threshold the history rewrite fires past
	// (Config.CompactHistoryBudget; 0 means the lever is OFF, body forwarded byte-for-byte).
	// Surfaced so the exit line can say whether compaction is ENABLED and merely idle
	// (budget>0, nothing exceeded it) vs DISABLED — the two readings of "0 fired" the bare
	// counters cannot tell apart.
	CompactionBudget int `json:"compaction_budget"`

	// KVPrefix* folds the local in-kernel RadixAttention prefix-cache reuse tap into
	// the same attribution frame as provider prompt-cache and compaction. These are
	// WITNESSED fak-authored savings: prompt prefill work the in-kernel planner did
	// not redo. They are distinct from CachedPromptTokens, which are OBSERVED
	// provider cache_read tokens.
	KVPrefixPromptTokens uint64 `json:"kv_prefix_prompt_tokens"`
	KVPrefixReusedTokens uint64 `json:"kv_prefix_reused_tokens"`
	// KVPrefixColdCliff is the LIVE frozen-trajectory cache-cliff finding (#3623): present
	// (PREFIX_COLD_CLIFF) only when the in-kernel KV-prefix reuse for this process has
	// collapsed — a cold-dominated turn distribution or an aggregate reuse ratio below the
	// floor. Absent (omitempty) on a healthy frozen/partial session, so its mere PRESENCE on
	// /debug/vars is the alarm. WITNESSED (fak's own kernel reuse), like the KVPrefix* counts.
	KVPrefixColdCliff *cacheobs.ColdCliffVerdict `json:"kv_prefix_cold_cliff,omitempty"`

	// ToolPrune* folds the INBOUND tool-definition prune lever into the same exit summary,
	// WITNESSED (fak authored): ToolPruneTurns is the count of turns that dropped at least one
	// unreachable tool def from the outbound tools[], and ToolPruneCount the total defs removed.
	// The pruner only drops tools strictly AFTER the cache_control breakpoint and re-proves the
	// protected prefix is byte-identical, so a counted prune is a pure uncached-token saving that
	// never bursts the upstream cache. Zero on the dominant Claude Code path (its single breakpoint
	// sits on the LAST tool, so nothing is droppable) — which is exactly the fact the operator
	// could not see before, since the prune result was discarded with no metric.
	ToolPruneTurns uint64 `json:"tool_prune_turns"`
	ToolPruneCount uint64 `json:"tool_prune_count"`

	// DeferCold* folds the OUTBOUND cold-tool DEFERRAL lever (--defer-cold-tools, #3232, the 10x
	// floor lever under epic #3229) into the same exit summary, WITNESSED (fak authored the splice):
	// DeferColdTurns is the count of turns on which fak deferred the cold tool tail (marked the
	// allowed-but-cold custom tools defer_loading:true and injected a tool_search_tool), and
	// DeferColdCount the total cold defs deferred across them. Unlike the prune lever this does NOT
	// shrink the request bytes — the reduction is provider-side (the provider loads only the hot
	// core into context, faulting a cold schema in on demand), so it lands in the OBSERVED usage
	// relay (input_tokens / cache_read), never in the ESTIMATED byte footprint. The transform is
	// deterministic + cache-safe, so a counted deferral never bursts the upstream prompt cache.
	// Zero when the lever is off (its DEFAULT) or when a turn had no cold tools to defer.
	DeferColdTurns uint64 `json:"defer_cold_turns"`
	DeferColdCount uint64 `json:"defer_cold_count"`
	// DeferColdToolNames is the DISTINCT set of custom tool names fak deferred across those
	// turns (#3647) — surfaced so the operator pane can name WHICH tools were made cold, not
	// just how many. Sorted for a stable render; empty/absent when the lever never fired.
	DeferColdToolNames []string `json:"defer_cold_tool_names,omitempty"`
	// DeferStandDownTurns / DeferStandDownReasons are the missing DENOMINATOR of the DeferCold*
	// numerator above (#3621): turns on which the lever was armed and the transform ran, yet it
	// stood down to identity, plus the per-reason breakdown (no_cold_tools, already_deferred,
	// decode_failed, …). They are what makes "the lever is off" distinguishable from "the lever
	// is on and never bit" — the silent-identity failure mode the DEFER_ENABLED_BUT_INERT
	// watchdog (defer_inert.go) keys on. Absent (omitempty) when nothing ever stood down.
	DeferStandDownTurns   uint64            `json:"defer_stand_down_turns,omitempty"`
	DeferStandDownReasons map[string]uint64 `json:"defer_stand_down_reasons,omitempty"`

	// ToolRefTurns / ToolRefConverted witness the tool_reference SANITIZE correctness transform:
	// the client's INTERNAL tool_reference blocks (emitted inside a ToolSearch tool_result) are not
	// a valid Anthropic tool_result.content type, so fak rewrites each into a wire-valid text block
	// before relay to keep the body from being 400'd upstream. ToolRefTurns counts turns with >=1
	// conversion; ToolRefConverted the total blocks converted. Zero on a harness that never surfaces
	// tool-discovery results back into a tool_result — nonzero means fak repaired a body the provider
	// would otherwise have rejected as malformed.
	ToolRefTurns     uint64 `json:"tool_ref_turns"`
	ToolRefConverted uint64 `json:"tool_ref_converted"`

	// DenyAllStops is how many served turns this session had EVERY proposed tool call
	// refused by the floor, which the wire reports as end_turn — a STOP the agent did not
	// choose (it wanted to act, the floor blocked all of it). Surfaced in the guard exit
	// summary so the operator SEES how often fak ended a turn, the otherwise-invisible
	// false-stop the --deny-all-continue Stop-hook auto-resumes the agent past.
	DenyAllStops uint64 `json:"deny_all_stops"`
	// ToolFeedbackTurns is the sibling non-terminal count: every proposed tool call was
	// rejected as retryable/model-fixable feedback, so the guard may continue the turn,
	// but this is NOT a session-stop/give-up policy input.
	ToolFeedbackTurns uint64 `json:"tool_feedback_turns"`

	// CacheTTLUpgrade* folds the managed-cache 1h TTL-upgrade lever (--managed-cache,
	// epic #1844 C6) into the same exit frame, WITNESSED (fak authored the cache_control
	// splice, and the upgrader re-proves the body redecodes before returning changed
	// bytes): CacheTTLUpgraded counts turns whose stable-head breakpoint fak actually
	// extended to the 1h tier — BOTH authoring outcomes, an existing breakpoint upgraded
	// ("upgraded") and a breakpoint placed-and-upgraded in one composed turn (#2175);
	// CacheTTLUpgradeReasons is the per-refusal breakdown over the closed
	// agent.TTLUpgradeReason* vocabulary (an authoring outcome is never a refusal, so it
	// never appears here). The family is recorded only while
	// the lever is on, so zero/absent means OFF, while a zero upgraded count WITH reason
	// rows means ON-but-ineligible (every head refused) — visible, not silent.
	CacheTTLUpgraded       uint64            `json:"cache_ttl_upgrades_upgraded"`
	CacheTTLUpgradeReasons map[string]uint64 `json:"cache_ttl_upgrade_reasons,omitempty"`

	// E2ELatency* fold the session's OBSERVED end-to-end per-turn latency distribution
	// (inferE2EHist, fed by observeInferenceTimed for every served turn) into the summary
	// so the guard exit line can price WITNESSED turns-saved into wall-clock at the
	// session's OWN measured per-turn cost — never a fabricated tokens/sec constant.
	// Sum is the total observed seconds across Count timed turns; mean = Sum/Count (see
	// MeanTurnLatencySeconds). Both are OBSERVED (provider round-trip timing), and 0 when
	// no turn was timed (a replayed/offline summary, or a session that served nothing) —
	// in which case the exit line prints turns only and omits the wall-clock dual.
	E2ELatencySumSeconds float64 `json:"e2e_latency_sum_seconds,omitempty"`
	E2ELatencyCount      uint64  `json:"e2e_latency_count,omitempty"`

	// AnchorPlacement is the session's breakpoint-placement outcome MIX and the rolling
	// volatile-head verdict over it (#3622) — the live-loop half of the placement counters the
	// Prometheus surface already renders. The counters are cumulative and unwatched: a session
	// whose head turns volatile mid-conversation simply stops incrementing "placed", which reads
	// identically to a session that went quiet. This carries the fraction, the dominant refusal
	// outcome, and how many times ANCHOR_REFUSED_RISING was raised, so the guard exit banner can
	// say the anchor stopped earning caching instead of leaving the operator to diff two counters.
	// A pointer with omitempty: nil — and the JSON field absent — on any session that never
	// attempted a placement, so a non-Anthropic or cold session's summary is byte-identical to
	// before this field existed.
	AnchorPlacement *metrics.AnchorRefusalReport `json:"anchor_placement,omitempty"`
}

// adjudicationSummary folds the live operation counters into a verdict roll-up.
func (m *gatewayMetrics) adjudicationSummary() AdjudicationSummary {
	sum := AdjudicationSummary{ByReason: map[string]uint64{}}
	if m == nil {
		return sum
	}
	// Provider prompt-cache reuse rides the inference counters (a separate lock from the
	// operation ledger); read it first so the two critical sections never nest.
	m.inferenceMu.Lock()
	sum.CachedPromptTokens = m.inferCachedTokens
	sum.CachedTurns = m.inferCachedHits
	sum.InputTokens = m.inferPromptTokens
	sum.OutputTokens = m.inferComplTokens
	sum.SelfHostedTurns = m.inferSelfHostedTurns
	sum.SelfHostedInputTokens = m.inferSelfHostedPromptTokens
	sum.SelfHostedOutputTokens = m.inferSelfHostedComplTokens
	sum.VendorTurns = m.inferVendorTurns
	sum.VendorInputTokens = m.inferVendorPromptTokens
	sum.VendorOutputTokens = m.inferVendorComplTokens
	sum.CacheCreationTokens = m.inferCacheCreationTokens
	sum.CacheCreationTokensUpgraded = m.inferCacheCreationTokensUpgraded
	sum.CacheCreationTokensTierObserved = m.inferCacheCreationTokensTierObserved
	sum.CacheCreationTokensHeadOnly = m.inferCacheCreationTokensHeadOnly
	sum.CacheCreationTokensMessagePrefix = m.inferCacheCreationTokensMessagePrefix
	// Observed per-turn E2E latency distribution (same lock observeInferenceTimed holds
	// when it writes inferE2EHist), so the exit line can price turns-saved into wall-clock
	// at the session's own measured per-turn cost.
	if m.inferE2EHist != nil {
		sum.E2ELatencySumSeconds = m.inferE2EHist.sum
		sum.E2ELatencyCount = m.inferE2EHist.count
	}
	m.inferenceMu.Unlock()

	comp := m.compactionSnapshotData()
	sum.CompactionFired = comp.attempts["fired"]
	sum.CompactionBailed = comp.attempts["bailed"]
	sum.CompactionOff = comp.attempts["off"]
	sum.CompactionDroppedTurns = comp.dropped
	sum.CompactionShedTokens = comp.shed
	sum.CompactionCacheReadTokens = comp.cacheReads
	sum.LastCompactionCacheRead = comp.lastCacheRd
	sum.CompactionAnchorStarved = comp.anchorStarved
	sum.CompactionSolvencyForced = comp.solvencyForced
	// Carry the per-reason bail breakdown so the banner can explain the bailed lump (the
	// snapshot already copied it under compactMu); only attach a non-empty map so a clean
	// session keeps the JSON field absent (omitempty).
	sum.ToolPruneTurns, sum.ToolPruneCount = m.inboundToolPruneSnapshot()
	sum.DeferColdTurns, sum.DeferColdCount = m.toolDeferSnapshot()
	sum.DeferColdToolNames = m.toolDeferNamesSnapshot()
	sum.DeferStandDownTurns, sum.DeferStandDownReasons = m.toolDeferStandDownSnapshot()
	sum.AnchorPlacement = m.anchorRefusalReport()
	sum.ToolRefTurns, sum.ToolRefConverted = m.toolRefSanitizeSnapshot()
	sum.DenyAllStops, _ = m.denyAllSnapshot()
	sum.ToolFeedbackTurns, _ = m.toolFeedbackSnapshot()
	// COMPACTION_THRASH (#2424) rides this same map. It is NOT part of the CompactionBailed
	// lump — the lever FIRED on every thrashing turn — but an operator reading "why is
	// compaction not holding this session" needs it beside the bail reasons, and a verdict
	// published nowhere is the gap the issue reports. Kept out of comp.bailReasons (and so
	// out of bail_reason_total's closed set and compactBailPartition's alertable rate) by
	// folding into a fresh map here; its own /metrics counter is
	// fak_gateway_compaction_thrash_sessions_total.
	if len(comp.bailReasons) > 0 || comp.thrashSessions > 0 {
		reasons := make(map[string]uint64, len(comp.bailReasons)+1)
		for k, v := range comp.bailReasons {
			reasons[k] = v
		}
		if comp.thrashSessions > 0 {
			reasons[ReasonCompactionThrash] = comp.thrashSessions
		}
		sum.CompactionBailReasons = reasons
	}
	// Managed-cache TTL-upgrade outcomes (#1844 C6): split the snapshot's one outcome
	// map into the AUTHORED count and the refusal-reason breakdown, attaching the map only
	// when non-empty so a lever-off session keeps the JSON field absent (omitempty). BOTH
	// authoring outcomes count as authored — "upgraded" (an existing stable-head breakpoint
	// extended to 1h) AND "placed_and_upgraded" (the #2175 composed transform that placed a
	// breakpoint AND upgraded it in one turn) — matching the managed-cache self-check's
	// Authored = Upgraded + PlacedAndUpgraded contract (internal/cachewitness). A composed
	// fire is NOT a refusal, so it must never land in the reason map: booking it there
	// undercounted CacheTTLUpgraded AND double-counted it as a bail, sinking the fired-rate
	// the cache-health digest folds (cachevaluereport TTLRefusals → UpgradeFiredHealth).
	sum.CacheTTLUpgraded = comp.ttlUpgrades["upgraded"] + comp.ttlUpgrades[cacheTTLUpgradePlacedAndUpgraded]
	ttlReasons := map[string]uint64{}
	for reason, n := range comp.ttlUpgrades {
		if reason != "upgraded" && reason != cacheTTLUpgradePlacedAndUpgraded {
			ttlReasons[reason] = n
		}
	}
	if len(ttlReasons) > 0 {
		sum.CacheTTLUpgradeReasons = ttlReasons
	}
	kv := cacheobs.Default.Snapshot()
	sum.KVPrefixPromptTokens = kv.PromptTokens
	sum.KVPrefixReusedTokens = kv.ReusedTokens
	// Raise the live frozen-trajectory cache-cliff finding (#3623) only when this
	// process's realized KV-prefix reuse has collapsed; a healthy session leaves the
	// field nil, so its presence on the exit summary / /debug/vars is itself the alarm.
	if cliff := kv.ColdCliff(); cliff.Fired {
		sum.KVPrefixColdCliff = &cliff
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	for key, c := range m.operations {
		n := c.count
		if n == 0 {
			continue
		}
		sum.Total += n
		switch key.verdict {
		case "ALLOW":
			sum.Allowed += n
		case "TRANSFORM":
			sum.Transformed += n
		case "DENY":
			sum.Denied += n
			if key.reason != "" {
				sum.ByReason[key.reason] += n
			}
		case "QUARANTINE":
			sum.Quarantined += n
			if key.reason != "" {
				sum.ByReason[key.reason] += n
			}
		case "DEFER":
			// A non-blocking admit (the inbound result was let through, the call
			// was not refused). Distinct from ALLOW only in that the kernel held a
			// firm opinion in reserve; reporting it as "errored" alarmed the
			// operator over a perfectly healthy decision.
			sum.Deferred += n
		case "REQUIRE_WITNESS":
			// Held pending a witness / approval — an escalation, not an error.
			sum.Escalated += n
		default: // a genuine "ERROR", or an unknown future kind: counted, never silently dropped.
			sum.Errored += n
		}
	}
	return sum
}

// providerCacheEvidence classifies the gateway's recorded provider prompt-cache
// reuse through the cachemeta materialization bridge (issue #432, acceptance #3). A
// provider cache_read is a `provider_prefix` materialization: COST/LATENCY telemetry
// about a prefix the REMOTE engine kept resident, never a re-serveable LOCAL-trust
// artifact. Routing the live telemetry through the same proven gate the kernel uses
// (MaterializeVerdict(MatProviderPrefix, …)) makes the separation mechanical on the
// live path — the verdict is structurally non-serveable (CanServe()==false) and
// marked cost_latency_only — rather than a prose promise in a metric's HELP text.
// cachedTok is the cumulative cache_read tokens observed across served turns.
func providerCacheEvidence(cachedTok uint64) cachemeta.LookupVerdict {
	entry := cachemeta.FromProviderCache(cachemeta.ProviderCache{CachedTokens: int64(cachedTok)})
	return cachemeta.MaterializeVerdict(
		cachemeta.MatProviderPrefix, entry, cachemeta.MaterializationKey{}, cachemeta.QualityEvidence{})
}

// ProviderCacheEvidence classifies the summary's provider prompt-cache reuse
// (CachedPromptTokens) through the #432 bridge: provider cache is PERFORMANCE
// evidence (cost/latency), never local TRUST. The returned verdict is structurally
// non-serveable (CanServe()==false, Meta["provider_cache"]=="cost_latency_only"), so
// a consumer that prints the cached-token saving (the `fak guard` exit summary) can
// prove from the kernel's own gate that it is reporting performance, not trust — the
// cache reuse can never be promoted to authority that a local result may be re-served.
func (s AdjudicationSummary) ProviderCacheEvidence() cachemeta.LookupVerdict {
	return providerCacheEvidence(s.CachedPromptTokens)
}

// defaultBackendLabel trims a reported backend name and substitutes "unknown" for
// an empty one, so every metric/debug label that keys on the backend carries a
// stable, non-empty value. Centralizes the trim-or-unknown idiom the request/OOM/
// pressure-trim reporters each repeated verbatim.
func defaultBackendLabel(backend string) string {
	backend = strings.TrimSpace(backend)
	if backend == "" {
		return "unknown"
	}
	return backend
}

// addPositiveSignedToUint64 saturating-adds a signed counter delta onto a uint64
// running total: a non-positive delta is a no-op, and an add that would overflow
// clamps at the uint64 max instead of wrapping. Generic over the signed input so
// both the int64 byte counters and the int token counters share one body.
func addPositiveSignedToUint64[T ~int | ~int64](total uint64, value T) uint64 {
	if value <= 0 {
		return total
	}
	v := uint64(value)
	if ^uint64(0)-total < v {
		return ^uint64(0)
	}
	return total + v
}

func addPositiveInt64ToUint64(total uint64, value int64) uint64 {
	return addPositiveSignedToUint64(total, value)
}

func addPositiveIntToUint64(total uint64, value int) uint64 {
	return addPositiveSignedToUint64(total, value)
}

func (s *Server) logInferenceTurn(traceID, wire string, stream bool, usage agent.Usage, finishReason string, dur time.Duration, compacted bool) {
	s.logInferenceTurnWithContextEvent(traceID, wire, stream, usage, finishReason, dur, compacted, compacted || s.consumeDecodedCtxViewEvent(traceID))
}

func (s *Server) logInferenceTurnWithContextEvent(traceID, wire string, stream bool, usage agent.Usage, finishReason string, dur time.Duration, compacted, contextEvent bool) {
	if s == nil {
		return
	}
	// #2179: attribute this turn's cache-creation write to the 1h tier when the
	// managed-cache TTL-upgrade rung was already active for the session. GATEWAY-
	// ATTRIBUTED, not provider-reported — the Anthropic usage block never splits 5m
	// vs 1h creation tokens, so this is fak's own per-turn upgrade witness
	// (noteCtxValueTTL1h), read BEFORE this turn's write is folded into the total.
	s.metrics.recordCacheCreationTierSplit(usage.CacheCreationInputTokens, s.ttl1hActiveFor(traceID), s.ttl1hMessagePrefixFor(traceID))
	// Record this turn into the per-family live-observe window (#935) BEFORE the sink
	// gates below, so the per-family / governor / warmth view is populated even with
	// --log off and --debug-stats off. The family is the session/trace prefix; the token
	// axes are the provider's own counters (OBSERVED). This is purely observational — it
	// never feeds the request path (Law A2).
	//
	// The cache-read axis is the PROVIDER-NORMALIZED hit (CachedPromptTokens), not the
	// Anthropic-only cache_read_input_tokens field: OpenAI (chat + the codex Responses
	// wire) and Gemini report their prompt-cache hit in prompt_tokens_details, leaving
	// cache_read_input_tokens at 0, so reading the raw field made every codex/OpenAI cache
	// hit register as read=0 — the families/governor/warmth plane saw those sessions as
	// permanently COLD even when the upstream served a real cache. Pairing the normalized
	// read with the uncached remainder (UncachedPromptTokens — OpenAI/Gemini fold the hit
	// INTO prompt_tokens, so it is peeled back off) keeps input+read == the full resident
	// prompt on every provider, so a codex family's hit-rate/economics read identically to
	// a Claude family's rather than being understated by the double-counted cached span.
	cacheRead := usage.CachedPromptTokens()
	uncachedPrompt := usage.UncachedPromptTokens()
	// One timestamp shared by the native vcache row and the per-request usage
	// record (#10670), so the two join exactly instead of by a fragile
	// nearest-millisecond match.
	nowMillis := time.Now().UnixMilli()
	s.metrics.observeVCacheTurn(traceID, nowMillis,
		uncachedPrompt, cacheRead, usage.CacheCreationInputTokens)
	// The stable per-request usage record (#10670): exactly one row per
	// completed request, after the native plane above so its join receipts exist.
	usageRec, usageRecOK := s.metrics.recordUsageTurn(traceID, wire, stream, usage, nowMillis)
	// Roll the per-session managed-context record (ctxvalue.go) on the same always-on
	// rung: every served turn, all wires, before any sink gating, so the long-session
	// context report is answerable even with --log and --debug-stats off.
	s.observeCtxValue(traceID, uncachedPrompt, cacheRead, usage.CacheCreationInputTokens,
		usage.CompletionTokens, contextEvent)
	// The per-turn human debug render (#793) fires independently of the JSON --log sink, so
	// --debug-stats works on a clean (--log off) terminal. It is a no-op unless debugStatsf is
	// wired, and reuses the #792 rolling health (read-only peek; no double-roll).
	if s.debugStatsf != nil {
		s.renderTurnDebugStats(traceID, wire, stream, finishReason,
			uncachedPrompt, usage.CompletionTokens, cacheRead, usage.CacheCreationInputTokens, compacted)
	}
	if s.logf == nil {
		return
	}
	ev := map[string]any{
		"event":                       "gateway_inference_turn",
		"wire":                        wire,
		"stream":                      stream,
		"model":                       s.model,
		"finish_reason":               finishReason,
		"duration_ms":                 float64(dur.Microseconds()) / 1000.0,
		"prompt_tokens":               usage.PromptTokens,
		"completion_tokens":           usage.CompletionTokens,
		"cached_prompt_tokens":        usage.CachedPromptTokens(),
		"cache_read_input_tokens":     usage.CacheReadInputTokens,
		"cache_creation_input_tokens": usage.CacheCreationInputTokens,
		"cache_creation_span":         cacheCreationSpanLabel(usage.CacheCreationInputTokens, s.ttl1hActiveFor(traceID), s.ttl1hMessagePrefixFor(traceID)),
		"total_tokens":                usage.TotalTokens,
		"compaction_fired":            compacted,
		"context_event":               contextEvent,
	}
	if trace := strings.TrimSpace(traceID); trace != "" {
		ev["trace_id"] = trace
		// Attribute the served turn to the tenant principal that owns the trace — the
		// keyset-bound org/project (#5332) — so the access log is joinable to
		// /v1/fak/events by trace_id. Emitted only for a NAMED owner; the single-tenant
		// loopback ("" owner / unbound trace) leaves the line's shape unchanged.
		if owner, ok := s.traceOwnerOf(trace); ok && owner != "" {
			ev["principal"] = owner
		}
	}
	b, err := json.Marshal(ev)
	if err != nil {
		return
	}
	s.logf("%s", b)
	// The sibling per-request usage-record event (#10670) — same sink, same
	// trace attribution, emitted exactly once per completed request so a log
	// reader pairs the two lines by trace_id + request_ordinal.
	if usageRecOK {
		s.emitUsageRecordEvent(traceID, usageRec)
	}
}

// beginInflight records a request as live and returns a token to release it with.
// The returned id is 0 when m is nil so endInflight is always safe to defer.
func (m *gatewayMetrics) beginInflight(route string, start time.Time) uint64 {
	if m == nil {
		return 0
	}
	m.inflightMu.Lock()
	m.inflightSeq++
	id := m.inflightSeq
	m.inflightReq[id] = inflightEntry{route: route, start: start}
	m.inflightMu.Unlock()
	return id
}

func (m *gatewayMetrics) endInflight(id uint64) {
	if m == nil || id == 0 {
		return
	}
	m.inflightMu.Lock()
	delete(m.inflightReq, id)
	m.inflightMu.Unlock()
}

func newLatencyCounter() *latencyCounter {
	return &latencyCounter{buckets: make([]uint64, len(gatewayLatencyBuckets))}
}
