package gateway

import (
	"errors"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/compactcohere"
	"github.com/anthony-chaudhary/fak/internal/metrics"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/nativeperf"
	"github.com/anthony-chaudhary/fak/internal/perfledger"
	"github.com/anthony-chaudhary/fak/internal/vcacheobserve"
	"github.com/anthony-chaudhary/fak/pkg/deadlineadmit"
)

var gatewayLatencyBuckets = []float64{
	0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1,
	0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300, 600, 900, 1800,
}

type gatewayMetrics struct {
	// nativePhases receives the decode-window kernel phase split; it is the
	// server's receipt projection, shared so /metrics renders one family.
	nativePhases *nativeperf.ReceiptMetrics
	// requestOutcomes is a bounded, lossy memory-only terminal view.
	requestOutcomes    requestOutcomeStore
	featureActivations featureActivationMetrics
	start              time.Time
	inflight           int64
	featureProof       *FeatureProofCollector

	// deadlineEst is the deadline-aware admission estimator (see
	// deadline_admission.go); created lazily through deadlineEstimator.
	deadlineOnce sync.Once
	deadlineEst  *deadlineadmit.Estimator
	// warmPrefix is observational served-turn cache history
	// (deadline_cache_credit.go); created lazily through warmPrefixLedger.
	warmPrefixOnce sync.Once
	warmPrefix     *warmPrefixLedger
	// deadlineResidency holds engine /slots residency evidence for deadline
	// cache credit (deadline_residency.go).
	deadlineResidencyOnce sync.Once
	deadlineResidency     *deadlineResidencyMemory

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
	// progressReg is the live-heartbeat table (#12760) keyed by the
	// beginInflight request id. attachProgress registers at stream start and
	// endInflight unregisters at completion or cancel.
	progressReg ProgressRegistry

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
	// inferSyntheticTurns counts sender-marked synthetic turns (readiness canaries)
	// kept out of the latency histograms.
	inferSyntheticTurns uint64
	// The non-synthetic subset of inferPromptTokens / inferCachedTokens keyed by
	// the closed perfledger client class (metrics_by_client.go).
	inferPromptTokensByClient map[string]uint64
	inferCachedTokensByClient map[string]uint64
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
	// inferInputTokenHist / inferOutputTokenHist are the per-turn TOKEN-COUNT
	// distributions (provider-reported prompt tokens in / completion tokens out) that the
	// inferPromptTokens/inferComplTokens gauges only summarize as running totals. Their
	// bucket ladders scale to the deployment's token ceiling (tokenBuckets, TGI-adapted)
	// instead of the fixed seconds ladder, so P50/P95/P99 stay resolvable at any window
	// size. observeInferenceTimedDetail feeds both from the normalized counter axes;
	// writeInferenceMetrics renders their copied snapshots, empty at startup. Synthetic
	// turns remain included like token totals, unlike latency distributions. Omitted or
	// nonpositive counts do not enter either histogram. Guarded by inferenceMu.
	inferInputTokenHist  *tokenHistogram
	inferOutputTokenHist *tokenHistogram
	// inferRegimeHists cuts the same three distributions by the turn's cache regime
	// (#5630), booked from the same values in the same critical section, so the regime
	// rows of each family sum exactly to the unlabeled histogram above.
	inferRegimeHists map[string]*regimeLatencyHists
	// nativeExecutions is the ordinary-request native diagnostic fold. A row is
	// admitted only when the completion carries both the request-local native
	// decode marker and planner-authored timings; provider timings alone never
	// enter this map. It shares inferenceMu with the surrounding inference fold.
	nativeExecutions map[string]nativeExecutionStats

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
	compactRestored      uint64            // WITNESSED: dropped messages restored via fak_context_restore
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

	// softProgressStalls counts SOFT no-progress diagnostics (#10638): a stream stayed silent
	// (no turn-advancing frame; keepalives ignored) for the soft window and a content-free
	// elapsed-since-progress / retry-attempt receipt was captured WITHOUT ending the turn. It is
	// the observability that makes a stalled-but-alive turn visible before the hard deadline
	// produces a terminal 5xx. Bumped atomically from the SoftStallNotify hook, off the request
	// path. SEPARATE from upstreamErrors (a terminal failure) and upstreamRetries (a backoff).
	softProgressStalls uint64

	// softProgressStallElapsedNS / softProgressStallAttempt carry the most recent soft strike's
	// numbers (nanoseconds of silence and the upstream retry attempt) so the scrape and the
	// debug line can name them without reading the receipt file. Written with atomic stores by
	// observeSoftProgressStall off the request path.
	softProgressStallElapsedNS uint64
	softProgressStallAttempt   uint64

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

	// perfMu guards the per-request serving-performance ring (perf_ledger.go):
	// one perfledger.Record per served turn, drop-oldest at perfledger.RingCap,
	// seeded from the durable ledger tail at startup. perfSink is the async
	// durable writer; Offer never blocks the served turn.
	perfMu             sync.Mutex
	perfRecords        []perfledger.Record
	perfRecordsDropped bool
	perfSink           atomic.Pointer[perfledger.Writer]
	// phaseSink is the durable sibling of perfSink (phase_ledger_writer.go):
	// committed native phase receipts appended beside the perf ledger so the
	// six-bucket breakdown survives the process. Offer never blocks the turn.
	phaseSink    atomic.Pointer[phaseLedgerWriter]
	perfServedBy atomic.Pointer[perfledger.ServedBy]
	// requestPhases is the fak-request-phases/1 ring (request_phase_receipt.go).
	requestPhases requestPhaseRing
	// httpWriteTimeout is the serving http.Server's effective WriteTimeout (ns; 0 =
	// none or not serving through Serve). A buffered turn that outlives it never
	// reached the client, so its perf row is client_write_timeout.
	httpWriteTimeout atomic.Int64

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

	// scopeRefusalsMu guards the auth-scope refusal family (#12761): each
	// X-Fak-Auth-Scope claim the gateway refused with 403 scope_forbidden, keyed
	// by (scope, route_class) so an operator can tell an observation-polling
	// watcher from someone probing the admin plane. Refusals run from the auth
	// middleware on every request path; kept on its own lock off the families
	// above - a zero-or-one-increment-per-agent-refusal hot path.
	scopeRefusalsMu sync.Mutex
	scopeRefusals   map[scopeRefusalKey]uint64

	// toolCallParseMu guards the tool-call parse-outcome family: what happened to
	// the tool calls the upstream announced on a served turn — refused because
	// none survived parsing + the text-lift fallback (fail_closed), promoted from
	// a text-form dialect into structured calls (lifted), or carried into
	// adjudication as structured calls (structured). Keyed by (outcome, dialect,
	// reason) so an operator can tell a dialect that stopped lifting from a
	// dialect that started emitting unparseable calls. Kept on its own lock off
	// the families above — a zero-or-one-increment-per-served-turn path.
	toolCallParseMu sync.Mutex
	toolCallParse   map[toolCallParseKey]uint64
}

// scopeRefusalKey is the closed label set of fak_gateway_scope_refusals_total:
// the declared scope plus the coarse route class the refusal happened on.
type scopeRefusalKey struct {
	scope      string
	routeClass string
}

// observeScopeRefusal records one refused request in place, before the 403 is
// written, so a scrape taken mid-storm still sees the refusal exactly once.
func (m *gatewayMetrics) observeScopeRefusal(scope, routeClass string) {
	if scope == "" || routeClass == "" {
		return
	}
	m.scopeRefusalsMu.Lock()
	defer m.scopeRefusalsMu.Unlock()
	if m.scopeRefusals == nil {
		m.scopeRefusals = map[scopeRefusalKey]uint64{}
	}
	m.scopeRefusals[scopeRefusalKey{scope: scope, routeClass: routeClass}]++
}

// scopeRefusalSnapshot copies the live refusal map under its lock so the
// scrape renders in sorted (scope, route_class) order without holding a
// hot-path lock for the duration of the write.
func (m *gatewayMetrics) scopeRefusalSnapshot() map[scopeRefusalKey]uint64 {
	m.scopeRefusalsMu.Lock()
	defer m.scopeRefusalsMu.Unlock()
	snap := make(map[scopeRefusalKey]uint64, len(m.scopeRefusals))
	for k, v := range m.scopeRefusals {
		snap[k] = v
	}
	return snap
}

// toolCallParseKey is the closed label set of fak_gateway_toolcall_parse_total:
// the parse outcome, the text-lift dialect that produced it (empty when the
// outcome is not a lift), and the closed abi drop-reason name (empty unless the
// outcome is fail_closed).
type toolCallParseKey struct {
	outcome string
	dialect string
	reason  string
}

// The closed outcome vocabulary of fak_gateway_toolcall_parse_total.
const (
	// toolCallParseOutcomeFailClosed: the upstream announced tool calls but none
	// survived parsing + the text-lift fallback, so the turn was refused.
	toolCallParseOutcomeFailClosed = "fail_closed"
	// toolCallParseOutcomeLifted: a text-form tool-call dialect was promoted into
	// structured calls.
	toolCallParseOutcomeLifted = "lifted"
	// toolCallParseOutcomeStructured: the turn reached adjudication carrying
	// structured calls.
	toolCallParseOutcomeStructured = "structured"
)

// observeToolCallParse records one tool-call parse outcome in place, before the
// refusal is written, so a scrape taken mid-storm still sees the outcome exactly
// once. A no-op for a nil metrics or an empty outcome — the conformance helpers
// are called directly by tests that construct a Server with no metrics.
func (m *gatewayMetrics) observeToolCallParse(outcome, dialect, reason string) {
	if m == nil || outcome == "" {
		return
	}
	m.toolCallParseMu.Lock()
	defer m.toolCallParseMu.Unlock()
	if m.toolCallParse == nil {
		m.toolCallParse = map[toolCallParseKey]uint64{}
	}
	m.toolCallParse[toolCallParseKey{outcome: outcome, dialect: dialect, reason: reason}]++
}

// toolCallParseSnapshot copies the live parse-outcome map under its lock so the
// scrape renders in sorted (outcome, dialect, reason) order without holding a
// hot-path lock for the duration of the write. A nil metrics renders empty.
func (m *gatewayMetrics) toolCallParseSnapshot() map[toolCallParseKey]uint64 {
	if m == nil {
		return map[toolCallParseKey]uint64{}
	}
	m.toolCallParseMu.Lock()
	defer m.toolCallParseMu.Unlock()
	snap := make(map[toolCallParseKey]uint64, len(m.toolCallParse))
	for k, v := range m.toolCallParse {
		snap[k] = v
	}
	return snap
}

type inflightEntry struct {
	route string
	start time.Time
	// prefProg / decProg are the per-request progress tick counts wired from
	// the same ProgressHeartbeat path that reports live-stream progress; a
	// zero for both is a request whose wire has no observable tick. touched
	// is the last-progress marker (unix nano): zero means never ticked -
	// the field "wedged vs working" is read from.
	prefProg uint64
	decProg  uint64
	touched  int64
	// prefTouched and decTouched are the per-phase last-tick markers (unix
	// nano) backing the progress=1 per-phase ages. Zero means that phase has
	// never ticked on this request; TouchInflightProgress refreshes both.
	prefTouched int64
	decTouched  int64
}

type httpMetricKey struct {
	route  string
	method string
	status string
}

type operationMetricKey struct {
	operation      string
	verdict        string
	reason         string
	refusalSubtype string
	disposition    string
	by             string // which adjudicator decided (forensics) — answers WHO refused, not just that it was refused
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

type nativeExecutionStats struct {
	requests        uint64
	prefillSeconds  float64
	decodeSeconds   float64
	promptTokens    uint64
	generatedTokens uint64
	cachedTokens    uint64
}

func newGatewayMetrics(now time.Time) *gatewayMetrics {
	proof, _ := NewFeatureProofCollector(DefaultFeatureProofCapacity)
	return &gatewayMetrics{
		featureProof:             proof,
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
		inferInputTokenHist:      newTokenHistogram(defaultTokenHistogramMax),
		inferOutputTokenHist:     newTokenHistogram(defaultTokenHistogramMax),
		nativeExecutions:         map[string]nativeExecutionStats{},
	}
}

func (m *gatewayMetrics) nativeExecutionSnapshot() map[string]nativeExecutionStats {
	if m == nil {
		return nil
	}
	m.inferenceMu.Lock()
	defer m.inferenceMu.Unlock()
	out := make(map[string]nativeExecutionStats, len(m.nativeExecutions))
	for path, stats := range m.nativeExecutions {
		out[path] = stats
	}
	return out
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
	// A fail-closed device-stage failure (a *model.BackendForwardOperationError, e.g. the V4.1
	// mHC pre-collapse readback at a specific layer) is a LOCAL in-kernel engine fault, not an
	// upstream provider failure. It is in-kernel by construction, so it gets its OWN kind —
	// distinct from the coarse "other" bucket and from "oom" — so a /metrics scrape (and the
	// FAILED debug line) can tell a device-stage fault storm apart from everything else. Stays
	// in lockstep with upstreamErrorStatus's in_kernel_device_stage_failed arm (#3518).
	var bfo *model.BackendForwardOperationError
	if errors.As(err, &bfo) {
		return "device_stage"
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

func (m *gatewayMetrics) recordCompactionRestore(count int) {
	if m == nil || count <= 0 {
		return
	}
	m.compactMu.Lock()
	m.compactRestored += uint64(count)
	m.compactMu.Unlock()
}

func (s *Server) recordCompactionRestore(count int) {
	if s != nil && s.metrics != nil {
		s.metrics.recordCompactionRestore(count)
	}
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

func newLatencyCounter() *latencyCounter {
	return &latencyCounter{buckets: make([]uint64, len(gatewayLatencyBuckets))}
}
