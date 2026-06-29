package gateway

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/blob"
	"github.com/anthony-chaudhary/fak/internal/cachemeta"
	"github.com/anthony-chaudhary/fak/internal/cacheobs"
	"github.com/anthony-chaudhary/fak/internal/compactcohere"
	"github.com/anthony-chaudhary/fak/internal/kernel"
	"github.com/anthony-chaudhary/fak/internal/vcachegov"
	"github.com/anthony-chaudhary/fak/internal/vcacheobserve"
	"github.com/anthony-chaudhary/fak/internal/vdso"
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
	// inferCacheCreationTokens is the cumulative provider cache_creation_input_tokens —
	// the WRITE axis the read-only ProviderCacheSavingsUSD never retained. With it and
	// the read total, the session can report NET realized vcache economics (read saving
	// minus write premium) via the same engine `fak vcache observe` uses offline.
	inferCacheCreationTokens uint64
	inferDecodeSecs          float64
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
	compactMu          sync.Mutex
	compactAttempts    map[string]uint64 // WITNESSED: outcome -> count: fired | bailed | off
	compactBailReasons map[string]uint64 // WITNESSED: CompactReason* -> count (why a bail happened)
	compactDropped     uint64            // WITNESSED: whole messages stubbed out across all fires
	compactShed        uint64            // WITNESSED: estimated tokens fak removed from the body across all fires
	compactCacheReads  uint64            // OBSERVED: sum of provider-reported cache_read on compacted turns
	compactLastCacheRd float64           // OBSERVED: provider-reported cache_read on the MOST RECENT compacted turn

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
	toolPruneMu    sync.Mutex
	toolPruneTurns uint64 // WITNESSED: turns where >=1 unreachable tool def was pruned from tools[]
	toolPruneCount uint64 // WITNESSED: total tool defs removed across all prune turns

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

	// oomMu guards the in-kernel device-OOM visibility family. These are LOCAL resource
	// exhaustion faults: either recovered compute.DeviceAllocError allocations or a request
	// capacity precheck that refused a known-too-large plan before allocation, never provider
	// errors. The Prometheus labels stay class-only to avoid allocator-site cardinality;
	// /debug/vars keeps the most recent site for operator drilldown.
	oomMu       sync.Mutex
	inKernelOOM map[string]*inKernelOOMClassStats

	// upstreamErrMu guards the upstream-error visibility family: a count of proxy/planner
	// turn FAILURES keyed by a KIND (stalled / unreachable / oom / rate_limited / auth /
	// forbidden / status_4xx / status_5xx / other), so an operator can scrape WHY turns are
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
		start:              now,
		http:               map[httpMetricKey]*latencyCounter{},
		operations:         map[operationMetricKey]*latencyCounter{},
		inflightReq:        map[uint64]inflightEntry{},
		compactAttempts:    map[string]uint64{},
		compactBailReasons: map[string]uint64{},
		reqMemoryObserved:  map[string]uint64{},
		reqMemoryPlan:      map[requestMemoryMetricKey]*requestMemoryMetricStats{},
		reqMemoryTokens:    map[requestMemoryTokenKey]*requestMemoryTokenStats{},
		reqMemoryFit:       map[requestMemoryFitKey]*requestMemoryFitStats{},
		inKernelOOM:        map[string]*inKernelOOMClassStats{},
		upstreamErrors:     map[string]uint64{},
		harnessCoherence:   newHarnessCoherenceMetrics(compactcohere.DefaultProviderCacheTTL),
		routing:            newRoutingMetrics(),
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
		}
		if se.Status >= 400 && se.Status < 500 {
			return "status_4xx"
		}
		return "status_5xx"
	}
	return "other"
}

// observeUpstreamError increments the upstream-error counter for the error's KIND. It is the
// single fold point for every proxy/planner error path (called from plannerErrorStatus), so a
// turn failure is counted exactly once. A nil or unclassifiable error is a no-op.
func (m *gatewayMetrics) observeUpstreamError(err error) {
	if m == nil {
		return
	}
	kind := upstreamErrorKind(err)
	if kind == "" {
		return
	}
	m.upstreamErrMu.Lock()
	if m.upstreamErrors == nil {
		m.upstreamErrors = map[string]uint64{}
	}
	m.upstreamErrors[kind]++
	m.upstreamErrMu.Unlock()
}

// observeUpstreamRetry counts one upstream retry attempt (the planner's 429/5xx backoff). Atomic
// and off the request path, called from the RetryNotify hook.
func (m *gatewayMetrics) observeUpstreamRetry() {
	if m == nil {
		return
	}
	atomic.AddUint64(&m.upstreamRetries, 1)
}

// observeInKernelOOM folds a planner error into the local device-OOM visibility family when
// it is either an in-kernel allocation fault or the request-time capacity precheck refusal.
// It returns true only for that local OOM class, so callers can record without re-doing
// errors.As.
func (m *gatewayMetrics) observeInKernelOOM(err error) bool {
	if m == nil || err == nil {
		return false
	}
	class, bytes, site, ok := inKernelOOMObservation(err)
	if !ok {
		return false
	}
	m.oomMu.Lock()
	if m.inKernelOOM == nil {
		m.inKernelOOM = map[string]*inKernelOOMClassStats{}
	}
	st := m.inKernelOOM[class]
	if st == nil {
		st = &inKernelOOMClassStats{}
		m.inKernelOOM[class] = st
	}
	st.count++
	st.failedBytes += bytes
	st.lastFailedBytes = bytes
	st.lastSite = site
	m.oomMu.Unlock()
	return true
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

// observeCompaction records the outcome of one history-compaction attempt. off=true means the
// budget was unset (the lever is configured off); otherwise the outcome's Reason decides fired
// vs bailed and which bail-reason bucket increments.
func (m *gatewayMetrics) observeCompaction(out agent.CompactOutcome, off bool) {
	if m == nil {
		return
	}
	m.compactMu.Lock()
	defer m.compactMu.Unlock()
	switch {
	case off:
		m.compactAttempts["off"]++
	case out.Reason == agent.CompactReasonNone:
		m.compactAttempts["fired"]++
		m.compactDropped += uint64(out.Dropped)
		m.compactShed += uint64(out.ShedTokens)
	default:
		m.compactAttempts["bailed"]++
		m.compactBailReasons[out.Reason]++
	}
}

// recordCompactionCacheRead records the provider's cache_read_input_tokens on a turn whose body
// WAS compacted. This is an OBSERVED value relayed verbatim from the upstream, NOT a fak claim:
// fak's own guarantee (the protected prefix it shipped was byte-identical) is already witnessed
// by the turn counting `fired` with no `prefix_mismatch` bail. A low cache_read here therefore
// does not mean fak broke anything — pair it with shed_tokens to see the net effect, and read a
// crater as a provider-side miss (TTL expiry / eviction / the client moving its breakpoint)
// unless bail_reason{prefix_mismatch} is nonzero.
func (m *gatewayMetrics) recordCompactionCacheRead(cacheRead int) {
	if m == nil {
		return
	}
	m.compactMu.Lock()
	m.compactCacheReads += uint64(cacheRead)
	m.compactLastCacheRd = float64(cacheRead)
	m.compactMu.Unlock()
}

// observeInboundToolPrune records that a turn pruned n unreachable tool definitions from the
// outbound tools[] (the INBOUND tool-floor prune lever). n<=0 is a no-op, so the common turn —
// where no advertised tool is floor-denied past the cache_control breakpoint — records nothing,
// exactly as a clean compaction turn does. WITNESSED: fak chose what to drop, and the pruner
// proved the cached prefix stayed byte-identical before returning Changed=true, so a counted
// prune never bursts the upstream cache.
func (m *gatewayMetrics) observeInboundToolPrune(n int) {
	if m == nil || n <= 0 {
		return
	}
	m.toolPruneMu.Lock()
	m.toolPruneTurns++
	m.toolPruneCount += uint64(n)
	m.toolPruneMu.Unlock()
}

// inboundToolPruneSnapshot reads the WITNESSED tool-prune accumulators under their lock. Pure
// read — the exit summary and the Prometheus surface both fold the same two numbers, so the line
// can never disagree with the scrape.
func (m *gatewayMetrics) inboundToolPruneSnapshot() (turns, count uint64) {
	if m == nil {
		return 0, 0
	}
	m.toolPruneMu.Lock()
	defer m.toolPruneMu.Unlock()
	return m.toolPruneTurns, m.toolPruneCount
}

// recordResetShadow folds one compacted turn's resetScore SHADOW verdict into the recommend-only
// accumulators. The reset policy NEVER acts in shadow mode (reset_shadow.go); this only counts what
// it WOULD recommend, bucketed by the closed ResetReason, so an operator can watch the cut-vs-reset
// pressure build before reset is ever enabled. The verdict is WITNESSED (fak's own policy); the
// inputs it scored are OBSERVED (provider cache counters). Lazily inits the reason map like
// observeInference, so a Server built without this family present still records correctly.
func (m *gatewayMetrics) recordResetShadow(d ResetDecision) {
	if m == nil {
		return
	}
	m.resetShadowMu.Lock()
	if m.resetShadowReasons == nil {
		m.resetShadowReasons = map[string]uint64{}
	}
	m.resetShadowReasons[string(d.Reason)]++
	if d.ShouldReset {
		m.resetShadowRecommend++
	}
	m.resetShadowLastScore = d.Score
	m.resetShadowMu.Unlock()
}

func (m *gatewayMetrics) observeHTTP(route, method string, status int, dur time.Duration) {
	if m == nil {
		return
	}
	key := httpMetricKey{route: route, method: method, status: itoa(uint64(status))}
	m.mu.Lock()
	counter := m.http[key]
	if counter == nil {
		counter = newLatencyCounter()
		m.http[key] = counter
	}
	counter.observe(dur.Seconds())
	m.mu.Unlock()
}

func (m *gatewayMetrics) observeOperation(operation string, v WireVerdict, err error, dur time.Duration) {
	if m == nil {
		return
	}
	key := operationMetricKey{
		operation: operation,
		verdict:   v.Kind,
		reason:    v.Reason,
	}
	if err != nil || key.verdict == "" {
		key.verdict = "ERROR"
	}
	key.disposition = v.Disposition
	key.by = v.By
	m.mu.Lock()
	counter := m.operations[key]
	if counter == nil {
		counter = newLatencyCounter()
		m.operations[key] = counter
	}
	counter.observe(dur.Seconds())
	m.mu.Unlock()
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
	// InputTokens (the uncached input remainder) and CacheCreationTokens (the cache
	// WRITE axis) are retained alongside CachedPromptTokens (the READ axis) so the
	// summary can price the NET realized provider-cache saving — read rebate MINUS
	// write premium — via ProviderCacheNetSavings, the axis the read-only
	// ProviderCacheSavingsUSD deliberately omits. Both OBSERVED (provider-relayed).
	InputTokens         uint64 `json:"input_tokens"`
	CacheCreationTokens uint64 `json:"cache_creation_tokens"`

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
	// CompactionBudget is the resident-token threshold the history rewrite fires past
	// (Config.CompactHistoryBudget; 0 means the lever is OFF, body forwarded byte-for-byte).
	// Surfaced so the exit line can say whether compaction is ENABLED and merely idle
	// (budget>0, nothing exceeded it) vs DISABLED — the two readings of "0 fired" the bare
	// counters cannot tell apart.
	CompactionBudget int `json:"compaction_budget"`

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
	sum.CacheCreationTokens = m.inferCacheCreationTokens
	m.inferenceMu.Unlock()

	comp := m.compactionSnapshotData()
	sum.CompactionFired = comp.attempts["fired"]
	sum.CompactionBailed = comp.attempts["bailed"]
	sum.CompactionOff = comp.attempts["off"]
	sum.CompactionDroppedTurns = comp.dropped
	sum.CompactionShedTokens = comp.shed
	sum.CompactionCacheReadTokens = comp.cacheReads
	sum.LastCompactionCacheRead = comp.lastCacheRd
	// Carry the per-reason bail breakdown so the banner can explain the bailed lump (the
	// snapshot already copied it under compactMu); only attach a non-empty map so a clean
	// session keeps the JSON field absent (omitempty).
	sum.ToolPruneTurns, sum.ToolPruneCount = m.inboundToolPruneSnapshot()
	if len(comp.bailReasons) > 0 {
		sum.CompactionBailReasons = comp.bailReasons
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

// observeInference records one served model-generation turn: its token accounting,
// why decode stopped, and the wall-clock the planner spent producing it. promptTok /
// complTok / cachedTok come straight from the planner's reported Usage; dur is the
// time spent inside planner.Complete. Negative/zero values are ignored so a planner
// that omits a count never corrupts the running totals. This is the signal that makes
// a busy gateway look busy: fak_kernel_*/fak_vdso_* stay 0 on a pure chat workload
// (no syscall, no fast-path lookup), so without this family every panel reads 0 while
// the box is in fact decoding tokens.
func (m *gatewayMetrics) observeInference(promptTok, complTok, cachedTok, cacheCreateTok int, finishReason string, dur time.Duration) {
	// A buffered turn cannot observe the first-token boundary, so prefill is "not
	// measured": ttft<=0 routes the whole duration into the decode-total accumulator
	// and leaves the prefill split untouched (it stays an honest 0, never a phantom).
	m.observeInferenceTimed(promptTok, complTok, cachedTok, cacheCreateTok, finishReason, dur, 0)
}

// observeInferenceTimed is observeInference with an explicit time-to-first-token
// split. ttft is the wall-clock from the planner call starting to the FIRST content
// delta arriving (the prefill phase: prompt ingest + first token); dur is the whole
// turn. When ttft is in (0, dur] the turn's time is split — prefill = ttft, decode =
// dur-ttft — and the turn is counted toward the TTFT denominator so the prefill rate
// divides streamed prefill tokens by only the turns that measured them. When ttft<=0
// (a buffered turn, or a stream that produced no delta), the whole duration counts as
// decode total and the prefill split is left untouched. inferDecodeSecs stays the
// FULL inference wall-clock in both cases so the existing output_tokens_per_second and
// the fleet-value agent-seconds denominator are byte-identical to before.
func (m *gatewayMetrics) observeInferenceTimed(promptTok, complTok, cachedTok, cacheCreateTok int, finishReason string, dur, ttft time.Duration) {
	if m == nil {
		return
	}
	if finishReason == "" {
		finishReason = "unknown"
	}
	m.inferenceMu.Lock()
	if m.inferReqs == nil {
		m.inferReqs = map[string]uint64{}
	}
	m.inferReqs[finishReason]++
	if promptTok > 0 {
		m.inferPromptTokens += uint64(promptTok)
	}
	if complTok > 0 {
		m.inferComplTokens += uint64(complTok)
	}
	if cachedTok > 0 {
		m.inferCachedTokens += uint64(cachedTok)
		m.inferCachedHits++ // this turn got a provider prompt-cache READ
	}
	if cacheCreateTok > 0 {
		m.inferCacheCreationTokens += uint64(cacheCreateTok) // this turn WROTE the provider cache
	}
	if dur > 0 {
		m.inferDecodeSecs += dur.Seconds()
	}
	// Split prefill from decode only when TTFT was actually observed and is sane
	// (positive and within the total). A clamp guards against a clock skew producing
	// ttft>dur, which would otherwise make decode-seconds negative.
	if ttft > 0 && dur > 0 {
		pre := ttft
		if pre > dur {
			pre = dur
		}
		m.inferPrefillSecs += pre.Seconds()
		m.inferTTFTTurns++
		if promptTok > 0 {
			m.inferPrefillPromptTokens += uint64(promptTok)
		}
		m.inferMeasuredDecodeSecs += (dur - pre).Seconds()
		if complTok > 0 {
			m.inferMeasuredComplTokens += uint64(complTok)
		}
	}
	m.inferenceMu.Unlock()
}

func (s *Server) observePlannerRequestMemory() {
	if s == nil || s.metrics == nil || s.planner == nil {
		return
	}
	reporter, ok := s.planner.(agent.RequestMemoryReporter)
	if !ok {
		return
	}
	s.metrics.observeRequestMemory(reporter.RequestMemoryStats())
}

func (m *gatewayMetrics) observeRequestMemory(st agent.RequestMemoryStats) {
	if m == nil || !st.Observed {
		return
	}
	backend := defaultBackendLabel(st.Backend)
	planRows := requestMemoryPlanByClassScopeDType(st.MemoryPlan)
	fitRows := requestMemoryFitRows(st.MemoryPlan, st.Capacities, st.HeadroomRatio)

	m.reqMemoryMu.Lock()
	if m.reqMemoryObserved == nil {
		m.reqMemoryObserved = map[string]uint64{}
	}
	if m.reqMemoryPlan == nil {
		m.reqMemoryPlan = map[requestMemoryMetricKey]*requestMemoryMetricStats{}
	}
	if m.reqMemoryTokens == nil {
		m.reqMemoryTokens = map[requestMemoryTokenKey]*requestMemoryTokenStats{}
	}
	if m.reqMemoryFit == nil {
		m.reqMemoryFit = map[requestMemoryFitKey]*requestMemoryFitStats{}
	}
	m.reqMemoryObserved[backend]++
	for _, row := range planRows {
		k := requestMemoryMetricKey{backend: backend, class: row.Class, scope: row.Scope, dtype: row.DType}
		st := m.reqMemoryPlan[k]
		if st == nil {
			st = &requestMemoryMetricStats{}
			m.reqMemoryPlan[k] = st
		}
		st.observations++
		st.totalBytes = addPositiveInt64ToUint64(st.totalBytes, row.Bytes)
		if row.Bytes > st.highWaterBytes {
			st.highWaterBytes = row.Bytes
		}
	}
	m.observeRequestMemoryTokenLocked(backend, "prompt", st.PromptTokens)
	m.observeRequestMemoryTokenLocked(backend, "max_new", st.MaxNewTokens)
	m.observeRequestMemoryTokenLocked(backend, "planned", st.PlannedTokens)
	for _, row := range fitRows {
		k := requestMemoryFitKey{backend: backend, scope: row.Scope}
		st := m.reqMemoryFit[k]
		if st == nil {
			st = &requestMemoryFitStats{}
			m.reqMemoryFit[k] = st
		}
		st.observations++
		if row.WantBytes > st.wantHighWater {
			st.wantHighWater = row.WantBytes
		}
		if row.CapacityKnown && (!st.marginKnown || row.MarginBytes < st.marginLowWater) {
			st.marginKnown = true
			st.marginLowWater = row.MarginBytes
		}
	}
	m.reqMemoryMu.Unlock()
}

func (m *gatewayMetrics) observeRequestMemoryTokenLocked(backend, kind string, value int) {
	if value < 0 {
		return
	}
	k := requestMemoryTokenKey{backend: backend, kind: kind}
	st := m.reqMemoryTokens[k]
	if st == nil {
		st = &requestMemoryTokenStats{}
		m.reqMemoryTokens[k] = st
	}
	st.observations++
	st.total = addPositiveIntToUint64(st.total, value)
	if value > st.highWater {
		st.highWater = value
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

func (s *Server) logInferenceTurn(traceID, wire string, stream bool, usage agent.Usage, finishReason string, dur time.Duration, compacted bool) {
	if s == nil {
		return
	}
	// Record this turn into the per-family live-observe window (#935) BEFORE the sink
	// gates below, so the per-family / governor / warmth view is populated even with
	// --log off and --debug-stats off. The family is the session/trace prefix; the token
	// axes are the provider's own counters (OBSERVED). This is purely observational — it
	// never feeds the request path (Law A2).
	s.metrics.observeVCacheTurn(traceID, time.Now().UnixMilli(),
		usage.PromptTokens, usage.CacheReadInputTokens, usage.CacheCreationInputTokens)
	// The per-turn human debug render (#793) fires independently of the JSON --log sink, so
	// --debug-stats works on a clean (--log off) terminal. It is a no-op unless debugStatsf is
	// wired, and reuses the #792 rolling health (read-only peek; no double-roll).
	if s.debugStatsf != nil {
		s.renderTurnDebugStats(traceID, wire, stream, finishReason,
			usage.PromptTokens, usage.CompletionTokens, usage.CacheReadInputTokens, usage.CacheCreationInputTokens, compacted)
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
		"total_tokens":                usage.TotalTokens,
		"compaction_fired":            compacted,
	}
	if trace := strings.TrimSpace(traceID); trace != "" {
		ev["trace_id"] = trace
	}
	b, err := json.Marshal(ev)
	if err != nil {
		return
	}
	s.logf("%s", b)
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

// inflightSnapshot derives live signals from the set of requests running right
// now: how many are in flight per route, and the age of the oldest one. This is
// computed at scrape time precisely because the completion-time histograms can't
// see a request that has not finished; maxAge is the hung-request detector.
func (m *gatewayMetrics) inflightSnapshot(now time.Time) (byRoute map[string]int, maxAge float64) {
	byRoute = map[string]int{}
	if m == nil {
		return byRoute, 0
	}
	m.inflightMu.Lock()
	for _, e := range m.inflightReq {
		byRoute[e.route]++
		if age := now.Sub(e.start).Seconds(); age > maxAge {
			maxAge = age
		}
	}
	m.inflightMu.Unlock()
	return byRoute, maxAge
}

func newLatencyCounter() *latencyCounter {
	return &latencyCounter{buckets: make([]uint64, len(gatewayLatencyBuckets))}
}

func (c *latencyCounter) observe(seconds float64) {
	c.count++
	c.sum += seconds
	for i, le := range gatewayLatencyBuckets {
		if seconds <= le {
			c.buckets[i]++
		}
	}
}

type latencySnapshot struct {
	count   uint64
	sum     float64
	buckets []uint64
}

func (c *latencyCounter) snapshot() latencySnapshot {
	return latencySnapshot{
		count:   c.count,
		sum:     c.sum,
		buckets: append([]uint64(nil), c.buckets...),
	}
}

type httpMetricSnapshot struct {
	key httpMetricKey
	val latencySnapshot
}

type operationMetricSnapshot struct {
	key operationMetricKey
	val latencySnapshot
}

func (s *Server) handleMetrics(w http.ResponseWriter, r *http.Request) {
	if !requireMethod(w, r, http.MethodGet) {
		return
	}
	w.Header().Set("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
	_, _ = w.Write([]byte(s.renderMetrics()))
}

func (s *Server) renderMetrics() string {
	m := s.metrics
	if m == nil {
		m = newGatewayMetrics(time.Now())
	}
	httpRows, opRows := m.snapshot()
	var b strings.Builder

	writeHelpType(&b, "fak_gateway_up", "Whether the fak gateway process is scrapeable.", "gauge")
	fmt.Fprintln(&b, "fak_gateway_up 1")
	writeHelpType(&b, "fak_gateway_start_time_seconds", "Unix start time of this fak gateway process.", "gauge")
	fmt.Fprintf(&b, "fak_gateway_start_time_seconds %d\n", m.start.Unix())
	s.writeStartupMetrics(&b)
	writeHelpType(&b, "fak_gateway_inflight_requests", "HTTP requests currently executing in the fak gateway.", "gauge")
	fmt.Fprintf(&b, "fak_gateway_inflight_requests %d\n", atomic.LoadInt64(&m.inflight))

	// Live-request visibility: derived from the in-flight registry at scrape time.
	// max_age is the oldest currently-running request's age (0 when idle); it
	// surfaces a slow or wedged request at the next scrape, where the completion-
	// time histograms would show nothing until the request finally returned.
	byRoute, maxAge := m.inflightSnapshot(time.Now())
	writeHelpType(&b, "fak_gateway_inflight_max_age_seconds", "Age of the oldest HTTP request currently in flight (0 when idle).", "gauge")
	fmt.Fprintf(&b, "fak_gateway_inflight_max_age_seconds %s\n", promFloat(maxAge))
	writeHelpType(&b, "fak_gateway_inflight_requests_by_route", "HTTP requests currently executing, by route.", "gauge")
	inflightRoutes := make([]string, 0, len(byRoute))
	for route := range byRoute {
		inflightRoutes = append(inflightRoutes, route)
	}
	sort.Strings(inflightRoutes)
	for _, route := range inflightRoutes {
		fmt.Fprintf(&b, "fak_gateway_inflight_requests_by_route{route=\"%s\"} %d\n", promQuote(route), byRoute[route])
	}

	writeHelpType(&b, "fak_gateway_build_info", "Static fak gateway build and runtime labels.", "gauge")
	fmt.Fprintf(&b, "fak_gateway_build_info{version=\"%s\",engine=\"%s\",model=\"%s\",vdso=\"%s\"} 1\n",
		promQuote(s.version), promQuote(s.engineID), promQuote(s.model), promQuote(strconv.FormatBool(s.k.VDSOEnabled())))

	writeHelpType(&b, "fak_gateway_http_requests_total", "HTTP requests served by route, method, and status.", "counter")
	for _, row := range httpRows {
		fmt.Fprintf(&b, "fak_gateway_http_requests_total{route=\"%s\",method=\"%s\",status=\"%s\"} %d\n",
			promQuote(row.key.route), promQuote(row.key.method), promQuote(row.key.status), row.val.count)
	}
	writeHelpType(&b, "fak_gateway_http_request_duration_seconds", "HTTP request latency by route, method, and status.", "histogram")
	for _, row := range httpRows {
		baseLabels := fmt.Sprintf("route=\"%s\",method=\"%s\",status=\"%s\"",
			promQuote(row.key.route), promQuote(row.key.method), promQuote(row.key.status))
		writeHistogram(&b, "fak_gateway_http_request_duration_seconds", baseLabels, row.val)
	}

	// Upstream-error visibility (the metric twin of the per-turn FAILED debug line): WHY turns
	// failed this session, by coarse kind.
	m.writeUpstreamErrorMetrics(&b)

	// In-kernel background-loop runtime: per-loop tick/error/panic/restart counters,
	// last-tick gauge, and liveness — the proof the kernel's loops keep progressing.
	s.writeBgloopMetrics(&b)

	writeHelpType(&b, "fak_gateway_operations_total", "Gateway kernel operations by operation, verdict, and deciding adjudicator (by).", "counter")
	for _, row := range opRows {
		fmt.Fprintf(&b, "fak_gateway_operations_total{operation=\"%s\",verdict=\"%s\",reason=\"%s\",disposition=\"%s\",by=\"%s\"} %d\n",
			promQuote(row.key.operation), promQuote(row.key.verdict), promQuote(row.key.reason),
			promQuote(row.key.disposition), promQuote(row.key.by), row.val.count)
	}
	writeHelpType(&b, "fak_gateway_operation_duration_seconds", "Gateway kernel operation latency by operation, verdict, and deciding adjudicator (by).", "histogram")
	for _, row := range opRows {
		baseLabels := fmt.Sprintf("operation=\"%s\",verdict=\"%s\",reason=\"%s\",disposition=\"%s\",by=\"%s\"",
			promQuote(row.key.operation), promQuote(row.key.verdict), promQuote(row.key.reason),
			promQuote(row.key.disposition), promQuote(row.key.by))
		writeHistogram(&b, "fak_gateway_operation_duration_seconds", baseLabels, row.val)
	}

	c := s.k.Counters()
	writeCounter(&b, "fak_kernel_submits_total", "Kernel submissions since process start.", c.Submits)
	writeCounter(&b, "fak_kernel_vdso_hits_total", "Kernel submissions served by the vDSO fast path.", c.VDSOHits)
	writeCounter(&b, "fak_kernel_engine_calls_total", "Kernel submissions that reached the engine.", c.EngineCalls)
	writeCounter(&b, "fak_kernel_denies_total", "Kernel submissions denied before execution.", c.Denies)
	writeCounter(&b, "fak_kernel_transforms_total", "Kernel submissions transformed by adjudication.", c.Transforms)
	writeCounter(&b, "fak_kernel_quarantines_total", "Kernel result admissions quarantined by the result-side stack.", c.Quarantines)
	writeCounter(&b, "fak_kernel_result_denies_total", "Kernel result admissions hard-refused by the result-side stack.", c.ResultDenies)
	writeCounter(&b, "fak_kernel_admitted_total", "Kernel result admissions that were accepted or transformed.", c.Admitted)
	// Per-rung decision distribution (issue #693): which adjudication rung actually
	// decided each call, bucketed by (rung, kind, reason). Passive — re-derived off the
	// hot path; a vDSO-served call (no adjudication) lands in rung="vdso". Drill down on
	// one call with `fak preflight --explain`. nil (older construction) suppresses it.
	if s.rungObs != nil {
		writeHelpType(&b, "fak_kernel_decisions_total", "Kernel decisions by winning adjudication rung, verdict kind, and reason (passive; re-derived off the hot path).", "counter")
		for _, row := range s.rungObs.Snapshot() {
			fmt.Fprintf(&b, "fak_kernel_decisions_total{rung=\"%s\",kind=\"%s\",reason=\"%s\"} %d\n",
				promQuote(row.Rung), promQuote(row.Kind), promQuote(row.Reason), row.Count)
		}
	}
	writeHelpType(&b, "fak_gateway_vdso_hit_ratio", "Current cumulative vDSO hit ratio over kernel submissions.", "gauge")
	ratio := 0.0
	if c.Submits > 0 {
		ratio = float64(c.VDSOHits) / float64(c.Submits)
	}
	fmt.Fprintf(&b, "fak_gateway_vdso_hit_ratio %s\n", promFloat(ratio))
	writeVDSOMetrics(&b)
	// Unified cache-stream family (fak_cache_*): the per-(plane,tier,kind) fold over
	// the cachemeta.Entry stream, fed live by the vDSO tier-2 cache-event sink wired
	// in New. It sits beside the per-cache fak_vdso_* family above; the snapshot
	// carries its own HELP/TYPE lines, so it concatenates cleanly. A Server without a
	// stream (older construction paths) emits nothing rather than a phantom series.
	if s.cacheStream != nil {
		b.WriteString(s.cacheStream.Snapshot().Prometheus())
	}
	writeBlobMetrics(&b)
	writeKVPrefixMetrics(&b)
	s.writeKVMemoryMetrics(&b)
	s.writeRequestMemoryMetrics(&b)
	m.writeRequestMemoryAggregateMetrics(&b)
	inf := m.writeInferenceMetrics(&b)
	m.writeVCacheMetrics(&b)
	m.writeInKernelOOMMetrics(&b)
	s.writeInKernelOOMRetryMetrics(&b)
	s.writeInKernelPressureTrimMetrics(&b)
	m.writeCompactionMetrics(&b)
	m.writeResetShadowMetrics(&b)
	m.harnessCoherence.writeHarnessCoherenceMetrics(&b)
	m.writeRoutingMetrics(&b)         // #603: per-aspect model-routing decision distribution (rule/strategy/aspect)
	s.resumeProj.writeMetrics(&b)     // #941: resume projected-vs-observed residual (self-contained family)
	s.writeFleetMembershipMetrics(&b) // #42: live fleet membership/health/drain/failover transitions, per worker
	s.writeAdmissionMetrics(&b)       // #35: native serving-scheduler admission family (fak_sched_*), when a controller is wired
	s.writePreemptionMetrics(&b)      // #31: native KV preemption/swap/recompute family, when a preemptor is wired

	// Fleet-value (hero-axis) KPIs, derived live from the kernel counters + the
	// inference accumulators above. fak's product axis is agent-fleet serving
	// efficiency (HERO-BENCHMARK), and these are the per-node ingredients of that
	// headline: turns the kernel saved (engine round-trips + retry turns avoided),
	// context-window pollutions it blocked, and the wall-clock agents were served.
	// agentSeconds is the time spent generating completions plus adjudicating/
	// dispatching kernel operations — the denominator for a live per-second view.
	var opSecs float64
	for _, row := range opRows {
		opSecs += row.val.sum
	}
	writeFleetValueMetrics(&b, c, inf.decodeSecs+opSecs)

	s.writeModelLoadMetrics(&b)
	return b.String()
}

func (s *Server) writeRequestMemoryMetrics(b *strings.Builder) {
	reporter, ok := s.planner.(agent.RequestMemoryReporter)
	if !ok {
		return
	}
	st := reporter.RequestMemoryStats()
	if !st.Observed {
		return
	}
	backend := defaultBackendLabel(st.Backend)
	writeHelpType(b, "fak_gateway_in_kernel_request_memory_plan_bytes", "Most recent served in-kernel backend request memory plan, by class/scope/dtype. This is a last-request gauge, not a cumulative counter.", "gauge")
	for _, row := range requestMemoryPlanByClassScopeDType(st.MemoryPlan) {
		fmt.Fprintf(b, "fak_gateway_in_kernel_request_memory_plan_bytes{backend=\"%s\",class=\"%s\",scope=\"%s\",dtype=\"%s\"} %d\n",
			promQuote(backend), promQuote(row.Class), promQuote(row.Scope), promQuote(row.DType), row.Bytes)
	}
	writeHelpType(b, "fak_gateway_in_kernel_request_memory_tokens", "Token window used by the most recent served in-kernel backend request memory plan.", "gauge")
	fmt.Fprintf(b, "fak_gateway_in_kernel_request_memory_tokens{backend=\"%s\",kind=\"prompt\"} %d\n", promQuote(backend), st.PromptTokens)
	fmt.Fprintf(b, "fak_gateway_in_kernel_request_memory_tokens{backend=\"%s\",kind=\"max_new\"} %d\n", promQuote(backend), st.MaxNewTokens)
	fmt.Fprintf(b, "fak_gateway_in_kernel_request_memory_tokens{backend=\"%s\",kind=\"planned\"} %d\n", promQuote(backend), st.PlannedTokens)
	if st.HeadroomRatio > 0 {
		writeHelpType(b, "fak_gateway_in_kernel_request_memory_headroom_ratio", "Fraction of reported capacity reserved by the most recent in-kernel request fit check for runtime headroom.", "gauge")
		fmt.Fprintf(b, "fak_gateway_in_kernel_request_memory_headroom_ratio{backend=\"%s\"} %s\n", promQuote(backend), promFloat(st.HeadroomRatio))
	}
	if len(st.Capacities) > 0 {
		writeHelpType(b, "fak_gateway_in_kernel_request_memory_capacity_known", "Whether the backend reported capacity for a memory scope used by the most recent in-kernel request fit check.", "gauge")
		for _, cap := range sortedRequestMemoryCapacities(st.Capacities) {
			known := 0
			if cap.Known {
				known = 1
			}
			fmt.Fprintf(b, "fak_gateway_in_kernel_request_memory_capacity_known{backend=\"%s\",scope=\"%s\"} %d\n",
				promQuote(backend), promQuote(modelLoadScope(cap.Scope)), known)
		}
		writeHelpType(b, "fak_gateway_in_kernel_request_memory_capacity_free_known", "Whether the backend reported current free bytes for a memory scope used by the most recent in-kernel request fit check.", "gauge")
		for _, cap := range sortedRequestMemoryCapacities(st.Capacities) {
			known := 0
			if cap.Known && cap.FreeKnown {
				known = 1
			}
			fmt.Fprintf(b, "fak_gateway_in_kernel_request_memory_capacity_free_known{backend=\"%s\",scope=\"%s\"} %d\n",
				promQuote(backend), promQuote(modelLoadScope(cap.Scope)), known)
		}
		writeHelpType(b, "fak_gateway_in_kernel_request_memory_capacity_bytes", "Reported backend capacity bytes used by the most recent in-kernel request fit check. The free row is omitted when current free bytes are unknown.", "gauge")
		for _, cap := range sortedRequestMemoryCapacities(st.Capacities) {
			if !cap.Known {
				continue
			}
			scope := modelLoadScope(cap.Scope)
			fmt.Fprintf(b, "fak_gateway_in_kernel_request_memory_capacity_bytes{backend=\"%s\",scope=\"%s\",kind=\"total\"} %d\n",
				promQuote(backend), promQuote(scope), cap.TotalBytes)
			if cap.FreeKnown {
				fmt.Fprintf(b, "fak_gateway_in_kernel_request_memory_capacity_bytes{backend=\"%s\",scope=\"%s\",kind=\"free\"} %d\n",
					promQuote(backend), promQuote(scope), cap.FreeBytes)
			}
		}
	}
	if rows := requestMemoryFitRows(st.MemoryPlan, st.Capacities, st.HeadroomRatio); len(rows) > 0 {
		writeHelpType(b, "fak_gateway_in_kernel_request_memory_fit_bytes", "Headroom-adjusted fit summary for the most recent in-kernel request by backend and scope. kind=want is planned bytes; kind=budget and kind=margin are omitted when capacity is unknown.", "gauge")
		for _, row := range rows {
			fmt.Fprintf(b, "fak_gateway_in_kernel_request_memory_fit_bytes{backend=\"%s\",scope=\"%s\",kind=\"want\"} %d\n",
				promQuote(backend), promQuote(row.Scope), row.WantBytes)
			if !row.CapacityKnown {
				continue
			}
			fmt.Fprintf(b, "fak_gateway_in_kernel_request_memory_fit_bytes{backend=\"%s\",scope=\"%s\",kind=\"budget\"} %d\n",
				promQuote(backend), promQuote(row.Scope), row.BudgetBytes)
			fmt.Fprintf(b, "fak_gateway_in_kernel_request_memory_fit_bytes{backend=\"%s\",scope=\"%s\",kind=\"margin\"} %d\n",
				promQuote(backend), promQuote(row.Scope), row.MarginBytes)
		}
	}
}

func (m *gatewayMetrics) writeRequestMemoryAggregateMetrics(b *strings.Builder) {
	snap := m.requestMemoryAggregateSnapshotData()
	if len(snap.observed) == 0 {
		return
	}
	writeHelpType(b, "fak_gateway_in_kernel_request_memory_observations_total", "In-kernel backend request memory plans observed after served planner turns, by backend. Includes successful turns and local OOM/capacity refusals that produced a plan.", "counter")
	backends := make([]string, 0, len(snap.observed))
	for backend := range snap.observed {
		backends = append(backends, backend)
	}
	sort.Strings(backends)
	for _, backend := range backends {
		fmt.Fprintf(b, "fak_gateway_in_kernel_request_memory_observations_total{backend=\"%s\"} %d\n",
			promQuote(backend), snap.observed[backend])
	}
	writeHelpType(b, "fak_gateway_in_kernel_request_memory_plan_observations_total", "Observed in-kernel request memory plan rows by backend, class, scope, and dtype.", "counter")
	for _, row := range snap.plans {
		fmt.Fprintf(b, "fak_gateway_in_kernel_request_memory_plan_observations_total{backend=\"%s\",class=\"%s\",scope=\"%s\",dtype=\"%s\"} %d\n",
			promQuote(row.key.backend), promQuote(row.key.class), promQuote(row.key.scope), promQuote(row.key.dtype), row.observations)
	}
	writeHelpType(b, "fak_gateway_in_kernel_request_memory_plan_bytes_total", "Cumulative planned bytes observed for served in-kernel backend requests, by backend, class, scope, and dtype.", "counter")
	for _, row := range snap.plans {
		fmt.Fprintf(b, "fak_gateway_in_kernel_request_memory_plan_bytes_total{backend=\"%s\",class=\"%s\",scope=\"%s\",dtype=\"%s\"} %d\n",
			promQuote(row.key.backend), promQuote(row.key.class), promQuote(row.key.scope), promQuote(row.key.dtype), row.totalBytes)
	}
	writeHelpType(b, "fak_gateway_in_kernel_request_memory_plan_high_water_bytes", "Largest single observed in-kernel request memory plan row, by backend, class, scope, and dtype.", "gauge")
	for _, row := range snap.plans {
		fmt.Fprintf(b, "fak_gateway_in_kernel_request_memory_plan_high_water_bytes{backend=\"%s\",class=\"%s\",scope=\"%s\",dtype=\"%s\"} %d\n",
			promQuote(row.key.backend), promQuote(row.key.class), promQuote(row.key.scope), promQuote(row.key.dtype), row.highWaterBytes)
	}
	writeHelpType(b, "fak_gateway_in_kernel_request_memory_tokens_total", "Cumulative prompt/max_new/planned token windows from observed in-kernel request memory plans.", "counter")
	for _, row := range snap.tokens {
		fmt.Fprintf(b, "fak_gateway_in_kernel_request_memory_tokens_total{backend=\"%s\",kind=\"%s\"} %d\n",
			promQuote(row.key.backend), promQuote(row.key.kind), row.total)
	}
	writeHelpType(b, "fak_gateway_in_kernel_request_memory_tokens_high_water", "Largest prompt/max_new/planned token window from an observed in-kernel request memory plan.", "gauge")
	for _, row := range snap.tokens {
		fmt.Fprintf(b, "fak_gateway_in_kernel_request_memory_tokens_high_water{backend=\"%s\",kind=\"%s\"} %d\n",
			promQuote(row.key.backend), promQuote(row.key.kind), row.highWater)
	}
	writeHelpType(b, "fak_gateway_in_kernel_request_memory_fit_observations_total", "Observed in-kernel request memory fit rows by backend and scope.", "counter")
	for _, row := range snap.fits {
		fmt.Fprintf(b, "fak_gateway_in_kernel_request_memory_fit_observations_total{backend=\"%s\",scope=\"%s\"} %d\n",
			promQuote(row.key.backend), promQuote(row.key.scope), row.observations)
	}
	writeHelpType(b, "fak_gateway_in_kernel_request_memory_fit_want_high_water_bytes", "Largest observed planned in-kernel request memory bytes by backend and scope.", "gauge")
	for _, row := range snap.fits {
		fmt.Fprintf(b, "fak_gateway_in_kernel_request_memory_fit_want_high_water_bytes{backend=\"%s\",scope=\"%s\"} %d\n",
			promQuote(row.key.backend), promQuote(row.key.scope), row.wantHighWater)
	}
	writeHelpType(b, "fak_gateway_in_kernel_request_memory_fit_margin_low_water_bytes", "Smallest observed headroom-adjusted fit margin for known-capacity in-kernel requests, by backend and scope. Omitted for scopes whose capacity was unknown.", "gauge")
	for _, row := range snap.fits {
		if !row.marginKnown {
			continue
		}
		fmt.Fprintf(b, "fak_gateway_in_kernel_request_memory_fit_margin_low_water_bytes{backend=\"%s\",scope=\"%s\"} %d\n",
			promQuote(row.key.backend), promQuote(row.key.scope), row.marginLowWater)
	}
}

func (s *Server) writeInKernelOOMRetryMetrics(b *strings.Builder) {
	if s == nil || s.planner == nil {
		return
	}
	reporter, ok := s.planner.(agent.InKernelOOMRetryReporter)
	if !ok {
		return
	}
	st := reporter.InKernelOOMRetryStats()
	if len(st.Rows) == 0 {
		return
	}
	backend := defaultBackendLabel(st.Backend)
	writeHelpType(b, "fak_gateway_in_kernel_oom_retry_total", "Idle-pool trim retries attempted after local in-kernel device allocation OOMs, bucketed by backend, memory class, and outcome. These are decode retries only; capacity precheck refusals do not retry.", "counter")
	for _, row := range st.Rows {
		class := oomClassLabel(row.Class)
		fmt.Fprintf(b, "fak_gateway_in_kernel_oom_retry_total{backend=\"%s\",class=\"%s\",outcome=\"attempted\"} %d\n",
			promQuote(backend), promQuote(class), row.Attempts)
		fmt.Fprintf(b, "fak_gateway_in_kernel_oom_retry_total{backend=\"%s\",class=\"%s\",outcome=\"succeeded\"} %d\n",
			promQuote(backend), promQuote(class), row.Successes)
		fmt.Fprintf(b, "fak_gateway_in_kernel_oom_retry_total{backend=\"%s\",class=\"%s\",outcome=\"failed\"} %d\n",
			promQuote(backend), promQuote(class), row.Failures)
	}
	writeHelpType(b, "fak_gateway_in_kernel_oom_retry_last_failed_bytes", "Most recent allocation size that triggered an idle-pool trim retry for each backend and memory class.", "gauge")
	for _, row := range st.Rows {
		fmt.Fprintf(b, "fak_gateway_in_kernel_oom_retry_last_failed_bytes{backend=\"%s\",class=\"%s\"} %d\n",
			promQuote(backend), promQuote(oomClassLabel(row.Class)), row.LastFailedBytes)
	}
}

func (s *Server) writeInKernelPressureTrimMetrics(b *strings.Builder) {
	if s == nil || s.planner == nil {
		return
	}
	reporter, ok := s.planner.(agent.InKernelMemoryPressureTrimReporter)
	if !ok {
		return
	}
	st := reporter.InKernelMemoryPressureTrimStats()
	if len(st.Rows) == 0 {
		return
	}
	backend := defaultBackendLabel(st.Backend)
	writeHelpType(b, "fak_gateway_in_kernel_memory_pressure_trim_total", "Idle-pool trims attempted before local in-kernel decode when a known request memory plan is refused or close to the headroom-adjusted budget. resolved means a capacity-precheck refusal fit after trimming.", "counter")
	for _, row := range st.Rows {
		scope := modelLoadScope(row.Scope)
		class := oomClassLabel(row.Class)
		reason := pressureTrimReasonLabel(row.Reason)
		fmt.Fprintf(b, "fak_gateway_in_kernel_memory_pressure_trim_total{backend=\"%s\",scope=\"%s\",class=\"%s\",reason=\"%s\",outcome=\"attempted\"} %d\n",
			promQuote(backend), promQuote(scope), promQuote(class), promQuote(reason), row.Attempts)
		fmt.Fprintf(b, "fak_gateway_in_kernel_memory_pressure_trim_total{backend=\"%s\",scope=\"%s\",class=\"%s\",reason=\"%s\",outcome=\"trimmed\"} %d\n",
			promQuote(backend), promQuote(scope), promQuote(class), promQuote(reason), row.Trimmed)
		fmt.Fprintf(b, "fak_gateway_in_kernel_memory_pressure_trim_total{backend=\"%s\",scope=\"%s\",class=\"%s\",reason=\"%s\",outcome=\"no_hooks\"} %d\n",
			promQuote(backend), promQuote(scope), promQuote(class), promQuote(reason), row.NoHooks)
		fmt.Fprintf(b, "fak_gateway_in_kernel_memory_pressure_trim_total{backend=\"%s\",scope=\"%s\",class=\"%s\",reason=\"%s\",outcome=\"resolved\"} %d\n",
			promQuote(backend), promQuote(scope), promQuote(class), promQuote(reason), row.Resolved)
	}
	writeHelpType(b, "fak_gateway_in_kernel_memory_pressure_trim_last_bytes", "Most recent request memory pressure trim sizing by backend, scope, class, and reason. kind=margin may be negative for a refused precheck.", "gauge")
	for _, row := range st.Rows {
		scope := modelLoadScope(row.Scope)
		class := oomClassLabel(row.Class)
		reason := pressureTrimReasonLabel(row.Reason)
		fmt.Fprintf(b, "fak_gateway_in_kernel_memory_pressure_trim_last_bytes{backend=\"%s\",scope=\"%s\",class=\"%s\",reason=\"%s\",kind=\"want\"} %d\n",
			promQuote(backend), promQuote(scope), promQuote(class), promQuote(reason), row.LastWantBytes)
		fmt.Fprintf(b, "fak_gateway_in_kernel_memory_pressure_trim_last_bytes{backend=\"%s\",scope=\"%s\",class=\"%s\",reason=\"%s\",kind=\"budget\"} %d\n",
			promQuote(backend), promQuote(scope), promQuote(class), promQuote(reason), row.LastBudgetBytes)
		fmt.Fprintf(b, "fak_gateway_in_kernel_memory_pressure_trim_last_bytes{backend=\"%s\",scope=\"%s\",class=\"%s\",reason=\"%s\",kind=\"margin\"} %d\n",
			promQuote(backend), promQuote(scope), promQuote(class), promQuote(reason), row.LastMarginBytes)
	}
}

func pressureTrimReasonLabel(reason string) string {
	reason = strings.TrimSpace(reason)
	switch reason {
	case "capacity_precheck", "low_margin":
		return reason
	case "":
		return "unknown"
	default:
		return "other"
	}
}

func requestMemoryPlanByClassScopeDType(plan []agent.RequestMemoryDemand) []agent.RequestMemoryDemand {
	type key struct {
		class string
		scope string
		dtype string
	}
	by := map[key]int64{}
	for _, row := range plan {
		if row.Bytes <= 0 {
			continue
		}
		k := key{class: modelLoadClass(row.Class), scope: modelLoadScope(row.Scope), dtype: modelLoadDType(row.DType)}
		by[k] += row.Bytes
	}
	out := make([]agent.RequestMemoryDemand, 0, len(by))
	for k, bytes := range by {
		out = append(out, agent.RequestMemoryDemand{Class: k.class, Scope: k.scope, DType: k.dtype, Bytes: bytes})
	}
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].Scope != out[j].Scope {
			return out[i].Scope < out[j].Scope
		}
		if out[i].Class != out[j].Class {
			return out[i].Class < out[j].Class
		}
		return out[i].DType < out[j].DType
	})
	return out
}

func sortedRequestMemoryCapacities(in []agent.RequestMemoryCapacity) []agent.RequestMemoryCapacity {
	out := append([]agent.RequestMemoryCapacity(nil), in...)
	sort.SliceStable(out, func(i, j int) bool { return modelLoadScope(out[i].Scope) < modelLoadScope(out[j].Scope) })
	return out
}

// writeVDSOMetrics renders vDSO fast-path effectiveness from the live vdso.Default
// stats API — the SAME process-global instance the kernel consults on the fast path
// and the coherence feed subscribes to. The kernel's fak_kernel_vdso_hits_total above
// counts submissions the fast path served; these gauges instead report the cache's OWN
// view: how often a lookup hit, how many entries it has filled, and how often a write
// stranded cached reads.
//
// Two of the issue's three asks are rendered from a direct accessor; two are rendered
// from the nearest exported signal because the vDSO does not expose them:
//   - hit rate    -> Stats() hitRate, a direct accessor.
//   - entry count -> Stats() fills is the CUMULATIVE number of entries stored. The vDSO
//     does not export the live cache occupancy (len of the tier-2 map), so this is the
//     fills counter, not a current-size gauge.
//   - eviction rate -> the vDSO does not export a per-entry LRU-eviction counter. The
//     nearest exported invalidation signal is Mutations(): write-shaped completions
//     that strand cached reads by bumping the world/scope epoch. Reported as such.
func writeVDSOMetrics(b *strings.Builder) {
	lookups, hits, fills, hitRate := vdso.Default.Stats()
	writeHelpType(b, "fak_vdso_lookups_total", "vDSO fast-path lookups attempted (tier-1/2/3 consulted).", "counter")
	fmt.Fprintf(b, "fak_vdso_lookups_total %d\n", lookups)
	writeHelpType(b, "fak_vdso_hits_total", "vDSO fast-path lookups served locally (a hit at any tier).", "counter")
	fmt.Fprintf(b, "fak_vdso_hits_total %d\n", hits)
	writeHelpType(b, "fak_vdso_hit_rate", "vDSO lookup hit rate (hits/lookups) from the live vdso.Default stats API.", "gauge")
	fmt.Fprintf(b, "fak_vdso_hit_rate %s\n", promFloat(hitRate))
	writeHelpType(b, "fak_vdso_cache_fills_total", "vDSO tier-2 cache entries filled since start (cumulative; the vDSO exports no live occupancy).", "counter")
	fmt.Fprintf(b, "fak_vdso_cache_fills_total %d\n", fills)
	writeHelpType(b, "fak_vdso_invalidations_total", "Write-shaped completions that stranded cached reads (the vDSO exports no per-entry LRU-eviction counter; this is the nearest invalidation signal).", "counter")
	fmt.Fprintf(b, "fak_vdso_invalidations_total %d\n", vdso.Default.Mutations())

	// Miss attribution: every lookup that returned no local answer, by reason — so a
	// low hit rate is explainable (write-shaped tools vs missing readOnly/idempotent
	// hints vs genuine cache churn) instead of collapsing to a bare miss. This is the
	// aggregate complement to the per-call decision trace (fak preflight --explain).
	writeHelpType(b, "fak_vdso_misses_total", "vDSO fast-path lookups that returned no local answer, by reason (DESTRUCTIVE: write-shaped tool, never cacheable; MISSING_HINTS: no readOnly/idempotent hint; RESOURCE_MISNAMED: read cannot name its entity; WITNESS_REVOKED: entry's external witness was refuted; NOT_CACHED: cacheable but unfilled or epoch-stranded).", "counter")
	misses := vdso.Default.MissReasons()
	missReasons := make([]string, 0, len(misses))
	for r := range misses {
		missReasons = append(missReasons, r)
	}
	sort.Strings(missReasons)
	for _, r := range missReasons {
		fmt.Fprintf(b, "fak_vdso_misses_total{reason=\"%s\"} %d\n", promQuote(r), misses[r])
	}
}

// writeBlobMetrics renders the content-addressed blob store (internal/blob) — the ONE
// CAS the vDSO tier-2 cache AND the context-MMU page-out share, so it is the
// cross-cache footprint/dedup/eviction surface a level below the per-cache families
// above. The store kept concurrency-safe KPI taps (Stats/Resident/MaxBytes) but
// emitted no metrics; this lights them up so an operator can see resident footprint,
// content-dedup effectiveness, and whether the byte bound is actually evicting — a
// rising fak_blob_evicted_total while the resident gauges plateau is the
// leak-absorbed-by-the-bound signal the store's own doc comment calls out.
func writeBlobMetrics(b *strings.Builder) {
	puts, dedupHits, resolves := blob.Default.Stats()
	residentBlobs, residentBytes, evicted := blob.Default.Resident()

	writeCounter(b, "fak_blob_puts_total", "Payloads stored into the content-addressed blob store (CAS puts; small inline payloads never reach the store and are not counted).", puts)
	writeCounter(b, "fak_blob_dedup_hits_total", "CAS puts whose digest was already resident — content-addressed dedup, the byte stored once and shared by the vDSO cache and the context-MMU.", dedupHits)
	writeCounter(b, "fak_blob_resolves_total", "Blob materializations (Resolve) served from the CAS.", resolves)
	writeHelpType(b, "fak_blob_resident_blobs", "Distinct blobs currently resident in the shared CAS.", "gauge")
	fmt.Fprintf(b, "fak_blob_resident_blobs %d\n", residentBlobs)
	writeHelpType(b, "fak_blob_resident_bytes", "Total bytes currently resident in the shared CAS (the live footprint a leak/pressure alarm watches).", "gauge")
	fmt.Fprintf(b, "fak_blob_resident_bytes %d\n", residentBytes)
	writeCounter(b, "fak_blob_evicted_total", "Digests dropped by the CAS byte bound (only ever UNPINNED transient payloads; a rising count is real working pressure or a leak the bound is absorbing).", evicted)
	writeHelpType(b, "fak_blob_max_bytes", "Configured resident-footprint ceiling for the CAS in bytes (0 = unbounded).", "gauge")
	fmt.Fprintf(b, "fak_blob_max_bytes %d\n", blob.Default.MaxBytes())
	writeHelpType(b, "fak_blob_dedup_ratio", "Fraction of CAS puts served by content dedup (dedup_hits/puts; 0 when nothing has been put).", "gauge")
	ratio := 0.0
	if puts > 0 {
		ratio = float64(dedupHits) / float64(puts)
	}
	fmt.Fprintf(b, "fak_blob_dedup_ratio %s\n", promFloat(ratio))
}

// writeKVPrefixMetrics renders the in-kernel KV-prefix reuse family from the process-global
// cacheobs.Default tap (fed by the planner on every served in-kernel turn). It is the LIVE
// measurement of the frozen-trajectory cache cliff (docs/explainers/frozen-trajectory-cache-cliff.md):
// the reuse ratio is the realized cache-hit, and the per-regime turn buckets show when turns
// leave the frozen append-only regime. This is the LOCAL-KV analogue of the provider-side
// fak_gateway_inference_cached_prompt_* family — distinct signal (the in-kernel RadixAttention
// prefix match, not a remote provider's cache_read), so the two never double-count. On a pure
// proxy workload (no in-kernel model) the tap is never fed and these series stay 0.
func writeKVPrefixMetrics(b *strings.Builder) {
	s := cacheobs.Default.Snapshot()
	writeCounter(b, "fak_gateway_kv_prefix_turns_total",
		"In-kernel model turns observed for KV-prefix reuse (the planner reported a prompt-token count).", int64(s.Turns))
	writeCounter(b, "fak_gateway_kv_prefix_prompt_tokens_total",
		"Prompt (prefill) tokens summed across in-kernel turns — the denominator of the realized cache-hit.", int64(s.PromptTokens))
	writeCounter(b, "fak_gateway_kv_prefix_reused_tokens_total",
		"Prompt tokens served from the cached KV prefix (the RadixAttention match) across in-kernel turns — the prefill work the kernel did NOT redo. Distinct from the provider's cache_read (fak_gateway_inference_cached_prompt_tokens_total).", int64(s.ReusedTokens))
	writeHelpType(b, "fak_gateway_kv_prefix_turns_by_regime_total",
		"In-kernel turns by reuse regime — the live cliff distribution. frozen: reuse >= 0.90 (the append-only ceiling); partial: 0.10-0.90; cold: < 0.10 (a cold first prefill, or a head-mutated / fanned-out turn that left the frozen single-linear regime — see docs/explainers/frozen-trajectory-cache-cliff.md).", "counter")
	fmt.Fprintf(b, "fak_gateway_kv_prefix_turns_by_regime_total{regime=\"frozen\"} %d\n", s.FrozenTurns)
	fmt.Fprintf(b, "fak_gateway_kv_prefix_turns_by_regime_total{regime=\"partial\"} %d\n", s.PartialTurns)
	fmt.Fprintf(b, "fak_gateway_kv_prefix_turns_by_regime_total{regime=\"cold\"} %d\n", s.ColdTurns)
	writeHelpType(b, "fak_gateway_kv_prefix_reuse_ratio",
		"Realized in-kernel KV-prefix cache-hit: reused / prompt tokens across served turns (0 until the first turn). A single append-only agent climbs toward ~1 (the frozen ceiling); flexibility, cold fan-out, or a divergent prefix drives it down — the frozen-trajectory cache cliff, measured live.", "gauge")
	fmt.Fprintf(b, "fak_gateway_kv_prefix_reuse_ratio %s\n", promFloat(s.ReuseRatio))
}

func (s *Server) writeKVMemoryMetrics(b *strings.Builder) {
	if s == nil || s.planner == nil {
		return
	}
	reporter, ok := s.planner.(agent.KVMemoryReporter)
	if !ok {
		return
	}
	st := reporter.KVMemoryStats()
	class := strings.TrimSpace(st.MemoryClass)
	if class == "" {
		class = "kv_cache"
	}
	scope := strings.TrimSpace(st.Scope)
	if scope == "" {
		scope = "host"
	}
	backend := defaultBackendLabel(st.Backend)
	labels := fmt.Sprintf("class=\"%s\",scope=\"%s\",backend=\"%s\"", promQuote(class), promQuote(scope), promQuote(backend))
	dtype := modelLoadDType(st.DType)
	enabled := 0
	if st.Enabled {
		enabled = 1
	}
	writeHelpType(b, "fak_gateway_kv_memory_enabled", "Whether a local reusable KV prefix cache is active for this planner. Proxy/mock planners emit no resident-KV series.", "gauge")
	fmt.Fprintf(b, "fak_gateway_kv_memory_enabled{%s} %d\n", labels, enabled)
	writeHelpType(b, "fak_gateway_kv_memory_dtype_info", "Storage dtype for local KV-cache rows under this planner/backend. Current HAL KV rows are f32; proxy/mock planners emit no resident-KV series.", "gauge")
	fmt.Fprintf(b, "fak_gateway_kv_memory_dtype_info{%s,dtype=\"%s\"} 1\n", labels, promQuote(dtype))
	writeHelpType(b, "fak_gateway_kv_memory_bytes_per_token", "Estimated bytes for one resident KV position under this model layout (classed as kv_cache).", "gauge")
	fmt.Fprintf(b, "fak_gateway_kv_memory_bytes_per_token{%s} %d\n", labels, st.BytesPerToken)
	if st.HeadroomRatio > 0 {
		writeHelpType(b, "fak_gateway_kv_memory_headroom_ratio", "Fraction of reported capacity reserved when calculating resident KV-cache fit budget.", "gauge")
		fmt.Fprintf(b, "fak_gateway_kv_memory_headroom_ratio{%s} %s\n", labels, promFloat(st.HeadroomRatio))
	}
	known := 0
	if st.CapacityKnown {
		known = 1
	}
	writeHelpType(b, "fak_gateway_kv_memory_capacity_known", "Whether the planner reported capacity for the resident KV-cache memory scope.", "gauge")
	fmt.Fprintf(b, "fak_gateway_kv_memory_capacity_known{%s} %d\n", labels, known)
	freeKnown := 0
	if st.CapacityKnown && st.CapacityFreeKnown {
		freeKnown = 1
	}
	writeHelpType(b, "fak_gateway_kv_memory_capacity_free_known", "Whether the planner reported current free bytes for the resident KV-cache memory scope.", "gauge")
	fmt.Fprintf(b, "fak_gateway_kv_memory_capacity_free_known{%s} %d\n", labels, freeKnown)
	if st.CapacityKnown {
		writeHelpType(b, "fak_gateway_kv_memory_capacity_bytes", "Reported capacity bytes for the resident KV-cache memory scope. The free row is omitted when current free bytes are unknown.", "gauge")
		fmt.Fprintf(b, "fak_gateway_kv_memory_capacity_bytes{%s,kind=\"total\"} %d\n", labels, st.CapacityTotalBytes)
		if st.CapacityFreeKnown {
			fmt.Fprintf(b, "fak_gateway_kv_memory_capacity_bytes{%s,kind=\"free\"} %d\n", labels, st.CapacityFreeBytes)
		}
	}
	writeHelpType(b, "fak_gateway_kv_memory_fit_bytes", "Headroom-adjusted fit summary for resident KV-cache memory. kind=want is current resident bytes; kind=budget and kind=margin are omitted when capacity is unknown.", "gauge")
	fmt.Fprintf(b, "fak_gateway_kv_memory_fit_bytes{%s,kind=\"want\"} %d\n", labels, st.ResidentBytes)
	if st.CapacityKnown {
		fmt.Fprintf(b, "fak_gateway_kv_memory_fit_bytes{%s,kind=\"budget\"} %d\n", labels, st.FitBudgetBytes)
		fmt.Fprintf(b, "fak_gateway_kv_memory_fit_bytes{%s,kind=\"margin\"} %d\n", labels, st.FitMarginBytes)
	}
	if !st.Enabled {
		return
	}
	writeHelpType(b, "fak_gateway_kv_memory_resident_tokens", "True resident KV prefix positions held by the local prefix cache. This can exceed the LRU edge-token budget on deep radix chains.", "gauge")
	fmt.Fprintf(b, "fak_gateway_kv_memory_resident_tokens{%s} %d\n", labels, st.ResidentTokens)
	writeHelpType(b, "fak_gateway_kv_memory_resident_bytes", "Estimated resident local KV-cache bytes, derived from resident prefix positions and model KV geometry.", "gauge")
	fmt.Fprintf(b, "fak_gateway_kv_memory_resident_bytes{%s} %d\n", labels, st.ResidentBytes)
	writeHelpType(b, "fak_gateway_kv_memory_lru_tokens", "Radix KV edge-token count enforced by the LRU budget. This is not the same as true resident prefix positions.", "gauge")
	fmt.Fprintf(b, "fak_gateway_kv_memory_lru_tokens{%s} %d\n", labels, st.LRUTokens)
	writeHelpType(b, "fak_gateway_kv_memory_budget_tokens", "Configured radix KV LRU edge-token budget (0 means unbounded).", "gauge")
	fmt.Fprintf(b, "fak_gateway_kv_memory_budget_tokens{%s} %d\n", labels, st.BudgetTokens)
	writeHelpType(b, "fak_gateway_kv_memory_max_depth_tokens", "Longest cached KV prefix depth in tokens.", "gauge")
	fmt.Fprintf(b, "fak_gateway_kv_memory_max_depth_tokens{%s} %d\n", labels, st.MaxDepthTokens)
	writeHelpType(b, "fak_gateway_kv_memory_nodes", "Radix KV prefix-cache nodes currently resident.", "gauge")
	fmt.Fprintf(b, "fak_gateway_kv_memory_nodes{%s} %d\n", labels, st.Nodes)
	writeHelpType(b, "fak_gateway_kv_memory_leaves", "Radix KV prefix-cache leaves currently resident.", "gauge")
	fmt.Fprintf(b, "fak_gateway_kv_memory_leaves{%s} %d\n", labels, st.Leaves)
	writeHelpType(b, "fak_gateway_kv_memory_evictions_total", "Radix KV prefix-cache evictions by cause: lru pressure or policy/quarantine.", "counter")
	fmt.Fprintf(b, "fak_gateway_kv_memory_evictions_total{%s,kind=\"lru\"} %d\n", labels, st.Evictions)
	fmt.Fprintf(b, "fak_gateway_kv_memory_evictions_total{%s,kind=\"policy\"} %d\n", labels, st.PolicyEvictions)
	writeHelpType(b, "fak_gateway_kv_memory_splits_total", "Radix KV prefix-cache edge splits performed to expose reusable mid-edge prefixes.", "counter")
	fmt.Fprintf(b, "fak_gateway_kv_memory_splits_total{%s} %d\n", labels, st.Splits)
}

type inferenceSnapshot struct {
	reqs           map[string]uint64
	promptTok      uint64
	complTok       uint64
	cachedTok      uint64
	cacheCreateTok uint64
	cachedHits     uint64
	decodeSecs     float64
	// prefillSecs is the cumulative TTFT wall-clock over the ttftTurns that measured
	// it; prefillPromptTok is the prompt-token sum over those same turns. ttftTurns is
	// the denominator that keeps the prefill rate honest on a mixed workload.
	prefillSecs      float64
	ttftTurns        uint64
	prefillPromptTok uint64
	// measuredDecodeSecs / measuredComplTok are the decode-rate pair over ttftTurns
	// only (see the accumulator doc) — kept separate so a mixed workload never blends
	// measured and unmeasured denominators.
	measuredDecodeSecs float64
	measuredComplTok   uint64
}

type compactionSnapshot struct {
	attempts    map[string]uint64
	bailReasons map[string]uint64
	dropped     uint64
	shed        uint64
	cacheReads  uint64
	lastCacheRd float64
}

type requestMemoryAggregateSnapshot struct {
	observed map[string]uint64
	plans    []requestMemoryPlanSnapshot
	tokens   []requestMemoryTokenSnapshot
	fits     []requestMemoryFitSnapshot
}

type requestMemoryPlanSnapshot struct {
	key            requestMemoryMetricKey
	observations   uint64
	totalBytes     uint64
	highWaterBytes int64
}

type requestMemoryTokenSnapshot struct {
	key          requestMemoryTokenKey
	observations uint64
	total        uint64
	highWater    int
}

type requestMemoryFitSnapshot struct {
	key            requestMemoryFitKey
	observations   uint64
	wantHighWater  int64
	marginLowWater int64
	marginKnown    bool
}

type inKernelOOMSnapshot struct {
	class           string
	count           uint64
	failedBytes     uint64
	lastFailedBytes uint64
	lastSite        string
}

func (m *gatewayMetrics) inferenceSnapshotData() inferenceSnapshot {
	m.inferenceMu.Lock()
	defer m.inferenceMu.Unlock()
	reqs := make(map[string]uint64, len(m.inferReqs))
	for k, v := range m.inferReqs {
		reqs[k] = v
	}
	return inferenceSnapshot{
		reqs:               reqs,
		promptTok:          m.inferPromptTokens,
		complTok:           m.inferComplTokens,
		cachedTok:          m.inferCachedTokens,
		cacheCreateTok:     m.inferCacheCreationTokens,
		cachedHits:         m.inferCachedHits,
		decodeSecs:         m.inferDecodeSecs,
		prefillSecs:        m.inferPrefillSecs,
		ttftTurns:          m.inferTTFTTurns,
		prefillPromptTok:   m.inferPrefillPromptTokens,
		measuredDecodeSecs: m.inferMeasuredDecodeSecs,
		measuredComplTok:   m.inferMeasuredComplTokens,
	}
}

func (m *gatewayMetrics) compactionSnapshotData() compactionSnapshot {
	if m == nil {
		return compactionSnapshot{
			attempts:    map[string]uint64{},
			bailReasons: map[string]uint64{},
		}
	}
	m.compactMu.Lock()
	defer m.compactMu.Unlock()
	attempts := map[string]uint64{}
	for k, v := range m.compactAttempts {
		attempts[k] = v
	}
	bailReasons := map[string]uint64{}
	for k, v := range m.compactBailReasons {
		bailReasons[k] = v
	}
	return compactionSnapshot{
		attempts:    attempts,
		bailReasons: bailReasons,
		dropped:     m.compactDropped,
		shed:        m.compactShed,
		cacheReads:  m.compactCacheReads,
		lastCacheRd: m.compactLastCacheRd,
	}
}

func (m *gatewayMetrics) requestMemoryAggregateSnapshotData() requestMemoryAggregateSnapshot {
	if m == nil {
		return requestMemoryAggregateSnapshot{observed: map[string]uint64{}}
	}
	m.reqMemoryMu.Lock()
	defer m.reqMemoryMu.Unlock()
	out := requestMemoryAggregateSnapshot{observed: map[string]uint64{}}
	for backend, n := range m.reqMemoryObserved {
		out.observed[backend] = n
	}
	for key, st := range m.reqMemoryPlan {
		if st == nil {
			continue
		}
		out.plans = append(out.plans, requestMemoryPlanSnapshot{
			key:            key,
			observations:   st.observations,
			totalBytes:     st.totalBytes,
			highWaterBytes: st.highWaterBytes,
		})
	}
	for key, st := range m.reqMemoryTokens {
		if st == nil {
			continue
		}
		out.tokens = append(out.tokens, requestMemoryTokenSnapshot{
			key:          key,
			observations: st.observations,
			total:        st.total,
			highWater:    st.highWater,
		})
	}
	for key, st := range m.reqMemoryFit {
		if st == nil {
			continue
		}
		out.fits = append(out.fits, requestMemoryFitSnapshot{
			key:            key,
			observations:   st.observations,
			wantHighWater:  st.wantHighWater,
			marginLowWater: st.marginLowWater,
			marginKnown:    st.marginKnown,
		})
	}
	sort.SliceStable(out.plans, func(i, j int) bool {
		a, b := out.plans[i].key, out.plans[j].key
		if a.backend != b.backend {
			return a.backend < b.backend
		}
		if a.scope != b.scope {
			return a.scope < b.scope
		}
		if a.class != b.class {
			return a.class < b.class
		}
		return a.dtype < b.dtype
	})
	sort.SliceStable(out.tokens, func(i, j int) bool {
		a, b := out.tokens[i].key, out.tokens[j].key
		if a.backend != b.backend {
			return a.backend < b.backend
		}
		return a.kind < b.kind
	})
	sort.SliceStable(out.fits, func(i, j int) bool {
		a, b := out.fits[i].key, out.fits[j].key
		if a.backend != b.backend {
			return a.backend < b.backend
		}
		return a.scope < b.scope
	})
	return out
}

func (m *gatewayMetrics) inKernelOOMSnapshotData() []inKernelOOMSnapshot {
	if m == nil {
		return nil
	}
	m.oomMu.Lock()
	defer m.oomMu.Unlock()
	out := make([]inKernelOOMSnapshot, 0, len(m.inKernelOOM))
	for class, st := range m.inKernelOOM {
		if st == nil {
			continue
		}
		out = append(out, inKernelOOMSnapshot{
			class:           class,
			count:           st.count,
			failedBytes:     st.failedBytes,
			lastFailedBytes: st.lastFailedBytes,
			lastSite:        st.lastSite,
		})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].class < out[j].class })
	return out
}

func (m *gatewayMetrics) writeInKernelOOMMetrics(b *strings.Builder) {
	rows := m.inKernelOOMSnapshotData()
	writeHelpType(b, "fak_gateway_in_kernel_oom_total", "LOCAL in-kernel device OOMs and capacity precheck refusals, bucketed by memory class. These are WITNESSED fak-owned resource faults, distinct from provider-side errors.", "counter")
	for _, row := range rows {
		fmt.Fprintf(b, "fak_gateway_in_kernel_oom_total{class=\"%s\"} %d\n", promQuote(row.class), row.count)
	}
	writeHelpType(b, "fak_gateway_in_kernel_oom_failed_bytes_total", "Cumulative bytes requested by local in-kernel allocation OOMs and capacity refusals, bucketed by memory class.", "counter")
	for _, row := range rows {
		fmt.Fprintf(b, "fak_gateway_in_kernel_oom_failed_bytes_total{class=\"%s\"} %d\n", promQuote(row.class), row.failedBytes)
	}
	writeHelpType(b, "fak_gateway_in_kernel_oom_last_failed_bytes", "Most recent failed allocation or refused plan size for each memory class (0 until that class has faulted).", "gauge")
	for _, row := range rows {
		fmt.Fprintf(b, "fak_gateway_in_kernel_oom_last_failed_bytes{class=\"%s\"} %d\n", promQuote(row.class), row.lastFailedBytes)
	}
}

// writeUpstreamErrorMetrics renders the upstream/planner turn-failure family: a count of failed
// turns by kind (stalled / unreachable / oom / rate_limited / auth / forbidden / status_4xx /
// status_5xx / other). It is the cumulative metric twin of the glanceable per-turn `fak-turn …
// FAILED` debug line, so a scrape answers WHY turns failed — including a rate-limit storm vs an
// auth-failure storm — not just that the route returned a 502/504. Snapshot under the lock, then
// render in sorted key order so the scrape is byte-stable.
func (m *gatewayMetrics) writeUpstreamErrorMetrics(b *strings.Builder) {
	m.upstreamErrMu.Lock()
	snap := make(map[string]uint64, len(m.upstreamErrors))
	for k, v := range m.upstreamErrors {
		snap[k] = v
	}
	m.upstreamErrMu.Unlock()
	writeHelpType(b, "fak_gateway_upstream_errors_total", "Upstream/planner turn failures by kind (stalled, unreachable, oom, rate_limited, auth, forbidden, status_4xx, status_5xx, other).", "counter")
	kinds := make([]string, 0, len(snap))
	for k := range snap {
		kinds = append(kinds, k)
	}
	sort.Strings(kinds)
	for _, kind := range kinds {
		fmt.Fprintf(b, "fak_gateway_upstream_errors_total{kind=\"%s\"} %d\n", promQuote(kind), snap[kind])
	}
	writeCounter(b, "fak_gateway_upstream_retries_total", "Upstream retry attempts (the planner's 429/5xx exponential backoff) since process start.", int64(atomic.LoadUint64(&m.upstreamRetries)))
}

// writeInferenceMetrics renders the model-generation family from the live
// accumulators. fak_kernel_*/fak_vdso_* count the ADJUDICATION + dedup fast path,
// which a plain chat/messages turn never touches — so on a box serving real chat
// they read 0 while the model is busy decoding. This family is the missing signal:
// turns served (by finish reason), prompt/completion/cached token totals, the
// cumulative decode wall-clock, and the mean output tokens/sec derived from the two.
// Until the first served turn the counters carry no series and the derived rate is 0,
// so an idle gateway never publishes a phantom throughput. The snapshot is returned
// so the fleet-value block can reuse the same accumulator read (cached tokens, decode
// wall-clock) without re-locking.
func (m *gatewayMetrics) writeInferenceMetrics(b *strings.Builder) inferenceSnapshot {
	snap := m.inferenceSnapshotData()

	writeHelpType(b, "fak_gateway_inference_requests_total", "Model completion turns served by the gateway planner, by finish reason.", "counter")
	reasons := make([]string, 0, len(snap.reqs))
	for r := range snap.reqs {
		reasons = append(reasons, r)
	}
	sort.Strings(reasons)
	for _, r := range reasons {
		fmt.Fprintf(b, "fak_gateway_inference_requests_total{finish_reason=\"%s\"} %d\n", promQuote(r), snap.reqs[r])
	}

	writeCounter(b, "fak_gateway_inference_prompt_tokens_total", "Prompt (input) tokens summed across served model turns.", int64(snap.promptTok))
	writeCounter(b, "fak_gateway_inference_completion_tokens_total", "Completion (generated) tokens summed across served model turns.", int64(snap.complTok))
	writeCounter(b, "fak_gateway_inference_cached_prompt_tokens_total", "Prompt (input) tokens the upstream PROVIDER served from its own prompt cache (cache_read) across served turns, normalized across Anthropic/OpenAI/Gemini. This is provider-side reuse — distinct from the local fak_vdso_*/fak_cache_* caches — and reads 0 on the in-kernel path (no provider).", int64(snap.cachedTok))

	// fak_gateway_provider_cache_local_trust — the #432 acceptance-3 invariant,
	// exported live. The cached-prompt-tokens counter above is PERFORMANCE evidence
	// (cost/latency); it is NEVER local trust. The value is DERIVED from the cachemeta
	// provider_prefix materialization gate (the same gate the kernel uses), so it is
	// structurally 0 — and would flip to 1 only if that proven gate ever regressed to
	// treat a provider cache_read as a serveable local-trust hit.
	providerLocalTrust := 0
	if providerCacheEvidence(snap.cachedTok).CanServe() {
		providerLocalTrust = 1
	}
	writeHelpType(b, "fak_gateway_provider_cache_local_trust", "Whether the upstream PROVIDER's prompt-cache reuse counts as LOCAL TRUST (#432 acceptance 3). Structurally 0: provider cache is performance evidence (cost/latency) only — derived live from the cachemeta provider_prefix materialization gate, not a prose promise. A 1 would mean the trust/performance separation regressed.", "gauge")
	fmt.Fprintf(b, "fak_gateway_provider_cache_local_trust %d\n", providerLocalTrust)

	// Provider prompt-cache HIT rate: the token total above carried no denominator,
	// so a dashboard could see tokens-cached but not how OFTEN a turn hit the provider
	// cache. cached_prompt_hits_total counts served turns with a provider cache read
	// (>0 cached tokens); the ratio is hits/turns, mirroring fak_gateway_vdso_hit_ratio.
	var inferTurns uint64
	for _, n := range snap.reqs {
		inferTurns += n
	}
	writeCounter(b, "fak_gateway_inference_cached_prompt_hits_total", "Served model turns whose prompt got a provider prompt-cache READ (cached tokens > 0). The hit COUNT behind the cached-prompt-tokens total above.", int64(snap.cachedHits))
	writeHelpType(b, "fak_gateway_inference_cached_prompt_hit_ratio", "Fraction of served model turns that hit the provider prompt cache (cached_prompt_hits / turns; 0 until the first turn). The provider-cache analogue of fak_gateway_vdso_hit_ratio.", "gauge")
	cacheHitRatio := 0.0
	if inferTurns > 0 {
		cacheHitRatio = float64(snap.cachedHits) / float64(inferTurns)
	}
	fmt.Fprintf(b, "fak_gateway_inference_cached_prompt_hit_ratio %s\n", promFloat(cacheHitRatio))

	writeHelpType(b, "fak_gateway_inference_duration_seconds_total", "Cumulative wall-clock spent inside the planner generating completions (prefill+decode on the in-kernel path; round-trip on a proxy).", "counter")
	fmt.Fprintf(b, "fak_gateway_inference_duration_seconds_total %s\n", promFloat(snap.decodeSecs))

	writeHelpType(b, "fak_gateway_inference_output_tokens_per_second", "Mean output tokens/sec across served turns (completion tokens / total inference wall-clock; 0 until the first turn).", "gauge")
	tps := 0.0
	if snap.decodeSecs > 0 {
		tps = float64(snap.complTok) / snap.decodeSecs
	}
	fmt.Fprintf(b, "fak_gateway_inference_output_tokens_per_second %s\n", promFloat(tps))

	// Prefill vs decode split (time-to-first-token). The output_tokens_per_second
	// above blends prompt ingest (prefill) with generation (decode) into one mean,
	// which hides the first-request-slow story: a cold prefill dominates the first
	// turn's wall-clock while decode is fast. These two rates separate them, measured
	// ONLY over the turns whose TTFT was observable (the streaming passthrough path);
	// a buffered turn reports no boundary and is excluded so the rates never blend a
	// measured turn with an unmeasured one. Until the first measured turn the
	// denominators are 0 and the rates stay 0 (no phantom throughput).
	writeHelpType(b, "fak_gateway_inference_ttft_turns_total", "Served turns whose time-to-first-token (prefill boundary) was observable — the streaming passthrough path only. The denominator behind the prefill seconds/rate below; buffered turns are excluded.", "counter")
	fmt.Fprintf(b, "fak_gateway_inference_ttft_turns_total %d\n", snap.ttftTurns)
	writeHelpType(b, "fak_gateway_inference_prefill_seconds_total", "Cumulative time-to-first-token wall-clock (prompt ingest + first token) across turns that measured it. Decode wall-clock is fak_gateway_inference_duration_seconds_total minus this over the same turns.", "counter")
	fmt.Fprintf(b, "fak_gateway_inference_prefill_seconds_total %s\n", promFloat(snap.prefillSecs))
	writeHelpType(b, "fak_gateway_inference_prefill_tokens_per_second", "Prefill throughput: prompt (input) tokens ingested per second of TTFT wall-clock, over the turns that measured TTFT (prefill_prompt_tokens / prefill_seconds; 0 until the first measured turn). This is the cold-prefill rate that dominates a slow first request.", "gauge")
	prefillTPS := 0.0
	if snap.prefillSecs > 0 {
		prefillTPS = float64(snap.prefillPromptTok) / snap.prefillSecs
	}
	fmt.Fprintf(b, "fak_gateway_inference_prefill_tokens_per_second %s\n", promFloat(prefillTPS))
	// Decode-only rate: completion tokens over the decode phase (total minus prefill)
	// across the measured turns. This is the steady-state generation speed an operator
	// expects — distinct from output_tokens_per_second, which is diluted by prefill.
	// Falls back to 0 (not the blended rate) when no turn measured TTFT, so the two
	// rates never silently coincide.
	writeHelpType(b, "fak_gateway_inference_decode_tokens_per_second", "Decode throughput: completion (generated) tokens per second of DECODE wall-clock (total inference time minus prefill) over the turns that measured TTFT (0 until the first measured turn). The steady-state generation speed, undiluted by cold prefill — contrast fak_gateway_inference_output_tokens_per_second, which blends both.", "gauge")
	decodeTPS := 0.0
	if snap.measuredDecodeSecs > 0 {
		decodeTPS = float64(snap.measuredComplTok) / snap.measuredDecodeSecs
	}
	fmt.Fprintf(b, "fak_gateway_inference_decode_tokens_per_second %s\n", promFloat(decodeTPS))
	return snap
}

// writeVCacheMetrics emits the fak_vcache_* family: the NET realized provider-cache
// economics (read rebate MINUS write premium) over the session's cumulative cache
// counters, computed by the SAME engine `fak vcache observe` uses
// (vcachegov.ProveTelemetrySavings over one aggregate row), so the live gauge equals
// the offline observe Aggregate on the same totals. It is the write-premium-aware
// companion to fak_gateway_inference_cached_prompt_tokens_total (the read axis): a
// cold-write-dominated session reads NEGATIVE saved / proven=0 until the reads repay
// the writes — the honest break-even the read-only surface cannot show. Every value is
// OBSERVED (provider-relayed counters); a hit is a realized rebate, never local trust.
// Until a turn carries provider cache activity it emits nothing (no phantom series).
func (m *gatewayMetrics) writeVCacheMetrics(b *strings.Builder) {
	snap := m.inferenceSnapshotData()
	if snap.cachedTok == 0 && snap.cacheCreateTok == 0 {
		return
	}
	proof := vcacheProofFromCounters(snap.promptTok, snap.cachedTok, snap.cacheCreateTok)
	writeCounter(b, "fak_vcache_cache_creation_tokens_total", "OBSERVED (provider-relayed): cumulative cache_creation_input_tokens — the provider-cache WRITE axis, companion to fak_gateway_inference_cached_prompt_tokens_total (the READ axis). Net saving = read rebate minus this write premium.", int64(snap.cacheCreateTok))
	writeHelpType(b, "fak_vcache_baseline_token_equiv", "OBSERVED-derived: input-token-equivalents the session WOULD have cost with NO provider cache (every prompt token at 1x).", "gauge")
	fmt.Fprintf(b, "fak_vcache_baseline_token_equiv %s\n", promFloat(proof.BaselineTokenEquiv))
	writeHelpType(b, "fak_vcache_actual_token_equiv", "OBSERVED-derived: input-token-equivalents the session actually cost under provider caching (read at 0.1x, unsplit write at the 5m 1.25x tier).", "gauge")
	fmt.Fprintf(b, "fak_vcache_actual_token_equiv %s\n", promFloat(proof.ActualTokenEquiv))
	writeHelpType(b, "fak_vcache_saved_token_equiv", "NET realized provider-cache saving in input-token-equivalents (baseline minus actual). NEGATIVE on a cold-write-dominated session until reads repay writes. Same engine and number as `fak vcache observe`.", "gauge")
	fmt.Fprintf(b, "fak_vcache_saved_token_equiv %s\n", promFloat(proof.SavedTokenEquiv))
	writeHelpType(b, "fak_vcache_saved_ratio", "NET saved-token-equiv as a fraction of the uncached baseline (saved_pct/100).", "gauge")
	fmt.Fprintf(b, "fak_vcache_saved_ratio %s\n", promFloat(proof.SavedPct/100))
	hit := 0.0
	if proof.BaselineTokenEquiv > 0 {
		hit = proof.CacheReadTokens / proof.BaselineTokenEquiv
	}
	writeHelpType(b, "fak_vcache_hit_rate", "OBSERVED: cache_read share of the uncached baseline (equals `fak vcache observe` Report.HitRate).", "gauge")
	fmt.Fprintf(b, "fak_vcache_hit_rate %s\n", promFloat(hit))
	mult := 0.0
	if proof.ActualTokenEquiv > 0 {
		mult = proof.BaselineTokenEquiv / proof.ActualTokenEquiv
	}
	writeHelpType(b, "fak_vcache_multiplier", "OBSERVED-derived: baseline/actual token-equiv (equals `fak vcache observe` Report.Multiplier). 1.0 = no net saving.", "gauge")
	fmt.Fprintf(b, "fak_vcache_multiplier %s\n", promFloat(mult))
	proven := 0
	if proof.Status == vcachegov.ProofProven {
		proven = 1
	}
	writeHelpType(b, "fak_vcache_proven", "1 when the session's observed cache reads repaid the write premium (NET positive); else 0 (cold/write-dominated). The honest break-even gate.", "gauge")
	fmt.Fprintf(b, "fak_vcache_proven %d\n", proven)
}

// writeFleetValueMetrics renders the hero-axis KPIs the live gateway can derive from
// its own counters — the per-node ingredients of fak's product headline (agent-fleet
// serving efficiency, HERO-BENCHMARK-2026-06-21.md). They are deliberately the LIVE
// cumulative mechanism counts, not an A/B headline: the comparable "turns saved vs a
// no-kernel baseline" number is only honest when both arms ran the same workload
// (LIVE-RESULTS.md / TICKETS T2), which a single serving node cannot witness. So this
// reports what THIS process measurably did:
//
//   - turns_saved_total{mechanism} — vDSO dedup hits (an engine round-trip not taken)
//     and grammar repairs (a retry turn the baseline spends re-emitting a malformed
//     call). These are the two turn-saving levers LIVE-RESULTS attributes turns to.
//   - context_pollutions_blocked_total — untrusted/poisoned tool-result payloads the
//     context-MMU paged out before they entered the model's context window (the live
//     "context saved" KPI; on a weak model each is also a derailment averted).
//   - context_pollution_rate — quarantines per kernel submission, the same ratio the
//     A/B report computes as context_pollution_rate (internal/bench).
//   - agent_seconds_total — cumulative agent-serving wall-clock (generation + kernel
//     op adjudication/dispatch), the denominator for a per-agent-second view.
//
// The KV-reuse "context saved" KPI is fak_gateway_inference_cached_prompt_tokens_total
// in the inference family above; it is not re-emitted here to avoid a duplicate series.
func writeFleetValueMetrics(b *strings.Builder, c kernel.Counters, agentSeconds float64) {
	writeHelpType(b, "fak_gateway_turns_saved_total", "Agent turns the kernel saved the served fleet, by mechanism: vdso_dedup = a duplicate read served from the fast path with no engine round-trip; grammar_repair = a malformed tool call repaired in-syscall instead of costing the baseline a retry turn. The comparable 'turns saved' headline (LIVE-RESULTS.md) is this measured against a no-kernel baseline arm; this is the live cumulative count of each mechanism firing.", "counter")
	fmt.Fprintf(b, "fak_gateway_turns_saved_total{mechanism=\"vdso_dedup\"} %d\n", c.VDSOHits)
	fmt.Fprintf(b, "fak_gateway_turns_saved_total{mechanism=\"grammar_repair\"} %d\n", c.Transforms)

	writeCounter(b, "fak_gateway_context_pollutions_blocked_total", "Untrusted/poisoned tool-result payloads the context-MMU paged out before they reached the model's context window — each a context-window pollution (and on a weak model a derailment) prevented. The live 'context saved' KPI.", c.Quarantines)

	writeHelpType(b, "fak_gateway_context_pollution_rate", "Quarantines per kernel submission, computed live over this process's submissions (the A/B report's context_pollution_rate; 0 when nothing has been submitted).", "gauge")
	pollRate := 0.0
	if c.Submits > 0 {
		pollRate = float64(c.Quarantines) / float64(c.Submits)
	}
	fmt.Fprintf(b, "fak_gateway_context_pollution_rate %s\n", promFloat(pollRate))

	writeHelpType(b, "fak_gateway_agent_seconds_total", "Cumulative wall-clock the gateway spent doing agent work: model generation (inference) plus kernel operation adjudication/dispatch (syscall, adjudicate, admit). The denominator for a live tokens/turns-per-agent-second view.", "counter")
	fmt.Fprintf(b, "fak_gateway_agent_seconds_total %s\n", promFloat(agentSeconds))
}

func (m *gatewayMetrics) snapshot() ([]httpMetricSnapshot, []operationMetricSnapshot) {
	m.mu.Lock()
	defer m.mu.Unlock()
	httpRows := make([]httpMetricSnapshot, 0, len(m.http))
	for k, v := range m.http {
		httpRows = append(httpRows, httpMetricSnapshot{key: k, val: v.snapshot()})
	}
	sort.Slice(httpRows, func(i, j int) bool {
		a, b := httpRows[i].key, httpRows[j].key
		if a.route != b.route {
			return a.route < b.route
		}
		if a.method != b.method {
			return a.method < b.method
		}
		return a.status < b.status
	})
	opRows := make([]operationMetricSnapshot, 0, len(m.operations))
	for k, v := range m.operations {
		opRows = append(opRows, operationMetricSnapshot{key: k, val: v.snapshot()})
	}
	sort.Slice(opRows, func(i, j int) bool {
		a, b := opRows[i].key, opRows[j].key
		if a.operation != b.operation {
			return a.operation < b.operation
		}
		if a.verdict != b.verdict {
			return a.verdict < b.verdict
		}
		if a.reason != b.reason {
			return a.reason < b.reason
		}
		if a.disposition != b.disposition {
			return a.disposition < b.disposition
		}
		return a.by < b.by
	})
	return httpRows, opRows
}

// writeCompactionMetrics renders the history-compaction family, split along what fak CONTROLS
// versus what it can only OBSERVE — keep the two apart or a provider-side miss reads as a fak
// bug it cannot support:
//   - WITNESSED (fak authored): attempts{fired|bailed|off}, bail_reason, dropped_turns, shed.
//     A turn counts `fired` only when the protected-prefix bytes it shipped were byte-identical
//     to the input (else it bails to `prefix_mismatch`). These describe what fak SENT.
//   - OBSERVED (provider-reported, relayed verbatim): cache_read_tokens / post_fire_cache_read.
//     fak attributes nothing to itself here; it forwards the upstream's number.
//
// The single fak-fault signal is bail_reason{reason="prefix_mismatch"}>0. A cratered cache_read
// while fires climb is NOT that bug — the shipped prefix was byte-identical, so the provider
// missed for a reason fak does not control (cache TTL expiry, eviction, or the client moving its
// own breakpoint). Reading the crater as "the splice broke the cache" is the conflation this
// split exists to prevent.
func (m *gatewayMetrics) writeCompactionMetrics(b *strings.Builder) {
	snap := m.compactionSnapshotData()

	writeHelpType(b, "fak_gateway_compaction_attempts_total",
		"WITNESSED (fak authored): Anthropic history-compaction attempts by outcome: fired (body rewritten, protected prefix shipped byte-identical), bailed (returned identity), off (budget unset).", "counter")
	for _, o := range []string{"fired", "bailed", "off"} { // stable order; emit at 0 so the panel exists pre-first-fire
		fmt.Fprintf(b, "fak_gateway_compaction_attempts_total{outcome=%q} %d\n", o, snap.attempts[o])
	}

	writeHelpType(b, "fak_gateway_compaction_bail_reason_total",
		"WITNESSED (fak authored): why a compaction attempt bailed to identity (closed set: under_budget|non_json|no_messages_key|too_few_msgs|no_breakpoint|window_no_drop|splice_failed|redecode_failed|prefix_mismatch). prefix_mismatch>0 is the ONLY fak-fault cache signal and must stay 0; splice_failed/redecode_failed are splice bugs and must stay 0 too.", "counter")
	rs := make([]string, 0, len(snap.bailReasons))
	for r := range snap.bailReasons {
		rs = append(rs, r)
	}
	sort.Strings(rs)
	for _, r := range rs {
		fmt.Fprintf(b, "fak_gateway_compaction_bail_reason_total{reason=%q} %d\n", promQuote(r), snap.bailReasons[r])
	}

	writeCounter(b, "fak_gateway_compaction_dropped_turns_total", "WITNESSED (fak authored): whole messages stubbed out across all fires.", int64(snap.dropped))
	writeCounter(b, "fak_gateway_compaction_shed_tokens_total", "WITNESSED (fak authored): estimated tokens fak removed from the outbound body across all fires (same ~4ch/token currency as the budget and provider input_tokens). What fak SENT — not a claim about what the provider billed.", int64(snap.shed))
	writeCounter(b, "fak_gateway_compaction_cache_read_tokens_total", "OBSERVED (provider-reported, relayed verbatim): cumulative cache_read_input_tokens on compacted turns. Pair with shed_tokens to see the net effect; attribute nothing to fak from it alone — fak only guarantees the prefix it shipped was byte-identical (see attempts{fired} with prefix_mismatch=0).", int64(snap.cacheReads))
	writeHelpType(b, "fak_gateway_compaction_post_fire_cache_read_tokens",
		"OBSERVED (provider-reported): cache_read_input_tokens on the MOST RECENT compacted turn. If this craters while fires climb, the prefix fak shipped was still byte-identical (witnessed by fired with prefix_mismatch=0), so the provider did not reuse it for a reason fak does NOT control: cache TTL expiry, eviction, or the client moving its own breakpoint. Only bail_reason{reason=\"prefix_mismatch\"}>0 is fak's bug.", "gauge")
	fmt.Fprintf(b, "fak_gateway_compaction_post_fire_cache_read_tokens %s\n", promFloat(snap.lastCacheRd))

	// The INBOUND tool-floor prune family (the tools[] twin of the compaction shed above).
	// WITNESSED: how many unreachable tool DEFINITIONS fak dropped from the advertised surface,
	// a pure uncached-token saving the pruner makes only after the cache_control breakpoint so it
	// never bursts the cache. Both stay 0 on the dominant Claude Code path (its single breakpoint
	// sits on the LAST tool, so nothing is droppable) — which, before these rows existed, was the
	// invisible fact: the prune result was discarded with no counter.
	pruneTurns, pruneCount := m.inboundToolPruneSnapshot()
	writeCounter(b, "fak_gateway_inbound_tools_pruned_total", "WITNESSED (fak authored): cumulative unreachable tool DEFINITIONS dropped from the outbound tools[] across the session. A pure uncached-token saving — the pruner drops only tools after the cache_control breakpoint and re-proves the protected prefix is byte-identical, so a counted prune never bursts the upstream cache.", int64(pruneCount))
	writeCounter(b, "fak_gateway_inbound_tools_prune_turns_total", "WITNESSED (fak authored): turns on which at least one unreachable tool def was pruned from tools[]. Zero on a harness (e.g. Claude Code) whose single cache_control breakpoint sits on the LAST tool, since nothing is then droppable.", int64(pruneTurns))
}

// resetShadowSnapshot is a lock-free copy of the resetScore SHADOW accumulators for rendering.
type resetShadowSnapshot struct {
	reasons   map[string]uint64
	recommend uint64
	lastScore float64
}

func (m *gatewayMetrics) resetShadowSnapshotData() resetShadowSnapshot {
	out := resetShadowSnapshot{reasons: map[string]uint64{}}
	if m == nil {
		return out
	}
	m.resetShadowMu.Lock()
	for k, v := range m.resetShadowReasons {
		out.reasons[k] = v
	}
	out.recommend = m.resetShadowRecommend
	out.lastScore = m.resetShadowLastScore
	m.resetShadowMu.Unlock()
	return out
}

// writeResetShadowMetrics renders the per-session resetScore SHADOW family (#792). The reset
// policy recommends cut-vs-reset on the compaction crossover; in SHADOW mode it acts on NOTHING,
// so this surface is purely "what it WOULD recommend" — recommendations_total is the count of
// turns the verdict said reset, and it stays a recommendation until shadow evidence supports
// enabling reset. The verdict is WITNESSED (fak's policy); the cache ratios it scored are OBSERVED
// (provider-reported), the same split the compaction family keeps. Every reason bucket is emitted
// at 0 so the panel exists before the first compacted turn.
func (m *gatewayMetrics) writeResetShadowMetrics(b *strings.Builder) {
	snap := m.resetShadowSnapshotData()
	writeHelpType(b, "fak_gateway_compaction_reset_shadow_total",
		"WITNESSED (fak policy, recommend-only): compacted turns scored by the resetScore SHADOW policy, by reason (healthy_cache|stale_prefix|cache_decay|cooldown|unknown_provider). SHADOW mode acts on nothing — this is what the cut-vs-reset crossover WOULD recommend, not a reset that happened. The cache ratios scored are OBSERVED (provider-reported).", "counter")
	for _, reason := range []string{
		string(ResetReasonHealthy), string(ResetReasonStalePrefix), string(ResetReasonDecay),
		string(ResetReasonCooldown), string(ResetReasonUnknown),
	} { // stable order; emit at 0 so the panel exists pre-first-turn
		fmt.Fprintf(b, "fak_gateway_compaction_reset_shadow_total{reason=%q} %d\n", reason, snap.reasons[reason])
	}
	writeCounter(b, "fak_gateway_compaction_reset_recommendations_total",
		"WITNESSED (fak policy, recommend-only): compacted turns whose resetScore SHADOW verdict was ShouldReset. In SHADOW mode NONE were acted on — the count is the reset pressure the policy held back, cut-by-default.", int64(snap.recommend))
	writeHelpType(b, "fak_gateway_compaction_reset_score",
		"The most recent compacted turn's 0..1 resetScore reset-pressure magnitude (0 = clearly keep cutting, 1 = clearly reset). Reported even when the cooldown holds the recommendation, so the building pressure is visible. Advisory: nothing acts on it in SHADOW mode.", "gauge")
	fmt.Fprintf(b, "fak_gateway_compaction_reset_score %s\n", promFloat(snap.lastScore))
}

func writeHelpType(b *strings.Builder, name, help, typ string) {
	fmt.Fprintf(b, "# HELP %s %s\n", name, help)
	fmt.Fprintf(b, "# TYPE %s %s\n", name, typ)
}

func writeCounter(b *strings.Builder, name, help string, n int64) {
	writeHelpType(b, name, help, "counter")
	fmt.Fprintf(b, "%s %d\n", name, n)
}

func writeHistogram(b *strings.Builder, name, baseLabels string, s latencySnapshot) {
	for i, le := range gatewayLatencyBuckets {
		fmt.Fprintf(b, "%s_bucket{%s,le=\"%s\"} %d\n", name, baseLabels, promQuote(promFloat(le)), s.buckets[i])
	}
	fmt.Fprintf(b, "%s_bucket{%s,le=\"+Inf\"} %d\n", name, baseLabels, s.count)
	fmt.Fprintf(b, "%s_sum{%s} %s\n", name, baseLabels, promFloat(s.sum))
	fmt.Fprintf(b, "%s_count{%s} %d\n", name, baseLabels, s.count)
}

func promFloat(v float64) string {
	return strconv.FormatFloat(v, 'g', -1, 64)
}

func promQuote(s string) string {
	s = strings.ReplaceAll(s, `\`, `\\`)
	s = strings.ReplaceAll(s, "\n", `\n`)
	s = strings.ReplaceAll(s, `"`, `\"`)
	return s
}

type statusRecorder struct {
	http.ResponseWriter
	status int
	bytes  int64
}

// WriteHeader records the FIRST status code written (later calls are ignored) and
// forwards it to the wrapped ResponseWriter, so the metrics middleware can label the
// request by the status actually sent.
func (r *statusRecorder) WriteHeader(status int) {
	if r.status != 0 {
		return
	}
	r.status = status
	r.ResponseWriter.WriteHeader(status)
}

// Write forwards the body to the wrapped ResponseWriter (defaulting the status to 200
// on the first write) and accumulates the byte count for the request log.
func (r *statusRecorder) Write(p []byte) (int, error) {
	if r.status == 0 {
		r.WriteHeader(http.StatusOK)
	}
	n, err := r.ResponseWriter.Write(p)
	r.bytes += int64(n)
	return n, err
}

// Flush defaults the status to 200 if unset and flushes the wrapped ResponseWriter
// when it implements http.Flusher, preserving streaming (SSE) behavior through the
// recorder.
func (r *statusRecorder) Flush() {
	if r.status == 0 {
		r.WriteHeader(http.StatusOK)
	}
	if f, ok := r.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

func (s *Server) withMetrics(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		atomic.AddInt64(&s.metrics.inflight, 1)
		defer atomic.AddInt64(&s.metrics.inflight, -1)
		route := routeForMetrics(r.URL.Path)
		// Register the request as live so scrapes can see it WHILE it runs, not
		// only after it completes into the latency histogram. route is derived from
		// the path up front (routeForMetrics is pure), matching the completion-time
		// label below.
		if route != "/metrics" {
			liveID := s.metrics.beginInflight(route, start)
			defer s.metrics.endInflight(liveID)
		}
		traceID := ensureHTTPTrace(s, w, r)
		rec := &statusRecorder{ResponseWriter: w}
		next.ServeHTTP(rec, r)
		status := rec.status
		if status == 0 {
			status = http.StatusOK
		}
		dur := time.Since(start)
		s.metrics.observeHTTP(route, r.Method, status, dur)
		if current := requestTraceID(r); current != "" {
			traceID = current
		}
		s.logHTTPRequest(r, route, status, dur, rec.bytes, traceID)
	})
}

func (s *Server) logHTTPRequest(r *http.Request, route string, status int, dur time.Duration, bytes int64, traceID string) {
	if s == nil || s.logf == nil {
		return
	}
	ev := map[string]any{
		"event":       "gateway_http_request",
		"method":      r.Method,
		"route":       route,
		"path":        r.URL.Path,
		"status":      status,
		"duration_ms": float64(dur.Microseconds()) / 1000.0,
		"bytes":       bytes,
	}
	if traceID != "" {
		ev["trace_id"] = traceID
	}
	if ua := r.Header.Get("User-Agent"); ua != "" {
		ev["user_agent"] = ua
	}
	b, err := json.Marshal(ev)
	if err != nil {
		return
	}
	s.logf("%s", b)
}

func (s *Server) logGatewayOperation(operation, traceID, tool string, v WireVerdict, opErr error, dur time.Duration) {
	if s == nil || s.logf == nil {
		return
	}
	verdict := v.Kind
	if opErr != nil && verdict == "" {
		verdict = "ERROR"
	}
	ev := map[string]any{
		"event":       "gateway_operation",
		"operation":   operation,
		"tool":        tool,
		"trace_id":    strings.TrimSpace(traceID),
		"verdict":     verdict,
		"duration_ms": float64(dur.Microseconds()) / 1000.0,
	}
	if v.Reason != "" {
		ev["reason"] = v.Reason
	}
	if v.By != "" {
		ev["by"] = v.By
	}
	if v.Disposition != "" {
		ev["disposition"] = v.Disposition
	}
	if opErr != nil {
		ev["error"] = opErr.Error()
	}
	b, err := json.Marshal(ev)
	if err != nil {
		return
	}
	s.logf("%s", b)
}

const traceHeader = "X-Trace-Id"

func ensureHTTPTrace(s *Server, w http.ResponseWriter, r *http.Request) string {
	traceID := requestTraceID(r)
	if traceID == "" && s != nil {
		traceID = s.traceFor("")
		r.Header.Set(traceHeader, traceID)
	}
	if traceID != "" {
		w.Header().Set(traceHeader, traceID)
	}
	return traceID
}

func requestTraceID(r *http.Request) string {
	if r == nil {
		return ""
	}
	return strings.TrimSpace(r.Header.Get(traceHeader))
}

func (s *Server) useHTTPTrace(w http.ResponseWriter, r *http.Request, preferred string) string {
	traceID := strings.TrimSpace(preferred)
	if traceID == "" {
		traceID = requestTraceID(r)
	}
	if traceID == "" && s != nil {
		traceID = s.traceFor("")
	}
	if traceID != "" {
		r.Header.Set(traceHeader, traceID)
		w.Header().Set(traceHeader, traceID)
	}
	return traceID
}

func routeForMetrics(path string) string {
	switch path {
	case "/v1/chat/completions", "/v1/responses", "/v1/messages", "/v1/messages/count_tokens",
		"/v1/fak/syscall", "/v1/fak/adjudicate", "/v1/fak/admit",
		"/v1/fak/changes", "/v1/fak/session/changes", "/v1/fak/revoke", "/v1/fak/policy/reload",
		"/v1/fak/trace/reset", "/v1/models", "/mcp", "/healthz", "/metrics",
		"/debug/vars":
		return path
	default:
		if strings.HasPrefix(path, "/v1/fak/") {
			return "/v1/fak/*"
		}
		if strings.HasPrefix(path, "/v1beta/") {
			return "/v1beta/*"
		}
		return "other"
	}
}
