package gateway

import (
	"encoding/json"
	"fmt"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/anthony-chaudhary/fak/internal/blob"
	"github.com/anthony-chaudhary/fak/internal/kernel"
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
	inferDecodeSecs   float64
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

type latencyCounter struct {
	count   uint64
	sum     float64
	buckets []uint64
}

func newGatewayMetrics(now time.Time) *gatewayMetrics {
	return &gatewayMetrics{
		start:       now,
		http:        map[httpMetricKey]*latencyCounter{},
		operations:  map[operationMetricKey]*latencyCounter{},
		inflightReq: map[uint64]inflightEntry{},
	}
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
	m.inferenceMu.Unlock()
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

// observeInference records one served model-generation turn: its token accounting,
// why decode stopped, and the wall-clock the planner spent producing it. promptTok /
// complTok / cachedTok come straight from the planner's reported Usage; dur is the
// time spent inside planner.Complete. Negative/zero values are ignored so a planner
// that omits a count never corrupts the running totals. This is the signal that makes
// a busy gateway look busy: fak_kernel_*/fak_vdso_* stay 0 on a pure chat workload
// (no syscall, no fast-path lookup), so without this family every panel reads 0 while
// the box is in fact decoding tokens.
func (m *gatewayMetrics) observeInference(promptTok, complTok, cachedTok int, finishReason string, dur time.Duration) {
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
	if dur > 0 {
		m.inferDecodeSecs += dur.Seconds()
	}
	m.inferenceMu.Unlock()
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
	if r.Method != http.MethodGet {
		writeErr(w, http.StatusMethodNotAllowed, "use GET")
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
	writeCounter(&b, "fak_kernel_admitted_total", "Kernel result admissions that were accepted or transformed.", c.Admitted)
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
	inf := m.writeInferenceMetrics(&b)

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

type inferenceSnapshot struct {
	reqs       map[string]uint64
	promptTok  uint64
	complTok   uint64
	cachedTok  uint64
	cachedHits uint64
	decodeSecs float64
}

func (m *gatewayMetrics) inferenceSnapshotData() inferenceSnapshot {
	m.inferenceMu.Lock()
	defer m.inferenceMu.Unlock()
	reqs := make(map[string]uint64, len(m.inferReqs))
	for k, v := range m.inferReqs {
		reqs[k] = v
	}
	return inferenceSnapshot{
		reqs:       reqs,
		promptTok:  m.inferPromptTokens,
		complTok:   m.inferComplTokens,
		cachedTok:  m.inferCachedTokens,
		cachedHits: m.inferCachedHits,
		decodeSecs: m.inferDecodeSecs,
	}
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
	return snap
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

func (r *statusRecorder) WriteHeader(status int) {
	if r.status != 0 {
		return
	}
	r.status = status
	r.ResponseWriter.WriteHeader(status)
}

func (r *statusRecorder) Write(p []byte) (int, error) {
	if r.status == 0 {
		r.WriteHeader(http.StatusOK)
	}
	n, err := r.ResponseWriter.Write(p)
	r.bytes += int64(n)
	return n, err
}

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

func useHTTPTrace(w http.ResponseWriter, r *http.Request, preferred string) string {
	traceID := strings.TrimSpace(preferred)
	if traceID == "" {
		traceID = requestTraceID(r)
	}
	if traceID != "" {
		r.Header.Set(traceHeader, traceID)
		w.Header().Set(traceHeader, traceID)
	}
	return traceID
}

func routeForMetrics(path string) string {
	switch path {
	case "/v1/chat/completions", "/v1/messages", "/v1/messages/count_tokens",
		"/v1/fak/syscall", "/v1/fak/adjudicate", "/v1/fak/admit",
		"/v1/fak/changes", "/v1/fak/revoke", "/v1/fak/policy/reload",
		"/v1/fak/trace/reset", "/v1/models", "/mcp", "/healthz", "/metrics",
		"/debug/vars":
		return path
	default:
		if strings.HasPrefix(path, "/v1/fak/") {
			return "/v1/fak/*"
		}
		return "other"
	}
}
