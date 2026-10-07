// Package enginestep is the bounded, always-on step recorder for the native
// in-kernel serving loop: it makes every cycle of continuous batching observable
// end to end (admission wait -> device wait -> cohort forming -> prefix lookup ->
// chunked prefill -> prefix admit -> each decode step -> sampling -> request done).
//
// It is instrumentation only. The serving loop calls the Observe* methods on the
// process-wide Default recorder; the gateway renders it on /metrics as the
// fak_engine_* family and serves a bounded JSON/compact view at
// /v1/fak/observation/engine so agents read the cycle without a raw scrape dump.
//
// Memory is fixed at construction: a small set of fixed-bucket histograms and a
// ring of the most recent step records. Each Observe is O(buckets) integer work
// under one short mutex, which is negligible against a millisecond-scale decode
// forward. Metric vocabulary is adapted (not ported) from vLLM's v1 scheduler
// stats (Apache-2.0): iteration tokens, running/waiting, per-phase request time;
// the speculative families adapt vLLM's SpecDecodingStats (drafts, draft tokens,
// accepted tokens).
package enginestep

import (
	"fmt"
	"io"
	"math"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

// Phase names one stage of a request's trip through the native serving loop.
type Phase string

const (
	// PhaseAdmissionWait is time between gateway admission enqueue and grant.
	PhaseAdmissionWait Phase = "admission_wait"
	// PhaseDeviceWait is time a request waits for the planner's device lock.
	PhaseDeviceWait Phase = "device_wait"
	// PhaseCohortWait is time a request waits in the decode coalescer before its
	// cohort starts.
	PhaseCohortWait Phase = "cohort_wait"
	// PhasePrefixLookup is the KV prefix-cache lookup before prefill.
	PhasePrefixLookup Phase = "prefix_lookup"
	// PhasePrefill is the (possibly chunked) prompt prefill forward.
	PhasePrefill Phase = "prefill"
	// PhasePrefixAdmit is admission of the prefilled prefix into the KV cache.
	PhasePrefixAdmit Phase = "prefix_admit"
	// PhaseDecode is one request's whole decode loop (all of its steps).
	PhaseDecode Phase = "decode"
	// PhaseSample is one token's sampling (logits -> token id).
	PhaseSample Phase = "sample"
	// PhaseRequest is one request end to end inside the planner.
	PhaseRequest Phase = "request"
)

// Phases is the closed, ordered phase vocabulary (render and dashboard order).
var Phases = []Phase{
	PhaseAdmissionWait, PhaseDeviceWait, PhaseCohortWait, PhasePrefixLookup,
	PhasePrefill, PhasePrefixAdmit, PhaseDecode, PhaseSample, PhaseRequest,
}

// Decode step paths.
const (
	PathSerial  = "serial"  // one lane advanced by Session.Step
	PathBatched = "batched" // N lanes advanced by one BatchSession.StepBatchActive
	// PathSpeculative is one draft-verify round: one target forward that commits
	// the accepted draft prefix plus the bonus token.
	PathSpeculative = "speculative"
)

// Paths is the closed decode-path vocabulary.
var Paths = []string{PathSerial, PathBatched, PathSpeculative}

// Record kinds in the recent-step ring.
const (
	KindDecodeStep   = "decode_step"
	KindPrefillChunk = "prefill_chunk"
	KindCohort       = "cohort"
	KindPhase        = "phase"
)

// Metric family names emitted by WritePrometheus. Dashboards and readers pin
// against these; renaming one is a contract change.
const (
	MetricPhaseSeconds       = "fak_engine_phase_seconds"
	MetricDecodeStepSeconds  = "fak_engine_decode_step_seconds"
	MetricDecodeStepLanes    = "fak_engine_decode_step_lanes"
	MetricDecodeTokensTotal  = "fak_engine_decode_tokens_total"
	MetricCohortSize         = "fak_engine_cohort_size"
	MetricCoalesceQueueDepth = "fak_engine_coalesce_queue_depth"
	MetricPrefillTokensTotal = "fak_engine_prefill_tokens_total"
	MetricPrefillChunksTotal = "fak_engine_prefill_chunks_total"
	MetricPrefixMatchedTotal = "fak_engine_prefix_matched_tokens_total"
	// MetricPrefixQueriedTotal is the hit-rate denominator: prompt tokens looked
	// up in the prefix cache, hit or miss (vLLM prefix_cache_queries). Hit rate is
	// prefix_matched / prefix_queried.
	MetricPrefixQueriedTotal = "fak_engine_prefix_queried_tokens_total"
	// MetricPreemptionsTotal counts running lanes preempted under KV pressure, by
	// reason (swap / recompute / gpudirect_swap); vLLM vllm:num_preemptions_total.
	MetricPreemptionsTotal = "fak_engine_preemptions_total"
	// MetricIterationTokens is tokens processed per engine step: lanes for one
	// decode step, chunk tokens for one prefill chunk; vLLM vllm:iteration_tokens_total.
	MetricIterationTokens   = "fak_engine_iteration_tokens"
	MetricRequestsActive    = "fak_engine_requests_active"
	MetricLastStepTimestamp = "fak_engine_last_step_timestamp_seconds"

	MetricSpecRoundsTotal         = "fak_engine_spec_rounds_total"
	MetricSpecDraftTokensTotal    = "fak_engine_spec_draft_tokens_total"
	MetricSpecAcceptedTokensTotal = "fak_engine_spec_accepted_tokens_total"
	MetricSpecAcceptedPerRound    = "fak_engine_spec_accepted_per_round"
)

// MetricFamilies lists every family WritePrometheus emits, in render order.
var MetricFamilies = []string{
	MetricPhaseSeconds, MetricDecodeStepSeconds, MetricDecodeStepLanes,
	MetricDecodeTokensTotal, MetricCohortSize, MetricCoalesceQueueDepth,
	MetricPrefillTokensTotal, MetricPrefillChunksTotal, MetricPrefixMatchedTotal,
	MetricPrefixQueriedTotal, MetricPreemptionsTotal, MetricIterationTokens,
	MetricRequestsActive, MetricLastStepTimestamp,
	MetricSpecRoundsTotal, MetricSpecDraftTokensTotal, MetricSpecAcceptedTokensTotal,
	MetricSpecAcceptedPerRound,
}

var (
	secondsBuckets = []float64{0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60}
	lanesBuckets   = []float64{1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 64}
	acceptBuckets  = []float64{0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 16}
	// iterTokBuckets mirror vLLM's iteration_tokens_total cudagraph-ish ladder,
	// spanning one decode lane up to a large prefill chunk.
	iterTokBuckets = []float64{1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384}
)

// Preemption reasons (the closed label vocabulary of MetricPreemptionsTotal).
const (
	PreemptSwap          = "swap"
	PreemptRecompute     = "recompute"
	PreemptGPUDirectSwap = "gpudirect_swap"
)

// PreemptionReasons is the closed preemption reason vocabulary, in render order.
var PreemptionReasons = []string{PreemptSwap, PreemptRecompute, PreemptGPUDirectSwap}

// DefaultRingSize is the number of recent step records Default keeps.
const DefaultRingSize = 512

// Default is the process-wide recorder the serving loop feeds and the gateway renders.
var Default = New(DefaultRingSize)

// PhaseObserver is an optional second sink for one completed phase or decode step. It
// exists so the SUB-PLANNER step timings this recorder already measures can also reach a
// label-keyed metric surface, without this package importing one: the mapping from
// Phase to whatever vocabulary a downstream consumer wants belongs to the consumer.
type PhaseObserver func(Phase, time.Duration)

// StepObserver is the same hook for one decode step, which the Phase vocabulary cannot
// name (PhaseDecode is a request's WHOLE decode loop; one forward is the step).
type StepObserver func(path string, lanes int, d time.Duration)

var observers struct {
	sync.RWMutex
	phase PhaseObserver
	step  StepObserver
}

// SetPhaseObserver installs the process phase observer. Passing nil detaches it. The
// hook is a fan-out on a path that already holds the recorder lock's work, and it is a
// nil check when unset, so the decode loop pays nothing until an observer exists.
func SetPhaseObserver(fn PhaseObserver) {
	observers.Lock()
	observers.phase = fn
	observers.Unlock()
}

// SetStepObserver installs the process decode-step observer. Passing nil detaches it.
func SetStepObserver(fn StepObserver) {
	observers.Lock()
	observers.step = fn
	observers.Unlock()
}

func phaseObservers() PhaseObserver {
	observers.RLock()
	defer observers.RUnlock()
	return observers.phase
}

func stepObservers() StepObserver {
	observers.RLock()
	defer observers.RUnlock()
	return observers.step
}

// StepRecord is one entry of the recent-step ring.
type StepRecord struct {
	Seq        uint64 `json:"seq"`
	AtUnixNano int64  `json:"at_unix_nano"`
	Kind       string `json:"kind"`
	Phase      string `json:"phase,omitempty"`
	Path       string `json:"path,omitempty"`
	Lanes      int    `json:"lanes,omitempty"`
	Tokens     int    `json:"tokens,omitempty"`
	DurationNS int64  `json:"duration_ns"`
	// Proposed and Accepted are set on speculative decode steps only.
	Proposed int `json:"proposed,omitempty"`
	Accepted int `json:"accepted,omitempty"`
}

type histogram struct {
	bounds []float64
	counts []uint64 // per-bucket (non-cumulative); len(bounds)+1 with +Inf last
	count  uint64
	sum    float64
	max    float64
}

func newHistogram(bounds []float64) *histogram {
	return &histogram{bounds: bounds, counts: make([]uint64, len(bounds)+1)}
}

func (h *histogram) observe(v float64) {
	if math.IsNaN(v) || v < 0 {
		v = 0
	}
	i := sort.SearchFloat64s(h.bounds, v)
	h.counts[i]++
	h.count++
	h.sum += v
	if v > h.max {
		h.max = v
	}
}

// quantile returns the upper bound of the bucket holding quantile q (an
// over-estimate by at most one bucket width, clamped to the observed max).
func (h *histogram) quantile(q float64) float64 {
	if h.count == 0 {
		return 0
	}
	rank := uint64(math.Ceil(q * float64(h.count)))
	if rank == 0 {
		rank = 1
	}
	var cum uint64
	for i, c := range h.counts {
		cum += c
		if cum >= rank {
			if i < len(h.bounds) {
				return math.Min(h.bounds[i], h.max)
			}
			return h.max
		}
	}
	return h.max
}

func (h *histogram) write(w io.Writer, name, labels string) {
	sep := ""
	if labels != "" {
		sep = ","
	}
	var cum uint64
	for i, b := range h.bounds {
		cum += h.counts[i]
		fmt.Fprintf(w, "%s_bucket{%s%sle=\"%s\"} %d\n", name, labels, sep, formatFloat(b), cum)
	}
	cum += h.counts[len(h.bounds)]
	fmt.Fprintf(w, "%s_bucket{%s%sle=\"+Inf\"} %d\n", name, labels, sep, cum)
	if labels == "" {
		fmt.Fprintf(w, "%s_sum %s\n%s_count %d\n", name, formatFloat(h.sum), name, h.count)
		return
	}
	fmt.Fprintf(w, "%s_sum{%s} %s\n%s_count{%s} %d\n", name, labels, formatFloat(h.sum), name, labels, h.count)
}

// Recorder accumulates step telemetry. The zero value is not usable; use New.
type Recorder struct {
	mu sync.Mutex

	now func() time.Time

	phases       map[Phase]*histogram
	stepSeconds  map[string]*histogram
	stepLanes    map[string]*histogram
	decodeTokens map[string]uint64
	cohortSize   *histogram
	specAccepted *histogram
	iterTokens   *histogram
	preemptions  map[string]uint64

	prefillTokens   uint64
	prefillChunks   uint64
	prefixMatched   uint64
	prefixQueried   uint64
	specRounds      uint64
	specDrafted     uint64
	specAcceptedTok uint64
	queueDepth      int64
	requestsActive  int64
	lastStep        time.Time

	ring []StepRecord
	next uint64 // sequence of the next record; ring index = (seq-1) % len(ring)
}

// New returns a recorder keeping the last ringSize step records (minimum 1).
func New(ringSize int) *Recorder {
	if ringSize < 1 {
		ringSize = 1
	}
	r := &Recorder{
		now:          time.Now,
		phases:       make(map[Phase]*histogram, len(Phases)),
		stepSeconds:  make(map[string]*histogram, len(Paths)),
		stepLanes:    make(map[string]*histogram, len(Paths)),
		decodeTokens: make(map[string]uint64, len(Paths)),
		cohortSize:   newHistogram(lanesBuckets),
		specAccepted: newHistogram(acceptBuckets),
		iterTokens:   newHistogram(iterTokBuckets),
		preemptions:  make(map[string]uint64, len(PreemptionReasons)),
		ring:         make([]StepRecord, ringSize),
	}
	for _, p := range Phases {
		r.phases[p] = newHistogram(secondsBuckets)
	}
	for _, p := range Paths {
		r.stepSeconds[p] = newHistogram(secondsBuckets)
		r.stepLanes[p] = newHistogram(lanesBuckets)
	}
	return r
}

// SetClock replaces the recorder's clock (tests).
func (r *Recorder) SetClock(now func() time.Time) {
	if r == nil || now == nil {
		return
	}
	r.mu.Lock()
	r.now = now
	r.mu.Unlock()
}

func (r *Recorder) appendLocked(rec StepRecord) {
	r.next++
	rec.Seq = r.next
	if rec.AtUnixNano == 0 {
		rec.AtUnixNano = r.now().UnixNano()
	}
	r.ring[int((r.next-1)%uint64(len(r.ring)))] = rec
}

// ObservePhase records one completed phase. Unknown phases are ignored (closed
// vocabulary keeps the series set bounded). Sample phases are histogram-only:
// they fire per token and would flush the ring of the decode steps they explain.
func (r *Recorder) ObservePhase(p Phase, d time.Duration) {
	if r == nil {
		return
	}
	// The observer is read BEFORE the recorder lock is taken so the fan-out never runs
	// while this package's mutex is held — a downstream sink is not allowed to become
	// part of the decode loop's lock ordering.
	if fn := phaseObservers(); fn != nil {
		fn(p, d)
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	h, ok := r.phases[p]
	if !ok {
		return
	}
	h.observe(d.Seconds())
	if p == PhaseSample {
		return
	}
	r.appendLocked(StepRecord{Kind: KindPhase, Phase: string(p), DurationNS: d.Nanoseconds()})
}

// ObserveDecodeStep records one decode forward that advanced `lanes` sequences
// by one token each, over the named path.
func (r *Recorder) ObserveDecodeStep(path string, lanes int, d time.Duration) {
	if r == nil || lanes <= 0 {
		return
	}
	if fn := stepObservers(); fn != nil {
		fn(path, lanes, d)
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	hs, ok := r.stepSeconds[path]
	if !ok {
		return
	}
	hs.observe(d.Seconds())
	r.stepLanes[path].observe(float64(lanes))
	r.iterTokens.observe(float64(lanes))
	r.decodeTokens[path] += uint64(lanes)
	r.lastStep = r.now()
	r.appendLocked(StepRecord{Kind: KindDecodeStep, Path: path, Lanes: lanes, Tokens: lanes, DurationNS: d.Nanoseconds(), AtUnixNano: r.lastStep.UnixNano()})
}

// ObserveSpeculativeRound records one draft-verify round on PathSpeculative:
// `proposed` draft tokens were verified, `accepted` of them matched the target,
// and `emitted` tokens (accepted prefix plus bonus, after stop/maxNew trimming)
// reached the caller. A round with no proposal is a serial step, not a round.
func (r *Recorder) ObserveSpeculativeRound(proposed, accepted, emitted int, d time.Duration) {
	if r == nil || proposed <= 0 {
		return
	}
	accepted = max(0, min(accepted, proposed))
	emitted = max(0, emitted)
	if fn := stepObservers(); fn != nil {
		fn(PathSpeculative, 1, d)
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.stepSeconds[PathSpeculative].observe(d.Seconds())
	r.stepLanes[PathSpeculative].observe(1)
	r.decodeTokens[PathSpeculative] += uint64(emitted)
	r.specRounds++
	r.specDrafted += uint64(proposed)
	r.specAcceptedTok += uint64(accepted)
	r.specAccepted.observe(float64(accepted))
	// One verify forward processes the bonus position plus every draft token.
	r.iterTokens.observe(float64(proposed + 1))
	r.lastStep = r.now()
	r.appendLocked(StepRecord{Kind: KindDecodeStep, Path: PathSpeculative, Lanes: 1, Tokens: emitted, Proposed: proposed, Accepted: accepted, DurationNS: d.Nanoseconds(), AtUnixNano: r.lastStep.UnixNano()})
}

// ObservePrefillChunk records one prefill forward over `tokens` prompt tokens.
func (r *Recorder) ObservePrefillChunk(tokens int, d time.Duration) {
	if r == nil || tokens <= 0 {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.prefillTokens += uint64(tokens)
	r.prefillChunks++
	r.iterTokens.observe(float64(tokens))
	r.lastStep = r.now()
	r.appendLocked(StepRecord{Kind: KindPrefillChunk, Tokens: tokens, DurationNS: d.Nanoseconds(), AtUnixNano: r.lastStep.UnixNano()})
}

// ObservePrefixMatched records prompt tokens served from the KV prefix cache.
func (r *Recorder) ObservePrefixMatched(tokens int) {
	if r == nil || tokens <= 0 {
		return
	}
	r.mu.Lock()
	r.prefixMatched += uint64(tokens)
	r.mu.Unlock()
}

// ObservePrefixQueried records prompt tokens looked up in the KV prefix cache,
// hit or miss. Call it beside ObservePrefixMatched with the looked-up prompt
// length so matched/queried is the token hit rate.
func (r *Recorder) ObservePrefixQueried(tokens int) {
	if r == nil || tokens <= 0 {
		return
	}
	r.mu.Lock()
	r.prefixQueried += uint64(tokens)
	r.mu.Unlock()
}

// ObservePreemption records one running lane preempted under KV pressure, by
// reason (one of PreemptionReasons). Unknown reasons are dropped: the closed
// vocabulary keeps the series set bounded.
func (r *Recorder) ObservePreemption(reason string) {
	if r == nil {
		return
	}
	known := false
	for _, k := range PreemptionReasons {
		if k == reason {
			known = true
			break
		}
	}
	if !known {
		return
	}
	r.mu.Lock()
	r.preemptions[reason]++
	r.mu.Unlock()
}

// ObserveCohort records one decode cohort of `size` lanes formed by the coalescer.
func (r *Recorder) ObserveCohort(size int) {
	if r == nil || size <= 0 {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.cohortSize.observe(float64(size))
	r.appendLocked(StepRecord{Kind: KindCohort, Lanes: size})
}

// SetQueueDepth publishes the number of requests waiting in the decode coalescer.
func (r *Recorder) SetQueueDepth(n int) {
	if r == nil {
		return
	}
	if n < 0 {
		n = 0
	}
	r.mu.Lock()
	r.queueDepth = int64(n)
	r.mu.Unlock()
}

// RequestStart marks one request entering the planner; call the returned func
// exactly once when it leaves. The request's end-to-end phase is recorded then.
func (r *Recorder) RequestStart() (done func()) {
	if r == nil {
		return func() {}
	}
	r.mu.Lock()
	r.requestsActive++
	start := r.now()
	r.mu.Unlock()
	var once sync.Once
	return func() {
		once.Do(func() {
			r.mu.Lock()
			r.requestsActive--
			d := r.now().Sub(start)
			r.mu.Unlock()
			r.ObservePhase(PhaseRequest, d)
		})
	}
}

// PhaseStat summarizes one histogram.
type PhaseStat struct {
	Count      uint64  `json:"count"`
	SumSeconds float64 `json:"sum_seconds"`
	P50Seconds float64 `json:"p50_seconds"`
	P99Seconds float64 `json:"p99_seconds"`
	MaxSeconds float64 `json:"max_seconds"`
}

// PathStat summarizes decode steps on one path (or cohorts, where Steps counts
// cohorts and the lane fields describe cohort size).
type PathStat struct {
	Steps          uint64  `json:"steps"`
	Tokens         uint64  `json:"tokens"`
	MeanLanes      float64 `json:"mean_lanes"`
	MaxLanes       float64 `json:"max_lanes"`
	P50StepSeconds float64 `json:"p50_step_seconds"`
	P99StepSeconds float64 `json:"p99_step_seconds"`
}

// SpecStat summarizes speculative draft-verify rounds. AcceptanceRate is
// accepted/drafted; TokensPerRound is emitted tokens per target forward, the
// number speculation must push above 1 to pay.
type SpecStat struct {
	Rounds         uint64  `json:"rounds"`
	DraftTokens    uint64  `json:"draft_tokens"`
	AcceptedTokens uint64  `json:"accepted_tokens"`
	AcceptanceRate float64 `json:"acceptance_rate"`
	TokensPerRound float64 `json:"tokens_per_round"`
}

// IterStat summarizes the per-step token histogram.
type IterStat struct {
	Count uint64  `json:"count"`
	Mean  float64 `json:"mean"`
	P50   float64 `json:"p50"`
	P99   float64 `json:"p99"`
	Max   float64 `json:"max"`
}

// Snapshot is the bounded agent-facing view of the recorder.
type Snapshot struct {
	Schema             string               `json:"schema"`
	NowUnixNano        int64                `json:"now_unix_nano"`
	LastStepUnixNano   int64                `json:"last_step_unix_nano,omitempty"`
	RequestsActive     int64                `json:"requests_active"`
	CoalesceQueueDepth int64                `json:"coalesce_queue_depth"`
	Phases             map[string]PhaseStat `json:"phases"`
	Decode             map[string]PathStat  `json:"decode"`
	Cohorts            PathStat             `json:"cohorts"`
	Speculative        *SpecStat            `json:"speculative,omitempty"`
	PrefillTokens      uint64               `json:"prefill_tokens"`
	PrefillChunks      uint64               `json:"prefill_chunks"`
	PrefixMatched      uint64               `json:"prefix_matched_tokens"`
	PrefixQueried      uint64               `json:"prefix_queried_tokens"`
	// PrefixHitRate is PrefixMatched/PrefixQueried (0 when nothing was queried).
	PrefixHitRate float64           `json:"prefix_hit_rate"`
	Preemptions   map[string]uint64 `json:"preemptions"`
	// IterationTokens summarizes tokens processed per engine step (Count=steps).
	IterationTokens IterStat     `json:"iteration_tokens"`
	Recent          []StepRecord `json:"recent"`
}

// SnapshotSchema versions the Snapshot JSON shape.
const SnapshotSchema = "fak-engine-steps/1"

// MaxRecent caps how many ring records one Snapshot may return.
const MaxRecent = 512

// Snapshot returns the summary plus the newest `recent` ring records (oldest
// first), optionally filtered to one kind ("" = all). recent is clamped to
// [0, MaxRecent].
func (r *Recorder) Snapshot(recent int, kind string) Snapshot {
	if recent < 0 {
		recent = 0
	}
	if recent > MaxRecent {
		recent = MaxRecent
	}
	s := Snapshot{Schema: SnapshotSchema, Phases: map[string]PhaseStat{}, Decode: map[string]PathStat{}, Preemptions: map[string]uint64{}, Recent: []StepRecord{}}
	if r == nil {
		s.NowUnixNano = time.Now().UnixNano()
		return s
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	s.NowUnixNano = r.now().UnixNano()
	if !r.lastStep.IsZero() {
		s.LastStepUnixNano = r.lastStep.UnixNano()
	}
	s.RequestsActive = r.requestsActive
	s.CoalesceQueueDepth = r.queueDepth
	for _, p := range Phases {
		h := r.phases[p]
		if h.count == 0 {
			continue
		}
		s.Phases[string(p)] = PhaseStat{Count: h.count, SumSeconds: h.sum, P50Seconds: h.quantile(0.5), P99Seconds: h.quantile(0.99), MaxSeconds: h.max}
	}
	for _, p := range Paths {
		hs, hl := r.stepSeconds[p], r.stepLanes[p]
		if hs.count == 0 {
			continue
		}
		s.Decode[p] = PathStat{Steps: hs.count, Tokens: r.decodeTokens[p], MeanLanes: hl.sum / float64(hl.count), MaxLanes: hl.max, P50StepSeconds: hs.quantile(0.5), P99StepSeconds: hs.quantile(0.99)}
	}
	if c := r.cohortSize; c.count > 0 {
		s.Cohorts = PathStat{Steps: c.count, MeanLanes: c.sum / float64(c.count), MaxLanes: c.max}
	}
	s.PrefillTokens, s.PrefillChunks, s.PrefixMatched = r.prefillTokens, r.prefillChunks, r.prefixMatched
	s.PrefixQueried = r.prefixQueried
	if r.prefixQueried > 0 {
		s.PrefixHitRate = float64(r.prefixMatched) / float64(r.prefixQueried)
	}
	for _, k := range PreemptionReasons {
		s.Preemptions[k] = r.preemptions[k]
	}
	if h := r.iterTokens; h.count > 0 {
		s.IterationTokens = IterStat{Count: h.count, Mean: h.sum / float64(h.count), P50: h.quantile(0.5), P99: h.quantile(0.99), Max: h.max}
	}
	if r.specRounds > 0 {
		s.Speculative = &SpecStat{
			Rounds:         r.specRounds,
			DraftTokens:    r.specDrafted,
			AcceptedTokens: r.specAcceptedTok,
			AcceptanceRate: float64(r.specAcceptedTok) / float64(r.specDrafted),
			TokensPerRound: float64(r.decodeTokens[PathSpeculative]) / float64(r.specRounds),
		}
	}
	if recent == 0 || r.next == 0 {
		return s
	}
	held := r.next
	if held > uint64(len(r.ring)) {
		held = uint64(len(r.ring))
	}
	picked := make([]StepRecord, 0, recent)
	for i := uint64(0); i < held && len(picked) < recent; i++ {
		seq := r.next - i
		rec := r.ring[int((seq-1)%uint64(len(r.ring)))]
		if kind != "" && rec.Kind != kind {
			continue
		}
		picked = append(picked, rec)
	}
	for i, j := 0, len(picked)-1; i < j; i, j = i+1, j-1 {
		picked[i], picked[j] = picked[j], picked[i]
	}
	s.Recent = picked
	return s
}

// Compact renders the snapshot summary as one bounded line for agents.
func (s Snapshot) Compact() string {
	var b strings.Builder
	fmt.Fprintf(&b, "ENGINE active=%d queue=%d", s.RequestsActive, s.CoalesceQueueDepth)
	if s.LastStepUnixNano > 0 {
		fmt.Fprintf(&b, " last_step=%s ago", time.Duration(s.NowUnixNano-s.LastStepUnixNano).Round(time.Millisecond))
	} else {
		b.WriteString(" last_step=never")
	}
	for _, p := range Paths {
		st, ok := s.Decode[p]
		if !ok {
			continue
		}
		fmt.Fprintf(&b, " | %s steps=%d tok=%d lanes~%.1f(max %.0f) p50=%s p99=%s", p, st.Steps, st.Tokens, st.MeanLanes, st.MaxLanes, secs(st.P50StepSeconds), secs(st.P99StepSeconds))
	}
	if s.Cohorts.Steps > 0 {
		fmt.Fprintf(&b, " | cohorts=%d size~%.1f(max %.0f)", s.Cohorts.Steps, s.Cohorts.MeanLanes, s.Cohorts.MaxLanes)
	}
	if sp := s.Speculative; sp != nil {
		fmt.Fprintf(&b, " | spec rounds=%d accept=%.0f%% (%d/%d) tok/round=%.2f", sp.Rounds, 100*sp.AcceptanceRate, sp.AcceptedTokens, sp.DraftTokens, sp.TokensPerRound)
	}
	if s.PrefillChunks > 0 || s.PrefixMatched > 0 || s.PrefixQueried > 0 {
		fmt.Fprintf(&b, " | prefill tok=%d chunks=%d prefix_hit_tok=%d/%d (%.0f%%)", s.PrefillTokens, s.PrefillChunks, s.PrefixMatched, s.PrefixQueried, 100*s.PrefixHitRate)
	}
	if it := s.IterationTokens; it.Count > 0 {
		fmt.Fprintf(&b, " | iter_tok p50=%.0f p99=%.0f max=%.0f", it.P50, it.P99, it.Max)
	}
	var preempt uint64
	for _, v := range s.Preemptions {
		preempt += v
	}
	if preempt > 0 {
		fmt.Fprintf(&b, " | preempt=%d (swap=%d recompute=%d gpudirect=%d)", preempt, s.Preemptions[PreemptSwap], s.Preemptions[PreemptRecompute], s.Preemptions[PreemptGPUDirectSwap])
	}
	var phases []string
	for _, p := range Phases {
		st, ok := s.Phases[string(p)]
		if !ok {
			continue
		}
		phases = append(phases, fmt.Sprintf("%s=%s", p, secs(st.P50Seconds)))
	}
	if len(phases) > 0 {
		b.WriteString(" | p50 ")
		b.WriteString(strings.Join(phases, " "))
	}
	return b.String()
}

func secs(v float64) string {
	return time.Duration(v * float64(time.Second)).Round(10 * time.Microsecond).String()
}

// WritePrometheus renders the fak_engine_* families in Prometheus text format.
// Every family is always emitted (zero-valued before traffic) so dashboards can
// tell "engine idle" from "series missing".
func (r *Recorder) WritePrometheus(w io.Writer) {
	if r == nil {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	helpType(w, MetricPhaseSeconds, "Native serving loop phase durations (admission_wait, device_wait, cohort_wait, prefix_lookup, prefill, prefix_admit, decode, sample, request).", "histogram")
	for _, p := range Phases {
		r.phases[p].write(w, MetricPhaseSeconds, fmt.Sprintf("phase=%q", string(p)))
	}
	helpType(w, MetricDecodeStepSeconds, "Wall time of one decode forward step, by path (serial Session.Step or batched StepBatchActive).", "histogram")
	for _, p := range Paths {
		r.stepSeconds[p].write(w, MetricDecodeStepSeconds, fmt.Sprintf("path=%q", p))
	}
	helpType(w, MetricDecodeStepLanes, "Active sequences (batch size) advanced by one decode step, by path.", "histogram")
	for _, p := range Paths {
		r.stepLanes[p].write(w, MetricDecodeStepLanes, fmt.Sprintf("path=%q", p))
	}
	helpType(w, MetricDecodeTokensTotal, "Tokens advanced by decode steps (sum of step lanes), by path.", "counter")
	for _, p := range Paths {
		fmt.Fprintf(w, "%s{path=%q} %d\n", MetricDecodeTokensTotal, p, r.decodeTokens[p])
	}
	helpType(w, MetricCohortSize, "Lanes per decode cohort formed by the continuous-batching coalescer.", "histogram")
	r.cohortSize.write(w, MetricCohortSize, "")
	helpType(w, MetricCoalesceQueueDepth, "Requests waiting in the decode coalescer for the next cohort.", "gauge")
	fmt.Fprintf(w, "%s %d\n", MetricCoalesceQueueDepth, r.queueDepth)
	helpType(w, MetricPrefillTokensTotal, "Prompt tokens forwarded by prefill (excludes prefix-cache hits).", "counter")
	fmt.Fprintf(w, "%s %d\n", MetricPrefillTokensTotal, r.prefillTokens)
	helpType(w, MetricPrefillChunksTotal, "Prefill forward chunks executed.", "counter")
	fmt.Fprintf(w, "%s %d\n", MetricPrefillChunksTotal, r.prefillChunks)
	helpType(w, MetricPrefixMatchedTotal, "Prompt tokens served from the KV prefix cache instead of prefill.", "counter")
	fmt.Fprintf(w, "%s %d\n", MetricPrefixMatchedTotal, r.prefixMatched)
	helpType(w, MetricPrefixQueriedTotal, "Prompt tokens looked up in the KV prefix cache, hit or miss (hit-rate denominator).", "counter")
	fmt.Fprintf(w, "%s %d\n", MetricPrefixQueriedTotal, r.prefixQueried)
	helpType(w, MetricPreemptionsTotal, "Running lanes preempted under KV pressure, by reason (swap, recompute, gpudirect_swap).", "counter")
	for _, k := range PreemptionReasons {
		fmt.Fprintf(w, "%s{reason=%q} %d\n", MetricPreemptionsTotal, k, r.preemptions[k])
	}
	helpType(w, MetricIterationTokens, "Tokens processed per engine step (decode lanes per decode step, chunk tokens per prefill chunk).", "histogram")
	r.iterTokens.write(w, MetricIterationTokens, "")
	helpType(w, MetricRequestsActive, "Requests currently inside the native planner.", "gauge")
	fmt.Fprintf(w, "%s %d\n", MetricRequestsActive, r.requestsActive)
	helpType(w, MetricLastStepTimestamp, "Unix time of the last prefill chunk or decode step (0 = never stepped).", "gauge")
	last := 0.0
	if !r.lastStep.IsZero() {
		last = float64(r.lastStep.UnixNano()) / 1e9
	}
	fmt.Fprintf(w, "%s %s\n", MetricLastStepTimestamp, formatFloat(last))
	helpType(w, MetricSpecRoundsTotal, "Speculative draft-verify rounds (target forwards that verified a draft).", "counter")
	fmt.Fprintf(w, "%s %d\n", MetricSpecRoundsTotal, r.specRounds)
	helpType(w, MetricSpecDraftTokensTotal, "Draft tokens proposed to speculative verification.", "counter")
	fmt.Fprintf(w, "%s %d\n", MetricSpecDraftTokensTotal, r.specDrafted)
	helpType(w, MetricSpecAcceptedTokensTotal, "Draft tokens accepted by speculative verification.", "counter")
	fmt.Fprintf(w, "%s %d\n", MetricSpecAcceptedTokensTotal, r.specAcceptedTok)
	helpType(w, MetricSpecAcceptedPerRound, "Draft tokens accepted per speculative verify round.", "histogram")
	r.specAccepted.write(w, MetricSpecAcceptedPerRound, "")
}

func helpType(w io.Writer, name, help, typ string) {
	fmt.Fprintf(w, "# HELP %s %s\n# TYPE %s %s\n", name, help, name, typ)
}

func formatFloat(v float64) string {
	if math.IsInf(v, 1) {
		return "+Inf"
	}
	return strconv.FormatFloat(v, 'f', -1, 64)
}
