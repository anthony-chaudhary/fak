package engine

// This file is the live-engine cache-event seam: when fak runs in front of a live
// inference engine (vLLM / SGLang / LMCache / Dynamo KVBM), that engine emits KV
// routing/offload/restore/migrate events. Without this seam those events live only
// in the engine's own internal counters, so cache behavior across the live-engine
// boundary is invisible in fak's one cache-entry stream.
//
// CacheEventRecorder normalizes each such event into the SAME cachemeta.Entry
// stream as tool/context entries (via cachemeta.FromKVTransfer, plane kv_transfer),
// records residency tier + owner separately from the payload, derives the typed
// lookup verdict (cachemeta.KVTransferVerdict) so a failed restore/load is a typed
// MISS or FAULT and never a silent recompute, and aggregates the stream into a
// metric surface (CacheEventMetrics) that renders Prometheus text the way the
// gateway exposes kernel counters — not just the engine's internal counters.
//
// This package never touches KV tensors; it observes residency transitions and
// lowers their metadata. See AGENTIC-CACHING-SOTA-2026-06-19.md §2.2 (issue #113).

import (
	"sort"
	"strings"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/cachemeta"
	"github.com/anthony-chaudhary/fak/internal/compute"
)

// CacheEvent is the live-engine residency event a routing/offload adapter feeds the
// recorder. It is the field-only shape that lowers into cachemeta.KVTransfer; the
// recorder keeps residency tier and owner separate from the (unseen) payload.
type CacheEvent struct {
	Direction    cachemeta.KVTransferDirection
	SpanDigest   string
	Tokens       int64
	ModelID      string
	TokenizerID  string
	PositionMode cachemeta.PositionMode
	FromTier     cachemeta.ResidencyTier
	ToTier       cachemeta.ResidencyTier
	Owner        string
	Lease        string
	Outcome      cachemeta.KVTransferOutcome
	FaultReason  string
	BytesMoved   int64
}

// toTransfer projects a CacheEvent onto the cachemeta.KVTransfer shape so the single
// normalization helper (cachemeta.FromKVTransfer) does the lowering — the engine
// lane never re-implements the entry shape.
func (ev CacheEvent) toTransfer() cachemeta.KVTransfer {
	return cachemeta.KVTransfer{
		Direction:    ev.Direction,
		SpanDigest:   ev.SpanDigest,
		Tokens:       ev.Tokens,
		ModelID:      ev.ModelID,
		TokenizerID:  ev.TokenizerID,
		PositionMode: ev.PositionMode,
		FromTier:     ev.FromTier,
		ToTier:       ev.ToTier,
		Owner:        ev.Owner,
		Lease:        ev.Lease,
		Outcome:      ev.Outcome,
		FaultReason:  ev.FaultReason,
		BytesMoved:   ev.BytesMoved,
	}
}

// CacheEventResult is what Record returns: the normalized cache-entry (on the
// kv_transfer plane, in the SAME stream as tool/context entries) plus the typed
// lookup verdict derived from the event's outcome. A caller that asked for a
// restore/load reads Verdict and MUST treat a non-Hit as a typed MISS/FAULT — the
// whole point of this seam is that a failed restore is observable, never a silent
// recompute.
type CacheEventResult struct {
	Entry   cachemeta.Entry
	Verdict cachemeta.LookupVerdict
}

// SilentRecompute reports whether folding this result as "just recompute it" would
// hide a real cache fault/miss. It is true for any non-Hit outcome: callers use it
// to assert the never-silent-recompute rule (a recompute is fine, but only after the
// miss/fault has been recorded as such, never instead of recording it).
func (r CacheEventResult) SilentRecompute() bool {
	return !r.Verdict.CanServe()
}

// CacheEventMetrics is the metric surface over the cache-event stream. It is the
// "cache events exposed as metrics, not just internal engine counters" half of the
// §2.2 parity requirement. Counts are keyed by (direction, outcome, to_tier) so a
// scrape can separate, e.g., a restore fault to remote from an offload ok to dram.
type CacheEventMetrics struct {
	mu sync.Mutex

	byKey map[cacheEventKey]*cacheEventAgg

	// totals, kept so a scrape need not re-sum the keyed map for the headline gauges.
	events       uint64
	hits         uint64
	misses       uint64
	faults       uint64
	bytesMoved   int64
	tokensMoved  int64
	restoreMiss  uint64 // restore/load that found nothing usable (typed MISS)
	restoreFault uint64 // restore/load that errored (typed FAULT) — never silent
}

type cacheEventKey struct {
	direction   string
	outcome     string
	toTier      string
	memoryClass string
}

type cacheEventAgg struct {
	count       uint64
	bytesMoved  int64
	tokensMoved int64
}

// NewCacheEventMetrics returns an empty, ready metric surface.
func NewCacheEventMetrics() *CacheEventMetrics {
	return &CacheEventMetrics{byKey: map[cacheEventKey]*cacheEventAgg{}}
}

// observe folds one normalized entry + verdict into the counters.
func (mx *CacheEventMetrics) observe(e cachemeta.Entry, v cachemeta.LookupVerdict) {
	if mx == nil {
		return
	}
	mx.mu.Lock()
	defer mx.mu.Unlock()
	if mx.byKey == nil {
		mx.byKey = map[cacheEventKey]*cacheEventAgg{}
	}
	key := cacheEventKey{
		direction:   e.Labels["direction"],
		outcome:     e.Labels["outcome"],
		toTier:      string(e.Residency.Tier),
		memoryClass: string(CacheTierMemoryClass(e.Residency.Tier)),
	}
	agg := mx.byKey[key]
	if agg == nil {
		agg = &cacheEventAgg{}
		mx.byKey[key] = agg
	}
	agg.count++
	agg.bytesMoved += e.Metrics.BytesTransferred
	agg.tokensMoved += e.ID.Length

	mx.events++
	mx.bytesMoved += e.Metrics.BytesTransferred
	mx.tokensMoved += e.ID.Length
	switch v.Kind {
	case cachemeta.LookupHit:
		mx.hits++
	case cachemeta.LookupFault:
		mx.faults++
		if v.Reason == cachemeta.ReasonResidencyFault {
			mx.restoreFault++
		}
	case cachemeta.LookupMiss:
		mx.misses++
		if v.Reason == cachemeta.ReasonRestoreMiss {
			mx.restoreMiss++
		}
	}
}

// CacheEventSnapshot is an immutable read of the metric surface, safe to render
// without holding the lock.
type CacheEventSnapshot struct {
	Events       uint64
	Hits         uint64
	Misses       uint64
	Faults       uint64
	RestoreMiss  uint64
	RestoreFault uint64
	BytesMoved   int64
	TokensMoved  int64
	Rows         []CacheEventRow
}

// CacheEventRow is one (direction, outcome, to_tier, memory_class) bucket.
type CacheEventRow struct {
	Direction   string
	Outcome     string
	ToTier      string
	MemoryClass string
	Count       uint64
	BytesMoved  int64
	TokensMoved int64
}

// Snapshot copies the current counters into a stable, sorted read.
func (mx *CacheEventMetrics) Snapshot() CacheEventSnapshot {
	s := CacheEventSnapshot{}
	if mx == nil {
		return s
	}
	mx.mu.Lock()
	defer mx.mu.Unlock()
	s.Events = mx.events
	s.Hits = mx.hits
	s.Misses = mx.misses
	s.Faults = mx.faults
	s.RestoreMiss = mx.restoreMiss
	s.RestoreFault = mx.restoreFault
	s.BytesMoved = mx.bytesMoved
	s.TokensMoved = mx.tokensMoved
	s.Rows = make([]CacheEventRow, 0, len(mx.byKey))
	for k, agg := range mx.byKey {
		s.Rows = append(s.Rows, CacheEventRow{
			Direction:   k.direction,
			Outcome:     k.outcome,
			ToTier:      k.toTier,
			MemoryClass: k.memoryClass,
			Count:       agg.count,
			BytesMoved:  agg.bytesMoved,
			TokensMoved: agg.tokensMoved,
		})
	}
	sort.Slice(s.Rows, func(i, j int) bool {
		a, b := s.Rows[i], s.Rows[j]
		if a.Direction != b.Direction {
			return a.Direction < b.Direction
		}
		if a.Outcome != b.Outcome {
			return a.Outcome < b.Outcome
		}
		if a.ToTier != b.ToTier {
			return a.ToTier < b.ToTier
		}
		return a.MemoryClass < b.MemoryClass
	})
	return s
}

// CacheTierMemoryClass projects a residency tier into the operator-facing memory class used
// by the capacity/OOM surfaces. HBM is the hot device KV cache; byte-addressable host/far
// tiers are DDR-cache residency; disk/remote/provider are offload tiers; an unset tier is
// unknown. This is deliberately a projection over metadata only — it does not move bytes.
func CacheTierMemoryClass(t cachemeta.ResidencyTier) compute.MemoryClass {
	switch t {
	case cachemeta.TierHBM:
		return compute.MemoryKVCache
	case cachemeta.TierDRAM, cachemeta.TierNUMAFar, cachemeta.TierCXL:
		return compute.MemoryDDRCache
	case cachemeta.TierDisk, cachemeta.TierRemote, cachemeta.TierProvider:
		return compute.MemoryOffload
	default:
		return compute.MemoryUnknown
	}
}

// Prometheus renders the snapshot as Prometheus exposition text. This is the metric
// surface a scrape reads to see cache behavior across the live-engine boundary as one
// stream — not the engine's internal counters. Metric names are namespaced
// fak_engine_cache_* so they sit alongside the gateway's fak_gateway_*/fak_kernel_*.
func (s CacheEventSnapshot) Prometheus() string {
	var b strings.Builder
	help := func(name, h, typ string) {
		b.WriteString("# HELP " + name + " " + h + "\n")
		b.WriteString("# TYPE " + name + " " + typ + "\n")
	}
	help("fak_engine_cache_events_total", "Live-engine KV cache events normalized into the cache-entry stream.", "counter")
	b.WriteString("fak_engine_cache_events_total " + utoa(s.Events) + "\n")
	help("fak_engine_cache_hits_total", "Cache events whose typed verdict was a HIT (serveable).", "counter")
	b.WriteString("fak_engine_cache_hits_total " + utoa(s.Hits) + "\n")
	help("fak_engine_cache_misses_total", "Cache events whose typed verdict was a MISS.", "counter")
	b.WriteString("fak_engine_cache_misses_total " + utoa(s.Misses) + "\n")
	help("fak_engine_cache_faults_total", "Cache events whose typed verdict was a FAULT (never silent recompute).", "counter")
	b.WriteString("fak_engine_cache_faults_total " + utoa(s.Faults) + "\n")
	help("fak_engine_cache_restore_miss_total", "Restore/load events that found nothing usable (typed MISS).", "counter")
	b.WriteString("fak_engine_cache_restore_miss_total " + utoa(s.RestoreMiss) + "\n")
	help("fak_engine_cache_restore_fault_total", "Restore/load events that errored (typed FAULT) — surfaced, never a silent recompute.", "counter")
	b.WriteString("fak_engine_cache_restore_fault_total " + utoa(s.RestoreFault) + "\n")
	help("fak_engine_cache_bytes_moved_total", "Bytes moved across residency tiers by cache events.", "counter")
	b.WriteString("fak_engine_cache_bytes_moved_total " + itoa64(s.BytesMoved) + "\n")
	help("fak_engine_cache_tokens_moved_total", "KV span positions moved across residency tiers by cache events.", "counter")
	b.WriteString("fak_engine_cache_tokens_moved_total " + itoa64(s.TokensMoved) + "\n")
	help("fak_engine_cache_event_breakdown_total", "Cache events by direction, outcome, destination residency tier, and memory class.", "counter")
	for _, r := range s.Rows {
		b.WriteString("fak_engine_cache_event_breakdown_total{direction=\"" + promLabel(r.Direction) +
			"\",outcome=\"" + promLabel(r.Outcome) + "\",to_tier=\"" + promLabel(r.ToTier) +
			"\",memory_class=\"" + promLabel(r.MemoryClass) + "\"} " +
			utoa(r.Count) + "\n")
	}
	help("fak_engine_cache_bytes_moved_breakdown_total", "Bytes moved by cache events, bucketed by direction, outcome, destination residency tier, and memory class.", "counter")
	for _, r := range s.Rows {
		b.WriteString("fak_engine_cache_bytes_moved_breakdown_total{direction=\"" + promLabel(r.Direction) +
			"\",outcome=\"" + promLabel(r.Outcome) + "\",to_tier=\"" + promLabel(r.ToTier) +
			"\",memory_class=\"" + promLabel(r.MemoryClass) + "\"} " +
			itoa64(r.BytesMoved) + "\n")
	}
	help("fak_engine_cache_tokens_moved_breakdown_total", "KV span positions moved by cache events, bucketed by direction, outcome, destination residency tier, and memory class.", "counter")
	for _, r := range s.Rows {
		b.WriteString("fak_engine_cache_tokens_moved_breakdown_total{direction=\"" + promLabel(r.Direction) +
			"\",outcome=\"" + promLabel(r.Outcome) + "\",to_tier=\"" + promLabel(r.ToTier) +
			"\",memory_class=\"" + promLabel(r.MemoryClass) + "\"} " +
			itoa64(r.TokensMoved) + "\n")
	}
	return b.String()
}

// CacheEventRecorder is the live-engine cache-event seam. The routing/offload path
// calls Record for each KV residency transition; the recorder normalizes it into the
// shared cache-entry stream, derives the typed verdict, folds metrics, and (when set)
// fans the entry out to an observer sink (e.g. a structured logger). Safe for
// concurrent use.
type CacheEventRecorder struct {
	metrics *CacheEventMetrics

	mu   sync.Mutex
	sink func(cachemeta.Entry, cachemeta.LookupVerdict)
}

// NewCacheEventRecorder returns a recorder with its own metric surface.
func NewCacheEventRecorder() *CacheEventRecorder {
	return &CacheEventRecorder{metrics: NewCacheEventMetrics()}
}

// Metrics returns the recorder's metric surface for scraping.
func (r *CacheEventRecorder) Metrics() *CacheEventMetrics { return r.metrics }

// SetSink installs an observer called with every normalized (entry, verdict). It is
// for fan-out only (logging/tracing); it does not change the verdict.
func (r *CacheEventRecorder) SetSink(fn func(cachemeta.Entry, cachemeta.LookupVerdict)) {
	r.mu.Lock()
	r.sink = fn
	r.mu.Unlock()
}

// Record normalizes one live-engine cache event into the cache-entry stream and
// returns the entry plus its typed verdict. The verdict makes a failed restore/load
// a typed MISS/FAULT — the caller must honor that rather than silently recomputing.
func (r *CacheEventRecorder) Record(ev CacheEvent, opts ...cachemeta.Option) CacheEventResult {
	entry := cachemeta.FromKVTransfer(ev.toTransfer(), opts...)
	verdict := cachemeta.KVTransferVerdict(entry)
	if r != nil {
		r.metrics.observe(entry, verdict)
		r.mu.Lock()
		sink := r.sink
		r.mu.Unlock()
		if sink != nil {
			sink(entry, verdict)
		}
	}
	return CacheEventResult{Entry: entry, Verdict: verdict}
}

func promLabel(s string) string {
	s = strings.ReplaceAll(s, `\`, `\\`)
	s = strings.ReplaceAll(s, "\n", `\n`)
	s = strings.ReplaceAll(s, `"`, `\"`)
	return s
}

func utoa(n uint64) string {
	if n == 0 {
		return "0"
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	return string(buf[i:])
}

func itoa64(n int64) string {
	if n < 0 {
		return "-" + utoa(uint64(-n))
	}
	return utoa(uint64(n))
}
