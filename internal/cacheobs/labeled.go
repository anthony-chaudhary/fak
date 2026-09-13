package cacheobs

// labeled.go — the #3391 per-(model, tenant) attribution of the KV-prefix reuse tap,
// plus the eligibility-carrying tap that feeds it.
//
// The unlabeled counters answer "how well is THE cache doing"; a shared gateway also
// needs "for WHOM" — which model earns the reuse and whose tenant traffic rides it (the
// tenant here is the same authenticated prefix-cache identity radixkv's scoped tree
// isolates on, so the label never invents an identity the cache itself does not know).
// The breakdown is deliberately booked from the SAME clamped per-turn values as the
// global counters, inside the same critical section, so summing any column across
// LabeledSnapshot rows reconciles exactly with the global Stats — a label row can never
// drift from the aggregate it decomposes.

import (
	"sort"
	"strings"
)

// Labels keys one (model, tenant) series of the per-series breakdown (#3391). An absent
// component normalizes to "unknown" at observe time — mirroring the gateway's serving
// metrics label defaulting — so an unlabeled legacy tap and a labeled tap can never mint
// two spellings of the same series (which would render duplicate Prometheus series).
const (
	// PhasePrefill attributes cache work performed while ingesting the prompt.
	PhasePrefill = "prefill"
	// PhaseDecode attributes cache work performed while generating output tokens.
	PhaseDecode = "decode"
	// PhaseOther is the bounded fallback for absent or unrecognized pipeline phases.
	PhaseOther = "other"
)

// Labels keys one (model, tenant, phase) series of the per-series breakdown. Phase has
// a deliberately closed vocabulary: prefill, decode, or other. This prevents caller-
// supplied pipeline names from creating unbounded telemetry cardinality.
type Labels struct {
	Model  string
	Tenant string
	Phase  string
}

// normalized trims the identity components, maps empty identities onto "unknown", and
// collapses every phase outside the closed vocabulary onto PhaseOther.
func (l Labels) normalized() Labels {
	l.Model = labelOrUnknown(l.Model)
	l.Tenant = labelOrUnknown(l.Tenant)
	l.Phase = normalizePhase(l.Phase)
	return l
}

func normalizePhase(phase string) string {
	switch strings.TrimSpace(phase) {
	case PhasePrefill:
		return PhasePrefill
	case PhaseDecode:
		return PhaseDecode
	default:
		return PhaseOther
	}
}

func labelOrUnknown(v string) string {
	if v = strings.TrimSpace(v); v != "" {
		return v
	}
	return "unknown"
}

// labelTotals is one series' share of the global token counters. Only the columns the
// #3391 breakdown exposes are tracked per label (turns, prompt, eligible, reused) — the
// regime buckets, histogram, and miss-cause split remain global-only.
type labelTotals struct {
	turns          uint64
	promptTokens   uint64
	eligibleTokens uint64
	reusedTokens   uint64
	// Per-label source-axis counters (#12886), the same four buckets the global source axis
	// carries. Booking them in the SAME critical section as the label's depth counters means
	// a per-label source row can never drift from the global source total it decomposes:
	// summing these across rows always reconciles, exactly like the depth columns.
	srcLocalCompute     uint64
	srcLocalHit         uint64
	srcExternalTransfer uint64
	srcUnknown          uint64
}

// labelTotalsLocked returns (creating if needed) the row for labels. Caller holds o.mu;
// labels must already be normalized.
func (o *Observer) labelTotalsLocked(labels Labels) *labelTotals {
	if o.byLabel == nil {
		o.byLabel = make(map[Labels]*labelTotals)
	}
	lt := o.byLabel[labels]
	if lt == nil {
		lt = &labelTotals{}
		o.byLabel[labels] = lt
	}
	return lt
}

// ObserveLabeled records one served in-kernel turn exactly like ObserveSplit and
// additionally (#3391) attributes it to its (model, tenant) series and carries the
// turn's eligibility-filtered denominator: eligiblePromptTokens is how many of the
// promptTokens COULD have been served from the cached KV prefix — the prompt minus the
// turn's always-uncacheable share. A caller passes 0 for a turn that could not hit at
// all (the cold first prefill into an empty cache, or prefix reuse disabled) and
// promptTokens when the whole prompt was in play. The value is clamped into
// [cacheablePrefixTokens, promptTokens] after the ObserveSplit clamps — a token that
// matched the index at lookup was demonstrably cacheable, hence eligible — so a stale
// witness (e.g. a prewarmed tree serving a "first" prefill) can never push the filtered
// ratio reused/eligible above 1. Every pre-existing counter, regime bucket, and
// histogram slot accumulates exactly as ObserveSplit; the label row books the SAME
// clamped values as the globals, so LabeledSnapshot always reconciles with Snapshot.
// With no eviction witness the turn's missed tokens book to the cold bucket, as in
// ObserveSplit.
func (o *Observer) ObserveLabeled(labels Labels, promptTokens, cacheablePrefixTokens, reusedPrefixTokens, eligiblePromptTokens int) {
	o.observeAttributed(labels, promptTokens, cacheablePrefixTokens, reusedPrefixTokens, 0, eligiblePromptTokens)
}

// LabeledStats is one (model, tenant, phase) row of the per-series snapshot. Beyond the
// depth columns it carries the SOURCE axis for the same series (#12886), so a caller gets
// whose traffic earned the reuse and where that value came from, from one coherent read.
type LabeledStats struct {
	Labels         Labels
	Turns          uint64
	PromptTokens   uint64
	EligibleTokens uint64
	ReusedTokens   uint64
	// Source columns. Zero for a series that has only ever been fed by a depth-axis tap
	// (including the legacy Observe / ObserveSplit / ObserveLabeled taps), never a fabricated
	// classification — a source-less turn stays absent rather than looking local.
	LocalComputeTokens     uint64
	LocalHitTokens         uint64
	ExternalTransferTokens uint64
	// UnknownSourceTokens is the explicit un-witnessed provenance for this series (#12886).
	UnknownSourceTokens uint64
}

// LabeledSnapshot returns the per-(model, tenant, phase) rows in deterministic order so a
// renderer emits a deterministic series order. Unlabeled legacy taps land on the
// ("unknown","unknown", "other") row, so summing any column across the rows reconciles exactly
// with the corresponding global counter in Snapshot(). Nil-safe like Snapshot; empty
// until the first observation.
func (o *Observer) LabeledSnapshot() []LabeledStats {
	if o == nil {
		return nil
	}
	o.mu.Lock()
	rows := o.labeledRowsLocked()
	o.mu.Unlock()
	sortLabeledRows(rows)
	return rows
}

// labeledRowsLocked copies every label row's depth AND source columns. Caller holds o.mu;
// the result is unsorted (the exported readers sort it outside the lock). It is the single
// row builder behind LabeledSnapshot and the coherent combined snapshot, so the two can
// never disagree on a row's fields.
func (o *Observer) labeledRowsLocked() []LabeledStats {
	rows := make([]LabeledStats, 0, len(o.byLabel))
	for labels, lt := range o.byLabel {
		rows = append(rows, LabeledStats{
			Labels:                 labels,
			Turns:                  lt.turns,
			PromptTokens:           lt.promptTokens,
			EligibleTokens:         lt.eligibleTokens,
			ReusedTokens:           lt.reusedTokens,
			LocalComputeTokens:     lt.srcLocalCompute,
			LocalHitTokens:         lt.srcLocalHit,
			ExternalTransferTokens: lt.srcExternalTransfer,
			UnknownSourceTokens:    lt.srcUnknown,
		})
	}
	return rows
}

// sortLabeledRows imposes the deterministic (model, tenant, phase) order a renderer needs to
// emit a stable series order.
func sortLabeledRows(rows []LabeledStats) {
	sort.Slice(rows, func(i, j int) bool {
		if rows[i].Labels.Model != rows[j].Labels.Model {
			return rows[i].Labels.Model < rows[j].Labels.Model
		}
		if rows[i].Labels.Tenant != rows[j].Labels.Tenant {
			return rows[i].Labels.Tenant < rows[j].Labels.Tenant
		}
		return rows[i].Labels.Phase < rows[j].Labels.Phase
	})
}

// SourceSplit is the caller's provenance decomposition of one turn's served tokens along
// the source axis (#3896 / #12886): how many prompt tokens were recomputed locally, served
// from a locally-resident prefix, or pulled across the fabric. A nil *SourceSplit (or a
// zero-valued one whose buckets do not account for the whole turn) is understood as an
// ABSENT source witness, and the turn's reuse books as SourceUnknown — never as local.
type SourceSplit struct {
	LocalCompute     int
	LocalHit         int
	ExternalTransfer int
}

// sourceBuckets clamps a caller's source split into non-negative buckets and returns the
// remainder of the turn's reused tokens as UNKNOWN. With a nil split (legacy input, no
// source evidence) the entire reused share books UNKNOWN — explicit, never classified
// local. The four returned values always sum to reusedPrefixTokens (already clamped >= 0),
// so the source axis reconciles exactly with the turn's depth-axis reuse.
func sourceBuckets(src *SourceSplit, reusedPrefixTokens int) (compute, hit, external, unknown uint64) {
	if reusedPrefixTokens < 0 {
		reusedPrefixTokens = 0
	}
	if src == nil {
		return 0, 0, 0, uint64(reusedPrefixTokens)
	}
	if src.LocalCompute > 0 {
		compute = uint64(src.LocalCompute)
	}
	if src.LocalHit > 0 {
		hit = uint64(src.LocalHit)
	}
	if src.ExternalTransfer > 0 {
		external = uint64(src.ExternalTransfer)
	}
	booked := compute + hit + external
	if booked > uint64(reusedPrefixTokens) {
		// Over-claim: cap at the reuse the depth axis actually saw, folding the excess back
		// to UNKNOWN rather than inflating any known source above the tokens that exist.
		compute, hit, external, unknown = 0, 0, 0, uint64(reusedPrefixTokens)
		return
	}
	unknown = uint64(reusedPrefixTokens) - booked
	return
}

// ObserveLabeledSource is the ATOMIC labeled-source observation (#12886). It records one
// served turn exactly like ObserveLabeled — the same #3390 clamps, eligibility denominator,
// regime/histogram bookkeeping, and (model, tenant, phase) depth row — and IN THE SAME
// o.mu critical section books the SOURCE axis both globally and on the label row. src is
// the caller's provenance split; pass nil for a legacy tap that carries no source evidence,
// and the turn's reuse books as explicit UNKNOWN rather than as a local hit or compute.
//
// This is the deliberate replacement for the non-atomic pattern of calling ObserveBySource
// and ObserveLabeled separately, under which a concurrent reader can observe a depth total
// and a source total that never co-existed.
func (o *Observer) ObserveLabeledSource(labels Labels, promptTokens, cacheablePrefixTokens, reusedPrefixTokens, eligiblePromptTokens int, src *SourceSplit) {
	o.observeAttributedSource(labels, promptTokens, cacheablePrefixTokens, reusedPrefixTokens, 0, eligiblePromptTokens, src, true)
}

// CombinedStats is one coherent reading of every axis the observer accumulates: the global
// DEPTH snapshot, the global SOURCE snapshot, and the per-label rows — each carrying both
// depth and source columns. Every field was read under a single o.mu acquisition, so the
// three axes cannot be mutually inconsistent (no depth total from before an observation
// beside a source total from after it).
type CombinedStats struct {
	// Depth is the global depth-axis snapshot, identical to Snapshot().
	Depth Stats
	// Source is the global provenance snapshot, identical to SourceSnapshot().
	Source SourceStats
	// Labels are the per-(model, tenant, phase) rows in deterministic order, identical to
	// LabeledSnapshot() — but read under the same lock as Depth and Source, so summing their
	// columns reconciles with both global axes in THIS reading.
	Labels []LabeledStats
}

// CombinedSnapshot returns the global depth, global source, and labeled rows from one lock
// acquisition — a coherent snapshot (#12886). It is the operation a caller must use when it
// needs to reconcile the axes: sequentially calling Snapshot, SourceSnapshot, and
// LabeledSnapshot is NOT a coherent snapshot, because an observation may land between any
// two of those calls and leave a reader with totals that never co-existed. The label rows
// are sorted deterministically, as LabeledSnapshot sorts them. Nil-safe like Snapshot.
func (o *Observer) CombinedSnapshot() CombinedStats {
	if o == nil {
		return CombinedStats{}
	}
	o.mu.Lock()
	cs := CombinedStats{
		Depth:  o.snapshotLocked(),
		Source: o.sourceSnapshotLocked(),
		Labels: o.labeledRowsLocked(),
	}
	o.mu.Unlock()
	sortLabeledRows(cs.Labels)
	return cs
}
