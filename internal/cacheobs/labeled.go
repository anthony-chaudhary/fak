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

// LabeledStats is one (model, tenant, phase) row of the per-series snapshot.
type LabeledStats struct {
	Labels         Labels
	Turns          uint64
	PromptTokens   uint64
	EligibleTokens uint64
	ReusedTokens   uint64
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
	rows := make([]LabeledStats, 0, len(o.byLabel))
	for labels, lt := range o.byLabel {
		rows = append(rows, LabeledStats{
			Labels:         labels,
			Turns:          lt.turns,
			PromptTokens:   lt.promptTokens,
			EligibleTokens: lt.eligibleTokens,
			ReusedTokens:   lt.reusedTokens,
		})
	}
	o.mu.Unlock()
	sort.Slice(rows, func(i, j int) bool {
		if rows[i].Labels.Model != rows[j].Labels.Model {
			return rows[i].Labels.Model < rows[j].Labels.Model
		}
		if rows[i].Labels.Tenant != rows[j].Labels.Tenant {
			return rows[i].Labels.Tenant < rows[j].Labels.Tenant
		}
		return rows[i].Labels.Phase < rows[j].Labels.Phase
	})
	return rows
}
