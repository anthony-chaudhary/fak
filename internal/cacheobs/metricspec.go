package cacheobs

// metricspec.go — a DECLARATIVE map-reduce registry for the cache metrics this package
// reports, plus a fold that reconciles two independent collection paths against the SAME
// registry so they cannot silently disagree.
//
// Borrow (study-repo, LMCache @e38ee415): `lmcache/usage_telemetry/metric_specs.py:26-78`
// declares each telemetry metric as a data row — `MetricSpec{event_type, field, extract,
// reduce}` — where `extract` maps one event to a numeric sample and `reduce` folds an
// interval's samples into one wire field. LMCache runs two reporters (single-process and
// multiprocess); both consume the SAME spec list, and a startup check
// (`mp_continuous.py:99-112`) asserts the specs cover every wire field exactly once. Drift
// between the two reporters is then structurally impossible, because there is only one
// definition of each metric and both reporters read it.
//
// The fak seam this closes: cacheobs.go computes its counters imperatively (Observe*
// mutates hand-written fields; Snapshot reads them back). A metric that a second emitter —
// a batch re-aggregator, an off-box mirror, a Python control-pane twin — recomputes has no
// single declaration both sides share, so the two can drift field-for-field with nothing to
// catch it. This file introduces that single declaration:
//
//   - MetricSpec is one metric as a data row: which Event it consumes, the target Field, an
//     Extract map (event -> sample) and a Reduce fold (samples -> field value).
//   - DefaultSpecs is the registry covering the additive snapshot counters. CheckCoverage
//     asserts a spec set covers a schema exactly once (the mp_continuous.py:99-112 check).
//   - SpecFold and ObserverFold are two collection paths for the SAME registry: SpecFold
//     reduces the event stream directly; ObserverFold routes the events through the existing
//     imperative Observer and reads its Snapshot. Reconcile folds the events through every
//     reporter and returns each field where a reporter disagrees with the reference — empty
//     means the paths agree field-for-field. A divergence is reported loudly, never hidden.
//
// Route: inspire (clean-room Go; both Apache-2.0). Source cited, no bytes vendored (#5267).

import (
	"errors"
	"fmt"
	"math"
	"sort"
	"strings"
)

// EventKind names which event on the observation bus a MetricSpec consumes. The vocabulary
// is closed: a spec declares exactly one kind, and a fold only feeds it events of that kind,
// so a metric can never accidentally reduce samples from an unrelated event.
type EventKind int

const (
	// EventTurn is one served in-kernel turn — the same datum the imperative Observe path
	// books (prompt / cacheable / reused / eligible token counts for the turn).
	EventTurn EventKind = iota
	// EventTierAccess is one cache access against one explicit tier (#6422) — the datum the
	// imperative ObserveTier path books, carried on the Access field.
	EventTierAccess
)

// Event is one datum on the observation bus. It carries the per-turn token counts a
// MetricSpec's Extract reads. Keeping the readings on a plain value (rather than behind the
// Observer's methods) is what lets a spec be reduced by more than one collection path.
type Event struct {
	Kind            EventKind
	PromptTokens    float64
	CacheableTokens float64
	ReusedTokens    float64
	EligibleTokens  float64
	// Access carries the EventTierAccess dimensions (tiers.go) — the tier, operation,
	// outcome, bytes, latency, and coarse backend class of one cache access. Zero and unread
	// for an EventTurn, exactly as the token fields are for an EventTierAccess: a spec only
	// ever sees events of the kind it declares.
	Access TierAccess
}

// MetricSpec declares one cache metric as a map-reduce data row (LMCache
// metric_specs.py:26-78): Event is the event kind it consumes, Field is the target snapshot
// field, Extract maps an event to one numeric sample, and Reduce folds an interval's samples
// into the field's value. A metric is a data row, not new plumbing — a second collection
// path reproduces it by reading this same row, not by re-implementing the arithmetic.
type MetricSpec struct {
	Event   EventKind
	Field   string
	Extract func(Event) float64
	Reduce  func([]float64) float64
}

// The additive snapshot counter fields the default registry covers. Held as named constants
// so the specs, the schema, and the ObserverFold switch all bind to one spelling — a typo
// cannot mint a second field that Reconcile would then flag against itself.
const (
	FieldTurns           = "turns"
	FieldPromptTokens    = "prompt_tokens"
	FieldCacheableTokens = "cacheable_tokens"
	FieldReusedTokens    = "reused_tokens"
	FieldEligibleTokens  = "eligible_tokens"
)

// SnapshotFields is the additive counter schema the default registry must cover exactly
// once — the wire fields a second emitter has to reproduce. Ratios (ReuseRatio and kin) are
// deliberately absent: they are pure functions of these covered counters (see ReuseRatioFrom),
// so covering the additive base is enough to pin every derived rate too.
func SnapshotFields() []string {
	return []string{
		FieldTurns,
		FieldPromptTokens,
		FieldCacheableTokens,
		FieldReusedTokens,
		FieldEligibleTokens,
	}
}

// sumSamples folds a sample list by addition — the reduce every additive counter uses. A
// count metric (Turns) reaches it via an Extract that emits 1 per event, so the sum is the
// number of events.
func sumSamples(xs []float64) float64 {
	var s float64
	for _, x := range xs {
		s += x
	}
	return s
}

// DefaultSpecs is the declarative registry for the additive snapshot counters. Each field is
// declared exactly once (CheckCoverage enforces it against SnapshotFields), and every spec
// reduces by summation — Turns counts events, the token fields sum their per-turn readings —
// so the registry is the single definition both SpecFold and ObserverFold reproduce.
func DefaultSpecs() []MetricSpec {
	return []MetricSpec{
		{Event: EventTurn, Field: FieldTurns, Extract: func(Event) float64 { return 1 }, Reduce: sumSamples},
		{Event: EventTurn, Field: FieldPromptTokens, Extract: func(e Event) float64 { return e.PromptTokens }, Reduce: sumSamples},
		{Event: EventTurn, Field: FieldCacheableTokens, Extract: func(e Event) float64 { return e.CacheableTokens }, Reduce: sumSamples},
		{Event: EventTurn, Field: FieldReusedTokens, Extract: func(e Event) float64 { return e.ReusedTokens }, Reduce: sumSamples},
		{Event: EventTurn, Field: FieldEligibleTokens, Extract: func(e Event) float64 { return e.EligibleTokens }, Reduce: sumSamples},
	}
}

// CheckCoverage returns an error unless specs cover every field in schema EXACTLY once and
// target no field outside it — the same structural guard LMCache runs at
// mp_continuous.py:99-112, which makes an under- or over-covered wire schema a startup
// failure rather than a silent reporting gap. A missing field means a metric no reporter can
// reproduce; a doubly-covered field means two rival definitions of one metric; an unknown
// field means a spec that no schema slot receives. All three are collected and returned
// together (sorted) so the caller sees every problem at once, not just the first.
func CheckCoverage(specs []MetricSpec, schema []string) error {
	covered := make(map[string]int, len(specs))
	for _, s := range specs {
		covered[s.Field]++
	}
	want := make(map[string]bool, len(schema))
	for _, f := range schema {
		want[f] = true
	}
	var problems []string
	for f := range want {
		switch covered[f] {
		case 1:
			// exactly once: good
		case 0:
			problems = append(problems, fmt.Sprintf("field %q has no spec; declare a MetricSpec covering this field in DefaultSpecs", f))
		default:
			problems = append(problems, fmt.Sprintf("field %q covered by %d specs, want exactly 1; remove duplicate MetricSpec definitions", f, covered[f]))
		}
	}
	for _, s := range specs {
		if !want[s.Field] {
			problems = append(problems, fmt.Sprintf("spec targets unknown field %q; remove the unmapped MetricSpec or register the field in schema", s.Field))
		}
	}
	if len(problems) == 0 {
		return nil
	}
	sort.Strings(problems)
	return errors.New("metric spec coverage: " + strings.Join(problems, "; "))
}

// Report is one collection path's result: field name -> reduced value.
type Report map[string]float64

// NamedReporter is a collection path that folds events through a spec registry into a Report.
// Name is what Reconcile attributes a disagreement to, so a divergence points at the path
// that produced it.
type NamedReporter struct {
	Name string
	Fold func(specs []MetricSpec, events []Event) Report
}

// SpecFold is the reference collection path: it reduces the event stream directly through the
// registry — for each spec, map Extract over the events of that spec's kind, then Reduce the
// samples. This is the metric's definition executed literally, with no imperative counters in
// between; every other reporter is judged against it.
func SpecFold(specs []MetricSpec, events []Event) Report {
	out := make(Report, len(specs))
	for _, spec := range specs {
		var samples []float64
		for _, e := range events {
			if e.Kind == spec.Event {
				samples = append(samples, spec.Extract(e))
			}
		}
		out[spec.Field] = spec.Reduce(samples)
	}
	return out
}

// ObserverFold is the SECOND collection path: it routes each turn event through the existing
// imperative Observer (the hand-written counters in cacheobs.go) and reads the additive
// fields back off Snapshot. It reproduces the registry's fields by a completely different
// mechanism than SpecFold, which is exactly why reconciling the two is worth doing — if the
// imperative counters and the declarative reduction ever drift, Reconcile names the field.
// A spec whose Field the Observer does not expose reduces to 0 here, so a registry that grows
// a field the imperative path forgot to track surfaces as a divergence rather than silence.
func ObserverFold(specs []MetricSpec, events []Event) Report {
	o := New()
	for _, e := range events {
		if e.Kind != EventTurn {
			continue
		}
		o.ObserveLabeled(Labels{}, int(e.PromptTokens), int(e.CacheableTokens), int(e.ReusedTokens), int(e.EligibleTokens))
	}
	s := o.Snapshot()
	byField := map[string]float64{
		FieldTurns:           float64(s.Turns),
		FieldPromptTokens:    float64(s.PromptTokens),
		FieldCacheableTokens: float64(s.CacheableTokens),
		FieldReusedTokens:    float64(s.ReusedTokens),
		FieldEligibleTokens:  float64(s.EligibleTokens),
	}
	out := make(Report, len(specs))
	for _, spec := range specs {
		out[spec.Field] = byField[spec.Field] // absent -> 0, surfaced as a divergence
	}
	return out
}

// Disagreement is one field on which a reporter's reduced value differs from the reference
// reporter's. It carries both numbers so a reader never has to re-run the fold to see how far
// apart the two paths landed.
type Disagreement struct {
	Field     string
	Reporter  string
	Reference float64
	Got       float64
}

// reconcileEpsilon is the absolute float tolerance below which two paths are treated as
// agreeing — the additive counters are exact integers as floats, so this only absorbs the
// rounding a derived reducer might introduce, never a real one-token drift.
const reconcileEpsilon = 1e-9

// Reconcile folds events through every reporter using the SAME spec registry and returns each
// field where a non-reference reporter disagrees with the reference (the first reporter)
// beyond reconcileEpsilon. An empty result means every path computed every covered field
// identically — the agreement the shared registry is meant to guarantee. A non-empty result
// is the loud failure: it names the reporter, the field, and both values, so a divergence
// between two collection paths can never pass unnoticed. Fields are compared in registry
// order and the result is sorted (reporter, field) so the report is deterministic.
func Reconcile(specs []MetricSpec, events []Event, reporters ...NamedReporter) []Disagreement {
	if len(reporters) < 2 {
		return nil // nothing to cross-check against
	}
	reference := reporters[0].Fold(specs, events)
	var out []Disagreement
	for _, r := range reporters[1:] {
		got := r.Fold(specs, events)
		for _, spec := range specs {
			ref := reference[spec.Field]
			val := got[spec.Field]
			if math.Abs(ref-val) > reconcileEpsilon {
				out = append(out, Disagreement{
					Field:     spec.Field,
					Reporter:  r.Name,
					Reference: ref,
					Got:       val,
				})
			}
		}
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Reporter != out[j].Reporter {
			return out[i].Reporter < out[j].Reporter
		}
		return out[i].Field < out[j].Field
	})
	return out
}

// ReuseRatioFrom derives the realized token hit-ratio — reused over prompt tokens — from a
// spec-reduced Report. The headline ratio is thus a pure function of the covered additive
// fields, not a separately-maintained counter, so it cannot drift from the counters it is
// built on: proving parity of the base fields proves parity of the ratio for free. 0 when no
// prompt tokens were reduced (an idle interval reports no phantom ratio).
func ReuseRatioFrom(r Report) float64 {
	prompt := r[FieldPromptTokens]
	if prompt <= 0 {
		return 0
	}
	return r[FieldReusedTokens] / prompt
}
