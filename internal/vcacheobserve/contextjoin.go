package vcacheobserve

import (
	"sort"
)

// contextjoin.go — issue #1607: JOIN provider prompt-cache telemetry (Turn, already
// shipped by this package) to the managed-context LIFECYCLE — resets, compactions,
// page faults, prefix mutations — so a report can answer the issue's done condition:
// "did a cost change come from CONTEXT PLANNING or from PROVIDER CACHE BEHAVIOR?"
//
// REUSE, NOT REINVENTION. The issue explicitly wants this correlated against the
// event/type vocabulary the rest of the kernel already ships, not a parallel one:
//
//   - context reset / compaction: internal/resume.Strategy (resume_full/cut/reset) is
//     the shipped closed vocabulary for a re-entry decision; internal/ctxplan's
//     CompactionView names the "compaction" objective a Plan can carry.
//   - page fault: internal/ctxplan.PageFaultOutcome (page_in/query_user/deny/
//     continue_with_pointer) from the #1587 page-fault protocol.
//   - prefix mutation: internal/cachemeta.Diverge's TurnDivergence (the §A3
//     prefix-stability linter already computes exactly "did the byte-stable prefix
//     break, and how many tokens got re-billed").
//
// LifecycleEvent is a thin, timestamped ADAPTER over those four vocabularies — it
// does not redefine them, it echoes the caller's own closed-vocabulary token into one
// join-able stream keyed by wall-clock millis + prefix family, matching the Turn shape
// this package already uses. A caller holding a real resume.Report, ctxplan.Plan, or
// cachemeta.TurnDivergence builds a LifecycleEvent with the constructor for that source
// (ResetEvent/CompactionEvent/PageFaultEventFrom/PrefixMutationEvent below) so the enum
// value is never hand-typed as a string literal.
//
// THE JOIN. JoinContext walks each prefix family's turns in time order, and at every
// turn-to-turn transition it (a) measures whether a cost-relevant CHANGE occurred
// (a cache-creation spike, a hit-rate collapse — see costChange) and (b) checks whether
// a LifecycleEvent for that family falls inside the CorrelationWindow ending at the
// transition. A change with a nearby event is attributed CausePlanning (the deliberate
// reset/compaction/page-fault/prefix-mutation explains the cost); a change with no
// nearby event is attributed CauseProviderBehavior (a natural cache miss/TTL expiry
// unrelated to any context action fak took). This is the whole "join" the issue asks
// for: two independent streams, correlated by (family, time), reduced to one
// attribution per observed change.
//
// Tier: mechanism (2), same as the rest of vcacheobserve — it composes tier-1
// ctxplan/cachemeta/resume vocabulary tokens (as plain string echoes, no import
// required for the enum values themselves) plus this package's own Turn type.

// LifecycleEventKind is the closed vocabulary of managed-context events this join
// layer correlates against provider cache telemetry. Each kind names a REAL event type
// shipped elsewhere in the kernel; this package never invents a fifth kind without a
// corresponding upstream source.
type LifecycleEventKind string

const (
	// EventContextReset mirrors internal/resume.Strategy: a session re-entry decision
	// (resume_full / cut / reset) that intentionally reshapes what is resident before
	// the next turn — the "context reset" the issue names.
	EventContextReset LifecycleEventKind = "context_reset"
	// EventCompaction mirrors internal/ctxplan's compaction objective (CompactionView):
	// a lossy fold of elided spans that destroys their recovery handles.
	EventCompaction LifecycleEventKind = "compaction"
	// EventPageFault mirrors internal/ctxplan.PageFaultOutcome: a forecast MISS decided
	// by DecidePageFault (page_in / query_user / deny / continue_with_pointer).
	EventPageFault LifecycleEventKind = "page_fault"
	// EventPrefixMutation mirrors internal/cachemeta's §A3 divergence: the turn's
	// provider-cacheable prefix broke against the previous turn (TurnDivergence with
	// LostTokens > 0), independent of any fak-authored decision.
	EventPrefixMutation LifecycleEventKind = "prefix_mutation"
)

var validLifecycleEventKinds = map[LifecycleEventKind]bool{
	EventContextReset:   true,
	EventCompaction:     true,
	EventPageFault:      true,
	EventPrefixMutation: true,
}

// ValidLifecycleEventKind reports whether k is a member of the closed vocabulary.
func ValidLifecycleEventKind(k LifecycleEventKind) bool { return validLifecycleEventKinds[k] }

// LifecycleEvent is one managed-context lifecycle occurrence, timestamped and tagged
// with the prefix family it applies to (the same Family key Turn uses), so it can be
// joined against the provider-cache Turn stream by (family, time) alone — no shared
// object identity required between the two sources.
type LifecycleEvent struct {
	Kind       LifecycleEventKind `json:"kind"`
	Family     string             `json:"family"`
	UnixMillis int64              `json:"unix_millis"`
	// Outcome is the source vocabulary's own token echoed verbatim (e.g. "reset",
	// "compaction", "page_in", "deny") — kept as a free string so this package never
	// forks the upstream enum; ValidLifecycleEventKind is the only closed check this
	// package itself enforces.
	Outcome string `json:"outcome,omitempty"`
	Detail  string `json:"detail,omitempty"`
}

// ResetEvent adapts an internal/resume.Strategy decision into a LifecycleEvent. Pass
// the Strategy's string value (e.g. string(resume.StrategyReset)) as outcome so this
// package stays import-free of internal/resume while still echoing its real token.
func ResetEvent(family string, unixMillis int64, outcome, detail string) LifecycleEvent {
	return LifecycleEvent{Kind: EventContextReset, Family: family, UnixMillis: unixMillis, Outcome: outcome, Detail: detail}
}

// CompactionEvent adapts an internal/ctxplan compaction (CompactionView's "compaction"
// Plan.Objective, or any lossy fold) into a LifecycleEvent.
func CompactionEvent(family string, unixMillis int64, detail string) LifecycleEvent {
	return LifecycleEvent{Kind: EventCompaction, Family: family, UnixMillis: unixMillis, Outcome: "compaction", Detail: detail}
}

// PageFaultEventFrom adapts an internal/ctxplan.PageFaultDecision into a
// LifecycleEvent. Pass the decision's Outcome.String() so the real closed-vocabulary
// token (page_in/query_user/deny/continue_with_pointer) rides along unchanged.
func PageFaultEventFrom(family string, unixMillis int64, outcome, detail string) LifecycleEvent {
	return LifecycleEvent{Kind: EventPageFault, Family: family, UnixMillis: unixMillis, Outcome: outcome, Detail: detail}
}

// PrefixMutationEvent adapts an internal/cachemeta.TurnDivergence (a broken byte-stable
// prefix) into a LifecycleEvent. lostTokens is carried in Detail for the human report;
// the join itself only needs the timestamp.
func PrefixMutationEvent(family string, unixMillis int64, lostTokens int64, detail string) LifecycleEvent {
	return LifecycleEvent{Kind: EventPrefixMutation, Family: family, UnixMillis: unixMillis, Outcome: "diverged", Detail: detail}
}

// AttributionCause is the closed vocabulary JoinContext assigns to every observed cost
// change — the answer to the issue's done condition ("did a cost change come from
// context planning or from provider cache behavior").
type AttributionCause string

const (
	// CausePlanning: a LifecycleEvent for this family falls inside the correlation
	// window ending at the change — the change is EXPLAINED by a deliberate context
	// action fak (or the session harness) took.
	CausePlanning AttributionCause = "context_planning"
	// CauseProviderBehavior: no LifecycleEvent falls in the window — the change is
	// natural provider-cache behavior (a TTL expiry, an ordinary cold miss) with no
	// fak-authored context action to blame or credit.
	CauseProviderBehavior AttributionCause = "provider_cache_behavior"
	// CauseNone: no cost-relevant change was detected at this transition at all — most
	// turn-to-turn steps in a steady warm session land here.
	CauseNone AttributionCause = "none"
)

// CorrelationWindow bounds how close (in wall-clock millis) a LifecycleEvent must be
// to a turn transition to count as its explanation. Both bounds are inclusive.
type CorrelationWindow struct {
	// BeforeMillis: an event up to this many millis BEFORE the transition still
	// explains it (the common case: a reset/compaction happens, then the next turn's
	// telemetry shows the consequence).
	BeforeMillis int64
	// AfterMillis: an event up to this many millis AFTER the transition still counts
	// (a page-fault decision logged fractionally after the telemetry sample that
	// triggered it, e.g. clock skew between the provider response and the harness's
	// own event timestamp).
	AfterMillis int64
}

// DefaultCorrelationWindow is a 5-minute-TTL-scaled window: a context event up to one
// provider TTL (5 minutes) before the change, or up to 5 seconds after it (covers
// same-tick logging order), still counts as the explanation. This mirrors the same
// ttl5mMillis constant the warmth-belief estimator already uses elsewhere in this
// package, so the join's notion of "nearby" agrees with the rest of vcacheobserve.
func DefaultCorrelationWindow() CorrelationWindow {
	return CorrelationWindow{BeforeMillis: ttl5mMillis, AfterMillis: 5000}
}

func (w CorrelationWindow) normalized() CorrelationWindow {
	if w.BeforeMillis <= 0 && w.AfterMillis <= 0 {
		return DefaultCorrelationWindow()
	}
	if w.BeforeMillis < 0 {
		w.BeforeMillis = 0
	}
	if w.AfterMillis < 0 {
		w.AfterMillis = 0
	}
	return w
}

// contains reports whether eventMillis falls within the window ending at transitionMillis.
func (w CorrelationWindow) contains(transitionMillis, eventMillis int64) bool {
	delta := transitionMillis - eventMillis
	if delta >= 0 {
		return delta <= w.BeforeMillis
	}
	return -delta <= w.AfterMillis
}

// CostChangeKind is the closed vocabulary of turn-to-turn changes JoinContext treats as
// cost-relevant and worth attributing. Anything not matching one of these is CauseNone.
type CostChangeKind string

const (
	// ChangeCacheCreateSpike: cache_creation jumped materially versus the prior turn —
	// the prefix was (re)written at the WRITE multiplier instead of read.
	ChangeCacheCreateSpike CostChangeKind = "cache_create_spike"
	// ChangeHitRateDrop: this turn's per-turn hit rate (cache_read / resident tokens)
	// fell materially versus the prior turn — fewer tokens than expected were served
	// from cache.
	ChangeHitRateDrop CostChangeKind = "hit_rate_drop"
)

// spikeRatio is the minimum multiple of the family's baseline cache_creation a turn
// must exceed to count as a spike (rather than routine incremental writes).
const spikeRatio = 3.0

// hitRateDropAbs is the minimum absolute drop in per-turn hit rate (0..1 scale) versus
// the immediately preceding turn to count as a drop.
const hitRateDropAbs = 0.25

// AttributedChange is one detected cost-relevant change plus its attribution — the row
// of the join's output table.
type AttributedChange struct {
	Family       string           `json:"family"`
	UnixMillis   int64            `json:"unix_millis"`
	Change       CostChangeKind   `json:"change"`
	Cause        AttributionCause `json:"cause"`
	Detail       string           `json:"detail"`
	MatchedEvent *LifecycleEvent  `json:"matched_event,omitempty"`
	PriorTurn    Turn             `json:"-"`
	Turn         Turn             `json:"-"`
}

// JoinInput is everything JoinContext needs: the provider-cache Turn stream this
// package already ingests, the lifecycle events to correlate against it, and the
// correlation window (zero value normalizes to DefaultCorrelationWindow).
type JoinInput struct {
	Turns  []Turn            `json:"turns"`
	Events []LifecycleEvent  `json:"events"`
	Window CorrelationWindow `json:"window"`
}

// JoinReport is the queryable/renderable result: one AttributedChange per detected
// cost-relevant transition, plus a Summary a caller can headline without walking every
// row, plus the schema/turns/event counts for a self-describing artifact.
type JoinReport struct {
	Schema  string             `json:"schema"`
	Turns   int                `json:"turns"`
	Events  int                `json:"events"`
	Changes []AttributedChange `json:"changes"`
	Summary JoinSummary        `json:"summary"`
}

// JoinSummary folds Changes into the headline counts the done condition asks for: how
// much of the observed cost movement is explained by context planning versus provider
// cache behavior.
type JoinSummary struct {
	TotalChanges       int `json:"total_changes"`
	PlanningAttributed int `json:"planning_attributed"`
	ProviderAttributed int `json:"provider_attributed"`
}

// JoinSchema is the versioned report contract for `fak vcache context-join --json`.
const JoinSchema = "fak.vcache.contextjoin.v1"

// JoinContext is the pure correlation fold: given a stream of provider cache samples
// and a stream of context lifecycle events, it detects cost-relevant changes within
// each prefix family and attributes each to context planning (a nearby lifecycle
// event) or provider cache behavior (no nearby event). Deterministic: same input,
// same output, no clock, no I/O.
func JoinContext(in JoinInput) JoinReport {
	window := in.Window.normalized()
	rep := JoinReport{Schema: JoinSchema, Turns: len(in.Turns), Events: len(in.Events)}
	if len(in.Turns) == 0 {
		return rep
	}

	byFamily := map[string][]Turn{}
	order := []string{}
	for _, t := range in.Turns {
		if _, ok := byFamily[t.Family]; !ok {
			order = append(order, t.Family)
		}
		byFamily[t.Family] = append(byFamily[t.Family], t)
	}
	eventsByFamily := map[string][]LifecycleEvent{}
	for _, e := range in.Events {
		eventsByFamily[e.Family] = append(eventsByFamily[e.Family], e)
	}

	sort.Strings(order)
	var changes []AttributedChange
	for _, fam := range order {
		ts := append([]Turn(nil), byFamily[fam]...)
		sort.SliceStable(ts, func(i, j int) bool { return ts[i].UnixMillis < ts[j].UnixMillis })
		evs := eventsByFamily[fam]
		changes = append(changes, detectFamilyChanges(fam, ts, evs, window)...)
	}
	rep.Changes = changes
	rep.Summary = summarizeChanges(changes)
	return rep
}

// detectFamilyChanges walks one family's time-ordered turns and emits an
// AttributedChange for every turn-to-turn transition that trips a cost-change
// detector, matched against that family's lifecycle events.
func detectFamilyChanges(family string, ts []Turn, evs []LifecycleEvent, window CorrelationWindow) []AttributedChange {
	if len(ts) < 2 {
		return nil
	}
	// Baseline cache_creation is the running mean of positive cache_creation values
	// seen so far in the family's INCREMENTAL writes (turn index >= 1) — the "routine
	// write size so far" the NEXT turn is measured against. Turn 0 is deliberately
	// excluded: it is the initial cold warm of the whole prefix, not a routine
	// incremental write, so it would swamp the baseline and mask a real re-warm spike
	// later in the same family. The turn under test is never included in its own
	// baseline (no self-inclusion, no lookahead).
	var baseSum float64
	var baseN int

	var out []AttributedChange
	for i := 1; i < len(ts); i++ {
		prev, cur := ts[i-1], ts[i]
		// Fold every incremental turn strictly before cur (index >= 1, up to and
		// including prev) into the baseline before judging cur, so cur is compared
		// against the full incremental history that precedes it — never against
		// itself, never against the initial cold-warm turn 0.
		if i-1 >= 1 && prev.CacheCreation > 0 {
			baseSum += float64(prev.CacheCreation)
			baseN++
		}
		var baseline float64
		if baseN > 0 {
			baseline = baseSum / float64(baseN)
		}
		if kind, detail, ok := detectSpike(cur, baseline, baseN); ok {
			out = append(out, attribute(family, cur, prev, kind, detail, evs, window))
		}
		if kind, detail, ok := detectHitRateDrop(prev, cur); ok {
			out = append(out, attribute(family, cur, prev, kind, detail, evs, window))
		}
	}
	return out
}

func detectSpike(cur Turn, baseline float64, baseN int) (CostChangeKind, string, bool) {
	if baseN < 2 || baseline <= 0 {
		return "", "", false // not enough history to call anything a spike
	}
	if float64(cur.CacheCreation) < baseline*spikeRatio {
		return "", "", false
	}
	return ChangeCacheCreateSpike, cacheSpikeDetail(cur.CacheCreation, baseline), true
}

func cacheSpikeDetail(created int64, baseline float64) string {
	return "cache_creation " + itoa(created) + " vs family baseline ~" + itoa(int64(baseline+0.5))
}

func detectHitRateDrop(prev, cur Turn) (CostChangeKind, string, bool) {
	prevRate, prevOK := turnHitRate(prev)
	curRate, curOK := turnHitRate(cur)
	if !prevOK || !curOK {
		return "", "", false
	}
	if prevRate-curRate < hitRateDropAbs {
		return "", "", false
	}
	return ChangeHitRateDrop, "hit rate fell from " + pct(prevRate) + " to " + pct(curRate), true
}

// turnHitRate is a turn's own cache_read share of its resident tokens (input + read +
// creation) — a per-turn analog of Family.HitRate, defined only when the turn has any
// resident tokens at all.
func turnHitRate(t Turn) (float64, bool) {
	resident := float64(t.InputTokens + t.CacheRead + t.CacheCreation)
	if resident <= 0 {
		return 0, false
	}
	return float64(t.CacheRead) / resident, true
}

// attribute matches a detected change against the family's lifecycle events within the
// window and returns the finished AttributedChange. The closest matching event (by
// absolute time delta) is recorded when more than one falls in the window, so the
// report cites the most likely explanation rather than an arbitrary one.
func attribute(family string, cur, prev Turn, kind CostChangeKind, detail string, evs []LifecycleEvent, window CorrelationWindow) AttributedChange {
	ac := AttributedChange{
		Family: family, UnixMillis: cur.UnixMillis, Change: kind, Detail: detail,
		PriorTurn: prev, Turn: cur,
	}
	var best *LifecycleEvent
	var bestDelta int64 = -1
	for i := range evs {
		e := evs[i]
		if !window.contains(cur.UnixMillis, e.UnixMillis) {
			continue
		}
		delta := abs64(cur.UnixMillis - e.UnixMillis)
		if best == nil || delta < bestDelta {
			ev := e
			best = &ev
			bestDelta = delta
		}
	}
	if best != nil {
		ac.Cause = CausePlanning
		ac.MatchedEvent = best
	} else {
		ac.Cause = CauseProviderBehavior
	}
	return ac
}

func summarizeChanges(changes []AttributedChange) JoinSummary {
	var s JoinSummary
	s.TotalChanges = len(changes)
	for _, c := range changes {
		switch c.Cause {
		case CausePlanning:
			s.PlanningAttributed++
		case CauseProviderBehavior:
			s.ProviderAttributed++
		}
	}
	return s
}

func abs64(n int64) int64 {
	if n < 0 {
		return -n
	}
	return n
}

func itoa(n int64) string {
	if n == 0 {
		return "0"
	}
	neg := n < 0
	if neg {
		n = -n
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		i--
		buf[i] = '-'
	}
	return string(buf[i:])
}

func pct(f float64) string {
	// Render as an integer percentage for a compact human detail line.
	v := int64(f*100 + 0.5)
	return itoa(v) + "%"
}
