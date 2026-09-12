package gateway

import (
	"sort"
	"sync"
)

// crossagent_reuse.go — the cross-agent reuse axis of the agents pane (#12312), sibling of
// #2627's activity axis and #12311's hierarchical tree render. A multi-agent run (one
// coordinator plus parallel worker/tester/researcher subagents) shares a common system
// prompt, tool catalog, and repo context, so each subagent's prompt is largely a prefix
// the coordinator (or a peer) already prefilled. The activity axis answers "who is hot";
// this one answers "how much of the fan-out's prompt did the shared prefix already pay
// for" — the cross-agent reuse rate and the prefill latency it avoided.
//
// PROVENANCE-SAFE by construction: only the parent trace id, the subagent type token, and
// two integer token counts cross. No prompt text, no child identity beyond the trace id
// the pane already renders, no tool arguments.
//
// The coordinator's OWN turns are intra-agent reuse (already reported by the in-kernel
// cacheobs tap) and are deliberately EXCLUDED: a subagent turn is observed only when its
// trace differs from the parent trace. This keeps the rollup an honest cross-agent number
// rather than an aggregate that flatters itself with a coordinator's own warm cache.
//
// Rollup pricing: AvoidedPrefillLatencySeconds prices the cross-agent shared tokens at the
// coordinator's own measured prefill rate where one is supplied, so the "avoided" figure is
// a projection anchored on a real rate, never a fabricated speedup. With no rate witness
// the tokens are still counted and the seconds stay 0 (unmeasured, not invented).

// crossAgentCap bounds the ledger against a wide fan-out. A write that would exceed it
// evicts the coordinator with the fewest shared tokens first (the least cross-agent value);
// ties break on parent trace id for determinism. The /debug/vars read path additionally
// prunes to the live coordinator set (retain), so under normal operation the ledger tracks
// at most the live coordinators and the cap is the write-path backstop when reads are rare.
const crossAgentCap = 256

// crossAgentSeries accumulates one coordinator's cross-agent subagent turns. Every field is
// zero-valued when unknown so the wire keeps its omitempty tags; an all-zero series is never
// emitted.
type crossAgentSeries struct {
	subagents   int
	promptTotal uint64
	sharedTotal uint64
}

// crossAgentRow is the read projection for one coordinator: the lineage id the pane roots
// children under, the subagent census, the shared/prompt token totals, and the derived
// cross-agent reuse rate + avoided prefill latency.
type crossAgentRow struct {
	ParentSessionID              string
	SubagentCount                int
	PromptTokens                 uint64
	SharedTokens                 uint64
	AvoidedPrefillLatencySeconds float64
}

// ReusePct is the token-weighted cross-agent reuse rate for this coordinator in percent.
// 0 when no subagent prompt tokens were observed (never a phantom or NaN ratio).
func (r crossAgentRow) ReusePct() float64 {
	if r.PromptTokens == 0 {
		return 0
	}
	return float64(r.SharedTokens) / float64(r.PromptTokens) * 100
}

// crossAgentRollup is the fleet-wide fold over every coordinator series.
type crossAgentRollup struct {
	CrossAgentSubagentCount int
	CrossAgentPromptTokens  uint64
	CrossAgentSharedTokens  uint64
	// CrossAgentReusePct is the token-weighted fleet cross-agent reuse rate in percent.
	// 0 when no subagent prompt tokens were observed.
	CrossAgentReusePct float64
	// AvoidedPrefillLatencySeconds is the fleet-wide pricing of CrossAgentSharedTokens at
	// the observed prefill rate. 0 when no rate witness was supplied.
	AvoidedPrefillLatencySeconds float64
}

// crossAgentReuseLedger is the bounded per-coordinator registry behind the agents pane's
// cross-agent reuse cell. Every method is safe on a nil receiver (a bare Server that never
// went through New has none) and guarded by one mutex — each series is tiny and touched on
// the served path, so a single lock is cheaper than sharded bookkeeping.
type crossAgentReuseLedger struct {
	mu sync.Mutex
	// series keys on the PARENT (coordinator) trace id; child turns fold into their parent.
	series map[string]*crossAgentSeries
	// prefillSecondsPerToken is the coordinator's measured prefill rate witness (seconds
	// per prompt token) used to price avoided prefill; 0 when unmeasured.
	prefillSecondsPerToken float64
}

// newCrossAgentReuseLedger builds an empty ledger. New wires one onto every Server it
// returns.
func newCrossAgentReuseLedger() *crossAgentReuseLedger {
	return &crossAgentReuseLedger{series: map[string]*crossAgentSeries{}}
}

// getOrMakeLocked returns the series for parent, creating it (and evicting the
// lowest-shared coordinator first when at cap) if absent. The caller holds l.mu.
func (l *crossAgentReuseLedger) getOrMakeLocked(parent string) *crossAgentSeries {
	if s := l.series[parent]; s != nil {
		return s
	}
	if len(l.series) >= crossAgentCap {
		l.evictLeastSharedLocked()
	}
	s := &crossAgentSeries{}
	l.series[parent] = s
	return s
}

// evictLeastSharedLocked drops the coordinator series carrying the fewest shared tokens —
// the least cross-agent value — breaking ties on parent id for determinism. The caller
// holds l.mu and the map is known non-empty.
func (l *crossAgentReuseLedger) evictLeastSharedLocked() {
	var victim string
	var best uint64
	first := true
	for id, s := range l.series {
		if first || s.sharedTotal < best || (s.sharedTotal == best && id < victim) {
			victim, best, first = id, s.sharedTotal, false
		}
	}
	delete(l.series, victim)
}

// ObserveCrossAgentTurn records one subagent turn folded under parentTrace. A turn whose
// childTrace equals parentTrace is the coordinator's OWN turn — intra-agent reuse, already
// carried by the in-kernel cacheobs tap — and is ignored, so the ledger counts only genuine
// cross-agent reuse. promptTokens <= 0 is ignored (no turn to attribute); sharedTokens is
// clamped into [0, promptTokens] so a miscount can never push the ratio outside [0,1].
// subagentType is carried for the pane's role chip; this ledger keeps the census, not the
// per-type breakdown (the hierarchical render owns that).
func (l *crossAgentReuseLedger) ObserveCrossAgentTurn(parentTrace, childTrace, subagentType string, promptTokens, sharedTokens int) {
	if l == nil || parentTrace == "" || promptTokens <= 0 {
		return
	}
	if childTrace == parentTrace {
		return // coordinator's own turn: intra-agent, not cross-agent
	}
	if sharedTokens < 0 {
		sharedTokens = 0
	}
	if sharedTokens > promptTokens {
		sharedTokens = promptTokens
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	s := l.getOrMakeLocked(parentTrace)
	s.subagents++
	s.promptTotal += uint64(promptTokens)
	s.sharedTotal += uint64(sharedTokens)
}

// SetPrefillRate records the coordinator's measured prefill rate (seconds per prompt token)
// used to price avoided prefill latency. A non-positive rate leaves the price unmeasured
// (0) rather than inventing one.
func (l *crossAgentReuseLedger) SetPrefillRate(secondsPerToken float64) {
	if l == nil || secondsPerToken <= 0 {
		return
	}
	l.mu.Lock()
	l.prefillSecondsPerToken = secondsPerToken
	l.mu.Unlock()
}

// Snapshot returns the per-coordinator rows in deterministic (parent-id ascending) order so
// a renderer emits a stable series order. Nil-safe; empty until the first subagent turn.
func (l *crossAgentReuseLedger) Snapshot() []crossAgentRow {
	if l == nil {
		return nil
	}
	l.mu.Lock()
	rows := make([]crossAgentRow, 0, len(l.series))
	for parent, s := range l.series {
		rows = append(rows, l.rowLocked(parent, s))
	}
	l.mu.Unlock()
	sort.Slice(rows, func(i, j int) bool { return rows[i].ParentSessionID < rows[j].ParentSessionID })
	return rows
}

// Rollup folds every coordinator series into the fleet-wide cross-agent reuse rollup the
// /debug/vars sessions block exposes. Nil-safe; an idle ledger yields a zero rollup whose
// ratio is 0 (never a phantom or NaN ratio).
func (l *crossAgentReuseLedger) Rollup() crossAgentRollup {
	if l == nil {
		return crossAgentRollup{}
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	var roll crossAgentRollup
	for _, s := range l.series {
		roll.CrossAgentSubagentCount += s.subagents
		roll.CrossAgentPromptTokens += s.promptTotal
		roll.CrossAgentSharedTokens += s.sharedTotal
	}
	if roll.CrossAgentPromptTokens > 0 {
		roll.CrossAgentReusePct = float64(roll.CrossAgentSharedTokens) / float64(roll.CrossAgentPromptTokens) * 100
		roll.AvoidedPrefillLatencySeconds = float64(roll.CrossAgentSharedTokens) * l.prefillSecondsPerToken
	}
	return roll
}

// rowLocked projects one series; the caller holds l.mu.
func (l *crossAgentReuseLedger) rowLocked(parent string, s *crossAgentSeries) crossAgentRow {
	row := crossAgentRow{
		ParentSessionID: parent,
		SubagentCount:   s.subagents,
		PromptTokens:    s.promptTotal,
		SharedTokens:    s.sharedTotal,
	}
	row.AvoidedPrefillLatencySeconds = float64(s.sharedTotal) * l.prefillSecondsPerToken
	return row
}

// retain drops every series whose parent is not in live (the current non-stopped
// coordinator set), folding vanished coordinators so the ledger tracks at most the live
// set. Called on the /debug/vars read path with the session list already in hand.
func (l *crossAgentReuseLedger) retain(live map[string]struct{}) {
	if l == nil {
		return
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	for id := range l.series {
		if _, ok := live[id]; !ok {
			delete(l.series, id)
		}
	}
}
