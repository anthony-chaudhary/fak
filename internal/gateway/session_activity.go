package gateway

import (
	"sync"
	"time"
)

// session_activity.go — the activity axis of the agents pane (#2627), sibling of #2250's
// tokens axis under the sidecar epic #2209 ("one pane, every agent"). The agents sub-pane
// already shows each live session's lineage + remaining budget (debugSessionVars), but
// nothing about what a session is doing RIGHT NOW. This keeps a bounded, per-trace
// activity record — stamped off the adjudication path the gateway already walks — so the
// pane can answer "who is hot / who is stuck / who is idle" without a second registry or
// any transcript read.
//
// PAYLOAD-FREE by construction: only the tool NAME crosses (the identifier, never its
// arguments or any result text), matching the pane's existing redaction-safe guarantee.
//
// PER-TRACE, not per-sub-agent: it equals per-sub-agent only under distinct-trace
// subagents — the same fence #2250 carries. spawn_count counts subagent-SHAPED tool CALLS
// admitted for the trace: an activity signal ("is it fanning out?"), NOT a live-child
// census (#2397's registry owns that).

// traceActivity is the mutable per-trace record. Times are unix seconds (the pane renders
// whole-second ages). inflightSinceUnix is 0 when no served turn is open; lastActivityUnix
// is the moment of the last adjudication. The two are mutually exclusive by construction —
// a turn in flight has not yet been adjudicated — so a snapshot yields in-flight OR idle,
// never both.
type traceActivity struct {
	lastTool          string
	spawnCount        int
	lastActivityUnix  int64
	inflightSinceUnix int64
}

// traceActivityView is the read projection debugSessions folds onto a session row. Every
// field is zero-valued when unknown so the wire keeps its omitempty tags: a trace with no
// record contributes nothing and the pre-activity wire shape stays byte-identical.
type traceActivityView struct {
	LastTool        string
	SpawnCount      int
	InflightSeconds int64
	IdleSeconds     int64
}

// sessionActivityCap bounds the record against a wide sub-agent fan-out: a write that
// would exceed it evicts the least-recently-seen trace first (an open in-flight turn
// counts as "seen now", so a live request is never the victim). The /debug/vars read path
// additionally prunes to the live session set (retain), so under normal operation the map
// tracks at most the live sessions; the cap is the write-path backstop when reads are rare.
const sessionActivityCap = 256

// sessionActivity is the bounded per-trace activity registry behind the agents pane's
// live-status cell. Every method is safe on a nil receiver (a bare Server that never went
// through New has none) and guarded by one mutex — each record is tiny and touched on the
// proxy hot path, so a single lock is cheaper than sharded bookkeeping.
type sessionActivity struct {
	mu  sync.Mutex
	rec map[string]*traceActivity
}

// newSessionActivity builds an empty registry. New wires one onto every Server it returns.
func newSessionActivity() *sessionActivity {
	return &sessionActivity{rec: map[string]*traceActivity{}}
}

// getOrMakeLocked returns the record for trace, creating it (and evicting the coldest
// record first when at cap) if absent. The caller holds a.mu; trace is assumed non-empty.
func (a *sessionActivity) getOrMakeLocked(trace string) *traceActivity {
	if r := a.rec[trace]; r != nil {
		return r
	}
	if len(a.rec) >= sessionActivityCap {
		a.evictColdestLocked()
	}
	r := &traceActivity{}
	a.rec[trace] = r
	return r
}

// evictColdestLocked drops the record least-recently seen, where "seen" is the later of
// lastActivityUnix and inflightSinceUnix (so an open turn is treated as fresh and survives
// eviction). Ties break on trace id for determinism. The caller holds a.mu and the map is
// known non-empty.
func (a *sessionActivity) evictColdestLocked() {
	var victim string
	var best int64
	for id, r := range a.rec {
		seen := r.lastActivityUnix
		if r.inflightSinceUnix > seen {
			seen = r.inflightSinceUnix
		}
		if victim == "" || seen < best || (seen == best && id < victim) {
			victim, best = id, seen
		}
	}
	delete(a.rec, victim)
}

// stampProposed records an admitted tool call for trace: it sets last_tool, stamps
// last_activity, and increments spawn_count when the tool is a subagent-spawn shape. This
// is the per-adjudication activity signal the pane reads; now is injected so the record is
// deterministic under test.
func (a *sessionActivity) stampProposed(trace, tool string, now time.Time) {
	if a == nil || trace == "" || tool == "" {
		return
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	r := a.getOrMakeLocked(trace)
	r.lastTool = tool
	r.lastActivityUnix = now.Unix()
	if subagentSpawnTool(tool) {
		r.spawnCount++
	}
}

// beginTurn marks trace as holding an open served model request (the in-flight window),
// stamping the moment it opened. Paired with endTurn around completeServed so the pane's
// in-flight age reflects the request the trace holds RIGHT NOW.
func (a *sessionActivity) beginTurn(trace string, now time.Time) {
	if a == nil || trace == "" {
		return
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	a.getOrMakeLocked(trace).inflightSinceUnix = now.Unix()
}

// endTurn clears trace's in-flight marker when its served request returns. It deliberately
// does NOT stamp last_activity — that belongs to adjudication (stampProposed), which runs
// after the turn returns — so idle age measures time since the last ADJUDICATED call, not
// since the request merely closed.
func (a *sessionActivity) endTurn(trace string) {
	if a == nil || trace == "" {
		return
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if r := a.rec[trace]; r != nil {
		r.inflightSinceUnix = 0
	}
}

// snapshot returns trace's activity projected for the pane, relative to now, and whether a
// record exists. InflightSeconds is set only while a turn is open; otherwise IdleSeconds
// carries the time since the last adjudication — the two are mutually exclusive, so the
// row shows exactly one age. Ages are clamped at 0 so a clock skew never renders negative.
func (a *sessionActivity) snapshot(trace string, now time.Time) (traceActivityView, bool) {
	if a == nil || trace == "" {
		return traceActivityView{}, false
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	r := a.rec[trace]
	if r == nil {
		return traceActivityView{}, false
	}
	v := traceActivityView{LastTool: r.lastTool, SpawnCount: r.spawnCount}
	nowU := now.Unix()
	switch {
	case r.inflightSinceUnix > 0:
		if d := nowU - r.inflightSinceUnix; d > 0 {
			v.InflightSeconds = d
		}
	case r.lastActivityUnix > 0:
		if d := nowU - r.lastActivityUnix; d > 0 {
			v.IdleSeconds = d
		}
	}
	return v, true
}

// retain drops every record whose trace is not in live (the current non-stopped session
// set), folding stopped and vanished traces so the registry tracks at most the live
// sessions. Called on the /debug/vars read path with the session list already in hand.
func (a *sessionActivity) retain(live map[string]struct{}) {
	if a == nil {
		return
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	for id := range a.rec {
		if _, ok := live[id]; !ok {
			delete(a.rec, id)
		}
	}
}
