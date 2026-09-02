package sessionctl

// broadcast.go — the fleet / lane-scoped BROADCAST of one lifecycle control op
// (#2764, epic #2753): the one verb that steers a whole wave. `fak session` ops are
// per-trace; a running wave is otherwise steered only by the loopmgr governor
// throttle, a per-worker signal, or a janitor kill. This file adds the missing
// middle: resolve the set of live sessions matching a SELECTOR (lane / wave /
// label), apply ONE lifecycle op to each through the existing per-session
// drive-state plumbing, and report per-session applied/refused.
//
// Boundary honesty (the epic's #2754 grammar). Broadcast never lands an op
// mid-turn: it fans the SAME Table.Transition write the single-session verbs ride,
// so each session takes the op at its own boundary — pause/throttle/cancel at the
// next turn boundary, terminate at the next safe point — exactly as the vocabulary
// spine declares per op. The broadcast is the DISTRIBUTION of an op, not a new op:
// it accepts only the lifecycle subset of the closed vocabulary (vocab.go) and
// adds no row of its own, so the per-op boundary/witness/refusal contract is
// unchanged whether an op arrives alone or fanned.
//
// Selector metadata. The issue assumed the session table already exposes
// lane/wave/label metadata; at HEAD it does NOT (session.State carries no such
// fields) — that assumption is the invalidating one this artifact names. So, like
// redirect.go's objective store, this package owns the per-session metadata home:
// a process-global tag registry keyed by trace (TagSession / SessionTag /
// ClearSessionTag). The PRODUCER is the serve process's session-admission boundary
// — cmd/fak's tagServedSessionAdmit, on the decideSession hook every served turn
// crosses — which tags each trace from the routing identity the process was
// launched under and drops the tag when the session stops (#5640). Any in-process
// spawn site that knows a session's lane more precisely may TagSession it first;
// admission fills gaps and never overwrites. FAIL-CLOSED both ways: an UNTAGGED
// session never matches any selector (a broadcast cannot silently mutate work
// nobody scoped into it), and an EMPTY selector is refused malformed rather than
// matching everything (the "quiesce the fleet by accident" over-match the issue
// names as a confusion risk).
//
// Scope fence (per the issue). Per-gateway only — the fan-out is over ONE
// process's session.Table; cross-host fleet broadcast is out of scope. Lifecycle
// ops only — steer/redirect (payload-carrying semantic ops) and budget/priority
// (value-carrying re-allotments) are refused malformed; semantic broadcast is the
// epic's follow-on.

import (
	"container/list"
	"fmt"
	"strings"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/session"
)

// BroadcastMeta is one session's routing metadata — the coordinates a broadcast
// selector resolves against. It is deliberately NOT on session.State (the drive
// table stays selector-blind); this package owns it, exactly as it owns the
// redirect objective store.
type BroadcastMeta struct {
	// Lane is the dos.toml work lane the session is working (e.g. "cmd").
	Lane string `json:"lane,omitempty"`
	// Wave is the dispatch wave / cohort id the session was launched under.
	Wave string `json:"wave,omitempty"`
	// Labels are free-form operator tags (e.g. "dogfood", "nightrun").
	Labels []string `json:"labels,omitempty"`
}

// IsZero reports whether the metadata carries no coordinate at all. A zero tag is
// never stored — it would be unmatchable dead weight.
func (m BroadcastMeta) IsZero() bool {
	return m.Lane == "" && m.Wave == "" && len(m.Labels) == 0
}

// MaxSessionTags bounds the tag registry. A tag is dropped by ClearSessionTag at
// session teardown, but teardown is not guaranteed to run for every trace a live
// process ever admits: session.Table is itself a bounded LRU that EVICTS a cold
// record silently (no terminal transition, so no teardown edge fires), and a
// gateway minting a per-request trace churns ids faster than any of them stop. An
// unbounded map behind those two facts is a leak in a long-lived serve, which is
// the failure mode "tag without untag" trades a correctness bug for. So the
// registry carries its own ceiling and evicts the LEAST-RECENTLY-TAGGED trace past
// it — the same bounded-LRU discipline, and the same reason, as the drive table it
// shadows.
const MaxSessionTags = 8192

// broadcast tag state is per-session, keyed by the run's trace id — the same
// keying and lifecycle as redirect.go's objective store. Process-global: the serve
// process that owns the session.Table owns the tags. broadcastOrder/broadcastIndex
// are the recency ring enforcing MaxSessionTags; all three move together under
// broadcastMu.
var (
	broadcastMu    sync.Mutex
	broadcastTags  = map[string]BroadcastMeta{}
	broadcastOrder = list.New()
	broadcastIndex = map[string]*list.Element{}
)

// TagSession records trace's routing metadata for broadcast selection. The spawn
// site (the serve admission boundary — see cmd/fak's tagServedSessionAdmit) calls
// it once per session it starts. An empty trace or zero meta is ignored — an
// unmatchable tag is never stored. Overwrites any prior tag (a re-homed session
// re-tags) and refreshes its recency, evicting the coldest tag past MaxSessionTags.
func TagSession(trace string, meta BroadcastMeta) {
	trace = strings.TrimSpace(trace)
	if trace == "" || meta.IsZero() {
		return
	}
	broadcastMu.Lock()
	defer broadcastMu.Unlock()
	broadcastTags[trace] = meta
	broadcastTouchLocked(trace)
	broadcastTrimLocked()
}

// broadcastTouchLocked moves trace to the front of the recency ring, inserting it
// when unseen. Caller holds broadcastMu.
func broadcastTouchLocked(trace string) {
	if el := broadcastIndex[trace]; el != nil {
		broadcastOrder.MoveToFront(el)
		return
	}
	broadcastIndex[trace] = broadcastOrder.PushFront(trace)
}

// broadcastTrimLocked evicts from the cold end until the registry is within
// MaxSessionTags. Eviction is FAIL-CLOSED in the same direction the whole package
// is: an evicted trace becomes untagged, so it matches no selector — a broadcast
// can never reach a session on the strength of a stale tag. Caller holds
// broadcastMu.
func broadcastTrimLocked() {
	for len(broadcastTags) > MaxSessionTags {
		el := broadcastOrder.Back()
		if el == nil {
			return
		}
		trace := el.Value.(string)
		broadcastOrder.Remove(el)
		delete(broadcastIndex, trace)
		delete(broadcastTags, trace)
	}
}

// SessionTag returns trace's routing metadata and whether one is declared.
func SessionTag(trace string) (BroadcastMeta, bool) {
	trace = strings.TrimSpace(trace)
	if trace == "" {
		return BroadcastMeta{}, false
	}
	broadcastMu.Lock()
	defer broadcastMu.Unlock()
	m, ok := broadcastTags[trace]
	return m, ok
}

// ClearSessionTag drops trace's routing metadata — the session teardown hook, so a
// finished run leaks no per-trace state. Idempotent.
func ClearSessionTag(trace string) {
	trace = strings.TrimSpace(trace)
	if trace == "" {
		return
	}
	broadcastMu.Lock()
	defer broadcastMu.Unlock()
	delete(broadcastTags, trace)
	if el := broadcastIndex[trace]; el != nil {
		broadcastOrder.Remove(el)
		delete(broadcastIndex, trace)
	}
}

// BroadcastSelector names the set of sessions a broadcast steers: every STATED
// axis must match (AND), an unstated axis matches anything. The zero selector is
// MALFORMED, not match-all — refusing it at the edge is the fence against the
// silent whole-fleet mutation the issue names as a confusion risk.
type BroadcastSelector struct {
	// Lane selects sessions tagged with this work lane.
	Lane string `json:"lane,omitempty"`
	// Wave selects sessions tagged with this dispatch wave / cohort id.
	Wave string `json:"wave,omitempty"`
	// Label selects sessions carrying this label among their tags.
	Label string `json:"label,omitempty"`
}

// IsZero reports whether no axis is stated — the malformed match-nothing form.
func (s BroadcastSelector) IsZero() bool {
	return s.Lane == "" && s.Wave == "" && s.Label == ""
}

// Matches reports whether a session tagged meta is in the selector's set: every
// stated axis must agree. Exact, case-sensitive token equality — a selector is an
// operator-issued closed token, never a pattern.
func (s BroadcastSelector) Matches(meta BroadcastMeta) bool {
	if s.IsZero() {
		return false
	}
	if s.Lane != "" && meta.Lane != s.Lane {
		return false
	}
	if s.Wave != "" && meta.Wave != s.Wave {
		return false
	}
	if s.Label != "" {
		found := false
		for _, l := range meta.Labels {
			if l == s.Label {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}
	return true
}

// String renders the stated axes for reports and logs (e.g. "lane=cmd wave=w7").
func (s BroadcastSelector) String() string {
	parts := make([]string, 0, 3)
	if s.Lane != "" {
		parts = append(parts, "lane="+s.Lane)
	}
	if s.Wave != "" {
		parts = append(parts, "wave="+s.Wave)
	}
	if s.Label != "" {
		parts = append(parts, "label="+s.Label)
	}
	return strings.Join(parts, " ")
}

// BroadcastRefuseReason is the closed refusal vocabulary for the broadcast EDGE —
// the same closed-reason discipline as redirect.go's RedirectRefuseReason. One
// token: a broadcast whose SHAPE cannot be applied (empty selector, unknown op,
// non-lifecycle op) is refused synchronously before any session is touched.
// Per-SESSION refusals inside an accepted broadcast reuse the drive table's own
// closed tokens (session.ControlRefusalTokens) via ControlRefusalFor — a fanned
// op refuses exactly as it would alone.
type BroadcastRefuseReason string

const (
	// BroadcastMalformed refuses a broadcast whose shape cannot be applied: an
	// empty (match-all) selector, a nil table, an op outside the closed control
	// vocabulary, or a non-lifecycle op (steer/redirect/budget/priority) that
	// cannot be fanned without a payload.
	BroadcastMalformed BroadcastRefuseReason = "BROADCAST_MALFORMED"
)

// BroadcastRefusal is the structured, closed-reason refusal of one broadcast op.
// It implements error so plumbing can thread it, but callers should switch on
// Reason, never parse Detail.
type BroadcastRefusal struct {
	Reason BroadcastRefuseReason `json:"reason"`
	Detail string                `json:"detail,omitempty"`
}

func (r *BroadcastRefusal) Error() string { return refusalString(string(r.Reason), r.Detail) }

// broadcastRefuse builds a BroadcastRefusal in one line.
func broadcastRefuse(reason BroadcastRefuseReason, format string, args ...any) *BroadcastRefusal {
	return &BroadcastRefusal{Reason: reason, Detail: fmt.Sprintf(format, args...)}
}

// broadcastRunStates maps each BROADCASTABLE op — the lifecycle subset of the
// closed vocabulary (vocab.go) — onto the drive-state it enqueues. This is the
// same op→run-state mapping the single-session CLI verbs apply (`fak session
// pause` ⇒ Paused, `stop` ⇒ Draining, ...), spelled once. An op absent here is
// not broadcastable: steer/redirect carry payloads (semantic broadcast is out of
// scope, per the issue), budget/priority carry values a bare fan-out cannot name.
var broadcastRunStates = map[ControlOp]session.RunState{
	OpPause:     session.Paused,
	OpResume:    session.Running,
	OpCancel:    session.Draining,
	OpTerminate: session.Terminating,
	OpThrottle:  session.Throttled,
}

// BroadcastRunState returns the drive-state a broadcastable lifecycle op enqueues,
// and whether op is broadcastable at all. It exists so a fan-out that resolves its
// OWN session set — the cross-process fleet bus, whose instance-level selector is
// already the deliberate-widening gate this package's session selector provides —
// can ride the same op→run-state table instead of forking a second copy of it. A
// forked copy is how `resume` and `running` drift apart.
func BroadcastRunState(op ControlOp) (session.RunState, bool) {
	run, ok := broadcastRunStates[op]
	return run, ok
}

// BroadcastableOps returns the closed set of ops a broadcast may fan, in the
// vocabulary's stable op order — the CLI help / validation surface.
func BroadcastableOps() []ControlOp {
	out := make([]ControlOp, 0, len(broadcastRunStates))
	for _, op := range Ops() {
		if _, ok := broadcastRunStates[op]; ok {
			out = append(out, op)
		}
	}
	return out
}

// BroadcastResult is one session's row of the per-session applied/refused report
// — the issue's required witness shape. Applied=false always carries the drive
// table's own closed refusal (ControlRefusalFor), never a bare boolean.
type BroadcastResult struct {
	// Trace is the session the op was fanned to.
	Trace string `json:"trace"`
	// Applied reports whether the drive-state write was taken.
	Applied bool `json:"applied"`
	// Run is the session's run-state token after the attempt (the enqueued state
	// on applied; the unchanged live state on refused).
	Run string `json:"run"`
	// Refusal is the closed drive-table refusal when Applied is false (a terminal
	// session refuses CONTROL_SESSION_TERMINAL, exactly as it would a single op).
	Refusal *session.ControlRefusal `json:"refusal,omitempty"`
}

// BroadcastReport is the whole broadcast's outcome: the resolved set and one row
// per matched session, in the table's snapshot order (priority rank, stable).
type BroadcastReport struct {
	// Op is the lifecycle op that was fanned.
	Op ControlOp `json:"op"`
	// Selector is the set definition the broadcast resolved.
	Selector BroadcastSelector `json:"selector"`
	// Matched is how many live sessions the selector resolved.
	Matched int `json:"matched"`
	// Applied / Refused partition Matched by per-session outcome.
	Applied int `json:"applied"`
	Refused int `json:"refused"`
	// Results is the per-session applied/refused report, one row per match.
	Results []BroadcastResult `json:"results"`
}

// Broadcast resolves the live sessions in tbl matching sel and applies op to each
// through the existing per-session drive-state plumbing (Table.Transition — the
// write the single-session verbs ride), reporting per-session applied/refused.
// Each session takes the op at ITS boundary, exactly as it would a single op;
// nothing lands mid-turn. A session the selector does not match — including every
// untagged session — is never touched. A malformed broadcast (empty selector,
// nil table, unknown or non-lifecycle op) is refused whole at the edge, before
// any session is touched.
func Broadcast(tbl *session.Table, sel BroadcastSelector, op ControlOp, reason string) (BroadcastReport, *BroadcastRefusal) {
	if tbl == nil {
		return BroadcastReport{}, broadcastRefuse(BroadcastMalformed, "broadcast needs a session table")
	}
	if sel.IsZero() {
		return BroadcastReport{}, broadcastRefuse(BroadcastMalformed,
			"broadcast needs a selector (lane / wave / label); an empty selector never means \"every session\"")
	}
	if _, known := Spec(op); !known {
		return BroadcastReport{}, broadcastRefuse(BroadcastMalformed,
			"unknown control op %q (closed vocabulary: %v)", op, Ops())
	}
	run, lifecycle := broadcastRunStates[op]
	if !lifecycle {
		return BroadcastReport{}, broadcastRefuse(BroadcastMalformed,
			"op %q is not broadcastable — only the lifecycle ops %v fan without a payload", op, BroadcastableOps())
	}

	report := BroadcastReport{Op: op, Selector: sel}
	for _, st := range tbl.Snapshot() {
		meta, tagged := SessionTag(st.TraceID)
		if !tagged || !sel.Matches(meta) {
			continue
		}
		report.Matched++
		next, ok := tbl.Transition(st.TraceID, run, reason)
		row := BroadcastResult{
			Trace:   st.TraceID,
			Applied: ok,
			Run:     next.Run.String(),
			Refusal: session.ControlRefusalFor(string(op), next, ok),
		}
		if ok {
			report.Applied++
		} else {
			report.Refused++
		}
		report.Results = append(report.Results, row)
	}
	return report, nil
}
