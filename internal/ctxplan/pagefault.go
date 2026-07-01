package ctxplan

import (
	"context"
	"strconv"
	"strings"
)

// pagefault.go — issue #1587: the context PAGE-FAULT PROTOCOL, the runtime continuity
// contract managed context needs across a HIDDEN RESTART (a session reset, a fresh
// process, a re-armed context window the user never asked for). Session budgets
// (sessionreset), reset carryover (sessionreset.BuildSeed), and the planned view
// (ctxplan.Plan/Forecast) already exist; what was missing is the DECISION a caller
// makes the instant a forecast determines a needed span is missing from the resident
// view. Silent omission is the failure this closes: today a miss can either demand-page
// (fault.go's DemandPage) or say nothing. This type makes "say nothing" impossible by
// construction — every miss produces exactly one closed-vocabulary decision.
//
// RELATIONSHIP TO fault.go. fault.go's Fault/DemandPage is the MECHANISM: given a span
// id, it pages the span in through the trust gate and reports what happened (served,
// already-resident, refused, absent). This file is the POLICY layer one level up: given
// a PageFaultEvent (a forecast MISS — the planner determined a span the turn needs is
// not resident) it decides WHAT TO DO before any page-in is attempted — page it in
// silently, ask the user, refuse outright, or proceed with a pointer instead of content.
// DecidePageFault never touches a Store; MaterializeFaultDecision below is the thin
// bridge that executes a PageFaultPageIn decision through DemandPage.
//
// DETERMINISM. DecidePageFault is a pure function of (PageFaultEvent, PageFaultPolicy):
// no clock, no randomness, no hidden state. The same inputs always produce the same
// PageFaultDecision, so a decision is REPLAYABLE — a caller (or a test, or an audit) can
// re-run it from a persisted PageFaultEvent and get back the identical decision, which is
// exactly the witness TestPageFaultDecisionIsReplayable checks.
//
// PERSISTED STATE. PageFaultLog is the append-only, replayable ledger of decisions: each
// entry is a (PageFaultEvent, PageFaultDecision) pair in occurrence order. It is the
// "deterministic runtime object with persisted state, typed transitions, and replayable
// witnesses" the issue asks for — Append records a typed transition, Replay recomputes
// every logged decision from its event and reports any decision that no longer matches
// (DIVERGED), which would mean the policy changed underneath a stale log.

// PageFaultOutcome is the CLOSED vocabulary of dispositions for a context page fault —
// the decision a caller makes the instant a forecast determines a needed span is
// missing. Exactly one outcome is produced per event; there is no "no decision" state
// once DecidePageFault runs, which is what keeps a miss from silently becoming an
// omission.
type PageFaultOutcome string

const (
	// PageFaultPageIn: fetch/reload the missing span and splice it back into the
	// resident view (the fault.go DemandPage mechanism this decision authorizes). Chosen
	// when the span is live (not sealed/tombstoned/absent) and can be silently
	// reconstructed without asking the user — the common case for an ordinary forecast
	// miss during a hidden restart.
	PageFaultPageIn PageFaultOutcome = "page_in"
	// PageFaultQueryUser: ask the user. Chosen when the span cannot be silently
	// reconstructed — it never entered the durable/lossless store (a hidden-restart
	// boundary dropped it, e.g. a turn-scoped ephemeral that a prior process held only
	// in memory) — so continuing without asking would be a guess dressed as a fact.
	PageFaultQueryUser PageFaultOutcome = "query_user"
	// PageFaultDeny: refuse to proceed without the span. Chosen when the span is
	// SEALED or TOMBSTONED (the trust gate holds — the fault.go FaultRefused mirror)
	// and the caller marked the reference as required: paging it in would launder
	// poison/suppression into context, and continuing anyway would silently drop a
	// requirement, so the safe closed-vocabulary move is an explicit refusal.
	PageFaultDeny PageFaultOutcome = "deny"
	// PageFaultContinuePointer: proceed but carry a pointer/reference to the missing
	// span instead of its content. Chosen when the span is reconstructable but NOT
	// required this turn (a durable/bounded fact the turn only tangentially touches),
	// or when the caller has opted into pointer-only continuity — the turn is not
	// blocked, but the fact is not silently dropped either: its handle rides along for
	// a later demand-page.
	PageFaultContinuePointer PageFaultOutcome = "continue_with_pointer"
)

// validPageFaultOutcomes is the membership set DecidePageFault always produces one
// member of — used by tests and by any caller (de)serializing a persisted decision to
// fail closed on a corrupt/foreign value rather than silently accept it.
var validPageFaultOutcomes = map[PageFaultOutcome]bool{
	PageFaultPageIn:          true,
	PageFaultQueryUser:       true,
	PageFaultDeny:            true,
	PageFaultContinuePointer: true,
}

// ValidPageFaultOutcome reports whether o is a member of the closed vocabulary.
func ValidPageFaultOutcome(o PageFaultOutcome) bool { return validPageFaultOutcomes[o] }

func (o PageFaultOutcome) String() string {
	if ValidPageFaultOutcome(o) {
		return string(o)
	}
	if o == "" {
		return "(unset)"
	}
	return "unknown(" + string(o) + ")"
}

// PageFaultSpanState is the closed-vocabulary state of the missing span in the backing
// store at decision time — the fact DecidePageFault branches its outcome on first. It
// deliberately mirrors the fault.go Fault* status names (served/resident/refused/absent)
// but at the PRE-materialize decision point: this is what the forecast-miss handler
// knows BEFORE it attempts a page-in, not what happened after.
type PageFaultSpanState string

const (
	// PageFaultSpanLive: the span is present in the lossless store and not
	// sealed/tombstoned — silently reconstructable by a page-in.
	PageFaultSpanLive PageFaultSpanState = "live"
	// PageFaultSpanSealed: the span is quarantined by the trust gate. Never silently
	// reconstructed (mirrors fault.go's FaultRefused / ErrSealed).
	PageFaultSpanSealed PageFaultSpanState = "sealed"
	// PageFaultSpanTombstoned: the span was suppressed by context control. Never
	// silently reconstructed (mirrors fault.go's ErrTombstoned).
	PageFaultSpanTombstoned PageFaultSpanState = "tombstoned"
	// PageFaultSpanGone: the span never entered the durable store at all — the
	// hidden-restart case that MemGPT calls a working-context eviction with no
	// swap-in path: a prior process held it only in volatile/turn-scoped memory and a
	// hidden restart dropped it before it was ever durably captured. This is NOT the
	// same as fault.go's FaultAbsent (an id naming no span, usually a caller typo);
	// PageFaultSpanGone specifically means the reference is believed real but
	// unrecoverable from any store.
	PageFaultSpanGone PageFaultSpanState = "gone"
)

func normalizePageFaultSpanState(s PageFaultSpanState) PageFaultSpanState {
	switch s {
	case PageFaultSpanLive, PageFaultSpanSealed, PageFaultSpanTombstoned, PageFaultSpanGone:
		return s
	default:
		// Fail closed to the most conservative state: an unrecognized/garbage state is
		// treated as unrecoverable, never as silently live.
		return PageFaultSpanGone
	}
}

// PageFaultEvent is the forecast-MISS input: the planner (or a runtime re-arm across a
// hidden restart) determined the turn references SpanID and it is not in the resident
// view. It carries exactly the facts DecidePageFault needs and nothing that could
// introduce nondeterminism (no timestamps, no random ids) — the same event always
// yields the same decision.
type PageFaultEvent struct {
	SpanID string `json:"span_id"`
	Step   int    `json:"step,omitempty"`
	Role   string `json:"role,omitempty"`
	// State is the span's status in the backing store at decision time (see
	// PageFaultSpanState). An empty/unrecognized value normalizes to
	// PageFaultSpanGone (fail closed).
	State PageFaultSpanState `json:"state"`
	// Durability classifies the missing span (see the ctxplan durability constants).
	// A durable/bounded fact crossing a hidden restart is a stronger candidate for
	// silent page-in than a turn-scoped ephemeral, which a fresh process may never
	// have durably captured in the first place.
	Durability string `json:"durability,omitempty"`
	// Required marks whether the current turn cannot correctly proceed without this
	// span's CONTENT (not just its existence) — e.g. the active refund amount, not a
	// backgrounded aside. Required + unrecoverable (sealed/tombstoned/gone) forces
	// PageFaultDeny rather than silently continuing.
	Required bool `json:"required,omitempty"`
	// SilentlyReconstructable marks that even though the span is PageFaultSpanLive,
	// the caller has determined it can be reconstructed WITHOUT asking the user (its
	// content is derivable from durable facts, a tool re-query, or a cache — not a
	// one-time human input that only the user can supply again). When false on a live
	// span, the protocol still queries the user rather than assume a silent page-in is
	// safe (e.g. a live span that only paraphrases something the user said once and a
	// verbatim re-ask is safer than trusting the paraphrase).
	SilentlyReconstructable bool `json:"silently_reconstructable,omitempty"`
}

// PageFaultPolicy holds the knobs that turn a PageFaultEvent into a PageFaultOutcome.
// It follows the same normalize-with-conservative-defaults shape as AssumptionPolicy so
// a caller does not have to hand-fill every field to get sane behavior.
type PageFaultPolicy struct {
	// PointerDurabilityFloor is the minimum durability rank (see durabilityRank) a
	// non-required LIVE span must meet to qualify for PageFaultContinuePointer instead
	// of an outright PageFaultPageIn. Below the floor, a non-required live span still
	// pages in (it is cheap and the safest default); at or above it, a non-required
	// durable/bounded fact is treated as safe to defer via pointer, saving the fault
	// cost for spans the turn is not actually blocked on. Default: DurabilityBounded.
	PointerDurabilityFloor string `json:"pointer_durability_floor,omitempty"`
}

// DefaultPageFaultPolicy is the conservative default: only a non-required LIVE span at
// bounded durability or higher defers to a pointer; everything else that is silently
// reconstructable pages in immediately.
func DefaultPageFaultPolicy() PageFaultPolicy {
	return PageFaultPolicy{PointerDurabilityFloor: DurabilityBounded}
}

func normalizePageFaultPolicy(p PageFaultPolicy) PageFaultPolicy {
	if _, ok := durabilityRank[p.PointerDurabilityFloor]; !ok {
		p.PointerDurabilityFloor = DefaultPageFaultPolicy().PointerDurabilityFloor
	}
	return p
}

// PageFaultDecision is the typed transition DecidePageFault produces: the outcome, the
// reason it was chosen (an operator-readable EXPLAIN, mirroring Plan.Explain's style),
// and the event it was computed from (echoed back so a decision is self-describing and
// a persisted log entry does not need a side join to know what it decided about).
type PageFaultDecision struct {
	SpanID  string             `json:"span_id"`
	Step    int                `json:"step,omitempty"`
	Outcome PageFaultOutcome   `json:"outcome"`
	Reason  string             `json:"reason"`
	State   PageFaultSpanState `json:"state"`
}

// DecidePageFault is the deterministic decision function: given a forecast-MISS event
// and a policy, it returns EXACTLY ONE closed-vocabulary outcome — never a silent
// no-decision. It is pure (no clock, no randomness, no I/O) so the same (event, policy)
// always reproduces the same decision, which is what makes a persisted PageFaultLog
// replayable.
//
// Decision order (first match wins, most safety-critical first):
//
//  1. Unrecoverable (sealed/tombstoned/gone) AND Required -> PageFaultDeny. The turn
//     cannot proceed correctly without the content, and the content cannot be produced
//     (the gate holds, or it was never durably captured) — continuing would silently
//     drop a requirement, so the closed-vocabulary move is a refusal, not a guess.
//  2. Unrecoverable AND NOT Required -> PageFaultContinuePointer. The turn is not
//     blocked on it, so proceed — but the missing reference is NOT silently dropped: a
//     pointer/handle (or, for a sealed/tombstoned span, the audit reason) rides along
//     so a later demand-page or a human review can still resolve it.
//  3. Live AND NOT SilentlyReconstructable -> PageFaultQueryUser. The span exists in
//     the store, but the caller has determined its content cannot be safely
//     reconstructed without the user (e.g. a one-time human input) — ask rather than
//     assume the reload is faithful.
//  4. Live AND SilentlyReconstructable AND Required -> PageFaultPageIn. The turn needs
//     the content and it can be safely reloaded — the cheap, silent recovery path
//     (authorizes fault.go's DemandPage).
//  5. Live AND SilentlyReconstructable AND NOT Required:
//     - durability at or above policy.PointerDurabilityFloor -> PageFaultContinuePointer
//     (defer the reload; the turn is not blocked and the fact is durable enough to
//     recover later without loss).
//     - below the floor (a turn/session-scoped span) -> PageFaultPageIn (cheap, and a
//     short-lived fact deferred now may not be recoverable as cleanly later).
func DecidePageFault(ev PageFaultEvent, policy PageFaultPolicy) PageFaultDecision {
	policy = normalizePageFaultPolicy(policy)
	state := normalizePageFaultSpanState(ev.State)
	durability := NormDurability(ev.Durability)
	out := PageFaultDecision{SpanID: ev.SpanID, Step: ev.Step, State: state}

	unrecoverable := state == PageFaultSpanSealed || state == PageFaultSpanTombstoned || state == PageFaultSpanGone
	switch {
	case unrecoverable && ev.Required:
		out.Outcome = PageFaultDeny
		out.Reason = "required span is " + string(state) + ": cannot silently reconstruct, refusing rather than guessing"
	case unrecoverable && !ev.Required:
		out.Outcome = PageFaultContinuePointer
		out.Reason = "span is " + string(state) + " but not required this turn: continuing with a pointer, not dropping the reference"
	case !ev.SilentlyReconstructable:
		out.Outcome = PageFaultQueryUser
		out.Reason = "span is live but not safely reconstructable without the user: asking rather than assuming"
	case ev.Required:
		out.Outcome = PageFaultPageIn
		out.Reason = "span is live, reconstructable, and required this turn: paging it back in"
	case durabilityRank[durability] >= durabilityRank[policy.PointerDurabilityFloor]:
		out.Outcome = PageFaultContinuePointer
		out.Reason = "span is live and reconstructable but not required, and durable enough to defer: continuing with a pointer"
	default:
		out.Outcome = PageFaultPageIn
		out.Reason = "span is live, reconstructable, and short-lived: paging it back in now rather than risking loss"
	}
	return out
}

// PageFaultLog is the append-only, replayable ledger of page-fault decisions — the
// PERSISTED-STATE half of the runtime continuity contract. Entries are kept in
// occurrence order; Replay recomputes every entry's decision from its stored event and
// reports any DIVERGED entry (the policy changed since the entry was logged), so a
// caller can tell "was this decision reproduced" from real evidence instead of trusting
// the stored Decision field blindly. The zero value is a usable empty log.
type PageFaultLog struct {
	entries []PageFaultLogEntry
}

// PageFaultLogEntry pairs one logged event with the decision computed for it, so the
// log is self-contained: replaying it needs no external state beyond the policy.
type PageFaultLogEntry struct {
	Event    PageFaultEvent    `json:"event"`
	Decision PageFaultDecision `json:"decision"`
	Policy   PageFaultPolicy   `json:"policy"`
}

// Append decides ev under policy, records the (event, decision, policy) as the next
// entry, and returns the decision — the one call sites need to both persist the
// transition and act on it. It never mutates a prior entry: PageFaultLog is
// append-only, matching the audit posture the rest of ctxplan uses (Elision/Selection
// records are never edited in place, only added to).
func (l *PageFaultLog) Append(ev PageFaultEvent, policy PageFaultPolicy) PageFaultDecision {
	d := DecidePageFault(ev, policy)
	l.entries = append(l.entries, PageFaultLogEntry{Event: ev, Decision: d, Policy: policy})
	return d
}

// Entries returns a defensive copy of the logged entries in occurrence order.
func (l *PageFaultLog) Entries() []PageFaultLogEntry {
	return append([]PageFaultLogEntry(nil), l.entries...)
}

// PageFaultReplayVerdict is the outcome of replaying one logged entry: whether
// recomputing DecidePageFault from the stored event+policy reproduces the stored
// decision.
type PageFaultReplayVerdict struct {
	SpanID     string           `json:"span_id"`
	Step       int              `json:"step,omitempty"`
	Stored     PageFaultOutcome `json:"stored"`
	Recomputed PageFaultOutcome `json:"recomputed"`
	Diverged   bool             `json:"diverged"`
}

// Replay recomputes DecidePageFault for every logged entry from its own stored
// (Event, Policy) and compares the result against the stored Decision. A log built
// from pure inputs (no clock/randomness snuck into a PageFaultEvent) always replays
// with zero diverged entries — that is the replayability witness this type exists to
// make checkable. A non-empty diverged slice means either the log was tampered with or
// DecidePageFault's logic changed incompatibly since the entry was recorded.
func (l *PageFaultLog) Replay() (verdicts []PageFaultReplayVerdict, allMatch bool) {
	allMatch = true
	for _, e := range l.entries {
		got := DecidePageFault(e.Event, e.Policy)
		v := PageFaultReplayVerdict{
			SpanID:     e.Event.SpanID,
			Step:       e.Event.Step,
			Stored:     e.Decision.Outcome,
			Recomputed: got.Outcome,
			Diverged:   got.Outcome != e.Decision.Outcome,
		}
		if v.Diverged {
			allMatch = false
		}
		verdicts = append(verdicts, v)
	}
	return verdicts, allMatch
}

// Summary folds the log into outcome counts — the O(1) health signal a debug surface
// prints instead of walking every entry (mirrors AssumptionSummary's role for
// assumptions).
type PageFaultSummary struct {
	PageIn          int `json:"page_in"`
	QueryUser       int `json:"query_user"`
	Deny            int `json:"deny"`
	ContinuePointer int `json:"continue_with_pointer"`
}

// Summary computes the outcome-class counts over every logged entry.
func (l *PageFaultLog) Summary() PageFaultSummary {
	var s PageFaultSummary
	for _, e := range l.entries {
		switch e.Decision.Outcome {
		case PageFaultPageIn:
			s.PageIn++
		case PageFaultQueryUser:
			s.QueryUser++
		case PageFaultDeny:
			s.Deny++
		case PageFaultContinuePointer:
			s.ContinuePointer++
		}
	}
	return s
}

// Explain renders the log as an operator-readable report, in the Plan.Explain /
// AssumptionReport style: one line per decision plus the outcome-count footer.
func (l *PageFaultLog) Explain() string {
	var b strings.Builder
	s := l.Summary()
	b.WriteString("ctxplan page-fault log: " + strconv.Itoa(len(l.entries)) + " decision(s)\n")
	for _, e := range l.entries {
		b.WriteString("  [step ")
		b.WriteString(strconv.Itoa(e.Event.Step))
		b.WriteString("] ")
		b.WriteString(e.Event.SpanID)
		b.WriteString(" state=")
		b.WriteString(string(e.Decision.State))
		b.WriteString(" -> ")
		b.WriteString(string(e.Decision.Outcome))
		b.WriteString(": ")
		b.WriteString(e.Decision.Reason)
		b.WriteString("\n")
	}
	b.WriteString("  totals: page_in=" + strconv.Itoa(s.PageIn) +
		" query_user=" + strconv.Itoa(s.QueryUser) +
		" deny=" + strconv.Itoa(s.Deny) +
		" continue_with_pointer=" + strconv.Itoa(s.ContinuePointer) + "\n")
	return b.String()
}

// EventFromMiss builds a PageFaultEvent from a Span the planner already knows is
// missing (its Elision record) plus the two policy facts only the CALLER can supply —
// whether the turn requires the content and whether it is safely reconstructable
// without the user. This is the thin adapter from ctxplan's own vocabulary (Span,
// Elision) into the page-fault protocol's event, so a caller already holding a Plan
// does not have to hand-build a PageFaultEvent field by field.
func EventFromMiss(span Span, required, silentlyReconstructable bool) PageFaultEvent {
	state := PageFaultSpanLive
	switch {
	case span.Sealed:
		state = PageFaultSpanSealed
	case span.Tombstoned:
		state = PageFaultSpanTombstoned
	}
	return PageFaultEvent{
		SpanID:                  span.ID,
		Step:                    span.Step,
		Role:                    span.Role,
		State:                   state,
		Durability:              NormDurability(span.Durability),
		Required:                required,
		SilentlyReconstructable: silentlyReconstructable,
	}
}

// MaterializeFaultDecision is the thin bridge from a PageFaultDecision to the fault.go
// mechanism: it EXECUTES a PageFaultPageIn decision through DemandPage and passes every
// other outcome through untouched. This is deliberately the only place the two files
// meet — DecidePageFault itself never imports Store or touches bytes, keeping the
// decision pure and the execution effectful in one clearly-marked seam.
//
//   - PageFaultPageIn:          calls DemandPage; the returned Fault mirrors the
//     underlying disposition (served/resident/refused/absent) for the caller's audit.
//   - PageFaultContinuePointer: no page-in attempted; the input View is returned
//     unchanged and the Fault reports FaultResident-shaped bookkeeping is skipped —
//     the caller keeps only the pointer (decision.SpanID) for a later demand-page.
//   - PageFaultQueryUser / PageFaultDeny: no page-in attempted; the View is returned
//     unchanged. The caller is expected to surface decision.Reason to the human (query)
//     or to the refusal path (deny) — this function never blocks or prompts, it only
//     refrains from touching the store.
//
// The returned Fault is the zero value (Status "") for the three non-page-in outcomes,
// which is intentionally NOT a member of fault.go's closed vocabulary (served/resident/
// refused/absent): a caller must branch on the PageFaultDecision first, and a zero
// Fault is a visible tell that no page-in was attempted, not a silent "resident" or
// "absent" mislabel.
func MaterializeFaultDecision(ctx context.Context, store Store, in View, decision PageFaultDecision) (View, Fault, error) {
	if decision.Outcome != PageFaultPageIn {
		return in, Fault{}, nil
	}
	return DemandPage(ctx, store, in, decision.SpanID)
}
