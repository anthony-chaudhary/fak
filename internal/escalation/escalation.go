// Package escalation defines fak.escalation.v1 — the ONE typed escalation
// packet every interrupt surface shares (#2271, epic #2269, spine R2 of
// docs/notes/CONCEPT-NO-BABYSITTING-2026-07-01.md).
//
// Before this leaf every interrupt carried its own ad-hoc shape: `fak notify`
// stop events, refusal dispositions, operator-brief items. Nothing downstream
// could route, measure, or acknowledge them uniformly. Now one schema: a
// packet is a closed-token object — reason, loop/run/session/trace ids, a
// state digest, evidence refs, a bounded action menu with a safe default +
// expiry, and a cost-of-delay class. NO free-text field exists by
// construction ("decidable in seconds" is reviewed against the packet fields
// alone — TestPacketCarriesNoProseField pins that), which is what makes the
// packet routable and its handling measurable.
//
// Producers: FromNotify adapts one `fak notify` stop-event fire; FromRefusal
// adapts one ESCALATE-disposition refusal (any other disposition is refused —
// RETRYABLE/WAIT/TERMINAL never wake a human). The supervisor seat's
// EmitEscalation (#4479) builds a SourceSupervisor packet directly and emits
// through the same Ledger. Consumers: Fold projects the rows into open /
// expired / acked, pairing each packet with its FIRST ack row so R1
// (internal/operatortouches, escalation_handling_p50) computes handling time
// from the pair, and internal/supervisoragent takes the typed head fields.
//
// An ack lands as a ledger row and closure is idempotent per Rev: a
// re-emitted packet folds to one row by its deterministic ID (which embeds
// Rev), and duplicate acks fold to the first (PacketID, Rev) row. Sinks stay
// non-authoritative: they render the packet; an inbound action re-adjudicates
// through policy (#748 pillar 3, #743 owns the webhook half).
//
// This leaf imports nothing internal ON PURPOSE: waiting (R3) and
// operatortouches (R1) sit above it in the no-babysit stack, so the packet
// schema must stay at the bottom with no sibling edge back up.
package escalation

import (
	"errors"
	"fmt"
	"strconv"
	"strings"
	"time"
)

// Schema tags every packet row; AckSchema tags every ack row. One JSONL
// ledger carries both, distinguished by this field alone.
const (
	Schema    = "fak.escalation.v1"
	AckSchema = "fak.escalation.ack.v1"
)

// DefaultExpiry is the bounded wait before a packet's safe default fires
// (babysitting law B4: silence is bounded — every escalation carries an
// expiry, never an open-ended wait on the human). It mirrors the R3 queue's
// DefaultDeadline (internal/waiting); producers narrow it per head.
const DefaultExpiry = 2 * time.Hour

// Source is the closed producer token: which surface emitted the packet.
type Source string

const (
	SourceNotify     Source = "notify"     // a `fak notify` stop-event fire
	SourceRefusal    Source = "refusal"    // an ESCALATE-disposition refusal
	SourceSupervisor Source = "supervisor" // the supervisor seat's EscalateAction (#4479)
)

// Severity is the closed urgency token: does this packet block on a human
// decision (operator) or merely inform one (status)? The split mirrors the
// supervisoragent head (internal/supervisoragent Escalation.Severity).
type Severity string

const (
	SeverityStatus   Severity = "status"   // informational: nothing waits on the ack
	SeverityOperator Severity = "operator" // a human decision is the unblock
)

// CostOfDelay is the closed cost class of leaving the packet unhandled — the
// ranking signal, kept a token so the fold never invents an economic model.
type CostOfDelay string

const (
	CostNone      CostOfDelay = "none"        // nothing held: delay costs attention only
	CostSeatHeld  CostOfDelay = "seat_held"   // an in-flight run holds a dispatch seat
	CostLeaseHeld CostOfDelay = "lease_held"  // a lane lease is held while it waits
	CostRunBlock  CostOfDelay = "run_blocked" // forward progress on a run is stopped
)

// Actor is the closed ack-author token. Deliberately NOT an identity: no
// personal id ever lands in the ledger — policy re-adjudicates the action.
type Actor string

const (
	ActorOperator Actor = "operator" // a human decided one of the bounded actions
	ActorExpiry   Actor = "expiry"   // the deadline elapsed; the safe default fired
)

// EvidenceRef points at one witness artifact backing the packet — a kind
// token plus a machine ref (a path, a SHA, a ledger key). No summary field:
// evidence is pointed at, never quoted.
type EvidenceRef struct {
	Kind string `json:"kind"`
	Ref  string `json:"ref"`
}

// Packet is one fak.escalation.v1 escalation: the typed object every
// interrupt surface emits and every consumer routes on. Every field is a
// closed token, an id, a ref, or a number — Validate refuses prose.
type Packet struct {
	Schema   string   `json:"schema"` // always "fak.escalation.v1"
	ID       string   `json:"id"`     // deterministic: <source>/<anchor>/<run|_>#<rev> — assigned at emit
	Source   Source   `json:"source"`
	Reason   string   `json:"reason"` // closed reason token (UPPER_SNAKE) from the producer's own closed vocabulary; never prose
	Class    string   `json:"class"`  // escalation class token (lower_snake), e.g. "stop" / "refusal"
	Severity Severity `json:"severity"`

	LoopID    string `json:"loop_id,omitempty"`
	RunID     string `json:"run_id,omitempty"`
	Issue     string `json:"issue,omitempty"`
	SessionID string `json:"session_id,omitempty"`
	TraceID   string `json:"trace_id,omitempty"`
	Rev       uint64 `json:"rev"` // producer idempotency key, monotonic per anchor (the notify Rev contract)

	StateDigest string        `json:"state_digest"`       // closed run-state token at emit (lower_snake), e.g. "paused" / "budget" / "denied"
	Evidence    []EvidenceRef `json:"evidence,omitempty"` // witness refs, pointed at, never quoted

	Actions     []string `json:"actions"`      // the bounded menu (lower_snake tokens); the operator picks from these and nothing else
	SafeDefault string   `json:"safe_default"` // the menu member that fires on expiry (law B4)

	EmittedAtUnixNano int64       `json:"emitted_at_unix_nano"`
	ExpiresAtUnixNano int64       `json:"expires_at_unix_nano"`
	CostOfDelay       CostOfDelay `json:"cost_of_delay"`
}

// Ack is the closure row for one packet: which bounded action was taken and
// by which closed actor class. Idempotent per (PacketID, Rev) — a re-delivery
// folds to the first row. It carries no identity and no prose.
type Ack struct {
	Schema          string `json:"schema"`    // always "fak.escalation.ack.v1"
	PacketID        string `json:"packet_id"` // the Packet.ID this closes
	Action          string `json:"action"`    // one of the packet's bounded actions (lower_snake)
	Actor           Actor  `json:"actor"`
	Rev             uint64 `json:"rev"` // mirrors the packet's Rev — the idempotency key
	AckedAtUnixNano int64  `json:"acked_at_unix_nano"`
}

// Sentinel refusals. Every validation failure wraps one of these, so a caller
// branches on the class, never on message prose.
var (
	ErrSchema        = errors.New("escalation: wrong or missing schema tag")
	ErrProse         = errors.New("escalation: field must be a closed token, never prose")
	ErrUnroutable    = errors.New("escalation: no routing id (need loop/run/session/trace)")
	ErrUnsafeDefault = errors.New("escalation: safe default missing or outside the bounded action menu")
	ErrExpiry        = errors.New("escalation: emitted/expiry timestamps missing or inverted")
	ErrRev           = errors.New("escalation: rev must be >= 1 (the idempotency key)")
	ErrNotEscalate   = errors.New("escalation: refusal disposition is not ESCALATE — no packet")
)

// Invariant: escalation packets are fail-closed and validate schema strictly.
// Every vocabulary field must conform to closed tokens, at least one routing
// identifier must be present, actions must contain the safe default, and timestamps
// must maintain strictly forward temporal progression without prose.
//
// Validate is the fail-closed schema gate: a packet that does not pass never
// reaches the ledger. It enforces closed-token shape on every vocabulary
// field — the "no prose" law — plus routability, the bounded-menu/safe-default
// pair, and the expiry contract.
func (p Packet) Validate() error {
	if p.Schema != Schema {
		return fmt.Errorf("%w: got %q want %q", ErrSchema, p.Schema, Schema)
	}
	switch p.Source {
	case SourceNotify, SourceRefusal, SourceSupervisor:
	default:
		return fmt.Errorf("%w: source %q", ErrProse, p.Source)
	}
	if !isReasonToken(p.Reason) {
		return fmt.Errorf("%w: reason %q", ErrProse, p.Reason)
	}
	if !isLowerToken(p.Class) {
		return fmt.Errorf("%w: class %q", ErrProse, p.Class)
	}
	switch p.Severity {
	case SeverityStatus, SeverityOperator:
	default:
		return fmt.Errorf("%w: severity %q", ErrProse, p.Severity)
	}
	if p.LoopID == "" && p.RunID == "" && p.SessionID == "" && p.TraceID == "" {
		return ErrUnroutable
	}
	if p.Rev == 0 {
		return ErrRev
	}
	if !isLowerToken(p.StateDigest) {
		return fmt.Errorf("%w: state_digest %q", ErrProse, p.StateDigest)
	}
	for _, ev := range p.Evidence {
		if !isLowerToken(strings.ToLower(ev.Kind)) || ev.Ref == "" {
			return fmt.Errorf("%w: evidence ref %q:%q", ErrProse, ev.Kind, ev.Ref)
		}
	}
	if len(p.Actions) == 0 {
		return fmt.Errorf("%w: empty action menu", ErrUnsafeDefault)
	}
	seen := map[string]bool{}
	for _, a := range p.Actions {
		if !isLowerToken(a) {
			return fmt.Errorf("%w: action %q", ErrProse, a)
		}
		if seen[a] {
			return fmt.Errorf("%w: duplicate action %q", ErrProse, a)
		}
		seen[a] = true
	}
	if p.SafeDefault == "" || !seen[p.SafeDefault] {
		return fmt.Errorf("%w: safe_default %q", ErrUnsafeDefault, p.SafeDefault)
	}
	if p.EmittedAtUnixNano <= 0 || p.ExpiresAtUnixNano <= p.EmittedAtUnixNano {
		return fmt.Errorf("%w: emitted=%d expires=%d", ErrExpiry, p.EmittedAtUnixNano, p.ExpiresAtUnixNano)
	}
	switch p.CostOfDelay {
	case CostNone, CostSeatHeld, CostLeaseHeld, CostRunBlock:
	default:
		return fmt.Errorf("%w: cost_of_delay %q", ErrProse, p.CostOfDelay)
	}
	return nil
}

// Validate is the ack row's fail-closed gate: closed actor class, token-shaped
// action, a packet to bind to, and the Rev idempotency key.
func (a Ack) Validate() error {
	if a.Schema != AckSchema {
		return fmt.Errorf("%w: got %q want %q", ErrSchema, a.Schema, AckSchema)
	}
	if a.PacketID == "" {
		return fmt.Errorf("%w: ack without packet_id", ErrUnroutable)
	}
	if !isLowerToken(a.Action) {
		return fmt.Errorf("%w: action %q", ErrProse, a.Action)
	}
	switch a.Actor {
	case ActorOperator, ActorExpiry:
	default:
		return fmt.Errorf("%w: actor %q", ErrProse, a.Actor)
	}
	if a.Rev == 0 {
		return ErrRev
	}
	if a.AckedAtUnixNano <= 0 {
		return fmt.Errorf("%w: acked_at=%d", ErrExpiry, a.AckedAtUnixNano)
	}
	return nil
}

// deriveID is the deterministic packet identity: source/anchor/run#rev. The
// anchor is the first routing id present (trace, then loop, then session), so
// a producer re-firing the same (anchor, rev) — the notify idempotency
// contract — derives the SAME id and the fold collapses the rows.
func (p Packet) deriveID() string {
	anchor := p.TraceID
	if anchor == "" {
		anchor = p.LoopID
	}
	if anchor == "" {
		anchor = p.SessionID
	}
	run := p.RunID
	if run == "" {
		run = "_"
	}
	return string(p.Source) + "/" + anchor + "/" + run + "#" + strconv.FormatUint(p.Rev, 10)
}

// NotifyFire is the typed head of one `fak notify` stop-event fire (the
// cmd/fak StopEvent contract: closed stop-reason token, lowercase run-state
// token, Rev monotonic per trace). The producer passes what it has; FromNotify
// derives the rest.
type NotifyFire struct {
	TraceID     string
	LoopID      string
	RunID       string
	Issue       string
	Reason      string // closed stop-reason token (session/decide.go vocabulary)
	To          string // lowercase run-state token, or "budget" for a budget-origin event
	Rev         uint64
	At          time.Time
	Expiry      time.Duration // bounded wait before the safe default; zero -> DefaultExpiry
	CostOfDelay CostOfDelay   // zero -> CostNone
	Evidence    []EvidenceRef
}

// operatorReasonHints marks a stop-reason as blocked-on-operator. It MIRRORS
// internal/waiting BlockedReasonHints verbatim — this leaf cannot import
// waiting (R3 folds R2's ack rows, so the edge must point the other way) and
// the two tables are kept in sync by review, both deliberately conservative.
var operatorReasonHints = []string{
	"ESCALAT", "APPROV", "AUTH", "LOGIN", "PERMISSION",
	"MANUAL", "OPERATOR", "NEEDS_HUMAN", "WAITING_ON_HUMAN", "BLOCKED_ON",
}

// FromNotify builds the packet for one notify fire — producer #1. A
// blocked-on-operator reason makes an operator-severity packet whose menu is
// {resume, hold, release_held_resources} with the conservative B4 default
// (release what it holds); any other stop-reason is a status packet whose only
// action is to note it. The packet id is derived, not chosen.
func FromNotify(f NotifyFire) (Packet, error) {
	if f.At.IsZero() {
		return Packet{}, fmt.Errorf("%w: notify fire without a timestamp", ErrExpiry)
	}
	expiry := f.Expiry
	if expiry <= 0 {
		expiry = DefaultExpiry
	}
	cost := f.CostOfDelay
	if cost == "" {
		cost = CostNone
	}
	severity := SeverityStatus
	actions := []string{"ack_noted"}
	safeDefault := "ack_noted"
	if isOperatorReason(f.Reason) {
		severity = SeverityOperator
		actions = []string{"resume", "hold", "release_held_resources"}
		safeDefault = "release_held_resources"
	}
	state := strings.ToLower(strings.TrimSpace(f.To))
	p := Packet{
		Schema:            Schema,
		Source:            SourceNotify,
		Reason:            strings.ToUpper(strings.TrimSpace(f.Reason)),
		Class:             "stop",
		Severity:          severity,
		LoopID:            f.LoopID,
		RunID:             f.RunID,
		Issue:             f.Issue,
		TraceID:           f.TraceID,
		Rev:               f.Rev,
		StateDigest:       state,
		Evidence:          append([]EvidenceRef(nil), f.Evidence...),
		Actions:           actions,
		SafeDefault:       safeDefault,
		EmittedAtUnixNano: f.At.UnixNano(),
		ExpiresAtUnixNano: f.At.Add(expiry).UnixNano(),
		CostOfDelay:       cost,
	}
	p.ID = p.deriveID()
	if err := p.Validate(); err != nil {
		return Packet{}, err
	}
	return p, nil
}

// RefusalHead is the typed head of one adjudicated refusal: the closed reason
// token and its disposition (the kernel's RETRYABLE/WAIT/ESCALATE/TERMINAL
// fold). Only ESCALATE reaches a human.
type RefusalHead struct {
	Disposition string // must be "ESCALATE"
	Reason      string // closed refusal-reason token (e.g. SELF_MODIFY)
	SessionID   string
	TraceID     string
	RunID       string
	Issue       string
	Rev         uint64
	At          time.Time
	Expiry      time.Duration // zero -> DefaultExpiry
	CostOfDelay CostOfDelay   // zero -> CostRunBlock (a refused turn stops the run)
	Evidence    []EvidenceRef
}

// FromRefusal builds the packet for one ESCALATE-disposition refusal —
// producer #2. Any other disposition returns ErrNotEscalate: RETRYABLE, WAIT
// and TERMINAL are the machine's to handle and never page a human. The menu is
// {approve_retry, deny} and the safe default is deny — an unreviewed escalated
// refusal STAYS refused (fail closed), it never auto-approves on expiry.
func FromRefusal(h RefusalHead) (Packet, error) {
	if strings.ToUpper(strings.TrimSpace(h.Disposition)) != "ESCALATE" {
		return Packet{}, fmt.Errorf("%w: disposition %q", ErrNotEscalate, h.Disposition)
	}
	if h.At.IsZero() {
		return Packet{}, fmt.Errorf("%w: refusal head without a timestamp", ErrExpiry)
	}
	expiry := h.Expiry
	if expiry <= 0 {
		expiry = DefaultExpiry
	}
	cost := h.CostOfDelay
	if cost == "" {
		cost = CostRunBlock
	}
	p := Packet{
		Schema:            Schema,
		Source:            SourceRefusal,
		Reason:            strings.ToUpper(strings.TrimSpace(h.Reason)),
		Class:             "refusal",
		Severity:          SeverityOperator,
		RunID:             h.RunID,
		Issue:             h.Issue,
		SessionID:         h.SessionID,
		TraceID:           h.TraceID,
		Rev:               h.Rev,
		StateDigest:       "denied",
		Evidence:          append([]EvidenceRef(nil), h.Evidence...),
		Actions:           []string{"approve_retry", "deny"},
		SafeDefault:       "deny",
		EmittedAtUnixNano: h.At.UnixNano(),
		ExpiresAtUnixNano: h.At.Add(expiry).UnixNano(),
		CostOfDelay:       cost,
	}
	p.ID = p.deriveID()
	if err := p.Validate(); err != nil {
		return Packet{}, err
	}
	return p, nil
}

// isOperatorReason is the data-table verdict: does this stop-reason mean a
// human decision is the unblock? (The mirror of waiting.IsBlockedOnOperator.)
func isOperatorReason(reason string) bool {
	r := strings.ToUpper(strings.TrimSpace(reason))
	if r == "" {
		return false
	}
	for _, h := range operatorReasonHints {
		if strings.Contains(r, h) {
			return true
		}
	}
	return false
}

// isSnakeToken accepts a closed snake-case token in ONE letter register: a first byte
// in [lo, hi], then at most 63 more bytes drawn from that same register, the digits, and
// '_'. Anything with spacing, the other casing, or punctuation beyond that is prose and
// refused. The two registers the ledger admits are spelled out by the wrappers below.
func isSnakeToken(s string, lo, hi byte) bool {
	if s == "" || len(s) > 64 {
		return false
	}
	if s[0] < lo || s[0] > hi {
		return false
	}
	for i := 1; i < len(s); i++ {
		c := s[i]
		if (c < lo || c > hi) && (c < '0' || c > '9') && c != '_' {
			return false
		}
	}
	return true
}

// isReasonToken accepts a closed UPPER_SNAKE reason token — [A-Z][A-Z0-9_]*,
// at most 64 bytes.
func isReasonToken(s string) bool { return isSnakeToken(s, 'A', 'Z') }

// isLowerToken accepts a closed lower_snake token — [a-z][a-z0-9_]*, at most
// 64 bytes. Same prose refusal as isReasonToken, lowercase register.
func isLowerToken(s string) bool { return isSnakeToken(s, 'a', 'z') }
