package waiting

import (
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/loopmgr"
)

// Schema tags the JSON queue.
const Schema = "fak.waiting-on-human.v1"

// DefaultDeadline is the bounded wait before the safe default fires (babysitting
// law B4: silence is bounded; every blocked item carries a deadline, never an
// open-ended wait on the human). Two hours matches the conservative operator
// envelope; callers narrow it per reason class via Params.Deadline.
const DefaultDeadline = 2 * time.Hour

// ItemStatus is the lifecycle of one queue row.
type ItemStatus string

const (
	// StatusWaiting: filed, its run still in-flight, age under the deadline.
	StatusWaiting ItemStatus = "waiting"
	// StatusExpiredDefault: age reached the deadline; SafeDefault is the
	// prescribed action that fires through the normal admission path.
	StatusExpiredDefault ItemStatus = "expired_default"
	// StatusClosedRunEnded: the run reached a terminal end, so the item is no
	// longer waiting. The SOURCE is the run end, NOT a proven operator ack —
	// see Queue.AckClosureNotYet; operator-ack closure is not_yet pending R2.
	StatusClosedRunEnded ItemStatus = "closed_run_ended"
)

// BlockedReasonHints are reason-token substrings (upper-cased match) that mark a
// notify as blocked-on-operator: the fleet filed a ticket on the human. Declared
// data, tuned as real reasons appear — never inferred. Deliberately conservative:
// it lists only tokens that unambiguously mean "waiting on a human decision", so
// an informational notify (e.g. a DONE_WITNESSED completion) is NOT binned as
// blocked. A notify whose reason matches none of these is simply not a queue row.
var BlockedReasonHints = []string{
	"ESCALAT", "APPROV", "AUTH", "LOGIN", "PERMISSION",
	"MANUAL", "OPERATOR", "NEEDS_HUMAN", "WAITING_ON_HUMAN", "BLOCKED_ON",
}

// IsBlockedOnOperator is the data-table verdict: does this notify reason mean
// the fleet is waiting on a human decision?
func IsBlockedOnOperator(reason string) bool {
	r := strings.ToUpper(strings.TrimSpace(reason))
	if r == "" {
		return false
	}
	for _, h := range BlockedReasonHints {
		if strings.Contains(r, h) {
			return true
		}
	}
	return false
}

// SafeDefaultFor maps a blocked reason to the safe default that fires on expiry.
// Declared data: the expiry action is never invented — it is the bounded, safe
// recovery for that reason class. The fold PRESCRIBES the action; execution goes
// through the normal admission path (out of scope for this pure fold).
func SafeDefaultFor(reason string) string {
	r := strings.ToUpper(reason)
	switch {
	case strings.Contains(r, "APPROV"):
		return "release_held_resources_and_requeue" // an unreviewed item expires: free its seat/lease, requeue the work
	case strings.Contains(r, "AUTH"), strings.Contains(r, "LOGIN"), strings.Contains(r, "PERMISSION"):
		return "release_held_resources_and_skip" // a cred-gated item expires: free its seat, skip until the cred is refreshed
	case strings.Contains(r, "ESCALAT"):
		return "release_held_resources" // a generic escalation expires: free what it holds (the conservative default)
	default:
		return "release_held_resources" // law B4 — silence is bounded; an expired hold is always released
	}
}

// HeldResources is what the blocked item is sitting on while it waits. Derived
// purely from the loop ledger: a run that has started with no terminal end holds
// a dispatch seat; the loop id names the lane it occupies. Lease/worker refs the
// notify event itself carried (in its metrics/evidence) are surfaced verbatim.
type HeldResources struct {
	WorkerSeat bool     `json:"worker_seat"`         // the run is in-flight (started, no end) — it holds a dispatch seat
	LoopID     string   `json:"loop_id,omitempty"`   // the lane the run occupies
	LeaseRefs  []string `json:"lease_refs,omitempty"` // lease/worker refs the notify event carried, if any
}

// Item is one kernel object in the waiting-on-human queue: a blocked-on-operator
// notify filed as a ticket on the human, with age, held resources, deadline, and
// the safe default that fires on expiry. Babysitting inverted: the human never
// scans for these — the queue ranks them and surfaces the cost-of-delay.
type Item struct {
	Key              string        `json:"key"` // stable identity: loop/run#notify_seq
	LoopID           string        `json:"loop_id"`
	RunID            string        `json:"run_id,omitempty"`
	Reason           string        `json:"reason,omitempty"`
	Summary          string        `json:"summary,omitempty"`
	NotifySeq        uint64        `json:"notify_seq"`
	FiledAtUnixNano  int64         `json:"filed_at_unix_nano"` // the notify timestamp
	AgeSeconds       float64       `json:"age_seconds"`        // AsOf - filed
	DeadlineUnixNano int64         `json:"deadline_unix_nano"` // filed + deadline
	PastDeadline     bool          `json:"past_deadline"`
	SafeDefault      string        `json:"safe_default,omitempty"` // prescribed only once expired
	Status           ItemStatus    `json:"status"`
	Held             HeldResources `json:"held"`
}

// Queue is the fak.waiting-on-human.v1 fold.
type Queue struct {
	Schema       string `json:"schema"`
	AsOfUnixNano int64  `json:"as_of_unix_nano"`

	Items    []Item `json:"items"`    // active rows: waiting + expired_default, oldest-first (ranked by cost-of-delay)
	Resolved []Item `json:"resolved"` // closed this window: the run reached a terminal end (source = run, not an ack)

	ByStatus         map[string]int `json:"by_status"`
	OldestAgeSeconds float64        `json:"oldest_age_seconds,omitempty"` // the longest any active row has waited
	PastDeadline     int            `json:"past_deadline"`                // active rows whose deadline elapsed

	// AckClosureNotYet is the honesty fence (mirrors R1 / internal/operatortouches):
	// an active row closes today only on a terminal run end (source = run) or on
	// expiry (the safe default). It NEVER closes on an explicit operator ack,
	// because the R2 escalation packet's ack row does not exist yet (#2271,
	// dos verify no-babysit/R2 -> shipped=false). When R2 lands, ack rows fold in
	// here as a fourth closure source — until then the fence names the gap.
	AckClosureNotYet string `json:"ack_closure_not_yet"`
}

// Params tunes the fold; zero values take the documented defaults.
type Params struct {
	AsOf     time.Time
	Deadline time.Duration // the bounded wait before the safe default fires
}

// Fold computes the R3 queue from loop-event rows (any number of ledgers,
// pre-concatenated). Pure: no I/O, no clock — AsOf comes from the caller.
//
// A row enters the queue when a notify is filed that IsBlockedOnOperator. It
// leaves when its run reaches a terminal end (closed_run_ended) — or it is
// flagged expired_default once its age passes the deadline, prescribing the safe
// default. Each row carries the resources it holds while it waits.
func Fold(events []loopmgr.Event, p Params) Queue {
	if p.Deadline <= 0 {
		p.Deadline = DefaultDeadline
	}
	asOf := p.AsOf
	if asOf.IsZero() {
		// Deterministic fallback: the newest row. Callers wanting "now" pass it.
		var maxTS int64
		for _, ev := range events {
			if ev.TSUnixNano > maxTS {
				maxTS = ev.TSUnixNano
			}
		}
		asOf = time.Unix(0, maxTS).UTC()
	}

	// Run state: which (loop,run) holds a dispatch seat (started, no terminal end),
	// and which reached a terminal end. Derived from the whole ledger, not just
	// the notify, so Held.WorkerSeat reflects the true in-flight state.
	type runKey struct{ loop, run string }
	inflight := map[runKey]bool{}
	ended := map[runKey]bool{}
	for _, ev := range events {
		if ev.RunID == "" {
			continue
		}
		k := runKey{ev.LoopID, ev.RunID}
		if ev.Kind == loopmgr.EventStart {
			inflight[k] = true
		}
		if ev.Kind == loopmgr.EventEnd {
			ended[k] = true
		}
	}

	q := Queue{
		Schema:       Schema,
		AsOfUnixNano: asOf.UnixNano(),
		ByStatus:     map[string]int{},
	}

	for _, ev := range events {
		if ev.Kind != loopmgr.EventNotify || !IsBlockedOnOperator(ev.Reason) {
			continue
		}
		filedAt := ev.TSUnixNano
		age := asOf.Sub(time.Unix(0, filedAt))
		if age < 0 {
			age = 0 // a notify dated after AsOf (clock skew) waits zero, never negative
		}
		k := runKey{ev.LoopID, ev.RunID}
		held := HeldResources{
			WorkerSeat: inflight[k] && !ended[k],
			LoopID:     ev.LoopID,
			LeaseRefs:  leaseRefs(ev),
		}

		it := Item{
			Key:              itemKey(ev),
			LoopID:           ev.LoopID,
			RunID:           ev.RunID,
			Reason:           ev.Reason,
			Summary:          ev.Summary,
			NotifySeq:        ev.Seq,
			FiledAtUnixNano:  filedAt,
			AgeSeconds:       age.Seconds(),
			DeadlineUnixNano: time.Unix(0, filedAt).Add(p.Deadline).UnixNano(),
			Held:             held,
		}
		it.PastDeadline = asOf.UnixNano() >= it.DeadlineUnixNano

		switch {
		case ended[k]:
			it.Status = StatusClosedRunEnded
			q.Resolved = append(q.Resolved, it)
		case it.PastDeadline:
			it.Status = StatusExpiredDefault
			it.SafeDefault = SafeDefaultFor(ev.Reason)
			q.Items = append(q.Items, it)
		default:
			it.Status = StatusWaiting
			q.Items = append(q.Items, it)
		}
		q.ByStatus[string(it.Status)]++
	}

	// Rank active rows by cost-of-delay: oldest (longest-waiting) first, with
	// expired rows ahead of waiting rows at equal age. This is the order an
	// operator (or the expiry path) consumes the queue.
	sort.SliceStable(q.Items, func(i, j int) bool {
		if q.Items[i].PastDeadline != q.Items[j].PastDeadline {
			return q.Items[i].PastDeadline // expired rows first
		}
		return q.Items[i].FiledAtUnixNano < q.Items[j].FiledAtUnixNano
	})
	sort.SliceStable(q.Resolved, func(i, j int) bool {
		return q.Resolved[i].FiledAtUnixNano < q.Resolved[j].FiledAtUnixNano
	})

	for _, it := range q.Items {
		if it.AgeSeconds > q.OldestAgeSeconds {
			q.OldestAgeSeconds = it.AgeSeconds
		}
		if it.PastDeadline {
			q.PastDeadline++
		}
	}

	q.AckClosureNotYet = "operator-ack closure not_yet until the R2 escalation packet's ack row exists (#2271); rows close today only on a terminal run end or on expiry"
	return q
}

// itemKey is the stable identity of one queue row.
func itemKey(ev loopmgr.Event) string {
	run := ev.RunID
	if run == "" {
		run = "_"
	}
	var b strings.Builder
	b.Grow(len(ev.LoopID) + 1 + len(run) + 16)
	b.WriteString(ev.LoopID)
	b.WriteByte('/')
	b.WriteString(run)
	b.WriteByte('#')
	// Seq is the row identity inside the ledger; the key pins the exact ticket.
	b.WriteString(utoa(ev.Seq))
	return b.String()
}

// leaseRefs surfaces lease/worker refs the notify event itself carried, verbatim.
// Honest pass-through: the loop ledger does not own the lease store, so a ref is
// only present when the producer attached one to the notify (metrics/evidence).
func leaseRefs(ev loopmgr.Event) []string {
	var refs []string
	for _, er := range ev.EvidenceRefs {
		k := strings.ToUpper(er.Kind)
		if strings.Contains(k, "LEASE") || strings.Contains(k, "WORKER") || strings.Contains(k, "SEAT") {
			refs = append(refs, er.Kind+":"+er.Ref)
		}
	}
	for k := range ev.Metrics {
		ku := strings.ToUpper(k)
		if strings.Contains(ku, "LEASE") || strings.Contains(ku, "WORKER") || strings.Contains(ku, "SEAT") {
			refs = append(refs, k)
		}
	}
	return refs
}

// utoa is a stdlib-free uint64 -> string (avoids fmt for a hot-path-free leaf).
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
