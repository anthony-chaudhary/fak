package wiprecon

// adopt.go — WITNESSED ADOPTION AND RESUME (#5998): the pure half of claiming a crashed
// session's checkpoint and carrying its work to a landing.
//
// reclaim.go produces a RANKED QUEUE of RECLAIM checkpoints. Nothing consumed it, so the
// recovery story stopped at "here is a lever". The missing piece is not a lander — it is
// OWNERSHIP, and two facts make that the missing piece rather than a nicety:
//
//   - A shared trunk runs many agents against ONE queue. With no claim, two successors
//     read the same head row, both materialize the same delta, and the second one lands on
//     top of the first — the exact multi-writer collision the rest of this tree refuses.
//   - A successor is exactly as mortal as the session it is rescuing. A claim that lives
//     only in a process dies with it, and the checkpoint is stranded a SECOND time — now
//     with a half-written working copy nobody can attribute to anybody.
//
// So the claim is a RECEIPT: a durable, compare-and-swap-guarded record bound to the
// checkpoint's ref AND its object SHA, naming the successor session. The CAS is what makes
// "exactly one successor wins" a property of git rather than of timing — the loser
// observes a failed swap, not a corrupted tree. The durability is what makes a successor
// crash RESUMABLE rather than terminal: the receipt is journaled BEFORE the first byte is
// materialized, so recovery never has to guess whether mutation had already started.
//
// FAIL-SAFE, in the same direction as recon.go. Every state this file cannot prove safe
// resolves to a REFUSAL, never to a claim:
//
//   - a checkpoint that is not RECLAIM is never adopted (QUARANTINE stays an operator's
//     call, and a SKIP checkpoint's owner is alive and still holds its own work);
//   - a receipt bound to a DIFFERENT object SHA refuses rather than materializing bytes it
//     does not describe — the binding is to ref+SHA precisely so a checkpoint that moved
//     under a claim cannot be silently inherited by it;
//   - a live or unexpired incumbent HOLDS.
//
// The single path that overrides an existing claim — TAKEOVER — requires ALL THREE of
// liveness (the incumbent holds no lease), TTL (its claim has lapsed), and CAS (nobody
// moved the receipt meanwhile), and it writes an AUDIT EVENT naming who it displaced. Two
// of the three are not enough: a lapsed TTL on a live successor is a slow successor, and a
// dead successor inside its TTL may still have a peer mid-flight.
//
// Pure: no git, no clock, no I/O. cmd/fak/wip_adopt.go supplies the receipt bytes, the
// liveness bit, and the clock, and performs the swap.

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
)

// Phase is how far one adoption has carried its checkpoint. It advances forward only, and
// it is the resume anchor: a successor restarting after a crash reads the phase to learn
// what its predecessor self had already done.
type Phase string

const (
	// PhaseAdopted: the claim is journaled and NOTHING has been materialized. A crash
	// here is free — resume re-materializes from the checkpoint object.
	PhaseAdopted Phase = "ADOPTED"
	// PhaseMaterialized: the delta is written to the receipt's Target and its bytes were
	// verified against the checkpoint object. A crash here resumes at the landing step.
	PhaseMaterialized Phase = "MATERIALIZED"
	// PhaseLanded: a commit witnessed the delta. The checkpoint ref is STILL PRESENT —
	// this package never authorizes deleting it; `fak wip reap` does that, and only once
	// the delta is provably in HEAD.
	PhaseLanded Phase = "LANDED"
)

// DefaultTTLSeconds bounds how long one claim is honoured without a renewal. Fifteen
// minutes is longer than any materialize-and-land takes and far shorter than a stranded
// checkpoint's useful life, so the common case never contends and an abandoned claim never
// pins a recoverable delta for an operator-visible length of time.
const DefaultTTLSeconds int64 = 900

// receiptMarker prefixes the one line of a receipt object's body that carries JSON, the
// same shape wipref.Stamp uses for checkpoint metadata: a human reading the raw object
// sees a labelled line, and a parser needs no schema negotiation to find it.
const receiptMarker = "fak-wip-adopt: "

// auditMax bounds the trail one receipt carries. A crash-loop appends an event per
// attempt and the receipt is a single object rewritten on every transition, so an
// unbounded trail turns a loop into unbounded object growth. The OLDEST events are
// dropped: what matters about a contested checkpoint is who holds it now and who was
// displaced most recently.
const auditMax = 32

// Audit event names. Closed vocabulary — an unknown value in a decoded receipt is data
// from a newer writer, not a reason to refuse.
const (
	EventAdopted      = "ADOPTED"
	EventResumed      = "RESUMED"
	EventTakeover     = "TAKEOVER"
	EventMaterialized = "MATERIALIZED"
	EventLanded       = "LANDED"
)

// AuditEvent is one entry in a receipt's history. From is set only on a TAKEOVER, naming
// the displaced successor — the done-condition that a takeover is never silent.
type AuditEvent struct {
	At     int64  `json:"at"`
	Event  string `json:"event"`
	Actor  string `json:"actor"`
	From   string `json:"from,omitempty"`
	Detail string `json:"detail,omitempty"`
}

// Receipt is the durable ownership record for one adopted checkpoint.
//
// CheckpointRef and CheckpointSHA together are the BINDING: the claim is over one object,
// not over a name that may later point somewhere else. DeltaDigest is the third leg —
// content addressing over the patch bytes, so a successor that resumes can prove the delta
// it is about to write is the delta it claimed, not merely that a ref still resolves.
type Receipt struct {
	Session       string `json:"session"`
	CheckpointRef string `json:"checkpoint_ref"`
	CheckpointSHA string `json:"checkpoint_sha"`
	// DeltaDigest is a content hash of the checkpoint's patch bytes. Empty is allowed —
	// a caller that could not compute it must then re-verify against CheckpointSHA alone,
	// which is weaker but never wrong.
	DeltaDigest string `json:"delta_digest,omitempty"`
	Successor   string `json:"successor"`
	// SuccessorHost is diagnostic only. Ownership is decided by session identity and the
	// CAS, never by host, because a fleet routinely runs two sessions on one host.
	SuccessorHost string `json:"successor_host,omitempty"`
	Phase         Phase  `json:"phase"`
	// Target is where the successor materialized the delta — an isolated worker path or
	// an explicit patch file. Carried in the receipt so a resume finds its own prior work
	// instead of choosing a fresh location and orphaning it.
	Target     string `json:"target,omitempty"`
	AdoptedAt  int64  `json:"adopted_at"`
	RenewedAt  int64  `json:"renewed_at"`
	TTLSeconds int64  `json:"ttl_seconds"`
	// Attempt counts claims over this checkpoint, resumes and takeovers included. It is
	// the attempt history the recovery queue surfaces: a row on its fifth attempt is a
	// row that wants an operator, not a sixth automatic try.
	Attempt   int          `json:"attempt"`
	LandedSHA string       `json:"landed_sha,omitempty"`
	Audit     []AuditEvent `json:"audit,omitempty"`
}

// AdoptVerdict is the closed decision vocabulary. Three grant, five refuse.
type AdoptVerdict string

const (
	// AdoptGrant: unclaimed and reclaimable — this successor takes it.
	AdoptGrant AdoptVerdict = "GRANT"
	// AdoptResume: this successor already holds the claim over this exact object and is
	// continuing from the recorded phase.
	AdoptResume AdoptVerdict = "RESUME"
	// AdoptTakeover: the incumbent is provably gone (no live lease) AND its claim has
	// lapsed. Emits an audit event naming the displaced successor.
	AdoptTakeover AdoptVerdict = "TAKEOVER"
	// AdoptHeld: another successor holds a claim that is live or still inside its TTL.
	AdoptHeld AdoptVerdict = "HELD"
	// AdoptMoved: a receipt exists but is bound to a different checkpoint object — the
	// ref moved under the claim. Refuse rather than inherit.
	AdoptMoved AdoptVerdict = "CHECKPOINT_MOVED"
	// AdoptSettled: this checkpoint already landed under this receipt. A benign no-op, so
	// a re-run of a completed recovery is idempotent rather than an error.
	AdoptSettled AdoptVerdict = "ALREADY_LANDED"
	// AdoptRefused: the checkpoint does not reconcile RECLAIM. Notably QUARANTINE, which
	// is never auto-applied, and SKIP, whose owner is still alive.
	AdoptRefused AdoptVerdict = "NOT_RECLAIMABLE"
	// AdoptMalformed: the request itself is incomplete.
	AdoptMalformed AdoptVerdict = "INCOMPLETE_REQUEST"
)

// Granted reports whether a verdict authorizes writing a receipt and materializing bytes.
// Everything else is a refusal a caller must surface without mutating anything.
func (v AdoptVerdict) Granted() bool {
	switch v {
	case AdoptGrant, AdoptResume, AdoptTakeover:
		return true
	}
	return false
}

// AdoptRequest is one successor's bid, carrying only facts its caller witnessed.
type AdoptRequest struct {
	Session       string // the CRASHED session whose checkpoint is being adopted
	Action        Action // that checkpoint's current reconciliation verdict
	CheckpointRef string
	CheckpointSHA string
	DeltaDigest   string
	Successor     string // the adopting session
	SuccessorHost string
	Now           int64
	TTLSeconds    int64
	// IncumbentLive is whether the CURRENT receipt's successor holds a live lease. False
	// when there is no incumbent. A caller that cannot read liveness must pass TRUE: that
	// direction refuses a takeover, which is the safe way to be wrong.
	IncumbentLive bool
}

// AdoptDecision pairs the verdict with the sentence a refusal has to be readable as.
type AdoptDecision struct {
	Verdict AdoptVerdict `json:"verdict"`
	Reason  string       `json:"reason"`
}

// DecideAdopt is the whole ownership rule. cur is the receipt currently stored for this
// checkpoint, or nil when none is.
//
// Order is deliberate and each step is the fail-safe one:
//
//  1. an incomplete request can never be a claim;
//  2. an already-landed receipt short-circuits before liveness matters, so re-running a
//     finished recovery is idempotent instead of racing its own success;
//  3. a non-RECLAIM checkpoint is refused BEFORE any receipt reasoning — a QUARANTINE
//     checkpoint must not become adoptable merely because a stale receipt exists;
//  4. no receipt → grant;
//  5. a receipt bound to a different SHA refuses even for the SAME successor, because the
//     bytes it authorized are not the bytes now on the ref;
//  6. same successor → resume;
//  7. live or unexpired incumbent → hold;
//  8. otherwise → takeover, audited.
func DecideAdopt(cur *Receipt, req AdoptRequest) AdoptDecision {
	if req.Session == "" || req.Successor == "" || req.CheckpointSHA == "" {
		return AdoptDecision{AdoptMalformed, "an adoption needs a crashed session, an adopting successor, and a checkpoint object SHA"}
	}
	if cur != nil && cur.Phase == PhaseLanded && cur.CheckpointSHA == req.CheckpointSHA {
		return AdoptDecision{AdoptSettled, fmt.Sprintf(
			"%s already landed the %s checkpoint (%s)%s — nothing left to adopt",
			cur.Successor, req.Session, shortOID(req.CheckpointSHA), landedAs(cur.LandedSHA))}
	}
	if req.Action != ActReclaim {
		return AdoptDecision{AdoptRefused, fmt.Sprintf(
			"checkpoint %s reconciles %s; only %s is adoptable, because %s is an operator's call and %s belongs to a session that is still alive",
			req.Session, req.Action, ActReclaim, ActQuarantine, ActSkip)}
	}
	if cur == nil {
		return AdoptDecision{AdoptGrant, fmt.Sprintf(
			"checkpoint %s is %s and unclaimed; %s takes it at %s",
			req.Session, ActReclaim, req.Successor, shortOID(req.CheckpointSHA))}
	}
	if cur.CheckpointSHA != req.CheckpointSHA {
		return AdoptDecision{AdoptMoved, fmt.Sprintf(
			"the %s adoption receipt is bound to checkpoint %s but the ref now holds %s — the checkpoint moved under the claim, so its bytes are not the bytes that were adopted",
			req.Session, shortOID(cur.CheckpointSHA), shortOID(req.CheckpointSHA))}
	}
	if cur.Successor == req.Successor {
		return AdoptDecision{AdoptResume, fmt.Sprintf(
			"%s already holds the %s adoption at phase %s (attempt %d); resuming",
			req.Successor, req.Session, cur.Phase, cur.Attempt)}
	}
	if req.IncumbentLive {
		return AdoptDecision{AdoptHeld, fmt.Sprintf(
			"%s holds the %s adoption and its session is LIVE — wait for it to finish or fail, rather than racing it",
			cur.Successor, req.Session)}
	}
	if !cur.Expired(req.Now) {
		return AdoptDecision{AdoptHeld, fmt.Sprintf(
			"%s holds the %s adoption for another %ds (TTL %ds); a lapsed lease alone does not authorize a takeover",
			cur.Successor, req.Session, cur.ExpiresAt()-req.Now, cur.ttl())}
	}
	return AdoptDecision{AdoptTakeover, fmt.Sprintf(
		"%s held the %s adoption at phase %s, holds no live lease, and its claim lapsed at %d — %s takes over on attempt %d",
		cur.Successor, req.Session, cur.Phase, cur.ExpiresAt(), req.Successor, cur.Attempt+1)}
}

// ApplyAdopt builds the receipt a granted verdict should be swapped to. It never mutates
// cur. The second return is false for a refusal, so a caller cannot accidentally write a
// receipt for a decision that did not grant one.
//
// A RESUME preserves the phase, the target, and the original adoption time: the successor
// is continuing its own work, and forgetting where it materialized would orphan those
// bytes. A TAKEOVER resets the phase to ADOPTED and drops the target: the displaced
// successor's materialization is its own, possibly half-written, and the new owner must
// re-materialize from the checkpoint rather than trust it. Both increment Attempt, so the
// history is continuous across the handoff.
func ApplyAdopt(cur *Receipt, req AdoptRequest, v AdoptVerdict) (Receipt, bool) {
	if !v.Granted() {
		return Receipt{}, false
	}
	next := Receipt{
		Session:       req.Session,
		CheckpointRef: req.CheckpointRef,
		CheckpointSHA: req.CheckpointSHA,
		DeltaDigest:   req.DeltaDigest,
		Successor:     req.Successor,
		SuccessorHost: req.SuccessorHost,
		Phase:         PhaseAdopted,
		AdoptedAt:     req.Now,
		RenewedAt:     req.Now,
		TTLSeconds:    ttlOr(req.TTLSeconds),
		Attempt:       1,
	}
	switch v {
	case AdoptResume:
		next.Phase = cur.Phase
		next.Target = cur.Target
		next.AdoptedAt = cur.AdoptedAt
		next.LandedSHA = cur.LandedSHA
		next.Attempt = cur.Attempt + 1
		next.Audit = appendAudit(cur.Audit, AuditEvent{
			At: req.Now, Event: EventResumed, Actor: req.Successor,
			Detail: "resumed at phase " + string(cur.Phase),
		})
	case AdoptTakeover:
		next.Attempt = cur.Attempt + 1
		next.Audit = appendAudit(cur.Audit, AuditEvent{
			At: req.Now, Event: EventTakeover, Actor: req.Successor, From: cur.Successor,
			Detail: fmt.Sprintf("claim lapsed at %d with no live lease; phase %s discarded", cur.ExpiresAt(), cur.Phase),
		})
	default:
		next.Audit = appendAudit(nil, AuditEvent{At: req.Now, Event: EventAdopted, Actor: req.Successor})
	}
	return next, true
}

// MarkPhase advances a held receipt without touching its ownership, renewing the claim as
// it goes — progress IS the renewal, so a successor that is working never has its claim
// taken from under it and one that has stopped working always eventually loses it.
func MarkPhase(cur Receipt, phase Phase, now int64, event, detail string) Receipt {
	next := cur
	next.Phase = phase
	next.RenewedAt = now
	next.Audit = appendAudit(cur.Audit, AuditEvent{At: now, Event: event, Actor: cur.Successor, Detail: detail})
	return next
}

// ExpiresAt is the unix second at which this claim stops being self-evidently live.
func (r Receipt) ExpiresAt() int64 {
	at := r.RenewedAt
	if at <= 0 {
		at = r.AdoptedAt
	}
	return at + r.ttl()
}

// Expired reports a claim whose TTL has lapsed. Expiry ALONE never authorizes a takeover;
// DecideAdopt also requires the incumbent to hold no live lease.
func (r Receipt) Expired(now int64) bool { return now >= r.ExpiresAt() }

func (r Receipt) ttl() int64 {
	if r.TTLSeconds > 0 {
		return r.TTLSeconds
	}
	return DefaultTTLSeconds
}

// Validate rejects a receipt that cannot be reasoned about. Called on both encode and
// decode so a malformed record can neither be written nor silently honoured.
func (r Receipt) Validate() error {
	switch {
	case r.Session == "":
		return errors.New("receipt has no session")
	case r.Successor == "":
		return errors.New("receipt has no successor")
	case r.CheckpointSHA == "":
		return errors.New("receipt has no checkpoint SHA")
	case r.Phase != PhaseAdopted && r.Phase != PhaseMaterialized && r.Phase != PhaseLanded:
		return fmt.Errorf("receipt has unknown phase %q", r.Phase)
	}
	return nil
}

// EncodeReceipt renders a receipt as the body of the object the claim ref points at: a
// human-readable header, then one marker line carrying the JSON. Same two-audience shape
// as wipref.EncodeStamp — `git cat-file -p` is readable, and the parser needs one line.
func EncodeReceipt(r Receipt) (string, error) {
	if err := r.Validate(); err != nil {
		return "", err
	}
	blob, err := json.Marshal(r)
	if err != nil {
		return "", fmt.Errorf("encode adoption receipt: %w", err)
	}
	var b strings.Builder
	fmt.Fprintf(&b, "fak wip adoption receipt for %s\n", r.Session)
	fmt.Fprintf(&b, "successor: %s  phase: %s  attempt: %d\n", r.Successor, r.Phase, r.Attempt)
	fmt.Fprintf(&b, "checkpoint: %s @ %s\n\n", r.CheckpointRef, r.CheckpointSHA)
	b.WriteString(receiptMarker)
	b.Write(blob)
	b.WriteString("\n")
	return b.String(), nil
}

// DecodeReceipt recovers a receipt from an object body. A body with no marker line, or one
// whose JSON or contents do not validate, is an ERROR rather than a zero receipt: a
// caller that cannot read the incumbent claim must refuse, not conclude "unclaimed".
func DecodeReceipt(body string) (Receipt, error) {
	for _, ln := range strings.Split(body, "\n") {
		s := strings.TrimSpace(ln)
		if !strings.HasPrefix(s, receiptMarker) {
			continue
		}
		var r Receipt
		if err := json.Unmarshal([]byte(strings.TrimPrefix(s, receiptMarker)), &r); err != nil {
			return Receipt{}, fmt.Errorf("decode adoption receipt: %w", err)
		}
		if err := r.Validate(); err != nil {
			return Receipt{}, err
		}
		return r, nil
	}
	return Receipt{}, errors.New("no adoption receipt marker in object body")
}

func appendAudit(prior []AuditEvent, ev AuditEvent) []AuditEvent {
	out := make([]AuditEvent, 0, len(prior)+1)
	out = append(out, prior...)
	out = append(out, ev)
	if len(out) > auditMax {
		out = out[len(out)-auditMax:]
	}
	return out
}

func ttlOr(ttl int64) int64 {
	if ttl > 0 {
		return ttl
	}
	return DefaultTTLSeconds
}

func shortOID(oid string) string {
	if len(oid) > 12 {
		return oid[:12]
	}
	return oid
}

func landedAs(sha string) string {
	if sha == "" {
		return ""
	}
	return " as " + shortOID(sha)
}
