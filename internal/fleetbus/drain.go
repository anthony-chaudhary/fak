package fleetbus

import (
	"fmt"
	"time"
)

// Outcome is what an applier reports back for one directive. It is deliberately not
// (error, bool): a control plane needs to distinguish "took it", "understood it and
// said no, HERE is the closed token", and "that op means nothing to me" — and a bare
// error flattens all three into noise at the control point.
type Outcome struct {
	// Status is the closed outcome. A zero Status is treated as a refusal (see
	// normalizeOutcome): an applier that answers nothing must not read as success.
	Status AckStatus
	// Reason is the closed token when Status is not applied.
	Reason RefuseReason
	// Detail carries the underlying LOCAL refusal token and its text verbatim, so a
	// fanned op refuses exactly as it would alone.
	Detail string
	// Witness names what the applier OBSERVED change — not what it enqueued.
	Witness string
	// Affected counts the local subjects the op landed on.
	Affected int
}

// OutcomeApplied reports a directive taken, with the witness of what changed.
func OutcomeApplied(witness string, affected int) Outcome {
	return Outcome{Status: AckApplied, Witness: witness, Affected: affected}
}

// OutcomeRefused reports a directive understood and declined, under a closed token.
func OutcomeRefused(reason RefuseReason, format string, args ...any) Outcome {
	return Outcome{Status: AckRefused, Reason: reason, Detail: fmt.Sprintf(format, args...)}
}

// Applier is the instance half of the bus: the thing that knows what an op MEANS and
// owns the real local write. The bus never interprets an op (see the package doc), so
// everything domain-specific about fleet control lives behind this one method.
type Applier interface {
	Apply(d Directive) Outcome
}

// ApplierFunc adapts a plain func into an Applier.
type ApplierFunc func(Directive) Outcome

func (f ApplierFunc) Apply(d Directive) Outcome { return f(d) }

// DrainReport is one drain pass's accounting, for logs and tests. It is NOT the
// witness a control point reads — that is Fold over the acks, which is the whole
// point of a return path.
type DrainReport struct {
	// Instance is who drained.
	Instance string `json:"instance"`
	// Seen is every directive in the log; Matched is the subset addressed to this
	// instance.
	Seen    int `json:"seen"`
	Matched int `json:"matched"`
	// AlreadyDone is matched directives this instance had already claimed — the
	// at-least-once redelivery this drain correctly did nothing about.
	AlreadyDone int `json:"already_done"`
	// Applied / Refused / Expired partition the directives this pass answered.
	Applied int `json:"applied"`
	Refused int `json:"refused"`
	Expired int `json:"expired"`
	// Acks is what this pass appended.
	Acks []Ack `json:"acks,omitempty"`
	// Errors records transport failures (a lost claim, an unwritable ack). They are
	// collected rather than returned so one bad directive cannot stop the drain of
	// the rest — and they are never silent.
	Errors []string `json:"errors,omitempty"`
}

// Drain is the instance side of the bus: read the log, answer everything addressed
// to self exactly once, and leave an ack for each answer.
//
// Three rules carry the design:
//
//	A directive this instance does not match is skipped SILENTLY and is not an
//	ack. Silence is unambiguous here because the control point computes the
//	expected set from the roster, not from who happened to answer.
//
//	The claim is taken BEFORE the apply. See DirBus.ClaimApply for why at-most-once
//	plus a visible OUTSTANDING beats a possible double-apply.
//
//	An expired directive is ACKED expired and never applied. "Nobody was listening
//	in time" and "nobody ever answered" are different facts and must stay
//	distinguishable at the control point.
func Drain(b Bus, self Instance, ap Applier, now time.Time) (DrainReport, error) {
	rep := DrainReport{Instance: self.ID}
	if r := self.Validate(); r != nil {
		return rep, r
	}
	if b == nil || ap == nil {
		return rep, fmt.Errorf("fleetbus: drain needs a bus and an applier")
	}
	directives, err := b.Directives()
	if err != nil {
		return rep, err
	}
	for _, d := range directives {
		rep.Seen++
		if !d.TargetsInstance(self) {
			continue
		}
		rep.Matched++

		fresh, err := b.ClaimApply(self.ID, d.ID)
		if err != nil {
			rep.Errors = append(rep.Errors, fmt.Sprintf("%s: claim: %v", d.ID, err))
			continue
		}
		if !fresh {
			rep.AlreadyDone++
			continue
		}

		out := normalizeOutcome(d, ap, now)
		ack := Ack{
			Schema:    AckSchema,
			Directive: d.ID,
			Instance:  self.ID,
			Status:    out.Status,
			Reason:    out.Reason,
			Detail:    out.Detail,
			Witness:   out.Witness,
			Affected:  out.Affected,
			AckedUTC:  utc(now),
		}
		switch out.Status {
		case AckApplied:
			rep.Applied++
		case AckExpired:
			rep.Expired++
		default:
			rep.Refused++
		}
		if err := b.Ack(ack); err != nil {
			rep.Errors = append(rep.Errors, fmt.Sprintf("%s: ack: %v", d.ID, err))
			continue
		}
		rep.Acks = append(rep.Acks, ack)
	}
	return rep, nil
}

// normalizeOutcome resolves what this instance actually reports for d, and enforces
// the one invariant an applier must not be able to break: a non-applied answer ALWAYS
// carries a closed token. An applier that refuses without one would produce an ack
// that fails validation and is therefore never written — turning a refusal into
// silence, which is the exact phantom this package prevents. So the missing token is
// supplied here, loudly, rather than the ack being dropped.
func normalizeOutcome(d Directive, ap Applier, now time.Time) Outcome {
	if d.IsExpired(now) {
		deadline, _ := d.ExpiresAt()
		return Outcome{
			Status: AckExpired,
			Reason: Expired,
			Detail: fmt.Sprintf("directive ttl %ds lapsed at %s; not applied", d.TTLSec, utc(deadline)),
		}
	}
	out := ap.Apply(d)
	switch out.Status {
	case AckApplied:
		return out
	case AckRefused, AckExpired:
		if out.Reason == "" {
			out.Reason = ApplyRefused
			out.Detail = "applier refused without a closed token; " + out.Detail
		}
		return out
	default:
		return Outcome{
			Status: AckRefused,
			Reason: ApplyRefused,
			Detail: fmt.Sprintf("applier returned no status for op %q", d.Op),
		}
	}
}
