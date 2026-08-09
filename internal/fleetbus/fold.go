package fleetbus

import (
	"sort"
	"time"
)

// RowStatus is the closed per-instance verdict a folded report carries. It is the
// ack statuses PLUS the one an ack cannot express, because no ack exists.
type RowStatus string

const (
	RowApplied RowStatus = "applied"
	RowRefused RowStatus = "refused"
	RowExpired RowStatus = "expired"
	// RowOutstanding is an addressed instance that has not answered. It is the whole
	// reason the roster is a first-class record: without a denominator this row
	// cannot exist, and a report that cannot say "one instance never answered" is
	// indistinguishable from one where everybody did.
	RowOutstanding RowStatus = "outstanding"
)

// Row is one instance's line of a folded report.
type Row struct {
	Instance string    `json:"instance"`
	Machine  string    `json:"machine,omitempty"`
	Role     string    `json:"role,omitempty"`
	Status   RowStatus `json:"status"`
	// Reason / Detail carry the closed refusal token and the local one verbatim.
	Reason RefuseReason `json:"reason,omitempty"`
	Detail string       `json:"detail,omitempty"`
	// Witness is what the instance observed change.
	Witness  string `json:"witness,omitempty"`
	Affected int    `json:"affected,omitempty"`
	AckedUTC string `json:"acked_utc,omitempty"`
	// InRoster is false for an instance that answered but is no longer fresh in the
	// roster (it answered and then died, or its presence lapsed). Its ack still
	// counts: an ack is evidence of what happened, and evidence does not stop being
	// true when the witness goes offline.
	InRoster bool `json:"in_roster"`
}

// Report is what a control point actually reads: the fold of one directive against
// the roster that was addressed and the acks that came back.
type Report struct {
	Directive string `json:"directive"`
	Op        Op     `json:"op"`
	Issuer    string `json:"issuer"`
	IssuedUTC string `json:"issued_utc"`
	Selector  string `json:"selector"`
	// Targeted is the denominator: the instances addressed at PUBLISH time
	// (Directive.Targets), plus any the selector addresses in the roster now, plus
	// any that acked (see Row.InRoster). An addressed instance that dies without
	// answering stays in this count as OUTSTANDING — it does not leave.
	Targeted int `json:"targeted"`
	// Applied / Refused / Expired / Outstanding partition Targeted exactly.
	Applied     int `json:"applied"`
	Refused     int `json:"refused"`
	Expired     int `json:"expired"`
	Outstanding int `json:"outstanding"`
	// AffectedTotal sums what actually changed across the fleet — sessions steered,
	// sessions paused. The number an operator came for.
	AffectedTotal int `json:"affected_total"`
	// Complete is the honest "everyone answered" bit: every addressed instance
	// answered SOMETHING. It says nothing about whether they all said yes — read
	// Applied for that. A directive nobody was addressed to is never Complete.
	Complete bool `json:"complete"`
	// DirectiveExpired reports whether the directive's own TTL has lapsed at the
	// fold instant. Once true, Outstanding will never fall again: those instances
	// are not slow, they are permanently unaccounted for.
	DirectiveExpired bool `json:"directive_expired"`
	// Rows is one line per targeted instance, sorted by instance id.
	Rows []Row `json:"rows"`
}

// Fold resolves one directive's outcome from the roster and the acks. It is pure:
// now is injected, nothing is read or written, and the same inputs always produce
// the same report.
func Fold(d Directive, roster []Instance, acks []Ack, now time.Time) Report {
	rep := Report{
		Directive:        d.ID,
		Op:               d.Op,
		Issuer:           d.Issuer,
		IssuedUTC:        d.IssuedUTC,
		Selector:         d.Selector.String(),
		DirectiveExpired: d.IsExpired(now),
	}

	// The denominator FLOOR: whoever the selector actually addressed at publish time,
	// recorded on the directive itself. This is seeded FIRST and never removed.
	//
	// Deriving the denominator from the roster alone is the subtle way this report
	// learns to lie. The roster is a liveness claim with an expiry, so an instance
	// that was addressed and then died stops announcing and ages out of it — and a
	// roster-only denominator would then drop its row entirely rather than leave it
	// OUTSTANDING, making Outstanding==0, Complete==true, and Applied==Targeted for a
	// directive that instance never received. The asymmetry to keep in view: an
	// instance that acked and THEN left is deliberately kept (its ack is evidence),
	// so erasing the one that left while silent is exactly backwards.
	//
	// A directive published before Targets existed carries none and folds back to the
	// roster-only denominator — the old behaviour, for old records only.
	rows := map[string]Row{}
	for _, id := range d.Targets {
		rows[id] = Row{Instance: id, Status: RowOutstanding, InRoster: false}
	}

	// ...plus every addressed instance in the roster NOW. An instance that announced
	// after the publish is a real new target (a fleet grows mid-fan-out), and one that
	// is still present gets its machine/role and InRoster filled in here.
	for _, inst := range roster {
		if !d.Selector.MatchesInstance(inst) {
			continue
		}
		row := rows[inst.ID]
		row.Instance, row.Machine, row.Role, row.InRoster = inst.ID, inst.Machine, inst.Role, true
		if row.Status == "" {
			row.Status = RowOutstanding
		}
		rows[inst.ID] = row
	}

	// ...plus anyone who actually answered. An ack from an instance that has since
	// dropped out of the roster is still evidence and must not be discarded, or a
	// fleet that churns would report phantom outstanding work forever.
	for _, a := range acks {
		if a.Directive != d.ID || a.Validate() != nil {
			continue
		}
		row, known := rows[a.Instance]
		if !known {
			row = Row{Instance: a.Instance, InRoster: false}
		}
		// Later acks win. An instance appends once per directive (the claim
		// guarantees it), so this only matters for a hand-edited or replayed log —
		// where taking the newest is the least surprising rule.
		if row.AckedUTC != "" && a.AckedUTC < row.AckedUTC {
			continue
		}
		row.Status = rowStatus(a.Status)
		row.Reason, row.Detail = a.Reason, a.Detail
		row.Witness, row.Affected, row.AckedUTC = a.Witness, a.Affected, a.AckedUTC
		rows[a.Instance] = row
	}

	for _, row := range rows {
		rep.Rows = append(rep.Rows, row)
	}
	sort.Slice(rep.Rows, func(i, j int) bool { return rep.Rows[i].Instance < rep.Rows[j].Instance })

	for _, row := range rep.Rows {
		rep.Targeted++
		rep.AffectedTotal += row.Affected
		switch row.Status {
		case RowApplied:
			rep.Applied++
		case RowRefused:
			rep.Refused++
		case RowExpired:
			rep.Expired++
		default:
			rep.Outstanding++
		}
	}
	rep.Complete = rep.Targeted > 0 && rep.Outstanding == 0
	return rep
}

func rowStatus(s AckStatus) RowStatus {
	switch s {
	case AckApplied:
		return RowApplied
	case AckExpired:
		return RowExpired
	case AckRefused:
		return RowRefused
	default:
		// An ack that reached here passed Validate, so its status is one of the
		// three. Anything else is a future schema leaking through a hand-edited log:
		// count it as unanswered rather than as success.
		return RowOutstanding
	}
}

// PublishTargets resolves how many live instances a selector addresses, which is what
// makes FLEETBUS_NO_TARGET checkable at the edge: publishing into an empty fleet is
// the accepted-but-never-applied phantom in its purest form, so it is refused rather
// than accepted and left to time out.
func PublishTargets(sel Selector, roster []Instance) []Instance {
	var out []Instance
	for _, inst := range roster {
		if sel.MatchesInstance(inst) {
			out = append(out, inst)
		}
	}
	return out
}
