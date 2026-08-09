package fleetbus

import "sort"

// OpSupport is one op's roster-wide capability answer: of the instances visible
// right now, how many say they can apply it, how many say they cannot, and how many
// said nothing either way.
//
// Declared + Unsupported + Silent == Total, always. Silent is a first-class count
// rather than folded into either side because it is the honest UNKNOWN rung and the
// two neighbouring readings are both wrong: counting a silent instance as capable
// over-promises (it may be an old binary that never heard of the op), and counting it
// as incapable under-promises (an op an instance did not declare still gets a real
// attempt and a real ack). The only true statement is that it did not say.
type OpSupport struct {
	Op Op `json:"op"`
	// Declared named this op in Instance.Ops; Unsupported named it in
	// Instance.Unsupported; Silent named it in neither.
	Declared    int `json:"declared"`
	Unsupported int `json:"unsupported"`
	Silent      int `json:"silent"`
	// Total is the roster size — the denominator for all three, and the same one
	// Fold would use, so "0 of 16" here and "16 targeted" there refer to one fleet.
	Total int `json:"total"`
}

// Capability folds a roster into one row per op anybody mentioned, sorted by op.
//
// This is the pre-fan-out question. Fold answers "what happened when I sent it";
// this answers "what would happen if I did", from claims alone — which is why it is
// a separate call and not a field on Report. Nothing here routes, selects or
// suppresses: an operator who reads "0 of 16 can steer" and sends steer anyway gets
// 16 real attempts and 16 real acks, because the roster's claim is not a witness.
//
// An instance that names one op BOTH ways counts as Unsupported. Capability exists
// to stop a fleet over-claiming what it can do, so when an instance contradicts
// itself the reading that cannot manufacture false capacity wins; the contradiction
// is visible on the instance record either way.
//
// Ops nobody mentions produce no row. A roster is not a vocabulary — the op set is
// open (a tier-1 bus declares none), so enumerating "every op that exists" is not
// something this package can honestly do. An op with no row is Silent across the
// whole roster; ask for it with CapabilityFor, which says exactly that.
func Capability(roster []Instance) []OpSupport {
	rows := map[Op]*OpSupport{}
	row := func(op Op) *OpSupport {
		if r, ok := rows[op]; ok {
			return r
		}
		r := &OpSupport{Op: op, Total: len(roster), Silent: len(roster)}
		rows[op] = r
		return r
	}
	// Two passes over the roster per op is what keeps Silent correct: a row is
	// created with every instance silent, and each instance that spoke about that op
	// moves itself out of Silent — including instances announced BEFORE the row's op
	// was first seen, because Total/Silent are seeded from the whole roster, not from
	// how far the loop had walked.
	for _, inst := range roster {
		for _, op := range inst.Ops {
			row(op)
		}
		for _, op := range inst.Unsupported {
			row(op)
		}
	}
	for op, r := range rows {
		for _, inst := range roster {
			switch {
			case inst.DeclaresUnsupported(op):
				r.Unsupported++
				r.Silent--
			case inst.DeclaresOp(op):
				r.Declared++
				r.Silent--
			}
		}
	}
	out := make([]OpSupport, 0, len(rows))
	for _, r := range rows {
		out = append(out, *r)
	}
	sort.Slice(out, func(a, b int) bool { return out[a].Op < out[b].Op })
	return out
}

// CapabilityFor is Capability narrowed to one op, including the case no instance
// mentioned it at all — which returns a row of all-Silent rather than nothing, so a
// caller asking "can this fleet steer?" gets a denominator instead of a zero value it
// has to guess the meaning of.
func CapabilityFor(roster []Instance, op Op) OpSupport {
	out := OpSupport{Op: op, Total: len(roster), Silent: len(roster)}
	for _, inst := range roster {
		switch {
		case inst.DeclaresUnsupported(op):
			out.Unsupported++
			out.Silent--
		case inst.DeclaresOp(op):
			out.Declared++
			out.Silent--
		}
	}
	return out
}
