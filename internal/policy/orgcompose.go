// orgcompose.go is the ASSEMBLY half of the org precedence fold (#5322, W4 of
// epic #5315). orgprecedence.go states the lattice and attributes a floor that
// already assembled; this file is the thing that does the assembling.
//
// The split matters. FoldOrgProvenance is deliberately a reporter — hand it three
// snapshots and it says who moved what, and it cannot lie about a floor it did
// not build. But something has to BUILD those three snapshots under the lattice,
// and if that something lived at the call site then every call site would carry
// its own copy of "may central touch this knob?" — which is how two boxes end up
// enforcing two different policies from the same manifest.
//
// ComposeOrgFloor is that one place. It applies each channel in authority order
// and CLAMPS rather than refuses:
//
//	compiled-in  the shipped FROZEN floor. Nothing below may move a knob the
//	             registry marks FROZEN, so any such field is restored to this.
//	central      the verified org manifest. It may RATCHET the floor fleet-wide
//	             and GATED_WIDEN a knob — up to, never past, the compiled cap.
//	operator     the local overlays. Under a central authority they may tighten
//	             further but may not climb back past what central granted.
//
// Clamping, not wholesale refusal, is the deliberate choice. The R3 note's two
// load-bearing answers are both phrased as clamps ("operator asks 150 over a
// central grant of 100; it clamps to 100"), and the alternative is worse in
// practice: throwing away an entire operator overlay because one field in it
// reached too far would take a box's twenty legitimate local denies offline over
// a single over-eager allow. A clamp keeps every knob the channel was entitled to
// set and rolls back only the ones it was not — and reports each rollback, so
// nothing is absorbed silently.
//
// The un-enrolled box is the invariant this file must not break. With no central
// authority (Central == nil) the operator's widening is ORDINARY LOCAL
// CONFIGURATION — the allow overlay is the whole point of `fak guard allow` — so
// no clamp applies and the assembled floor is byte-for-byte what it is today.
// Central authority is what turns an operator widening into a violation, which is
// exactly why CentralApplied is tracked rather than inferred from whether the two
// snapshots differ.
package policy

import (
	"reflect"

	"github.com/anthony-chaudhary/fak/internal/adjudicator"
)

// OrgComposition is one assembled floor plus everything the assembly absorbed.
//
// Floor is the policy that RUNS. The other fields exist so no absorption is
// silent: CentralRefused and OperatorClamped are the two rollbacks the lattice
// performed, and Fold is the per-knob attribution of the result.
type OrgComposition struct {
	// Floor is the assembled policy — the value of this whole file.
	Floor adjudicator.Policy
	// Stages are the three snapshots Floor passed through, suitable for handing
	// straight to FoldOrgProvenance (and already folded into Fold below).
	Stages OrgStages
	// Fold attributes each knob of the assembled floor to a channel.
	Fold OrgFold
	// CentralRefused is every field the central manifest moved that it has no
	// authority over — a FROZEN knob. Restored to the compiled-in value.
	//
	// This is not a harmless no-op to swallow. A central plane pushing a manifest
	// that reaches at the compiled floor is either misconfigured or probing the
	// one surface that is supposed to be unreachable, and an auditor has to be
	// able to see that it happened.
	CentralRefused []AmendmentChange
	// OperatorClamped is every field the local overlay widened past the central
	// grant, restored to central's value. Always empty without a central
	// authority — see the package comment.
	OperatorClamped []AmendmentChange
}

// CentralAuthority reports whether a verified central manifest was in force for
// this assembly. It is the difference between "the org allows exactly this" and
// "nobody has told this box anything", which render identically in the floor and
// mean opposite things.
func (c OrgComposition) CentralAuthority() bool { return c.Stages.CentralApplied }

// Clamped reports whether the lattice rolled anything back. A caller that wants
// one cheap "is there something to tell the operator about?" test uses this
// rather than checking both slices.
func (c OrgComposition) Clamped() bool {
	return len(c.CentralRefused) > 0 || len(c.OperatorClamped) > 0
}

// ComposeOrgFloor assembles the running floor from the compiled-in floor, an
// optional verified central manifest, and an optional operator overlay.
//
// Both channel arguments are POINTERS, and nil is meaningful rather than
// convenient: nil means "this channel does not exist on this box", which is not
// the same as "this channel exists and proposed the compiled floor unchanged".
// The two produce identical snapshots and differ in whether the operator may
// widen — collapsing them is precisely the bug that would make every un-enrolled
// box start refusing its own allow overlay.
func ComposeOrgFloor(compiledIn adjudicator.Policy, central, operator *adjudicator.Policy) OrgComposition {
	out := OrgComposition{Stages: OrgStages{CompiledIn: compiledIn, Central: compiledIn}}

	// Stage 1 — central, capped by the compiled-in FROZEN floor.
	//
	// DiffAmendment routes a change to a FROZEN field, or to a field no registry
	// entry classifies, into the Frozen bucket. Both are refusals here for the
	// same fail-closed reason: a knob whose amendment class this build does not
	// know about must not be movable by a document fetched off the network.
	if central != nil {
		delta := DiffAmendment(compiledIn, *central)
		out.CentralRefused = delta.Frozen
		out.Stages.Central = restoreFields(*central, compiledIn, changedFields(delta.Frozen))
		out.Stages.CentralApplied = true
	}

	// Stage 2 — operator, capped by whatever central left standing.
	out.Stages.Operator = out.Stages.Central
	if operator != nil {
		delta := DiffAmendment(out.Stages.Central, *operator)
		// A FROZEN reach is refused from the operator for the same reason it is
		// refused from central: the compiled floor is under both of them.
		rollback := changedFields(delta.Frozen)
		clamped := delta.Frozen
		if out.Stages.CentralApplied {
			// THE central-control rule. Only under a central authority: without
			// one this is the ordinary allow overlay and clamping it would break
			// every box that has ever run `fak guard allow`.
			for _, c := range delta.Widen {
				rollback[c.Field] = true
			}
			clamped = append(clamped, delta.Widen...)
		}
		out.OperatorClamped = clamped
		out.Stages.Operator = restoreFields(*operator, out.Stages.Central, rollback)
	}

	out.Floor = out.Stages.Operator
	out.Fold = FoldOrgProvenance(out.Stages)
	return out
}

// changedFields is the set of struct field names a change list touches. A change
// carrying no field name is skipped rather than rolling back a field called "",
// which would match nothing but reads as if it might.
func changedFields(changes []AmendmentChange) map[string]bool {
	out := make(map[string]bool, len(changes))
	for _, c := range changes {
		if c.Field != "" {
			out[c.Field] = true
		}
	}
	return out
}

// restoreFields returns a copy of next with every named field reset to base's
// value — the clamp, expressed generically.
//
// It reflects over the struct rather than switching on known field names on
// purpose, and the reason is the same one that put the reflection backstop in
// residualAmendmentChanges: a knob added to adjudicator.Policy tomorrow must be
// clampable WITHOUT anyone remembering to come here. A hand-written switch would
// silently let the new field through the clamp, which is a fail-open in the one
// path built to fail closed.
func restoreFields(next, base adjudicator.Policy, fields map[string]bool) adjudicator.Policy {
	if len(fields) == 0 {
		return next
	}
	out := next
	ov := reflect.ValueOf(&out).Elem()
	bv := reflect.ValueOf(base)
	t := ov.Type()
	for i := 0; i < t.NumField(); i++ {
		f := t.Field(i)
		if !f.IsExported() || !fields[f.Name] {
			continue
		}
		ov.Field(i).Set(bv.Field(i))
	}
	return out
}
