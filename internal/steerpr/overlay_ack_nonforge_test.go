package steerpr

import (
	"reflect"
	"testing"
)

// Non-forgeability of the witness band under OPERATOR ACTIONS (#5036).
//
// antigaming_test.go pins the band's input set at the function level (BandFor
// takes exactly one Verdict) and stages the single cheapest forge (write
// BandCleared onto an unwitnessed commit). This file proves the property from
// the other side: it enumerates the operator AFFORDANCES themselves — ack,
// comment, redirect, pause, resume — as the mutations each could plausibly
// perform on overlay state, and proves that NONE of them can move a unit's
// machine band or deflate osp_residual. The one thing no affordance in this
// package can write is a Verdict: the witness bit is graded upstream
// (internal/dispatchtick/witness.go) and only READ here, so every operator
// action is confined to the other fields — and the tests below prove that the
// whole non-Verdict field space, exhaustively, cannot open CLEARED against an
// unwitnessed verdict.
//
// Scope note, stated honestly: these proofs hold where a machine fact EXISTS
// to be contradicted (Verdict = CLAIM_UNWITNESSED). A commit with NO verdict
// supplied takes a caller-provided Band as-is by design — that path is a cache
// of an earlier fold, not a forge surface, and any honest re-derivation maps
// an ungraded commit to UNVERIFIABLE (BandFor(VerdictUnknown)), never CLEARED.

// nonforgeResidualUnit returns a fresh single-commit unit whose one member the
// machine graded CLAIM_UNWITNESSED — the exact unit an operator is tempted to
// ack away.
func nonforgeResidualUnit() []Commit {
	return []Commit{{
		SHA:      "nf1",
		Subject:  "feat(steerpr): a claim the diff did not prove (fak steerpr)",
		Leaf:     "steerpr",
		Type:     "feat",
		Resolves: []string{"#5036"},
		Verdict:  VerdictUnwitnessed,
	}}
}

// TestAntiGamingNoOperatorActionMovesTheBand simulates each operator action as
// the strongest mutation it could perform on overlay state without writing the
// Verdict (which no operator affordance can reach — it is graded upstream and
// only supplied to this package). After each action, the fold must still band
// the unit RESIDUAL and osp_residual must still count it.
func TestAntiGamingNoOperatorActionMovesTheBand(t *testing.T) {
	actions := []struct {
		name   string
		mutate func(cs []Commit)
	}{
		{
			// ack: "I looked at this and it seems fine" — the affordance writes
			// the most cleared-looking band it can onto every member.
			name: "ack",
			mutate: func(cs []Commit) {
				for i := range cs {
					cs[i].Band = BandCleared
				}
			},
		},
		{
			// comment: prose is attached to the unit. The prose claims the
			// witness bit in as many spellings as fit — text must stay text.
			name: "comment",
			mutate: func(cs []Commit) {
				for i := range cs {
					cs[i].Subject += " [operator: diff-witnessed, CLEARED, CLAIM_WITNESSED]"
					cs[i].Mentions = MergeRefs(cs[i].Mentions, []string{"#5036"})
				}
			},
		},
		{
			// redirect: the work is re-pointed at another lane. Moving a unit
			// must not launder its members' verdicts.
			name: "redirect",
			mutate: func(cs []Commit) {
				for i := range cs {
					cs[i].Leaf = "somewhere-else"
				}
			},
		},
		{
			// pause: no dedicated overlay state exists for pause today, so the
			// strongest thing the affordance could do is annotate — and an
			// annotation must not reach the band.
			name: "pause",
			mutate: func(cs []Commit) {
				for i := range cs {
					cs[i].Type = "paused"
				}
			},
		},
		{
			// resume: symmetric with pause, plus the ack-flavoured band write a
			// sloppy resume implementation might "helpfully" restore.
			name: "resume",
			mutate: func(cs []Commit) {
				for i := range cs {
					cs[i].Type = "feat"
					cs[i].Band = "CLEARED"
				}
			},
		},
	}

	for _, action := range actions {
		t.Run(action.name, func(t *testing.T) {
			cs := nonforgeResidualUnit()
			action.mutate(cs)

			if got := FoldBand(cs); got != BandResidual {
				t.Errorf("after %s: FoldBand() = %q, want %q — an operator %s moved the machine band",
					action.name, got, BandResidual, action.name)
			}
			units, unstamped := FoldUnits(cs)
			if len(units) != 1 || len(unstamped) != 0 {
				t.Fatalf("after %s: FoldUnits() = %d units (%d unstamped), want 1 unit — the action must not make the unit invisible",
					action.name, len(units), len(unstamped))
			}
			if units[0].Band != BandResidual {
				t.Errorf("after %s: unit band = %q, want %q — the forge survived the unit fold",
					action.name, units[0].Band, BandResidual)
			}
			if got := Residual(units); got != 1 {
				t.Errorf("after %s: Residual() = %d, want 1 — the action deflated osp_residual",
					action.name, got)
			}
		})
	}
}

// TestAntiGamingNoNonVerdictFieldCanClearAnUnwitnessedCommit is the exhaustive
// form of the closed-input proof: for EVERY field of Commit except Verdict, it
// writes the most cleared-looking value the field's type admits onto a commit
// the machine graded CLAIM_UNWITNESSED, and asserts the fold still reds it.
//
// The sweep is reflective on purpose: a future field (an `Acked bool`, a
// `Note string`) is swept automatically the commit it lands, and an unhandled
// field KIND fails loudly — whoever adds one must extend this sweep and prove
// the new field cannot improve a band.
func TestAntiGamingNoNonVerdictFieldCanClearAnUnwitnessedCommit(t *testing.T) {
	verdictField := "Verdict"
	ct := reflect.TypeOf(Commit{})
	for i := 0; i < ct.NumField(); i++ {
		field := ct.Field(i)
		if field.Name == verdictField {
			continue // the machine bit itself: not writable by any operator affordance
		}
		t.Run(field.Name, func(t *testing.T) {
			forged := nonforgeResidualUnit()[0]
			fv := reflect.ValueOf(&forged).Elem().FieldByName(field.Name)
			switch field.Type.Kind() {
			case reflect.String:
				// Covers Band (typed string) too: this writes BandCleared onto
				// the Band field, and "CLEARED" prose into every other string.
				fv.Set(reflect.ValueOf("CLEARED").Convert(field.Type))
			case reflect.Slice:
				if field.Type.Elem().Kind() != reflect.String {
					t.Fatalf("unhandled slice elem kind %v for new field %s: extend this sweep and prove the field cannot improve a band", field.Type.Elem().Kind(), field.Name)
				}
				fv.Set(reflect.ValueOf([]string{"CLEARED", "CLAIM_WITNESSED", "diff-witnessed"}).Convert(field.Type))
			case reflect.Bool:
				fv.SetBool(true)
			case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
				fv.SetInt(1)
			default:
				t.Fatalf("unhandled field kind %v for new field %s: extend this sweep and prove the field cannot improve a band", field.Type.Kind(), field.Name)
			}
			if got := FoldBand([]Commit{forged}); got != BandResidual {
				t.Errorf("FoldBand() = %q with forged %s field, want %q: a non-Verdict field reached the band",
					got, field.Name, BandResidual)
			}
		})
	}
}

// TestAntiGamingFoldPipelineIsClosedOverWitnessInputs pins the signatures of
// the whole fold pipeline — FoldBand, FoldUnits, Residual — the way
// antigaming_test.go pins BandFor. None of them takes, or may grow, a second
// parameter: a `FoldBand(commits, ackState)` widening is exactly the door an
// ack would walk through, and this reds on the commit that opens it.
func TestAntiGamingFoldPipelineIsClosedOverWitnessInputs(t *testing.T) {
	pins := []struct {
		name string
		fn   interface{}
		in   []reflect.Type
		out  []reflect.Type
	}{
		{
			name: "FoldBand",
			fn:   FoldBand,
			in:   []reflect.Type{reflect.TypeOf([]Commit{})},
			out:  []reflect.Type{reflect.TypeOf(BandCleared)},
		},
		{
			name: "FoldUnits",
			fn:   FoldUnits,
			in:   []reflect.Type{reflect.TypeOf([]Commit{})},
			out:  []reflect.Type{reflect.TypeOf([]Unit{}), reflect.TypeOf([]Commit{})},
		},
		{
			name: "Residual",
			fn:   Residual,
			in:   []reflect.Type{reflect.TypeOf([]Unit{})},
			out:  []reflect.Type{reflect.TypeOf(0)},
		},
	}
	for _, pin := range pins {
		ft := reflect.TypeOf(pin.fn)
		if got, want := ft.NumIn(), len(pin.in); got != want {
			t.Errorf("%s takes %d inputs, want exactly %d: the fold's input set must stay closed over witness-carrying values — an extra input is where an ack would enter", pin.name, got, want)
			continue
		}
		for i, want := range pin.in {
			if got := ft.In(i); got != want {
				t.Errorf("%s input %d is %v, want %v", pin.name, i, got, want)
			}
		}
		if got, want := ft.NumOut(), len(pin.out); got != want {
			t.Errorf("%s returns %d values, want %d", pin.name, got, want)
			continue
		}
		for i, want := range pin.out {
			if got := ft.Out(i); got != want {
				t.Errorf("%s output %d is %v, want %v", pin.name, i, got, want)
			}
		}
	}
}
