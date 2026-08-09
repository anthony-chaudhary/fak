package wiprecon

import (
	"reflect"
	"strings"
	"testing"
)

// TestDecideVocabulary walks every branch of the decision rule.
func TestDecideVocabulary(t *testing.T) {
	cases := []struct {
		name string
		c    Candidate
		want Action
	}{
		{"live owner is skipped regardless of facts", Candidate{Owner: OwnerLive, Landed: true, Applies: true}, ActSkip},
		{"crashed + landed -> discard witnessed", Candidate{Owner: OwnerCrashed, Landed: true}, ActDiscardWitnessed},
		{"crashed + landed wins over apply state", Candidate{Owner: OwnerCrashed, Landed: true, Applies: false}, ActDiscardWitnessed},
		{"crashed + unlanded + applies -> reclaim", Candidate{Owner: OwnerCrashed, Landed: false, Applies: true}, ActReclaim},
		{"crashed + unlanded + !applies -> quarantine", Candidate{Owner: OwnerCrashed, Landed: false, Applies: false}, ActQuarantine},
		{"unknown owner falls closed to quarantine", Candidate{Owner: Owner("BOGUS"), Landed: true, Applies: true}, ActQuarantine},
	}
	for _, tc := range cases {
		if got := Decide(tc.c).Action; got != tc.want {
			t.Errorf("%s: want %s, got %s", tc.name, tc.want, got)
		}
	}
}

// TestDiscardRequiresWitness is the load-bearing safety invariant: across the entire
// candidate space, DISCARD_WITNESSED is emitted ONLY when the delta actually landed.
// Nothing is ever dropped without a git witness.
func TestDiscardRequiresWitness(t *testing.T) {
	owners := []Owner{OwnerLive, OwnerCrashed, Owner("UNKNOWN")}
	for _, o := range owners {
		for _, landed := range []bool{false, true} {
			for _, applies := range []bool{false, true} {
				d := Decide(Candidate{Session: "s", Owner: o, Landed: landed, Applies: applies})
				if d.Action == ActDiscardWitnessed && !landed {
					t.Fatalf("SAFETY VIOLATION: discarded without witness for owner=%s landed=%v applies=%v", o, landed, applies)
				}
				// And a live owner is never acted on destructively.
				if o == OwnerLive && d.Action != ActSkip {
					t.Fatalf("live owner not skipped: owner=%s -> %s", o, d.Action)
				}
			}
		}
	}
}

// TestReconcileTotalAndDeterministic proves one decision per candidate, stable sort by
// session, and independence from input order.
func TestReconcileTotalAndDeterministic(t *testing.T) {
	cands := []Candidate{
		{Session: "zeta", Owner: OwnerCrashed, Landed: true},
		{Session: "alpha", Owner: OwnerCrashed, Applies: true},
		{Session: "mid", Owner: OwnerLive},
	}
	got := Reconcile(cands)
	if len(got) != len(cands) {
		t.Fatalf("totality broken: %d decisions for %d candidates", len(got), len(cands))
	}
	wantOrder := []string{"alpha", "mid", "zeta"}
	for i, w := range wantOrder {
		if got[i].Session != w {
			t.Errorf("sort[%d]: want %s, got %s", i, w, got[i].Session)
		}
	}
	// Reversing the input must not change the output.
	rev := []Candidate{cands[2], cands[1], cands[0]}
	if got2 := Reconcile(rev); !reflect.DeepEqual(got2, got) {
		t.Errorf("non-deterministic across input order:\n a=%+v\n b=%+v", got, got2)
	}
	// Spot-check the actions carried through the sort.
	byS := map[string]Action{}
	for _, d := range got {
		byS[d.Session] = d.Action
	}
	if byS["zeta"] != ActDiscardWitnessed || byS["alpha"] != ActReclaim || byS["mid"] != ActSkip {
		t.Errorf("actions wrong after fold: %+v", byS)
	}
}

// TestReconcileEmpty: no candidates -> empty (non-nil) result.
func TestReconcileEmpty(t *testing.T) {
	if got := Reconcile(nil); got == nil || len(got) != 0 {
		t.Errorf("want empty non-nil, got %#v", got)
	}
}

func TestDecideDivergedRefusesReclaim(t *testing.T) {
	got := Decide(Candidate{Session: "dead", Owner: OwnerCrashed, Applies: true, DivergedPaths: 1})
	if got.Action != ActQuarantine {
		t.Fatalf("action=%q, want %q", got.Action, ActQuarantine)
	}
	if !strings.Contains(got.Reason, "DIVERGED") || !strings.Contains(got.Reason, "git diff HEAD...") {
		t.Fatalf("reason=%q, want three-way-diff guidance", got.Reason)
	}
}
