package loopunblock

import (
	"reflect"
	"testing"
)

// enterable / blocked helpers keep the table rows short.
func cand(id string, rank int, a AdmitStatus) Candidate {
	return Candidate{ID: id, Rank: rank, Admit: a}
}

func TestDecide_TableDrivenActions(t *testing.T) {
	tests := []struct {
		name         string
		cands        []Candidate
		pol          Policy
		wantAction   Action
		wantAuto     bool
		wantEnter    string
		wantBypassed []string
		wantCause    BlockCause
	}{
		{
			name:       "empty worklist stands down",
			cands:      nil,
			wantAction: ActionStandDown,
			wantAuto:   true,
			wantEnter:  "",
			wantCause:  CauseNone,
		},
		{
			name:       "admittable head is entered",
			cands:      []Candidate{cand("a", 1, Admittable())},
			wantAction: ActionEnter,
			wantAuto:   true,
			wantEnter:  "a",
			wantCause:  CauseNone,
		},
		{
			name:       "stale lease head is cleared in place, preserving worst-first",
			cands:      []Candidate{cand("a", 1, Blocked(CauseLeaseStale, "pid 1234 gone")), cand("b", 2, Admittable())},
			wantAction: ActionClearThenEnter,
			wantAuto:   true,
			wantEnter:  "a", // the head, NOT the admittable b — a stale lease is cleared, not bypassed
			wantCause:  CauseLeaseStale,
		},
		{
			name:         "live-lease head bypasses to the next admittable candidate",
			cands:        []Candidate{cand("a", 1, Blocked(CauseLeaseLive, "peer holds it")), cand("b", 2, Admittable())},
			wantAction:   ActionBypass,
			wantAuto:     true,
			wantEnter:    "b",
			wantBypassed: []string{"a"},
			wantCause:    CauseLeaseLive,
		},
		{
			name: "bypass skips multiple blocked heads to reach the admittable one",
			cands: []Candidate{
				cand("a", 1, Blocked(CauseCapped, "seat capped")),
				cand("b", 2, Blocked(CauseLeaseLive, "peer")),
				cand("c", 3, Admittable()),
			},
			wantAction:   ActionBypass,
			wantAuto:     true,
			wantEnter:    "c",
			wantBypassed: []string{"a", "b"},
			wantCause:    CauseCapped,
		},
		{
			name:       "transient head with nothing admittable waits",
			cands:      []Candidate{cand("a", 1, Blocked(CauseCapped, "all seats capped")), cand("b", 2, Blocked(CauseLeaseLive, "peer"))},
			wantAction: ActionWait,
			wantAuto:   true,
			wantEnter:  "",
			wantCause:  CauseCapped,
		},
		{
			name:       "non-transient head with nothing admittable escalates",
			cands:      []Candidate{cand("a", 1, Blocked(CauseUnmeasured, "status unreadable"))},
			wantAction: ActionEscalate,
			wantAuto:   false, // escalate is never auto — only an operator can move it
			wantEnter:  "",
			wantCause:  CauseUnmeasured,
		},
		{
			name:       "unknown head with nothing admittable escalates (fail conservative)",
			cands:      []Candidate{cand("a", 1, Blocked(CauseUnknown, "??"))},
			wantAction: ActionEscalate,
			wantAuto:   false,
			wantCause:  CauseUnknown,
		},
		{
			name:       "budget-held head with nothing admittable escalates, not waits",
			cands:      []Candidate{cand("a", 1, Blocked(CauseBudgetHeld, "unbudgeted dimension"))},
			wantAction: ActionEscalate,
			wantAuto:   false,
			wantCause:  CauseBudgetHeld,
		},
		{
			name:         "budget-held head bypasses when a budgeted member is admittable",
			cands:        []Candidate{cand("a", 1, Blocked(CauseBudgetHeld, "held")), cand("b", 2, Admittable())},
			wantAction:   ActionBypass,
			wantAuto:     true,
			wantEnter:    "b",
			wantBypassed: []string{"a"},
			wantCause:    CauseBudgetHeld,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Decide(tt.cands, tt.pol)
			if got.Action != tt.wantAction {
				t.Errorf("Action = %q, want %q (reason: %s)", got.Action, tt.wantAction, got.Reason)
			}
			if got.Auto != tt.wantAuto {
				t.Errorf("Auto = %v, want %v", got.Auto, tt.wantAuto)
			}
			if got.Enter != tt.wantEnter {
				t.Errorf("Enter = %q, want %q", got.Enter, tt.wantEnter)
			}
			if got.HeadCause != tt.wantCause {
				t.Errorf("HeadCause = %q, want %q", got.HeadCause, tt.wantCause)
			}
			if !reflect.DeepEqual(got.Bypassed, tt.wantBypassed) {
				t.Errorf("Bypassed = %v, want %v", got.Bypassed, tt.wantBypassed)
			}
			if got.Schema != Schema {
				t.Errorf("Schema = %q, want %q", got.Schema, Schema)
			}
		})
	}
}

// A Manual policy must still classify and name the exact action, but never mark it
// Auto — the operator applies it. Escalate stays Auto=false either way.
func TestDecide_ManualPolicyIsAdvisory(t *testing.T) {
	cases := []struct {
		name   string
		cands  []Candidate
		action Action
	}{
		{"enter", []Candidate{cand("a", 1, Admittable())}, ActionEnter},
		{"clear", []Candidate{cand("a", 1, Blocked(CauseLeaseStale, ""))}, ActionClearThenEnter},
		{"bypass", []Candidate{cand("a", 1, Blocked(CauseLeaseLive, "")), cand("b", 2, Admittable())}, ActionBypass},
		{"wait", []Candidate{cand("a", 1, Blocked(CauseCapped, ""))}, ActionWait},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := Decide(c.cands, Policy{Manual: true})
			if got.Action != c.action {
				t.Fatalf("Action = %q, want %q", got.Action, c.action)
			}
			if got.Auto {
				t.Errorf("Manual policy must not mark %q Auto", got.Action)
			}
		})
	}
}

// NoBypass forbids routing around a blocked head: a transient head that would have
// bypassed instead waits; a non-transient one escalates. A stale lease is still
// cleared in place (that does not change which member runs).
func TestDecide_NoBypass(t *testing.T) {
	transient := []Candidate{cand("a", 1, Blocked(CauseLeaseLive, "peer")), cand("b", 2, Admittable())}
	if got := Decide(transient, Policy{NoBypass: true}); got.Action != ActionWait {
		t.Errorf("transient head under NoBypass: Action = %q, want wait", got.Action)
	} else if got.Enter != "" || got.Bypassed != nil {
		t.Errorf("NoBypass must not enter/skip: Enter=%q Bypassed=%v", got.Enter, got.Bypassed)
	}

	nonTransient := []Candidate{cand("a", 1, Blocked(CauseUnmeasured, "")), cand("b", 2, Admittable())}
	if got := Decide(nonTransient, Policy{NoBypass: true}); got.Action != ActionEscalate {
		t.Errorf("non-transient head under NoBypass: Action = %q, want escalate", got.Action)
	}

	// A stale lease is cleared in place even under NoBypass — order is preserved.
	stale := []Candidate{cand("a", 1, Blocked(CauseLeaseStale, "")), cand("b", 2, Admittable())}
	if got := Decide(stale, Policy{NoBypass: true}); got.Action != ActionClearThenEnter || got.Enter != "a" {
		t.Errorf("stale head under NoBypass: Action=%q Enter=%q, want clear_then_enter/a", got.Action, got.Enter)
	}
}

// Blocked() must never let a blocked candidate masquerade as admittable: a CauseNone
// passed to Blocked is coerced to CauseUnknown so the fold fails conservative.
func TestBlocked_CoercesNoneToUnknown(t *testing.T) {
	a := Blocked(CauseNone, "oops no cause")
	if a.Admittable {
		t.Fatal("Blocked() must produce a non-admittable status")
	}
	if a.Cause != CauseUnknown {
		t.Errorf("Blocked(CauseNone) coerced to %q, want %q", a.Cause, CauseUnknown)
	}
}

func TestBlockCause_TransientAndClearable(t *testing.T) {
	transient := map[BlockCause]bool{
		CauseNone: false, CauseLeaseStale: false, CauseLeaseLive: true,
		CauseCapped: true, CauseBudgetHeld: false, CauseUnmeasured: false, CauseUnknown: false,
	}
	for c, want := range transient {
		if c.Transient() != want {
			t.Errorf("%q.Transient() = %v, want %v", c, c.Transient(), want)
		}
	}
	// Only a stale lease is clearable in place.
	for c := range transient {
		wantClear := c == CauseLeaseStale
		if c.Clearable() != wantClear {
			t.Errorf("%q.Clearable() = %v, want %v", c, c.Clearable(), wantClear)
		}
	}
}

// The head reported is always the worst-first head (rank 1), even when a lower-ranked
// candidate is the one actually entered via bypass.
func TestDecide_HeadIsAlwaysRankOne(t *testing.T) {
	cands := []Candidate{
		cand("head", 1, Blocked(CauseLeaseLive, "peer")),
		cand("next", 2, Admittable()),
	}
	got := Decide(cands, Policy{})
	if got.Head != "head" || got.HeadRank != 1 {
		t.Errorf("Head=%q rank=%d, want head/1", got.Head, got.HeadRank)
	}
	if got.Enter != "next" {
		t.Errorf("Enter=%q, want next (bypass target)", got.Enter)
	}
}
