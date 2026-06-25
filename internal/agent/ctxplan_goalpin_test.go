package agent

import (
	"context"
	"reflect"
	"testing"
)

// Goal-as-pin-root (#845, epic #844): the active goal is the intentional GC root of the
// context heap. A RoleGoal message is pinned RESIDENT regardless of its relevance/recency
// score, so a session pursuing one goal never elides the span that goal depends on — and a
// session with NO goal set plans byte-identically to before (behavior-preserving).

// goalDecaySession is a frozen transcript whose GOAL is content-disjoint from the LATEST
// user turn: the goal is about the auth-token migration, but the current turn asks about
// the weather. The goal therefore scores LOW on relevance to the live intents, so under a
// tight budget a non-pinned goal span would be elided. The test asserts the pin keeps it
// resident anyway — the whole point of an intentional root.
//
// The goal sits at index 1, AFTER the system prompt and a first user turn, so when the goal
// is demoted to a plain user role for the load-bearing probe it does NOT become the
// firstUserPin (index 2 is the first user turn) — it competes purely on relevance, exactly
// the score-decay regime the pin must survive.
func goalDecaySession() []Message {
	return []Message{
		{Role: RoleSystem, Content: "You are a support agent. Use the tools to help."},
		{Role: RoleUser, Content: "let's get started on today's work please"},
		{Role: RoleGoal, Content: "migrate the auth token store to the new rotation runbook end to end"},
		{Role: RoleTool, Name: "WebSearch", Content: "weather sunny 22C light wind from the west today outside"},
		{Role: RoleTool, Name: "Read", Content: "forecast tomorrow rain 14C gusty from the north region"},
		{Role: RoleUser, Content: "what is the weather and the wind speed in detail today"},
	}
}

// goalSpanIdx is the index of the RoleGoal message in goalDecaySession (its span id is
// span:<goalSpanIdx>, since the MemStore assigns ids by insertion order == message index).
const goalSpanIdx = 2

// TestGoalPinNoGoalIsByteIdentical is the behavior-preserving witness: a session with no
// RoleGoal message produces the EXACT SAME pin set and plan as the same session would have
// before the goal-pin edge existed. We prove it by stripping the goal and asserting the pin
// set has no goal id and equals the structural-pins-only list.
func TestGoalPinNoGoalIsByteIdentical(t *testing.T) {
	// recordedSession() (from ctxplan_seam_test.go) carries NO RoleGoal message:
	// [system, user, tool, tool, tool] -> system=0, firstUser=1, no second user turn.
	session := recordedSession()
	_, pins := messagesToStore(session)

	// The pre-edge contract: pins are exactly {system(0), firstUser(1)} — no goal id, in the
	// same order. A goal pin leaking in (or the structural pins reordering) fails here.
	want := []string{ctxplanSpanID(0), ctxplanSpanID(1)}
	if !reflect.DeepEqual(pins, want) {
		t.Fatalf("no-goal pin set drifted: got %v, want %v (must be byte-identical to pre-edge)", pins, want)
	}
}

// TestGoalPinChargedFirst asserts the goal root is the FIRST pin — charged against the
// resident budget ahead of the structural pins, as an intentional root must be.
func TestGoalPinChargedFirst(t *testing.T) {
	session := goalDecaySession()
	_, pins := messagesToStore(session)
	if len(pins) == 0 {
		t.Fatal("expected a non-empty pin set")
	}
	if pins[0] != ctxplanSpanID(goalSpanIdx) {
		t.Fatalf("the goal root must be pinned FIRST: got pins[0]=%q, want %q", pins[0], ctxplanSpanID(goalSpanIdx))
	}
}

// TestGoalPinResidentUnderScoreDecay is the headline: even when the goal is content-disjoint
// from the live turn (so it scores LOW and a tight budget would elide a non-pinned span), the
// goal span stays RESIDENT and is marked Pinned. This is the property that makes the goal a
// real GC root: its retention does not depend on its relevance score decaying.
func TestGoalPinResidentUnderScoreDecay(t *testing.T) {
	session := goalDecaySession()
	goalID := ctxplanSpanID(goalSpanIdx)

	// goalDecayBudget is tight enough that, WITHOUT the pin, the low-relevance goal span loses
	// the knapsack to the spans the current turn is actually about. The load-bearing probe
	// below proves that premise, so this test cannot pass vacuously.
	const goalDecayBudget = 44

	sp := NewSessionPlanner(goalDecayBudget)
	plan := sp.PlanTurn(session)

	found := false
	pinned := false
	for _, s := range plan.Selected {
		if s.ID == goalID {
			found = true
			pinned = s.Pinned
			break
		}
	}
	if !found {
		// The negative case is the bug this edge fixes: the goal was elided by score decay.
		t.Fatalf("the goal root was not resident — an intentional root must never be elided by score decay; selected=%+v", plan.Selected)
	}
	if !pinned {
		t.Fatalf("the goal span is resident but not marked Pinned; it must be a charged pin, not an incidental selection")
	}

	// Load-bearing premise: demote the goal to a plain (unpinned) user role at the SAME
	// position and budget, and confirm it is now ELIDED. If the unpinned span stays resident
	// the budget is too loose and the assertion above is vacuous — this guards against that.
	demoted := goalDecaySession()
	demoted[goalSpanIdx].Role = RoleUser
	probe := NewSessionPlanner(goalDecayBudget)
	pplan := probe.PlanTurn(demoted)
	for _, s := range pplan.Selected {
		if s.ID == goalID {
			t.Fatalf("premise broken: the UNPINNED goal-position span stayed resident at budget %d — the pin test is vacuous; tighten the budget. selected=%+v", goalDecayBudget, pplan.Selected)
		}
	}
}

// TestGoalPinStatefulMatchesStateless ties the persistent SessionPlanner to the stateless
// CtxViewPlanner for a goal-bearing session in the fits-in-window regime: both must pin the
// SAME goal span and render the SAME history, so the two seams agree on the new root.
func TestGoalPinStatefulMatchesStateless(t *testing.T) {
	ctx := context.Background()
	session := goalDecaySession()
	const budget = 256 // wide enough that everything fits -> the two paths must be byte-identical

	stateless := &CtxViewPlanner{Enabled: true, Budget: budget}
	out, err := stateless.RenderTurn(ctx, session)
	if err != nil {
		t.Fatal(err)
	}

	stateful := NewSessionPlanner(budget)
	got := stateful.RenderTurn(ctx, session)

	if !reflect.DeepEqual(got, out) {
		t.Fatalf("stateful render != stateless render for a goal-bearing session:\n stateful=%v\n stateless=%v", got, out)
	}

	// And both pin sets must name the goal span first.
	_, pins := messagesToStore(session)
	if len(pins) == 0 || pins[0] != ctxplanSpanID(goalSpanIdx) {
		t.Fatalf("stateless pin set must charge the goal first; got %v", pins)
	}
	if gp := stateful.pins(); len(gp) == 0 || gp[0] != ctxplanSpanID(goalSpanIdx) {
		t.Fatalf("stateful pin set must charge the goal first; got %v", gp)
	}
}
