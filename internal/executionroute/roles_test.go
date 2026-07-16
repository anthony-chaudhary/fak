package executionroute

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/harnessprofile"
	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// multiRolePlan is the multi-role fixture named in the Witness: a cheap scout, a
// bounded worker, a stronger judge, and a driving primary — each with its OWN
// model-routing subject and its OWN independent budget. Over DefaultManifest the
// scout routes to the small model, the judge to the large model, and the worker
// to the guard ensemble, so distinct roles never overload one model id.
func multiRolePlan() RolePlanSet {
	return RolePlanSet{
		Fold: modelroute.ReduceFirst, // top-level composition absent any escalation
		Plans: []RolePlan{
			{
				Role: RoleScout,
				// interactive short probe -> DefaultManifest "small" (cheap) model.
				Subject: modelroute.Subject{Aspect: modelroute.AspectScout, Latency: modelroute.LatencyInteractive, PromptTokens: 256},
				Budget:  Budget{MaxTokens: 512, MaxLatencyMS: 300, MaxAttempts: 1, MaxCost: 0.01},
			},
			{
				Role: RoleWorker,
				// write-shaped tool call -> DefaultManifest guard ensemble (bounded doer).
				Subject:    modelroute.Subject{Aspect: modelroute.AspectToolCall, Tool: "write_repository"},
				Budget:     Budget{MaxTokens: 4000, MaxLatencyMS: 5000, MaxAttempts: 3, MaxCost: 0.20},
				EscalateTo: RoleJudge,
				Fold:       modelroute.ReduceConcat,
			},
			{
				Role: RoleJudge,
				// high-complexity adjudication -> DefaultManifest "large" (stronger) model.
				Subject: modelroute.Subject{Aspect: modelroute.AspectRequest, Complexity: modelroute.ComplexityHigh},
				Budget:  Budget{MaxTokens: 8000, MaxLatencyMS: 20000, MaxAttempts: 2, MaxCost: 1.00},
				Fold:    modelroute.ReduceBestOf,
			},
			{
				Role:    RolePrimary,
				Subject: modelroute.Subject{Aspect: modelroute.AspectRequest},
				Budget:  Budget{MaxTokens: 16000, MaxLatencyMS: 60000, MaxAttempts: 1, MaxCost: 2.00},
			},
		},
	}
}

// TestRouteRolesSelectsCheapScoutBoundedWorkerStrongerJudge proves the fixture's
// role→model split: the scout gets the cheap model, the judge the stronger one,
// and the bounded worker fans out to its ensemble — four distinct roles carrying
// four distinct model plans, no id overloaded.
func TestRouteRolesSelectsCheapScoutBoundedWorkerStrongerJudge(t *testing.T) {
	env, err := RouteRoles(multiRolePlan(), modelroute.DefaultManifest(), nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(env.Roles) != 4 { // Target operating envelope: routed_roles >= 4
		t.Fatalf("routed_roles=%d want 4", len(env.Roles))
	}
	scout, ok := env.Role(RoleScout)
	if !ok {
		t.Fatal("scout role missing from envelope")
	}
	judge, ok := env.Role(RoleJudge)
	if !ok {
		t.Fatal("judge role missing from envelope")
	}
	worker, ok := env.Role(RoleWorker)
	if !ok {
		t.Fatal("worker role missing from envelope")
	}
	if got := scout.Plan.Primary(); got != "small" {
		t.Fatalf("scout model=%q want cheap 'small'", got)
	}
	if got := judge.Plan.Primary(); got != "large" {
		t.Fatalf("judge model=%q want stronger 'large'", got)
	}
	if scout.Plan.Primary() == judge.Plan.Primary() {
		t.Fatal("scout and judge share a model id; roles are overloaded")
	}
	if !worker.Plan.IsEnsemble() {
		t.Fatalf("worker plan=%v want a bounded ensemble", worker.Plan.Models())
	}
	// Fresh spends: nothing is exhausted, so the top-level fold governs.
	if env.Escalated() {
		t.Fatal("no budget spent yet, but envelope reports an escalation")
	}
	if got := env.EffectiveFold(); got != modelroute.ReduceFirst {
		t.Fatalf("effective fold=%q want top-level %q", got, modelroute.ReduceFirst)
	}
}

// TestRouteRolesEscalatesAndFoldsOnBudgetExhaustion is the Witness proper: the
// worker's attempt budget is spent, which must (1) mark the worker exhausted on
// the attempts axis, (2) fire the declared escalation to the judge, and (3) hand
// the final fold to the judge's best_of policy — and that fold, run for real,
// must pick the highest-scored partial.
func TestRouteRolesEscalatesAndFoldsOnBudgetExhaustion(t *testing.T) {
	set := multiRolePlan()
	// The bounded worker has burned all three of its allowed attempts.
	spends := map[Role]Spend{
		RoleWorker: {Attempts: 3, Tokens: 1200, Cost: 0.05},
	}
	env, err := RouteRoles(set, modelroute.DefaultManifest(), spends)
	if err != nil {
		t.Fatal(err)
	}
	worker, _ := env.Role(RoleWorker)
	if worker.Status != RoleExhausted {
		t.Fatalf("worker status=%q want %q", worker.Status, RoleExhausted)
	}
	if worker.ExhaustedAxis != "attempts" {
		t.Fatalf("worker exhausted axis=%q want attempts", worker.ExhaustedAxis)
	}
	if worker.EscalatedTo != RoleJudge {
		t.Fatalf("worker escalated to=%q want %q", worker.EscalatedTo, RoleJudge)
	}
	if !env.Escalated() {
		t.Fatal("worker budget exhausted but envelope reports no escalation")
	}
	// The declared escalation/fold policy: the judge's fold now governs.
	fold := env.EffectiveFold()
	if fold != modelroute.ReduceBestOf {
		t.Fatalf("effective fold=%q want judge's %q after escalation", fold, modelroute.ReduceBestOf)
	}
	// Prove the fold actually executes the declared policy: best_of picks the
	// highest-scored worker partial, not the first.
	result, err := modelroute.Combine(fold, []modelroute.Vote{
		{Member: modelroute.Member{Model: "guard-a"}, Output: "weak", Score: 0.2},
		{Member: modelroute.Member{Model: "guard-b"}, Output: "strong", Score: 0.9},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Output != "strong" || result.Winner != "guard-b" {
		t.Fatalf("fold result=%+v want best_of to pick guard-b/'strong'", result)
	}
}

// TestBudgetExhaustionIsIndependentPerAxis proves the >= 3 independent budget
// axes: each of tokens, latency, attempts, and cost bounds the role on its own,
// and an all-zero budget is unbounded.
func TestBudgetExhaustionIsIndependentPerAxis(t *testing.T) {
	b := Budget{MaxTokens: 100, MaxLatencyMS: 200, MaxAttempts: 3, MaxCost: 0.50}
	cases := []struct {
		name string
		s    Spend
		axis string
	}{
		{"tokens", Spend{Tokens: 100}, "tokens"},
		{"latency", Spend{LatencyMS: 250}, "latency"},
		{"attempts", Spend{Attempts: 3}, "attempts"},
		{"cost", Spend{Cost: 0.50}, "cost"},
	}
	axes := map[string]bool{}
	for _, tc := range cases {
		axis, spent := b.Exhausted(tc.s)
		if !spent || axis != tc.axis {
			t.Fatalf("%s: axis=%q spent=%v want axis=%q spent=true", tc.name, axis, spent, tc.axis)
		}
		axes[axis] = true
	}
	if len(axes) < 3 { // Target operating envelope: budget_axes >= 3
		t.Fatalf("budget_axes=%d want >= 3", len(axes))
	}
	if _, spent := b.Exhausted(Spend{Tokens: 99, LatencyMS: 199, Attempts: 2, Cost: 0.49}); spent {
		t.Fatal("under-limit spend reported exhausted")
	}
	if _, spent := (Budget{}).Exhausted(Spend{Tokens: 1e6}); spent {
		t.Fatal("all-zero budget must be unbounded")
	}
}

// TestRouteWithRolesKeepsRolesInParentEnvelope proves the role decisions are
// inspectable in the parent execution envelope alongside harness/model/session.
func TestRouteWithRolesKeepsRolesInParentEnvelope(t *testing.T) {
	dec, err := RouteWithRoles(
		Request{
			Model:   modelroute.Subject{Aspect: modelroute.AspectRequest},
			Session: SessionSubject{ID: "s1", Portable: true},
		},
		multiRolePlan(),
		harnessprofile.Builtins(),
		modelroute.DefaultManifest(),
		map[Role]Spend{RoleWorker: {Attempts: 3}},
	)
	if err != nil {
		t.Fatal(err)
	}
	if dec.Roles == nil {
		t.Fatal("parent Decision.Roles is nil; role plan not attached")
	}
	if dec.Session.Action != SessionFork {
		t.Fatalf("session=%q want fork (parent envelope still composed)", dec.Session.Action)
	}
	worker, ok := dec.Roles.Role(RoleWorker)
	if !ok || worker.EscalatedTo != RoleJudge {
		t.Fatalf("parent envelope lost the worker->judge escalation: %+v", worker)
	}
}

// TestRouteRolesRejectsMalformedPlans keeps the role vocabulary closed and
// escalation targets bound to declared roles.
func TestRouteRolesRejectsMalformedPlans(t *testing.T) {
	if _, err := RouteRoles(RolePlanSet{}, modelroute.DefaultManifest(), nil); err == nil {
		t.Fatal("empty plan set should fail")
	}
	unknown := RolePlanSet{Plans: []RolePlan{{Role: Role("oracle")}}}
	if _, err := RouteRoles(unknown, modelroute.DefaultManifest(), nil); err == nil {
		t.Fatal("unknown role should fail")
	}
	dangling := RolePlanSet{Plans: []RolePlan{{Role: RoleWorker, EscalateTo: RoleJudge}}}
	if _, err := RouteRoles(dangling, modelroute.DefaultManifest(), nil); err == nil {
		t.Fatal("escalation to an undeclared role should fail")
	}
	dup := RolePlanSet{Plans: []RolePlan{{Role: RoleWorker}, {Role: RoleWorker}}}
	if _, err := RouteRoles(dup, modelroute.DefaultManifest(), nil); err == nil {
		t.Fatal("duplicate role should fail")
	}
}
