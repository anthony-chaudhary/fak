package dispatchtick

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// highRiskVerbs is the exact set #3044 names as requiring escalation: mutate git,
// kill a process, launch a worker, edit a label, claim done.
var highRiskVerbs = []MetaAction{
	ActionGitMutate, ActionKillProcess, ActionLaunchWorker, ActionEditLabel, ActionClaimDone,
}

var readOnlyVerbs = []MetaAction{
	ActionReadStatus, ActionRank, ActionSummarize, ActionSuggestNext,
}

// The core done-condition: a T2 meta-orchestrator is READ-ONLY. It may read,
// rank, summarize, and suggest — but every mutation verb #3044 names is refused
// direct execution and produces a typed escalation instead.
func TestOrchestratorT2IsReadOnly(t *testing.T) {
	// Read-only verbs are allowed for the cheapest tier.
	for _, a := range readOnlyVerbs {
		d := Authorize(modelroute.TierT2, a, false)
		if !d.Allowed() {
			t.Fatalf("T2 must be allowed read-only verb %q, got %+v", a, d)
		}
		if d.Reason != MetaReasonReadOnlyAllowed {
			t.Fatalf("verb %q reason = %q, want %q", a, d.Reason, MetaReasonReadOnlyAllowed)
		}
	}

	// Every high-risk verb is NOT executable by T2 and yields a typed escalation.
	for _, a := range highRiskVerbs {
		d := Authorize(modelroute.TierT2, a, false)
		if d.Allowed() {
			t.Fatalf("T2 must NOT execute high-risk verb %q, got allowed: %+v", a, d)
		}
		if d.Outcome != OutcomeEscalate {
			t.Fatalf("verb %q outcome = %q, want %q", a, d.Outcome, OutcomeEscalate)
		}
		if d.Reason != MetaReasonEscalationRequired {
			t.Fatalf("verb %q reason = %q, want %q", a, d.Reason, MetaReasonEscalationRequired)
		}
		if d.RequiredTier != modelroute.TierT0 {
			t.Fatalf("verb %q required tier = %s, want T0", a, d.RequiredTier)
		}
	}

	// The envelope readout agrees: T2's ceiling is read-only.
	env := Envelope(modelroute.TierT2)
	if env.MaxRisk != RiskReadOnly {
		t.Fatalf("T2 envelope MaxRisk = %s, want read-only", env.MaxRisk)
	}
	if len(env.Allowed) != len(readOnlyVerbs) {
		t.Fatalf("T2 envelope should allow exactly the read-only verbs, got %+v", env.Allowed)
	}
}

// High-risk actions escalate for every sub-T0 tier without a witness, and only T0
// executes them directly.
func TestOrchestratorHighRiskNeedsT0OrWitness(t *testing.T) {
	for _, a := range highRiskVerbs {
		// T0 executes directly.
		if d := Authorize(modelroute.TierT0, a, false); !d.Allowed() || d.Reason != MetaReasonTierAuthorized {
			t.Fatalf("T0 must execute high-risk %q directly, got %+v", a, d)
		}
		// T1 without a witness escalates (it can propose, not perform high-risk).
		if d := Authorize(modelroute.TierT1, a, false); d.Outcome != OutcomeEscalate {
			t.Fatalf("T1 high-risk %q must escalate without a witness, got %+v", a, d)
		}
		// A witness authorizes the otherwise over-authority action for a cheap tier.
		d := Authorize(modelroute.TierT2, a, true)
		if !d.Allowed() {
			t.Fatalf("witnessed high-risk %q must be allowed for T2, got %+v", a, d)
		}
		if d.Reason != MetaReasonWitnessAuthorized || !d.Witnessed {
			t.Fatalf("witnessed %q reason = %q witnessed=%v, want witnessed-escalation", a, d.Reason, d.Witnessed)
		}
	}
}

// The T1/T0 paths remain available (the done-condition's third clause): T1 may
// propose guarded mutations, which T2 must escalate and T0 may also perform.
func TestOrchestratorGuardedMutationTierPaths(t *testing.T) {
	guarded := ActionProposeMutation

	if d := Authorize(modelroute.TierT1, guarded, false); !d.Allowed() || d.Reason != MetaReasonTierAuthorized {
		t.Fatalf("T1 must be authorized to propose guarded mutation, got %+v", d)
	}
	if d := Authorize(modelroute.TierT0, guarded, false); !d.Allowed() {
		t.Fatalf("T0 must also be able to propose guarded mutation, got %+v", d)
	}
	if d := Authorize(modelroute.TierT2, guarded, false); d.Outcome != OutcomeEscalate {
		t.Fatalf("T2 guarded mutation must escalate, got %+v", d)
	}
	if d := Authorize(modelroute.TierT2, guarded, false); d.RequiredTier != modelroute.TierT1 {
		t.Fatalf("guarded mutation required tier = %s, want T1", d.RequiredTier)
	}
}

// An unknown verb fails closed — a cheap model cannot invent authority by naming a
// verb the envelope has never classified.
func TestOrchestratorUnknownActionRefused(t *testing.T) {
	// Even the frontier tier cannot run an unclassified verb.
	d := Authorize(modelroute.TierT0, MetaAction("rm-rf-everything"), false)
	if d.Outcome != OutcomeRefuse {
		t.Fatalf("unknown verb must refuse, got %+v", d)
	}
	if d.Reason != MetaReasonUnknownActionRefused {
		t.Fatalf("unknown verb reason = %q, want %q", d.Reason, MetaReasonUnknownActionRefused)
	}
	// A witness does not rescue an unknown verb — refusal is not an escalation.
	if d := Authorize(modelroute.TierT0, MetaAction("unknown"), true); d.Outcome != OutcomeRefuse {
		t.Fatalf("a witness must not rescue an unknown verb, got %+v", d)
	}
}

// A T2 suggestion is useful but is NOT an execution witness (a #3044 confusion
// risk): suggesting a next action stays read-only, and a later high-risk claim by
// the same T2 orchestrator still escalates — the suggestion never became authority.
func TestOrchestratorSuggestionIsNotAWitness(t *testing.T) {
	sug := Authorize(modelroute.TierT2, ActionSuggestNext, false)
	if !sug.Allowed() || sug.Risk != RiskReadOnly {
		t.Fatalf("T2 suggestion must be an allowed read-only action, got %+v", sug)
	}
	// The suggestion must not implicitly witness a subsequent high-risk action.
	claim := Authorize(modelroute.TierT2, ActionClaimDone, false)
	if claim.Allowed() {
		t.Fatalf("a prior T2 suggestion must not authorize claim-done, got %+v", claim)
	}
	if claim.Outcome != OutcomeEscalate {
		t.Fatalf("T2 claim-done must escalate, got %+v", claim)
	}
}

// The envelope readout is coherent across the ladder: the ceiling rises with the
// tier, and the exact high-risk verb set is what a sub-T0 tier must escalate.
func TestOrchestratorEnvelopeLadder(t *testing.T) {
	cases := []struct {
		tier    OrchestratorTier
		maxRisk ActionRisk
	}{
		{modelroute.TierT2, RiskReadOnly},
		{modelroute.TierT1, RiskGuardedMutation},
		{modelroute.TierT0, RiskHighRisk},
	}
	for _, c := range cases {
		env := Envelope(c.tier)
		if env.MaxRisk != c.maxRisk {
			t.Fatalf("%s envelope MaxRisk = %s, want %s", c.tier, env.MaxRisk, c.maxRisk)
		}
	}

	// T0 escalates nothing; every classified verb is within its authority.
	if env := Envelope(modelroute.TierT0); len(env.EscalationRequired) != 0 {
		t.Fatalf("T0 must escalate nothing, got %+v", env.EscalationRequired)
	}

	// Sub-T0 tiers must escalate every high-risk verb #3044 names.
	for _, tier := range []OrchestratorTier{modelroute.TierT2, modelroute.TierT1} {
		env := Envelope(tier)
		for _, a := range highRiskVerbs {
			if !containsMetaAction(env.EscalationRequired, a) {
				t.Fatalf("%s envelope must escalate high-risk verb %q, got %+v", tier, a, env.EscalationRequired)
			}
			if containsMetaAction(env.Allowed, a) {
				t.Fatalf("%s envelope must NOT directly allow high-risk verb %q", tier, a)
			}
		}
	}
}

func containsMetaAction(xs []MetaAction, want MetaAction) bool {
	for _, x := range xs {
		if x == want {
			return true
		}
	}
	return false
}
