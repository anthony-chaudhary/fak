// Admission wiring QA for the steerpr overlay-maintenance loop (#5023): the
// loop's tick done-claim binds to the commit-audit witness over the tick's
// base..head range, so the gate admits it ONLY on an external
// OutcomeWitnessed. A self-reported/fabricated done re-arms with
// LOOP_DONE_UNWITNESSED — the exact refusal loopgate exists to make.
package loopgate

import (
	"context"
	"strings"
	"testing"
)

func TestTurnForSteerprTickBindsCommitAuditOverRange(t *testing.T) {
	turn := TurnForSteerprTick("aaa111", "bbb222", "steerpr overlay tick claim", true)
	if !turn.ClaimedDone {
		t.Fatal("turn does not carry the done claim")
	}
	if turn.Criterion.Kind != CriterionCommitAudit {
		t.Fatalf("criterion kind = %q, want %q", turn.Criterion.Kind, CriterionCommitAudit)
	}
	if turn.Criterion.Ref != "aaa111..bbb222" {
		t.Fatalf("criterion ref = %q, want the base..head range", turn.Criterion.Ref)
	}
	if turn.HeadRef != "bbb222" {
		t.Fatalf("head ref = %q, want bbb222", turn.HeadRef)
	}
	// A rangeless (genesis) tick still binds a usable audit ref: the head.
	genesis := TurnForSteerprTick("", "bbb222", "claim", true)
	if genesis.Criterion.Ref != "bbb222" {
		t.Fatalf("genesis criterion ref = %q, want bbb222", genesis.Criterion.Ref)
	}
}

func TestSteerprTickFabricatedDoneReArmsUnwitnessed(t *testing.T) {
	turn := TurnForSteerprTick("aaa111", "bbb222", "steerpr overlay tick claim", true)

	// The external surface does not corroborate the claim: the only honest
	// verdict is NOT_YET with the closed re-arm reason, never an admit.
	d := Adjudicate(context.Background(), turn, func(context.Context, Request) (WitnessResult, error) {
		return WitnessResult{Outcome: OutcomeNotYet, Detail: "self-report only"}, nil
	})
	if d.Verdict != VerdictNotYet {
		t.Fatalf("fabricated done verdict = %q, want %q", d.Verdict, VerdictNotYet)
	}
	if d.Reason != ReasonDoneUnwitnessed {
		t.Fatalf("fabricated done reason = %q, want %q", d.Reason, ReasonDoneUnwitnessed)
	}
	if got := SteerprAuditRef(d); got != "" {
		t.Fatalf("unwitnessed decision rendered witness binding %q, want empty", got)
	}

	// A turn that never claimed done just continues — also unwitnessed, and no
	// witness call is owed at all.
	quiet := Adjudicate(context.Background(), TurnForSteerprTick("aaa111", "bbb222", "", false),
		func(context.Context, Request) (WitnessResult, error) {
			t.Fatal("witness invoked for a turn that claimed nothing")
			return WitnessResult{}, nil
		})
	if quiet.Verdict != VerdictNotYet || quiet.Reason != ReasonDoneUnwitnessed {
		t.Fatalf("unclaimed turn = (%q, %q), want (NOT_YET, %s)", quiet.Verdict, quiet.Reason, ReasonDoneUnwitnessed)
	}
}

func TestSteerprTickWitnessedDoneAdmittedWithBinding(t *testing.T) {
	turn := TurnForSteerprTick("aaa111", "bbb222", "steerpr overlay tick claim", true)
	d := Adjudicate(context.Background(), turn, func(_ context.Context, req Request) (WitnessResult, error) {
		if req.Kind != CriterionCommitAudit || req.Ref != "aaa111..bbb222" {
			t.Fatalf("witness request = %+v, want commit-audit over aaa111..bbb222", req)
		}
		return WitnessResult{Outcome: OutcomeWitnessed, Rung: "commit-audit-range"}, nil
	})
	if d.Verdict != VerdictWitnessed {
		t.Fatalf("witnessed done verdict = %q, want %q", d.Verdict, VerdictWitnessed)
	}
	binding := SteerprAuditRef(d)
	if binding == "" || !strings.Contains(binding, "commit-audit") || !strings.Contains(binding, "aaa111..bbb222") {
		t.Fatalf("witness binding = %q, want a re-checkable commit-audit reference over the range", binding)
	}
}
