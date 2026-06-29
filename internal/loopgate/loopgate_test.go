package loopgate

import (
	"context"
	"errors"
	"strings"
	"testing"
)

func TestAdjudicateCommitAuditWitnessed(t *testing.T) {
	var gotReq Request
	dec := Adjudicate(context.Background(), Turn{
		ClaimedDone: true,
		Claim:       "implemented the exit gate",
		HeadRef:     "HEAD",
	}, func(_ context.Context, req Request) (WitnessResult, error) {
		gotReq = req
		return WitnessResult{Outcome: OutcomeWitnessed, Reason: "OK", Detail: "diff-witnessed", Rung: "diff-witnessed"}, nil
	})
	if dec.Verdict != VerdictWitnessed {
		t.Fatalf("verdict = %+v, want WITNESSED", dec)
	}
	if gotReq.Kind != CriterionCommitAudit || gotReq.Ref != "HEAD" || gotReq.Claim == "" {
		t.Fatalf("request = %+v, want commit-audit HEAD with claim", gotReq)
	}
	if got := strings.Join(gotReq.Argv(), " "); got != "commit-audit --json HEAD" {
		t.Fatalf("argv = %q", got)
	}
}

func TestAdjudicateClaimUnwitnessedRearms(t *testing.T) {
	dec := Adjudicate(context.Background(), Turn{ClaimedDone: true}, func(_ context.Context, req Request) (WitnessResult, error) {
		return WitnessResult{
			Outcome:    OutcomeNotYet,
			Reason:     "CLAIM_UNWITNESSED",
			Detail:     "subject-only claim",
			RawVerdict: "CLAIM_UNWITNESSED",
			Rung:       "subject-only",
		}, nil
	})
	if dec.Verdict != VerdictNotYet || dec.Reason != ReasonDoneUnwitnessed {
		t.Fatalf("decision = %+v, want NOT_YET/%s", dec, ReasonDoneUnwitnessed)
	}
	if !strings.Contains(dec.Summary, "subject-only") {
		t.Fatalf("summary should surface witness reason, got %q", dec.Summary)
	}
}

func TestAdjudicateVerifyPlanPhase(t *testing.T) {
	var gotReq Request
	dec := Adjudicate(context.Background(), Turn{
		ClaimedDone: true,
		Criterion: Criterion{
			Kind:  CriterionVerify,
			Plan:  "fak",
			Phase: "loopgate",
		},
	}, func(_ context.Context, req Request) (WitnessResult, error) {
		gotReq = req
		return WitnessResult{Outcome: OutcomeNotYet, Reason: "NOT_SHIPPED", Detail: "source=none", Rung: "none"}, nil
	})
	if dec.Verdict != VerdictNotYet || dec.Reason != ReasonDoneUnwitnessed {
		t.Fatalf("decision = %+v, want NOT_YET/%s", dec, ReasonDoneUnwitnessed)
	}
	if gotReq.Kind != CriterionVerify || gotReq.Plan != "fak" || gotReq.Phase != "loopgate" {
		t.Fatalf("request = %+v, want verify fak loopgate", gotReq)
	}
	if got := strings.Join(gotReq.Argv(), " "); got != "verify --json fak loopgate" {
		t.Fatalf("argv = %q", got)
	}
}

func TestAdjudicateTestWitnessVacuousRearms(t *testing.T) {
	var gotReq Request
	dec := Adjudicate(context.Background(), Turn{
		ClaimedDone: true,
		Claim:       "added a test",
		Criterion: Criterion{
			Kind:      CriterionTestWitness,
			Baseline:  "pass",
			Candidate: "pass",
		},
	}, func(_ context.Context, req Request) (WitnessResult, error) {
		gotReq = req
		return TestWitnessResultFromJSON([]byte(`{"verdict":"VACUOUS","witnesses":false,"reason":"pass/pass witnesses nothing","evidence":{"rung":"OS_RECORDED"}}`))
	})
	if dec.Verdict != VerdictNotYet || dec.Reason != ReasonDoneUnwitnessed {
		t.Fatalf("decision = %+v, want NOT_YET/%s", dec, ReasonDoneUnwitnessed)
	}
	if got := strings.Join(gotReq.Argv(), " "); got != "test-witness --json --baseline pass --candidate pass" {
		t.Fatalf("argv = %q", got)
	}
	if !strings.Contains(dec.Summary, "witnesses nothing") {
		t.Fatalf("summary = %q, want vacuity reason", dec.Summary)
	}
}

func TestAdjudicateCitationResolveRefutesFabricatedCitation(t *testing.T) {
	var gotReq Request
	dec := Adjudicate(context.Background(), Turn{
		ClaimedDone: true,
		Claim:       "relied on a cited case",
		Criterion:   Criterion{Kind: CriterionCitationResolve, Subject: "999 F.999 1"},
	}, func(_ context.Context, req Request) (WitnessResult, error) {
		gotReq = req
		return GenericWitnessResultFromJSON([]byte(`{"facts":{"source_name":"citation_resolve","accountability":"THIRD_PARTY","stance":"REFUTED","detail":"no reporter cluster carries citation"},"belief":{"believe":false,"refuted":true,"reason":"refuted by citation_resolve"}}`))
	})
	if dec.Verdict != VerdictNotYet || dec.Reason != ReasonDoneUnwitnessed {
		t.Fatalf("decision = %+v, want NOT_YET/%s", dec, ReasonDoneUnwitnessed)
	}
	if got := strings.Join(gotReq.Argv(), " "); got != "witness --json citation_resolve 999 F.999 1" {
		t.Fatalf("argv = %q", got)
	}
	if !strings.Contains(dec.Summary, "no reporter cluster") {
		t.Fatalf("summary = %q, want resolver detail", dec.Summary)
	}
}

func TestAdjudicateNoDoneClaimDoesNotCallWitness(t *testing.T) {
	called := false
	dec := Adjudicate(context.Background(), Turn{ClaimedDone: false}, func(_ context.Context, req Request) (WitnessResult, error) {
		called = true
		return WitnessResult{Outcome: OutcomeWitnessed}, nil
	})
	if called {
		t.Fatal("witness must not be called when the turn did not claim done")
	}
	if dec.Verdict != VerdictNotYet || dec.Reason != ReasonDoneUnwitnessed {
		t.Fatalf("decision = %+v, want NOT_YET/%s", dec, ReasonDoneUnwitnessed)
	}
}

func TestMetricOnlyNoopCannotProduceWitnessed(t *testing.T) {
	called := false
	dec := Adjudicate(context.Background(), Turn{
		ClaimedDone: true,
		Criterion:   Criterion{Kind: CriterionMetric, Subject: "slop_score<=0"},
	}, func(_ context.Context, req Request) (WitnessResult, error) {
		called = true
		return WitnessResult{Outcome: OutcomeWitnessed, Reason: "metric improved"}, nil
	})
	if called {
		t.Fatal("raw metric criteria must not be handed to the witness adapter as a passable oracle")
	}
	if dec.Verdict == VerdictWitnessed {
		t.Fatalf("metric-only criterion produced WITNESSED: %+v", dec)
	}
	if dec.Verdict != VerdictNotYet || dec.Reason != ReasonDoneUnwitnessed {
		t.Fatalf("decision = %+v, want NOT_YET/%s", dec, ReasonDoneUnwitnessed)
	}
}

func TestWitnessRefusalTerminates(t *testing.T) {
	dec := Adjudicate(context.Background(), Turn{ClaimedDone: true}, func(_ context.Context, req Request) (WitnessResult, error) {
		return WitnessResult{Outcome: OutcomeRefused, Reason: "FORBIDDEN_CALL_SHAPE", Detail: "witness command forbidden"}, nil
	})
	if dec.Verdict != VerdictRefused || dec.Reason != "FORBIDDEN_CALL_SHAPE" {
		t.Fatalf("decision = %+v, want REFUSED/FORBIDDEN_CALL_SHAPE", dec)
	}
}

func TestMalformedCriterionRefuses(t *testing.T) {
	dec := Adjudicate(context.Background(), Turn{
		ClaimedDone: true,
		Criterion:   Criterion{Kind: CriterionVerify, Plan: "fak"},
	}, nil)
	if dec.Verdict != VerdictRefused || dec.Reason != ReasonSchemaUnreadable {
		t.Fatalf("decision = %+v, want REFUSED/%s", dec, ReasonSchemaUnreadable)
	}
}

func TestWitnessAdapterErrorRefuses(t *testing.T) {
	dec := Adjudicate(context.Background(), Turn{ClaimedDone: true}, func(_ context.Context, req Request) (WitnessResult, error) {
		return WitnessResult{}, errors.New("dos unavailable")
	})
	if dec.Verdict != VerdictRefused || dec.Reason != ReasonSchemaUnreadable {
		t.Fatalf("decision = %+v, want REFUSED/%s", dec, ReasonSchemaUnreadable)
	}
	if !strings.Contains(dec.Summary, "dos unavailable") {
		t.Fatalf("summary = %q, want adapter error", dec.Summary)
	}
}

func TestJSONParsers(t *testing.T) {
	audit, err := CommitAuditResultFromJSON([]byte(`[{"verdict":"CLAIM_UNWITNESSED","witness":"subject-only","reason":"README only"}]`))
	if err != nil {
		t.Fatal(err)
	}
	if audit.Outcome != OutcomeNotYet || audit.Reason != "CLAIM_UNWITNESSED" {
		t.Fatalf("audit = %+v", audit)
	}

	verify, err := VerifyResultFromJSON([]byte(`{"plan":"fak","phase":"loopgate","shipped":true,"source":"grep-subject"}`))
	if err != nil {
		t.Fatal(err)
	}
	if verify.Outcome != OutcomeWitnessed || verify.Rung != "grep-subject" {
		t.Fatalf("verify = %+v", verify)
	}

	testWitness, err := TestWitnessResultFromJSON([]byte(`{"verdict":"VACUOUS","witnesses":false,"reason":"pass/pass","evidence":{"rung":"OS_RECORDED"}}`))
	if err != nil {
		t.Fatal(err)
	}
	if testWitness.Outcome != OutcomeNotYet || testWitness.RawVerdict != "VACUOUS" {
		t.Fatalf("test witness = %+v", testWitness)
	}

	generic, err := GenericWitnessResultFromJSON([]byte(`{"facts":{"accountability":"OS_RECORDED","stance":"NO_SIGNAL","detail":"no file"},"belief":{"believe":false,"refuted":false,"reason":"abstain"}}`))
	if err != nil {
		t.Fatal(err)
	}
	if generic.Outcome != OutcomeNotYet || generic.Rung != "OS_RECORDED" {
		t.Fatalf("generic witness = %+v", generic)
	}
}
