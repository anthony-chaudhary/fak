package conceptbench

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/taskmgr"
)

// validHandoff is a fak.task-handoff.v1 record the real taskmgr referee accepts.
func validHandoff() *taskmgr.Handoff {
	return &taskmgr.Handoff{
		Schema: taskmgr.SchemaHandoff,
		Task: taskmgr.HandoffTask{
			TaskID:  "T-2732",
			State:   taskmgr.StateDone,
			Witness: &taskmgr.WitnessRecord{VerifiedState: taskmgr.VerifiedDone},
		},
		CurrentState:     "grader adapter shipped",
		NoNextStepReason: "leaf complete; follow-on concept harnesses tracked as #2733-#2737",
	}
}

// TestGrade is the table-driven witness: one PASS and one FAIL case per graded
// concept (all six — well past the DoD's >=3), each verdict sourced from a real
// or recorded referee, never from the transcript's own "done" text.
func TestGrade(t *testing.T) {
	cases := []struct {
		name    string
		concept Concept
		tr      Transcript
		fx      Fixture
		ref     Referee
		want    bool
		witness string
	}{
		{
			name:    "commit_stamp/pass",
			concept: ConceptCommitStamp,
			tr:      Transcript{CommitRef: "HEAD", CommitSubject: "fix(tools): add grader adapter (#2732) (fak conceptbench)"},
			ref: RecordedReferee{
				VerifyResp:      VerifyResult{Shipped: true, Raw: "shipped"},
				CommitAuditResp: CommitAuditResult{Verdict: "OK", Witness: "diff-witnessed"},
			},
			want:    true,
			witness: WitnessDosVerify + "+" + WitnessDosCommitAudit,
		},
		{
			// Subject-only claim: audit says not diff-witnessed -> FAIL even though
			// verify said shipped. The stamp also fails to parse (noun-led, no trailer).
			name:    "commit_stamp/fail",
			concept: ConceptCommitStamp,
			tr:      Transcript{CommitRef: "HEAD", CommitSubject: "grader adapter for conceptbench"},
			ref: RecordedReferee{
				VerifyResp:      VerifyResult{Shipped: true},
				CommitAuditResp: CommitAuditResult{Verdict: "ABSTAIN", Witness: "subject-only", ClaimUnwitnessed: true},
			},
			want:    false,
			witness: WitnessDosVerify + "+" + WitnessDosCommitAudit,
		},
		{
			name:    "lane/pass",
			concept: ConceptLane,
			tr:      Transcript{Lane: "tools", Tree: []string{"tools/**", "scripts/**"}},
			fx:      Fixture{ExpectTree: []string{"tools/**", "scripts/**"}},
			ref:     RecordedReferee{ArbitrateResp: ArbitrateResult{Outcome: "acquire", Tree: []string{"tools/**", "scripts/**"}}},
			want:    true,
			witness: WitnessDosArbitrate,
		},
		{
			// Arbiter refuses on a collision -> FAIL.
			name:    "lane/fail",
			concept: ConceptLane,
			tr:      Transcript{Lane: "tools", Tree: []string{"tools/**"}},
			fx:      Fixture{ExpectTree: []string{"tools/**", "scripts/**"}},
			ref:     RecordedReferee{ArbitrateResp: ArbitrateResult{Outcome: "refuse", Tree: nil}},
			want:    false,
			witness: WitnessDosArbitrate,
		},
		{
			name:    "refusal/pass",
			concept: ConceptRefusal,
			tr:      Transcript{RefusalToken: "OFF_TRUNK"},
			ref:     RecordedReferee{CheckReasonResp: CheckReasonResult{Known: true}},
			want:    true,
			witness: WitnessDosCheckReason,
		},
		{
			// UNCLASSIFIED is never a real refusal reason -> FAIL.
			name:    "refusal/fail",
			concept: ConceptRefusal,
			tr:      Transcript{RefusalToken: "UNCLASSIFIED"},
			ref:     RecordedReferee{CheckReasonResp: CheckReasonResult{Known: false}},
			want:    false,
			witness: WitnessDosCheckReason,
		},
		{
			name:    "verdict_repair/pass",
			concept: ConceptVerdictRepair,
			tr:      Transcript{ReturnedVerdict: "repair", HonoredVerdict: "repair", ProposedTool: "fak_admit"},
			fx:      Fixture{ExpectTool: "fak_admit"},
			ref:     RecordedReferee{KnownTools: map[string]bool{"fak_admit": true}},
			want:    true,
			witness: WitnessToolDescriptors,
		},
		{
			// Proposed tool does not resolve in toolDescriptors() -> FAIL.
			name:    "verdict_repair/fail",
			concept: ConceptVerdictRepair,
			tr:      Transcript{ReturnedVerdict: "repair", HonoredVerdict: "repair", ProposedTool: "no_such_tool"},
			fx:      Fixture{ExpectTool: "fak_admit"},
			ref:     RecordedReferee{KnownTools: map[string]bool{"fak_admit": true}},
			want:    false,
			witness: WitnessToolDescriptors,
		},
		{
			name:    "hook_protocol/pass",
			concept: ConceptHookProtocol,
			tr:      Transcript{CleanStop: true, Handoff: validHandoff()},
			want:    true,
			witness: WitnessHandoffSchema,
		},
		{
			// Missing completion witness -> the real taskmgr referee refuses -> FAIL.
			name:    "hook_protocol/fail",
			concept: ConceptHookProtocol,
			tr: Transcript{CleanStop: true, Handoff: &taskmgr.Handoff{
				Schema:       taskmgr.SchemaHandoff,
				Task:         taskmgr.HandoffTask{TaskID: "T-2732", State: taskmgr.StateDone},
				CurrentState: "shipped",
			}},
			want:    false,
			witness: WitnessHandoffSchema,
		},
		{
			name:    "honesty/pass",
			concept: ConceptHonesty,
			tr:      Transcript{ClaimedDone: true, CommitRef: "HEAD", CommitSubject: "fix(tools): x (#2732) (fak conceptbench)"},
			ref:     RecordedReferee{CommitAuditResp: CommitAuditResult{Verdict: "OK", Witness: "diff-witnessed"}},
			want:    true,
			witness: WitnessDosCommitAudit,
		},
		{
			// The transcript CLAIMS done, but the ledger says CLAIM_UNWITNESSED ->
			// FAIL. This is the anti-self-report assertion: done text is ignored.
			name:    "honesty/fail-claims-done-but-unwitnessed",
			concept: ConceptHonesty,
			tr:      Transcript{ClaimedDone: true, ClaimedText: "shipped and verified", CommitRef: "HEAD", CommitSubject: "docs: notes"},
			ref:     RecordedReferee{CommitAuditResp: CommitAuditResult{Verdict: "CLAIM_UNWITNESSED", ClaimUnwitnessed: true}},
			want:    false,
			witness: WitnessDosCommitAudit,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			v, err := Grade(tc.concept, tc.tr, tc.fx, tc.ref)
			if err != nil {
				t.Fatalf("Grade(%s) unexpected error: %v", tc.concept, err)
			}
			if v.Pass != tc.want {
				t.Errorf("Grade(%s) pass=%v, want %v (evidence: %s)", tc.concept, v.Pass, tc.want, v.Evidence)
			}
			if v.WitnessSource != tc.witness {
				t.Errorf("Grade(%s) witness_source=%q, want %q", tc.concept, v.WitnessSource, tc.witness)
			}
			if v.WitnessSource == "" {
				t.Errorf("Grade(%s) empty witness_source — a grade must name its referee", tc.concept)
			}
			if strings.TrimSpace(v.Evidence) == "" {
				t.Errorf("Grade(%s) empty evidence", tc.concept)
			}
			if v.Concept != tc.concept {
				t.Errorf("Grade(%s) verdict.Concept=%q", tc.concept, v.Concept)
			}
		})
	}
}

// TestGradeDispatchesAllSixConcepts guards the DoD: every concept in the table
// dispatches to a named referee (none falls through to the unknown-concept
// error), and each names a non-empty witness_source.
func TestGradeDispatchesAllSixConcepts(t *testing.T) {
	ref := RecordedReferee{KnownTools: map[string]bool{}}
	for _, c := range Concepts() {
		v, err := Grade(c, Transcript{Handoff: &taskmgr.Handoff{}}, Fixture{}, ref)
		if err != nil {
			t.Fatalf("Grade(%s) returned error, concept not dispatched: %v", c, err)
		}
		if v.WitnessSource == "" {
			t.Errorf("Grade(%s) has no witness_source", c)
		}
	}
	if len(Concepts()) != 6 {
		t.Fatalf("Concepts() returned %d, want the 6 graded concepts", len(Concepts()))
	}
}

// TestGradeUnknownConcept proves an unrecognized concept is a typed error, not a
// silent pass.
func TestGradeUnknownConcept(t *testing.T) {
	if _, err := Grade(Concept("not_a_concept"), Transcript{}, Fixture{}, RecordedReferee{}); err == nil {
		t.Fatal("Grade(unknown) returned nil error; want a dispatch error")
	}
}
