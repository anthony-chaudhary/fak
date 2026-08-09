package trajctl

import "testing"

func TestStampAuthorityConformance(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name       string
		req        AuthorityRequest
		want       Authority
		refused    bool
		wantReason string
	}{
		{name: "w3 fact", req: AuthorityRequest{AuthorityAssert, W3, W3}, want: AuthorityAssert},
		{name: "supervisor bounds fact", req: AuthorityRequest{AuthorityAssert, W3, W2}, want: AuthorityVerify, refused: true, wantReason: ReasonLessonOverclaims},
		{name: "evidence bounds fact", req: AuthorityRequest{AuthorityAssert, W2, W3}, want: AuthorityVerify, refused: true, wantReason: ReasonLessonOverclaims},
		{name: "w2 verification", req: AuthorityRequest{AuthorityVerify, W2, W3}, want: AuthorityVerify},
		{name: "w1 verification", req: AuthorityRequest{AuthorityVerify, W1, W3}, want: AuthorityVerify},
		{name: "w0 can only ask", req: AuthorityRequest{AuthorityVerify, W0, W3}, want: AuthorityAsk, refused: true, wantReason: ReasonLessonOverclaims},
		{name: "asking never overclaims", req: AuthorityRequest{AuthorityAsk, W0, W0}, want: AuthorityAsk},
	}
	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got, outcome, err := StampAuthority(tc.req)
			if err != nil {
				t.Fatalf("StampAuthority: %v", err)
			}
			if got.Authority != tc.want || outcome.Refused != tc.refused || outcome.Reason != tc.wantReason {
				t.Fatalf("got stamp=%+v outcome=%+v, want authority=%q refused=%v reason=%q", got, outcome, tc.want, tc.refused, tc.wantReason)
			}
			if got.EvidenceWitness != tc.req.EvidenceWitness || got.SupervisorRung != tc.req.SupervisorRung {
				t.Fatalf("stamp lost witness provenance: got %+v, request %+v", got, tc.req)
			}
		})
	}
}

func TestStampAuthorityRejectsUnknownContractValues(t *testing.T) {
	t.Parallel()
	for _, req := range []AuthorityRequest{
		{Requested: Authority("fact"), EvidenceWitness: W3, SupervisorRung: W3},
		{Requested: AuthorityAsk, EvidenceWitness: WitnessRung("W9"), SupervisorRung: W3},
	} {
		if _, _, err := StampAuthority(req); err == nil {
			t.Fatalf("StampAuthority(%+v) accepted an undeclared contract value", req)
		}
	}
}

func TestAuthorityRefusalUsesClosedDOSReason(t *testing.T) {
	t.Parallel()
	// This token is declared by [reasons.LESSON_OVERCLAIMS] in dos.toml. Pinning
	// it here prevents an intervention from inventing a free-text refusal class.
	if ReasonLessonOverclaims != "LESSON_OVERCLAIMS" {
		t.Fatalf("reason = %q", ReasonLessonOverclaims)
	}
	_, outcome, err := StampAuthority(AuthorityRequest{AuthorityAssert, W3, W2})
	if err != nil {
		t.Fatal(err)
	}
	if !outcome.Refused || outcome.Reason != ReasonLessonOverclaims {
		t.Fatalf("unwitnessed supervisor assertion was not fail-closed: %+v", outcome)
	}
}
