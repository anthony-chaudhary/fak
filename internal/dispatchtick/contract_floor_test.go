package dispatchtick

import "testing"

func TestDispatchContractRejectsPrivateBoundaryWithoutDevPackage(t *testing.T) {
	issue := routerIssue(12, "gateway: scoped private leaf", nil, scopedGatewayIssueBody("4")+"\n\n## Boundary detail\nUse the fak-private control bridge.")
	review := reviewDispatchContract(issue)
	if review.OK || firstIssueContractReason(review) != reasonPrivateBoundary {
		t.Fatalf("private review = %+v, want %s refusal", review, reasonPrivateBoundary)
	}
}

func TestDispatchContractRejectsMissingScope(t *testing.T) {
	issue := routerIssue(13, "gateway: thin row", nil, "## Lane\ngateway\n\n## Path hints\n- `internal/gateway/http.go`")
	review := reviewDispatchContract(issue)
	if review.OK || firstIssueContractReason(review) != reasonScopeIncomplete {
		t.Fatalf("thin review = %+v, want %s refusal", review, reasonScopeIncomplete)
	}
}
