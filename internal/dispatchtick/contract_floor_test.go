package dispatchtick

import (
	"strings"
	"testing"
)

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

func TestDispatchContractAcceptsCanonicalScopeAliases(t *testing.T) {
	issue := routerIssue(14, "gateway: canonical scope aliases", nil, canonicalScopeIssueBody())
	if review := reviewDispatchContract(issue); !review.OK {
		t.Fatalf("canonical scope review = %+v, want accepted", review)
	}
}

func TestDispatchContractRejectsCanonicalScopeAliasHalf(t *testing.T) {
	for _, heading := range []string{"## Core through-line", "## Gold-plating boundary"} {
		t.Run(strings.TrimPrefix(heading, "## "), func(t *testing.T) {
			body := strings.Replace(canonicalScopeIssueBody(), heading, "## Unrecognized scope heading", 1)
			review := reviewDispatchContract(routerIssue(15, "gateway: incomplete canonical scope", nil, body))
			if review.OK || firstIssueContractReason(review) != reasonScopeIncomplete {
				t.Fatalf("review without %q = %+v, want %s refusal", heading, review, reasonScopeIncomplete)
			}
		})
	}
}

func TestDispatchContractKeepsCompatibleScopeAliases(t *testing.T) {
	legacy := scopedGatewayIssueBody("2")
	scope := strings.NewReplacer(
		"## In scope", "## Scope",
		"## Out of scope", "## Compatibility boundary",
	).Replace(legacy)
	for name, body := range map[string]string{"scope": scope, "in_scope_out_of_scope": legacy} {
		t.Run(name, func(t *testing.T) {
			if review := reviewDispatchContract(routerIssue(16, "gateway: compatible scope aliases", nil, body)); !review.OK {
				t.Fatalf("compatible scope review = %+v, want accepted", review)
			}
		})
	}
}

func canonicalScopeIssueBody() string {
	return strings.NewReplacer(
		"## In scope", "## Core through-line",
		"## Out of scope", "## Gold-plating boundary",
	).Replace(scopedGatewayIssueBody("2"))
}
