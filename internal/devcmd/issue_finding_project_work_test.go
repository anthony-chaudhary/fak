package devcmd

import (
	"github.com/anthony-chaudhary/fak/internal/modelroute"
	"strings"
	"testing"
)

func TestFindingProjectWorkProductionAndDemo(t *testing.T) {
	item := modelroute.FindingPlanItem{Key: "k", AuditedIssue: 36, ReceiptDigest: "abc", Verdict: modelroute.CrossAuditRefute, Severity: modelroute.AuditSeverityHigh, Subject: modelroute.IssueAuditSubject{IssueNumber: 36, CommitSHA: "abcdef123456"}, Detail: "toy path only"}
	prod := findingProjectWork{Baseline: 20, Standard: "production", TargetEnvelope: "- concurrent users: 10 users", WitnessedEnvelope: "- concurrent users: 10 users"}
	c, err := buildFindingCandidateWithProjectWork(item, "crossaudit", 50, prod)
	if err != nil {
		t.Fatal(err)
	}
	if c.ScopeContribution != "Contribution: 3/20 points" || c.CompletionStandard != "production" {
		t.Fatalf("candidate=%+v", c)
	}
	body, err := renderFindingIssueBodyWithProjectWork(item, findingProjectWork{Baseline: 20, Standard: "demo"})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(body, "## Completion standard\ndemo") || strings.Contains(body, "## Completion standard\nproduction") {
		t.Fatalf("demo hidden:\n%s", body)
	}
}
func TestFindingProjectWorkRefusesUnknownBaseline(t *testing.T) {
	item := modelroute.FindingPlanItem{AuditedIssue: 36}
	if _, err := buildFindingCandidateWithProjectWork(item, "crossaudit", 1, findingProjectWork{}); err == nil {
		t.Fatal("missing baseline accepted")
	}
}
