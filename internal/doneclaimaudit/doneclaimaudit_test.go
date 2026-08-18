package doneclaimaudit

import "testing"

func TestAuditFlagsDoneClaimWithoutShownWork(t *testing.T) {
	report := Audit([]Issue{{Number: 7, Title: "fix it", State: "CLOSED", Comments: []Comment{{Body: "Shipped and verified.", URL: "c7"}}}}, nil)
	if report.Verdict != "ACTION" || len(report.Findings) != 1 || report.Findings[0].Number != 7 {
		t.Fatalf("report = %#v", report)
	}
}

func TestAuditAcceptsTrackedDiffAndExplicitUntrackedPath(t *testing.T) {
	issues := []Issue{
		{Number: 8, Comments: []Comment{{Body: "Landed in abcdef1."}}},
		{Number: 9, Comments: []Comment{{Body: "Done.\n?? internal/newleaf/new.go"}}},
	}
	lookup := func(sha string) ([]string, bool) {
		if sha == "abcdef1" {
			return []string{"internal/x/x.go"}, true
		}
		return nil, false
	}
	report := Audit(issues, lookup)
	if report.Verdict != "OK" || len(report.Findings) != 0 || report.ClaimsScanned != 2 {
		t.Fatalf("report = %#v", report)
	}
}

func TestAuditRejectsUnknownOrEmptyCommit(t *testing.T) {
	issues := []Issue{{Number: 10, Comments: []Comment{{Body: "Fixed in deadbee and 1234567."}}}}
	report := Audit(issues, func(sha string) ([]string, bool) {
		if sha == "1234567" {
			return []string{}, true
		}
		return nil, false
	})
	if len(report.Findings) != 1 {
		t.Fatalf("findings = %#v", report.Findings)
	}
}

func TestAuditIgnoresProgressAndSortsNewestFirst(t *testing.T) {
	issues := []Issue{
		{Number: 2, Comments: []Comment{{Body: "A component completed successfully."}, {Body: "Implementation is now complete."}}},
		{Number: 11, Comments: []Comment{{Body: "Shipped."}}},
	}
	report := Audit(issues, nil)
	if report.ClaimsScanned != 2 || len(report.Findings) != 2 {
		t.Fatalf("report = %#v", report)
	}
	if report.Findings[0].Number != 11 || report.Findings[1].Number != 2 {
		t.Fatalf("order = %#v", report.Findings)
	}
}
