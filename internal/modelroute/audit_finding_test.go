package modelroute

import "testing"

// refuteReceipt builds a minimal REFUTE receipt for planner tests. The planner
// reads only the subject, verdict, severity, receipt digest, and reason — it
// never calls Verify — so a hand-built receipt is sufficient and keeps the test
// free of the heavy identity/independence machinery.
func findingReceipt(issue int, digest string, verdict CrossAuditVerdict, sev AuditFindingSeverity, receiptDigest string) IssueAuditReceipt {
	return IssueAuditReceipt{
		Schema: CrossAuditReceiptSchema,
		Subject: IssueAuditSubject{
			IssueNumber: issue,
			IssueURL:    "https://example.test/issues/" + itoa(issue),
			CommitSHA:   "c0ffee" + itoa(issue),
			Digest:      digest,
		},
		Verdict:       verdict,
		Severity:      sev,
		Reason:        string(verdict) + " reason",
		ReceiptDigest: receiptDigest,
	}
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	neg := n < 0
	if neg {
		n = -n
	}
	var b [20]byte
	i := len(b)
	for n > 0 {
		i--
		b[i] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		i--
		b[i] = '-'
	}
	return string(b[i:])
}

func onlyItem(t *testing.T, plan FindingPlan) FindingPlanItem {
	t.Helper()
	if len(plan.Items) != 1 {
		t.Fatalf("want exactly 1 plan item, got %d", len(plan.Items))
	}
	return plan.Items[0]
}

func TestFindingKeyStableAndValidGrammar(t *testing.T) {
	subject := IssueAuditSubject{IssueNumber: 3857, Digest: "sha256:abcdef0123456789aabbccddeeff0011"}
	got := FindingKey(subject)
	want := "crossaudit/finding/3857/abcdef0123456789"
	if got != want {
		t.Fatalf("FindingKey = %q, want %q", got, want)
	}
	// Same subject -> same key (dedupe stability); different digest -> new key.
	if FindingKey(subject) != got {
		t.Fatalf("FindingKey is not stable for identical subjects")
	}
	other := IssueAuditSubject{IssueNumber: 3857, Digest: "sha256:1111111111111111ffff"}
	if FindingKey(other) == got {
		t.Fatalf("FindingKey collided across distinct subject digests")
	}
}

func TestPlanNewRefuteCreates(t *testing.T) {
	receipts := []IssueAuditReceipt{findingReceipt(10, "d10", CrossAuditRefute, AuditSeverityMedium, "r-10-a")}
	item := onlyItem(t, PlanCrossAuditFindings(receipts, nil))
	if item.Action != FindingCreate || item.Reason != FindingReasonNewRefute {
		t.Fatalf("want CREATE/NEW_REFUTE, got %s/%s", item.Action, item.Reason)
	}
	if item.Severity != AuditSeverityMedium {
		t.Fatalf("severity not preserved: %s", item.Severity)
	}
}

func TestPlanDuplicateReceiptsUpdateOneFinding(t *testing.T) {
	// Two identical REFUTE receipts in one batch: the first opens a finding, the
	// second dedupes to NOOP rather than filing a second issue.
	r := findingReceipt(11, "d11", CrossAuditRefute, AuditSeverityHigh, "r-11-same")
	plan := PlanCrossAuditFindings([]IssueAuditReceipt{r, r}, nil)
	if len(plan.Items) != 2 {
		t.Fatalf("want 2 items, got %d", len(plan.Items))
	}
	if plan.Items[0].Action != FindingCreate {
		t.Fatalf("first identical receipt: want CREATE, got %s", plan.Items[0].Action)
	}
	if plan.Items[1].Action != FindingNoop || plan.Items[1].Reason != FindingReasonDuplicateReceipt {
		t.Fatalf("second identical receipt: want NOOP/DUPLICATE_RECEIPT, got %s/%s", plan.Items[1].Action, plan.Items[1].Reason)
	}
	if plan.Counts[FindingCreate] != 1 || plan.Counts[FindingNoop] != 1 {
		t.Fatalf("counts wrong: %+v", plan.Counts)
	}
}

func TestPlanDuplicateAgainstExistingOpenFindingNoops(t *testing.T) {
	r := findingReceipt(12, "d12", CrossAuditRefute, AuditSeverityLow, "r-12")
	existing := []ExistingFinding{{
		Key:           FindingKey(r.Subject),
		IssueNumber:   500,
		State:         FindingStateOpen,
		ReceiptDigest: "r-12",
	}}
	item := onlyItem(t, PlanCrossAuditFindings([]IssueAuditReceipt{r}, existing))
	if item.Action != FindingNoop || item.Reason != FindingReasonDuplicateReceipt {
		t.Fatalf("want NOOP/DUPLICATE_RECEIPT, got %s/%s", item.Action, item.Reason)
	}
	if item.TargetIssue != 500 {
		t.Fatalf("want target issue 500, got %d", item.TargetIssue)
	}
}

func TestPlanNewEvidenceUpdatesExistingOpenFinding(t *testing.T) {
	r := findingReceipt(13, "d13", CrossAuditRefute, AuditSeverityHigh, "r-13-new")
	existing := []ExistingFinding{{
		Key:           FindingKey(r.Subject),
		IssueNumber:   501,
		State:         FindingStateOpen,
		ReceiptDigest: "r-13-old",
	}}
	item := onlyItem(t, PlanCrossAuditFindings([]IssueAuditReceipt{r}, existing))
	if item.Action != FindingUpdate || item.Reason != FindingReasonNewEvidence {
		t.Fatalf("want UPDATE/NEW_EVIDENCE, got %s/%s", item.Action, item.Reason)
	}
	if item.TargetIssue != 501 {
		t.Fatalf("want target issue 501, got %d", item.TargetIssue)
	}
}

func TestPlanResolvedRecurrenceReopens(t *testing.T) {
	r := findingReceipt(14, "d14", CrossAuditRefute, AuditSeverityCritical, "r-14")
	existing := []ExistingFinding{{
		Key:           FindingKey(r.Subject),
		IssueNumber:   502,
		State:         FindingStateClosed,
		ReceiptDigest: "r-14-prev",
	}}
	plan := PlanCrossAuditFindings([]IssueAuditReceipt{r, r}, existing)
	if plan.Items[0].Action != FindingReopen || plan.Items[0].Reason != FindingReasonResolvedRecurrence {
		t.Fatalf("want REOPEN/RESOLVED_RECURRENCE, got %s/%s", plan.Items[0].Action, plan.Items[0].Reason)
	}
	if plan.Items[0].TargetIssue != 502 {
		t.Fatalf("want target issue 502, got %d", plan.Items[0].TargetIssue)
	}
	// After a reopen, a following identical receipt must not reopen again.
	if plan.Items[1].Action != FindingNoop {
		t.Fatalf("recurrence then duplicate: want NOOP, got %s", plan.Items[1].Action)
	}
}

func TestPlanInconclusiveEscalatesNotCreate(t *testing.T) {
	// The core confusion-risk guard: model-only uncertainty must escalate, never
	// file a corruption finding.
	r := findingReceipt(15, "d15", CrossAuditInconclusive, AuditSeverityNone, "r-15")
	item := onlyItem(t, PlanCrossAuditFindings([]IssueAuditReceipt{r}, nil))
	if item.Action != FindingEscalate || item.Reason != FindingReasonInconclusive {
		t.Fatalf("want ESCALATE/INCONCLUSIVE_ESCALATION, got %s/%s", item.Action, item.Reason)
	}
	if item.Action == FindingCreate {
		t.Fatalf("INCONCLUSIVE must never CREATE a finding")
	}
}

func TestPlanUnavailableEscalates(t *testing.T) {
	r := findingReceipt(16, "d16", CrossAuditUnavailable, AuditSeverityNone, "r-16")
	item := onlyItem(t, PlanCrossAuditFindings([]IssueAuditReceipt{r}, nil))
	if item.Action != FindingEscalate || item.Reason != FindingReasonUnavailable {
		t.Fatalf("want ESCALATE/AUDITOR_UNAVAILABLE, got %s/%s", item.Action, item.Reason)
	}
}

func TestPlanCleanPassNoops(t *testing.T) {
	r := findingReceipt(17, "d17", CrossAuditPass, AuditSeverityNone, "r-17")
	item := onlyItem(t, PlanCrossAuditFindings([]IssueAuditReceipt{r}, nil))
	if item.Action != FindingNoop || item.Reason != FindingReasonCleanPass {
		t.Fatalf("want NOOP/CLEAN_PASS, got %s/%s", item.Action, item.Reason)
	}
}

func TestPlanPassOnOpenFindingCommentsDoesNotClose(t *testing.T) {
	// An independent re-audit PASS records progress but does NOT close: closure
	// still requires a witnessed fix, so the planner only comments.
	r := findingReceipt(18, "d18", CrossAuditPass, AuditSeverityNone, "r-18")
	existing := []ExistingFinding{{
		Key:         FindingKey(r.Subject),
		IssueNumber: 503,
		State:       FindingStateOpen,
	}}
	item := onlyItem(t, PlanCrossAuditFindings([]IssueAuditReceipt{r}, existing))
	if item.Action != FindingComment || item.Reason != FindingReasonReauditPassed {
		t.Fatalf("want COMMENT/REAUDIT_PASSED, got %s/%s", item.Action, item.Reason)
	}
	if item.Action == FindingReopen {
		t.Fatalf("a PASS re-audit must never reopen or close")
	}
}

func TestPlanHighSeverityCreatesWithSeverity(t *testing.T) {
	for _, sev := range []AuditFindingSeverity{AuditSeverityHigh, AuditSeverityCritical} {
		r := findingReceipt(19, "d19-"+string(sev), CrossAuditRefute, sev, "r-19-"+string(sev))
		item := onlyItem(t, PlanCrossAuditFindings([]IssueAuditReceipt{r}, nil))
		if item.Action != FindingCreate {
			t.Fatalf("severity %s: want CREATE, got %s", sev, item.Action)
		}
		if item.Severity != sev {
			t.Fatalf("severity %s not carried onto the plan item (got %s)", sev, item.Severity)
		}
	}
}

func TestPlanActionCountsOrdered(t *testing.T) {
	plan := PlanCrossAuditFindings([]IssueAuditReceipt{
		findingReceipt(20, "d20", CrossAuditRefute, AuditSeverityHigh, "r-20"),
		findingReceipt(21, "d21", CrossAuditInconclusive, AuditSeverityNone, "r-21"),
	}, nil)
	counts := FindingPlanActionCounts(plan)
	if counts[0].Action != FindingCreate {
		t.Fatalf("first count row should be CREATE, got %s", counts[0].Action)
	}
	byAction := map[FindingAction]int{}
	for _, c := range counts {
		byAction[c.Action] = c.Count
	}
	if byAction[FindingCreate] != 1 || byAction[FindingEscalate] != 1 {
		t.Fatalf("counts wrong: %+v", byAction)
	}
}
