package devcmd

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/issuepolicy"
	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

func findingTestReceipt(issue int, digest string, verdict modelroute.CrossAuditVerdict, sev modelroute.AuditFindingSeverity, receiptDigest string, refs ...modelroute.EvidenceRef) modelroute.IssueAuditReceipt {
	return modelroute.IssueAuditReceipt{
		Schema: modelroute.CrossAuditReceiptSchema,
		Subject: modelroute.IssueAuditSubject{
			IssueNumber: issue,
			IssueURL:    "https://github.com/anthony-chaudhary/fak/issues/" + itoaFinding(issue),
			CommitSHA:   "deadbeefcafe1234",
			Digest:      digest,
		},
		Verdict:       verdict,
		Severity:      sev,
		Reason:        "the closing diff does not establish the claimed behaviour",
		ReceiptDigest: receiptDigest,
		EvidenceRefs:  refs,
	}
}

// failIfCalledGH is a gh runner that fails the test if invoked — proving the
// dry-run path never touches GitHub.
func failIfCalledGH(t *testing.T) issueCreateRunner {
	return func(args []string) (string, string, bool) {
		t.Fatalf("gh must not run in dry-run mode; got args: %v", args)
		return "", "", false
	}
}

func TestIssueFindingDryRunCandidatesAreDispatchable(t *testing.T) {
	deps := issueFindingDeps{
		receipts: []modelroute.IssueAuditReceipt{
			findingTestReceipt(4001, "sha256:aaaa1111bbbb2222", modelroute.CrossAuditRefute, modelroute.AuditSeverityHigh, "r-4001"),
		},
		receiptsSet: true,
		gh:          failIfCalledGH(t),
	}
	var stdout, stderr bytes.Buffer
	code := runIssueFindingWith(&stdout, &stderr, []string{"--parent-baseline-points", "8", "--target-envelope", "- re-audit pass rate: = 100 percent", "--witnessed-envelope", "- re-audit pass rate: = 100 percent", "--json"}, deps)
	if code != 0 {
		t.Fatalf("dry-run exit=%d stdout=%s stderr=%s", code, stdout.String(), stderr.String())
	}
	var result issueFindingResult
	if err := json.Unmarshal(stdout.Bytes(), &result); err != nil {
		t.Fatalf("decode result: %v", err)
	}
	if !result.DryRun || result.Live {
		t.Fatalf("expected dry-run, got live=%v dryrun=%v", result.Live, result.DryRun)
	}
	if len(result.Candidates) != 1 {
		t.Fatalf("want 1 generated candidate, got %d", len(result.Candidates))
	}
	// The generated candidate must be admitted by the strict, armed contract.
	review := issuepolicy.ReviewCandidate(result.Candidates[0], issuepolicy.Options{
		Live: true, DedupeChecked: true, DedupeCap: 50,
		StrictModelTier: true, StrictScale: true, StrictBornRouted: true,
	})
	if !review.OK || review.Dispatchability != issuepolicy.Dispatchable {
		t.Fatalf("generated candidate not dispatchable: verdict=%s reasons=%v", review.Dispatchability, review.Reasons)
	}
	if result.Items[0].ContractOK == nil || !*result.Items[0].ContractOK {
		t.Fatalf("adapter did not mark the CREATE item contract-ok: %+v", result.Items[0])
	}
}

// The witness: a dry-run plan emitted by the adapter is admitted by
// `fak-dev issue contract --from-plan --live`.
func TestIssueFindingPlanAdmittedByIssueContract(t *testing.T) {
	deps := issueFindingDeps{
		receipts: []modelroute.IssueAuditReceipt{
			findingTestReceipt(4100, "sha256:11112222333344aa", modelroute.CrossAuditRefute, modelroute.AuditSeverityMedium, "r-4100"),
			findingTestReceipt(4101, "sha256:5555666677778bbb", modelroute.CrossAuditRefute, modelroute.AuditSeverityCritical, "r-4101"),
		},
		receiptsSet: true,
	}
	var stdout, stderr bytes.Buffer
	if code := runIssueFindingWith(&stdout, &stderr, []string{"--parent-baseline-points", "8", "--target-envelope", "- re-audit pass rate: = 100 percent", "--witnessed-envelope", "- re-audit pass rate: = 100 percent", "--json", "--dedupe-cap", "50"}, deps); code != 0 {
		t.Fatalf("dry-run exit=%d stdout=%s stderr=%s", code, stdout.String(), stderr.String())
	}
	dir := t.TempDir()
	planPath := filepath.Join(dir, "plan.json")
	if err := os.WriteFile(planPath, stdout.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}

	var cStdout, cStderr bytes.Buffer
	code := runIssueContract(&cStdout, &cStderr, []string{
		"--from-plan", planPath, "--live", "--dedupe-checked", "--dedupe-cap", "50",
		"--strict-model-tier", "--strict-scale", "--strict-born-routed", "--json",
	})
	if code != 0 {
		t.Fatalf("fak-dev issue contract rejected the finding plan: exit=%d\nstdout=%s\nstderr=%s", code, cStdout.String(), cStderr.String())
	}
	var contract issueContractResult
	if err := json.Unmarshal(cStdout.Bytes(), &contract); err != nil {
		t.Fatalf("decode contract result: %v", err)
	}
	if !contract.OK || contract.Counts.Dispatchable != 2 || contract.Counts.Total != 2 {
		t.Fatalf("plan candidates not all dispatchable: ok=%v counts=%+v", contract.OK, contract.Counts)
	}
}

func TestIssueFindingLiveRequiresArming(t *testing.T) {
	deps := issueFindingDeps{
		receipts: []modelroute.IssueAuditReceipt{
			findingTestReceipt(4200, "sha256:9999aaaabbbbcccc", modelroute.CrossAuditRefute, modelroute.AuditSeverityHigh, "r-4200"),
		},
		receiptsSet: true,
		gh:          failIfCalledGH(t), // must not run: unarmed live is refused before any gh call
	}
	var stdout, stderr bytes.Buffer
	code := runIssueFindingWith(&stdout, &stderr, []string{"--live"}, deps)
	if code != 2 {
		t.Fatalf("unarmed --live: want exit 2, got %d (stderr=%s)", code, stderr.String())
	}
	if !strings.Contains(stderr.String(), "dedupe-cap") {
		t.Fatalf("unarmed --live error should mention dedupe-cap: %s", stderr.String())
	}
}

func TestIssueFindingLiveAppliesBoundedCreate(t *testing.T) {
	var calls [][]string
	runner := func(args []string) (string, string, bool) {
		calls = append(calls, args)
		return "https://github.com/anthony-chaudhary/fak/issues/9001", "", true
	}
	deps := issueFindingDeps{
		receipts: []modelroute.IssueAuditReceipt{
			findingTestReceipt(4300, "sha256:1212121234343434", modelroute.CrossAuditRefute, modelroute.AuditSeverityHigh, "r-4300"),
		},
		receiptsSet: true,
		gh:          runner,
	}
	var stdout, stderr bytes.Buffer
	code := runIssueFindingWith(&stdout, &stderr, []string{"--parent-baseline-points", "8", "--target-envelope", "- re-audit pass rate: = 100 percent", "--witnessed-envelope", "- re-audit pass rate: = 100 percent", "--json", "--live", "--dedupe-cap", "25", "--max-apply", "5"}, deps)
	if code != 0 {
		t.Fatalf("armed live exit=%d stderr=%s", code, stderr.String())
	}
	if len(calls) != 1 {
		t.Fatalf("want exactly 1 gh call (one CREATE), got %d: %v", len(calls), calls)
	}
	if calls[0][0] != "issue" || calls[0][1] != "create" {
		t.Fatalf("expected an `issue create` gh call, got %v", calls[0])
	}
	// The created body must carry the dedupe markers so a later run recovers state.
	joined := strings.Join(calls[0], "\n")
	if !strings.Contains(joined, "fak-crossaudit-finding-key:") || !strings.Contains(joined, "fak-crossaudit-finding-receipt:") {
		t.Fatalf("created body missing dedupe markers: %v", calls[0])
	}
}

func TestIssueFindingBlastRadiusRefusal(t *testing.T) {
	var receipts []modelroute.IssueAuditReceipt
	for i := 0; i < 4; i++ {
		receipts = append(receipts, findingTestReceipt(4400+i, "sha256:cap"+itoaFinding(i)+"0000000000000", modelroute.CrossAuditRefute, modelroute.AuditSeverityMedium, "r-cap"+itoaFinding(i)))
	}
	deps := issueFindingDeps{
		receipts:    receipts,
		receiptsSet: true,
		gh:          failIfCalledGH(t), // cap exceeded -> refuse before any gh call
	}
	var stdout, stderr bytes.Buffer
	code := runIssueFindingWith(&stdout, &stderr, []string{"--parent-baseline-points", "8", "--target-envelope", "- re-audit pass rate: = 100 percent", "--witnessed-envelope", "- re-audit pass rate: = 100 percent", "--json", "--live", "--dedupe-cap", "10", "--max-apply", "2"}, deps)
	if code != 2 {
		t.Fatalf("blast-radius over cap: want exit 2, got %d (stderr=%s)", code, stderr.String())
	}
	var result issueFindingResult
	_ = json.Unmarshal(stdout.Bytes(), &result)
	if result.Refusal == "" || !strings.Contains(result.Refusal, "max-apply") {
		t.Fatalf("expected a max-apply refusal, got %q", result.Refusal)
	}
}

func TestIssueFindingInconclusiveEscalatesNoCandidate(t *testing.T) {
	deps := issueFindingDeps{
		receipts: []modelroute.IssueAuditReceipt{
			findingTestReceipt(4500, "sha256:incon000011112222", modelroute.CrossAuditInconclusive, modelroute.AuditSeverityNone, "r-4500"),
		},
		receiptsSet: true,
		gh:          failIfCalledGH(t),
	}
	var stdout, stderr bytes.Buffer
	code := runIssueFindingWith(&stdout, &stderr, []string{"--parent-baseline-points", "8", "--target-envelope", "- re-audit pass rate: = 100 percent", "--witnessed-envelope", "- re-audit pass rate: = 100 percent", "--json"}, deps)
	if code != 0 {
		t.Fatalf("dry-run exit=%d stdout=%s stderr=%s", code, stdout.String(), stderr.String())
	}
	var result issueFindingResult
	if err := json.Unmarshal(stdout.Bytes(), &result); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(result.Candidates) != 0 {
		t.Fatalf("INCONCLUSIVE must not generate a finding candidate, got %d", len(result.Candidates))
	}
	if len(result.Escalations) != 1 || result.Escalations[0].Reason != modelroute.FindingReasonInconclusive {
		t.Fatalf("want one INCONCLUSIVE escalation, got %+v", result.Escalations)
	}
}

// existing findings parsed from --from-issues drive dedupe and reopen decisions.
func TestIssueFindingFromIssuesReopenAndDedupe(t *testing.T) {
	// Two receipts: one whose subject matches a CLOSED finding (recurrence ->
	// reopen), one whose subject matches an OPEN finding with the same recorded
	// digest (duplicate -> noop).
	recReopen := findingTestReceipt(4600, "sha256:reopen0000111122", modelroute.CrossAuditRefute, modelroute.AuditSeverityHigh, "r-reopen-new")
	recDup := findingTestReceipt(4601, "sha256:dup0000011112222a", modelroute.CrossAuditRefute, modelroute.AuditSeverityLow, "r-dup-same")

	closedFinding := findingIssueRow{
		Number: 7000,
		State:  "closed",
		Body:   "<!-- fak-crossaudit-finding-key: " + modelroute.FindingKey(recReopen.Subject) + " -->\n<!-- fak-crossaudit-finding-receipt: r-reopen-old -->\n",
	}
	openFinding := findingIssueRow{
		Number: 7001,
		State:  "open",
		Body:   "<!-- fak-crossaudit-finding-key: " + modelroute.FindingKey(recDup.Subject) + " -->\n<!-- fak-crossaudit-finding-receipt: r-dup-same -->\n",
	}
	issuesJSON, err := json.Marshal([]findingIssueRow{closedFinding, openFinding})
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	issuesPath := filepath.Join(dir, "issues.json")
	if err := os.WriteFile(issuesPath, issuesJSON, 0o644); err != nil {
		t.Fatal(err)
	}

	deps := issueFindingDeps{
		receipts:    []modelroute.IssueAuditReceipt{recReopen, recDup},
		receiptsSet: true,
		gh:          failIfCalledGH(t),
	}
	var stdout, stderr bytes.Buffer
	code := runIssueFindingWith(&stdout, &stderr, []string{"--json", "--from-issues", issuesPath}, deps)
	if code != 0 {
		t.Fatalf("dry-run exit=%d stdout=%s stderr=%s", code, stdout.String(), stderr.String())
	}
	var result issueFindingResult
	if err := json.Unmarshal(stdout.Bytes(), &result); err != nil {
		t.Fatalf("decode: %v", err)
	}
	byKey := map[string]issueFindingItem{}
	for _, it := range result.Items {
		byKey[it.Key] = it
	}
	reopen := byKey[modelroute.FindingKey(recReopen.Subject)]
	if reopen.Action != string(modelroute.FindingReopen) || reopen.TargetIssue != 7000 {
		t.Fatalf("recurrence should reopen #7000, got %+v", reopen)
	}
	dup := byKey[modelroute.FindingKey(recDup.Subject)]
	if dup.Action != string(modelroute.FindingNoop) || dup.Reason != modelroute.FindingReasonDuplicateReceipt {
		t.Fatalf("duplicate receipt should noop, got %+v", dup)
	}
}

// The backlog-flood guard the done-condition names: one subject audited twice
// in a single batch with NEW evidence (a different receipt digest) must still
// yield exactly one candidate — the first REFUTE creates, the second appends
// as UPDATE onto the same finding key, never a second issue.
func TestIssueFindingNewEvidenceSameSubjectYieldsSingleCandidate(t *testing.T) {
	first := findingTestReceipt(4800, "sha256:flood00001111aaaa", modelroute.CrossAuditRefute, modelroute.AuditSeverityHigh, "r-4800-a")
	second := findingTestReceipt(4800, "sha256:flood00001111aaaa", modelroute.CrossAuditRefute, modelroute.AuditSeverityHigh, "r-4800-b")
	deps := issueFindingDeps{
		receipts:    []modelroute.IssueAuditReceipt{first, second},
		receiptsSet: true,
		gh:          failIfCalledGH(t),
	}
	var stdout, stderr bytes.Buffer
	if code := runIssueFindingWith(&stdout, &stderr, []string{"--parent-baseline-points", "8", "--target-envelope", "- re-audit pass rate: = 100 percent", "--witnessed-envelope", "- re-audit pass rate: = 100 percent", "--json"}, deps); code != 0 {
		t.Fatalf("dry-run exit=%d stdout=%s stderr=%s", code, stdout.String(), stderr.String())
	}
	var result issueFindingResult
	if err := json.Unmarshal(stdout.Bytes(), &result); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(result.Candidates) != 1 {
		t.Fatalf("same subject with new evidence must not add a second candidate, got %d", len(result.Candidates))
	}
	if len(result.Items) != 2 {
		t.Fatalf("want 2 plan items, got %d", len(result.Items))
	}
	if result.Items[0].Action != string(modelroute.FindingCreate) {
		t.Fatalf("first refute: want CREATE, got %s", result.Items[0].Action)
	}
	if result.Items[1].Action != string(modelroute.FindingUpdate) || result.Items[1].Reason != modelroute.FindingReasonNewEvidence {
		t.Fatalf("second refute with new digest: want UPDATE/NEW_EVIDENCE, got %s/%s", result.Items[1].Action, result.Items[1].Reason)
	}
	if result.Items[0].Key != result.Items[1].Key {
		t.Fatalf("both receipts must map to one finding key: %q vs %q", result.Items[0].Key, result.Items[1].Key)
	}
}

// A filed finding body must itself re-parse to a dispatchable issue under
// `fak-dev issue contract --from-issues`, closing the loop that the adapter files
// contract-clean issues (not just contract-clean candidate structs).
func TestIssueFindingRenderedBodyReparsesDispatchable(t *testing.T) {
	receipt := findingTestReceipt(4700, "sha256:body00001111aaaa", modelroute.CrossAuditRefute, modelroute.AuditSeverityHigh, "r-4700",
		modelroute.EvidenceRef{Kind: "tests", Ref: "internal/modelroute/audit_finding_test.go"},
	)
	plan := modelroute.PlanCrossAuditFindings([]modelroute.IssueAuditReceipt{receipt}, nil)
	body := renderFindingIssueBody(plan.Items[0])

	draft := issuepolicy.IssueDraft{
		Number: 7100,
		Title:  findingIssueTitle(4700),
		Body:   body,
		Labels: []issuepolicy.IssueLabel{{Name: "crossaudit-finding"}, {Name: "class:bug"}, {Name: "priority/p2"}},
	}
	review := issuepolicy.ReviewIssueDraft(draft, issuepolicy.Options{
		Live: true, DedupeChecked: true, DedupeCap: 50,
		StrictModelTier: true, StrictScale: true, StrictBornRouted: true,
	})
	if !review.OK || review.Dispatchability != issuepolicy.Dispatchable {
		t.Fatalf("filed finding body not dispatchable on re-parse: verdict=%s reasons=%v missing=%v sections=%v",
			review.Dispatchability, review.Reasons, review.MissingFields, review.MissingSections)
	}
}
