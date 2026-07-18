package shipgate

// crossaudit_closure_test.go is the #3860 failing-before/passing-after witness. Each
// bad-receipt class is a high-risk closure the PRE-gate world (legacyUngatedAllowed)
// would have let land; AdjudicateClosure under enforcement now BLOCKS it. It also pins
// the done-condition properties: a calibrated independent PASS cannot override a
// structural deny, the low-risk path is unchanged, break-glass is explicit/audited and
// cannot override structure, and staged enablement never blocks while prerequisites are
// unmet.

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

// legacyUngatedAllowed models the behavior BEFORE this gate: no closure class required
// an admitted independent audit receipt, so every closure was allowed. It makes the
// "failing-before" half of each case concrete.
func legacyUngatedAllowed(ClosureRequest) bool { return true }

const testNowNano int64 = 1_700_000_000_000_000_000

// enforcePolicy is the calibrated policy with the dogfood prerequisite satisfied, so
// enforcement is ON — the future state in which the gate fails closed.
func enforcePolicy() CrossAuditPolicy {
	p := DefaultCrossAuditPolicy()
	p.Prereqs.DogfoodGreen = true
	return p
}

// validReceipt is a calibrated, independent, fresh PASS bound to the closure subject.
func validReceipt() AuditReceiptView {
	return AuditReceiptView{
		Present:             true,
		SubjectDigest:       "sha256:subject-aaa",
		Verdict:             AuditPass,
		AuditorFamily:       "claude",
		AuthorFamily:        "gpt",
		CalibrationVersion:  "issue-resolution-audit/v2",
		CompletedAtUnixNano: testNowNano - 1,
	}
}

func baseHighRiskReq() ClosureRequest {
	return ClosureRequest{
		Issue:         3860,
		Risk:          RiskHigh,
		SubjectDigest: "sha256:subject-aaa",
		Receipt:       validReceipt(),
		NowUnixNano:   testNowNano,
	}
}

// TestClosureValidReceiptAllows: a calibrated independent fresh PASS opens the gate.
func TestClosureValidReceiptAllows(t *testing.T) {
	d := AdjudicateClosure(baseHighRiskReq(), enforcePolicy())
	if !d.Enforced || !d.Allowed || d.Reason != ReasonAuditPass {
		t.Fatalf("valid calibrated independent PASS should allow under enforcement: %+v", d)
	}
}

// TestClosureBadReceiptClassesBlock is the core done-condition table: every bad-receipt
// class blocks a high-risk closure under enforcement, while the legacy ungated path
// would have allowed it (failing-before/passing-after).
func TestClosureBadReceiptClassesBlock(t *testing.T) {
	pol := enforcePolicy()
	cases := []struct {
		name   string
		mutate func(*ClosureRequest)
		want   ClosureReason
	}{
		{"missing", func(r *ClosureRequest) { r.Receipt.Present = false }, ReasonReceiptMissing},
		{"refute", func(r *ClosureRequest) { r.Receipt.Verdict = AuditRefute }, ReasonReceiptNonPass},
		{"inconclusive", func(r *ClosureRequest) { r.Receipt.Verdict = AuditInconclusive }, ReasonReceiptNonPass},
		{"unavailable", func(r *ClosureRequest) { r.Receipt.Verdict = AuditUnavailable }, ReasonReceiptNonPass},
		{"digest-mismatch", func(r *ClosureRequest) { r.Receipt.SubjectDigest = "sha256:other" }, ReasonSubjectMismatch},
		{"uncalibrated-auditor", func(r *ClosureRequest) { r.Receipt.AuditorFamily = "mistral" }, ReasonAuditorUncalibrated},
		{"calibration-stale", func(r *ClosureRequest) { r.Receipt.CalibrationVersion = "issue-resolution-audit/v1" }, ReasonCalibrationMismatch},
		{"same-family", func(r *ClosureRequest) { r.Receipt.AuditorFamily = "gpt" }, ReasonNotIndependent}, // author is gpt
		{"stale", func(r *ClosureRequest) { r.Receipt.CompletedAtUnixNano = testNowNano - pol.MaxReceiptAgeNanos - 1 }, ReasonReceiptStale},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			req := baseHighRiskReq()
			tc.mutate(&req)
			if !legacyUngatedAllowed(req) {
				t.Fatalf("precondition: the legacy ungated path allowed every closure")
			}
			d := AdjudicateClosure(req, pol)
			if d.Allowed || !d.Enforced {
				t.Fatalf("%s: enforcement must BLOCK this closure; got %+v", tc.name, d)
			}
			if d.Reason != tc.want {
				t.Fatalf("%s: reason = %q, want %q (%s)", tc.name, d.Reason, tc.want, d.Detail)
			}
		})
	}
}

// TestClosureStructuralDenyBeatsPass: a calibrated independent PASS — and even a valid
// break-glass — can never flip a structural deny to allowed.
func TestClosureStructuralDenyBeatsPass(t *testing.T) {
	pol := enforcePolicy()

	req := baseHighRiskReq() // a perfectly valid PASS receipt
	req.StructuralDeny = true
	d := AdjudicateClosure(req, pol)
	if d.Allowed || d.Reason != ReasonStructuralDeny {
		t.Fatalf("a valid PASS must not override a structural deny: %+v", d)
	}

	req.BreakGlass = &BreakGlass{Operator: "op", Reason: "emergency", LedgerRef: "ledger:1", ExpiresAtUnixN: testNowNano + 1}
	d = AdjudicateClosure(req, pol)
	if d.Allowed || d.Reason != ReasonStructuralDeny {
		t.Fatalf("break-glass must not override a structural deny: %+v", d)
	}
}

// TestClosureLowRiskUnchanged: the ordinary low-risk path is never newly gated, even
// with no receipt at all.
func TestClosureLowRiskUnchanged(t *testing.T) {
	req := baseHighRiskReq()
	req.Risk = RiskLow
	req.Receipt = AuditReceiptView{Present: false}
	d := AdjudicateClosure(req, enforcePolicy())
	if !d.Allowed || d.Reason != ReasonLowRiskExempt {
		t.Fatalf("low-risk closure must remain ungated: %+v", d)
	}
}

// TestClosureBreakGlassAuditedOverride: an explicit, audited, unexpired break-glass
// waives a missing receipt; a blank or expired one does not.
func TestClosureBreakGlassAuditedOverride(t *testing.T) {
	pol := enforcePolicy()

	req := baseHighRiskReq()
	req.Receipt.Present = false
	req.BreakGlass = &BreakGlass{Operator: "alice", Reason: "sev1 rollback", LedgerRef: "ledger:bg-7", ExpiresAtUnixN: testNowNano + 1}
	if d := AdjudicateClosure(req, pol); !d.Allowed || d.Reason != ReasonBreakGlass {
		t.Fatalf("valid audited break-glass should override a missing receipt: %+v", d)
	}

	// Missing audit fields => invalid => the underlying block reason stands.
	for _, bad := range []*BreakGlass{
		{Operator: "", Reason: "r", LedgerRef: "l", ExpiresAtUnixN: testNowNano + 1},
		{Operator: "o", Reason: "r", LedgerRef: "", ExpiresAtUnixN: testNowNano + 1},
		{Operator: "o", Reason: "r", LedgerRef: "l", ExpiresAtUnixN: testNowNano - 1}, // expired
	} {
		req.BreakGlass = bad
		if d := AdjudicateClosure(req, pol); d.Allowed || d.Reason != ReasonReceiptMissing {
			t.Fatalf("an invalid break-glass must not override; got %+v (bg=%+v)", d, bad)
		}
	}
}

// TestClosureStagedEnablementDryRun: with prerequisites unmet (the shipped default, dark
// dogfood loop), the gate never blocks — it reports what enforcement WOULD do.
func TestClosureStagedEnablementDryRun(t *testing.T) {
	pol := DefaultCrossAuditPolicy() // DogfoodGreen == false
	if pol.Prereqs.Met() {
		t.Fatalf("default policy prerequisites must be UNMET while the dogfood loop is dark")
	}

	bad := baseHighRiskReq()
	bad.Receipt.Present = false
	d := AdjudicateClosure(bad, pol)
	if d.Enforced || !d.Allowed || !d.WouldBlock || d.Reason != ReasonPrereqsDryRun {
		t.Fatalf("dry-run must allow but report would-block on a bad receipt: %+v", d)
	}

	good := baseHighRiskReq()
	d = AdjudicateClosure(good, pol)
	if d.Enforced || !d.Allowed || d.WouldBlock {
		t.Fatalf("dry-run on a valid receipt must allow with would_block=false: %+v", d)
	}
}

// TestPrerequisitesMet pins the staged-enablement gate truth table.
func TestPrerequisitesMet(t *testing.T) {
	cases := []struct {
		name string
		p    Prerequisites
		want bool
	}{
		{"all satisfied", Prerequisites{CalibratedAuditorFamilies: 2, MinIndependent: 2, DogfoodGreen: true}, true},
		{"single family never independent", Prerequisites{CalibratedAuditorFamilies: 3, MinIndependent: 1, DogfoodGreen: true}, false},
		{"too few calibrated", Prerequisites{CalibratedAuditorFamilies: 1, MinIndependent: 2, DogfoodGreen: true}, false},
		{"dogfood not green", Prerequisites{CalibratedAuditorFamilies: 2, MinIndependent: 2, DogfoodGreen: false}, false},
	}
	for _, tc := range cases {
		if got := tc.p.Met(); got != tc.want {
			t.Errorf("%s: Met()=%v want %v", tc.name, got, tc.want)
		}
	}
}

// TestAdoptionReportReflectsMeasuredEvidence: the report recommends dry-run while the
// dogfood loop is dark, and enforce once it is green.
func TestAdoptionReportReflectsMeasuredEvidence(t *testing.T) {
	rep := CrossAuditAdoptionReport(DefaultCrossAuditPolicy())
	if rep.PrereqsMet || rep.RecommendedStage != StageDryRun {
		t.Fatalf("dark-dogfood policy must recommend dry-run: %+v", rep)
	}
	if !rep.Calibration.Met || rep.Calibration.FamilyCount != 2 {
		t.Fatalf("calibration prerequisite (2 independent families) should be satisfied: %+v", rep.Calibration)
	}
	if rep.Dogfood.Green {
		t.Fatalf("dogfood must read not-green: %+v", rep.Dogfood)
	}

	green := enforcePolicy()
	if rg := CrossAuditAdoptionReport(green); !rg.PrereqsMet || rg.RecommendedStage != StageEnforce {
		t.Fatalf("green-dogfood policy must recommend enforce: %+v", rg)
	}
}

// TestCommittedAdoptionReportMatchesCode: the committed adoption artifact must equal the
// report the code generates, so the witness can never silently drift from the gate.
func TestCommittedAdoptionReportMatchesCode(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("testdata", "crossaudit_adoption_report.json"))
	if err != nil {
		t.Fatalf("read committed adoption report: %v", err)
	}
	var committed AdoptionReport
	if err := json.Unmarshal(raw, &committed); err != nil {
		t.Fatalf("decode committed adoption report: %v", err)
	}
	want := CrossAuditAdoptionReport(DefaultCrossAuditPolicy())
	if !reflect.DeepEqual(committed, want) {
		t.Fatalf("committed adoption report drifted from code:\n committed=%+v\n code=%+v", committed, want)
	}
}
