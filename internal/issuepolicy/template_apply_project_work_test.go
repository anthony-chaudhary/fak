package issuepolicy

import "testing"

func TestApplyTemplateRepairPreservesProjectWorkMetadata(t *testing.T) {
	d := corruptGenerationDraft()
	d.Body += "\n\n## Parent context\nParent: #1625\n\n## Work estimate\nEstimate: 3 points\n\n## Overall completion contribution\nContribution: 3/20 points\n\n## Completion standard\ndemo"
	a, ok := ApplyTemplateRepair(d)
	if !ok || !a.Safe {
		t.Fatalf("apply=%+v ok=%v", a, ok)
	}
	before := projectWork(CandidateFromIssueDraft(d))
	after := projectWork(CandidateFromIssueDraft(IssueDraft{Body: a.NewBody}))
	if before.Status != ProjectWorkValid || after.Status != ProjectWorkValid || before.EstimatePoints != after.EstimatePoints || before.Contribution != after.Contribution || before.ParentBaseline != after.ParentBaseline || after.CompletionStandard != "demo" {
		t.Fatalf("before=%+v after=%+v", before, after)
	}
}
