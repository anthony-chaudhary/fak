package mlpscore

// kernel_test.go pins the shared-kernel migration: that this card's numbers come off
// pkg/scorecard's fold rather than a look-alike written here, that the migration did not
// move any number the card already published, and that the --compare / --markdown surfaces
// do the real thing (diff a prior payload, render the family page) rather than return prose.

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/pkg/scorecard"
)

// TestPayloadRidesTheSharedKernel is the migration witness: the folded corpus is the shared
// control-pane envelope, with the kernel's own value/grade/pressure keys and its one grade
// table -- not a hand-rolled copy that could drift from the rest of the family.
func TestPayloadRidesTheSharedKernel(t *testing.T) {
	s := Grade(memorySnapshot{}, testOpts())
	for _, key := range []string{"value", "value_unit", "score", "legacy_score", "grade", "pressure", "slack", DebtKey} {
		if _, ok := s.Corpus[key]; !ok {
			t.Fatalf("corpus is missing the kernel key %q: %+v", key, s.Corpus)
		}
	}
	composite, _ := s.Corpus["score"].(float64)
	if want := scorecard.GradeStd(composite); s.Corpus["grade"] != want {
		t.Fatalf("grade %v is not the kernel table's %q for score %v", s.Corpus["grade"], want, composite)
	}
	if len(s.KPIs) != len(criteria) {
		t.Fatalf("kpis = %d, want one per criterion (%d)", len(s.KPIs), len(criteria))
	}
	raw, err := json.Marshal(KernelPayload(s))
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(raw), `"defects":null`) || strings.Contains(string(raw), `"soft":null`) {
		t.Fatalf("kernel KPI lists must marshal as [], got %s", raw)
	}
}

// TestKernelDebtStaysOnePerUnwitnessedCriterion pins the number the migration had to
// preserve: mlp_debt counts CRITERIA still to witness, never individual missing claims.
func TestKernelDebtStaysOnePerUnwitnessedCriterion(t *testing.T) {
	s := Grade(memorySnapshot{}, testOpts())
	if s.Debt != s.Total-s.Witnessed || s.Debt != len(criteria) {
		t.Fatalf("debt %d, witnessed %d, total %d", s.Debt, s.Witnessed, s.Total)
	}
	defects := 0
	for _, k := range s.KPIs {
		if len(k.Defects) > 1 {
			t.Fatalf("kpi %s emitted %d defects; one criterion is one defect: %+v", k.Key, len(k.Defects), k.Defects)
		}
		defects += len(k.Defects)
		if len(k.Soft) != 0 {
			t.Fatalf("kpi %s emitted advisory soft entries this card does not define: %+v", k.Key, k.Soft)
		}
	}
	if defects != s.Debt {
		t.Fatalf("kernel debt %d is not the defect count %d", s.Debt, defects)
	}
	if got := scorecard.IntValue(s.Corpus[DebtKey]); got != s.Debt {
		t.Fatalf("corpus.%s %d disagrees with the envelope debt %d", DebtKey, got, s.Debt)
	}
	// Every defect must name the criterion and the witness that would retire it.
	for i, k := range s.KPIs {
		if len(k.Defects) == 0 {
			continue
		}
		if !strings.Contains(k.Defects[0], criteria[i].workstream) || !strings.Contains(k.Defects[0], WitnessDir+"/") {
			t.Fatalf("defect does not name the workstream and its witness: %q", k.Defects[0])
		}
	}

	lovable := Grade(completeSnapshot(t), testOpts())
	if lovable.Debt != 0 || !lovable.OK || scorecard.IntValue(lovable.Corpus[DebtKey]) != 0 {
		t.Fatalf("a fully witnessed card must fold to zero debt: %+v", lovable.Corpus)
	}
	if lovable.Corpus["grade"] != "A" {
		t.Fatalf("zero-debt grade = %v", lovable.Corpus["grade"])
	}
}

// TestEnvelopeIsStampedFromTheFold proves the card computes each number once: the published
// envelope fields are read back out of the same folded corpus, so they cannot disagree.
func TestEnvelopeIsStampedFromTheFold(t *testing.T) {
	for _, tc := range []struct {
		name     string
		snapshot memorySnapshot
	}{
		{"not-yet", memorySnapshot{}},
		{"lovable", completeSnapshot(t)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := Grade(tc.snapshot, testOpts())
			p := fold(s.Criteria)
			if s.OK != p.OK || s.Verdict != p.Verdict || s.Finding != p.Finding {
				t.Fatalf("envelope head drifted from the fold: %+v vs %+v", s, p)
			}
			if s.Reason != p.Reason || s.NextAction != p.NextAction {
				t.Fatalf("envelope prose drifted from the fold: %q/%q", s.Reason, s.NextAction)
			}
			if s.Witnessed != scorecard.IntValue(p.Corpus["witnessed"]) || s.Total != scorecard.IntValue(p.Corpus["total"]) {
				t.Fatalf("counts drifted from the fold: %+v", p.Corpus)
			}
			if s.Lovable != scorecard.True(p.Corpus["lovable"]) || s.MLPVerdict != scorecard.StringValue(p.Corpus["mlp_verdict"]) {
				t.Fatalf("verdict drifted from the fold: %v/%q vs %+v", s.Lovable, s.MLPVerdict, p.Corpus)
			}
			// mlp_verdict stays the closed two-word vocabulary the milestone report projects.
			if s.MLPVerdict != VerdictLovable && s.MLPVerdict != VerdictNotYet {
				t.Fatalf("mlp_verdict %q is outside the closed vocabulary", s.MLPVerdict)
			}
			if s.Lovable != (s.Debt == 0) {
				t.Fatalf("lovable %v disagrees with debt %d", s.Lovable, s.Debt)
			}
		})
	}
}

// TestKernelPayloadRePresentsTheGradedScore pins that --compare and any control-pane reader
// see exactly the payload Grade already folded, not a second derivation of it.
func TestKernelPayloadRePresentsTheGradedScore(t *testing.T) {
	s := Grade(memorySnapshot{}, testOpts())
	p := KernelPayload(s)
	if p.Schema != Schema || p.OK != s.OK || p.Verdict != s.Verdict || p.Finding != s.Finding {
		t.Fatalf("payload head = %+v, score head = %+v", p, s)
	}
	if p.Reason != s.Reason || p.NextAction != s.NextAction || p.Workspace != s.Workspace {
		t.Fatalf("payload body drifted: %+v", p)
	}
	if scorecard.IntValue(p.Corpus[DebtKey]) != s.Debt || len(p.KPIs) != len(s.KPIs) {
		t.Fatalf("payload corpus/kpis drifted: %+v", p.Corpus)
	}
	if got := scorecard.Render(p, DebtKey); !strings.Contains(got, DebtKey) {
		t.Fatalf("shared renderer does not name the debt key:\n%s", got)
	}
}

// TestCompareReadsAPriorPayloadAndProvesTheDirection is the --compare surface doing the real
// thing the 16 conforming cards do: diffing this fold against a prior --json run's numbers.
func TestCompareReadsAPriorPayloadAndProvesTheDirection(t *testing.T) {
	lovable := KernelPayload(Grade(completeSnapshot(t), testOpts()))
	prior := map[string]any{"corpus": map[string]any{DebtKey: 5, "pressure": 500.0}}
	line := scorecard.Compare(lovable, prior, DebtKey)
	if !strings.Contains(line, "compare: "+DebtKey+" 5 -> 0 (improved by 5)") {
		t.Fatalf("compare line missing the debt drop:\n%s", line)
	}

	// A regression must read as a regression, not as prose.
	notYet := KernelPayload(Grade(memorySnapshot{}, testOpts()))
	clean := map[string]any{"corpus": map[string]any{DebtKey: 0, "pressure": 0.0}}
	if got := scorecard.Compare(notYet, clean, DebtKey); !strings.Contains(got, "0 -> 5") {
		t.Fatalf("compare line does not report the regression:\n%s", got)
	}
}

// TestMarkdownRendersTheFamilyPageAndTheKernelHeadline covers both markdown surfaces: the
// shared kernel page body (the family standard) and this card's own rollup, which now carries
// the folded grade/value/debt triple straight off the corpus.
func TestMarkdownRendersTheFamilyPageAndTheKernelHeadline(t *testing.T) {
	s := Grade(memorySnapshot{}, testOpts())
	page := scorecard.Markdown(KernelPayload(s), MarkdownDoc(s))
	for _, want := range []string{
		"---\ntitle: ",
		"# MLP first-lovable-cut scorecard",
		"| KPI | value | legacy score | detail |",
		DebtKey,
		"0/5 criteria witnessed",
	} {
		if !strings.Contains(page, want) {
			t.Fatalf("kernel markdown missing %q:\n%s", want, page)
		}
	}
	if scorecard.Markdown(KernelPayload(s), MarkdownDoc(s)) != page {
		t.Fatal("kernel markdown render is not deterministic")
	}

	rollup := RenderMarkdown(s)
	if !strings.Contains(rollup, "**Grade F** - value 0 - "+DebtKey+" **5**") {
		t.Fatalf("rollup is missing the folded headline:\n%s", rollup)
	}
}
