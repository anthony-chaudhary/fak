package quality

import (
	"strings"
	"testing"
)

// citeTestEvidence is the hermetic evidence corpus for the citation tests:
// three declared sources in the id-plus-link/description entry form.
var citeTestEvidence = []string{
	"1: https://ops/rollup-w28",
	"S3 weekly support-queue export",
	"[E-12]: incident postmortem",
}

const (
	// citeFaithfulReport cites only declared sources, one of them by a
	// case-variant marker ([s3] must resolve to evidence "S3").
	citeFaithfulReport = "Throughput increased 12% week over week [1]. " +
		"Median latency held flat at 250ms [s3]. The outage was closed [E-12]."
	// citeDanglingReport adds a marker no evidence entry declares — the
	// fabricated source.
	citeDanglingReport = citeFaithfulReport + " Churn dropped 40% [9]."
)

// citationCase builds a valid case whose Rubric carries the allowed evidence
// ids the citation-validity oracle resolves report markers against.
func citationCase(evidence []string, minScore float64) QualityCase {
	return QualityCase{
		Schema:    CaseSchema,
		ID:        "citation-validity-exec-report",
		Version:   1,
		Prompt:    "Summarize the weekly ops evidence with citations for the executive rollup.",
		Params:    SamplingParams{Temperature: 0, MaxTokens: 64},
		Reference: Trace{Text: citeFaithfulReport},
		Oracles:   []string{"citation-validity"},
		Rubric:    CitationRubric(evidence, minScore),
	}
}

// TestCitationValidityRegistered proves the oracle registered under its stable
// name and kind, so cases can reference it by name.
func TestCitationValidityRegistered(t *testing.T) {
	os, err := Lookup([]string{"citation-validity"})
	if err != nil {
		t.Fatalf("Lookup(citation-validity): %v", err)
	}
	if got := os[0].Kind(); got != "rubric" {
		t.Errorf("Kind() = %q, want rubric", got)
	}
}

// TestCitationValidityFaithfulReportPasses is the happy path: every marker —
// bare-numeric, lettered (case-variant), and hyphenated — resolves to a
// declared evidence entry, so the oracle passes at score 1.0.
func TestCitationValidityFaithfulReportPasses(t *testing.T) {
	c := citationCase(citeTestEvidence, 1)
	v := CitationValidity{}.Judge(Trace{}, Trace{Text: citeFaithfulReport}, c)
	if !v.Pass {
		t.Fatalf("faithful report must pass; got %+v", v)
	}
	if v.Score != 1 {
		t.Errorf("score = %v, want 1.0", v.Score)
	}
	if strings.Contains(v.Detail, "uncited") {
		t.Errorf("all evidence cited; Detail must carry no uncited warning: %q", v.Detail)
	}
}

// TestCitationValidityDanglingCitationFails is the defect witness: a marker no
// evidence entry declares fails the oracle, and Detail names that exact
// dangling citation.
func TestCitationValidityDanglingCitationFails(t *testing.T) {
	c := citationCase(citeTestEvidence, 1)
	v := CitationValidity{}.Judge(Trace{}, Trace{Text: citeDanglingReport}, c)
	if v.Pass {
		t.Fatalf("dangling citation must not pass; got %+v", v)
	}
	if want := 3.0 / 4.0; v.Score != want {
		t.Errorf("score = %v, want %v (3 of 4 markers resolved)", v.Score, want)
	}
	if !strings.Contains(v.Detail, "[9]") {
		t.Errorf("Detail must name the dangling citation [9]; got %q", v.Detail)
	}
}

// TestCitationValiditySpineIntegration runs the dangling report through the
// full spine: the failure bundle names citation-validity as the failing oracle
// and carries the dangling marker in its detail.
func TestCitationValiditySpineIntegration(t *testing.T) {
	c := citationCase(citeTestEvidence, 1)
	eng := ScriptedRunner{Label: "engine-dangling-citation", Trace: Trace{Text: citeDanglingReport}}
	res, err := RunCase(c, ReferenceRunner{}, eng, oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if res.Pass {
		t.Fatalf("dangling citation must not pass the spine; got %s", Explain(res))
	}
	fb := res.FailureBundle
	if fb == nil {
		t.Fatal("failing run must carry a failure bundle")
	}
	if fb.FailingOracle != "citation-validity" {
		t.Errorf("failing oracle = %q, want citation-validity", fb.FailingOracle)
	}
	if !strings.Contains(fb.Detail, "[9]") {
		t.Errorf("bundle detail must name the dangling citation [9]; got %q", fb.Detail)
	}
}

// TestCitationValidityMinScoreTolerance proves the threshold gate: the same
// dangling report (3/4 resolved) passes when the case tolerates MinScore 0.75,
// and the Detail still names the tolerated dangling marker.
func TestCitationValidityMinScoreTolerance(t *testing.T) {
	c := citationCase(citeTestEvidence, 0.75)
	v := CitationValidity{}.Judge(Trace{}, Trace{Text: citeDanglingReport}, c)
	if !v.Pass {
		t.Fatalf("3/4 resolved must pass at MinScore 0.75; got %+v", v)
	}
	if !strings.Contains(v.Detail, "[9]") {
		t.Errorf("tolerated-dangling detail should still name the marker; got %q", v.Detail)
	}
}

// TestCitationValidityZeroCitations defines the zero-citation edge: a report
// with no markers has nothing to resolve and passes at score 1, with the
// declared-but-uncited evidence surfaced as a soft warning in Detail — never
// as a failure (must-cite requirements belong to the grounding/omission
// rubrics, not the citation machinery).
func TestCitationValidityZeroCitations(t *testing.T) {
	c := citationCase(citeTestEvidence, 1)
	v := CitationValidity{}.Judge(Trace{}, Trace{Text: "Throughput increased 12% week over week."}, c)
	if !v.Pass || v.Score != 1 {
		t.Fatalf("zero-citation report must pass at score 1; got %+v", v)
	}
	if !strings.Contains(v.Detail, "no citation markers") {
		t.Errorf("Detail must note the zero-citation edge; got %q", v.Detail)
	}
	if !strings.Contains(v.Detail, "uncited evidence (soft warning)") || !strings.Contains(v.Detail, "E-12") {
		t.Errorf("Detail must soft-warn about uncited declared evidence; got %q", v.Detail)
	}

	// No markers AND no evidence: nothing to resolve, nothing to warn about.
	empty := CitationValidity{}.Judge(Trace{}, Trace{Text: "No sources here."}, citationCase(nil, 1))
	if !empty.Pass || empty.Score != 1 {
		t.Fatalf("no markers, no evidence must pass at score 1; got %+v", empty)
	}
	if strings.Contains(empty.Detail, "uncited") {
		t.Errorf("no declared evidence: Detail must carry no uncited warning: %q", empty.Detail)
	}
}

// TestCitationValidityNoEvidenceFailsClosed defines the empty-evidence edge:
// markers judged against no declared evidence all dangle and fail closed at
// score 0, naming the first marker.
func TestCitationValidityNoEvidenceFailsClosed(t *testing.T) {
	c := citationCase(nil, 1)
	v := CitationValidity{}.Judge(Trace{}, Trace{Text: citeFaithfulReport}, c)
	if v.Pass {
		t.Fatalf("markers with no declared evidence must fail closed; got %+v", v)
	}
	if v.Score != 0 {
		t.Errorf("score = %v, want 0", v.Score)
	}
	if !strings.Contains(v.Detail, "[1]") {
		t.Errorf("Detail must name the first dangling citation [1]; got %q", v.Detail)
	}
}

// TestCitationValidityMalformedMarkers proves bracketed text outside the
// marker grammar is skipped without a panic and never counts toward the
// score: only the one well-formed, resolvable marker is judged.
func TestCitationValidityMalformedMarkers(t *testing.T) {
	c := citationCase(citeTestEvidence, 1)
	text := "Latency held [sic] flat [] at 250ms [1.2] — see [ 3 ] and [12a-] and [-] but really [1]."
	v := CitationValidity{}.Judge(Trace{}, Trace{Text: text}, c)
	if !v.Pass {
		t.Fatalf("only well-formed resolvable marker [1] present; must pass, got %+v", v)
	}
	if v.Score != 1 {
		t.Errorf("score = %v, want 1.0 (1/1 markers resolved)", v.Score)
	}
}

// TestCitationValidityUncitedEvidenceIsSoftWarning proves declared-but-uncited
// evidence never fails a passing report: the report resolves every marker it
// carries, and the never-cited ids appear only as the Detail warning.
func TestCitationValidityUncitedEvidenceIsSoftWarning(t *testing.T) {
	c := citationCase(citeTestEvidence, 1)
	v := CitationValidity{}.Judge(Trace{}, Trace{Text: "Throughput increased 12% week over week [1]."}, c)
	if !v.Pass || v.Score != 1 {
		t.Fatalf("all markers resolved; uncited evidence must not fail: %+v", v)
	}
	if !strings.Contains(v.Detail, "uncited evidence (soft warning): S3, E-12") {
		t.Errorf("Detail must soft-warn with the uncited ids in declaration order; got %q", v.Detail)
	}
}

// TestCiteEvidenceIDForms proves the evidence-entry parser accepts the
// documented forms — bare id, bracketed id, id-colon-link, id-space-description
// — normalizes case, and drops blanks/duplicates.
func TestCiteEvidenceIDForms(t *testing.T) {
	known, declared := citeEvidenceIDs([]string{
		"1", "[2]", "s3: https://ops/queue", "E-12 incident postmortem", "  ", "1",
	})
	want := []string{"1", "2", "S3", "E-12"}
	if len(declared) != len(want) {
		t.Fatalf("declared = %v, want %v", declared, want)
	}
	for i, id := range want {
		if declared[i] != id {
			t.Errorf("declared[%d] = %q, want %q", i, declared[i], id)
		}
		if !known[id] {
			t.Errorf("known set missing %q", id)
		}
	}
}
