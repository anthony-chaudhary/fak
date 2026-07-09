package livecodebench

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// fixedReportTime is the pinned generated_at the golden report renders with, so
// the golden markdown is deterministic and does not depend on the wall clock.
var fixedReportTime = time.Date(2026, 7, 9, 0, 0, 0, 0, time.UTC)

// reportFromSampleSuite builds the honest, unpromoted report the `report`
// renderer targets: NewReport's local-ungraded scaffold plus one per-problem
// verdict row per suite problem. It is the exact report the golden markdown and
// the CLI both render, so the golden file is the fixture run's contract.
func reportFromSampleSuite(t *testing.T) Report {
	t.Helper()
	s, err := LoadSuiteFile(filepath.Join("testdata", "suite_release_v2_sample.json"))
	if err != nil {
		t.Fatalf("load sample suite: %v", err)
	}
	r := NewReport(s, fixedReportTime)
	r.Problems = ProblemRowsFromSuite(s)
	if err := r.Validate(); err != nil {
		t.Fatalf("report does not validate: %v", err)
	}
	return r
}

// TestRenderReportMarkdownGolden is the golden-file test for the markdown of the
// fixture run (#2112). Run `GOLDEN_UPDATE=1 go test ./internal/livecodebench/...`
// to refresh the committed golden after an intentional renderer change.
func TestRenderReportMarkdownGolden(t *testing.T) {
	r := reportFromSampleSuite(t)
	got := RenderReportMarkdown(r)

	goldenPath := filepath.Join("testdata", "report_release_v2_sample.md")
	if os.Getenv("GOLDEN_UPDATE") == "1" {
		if err := os.WriteFile(goldenPath, []byte(got), 0o644); err != nil {
			t.Fatalf("update golden: %v", err)
		}
	}
	want, err := os.ReadFile(goldenPath)
	if err != nil {
		t.Fatalf("read golden %s: %v", goldenPath, err)
	}
	if got != string(want) {
		t.Fatalf("markdown does not match golden %s\n--- got ---\n%s\n--- want ---\n%s", goldenPath, got, want)
	}
}

// TestRenderReportMarkdownBanner asserts the load-bearing evidence-class /
// claim-boundary banner and the per-problem question_id -> verdict -> evidence
// id linkage are present, independent of exact byte layout.
func TestRenderReportMarkdownBanner(t *testing.T) {
	md := RenderReportMarkdown(reportFromSampleSuite(t))
	for _, want := range []string{
		"# LiveCodeBench Run Report",
		"- Evidence class: `" + EvidenceLocalUngraded + "`",
		"- Result claim allowed: `false`",
		"- Claim boundary:",
		"## Per-Problem Verdicts",
		"| question_id | scenario | arm | verdict | evidence_id |",
		"| `lcb-sample-001` | codegeneration | - | ungraded | `local-ungraded:lcb-sample-001` |",
	} {
		if !strings.Contains(md, want) {
			t.Fatalf("markdown missing %q\n---\n%s", want, md)
		}
	}
}

// TestRenderReportMarkdownArmDeltas covers the raw-vs-fak delta section: a graded
// two-arm report must render both arms' pass rates and the signed fak-minus-raw
// delta per scenario.
func TestRenderReportMarkdownArmDeltas(t *testing.T) {
	r := Report{
		Schema:         ReportSchema,
		GeneratedAt:    fixedReportTime.Format(time.RFC3339),
		Benchmark:      Benchmark,
		ReleaseVersion: "release_v6",
		EvidenceClass:  EvidenceOfficialLCBRunner,
		Arms: []ArmResult{
			{Arm: "raw", Scenario: ScenarioCodeGeneration, Problems: 10, Generations: 10, Graded: 10, Pass1: 0.40, Pass5: 0.60},
			{Arm: "fak", Scenario: ScenarioCodeGeneration, Problems: 10, Generations: 10, Graded: 10, Pass1: 0.50, Pass5: 0.70},
		},
		Summary: Summary{Problems: 10, Graded: 20},
	}
	if err := r.Validate(); err != nil {
		t.Fatalf("graded report does not validate: %v", err)
	}
	md := RenderReportMarkdown(r)
	for _, want := range []string{
		"## Arms (raw vs fak)",
		"| `codegeneration` | 0.4000 | 0.5000 | +0.1000 | 0.6000 | 0.7000 | +0.1000 |",
	} {
		if !strings.Contains(md, want) {
			t.Fatalf("markdown missing %q\n---\n%s", want, md)
		}
	}
}

// TestReportRejectsProblemRowMissingEvidence guards the honesty invariant: a
// per-problem row without an evidence id is refused, so a rendered verdict can
// never be untraceable to evidence.
func TestReportRejectsProblemRowMissingEvidence(t *testing.T) {
	r := NewReport(mustSampleSuite(t), fixedReportTime)
	r.Problems = []ProblemVerdict{{QuestionID: "q1", Scenario: ScenarioCodeGeneration, Verdict: VerdictUngraded}}
	if err := r.Validate(); err == nil {
		t.Fatal("expected validation error for a problem row with no evidence_id")
	}
}

func mustSampleSuite(t *testing.T) Suite {
	t.Helper()
	s, err := LoadSuiteFile(filepath.Join("testdata", "suite_release_v2_sample.json"))
	if err != nil {
		t.Fatalf("load sample suite: %v", err)
	}
	return s
}
