package swebenchsota

import (
	"testing"
)

// fixtureHTML mirrors the embedded leaderboard JSON fixture from the Python test
// (tools/swebench_sota_snapshot_test.py).
const fixtureHTML = `
<html><body>
<script type="application/json" id="leaderboard-data">
[
  {"name":"bash-only","results":[
    {"name":"GLM-5 (high reasoning)","resolved":72.8,"date":"2026-02-17","mini-swe-agent_version":"2.0.0"},
    {"name":"Claude 4.5 Opus (high reasoning)","resolved":76.8,"date":"2026-02-17","mini-swe-agent_version":"2.0.0"}
  ]},
  {"name":"Verified","results":[
    {"name":"mini-SWE-agent + Claude 4.5 Opus (high reasoning)","resolved":76.8,"date":"2026-02-17"},
    {"name":"live-SWE-agent + Claude 4.5 Opus medium","resolved":79.2,"date":"2025-12-15"},
    {"name":"mini-SWE-agent + GLM-5 (high reasoning)","resolved":72.8,"date":"2026-02-17"}
  ]}
]
</script>
</body></html>
`

func TestBuildSnapshotFromSavedHTML(t *testing.T) {
	doc, err := BuildSnapshot(fixtureHTML, "leaderboard.html", "2026-06-25T00:00:00Z", Options{})
	if err != nil {
		t.Fatalf("BuildSnapshot: %v", err)
	}

	if doc.Schema != "fak.swebench-sota-snapshot.v1" {
		t.Errorf("schema = %q, want fak.swebench-sota-snapshot.v1", doc.Schema)
	}
	if doc.OverallSOTA.Top == nil {
		t.Fatal("overall_sota.top is nil")
	}
	if got := doc.OverallSOTA.Top.Name; got != "live-SWE-agent + Claude 4.5 Opus medium" {
		t.Errorf("overall_sota.top.name = %v, want live-SWE-agent + Claude 4.5 Opus medium", got)
	}
	if got := doc.OverallSOTA.Top.ResolvedPct; !floatEq(got, 79.2) {
		t.Errorf("overall_sota.top.resolved_pct = %v, want 79.2", got)
	}
	if doc.SameScaffoldSOTA.Top == nil {
		t.Fatal("same_scaffold_sota.top is nil")
	}
	if got := doc.SameScaffoldSOTA.Top.ResolvedPct; !floatEq(got, 76.8) {
		t.Errorf("same_scaffold_sota.top.resolved_pct = %v, want 76.8", got)
	}
	if len(doc.SameScaffoldSOTA.FocalRows) == 0 {
		t.Fatal("same_scaffold_sota.focal_rows is empty")
	}
	if got := doc.SameScaffoldSOTA.FocalRows[0].Name; got != "GLM-5 (high reasoning)" {
		t.Errorf("focal_rows[0].name = %v, want GLM-5 (high reasoning)", got)
	}
	if probs := Check(doc); len(probs) != 0 {
		t.Errorf("Check returned problems: %v", probs)
	}
}

func TestExtractLeaderboardMissingTag(t *testing.T) {
	if _, err := ExtractLeaderboard("<html><body>no script here</body></html>"); err == nil {
		t.Fatal("expected an error when the leaderboard script tag is absent")
	}
}

func TestExtractLeaderboardNotAList(t *testing.T) {
	src := `<script type="application/json" id="leaderboard-data">{"name":"x"}</script>`
	_, err := ExtractLeaderboard(src)
	if err == nil {
		t.Fatal("expected an error when the leaderboard JSON is not a list")
	}
	if err.Error() != "leaderboard JSON is not a list" {
		t.Errorf("err = %q, want \"leaderboard JSON is not a list\"", err.Error())
	}
}

func TestRenderMarkdown(t *testing.T) {
	doc, err := BuildSnapshot(fixtureHTML, "leaderboard.html", "2026-06-25T00:00:00Z", Options{})
	if err != nil {
		t.Fatalf("BuildSnapshot: %v", err)
	}
	md := RenderMarkdown(doc)
	if !contains(md, "SWE-bench SOTA Snapshot") {
		t.Errorf("markdown missing title:\n%s", md)
	}
	if !contains(md, "GLM-5") {
		t.Errorf("markdown missing focal GLM-5 row:\n%s", md)
	}
}

func TestMissingFocalRowIsRejected(t *testing.T) {
	doc := Snapshot{
		Schema:           Schema,
		Benchmark:        "SWE-bench Verified",
		OverallSOTA:      GroupSummary{Top: &RowSummary{Name: "x", ResolvedPct: 1}},
		SameScaffoldSOTA: GroupSummary{Top: &RowSummary{Name: "y", ResolvedPct: 1}, FocalRows: []RowSummary{}},
	}
	probs := Check(doc)
	if !hasProblem(probs, "same_scaffold_sota.focal_rows missing") {
		t.Errorf("Check did not flag missing focal rows: %v", probs)
	}
}

func floatEq(v any, want float64) bool {
	f, ok := v.(float64)
	return ok && f == want
}

func contains(s, sub string) bool {
	return len(s) >= len(sub) && indexOf(s, sub) >= 0
}

func indexOf(s, sub string) int {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return i
		}
	}
	return -1
}

func hasProblem(probs []string, want string) bool {
	for _, p := range probs {
		if p == want {
			return true
		}
	}
	return false
}
