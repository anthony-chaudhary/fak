package livecodebench

import (
	"os"
	"strings"
	"testing"
)

func TestNormalizeFromUpstreamSample(t *testing.T) {
	raw, err := os.ReadFile("testdata/upstream_sample.json")
	if err != nil {
		t.Fatal(err)
	}
	ups, err := ParseUpstreamRows(raw)
	if err != nil {
		t.Fatal(err)
	}
	if len(ups) != 3 {
		t.Fatalf("parsed %d upstream rows, want 3", len(ups))
	}
	suite, err := Normalize(ups, NormalizeOptions{
		Release:   "release_v2",
		DatasetID: "livecodebench/code_generation_lite",
		Revision:  "release_v2",
		Split:     "test",
		Model:     "glm-5.2",
	})
	if err != nil {
		t.Fatalf("Normalize: %v", err)
	}
	if err := suite.Validate(); err != nil {
		t.Fatalf("normalized suite must validate: %v", err)
	}
	if suite.ReleaseVersion != "release_v2" {
		t.Errorf("release = %q, want release_v2", suite.ReleaseVersion)
	}
	if suite.Provenance.ProblemCount != 3 {
		t.Errorf("provenance.problem_count = %d, want 3", suite.Provenance.ProblemCount)
	}
	if suite.Provenance.DatasetID == "" || suite.Provenance.Revision != "release_v2" {
		t.Errorf("provenance = %+v, want dataset id and revision recorded", suite.Provenance)
	}
	if suite.Provenance.ContestDateFrom != "2024-09-01" || suite.Provenance.ContestDateTo != "2024-11-20" {
		t.Errorf("contest-date range = %s..%s, want 2024-09-01..2024-11-20",
			suite.Provenance.ContestDateFrom, suite.Provenance.ContestDateTo)
	}
	p0 := suite.Problems[0]
	if p0.Difficulty != "easy" {
		t.Errorf("difficulty = %q, want lowercased easy", p0.Difficulty)
	}
	if p0.ContestDate != "2024-09-01" {
		t.Errorf("contest_date = %q, want date-only 2024-09-01", p0.ContestDate)
	}
	if len(p0.PublicTests) != 1 || p0.PublicTests[0].TestType != "functional" {
		t.Errorf("public tests = %+v, want one functional case", p0.PublicTests)
	}
	if got := suite.Problems[2].PublicTests; len(got) != 0 {
		t.Errorf("row with empty public_test_cases must yield no tests, got %+v", got)
	}
}

func TestNormalizeRequiresProvenanceDataset(t *testing.T) {
	ups := []UpstreamProblem{{QuestionID: "q1", QuestionContent: "do a thing", ContestDate: "2024-01-01"}}
	_, err := Normalize(ups, NormalizeOptions{Release: "release_v2"})
	if err == nil || !strings.Contains(err.Error(), "provenance.dataset_id") {
		t.Fatalf("err = %v, want provenance.dataset_id refusal", err)
	}
}

func TestNormalizeRejectsUnknownRelease(t *testing.T) {
	ups := []UpstreamProblem{{QuestionID: "q1", QuestionContent: "x"}}
	_, err := Normalize(ups, NormalizeOptions{Release: "release_v99", DatasetID: "d", Revision: "r"})
	if err == nil || !strings.Contains(err.Error(), "unknown") {
		t.Fatalf("err = %v, want unknown-release refusal", err)
	}
}

func TestParseUpstreamRowsAcceptsBareArray(t *testing.T) {
	body := `[{"question_id":"q1","question_content":"c","platform":"leetcode","difficulty":"Easy"}]`
	ups, err := ParseUpstreamRows([]byte(body))
	if err != nil {
		t.Fatal(err)
	}
	if len(ups) != 1 || ups[0].QuestionID != "q1" {
		t.Fatalf("parsed = %+v, want one row q1", ups)
	}
}

func TestNormalizeContestDate(t *testing.T) {
	cases := []struct{ in, want string }{
		{"", ""},
		{"2024-05-20", "2024-05-20"},
		{"2024-05-20T00:00:00", "2024-05-20"},
		{"2024-05-20T13:45:07Z", "2024-05-20"},
	}
	for _, tc := range cases {
		got, err := normalizeContestDate(tc.in)
		if err != nil {
			t.Fatalf("normalizeContestDate(%q): %v", tc.in, err)
		}
		if got != tc.want {
			t.Errorf("normalizeContestDate(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
	if _, err := normalizeContestDate("not-a-date"); err == nil {
		t.Fatal("expected error for unparseable date")
	}
}
