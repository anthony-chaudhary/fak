package studymonitor

import (
	"bytes"
	"strings"
	"testing"
	"time"
)

func TestRegistryValidationRejectsDuplicate(t *testing.T) {
	r := Registry{Schema: Schema, Methodology: "ranked", Repositories: []Repository{
		{Repository: "owner/repo", URL: "https://github.com/owner/repo", Status: "candidate", Priority: 1, Why: "useful", LastChecked: "2026-08-14", CheckedRevision: "abc"},
		{Repository: "OWNER/REPO", URL: "https://github.com/OWNER/REPO", Status: "watch", Priority: 2, Why: "duplicate", LastChecked: "2026-08-14", CheckedRevision: "def"},
	}}
	if err := r.Validate(); err == nil || !strings.Contains(err.Error(), "duplicate repository") {
		t.Fatalf("Validate() error = %v, want duplicate repository", err)
	}
}

func TestBuildReportSortsAndRenderMarksDue(t *testing.T) {
	r := Registry{Schema: Schema, Methodology: "ranked", Repositories: []Repository{
		{Repository: "owner/later", URL: "https://example/later", Status: "watch", Priority: 2, Why: "later", LastChecked: "2026-08-13", CheckedRevision: "123456789012345", StarsAtCheck: 2},
		{Repository: "owner/first", URL: "https://example/first", Status: "candidate", Priority: 1, Why: "first", LastChecked: "2026-08-01", CheckedRevision: "abcdef", StarsAtCheck: 3},
	}}
	report := BuildReport("registry.json", r, time.Date(2026, 8, 14, 0, 0, 0, 0, time.UTC), 7)
	if got := report.Repositories[0].Repository; got != "owner/first" {
		t.Fatalf("first repository = %q", got)
	}
	var out bytes.Buffer
	RenderHuman(&out, report)
	text := out.String()
	if !strings.Contains(text, "owner/first status=candidate checked=2026-08-01 age_days=13 due=true") {
		t.Fatalf("render missing due witness:\n%s", text)
	}
	if !strings.Contains(text, "owner/later status=watch checked=2026-08-13 age_days=1 due=false") {
		t.Fatalf("render missing fresh witness:\n%s", text)
	}
}
