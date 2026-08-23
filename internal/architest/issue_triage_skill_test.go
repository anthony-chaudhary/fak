package architest

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestIssueTriageSkillUsesSupportedContractInput(t *testing.T) {
	root := repoRoot(t)
	body, err := os.ReadFile(filepath.Join(root, ".claude", "skills", "issue-triage", "SKILL.md"))
	if err != nil {
		t.Fatal(err)
	}
	text := string(body)
	if strings.Contains(text, "issue contract --repo") || strings.Contains(text, "issue contract --issue") {
		t.Fatal("issue-triage skill documents unsupported issue contract flags")
	}
	for _, want := range []string{"gh issue view N --repo owner/name --json number,title,body,labels", "issue contract --from-issues issue.json --json"} {
		if !strings.Contains(text, want) {
			t.Fatalf("issue-triage skill missing supported selected-issue flow %q", want)
		}
	}
}
