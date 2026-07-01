package branchrole

import (
	"strings"
	"testing"
)

func TestClassifyPublicDocRef(t *testing.T) {
	cases := []struct {
		path string
		line string
		want string
	}{
		{"README.md", `[![ci](https://github.com/anthony-chaudhary/fak/actions/workflows/ci.yml/badge.svg?branch=main)]`, PublicDocClassPublicFrontDoor},
		{"INSTALL.md", `curl -fsSL https://raw.githubusercontent.com/anthony-chaudhary/fak/main/install.sh | sh`, PublicDocClassPublicFrontDoor},
		{"GETTING-STARTED.md", `go install github.com/anthony-chaudhary/fak/cmd/fak@latest`, PublicDocClassReleaseArtifact},
		{"CONTRIBUTING.md", "- **Work directly on `main` — do not open a feature branch.**", PublicDocClassContributorWorkflow},
		{".github/copilot-instructions.md", "- Work directly on the trunk (`main`); never open a feature branch", PublicDocClassContributorWorkflow},
		{"docs/fak/deployment-guide.md", `-d '{"tool":"Bash","arguments":{"command":"git push origin main"}}'`, PublicDocClassAdjudicationFixture},
		{"docs/integrations/claude.md", "| `git push origin master` | POLICY_BLOCK |", PublicDocClassAdjudicationFixture},
		{"README.md", `go install github.com/anthony-chaudhary/fak/cmd/fak@main`, PublicDocClassUnclassified},
	}
	for _, tc := range cases {
		if got := ClassifyPublicDocRef(tc.path, tc.line); got != tc.want {
			t.Fatalf("ClassifyPublicDocRef(%q, %q) = %q, want %q", tc.path, tc.line, got, tc.want)
		}
	}
}

func TestPublicDocRefsCurrentTreeClassified(t *testing.T) {
	root := repoRootForRefAudit(t)
	findings, err := AuditPublicDocRefs(root)
	if err != nil {
		t.Fatalf("AuditPublicDocRefs: %v", err)
	}
	var unclassified []string
	classes := map[string]int{}
	for _, finding := range findings {
		classes[finding.Class]++
		if finding.Class == PublicDocClassUnclassified {
			unclassified = append(unclassified, finding.Path+":"+itoa(finding.Line)+" "+finding.Text)
		}
	}
	if len(unclassified) > 0 {
		t.Fatalf("unclassified public/contributor branch refs:\n%s", strings.Join(unclassified, "\n"))
	}
	for _, want := range []string{
		PublicDocClassPublicFrontDoor,
		PublicDocClassReleaseArtifact,
		PublicDocClassContributorWorkflow,
		PublicDocClassAdjudicationFixture,
	} {
		if classes[want] == 0 {
			t.Fatalf("audit saw no %s rows; classes=%v", want, classes)
		}
	}
}
