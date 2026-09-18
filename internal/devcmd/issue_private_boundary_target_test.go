package devcmd

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/issuepolicy"
)

func TestIssuePrivateBoundaryTargetRepoIsPrivate(t *testing.T) {
	for _, tc := range []struct {
		name string
		repo string
		want bool
	}{
		{name: "owner/private", repo: "anthony-chaudhary/fak-private", want: true},
		{name: "bare private", repo: "fak-private", want: true},
		{name: "mixed case", repo: "Anthony-Chaudhary/FAK-Private", want: true},
		{name: "surrounding whitespace", repo: "  anthony-chaudhary/fak-private  ", want: true},
		{name: "owner/public", repo: "anthony-chaudhary/fak", want: false},
		{name: "bare public", repo: "fak", want: false},
		{name: "empty", repo: "", want: false},
		{name: "unrelated owner", repo: "owner/other", want: false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := issueTargetRepoIsPrivate(tc.repo); got != tc.want {
				t.Fatalf("issueTargetRepoIsPrivate(%q) = %v, want %v", tc.repo, got, tc.want)
			}
		})
	}
}

// TestIssuePrivateBoundaryDiscoverabilityBothTargets proves the discoverability
// audit admits the private target and refuses the public one on the SAME draft
// body naming fak-private.
func TestIssuePrivateBoundaryDiscoverabilityBothTargets(t *testing.T) {
	draft := privateBoundaryProbeDraft()

	privateRow := auditIssueDraftDiscoverability(draft, issuepolicy.Options{TargetPrivate: true})
	if containsReason(privateRow.Reasons, issuepolicy.ReasonPrivateBoundary) {
		t.Fatalf("private target row must suppress %s; reasons = %+v", issuepolicy.ReasonPrivateBoundary, privateRow.Reasons)
	}

	publicRow := auditIssueDraftDiscoverability(draft, issuepolicy.Options{})
	if !containsReason(publicRow.Reasons, issuepolicy.ReasonPrivateBoundary) {
		t.Fatalf("public target row must keep %s; reasons = %+v", issuepolicy.ReasonPrivateBoundary, publicRow.Reasons)
	}
}

// TestIssueCreatePrivateTargetBoundaryAdmits proves issue create against the
// private repo reaches the injected runner (real gh never invoked).
func TestIssuePrivateBoundaryCreatePrivateTargetAdmits(t *testing.T) {
	body := privateBoundaryProbeBody()
	bodyFile := filepath.Join(t.TempDir(), "probe.md")
	if err := os.WriteFile(bodyFile, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	called := false
	runner := func(args []string) (string, string, bool) {
		called = true
		return "https://example.test/issues/1981", "", true
	}
	var out, errb bytes.Buffer
	code := runIssueCreateWithCleanScrub(&out, &errb, []string{
		"--title", "feat(compute): private target boundary",
		"--body-file", bodyFile,
		"--repo", "anthony-chaudhary/fak-private",
		"--estimate-points", "1",
		"--parent-baseline-points", "1",
		"--target-envelope", "- paths: >= 1 command",
		"--witnessed-envelope", "- paths: 1 command",
	}, runner)
	if code != 0 {
		t.Fatalf("private target exit = %d, want 0; stderr:\n%s", code, errb.String())
	}
	if !called {
		t.Fatalf("runner was not called for private target; stderr:\n%s", errb.String())
	}
	if strings.Contains(errb.String(), issuepolicy.ReasonPrivateBoundary) {
		t.Fatalf("private target stderr carries %s:\n%s", issuepolicy.ReasonPrivateBoundary, errb.String())
	}
}

// TestIssueCreatePublicTargetBoundaryRefuses proves the SAME body against the
// public repo is refused by the born-routed gate (exit 3) and the runner is
// never invoked.
func TestIssuePrivateBoundaryCreatePublicTargetRefuses(t *testing.T) {
	body := privateBoundaryProbeBody()
	bodyFile := filepath.Join(t.TempDir(), "probe.md")
	if err := os.WriteFile(bodyFile, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	called := false
	runner := func(args []string) (string, string, bool) {
		called = true
		return "https://example.test/issues/1981", "", true
	}
	var out, errb bytes.Buffer
	code := runIssueCreateWithCleanScrub(&out, &errb, []string{
		"--title", "feat(compute): public target boundary",
		"--body-file", bodyFile,
		"--repo", "anthony-chaudhary/fak",
		"--estimate-points", "1",
		"--parent-baseline-points", "1",
		"--target-envelope", "- paths: >= 1 command",
		"--witnessed-envelope", "- paths: 1 command",
	}, runner)
	if code != 3 {
		t.Fatalf("public target exit = %d, want 3; stderr:\n%s", code, errb.String())
	}
	if called {
		t.Fatalf("runner must not be called for public target; stderr:\n%s", errb.String())
	}
	if !strings.Contains(errb.String(), issuepolicy.ReasonPrivateBoundary) {
		t.Fatalf("public target stderr missing %s:\n%s", issuepolicy.ReasonPrivateBoundary, errb.String())
	}
}

// privateBoundaryProbeDraft is a dispatchable draft whose body names
// fak-private in its "Boundary notes" section and carries a private path in
// "Likely files" — the surfaces CandidateFromIssueDraft scans. It reuses the
// canonical dispatchable body so the only thing under test is the target-repo
// boundary.
func privateBoundaryProbeDraft() issuepolicy.IssueDraft {
	body := validDispatchableIssueBody("compute", []string{"platform/dispatch/x.go"}, 4) +
		"\n## Boundary notes\nRequires fak-private platform/dispatch/x.go.\n"
	return issuepolicy.IssueDraft{
		Number: 1981,
		Title:  "feat(compute): target-repo aware born-routed boundary",
		Body:   body,
		Labels: []issuepolicy.IssueLabel{{Name: "class:dev"}},
	}
}

func privateBoundaryProbeBody() string {
	return privateBoundaryProbeDraft().Body
}
func containsReason(reasons []string, want string) bool {
	for _, r := range reasons {
		if r == want {
			return true
		}
	}
	return false
}
