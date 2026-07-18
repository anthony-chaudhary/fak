package guardcomplaint

import (
	"strings"
	"testing"
)

// TestNormalizeDomainClosedSet pins the domain axis (#5191): guard is the default, workflow
// is admitted, and anything else is a named error over the closed set.
func TestNormalizeDomainClosedSet(t *testing.T) {
	if d, err := NormalizeDomain(""); err != nil || d != DefaultDomain {
		t.Fatalf("empty domain => (%q,%v), want default %q", d, err, DefaultDomain)
	}
	if d, err := NormalizeDomain("  Workflow "); err != nil || d != "workflow" {
		t.Fatalf("workflow normalize => (%q,%v)", d, err)
	}
	if _, err := NormalizeDomain("nonsense"); err == nil {
		t.Fatal("unknown domain must error")
	}
}

// TestNormalizeKindForIsDomainScoped pins that each domain validates against its OWN closed
// kind set: a guard kind is rejected in the workflow domain and vice-versa, and an empty kind
// resolves to the domain's default.
func TestNormalizeKindForIsDomainScoped(t *testing.T) {
	if k, err := NormalizeKindFor("workflow", ""); err != nil || k != DefaultWorkflowKind {
		t.Fatalf("empty workflow kind => (%q,%v), want %q", k, err, DefaultWorkflowKind)
	}
	if k, err := NormalizeKindFor("workflow", " Lane-Collision "); err != nil || k != "lane-collision" {
		t.Fatalf("lane-collision normalize => (%q,%v)", k, err)
	}
	// A guard kind is NOT valid in the workflow domain (and the error must name the workflow set).
	if _, err := NormalizeKindFor("workflow", "false-positive"); err == nil {
		t.Fatal("a guard kind must be rejected in the workflow domain")
	} else if !strings.Contains(err.Error(), "lane-collision") {
		t.Fatalf("workflow kind error must name the workflow vocabulary, got %v", err)
	}
	// A workflow kind is NOT valid in the guard domain.
	if _, err := NormalizeKindFor("guard", "lane-collision"); err == nil {
		t.Fatal("a workflow kind must be rejected in the guard domain")
	}
	// Guard-domain default is unchanged (backward compatibility).
	if k, err := NormalizeKindFor("guard", ""); err != nil || k != DefaultKind {
		t.Fatalf("empty guard kind => (%q,%v), want %q", k, err, DefaultKind)
	}
}

// TestWorkflowComplaintKeyLabelTitleDistinct pins that a workflow complaint routes through a
// distinct dedup key prefix, label, and title — while a guard complaint with the SAME summary
// stays on the byte-identical historical key, so the two domains never fold onto one issue.
func TestWorkflowComplaintKeyLabelTitleDistinct(t *testing.T) {
	wf := Complaint{
		Domain:    "workflow",
		Kind:      "lane-collision",
		Tool:      "fak commit",
		Summary:   "two workers raced the commit lock",
		Rationale: "compute lease and cmd lane both landed on cmd/fak within the same minute.",
	}
	if got, want := wf.Key(), "workflow-complaint/lane-collision/none/fak-commit/two-workers-raced-the-commit-lock"; got != want {
		t.Fatalf("workflow key = %q, want %q", got, want)
	}
	if !strings.HasPrefix(wf.Title(), "workflow friction [lane-collision]") {
		t.Fatalf("workflow title = %q, want 'workflow friction [lane-collision]' prefix", wf.Title())
	}
	if LabelFor("workflow") != WorkflowLabel {
		t.Fatalf("LabelFor(workflow) = %q, want %q", LabelFor("workflow"), WorkflowLabel)
	}
	if LabelFor("guard") != Label || LabelFor("") != Label {
		t.Fatalf("LabelFor(guard/empty) must be the guard Label %q", Label)
	}

	// A guard complaint with the same summary keeps the historical key prefix and stays distinct.
	guard := Complaint{Kind: "other", Summary: "two workers raced the commit lock"}
	if !strings.HasPrefix(guard.Key(), "guard-complaint/") {
		t.Fatalf("guard key lost its historical prefix: %q", guard.Key())
	}
	if guard.Key() == wf.Key() {
		t.Fatal("guard and workflow complaints with the same summary must not share a dedup key")
	}
}

// TestWorkflowBodyFramingAndMarker pins the workflow body: it carries the workflow framing and
// the kind's description, but REUSES the shared dedup marker tag so MarkerKey/occurrence folding
// keep working across both domains.
func TestWorkflowBodyFramingAndMarker(t *testing.T) {
	wf := Complaint{
		Domain:    "workflow",
		Kind:      "shared-tree-clobber",
		Summary:   "peer restaged my file",
		Rationale: "a peer's git add swept my unstaged hunk on the shared trunk.",
	}
	body := wf.Body(3)
	// Dedup marker tag is shared (guard-complaint-key) so MarkerKey extracts the workflow key.
	if MarkerKey(body) != wf.Key() {
		t.Fatalf("MarkerKey(workflow body) = %q, want %q", MarkerKey(body), wf.Key())
	}
	if occurrencesOf(body) != 3 {
		t.Fatalf("occurrencesOf(workflow body) = %d, want 3", occurrencesOf(body))
	}
	for _, want := range []string{
		"# Workflow friction (agent report)",
		"domain: `workflow`",
		WorkflowKinds["shared-tree-clobber"],
		"fak complain --domain workflow",
	} {
		if !strings.Contains(body, want) {
			t.Fatalf("workflow body missing %q:\n%s", want, body)
		}
	}
	// It must NOT carry the guard-appeal framing.
	if strings.Contains(body, "Guard complaint (agent appeal)") {
		t.Fatalf("workflow body leaked the guard-appeal framing:\n%s", body)
	}
}

// TestBuildPlanCarriesDomain pins that the plan row records the resolved domain (so the JSON
// result and Render surface which channel a complaint filed on), defaulting to guard.
func TestBuildPlanCarriesDomain(t *testing.T) {
	wf := BuildPlan(Complaint{Domain: "workflow", Kind: "tool-timeout", Summary: "go test hung"}, nil)
	if wf.Domain != "workflow" {
		t.Fatalf("workflow plan domain = %q, want workflow", wf.Domain)
	}
	guard := BuildPlan(Complaint{Kind: "other", Summary: "some guard thing"}, nil)
	if guard.Domain != "guard" {
		t.Fatalf("default plan domain = %q, want guard", guard.Domain)
	}
}
