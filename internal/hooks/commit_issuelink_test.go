package hooks

import (
	"reflect"
	"testing"
)

// commit_issuelink_test.go — #312 author-time issue-link gate. The grammar must agree with the
// closure auditor (tools/issue_closure_audit.py): a commit the preview calls "resolving" is one
// the auditor binds to its issue, so the two never disagree about what counts as a real close.

func TestLintIssueLink_resolvingForms(t *testing.T) {
	cases := []struct {
		name    string
		message string
		want    bool
	}{
		{"subject ref", "fix(gateway): correct the reclaim (#312)", true},
		{"subject ref bare", "fix(gateway): correct the reclaim #312", true},
		{"closes verb body", "fix(gateway): correct the reclaim\n\nCloses #312", true},
		{"fixes verb body", "fix(x): do it\n\nFixes #99", true},
		{"resolve verb body", "fix(x): do it\n\nresolve #5", true},
		{"resolved verb body", "fix(x): do it\n\nResolved #5", true},
		{"house noun form", "fix(x): do it\n\nThis fixes the bug (issue #42)", true},
		{"house noun plural", "fix(x): do it\n\nissues #1, #2 and #3", true},
		{"body mention only is NOT resolving", "fix(x): do it\n\nsee also #77 for context", false},
		{"no ref at all", "fix(gateway): correct the reclaim (fak gateway)", false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := lintIssueLink(c.message).resolving; got != c.want {
				t.Errorf("lintIssueLink(%q).resolving = %v, want %v", c.message, got, c.want)
			}
		})
	}
}

func TestIssueRefs_boundaryAndDedup(t *testing.T) {
	cases := []struct {
		in   string
		want []int
	}{
		{"closes #12 and #34", []int{12, 34}},
		{"see #5, then #5 again", []int{5}},          // dedup, first-seen order
		{"a token like abc#12 is not a ref", nil},     // word-char before '#'
		{"hash-joined foo-#9 is not a ref", nil},      // '-' before '#'
		{"snake_#7 is not a ref", nil},                // '_' before '#'
		{"#1 leads the line", []int{1}},               // start-of-string boundary
		{"no refs here", nil},
	}
	for _, c := range cases {
		got := issueRefs(c.in)
		if !reflect.DeepEqual(got, c.want) {
			t.Errorf("issueRefs(%q) = %v, want %v", c.in, got, c.want)
		}
	}
}

func TestLintCommitMessage_issueLinkAdvisoryByDefault(t *testing.T) {
	root := writeLintRepo(t)
	// No issue ref: default is advisory (a Note), so a clean stamped commit stays OK.
	r := LintCommitMessage("feat(gateway): add the slot reclaim (fak gateway)", []string{"internal/gateway/server.go"}, root)
	if !r.OK {
		t.Fatalf("missing issue link must be advisory by default, got issues=%v", r.Issues)
	}
	if r.IssueResolving {
		t.Errorf("no #N present, IssueResolving should be false")
	}
	if !hasNoteContaining(r, "no bindable issue link") {
		t.Errorf("want the advisory issue-link note, got %v", r.Notes)
	}
}

func TestLintCommitMessage_requireIssueBlocks(t *testing.T) {
	root := writeLintRepo(t)
	// --require-issue: a missing resolving #N becomes BLOCKING.
	r := LintCommitMessageWithOptions("feat(gateway): add the slot reclaim (fak gateway)", []string{"internal/gateway/server.go"}, root, true)
	if r.OK {
		t.Fatalf("with require-issue, a missing #N must block")
	}
	if !hasIssueContaining(r, "no bindable issue link") {
		t.Errorf("want the blocking issue-link defect, got %v", r.Issues)
	}
}

func TestLintCommitMessage_requireIssueSatisfiedBySubjectRef(t *testing.T) {
	root := writeLintRepo(t)
	r := LintCommitMessageWithOptions("feat(gateway): add the slot reclaim (#312) (fak gateway)", []string{"internal/gateway/server.go"}, root, true)
	if !r.OK {
		t.Fatalf("a subject #N satisfies require-issue, got issues=%v", r.Issues)
	}
	if !r.IssueResolving {
		t.Errorf("subject #312 should be resolving")
	}
	if !reflect.DeepEqual(r.IssueRefs, []int{312}) {
		t.Errorf("want refs [312], got %v", r.IssueRefs)
	}
}

func TestLintCommitMessage_requireIssueSatisfiedByClosesBody(t *testing.T) {
	root := writeLintRepo(t)
	msg := "feat(gateway): add the slot reclaim (fak gateway)\n\nCloses #312"
	r := LintCommitMessageWithOptions(msg, []string{"internal/gateway/server.go"}, root, true)
	if !r.OK {
		t.Fatalf("a `Closes #N` body satisfies require-issue, got issues=%v", r.Issues)
	}
	if !r.IssueResolving {
		t.Errorf("Closes #312 should be resolving")
	}
}

func TestLintCommitMessage_requireIssueMentionStillBlocks(t *testing.T) {
	root := writeLintRepo(t)
	// A body MENTION (no closing verb, not in subject) is a reference but NOT resolving — the
	// auditor would bucket this CLAIMED_CLOSED, so require-issue must still block, with a hint
	// that distinguishes "mentioned but not binding" from "no issue at all".
	msg := "feat(gateway): add the slot reclaim (fak gateway)\n\nrelated to #312"
	r := LintCommitMessageWithOptions(msg, []string{"internal/gateway/server.go"}, root, true)
	if r.OK {
		t.Fatalf("a non-resolving mention must still block under require-issue")
	}
	if !hasIssueContaining(r, "non-resolving position") {
		t.Errorf("want the mention-not-binding hint, got %v", r.Issues)
	}
}

func TestLintCommitMessage_requireIssueExemptSubjectSkips(t *testing.T) {
	root := writeLintRepo(t)
	// A merge/release subject owes no issue link even under require-issue.
	r := LintCommitMessageWithOptions("Merge origin/main into main", []string{"internal/gateway/x.go"}, root, true)
	if !r.OK {
		t.Fatalf("an exempt subject owes no issue link, got issues=%v", r.Issues)
	}
}
