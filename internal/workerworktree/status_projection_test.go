package workerworktree

import (
	"strings"
	"testing"
)

func TestProjectStatusMapsLifecycleWithoutInventingCompletion(t *testing.T) {
	const head = "1234567890abcdef"
	cases := []struct {
		name string
		in   StatusEvidence
		want DisplayState
	}{
		{"active owner", StatusEvidence{AssociationKnown: true, OwnerLive: true, Dirty: true}, DisplayActive},
		{"active lease", StatusEvidence{AssociationKnown: true, LeaseLive: true}, DisplayActive},
		{"dirty dead", StatusEvidence{AssociationKnown: true, Dirty: true}, DisplayUnlandedChanges},
		{"committed but unlanded", StatusEvidence{AssociationKnown: true, HeadSHA: head, BaseSHA: "base"}, DisplayUnlandedChanges},
		{"landed proof", StatusEvidence{AssociationKnown: true, HeadSHA: head, BaseSHA: "base", LandedWitnessed: true}, DisplayLandedWitnessed},
		{"cleanup is not completion", StatusEvidence{AssociationKnown: true, CleanupReady: true}, DisplayCleanupReady},
		{"unknown association", StatusEvidence{}, DisplayAssociationUnknown},
		{"insufficient stopped evidence", StatusEvidence{AssociationKnown: true}, DisplayAssociationUnknown},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := ProjectStatus(tc.in)
			if got.State != tc.want {
				t.Fatalf("state = %q, want %q", got.State, tc.want)
			}
			if got.State == DisplayLandedWitnessed && got.Commit != head[:12] {
				t.Fatalf("commit = %q, want scrubbed short SHA", got.Commit)
			}
		})
	}
}

func TestRenderIssueStatusCommentIsIdempotentAndPathFree(t *testing.T) {
	rows := []StatusProjection{
		ProjectStatus(StatusEvidence{IssueNumber: 10551, Lane: "workerworktree", Session: "worker-2", AssociationKnown: true, CleanupReady: true}),
		ProjectStatus(StatusEvidence{IssueNumber: 10551, Lane: "workerworktree", Session: "worker-1", AssociationKnown: true, Dirty: true}),
		ProjectStatus(StatusEvidence{IssueNumber: 9, Lane: "other", Session: "/Users/alice/private", AssociationKnown: true, Dirty: true}),
	}
	first, err := RenderIssueStatusComment("operator preface", 10551, rows)
	if err != nil {
		t.Fatal(err)
	}
	second, err := RenderIssueStatusComment(first, 10551, rows)
	if err != nil {
		t.Fatal(err)
	}
	if second != first {
		t.Fatalf("renderer is not idempotent:\nfirst=%q\nsecond=%q", first, second)
	}
	for _, forbidden := range []string{"/Users/", "private", "#9"} {
		if strings.Contains(first, forbidden) {
			t.Fatalf("comment leaked unrelated/local data %q:\n%s", forbidden, first)
		}
	}
	if strings.Index(first, "worker-1") > strings.Index(first, "worker-2") {
		t.Fatalf("rows are not deterministic:\n%s", first)
	}
	if !strings.Contains(first, "`unlanded_changes`") || !strings.Contains(first, "`cleanup_ready`") {
		t.Fatalf("comment omits lifecycle states:\n%s", first)
	}
}
