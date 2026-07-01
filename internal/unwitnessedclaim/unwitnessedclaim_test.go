package unwitnessedclaim

import (
	"strings"
	"testing"
)

func TestEvaluate_ClosedIssue_NeverFlagged(t *testing.T) {
	rep := Evaluate(Input{
		IssueNumber: 42,
		Open:        false,
		Comments:    []Comment{{Author: "alice", Body: "done, this shipped"}},
	})
	if rep.Flagged {
		t.Fatalf("want not flagged for a closed issue, got %+v", rep)
	}
}

func TestEvaluate_NoComments_NeverFlagged(t *testing.T) {
	rep := Evaluate(Input{IssueNumber: 42, Open: true})
	if rep.Flagged {
		t.Fatalf("want not flagged with no comments, got %+v", rep)
	}
}

func TestEvaluate_LatestCommentNoClaimWords_NotFlagged(t *testing.T) {
	rep := Evaluate(Input{
		IssueNumber: 42,
		Open:        true,
		Comments:    []Comment{{Author: "bob", Body: "still investigating, no repro yet"}},
	})
	if rep.Flagged {
		t.Fatalf("want not flagged for a non-claim comment, got %+v", rep)
	}
}

func TestEvaluate_LatestCommentClaimsDone_Flagged(t *testing.T) {
	rep := Evaluate(Input{
		IssueNumber: 1816,
		Open:        true,
		Comments:    []Comment{{Author: "carol", Body: "This is done, all tests pass."}},
	})
	if !rep.Flagged {
		t.Fatalf("want flagged for a completion claim on an open issue, got %+v", rep)
	}
	if rep.ClaimAuthor != "carol" {
		t.Errorf("want claim_author=carol, got %q", rep.ClaimAuthor)
	}
	if rep.CommentBody == "" {
		t.Fatal("want a rendered comment body")
	}
	if !strings.Contains(rep.CommentBody, "#1816") || !strings.Contains(rep.CommentBody, "@carol") {
		t.Errorf("comment body should name the issue and author, got %q", rep.CommentBody)
	}
	if !strings.Contains(rep.CommentBody, "close") {
		t.Errorf("comment body should mention closing behavior, got %q", rep.CommentBody)
	}
}

func TestEvaluate_OnlyLatestCommentChecked(t *testing.T) {
	rep := Evaluate(Input{
		IssueNumber: 42,
		Open:        true,
		Comments: []Comment{
			{Author: "dave", Body: "I think this is done"},
			{Author: "erin", Body: "actually reopening, found a regression"},
		},
	})
	if rep.Flagged {
		t.Fatalf("want not flagged -- only the latest comment counts, got %+v", rep)
	}
}

func TestEvaluate_WordBoundary_NoFalseMatchOnSubstring(t *testing.T) {
	rep := Evaluate(Input{
		IssueNumber: 42,
		Open:        true,
		Comments:    []Comment{{Author: "frank", Body: "this remains unfixed and undone as far as I can tell"}},
	})
	if rep.Flagged {
		t.Fatalf("want not flagged -- 'unfixed'/'undone' must not match the word-boundary claim regex, got %+v", rep)
	}
}
