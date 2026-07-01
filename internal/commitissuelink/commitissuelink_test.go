package commitissuelink

import "testing"

func TestFold_GoodSubject_NotAFinding(t *testing.T) {
	rep := Fold([]Commit{
		{SHA: "aaa", Subject: "feat(audit): add fak audit usage cross-session rollup #1612 (fak audit)"},
	})
	if len(rep.Findings) != 0 {
		t.Fatalf("want no findings for a subject that already carries #N, got %+v", rep.Findings)
	}
	if rep.Scanned != 1 {
		t.Fatalf("want scanned=1, got %d", rep.Scanned)
	}
}

func TestFold_MissingLinkSubject_GuessesFromBodyTrailer(t *testing.T) {
	rep := Fold([]Commit{
		{
			SHA:     "bbb",
			Subject: "feat(audit): add fak audit usage cross-session rollup",
			Body:    "Cross-session usage rollup over every durable sink.\n\nFixes #1612\n\n(fak audit)",
		},
	})
	if len(rep.Findings) != 1 {
		t.Fatalf("want 1 finding, got %d: %+v", len(rep.Findings), rep.Findings)
	}
	f := rep.Findings[0]
	if f.SHA != "bbb" || f.GuessedIssue != "1612" {
		t.Errorf("want sha=bbb guessed_issue=1612, got %+v", f)
	}
}

func TestFold_MissingLinkSubject_NoBodyTrailer_NoGuess(t *testing.T) {
	rep := Fold([]Commit{
		{
			SHA:     "ccc",
			Subject: "fix(guard): tighten the deny path (fak guard)",
			Body:    "No issue trailer here, just a fix.",
		},
	})
	if len(rep.Findings) != 1 {
		t.Fatalf("want 1 finding, got %d: %+v", len(rep.Findings), rep.Findings)
	}
	if got := rep.Findings[0].GuessedIssue; got != "" {
		t.Errorf("want no guess without a body trailer, got %q", got)
	}
}

func TestFold_NoLeafTrailer_NotHeldToTheBar(t *testing.T) {
	rep := Fold([]Commit{
		{SHA: "ddd", Subject: "typo: fix a comment", Body: "just a typo, no tracked issue"},
	})
	if len(rep.Findings) != 0 {
		t.Fatalf("want no findings for a commit with no ship-stamp trailer, got %+v", rep.Findings)
	}
}

func TestFold_BodyIssueRefWithoutSubjectRef_StillAFinding(t *testing.T) {
	// A #N anywhere in the BODY (not as a Fixes/Closes/Resolves trailer) does
	// not excuse a missing subject-line #N -- only the subject is scanned for
	// the "already linked" exemption.
	rep := Fold([]Commit{
		{SHA: "eee", Subject: "chore(deps): bump toolchain (fak build)", Body: "see discussion in #999 for context"},
	})
	if len(rep.Findings) != 1 {
		t.Fatalf("want 1 finding, got %d: %+v", len(rep.Findings), rep.Findings)
	}
	if got := rep.Findings[0].GuessedIssue; got != "" {
		t.Errorf("want no guess -- #999 is not a Fixes/Closes/Resolves trailer -- got %q", got)
	}
}

func TestFold_Empty(t *testing.T) {
	rep := Fold(nil)
	if rep.Scanned != 0 || len(rep.Findings) != 0 {
		t.Fatalf("want a zero report for no commits, got %+v", rep)
	}
}
