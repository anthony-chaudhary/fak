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

func TestFoldUnresolvedCommitLinkedIssues_MapsReasons(t *testing.T) {
	reachable := true
	stale := false
	rep := FoldUnresolvedCommitLinkedIssues([]CommitLinkedIssue{
		{
			Number:       10,
			SHA:          "aaa111",
			Subject:      "fix(dispatch): close shipped work (fak cmd)",
			AuditVerdict: "OK",
			AuditWitness: "diff-witnessed",
			Reachable:    &reachable,
		},
		{
			Number:       11,
			SHA:          "bbb222",
			Subject:      "fix(dispatch): close #11 (fak cmd)",
			AuditVerdict: "FAIL",
			AuditWitness: "diff-witnessed",
			Reachable:    &reachable,
		},
		{
			Number:       12,
			SHA:          "ccc333",
			Subject:      "fix(dispatch): close #12 (fak cmd)",
			AuditVerdict: "OK",
			AuditWitness: "diff-witnessed",
			Reachable:    &stale,
		},
		{
			Number:       13,
			SHA:          "ddd444",
			Subject:      "fix(dispatch): close #13 (fak cmd)",
			AuditVerdict: "OK",
			AuditWitness: "subject-only",
			Reachable:    &reachable,
		},
		{
			Number:       14,
			SHA:          "eee555",
			Subject:      "fix(dispatch): close #14 (fak cmd)",
			AuditVerdict: "OK",
			AuditWitness: "diff-witnessed",
			Reachable:    &reachable,
		},
	})
	if rep.Scanned != 5 {
		t.Fatalf("scanned = %d, want 5", rep.Scanned)
	}
	got := map[int]string{}
	for _, f := range rep.Findings {
		got[f.Number] = f.Reason
	}
	want := map[int]string{
		10: ReasonMissingIssueLink,
		11: ReasonFailedAudit,
		12: ReasonStaleSHA,
		13: ReasonInsufficientDiffEvidence,
	}
	if len(got) != len(want) {
		t.Fatalf("findings = %+v, want one per unresolved reason", rep.Findings)
	}
	for number, reason := range want {
		if got[number] != reason {
			t.Fatalf("#%d reason = %q, want %q (all findings %+v)", number, got[number], reason, rep.Findings)
		}
	}
	if _, ok := got[14]; ok {
		t.Fatalf("fully witnessed issue #14 must not be a finding: %+v", rep.Findings)
	}
}

func TestFold_Empty(t *testing.T) {
	rep := Fold(nil)
	if rep.Scanned != 0 || len(rep.Findings) != 0 {
		t.Fatalf("want a zero report for no commits, got %+v", rep)
	}
}
