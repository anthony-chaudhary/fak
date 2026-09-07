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

func TestFold_PrivateIssueInSubject_IsFinding(t *testing.T) {
	// When a commit quotes a fak-private issue (e.g. fak-private#21), it is clearly
	// from fak-private and must NOT be mistaken for a public #N issue reference.
	// Therefore the commit is still reported as missing a public issue link in its subject.
	rep := Fold([]Commit{
		{
			SHA:     "priv1",
			Subject: "fix(adjudicator): bind confirm token past client supervision knobs (fak-private#21) (fak adjudicator)",
			Body:    "Regression witness for companion fak-private#21.",
		},
	})
	if len(rep.Findings) != 1 {
		t.Fatalf("want 1 finding for commit with only fak-private#21 reference, got %d: %+v", len(rep.Findings), rep.Findings)
	}
	f := rep.Findings[0]
	if len(f.PrivateIssues) != 1 || f.PrivateIssues[0] != 21 {
		t.Errorf("expected PrivateIssues=[21], got %+v", f.PrivateIssues)
	}
}

func TestFold_PublicAndPrivateIssues_NotAFinding(t *testing.T) {
	// A commit that carries BOTH a public #12013 reference and a qualified fak-private#21
	// reference satisfies the public subject issue requirement while keeping the private
	// provenance clear.
	rep := Fold([]Commit{
		{
			SHA:     "pubpriv1",
			Subject: "fix(adjudicator): handle supervision knobs #12013 (fak-private#21) (fak adjudicator)",
			Body:    "Addresses public issue #12013 informed by private incident fak-private#21.",
		},
	})
	if len(rep.Findings) != 0 {
		t.Fatalf("want 0 findings for commit with public #12013 and private fak-private#21, got %+v", rep.Findings)
	}
}

func TestCommitTextNamesIssue_IgnoresPrivateIssue(t *testing.T) {
	reachable := true
	// Commit mentions fak-private#21, NOT public issue #21.
	rep := FoldUnresolvedCommitLinkedIssues([]CommitLinkedIssue{
		{
			Number:       21,
			SHA:          "priv21",
			Subject:      "fix(adjudicator): handle timeout (fak-private#21) (fak adjudicator)",
			Body:         "Addresses internal appeal in fak-private#21.",
			AuditVerdict: "OK",
			AuditWitness: "diff-witnessed",
			Reachable:    &reachable,
		},
	})
	if len(rep.Findings) != 1 {
		t.Fatalf("want 1 finding because commit text refers to fak-private#21, not public #21, got: %+v", rep.Findings)
	}
	if rep.Findings[0].Reason != ReasonMissingIssueLink {
		t.Errorf("expected ReasonMissingIssueLink, got %q", rep.Findings[0].Reason)
	}

	// Commit mentions public #21 and also quotes fak-private#456.
	rep2 := FoldUnresolvedCommitLinkedIssues([]CommitLinkedIssue{
		{
			Number:       21,
			SHA:          "pub21",
			Subject:      "fix(adjudicator): handle timeout #21 (fak-private#456) (fak adjudicator)",
			Body:         "Addresses public #21.",
			AuditVerdict: "OK",
			AuditWitness: "diff-witnessed",
			Reachable:    &reachable,
		},
	})
	if len(rep2.Findings) != 0 {
		t.Fatalf("want 0 findings for commit naming public #21, got: %+v", rep2.Findings)
	}
}

func TestPrivateIssueHelpers(t *testing.T) {
	text := "Incident reported in fak-private#456 and anthony-chaudhary/fak-private#789, see also #123."
	nums := ExtractPrivateIssues(text)
	if len(nums) != 2 || nums[0] != 456 || nums[1] != 789 {
		t.Errorf("ExtractPrivateIssues(%q) = %v, want [456, 789]", text, nums)
	}

	if got := FormatPrivateIssue(456); got != "fak-private#456" {
		t.Errorf("FormatPrivateIssue(456) = %q, want 'fak-private#456'", got)
	}
	if got := FormatPublicIssue(123); got != "#123" {
		t.Errorf("FormatPublicIssue(123) = %q, want '#123'", got)
	}

	if !IsPrivateIssueRef("fak-private#100") {
		t.Errorf("expected IsPrivateIssueRef('fak-private#100') == true")
	}
	if !IsPrivateIssueRef("anthony-chaudhary/fak-private#100") {
		t.Errorf("expected IsPrivateIssueRef('anthony-chaudhary/fak-private#100') == true")
	}
	if IsPrivateIssueRef("#100") {
		t.Errorf("expected IsPrivateIssueRef('#100') == false")
	}
}
