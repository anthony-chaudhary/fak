package commitissuelink

import (
	"testing"
)

// BenchmarkCommitIssueLink measures throughput and allocation overhead of
// commit issue link scanning and folding operations over representative commit batches.
func BenchmarkCommitIssueLink(b *testing.B) {
	commits := []Commit{
		{
			SHA:     "a1b2c3d4e5f6",
			Subject: "feat(audit): add fak audit usage cross-session rollup #1612 (fak audit)",
			Body:    "Detailed body explaining rollup metrics across sessions.",
		},
		{
			SHA:     "b2c3d4e5f6a1",
			Subject: "feat(audit): missing subject issue reference (fak audit)",
			Body:    "Resolved after testing.\n\nFixes #1612\n\n(fak audit)",
		},
		{
			SHA:     "c3d4e5f6a1b2",
			Subject: "fix(guard): sanitize inputs without link (fak guard)",
			Body:    "No trailer reference provided here.",
		},
		{
			SHA:     "d4e5f6a1b2c3",
			Subject: "typo: clean up comment formatting",
			Body:    "Trivial typo fix not requiring issue tracking.",
		},
	}

	reachable := true
	stale := false
	linkedIssues := []CommitLinkedIssue{
		{
			Number:       10,
			SHA:          "aaa111222333",
			Subject:      "fix(dispatch): close shipped work (fak cmd)",
			AuditVerdict: "OK",
			AuditWitness: "diff-witnessed",
			Reachable:    &reachable,
		},
		{
			Number:       11,
			SHA:          "bbb222333444",
			Subject:      "fix(dispatch): close #11 (fak cmd)",
			AuditVerdict: "FAIL",
			AuditWitness: "diff-witnessed",
			Reachable:    &reachable,
		},
		{
			Number:       12,
			SHA:          "ccc333444555",
			Subject:      "fix(dispatch): close #12 (fak cmd)",
			AuditVerdict: "OK",
			AuditWitness: "diff-witnessed",
			Reachable:    &stale,
		},
		{
			Number:       13,
			SHA:          "ddd444555666",
			Subject:      "fix(dispatch): close #13 (fak cmd)",
			AuditVerdict: "OK",
			AuditWitness: "subject-only",
			Reachable:    &reachable,
		},
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		rep := Fold(commits)
		if rep.Scanned != len(commits) {
			b.Fatalf("unexpected scanned count: %d", rep.Scanned)
		}
		unresolved := FoldUnresolvedCommitLinkedIssues(linkedIssues)
		if unresolved.Scanned != len(linkedIssues) {
			b.Fatalf("unexpected unresolved scanned count: %d", unresolved.Scanned)
		}
	}
}
