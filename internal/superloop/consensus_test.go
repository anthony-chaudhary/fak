package superloop

import (
	"strings"
	"testing"
)

func TestConsensus_SingleCandidatePass(t *testing.T) {
	diff := "--- a/pkg/foo.go\n+++ b/pkg/foo.go\n@@ -1,3 +1,3 @@\n-old\n+new\n"
	cand := PatchProposal{
		ID:     "cand-1",
		Diff:   diff,
		Score:  0.85,
		Passed: true,
		Files:  []string{"pkg/foo.go"},
	}

	winner, err := ClusterAndVotePatches([]PatchProposal{cand})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if winner == nil {
		t.Fatal("expected winner, got nil")
	}
	if winner.ID != "cand-1" {
		t.Errorf("expected winner ID 'cand-1', got %q", winner.ID)
	}
	if winner.Diff != diff {
		t.Errorf("expected winner Diff to match original diff, got %q", winner.Diff)
	}
	if winner.NormDiff == "" {
		t.Error("expected populated NormDiff on winner")
	}
}

func TestConsensus_MultipleCandidateClustersMajorityVote(t *testing.T) {
	diffA := "--- a/file.go\n+++ b/file.go\n@@ -1 +1 @@\n-x\n+a\n"
	diffB := "--- a/file.go\n+++ b/file.go\n@@ -1 +1 @@\n-x\n+b\n"

	proposals := []PatchProposal{
		{ID: "cand-a1", Diff: diffA, Score: 0.70, Passed: true},
		{ID: "cand-a2", Diff: diffA, Score: 0.90, Passed: true},
		{ID: "cand-b1", Diff: diffB, Score: 0.99, Passed: true},
		{ID: "cand-a3", Diff: diffA, Score: 0.80, Passed: true},
		{ID: "cand-b2", Diff: diffB, Score: 0.95, Passed: true},
	}

	winner, err := ClusterAndVotePatches(proposals)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if winner == nil {
		t.Fatal("expected winner, got nil")
	}
	// Cluster A has 3 votes, Cluster B has 2 votes.
	// Cluster A must win majority vote despite Cluster B having higher individual scores.
	if !strings.Contains(winner.Diff, "+a") {
		t.Errorf("expected majority cluster A to win, got diff: %s", winner.Diff)
	}
	// Within Cluster A, cand-a2 has the highest score (0.90).
	if winner.ID != "cand-a2" {
		t.Errorf("expected representative cand-a2 (highest score in cluster A), got %q", winner.ID)
	}
}

func TestConsensus_TieBreaking(t *testing.T) {
	diffA := "--- a/file.go\n+++ b/file.go\n@@ -1 +1 @@\n-x\n+a\n"
	diffB := "--- a/file.go\n+++ b/file.go\n@@ -1 +1 @@\n-x\n+b\n"

	t.Run("broken by score", func(t *testing.T) {
		proposals := []PatchProposal{
			{ID: "cand-a1", Diff: diffA, Score: 0.85, Passed: true},
			{ID: "cand-b1", Diff: diffB, Score: 0.95, Passed: true},
		}

		winner, err := ClusterAndVotePatches(proposals)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		// Each has 1 vote; cand-b1 has higher score.
		if winner.ID != "cand-b1" {
			t.Errorf("expected cand-b1 to win on score tie-break, got %q", winner.ID)
		}
	})

	t.Run("broken by candidate ID", func(t *testing.T) {
		proposals := []PatchProposal{
			{ID: "cand-z", Diff: diffB, Score: 0.90, Passed: true},
			{ID: "cand-a", Diff: diffA, Score: 0.90, Passed: true},
		}

		winner, err := ClusterAndVotePatches(proposals)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		// Equal votes and equal scores; cand-a wins lexicographical tie-break.
		if winner.ID != "cand-a" {
			t.Errorf("expected cand-a to win on ID tie-break, got %q", winner.ID)
		}
	})
}

func TestConsensus_FilterFailedCandidates(t *testing.T) {
	diffFail := "--- a/file.go\n+++ b/file.go\n@@ -1 +1 @@\n-x\n+fail\n"
	diffPass := "--- a/file.go\n+++ b/file.go\n@@ -1 +1 @@\n-x\n+pass\n"

	proposals := []PatchProposal{
		{ID: "fail-1", Diff: diffFail, Score: 0.99, Passed: false},
		{ID: "fail-2", Diff: diffFail, Score: 0.98, Passed: false},
		{ID: "fail-3", Diff: diffFail, Score: 0.97, Passed: false},
		{ID: "pass-1", Diff: diffPass, Score: 0.60, Passed: true},
	}

	winner, err := ClusterAndVotePatches(proposals)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if winner == nil {
		t.Fatal("expected winner, got nil")
	}
	if winner.ID != "pass-1" {
		t.Errorf("expected only passing candidate 'pass-1' to win, got %q", winner.ID)
	}
}

func TestConsensus_NormalizationWhitespaceCRLF(t *testing.T) {
	diffCRLF := "\r\n\r\nindex 1234567..89abcdef 100644\r\n--- a/pkg/mod.go\r\n+++ b/pkg/mod.go\r\n@@ -1,3 +1,3 @@\r\n-old    \r\n+new    \r\n\r\n\r\n"
	diffLF := "--- a/pkg/mod.go\t2026-09-04 12:00:00.000000000 +0000\n+++ b/pkg/mod.go\t2026-09-04 12:05:00.000000000 +0000\n@@ -1,3 +1,3 @@\n-old\n+new\n"

	norm1 := NormalizeDiff(diffCRLF)
	norm2 := NormalizeDiff(diffLF)

	if norm1 == "" {
		t.Fatal("NormalizeDiff returned empty string for diffCRLF")
	}
	if norm1 != norm2 {
		t.Fatalf("NormalizeDiff mismatch:\n--- norm1 ---\n%s\n--- norm2 ---\n%s", norm1, norm2)
	}

	// Verify they cluster together and accumulate votes
	proposals := []PatchProposal{
		{ID: "cand-crlf", Diff: diffCRLF, Score: 0.80, Passed: true},
		{ID: "cand-lf", Diff: diffLF, Score: 0.90, Passed: true},
	}

	clusters := ClusterPatches(proposals)
	if len(clusters) != 1 {
		t.Fatalf("expected 1 cluster from cosmetically different diffs, got %d", len(clusters))
	}
	if clusters[0].Votes != 2 {
		t.Errorf("expected 2 votes in cluster, got %d", clusters[0].Votes)
	}

	winner, err := ClusterAndVotePatches(proposals)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if winner.ID != "cand-lf" {
		t.Errorf("expected cand-lf (score 0.90) to be picked as representative, got %q", winner.ID)
	}
}

func TestConsensus_EmptyOrAllFailed(t *testing.T) {
	t.Run("nil slice", func(t *testing.T) {
		winner, err := ClusterAndVotePatches(nil)
		if err == nil {
			t.Errorf("expected error for nil proposals, got winner %+v", winner)
		}
	})

	t.Run("empty slice", func(t *testing.T) {
		winner, err := ClusterAndVotePatches([]PatchProposal{})
		if err == nil {
			t.Errorf("expected error for empty proposals, got winner %+v", winner)
		}
	})

	t.Run("all failed", func(t *testing.T) {
		proposals := []PatchProposal{
			{ID: "f1", Diff: "+1", Passed: false},
			{ID: "f2", Diff: "+2", Passed: false},
		}
		winner, err := ClusterAndVotePatches(proposals)
		if err == nil {
			t.Errorf("expected error when all proposals failed, got winner %+v", winner)
		}
	})
}
