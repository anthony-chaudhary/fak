package trajhook

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/trajectory"
)

func turn(trace string, seq int, query, verdict string, tokens int) trajectory.Turn {
	return trajectory.Turn{TraceID: trace, Seq: seq, Query: query, Verdict: verdict, TokenEstimate: tokens}
}

// TestDuplicateQueryFlagsParaphrase — the near-duplicate scorer flags a later turn
// whose query paraphrases an earlier one, even with different surface tokens, and
// names the earlier match. This is the signal lexical overlap misses.
func TestDuplicateQueryFlagsParaphrase(t *testing.T) {
	corpus := []trajectory.Turn{
		turn("a", 1, "delete all users from the accounts table", "ALLOW", 0),
		turn("a", 2, "what is the weather in London", "ALLOW", 0),
		turn("a", 3, "remove every user from the accounts table", "ALLOW", 0),
	}
	r := NewRegistry()
	r.Register("duplicate_query", DuplicateQuery(0.5))
	findings := r.Run(corpus)

	var dup *Finding
	for i := range findings {
		if findings[i].Label == "duplicate_query" && findings[i].Seq == 3 {
			dup = &findings[i]
		}
	}
	if dup == nil {
		t.Fatalf("paraphrase at a:3 not flagged; findings=%+v", findings)
	}
	if dup.Related != "a:1" {
		t.Fatalf("duplicate matched %q, want a:1", dup.Related)
	}
}

// TestDuplicateQueryIgnoresUnrelated — an unrelated query is not flagged as a
// duplicate (no false positives on distinct work).
func TestDuplicateQueryIgnoresUnrelated(t *testing.T) {
	corpus := []trajectory.Turn{
		turn("a", 1, "compile the kernel", "ALLOW", 0),
		turn("a", 2, "book a flight to Tokyo", "ALLOW", 0),
	}
	r := NewRegistry()
	r.Register("duplicate_query", DuplicateQuery(0.9))
	if f := r.Run(corpus); len(f) != 0 {
		t.Fatalf("unrelated queries flagged as duplicates: %+v", f)
	}
}

// TestCostOutlier — the cost scorer flags the expensive tail of the token
// distribution.
func TestCostOutlier(t *testing.T) {
	corpus := []trajectory.Turn{
		turn("a", 1, "cheap", "ALLOW", 100),
		turn("a", 2, "cheap", "ALLOW", 120),
		turn("a", 3, "cheap", "ALLOW", 110),
		turn("a", 4, "expensive", "ALLOW", 9000),
	}
	r := NewRegistry()
	r.RegisterCorpus("cost_outlier", CostOutlier(0.9))
	findings := r.Run(corpus)
	if len(findings) == 0 {
		t.Fatalf("no cost outlier flagged")
	}
	// The 9000-token turn must be flagged and be the top finding by score.
	if findings[0].Seq != 4 || findings[0].Label != "cost_outlier" {
		t.Fatalf("top finding is not the expensive turn: %+v", findings[0])
	}
}

// TestDenyRate — a trace the kernel kept refusing is flagged at the trace level.
func TestDenyRate(t *testing.T) {
	corpus := []trajectory.Turn{
		turn("bad", 1, "do a thing", "DENY", 0),
		turn("bad", 2, "do it anyway", "DENY", 0),
		turn("bad", 3, "please", "QUARANTINE", 0),
		turn("good", 1, "normal work", "ALLOW", 0),
		turn("good", 2, "more work", "ALLOW", 0),
	}
	r := NewRegistry()
	r.RegisterCorpus("deny_rate", DenyRate(0.5, 2))
	findings := r.Run(corpus)
	if len(findings) != 1 {
		t.Fatalf("deny-rate findings = %d, want 1: %+v", len(findings), findings)
	}
	if findings[0].TraceID != "bad" || findings[0].Score < 1.0 {
		t.Fatalf("wrong trace flagged or wrong score: %+v", findings[0])
	}
}

// TestDefaultRegistryWiresThree — the day-one toolkit registers exactly the three
// reference scorers, and Run tags each finding with its scorer name.
func TestDefaultRegistryWiresThree(t *testing.T) {
	names := Default().Names()
	if len(names) != 3 {
		t.Fatalf("default scorers = %v, want 3", names)
	}
	corpus := []trajectory.Turn{
		turn("a", 1, "delete users", "DENY", 50),
		turn("a", 2, "delete the users now", "DENY", 9000),
	}
	for _, f := range Default().Run(corpus) {
		if f.Scorer == "" {
			t.Fatalf("finding missing scorer name: %+v", f)
		}
	}
}

// TestRunSortedWorstFirst — Run returns findings sorted by descending score so a
// gardening skill reads the worst first.
func TestRunSortedWorstFirst(t *testing.T) {
	corpus := []trajectory.Turn{
		turn("a", 1, "x", "ALLOW", 100),
		turn("a", 2, "y", "ALLOW", 100),
		turn("a", 3, "z", "ALLOW", 5000),
	}
	r := NewRegistry()
	r.RegisterCorpus("cost_outlier", CostOutlier(0.5))
	findings := r.Run(corpus)
	for i := 1; i < len(findings); i++ {
		if findings[i-1].Score < findings[i].Score {
			t.Fatalf("not sorted worst-first: %+v", findings)
		}
	}
}
