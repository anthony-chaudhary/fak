package dispatchstatus

import (
	"testing"
	"time"
)

func i64(v int64) *int64 { return &v }

func TestLeaseLane(t *testing.T) {
	cases := map[string]string{
		"resolve-engine-123": "engine",
		"resolve-model":      "model",
		"resolve-gateway-7":  "gateway",
		"session-abc":        "session-abc", // non-resolve id unchanged
		"resolve-":           "resolve-",    // empty lane -> id
		"  resolve-docs  ":   "docs",        // trimmed
	}
	for in, want := range cases {
		if got := LeaseLane(in); got != want {
			t.Errorf("LeaseLane(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestTreesOverlap(t *testing.T) {
	if !TreesOverlap([]string{"internal/engine/**"}, []string{"internal/engine"}) {
		t.Error("equal-after-normalize trees should overlap")
	}
	if !TreesOverlap([]string{"internal/engine"}, []string{"internal/engine/sub"}) {
		t.Error("parent/child trees should overlap")
	}
	if TreesOverlap([]string{"docs"}, []string{"internal/engine"}) {
		t.Error("disjoint trees should not overlap")
	}
	if !TreesOverlap(nil, []string{"internal/engine"}) {
		t.Error("empty (unknown scope) should overlap everything")
	}
	if !TreesOverlap([]string{"**"}, []string{"docs"}) {
		t.Error("** should overlap everything")
	}
}

func backlogFixture() Backlog {
	return Backlog{
		Lanes: map[string]BacklogLane{
			"engine": {Tree: []string{"internal/engine/**"}, Issues: []int{42}},
		},
		Issues: []BacklogIssue{{Number: 42, Lane: "engine", Confidence: "path-confirmed"}},
	}
}

func TestSummarizeLeasesClassifiesAndBlocks(t *testing.T) {
	now := time.Unix(1_000_000, 0)
	records := []LeaseRecord{
		{ID: "resolve-docs", TreeGlobs: []string{"docs/**"}, AcquiredUnix: i64(now.Unix() - 60), TTLSeconds: 3600, Holder: "w2"},
		{ID: "resolve-engine-1", TreeGlobs: []string{"internal/engine/**"}, AcquiredUnix: i64(now.Unix() - 600), TTLSeconds: 3600, Holder: "w1"},
		{ID: "resolve-old-2", TreeGlobs: []string{"internal/old"}, AcquiredUnix: i64(now.Unix() - 5000), TTLSeconds: 1000, Holder: "w3"},
		{ID: "  ", TTLSeconds: 10}, // blank id skipped
	}
	st := SummarizeLeases(records, backlogFixture(), now)

	if st.ActiveCount != 2 {
		t.Fatalf("active_count = %d, want 2", st.ActiveCount)
	}
	if st.ExpiredCount != 1 {
		t.Fatalf("expired_count = %d, want 1", st.ExpiredCount)
	}
	if st.BlockingCount != 1 {
		t.Fatalf("blocking_count = %d, want 1", st.BlockingCount)
	}
	if !st.CandidateSourceAvailable || st.CandidateCount != 1 {
		t.Fatalf("candidate source avail=%v count=%d, want true/1", st.CandidateSourceAvailable, st.CandidateCount)
	}
	// Blocking lease sorts first.
	if st.Active[0].ID != "resolve-engine-1" {
		t.Fatalf("active[0] = %q, want resolve-engine-1 (blocking first)", st.Active[0].ID)
	}
	if st.Active[0].BlocksCandidate == nil || !*st.Active[0].BlocksCandidate {
		t.Fatalf("engine lease should block a candidate")
	}
	if len(st.Active[0].BlockingCandidates) != 1 || st.Active[0].BlockingCandidates[0].Issue != 42 {
		t.Fatalf("blocking candidate = %+v, want issue 42", st.Active[0].BlockingCandidates)
	}
	// docs lease: live but not blocking.
	if st.Active[1].ID != "resolve-docs" || st.Active[1].BlocksCandidate == nil || *st.Active[1].BlocksCandidate {
		t.Fatalf("docs lease should be live non-blocking, got %+v", st.Active[1])
	}
	// expired residue.
	if st.Expired[0].ID != "resolve-old-2" || st.Expired[0].Status != "EXPIRED" {
		t.Fatalf("expired[0] = %+v, want resolve-old-2 EXPIRED", st.Expired[0])
	}
	if st.Expired[0].BlocksCandidate == nil || *st.Expired[0].BlocksCandidate {
		t.Fatalf("expired lease must never block")
	}
	// age/expiry on the engine lease.
	if st.Active[0].AgeSeconds == nil || *st.Active[0].AgeSeconds != 600 {
		t.Fatalf("engine age_seconds = %v, want 600", st.Active[0].AgeSeconds)
	}
	if st.Active[0].ExpiresInSeconds == nil || *st.Active[0].ExpiresInSeconds != 3000 {
		t.Fatalf("engine expires_in = %v, want 3000", st.Active[0].ExpiresInSeconds)
	}
}

func TestSummarizeLeasesUnavailableCandidateSource(t *testing.T) {
	now := time.Unix(1_000_000, 0)
	records := []LeaseRecord{
		{ID: "resolve-engine-1", TreeGlobs: []string{"internal/engine/**"}, AcquiredUnix: i64(now.Unix() - 60), TTLSeconds: 3600},
	}
	// HasSkipped => candidate source not trustworthy => blocks_candidate is null.
	bl := backlogFixture()
	bl.HasSkipped = true
	st := SummarizeLeases(records, bl, now)
	if st.CandidateSourceAvailable {
		t.Fatal("candidate_source_available should be false when HasSkipped")
	}
	if st.Active[0].BlocksCandidate != nil {
		t.Fatalf("blocks_candidate = %v, want nil when candidate source unavailable", st.Active[0].BlocksCandidate)
	}
	if st.BlockingCount != 0 {
		t.Fatalf("blocking_count = %d, want 0", st.BlockingCount)
	}
}
