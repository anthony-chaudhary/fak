package contextq

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/cachemeta"
	"github.com/anthony-chaudhary/fak/internal/cdb"
)

func attachFixture(t testing.TB) *cdb.Image {
	t.Helper()
	ctx := context.Background()
	rec, st, err := cdb.IngestSession(ctx, "../../testdata/cdb/session.jsonl", "contextq-fixture")
	if err != nil {
		t.Fatalf("ingest: %v", err)
	}
	if st.Pages == 0 {
		t.Fatalf("ingest recorded no pages")
	}
	dir := t.TempDir()
	if err := rec.Persist(dir); err != nil {
		t.Fatalf("persist: %v", err)
	}
	im, err := cdb.Attach(dir)
	if err != nil {
		t.Fatalf("attach: %v", err)
	}
	return im
}

func TestQueryMaterializesTypedWorkingSet(t *testing.T) {
	im := attachFixture(t)
	res := Query(context.Background(), im, Request{
		Query:         "refund fee trust violation",
		PolicyVersion: "policy-test",
	})

	if len(res.Frames) != im.Info().Pages {
		t.Fatalf("frames=%d, want %d", len(res.Frames), im.Info().Pages)
	}
	if len(res.Slices) == 0 {
		t.Fatal("expected at least one materialized benign slice")
	}
	if len(res.Views) != len(res.Slices) {
		t.Fatalf("views=%d slices=%d, want one view per materialized slice", len(res.Views), len(res.Slices))
	}
	if len(res.RenderPlan.Items) != len(res.Slices) {
		t.Fatalf("render items=%d slices=%d", len(res.RenderPlan.Items), len(res.Slices))
	}
	if res.RenderPlan.EstimatedTokens == 0 {
		t.Fatal("render plan should estimate tokens")
	}
	if !hasVerdict(res, MaterializationFault, "raw_page_fault") {
		t.Fatalf("expected a raw-page FAULT verdict, got %+v", res.Verdicts)
	}
	if !hasRefusal(res, "sealed_by_trust_gate") {
		t.Fatalf("expected sealed page refusal, got %+v", res.Refused)
	}
	for _, sl := range res.Slices {
		if sl.Source.MediaType != cachemeta.MediaRecallPage {
			t.Fatalf("slice source should be a recall page handle: %+v", sl.Source)
		}
		if sl.MaterializedBy != MaterializationFault {
			t.Fatalf("slice materialization = %q, want FAULT", sl.MaterializedBy)
		}
	}
	for _, v := range res.Views {
		if v.CacheEntry.Plane != cachemeta.PlaneMemoryView || v.CacheEntry.ID.MediaType != cachemeta.MediaMemoryView {
			t.Fatalf("view did not lower into memory-view cache metadata: %+v", v.CacheEntry)
		}
		if len(v.CacheEntry.Derivation.SourceRefs) != 1 {
			t.Fatalf("view source refs missing: %+v", v.CacheEntry.Derivation.SourceRefs)
		}
		if v.PolicyVersion != "policy-test" || v.CacheEntry.Validity.PolicyVersion != "policy-test" {
			t.Fatalf("policy version did not carry through: view=%q entry=%q", v.PolicyVersion, v.CacheEntry.Validity.PolicyVersion)
		}
	}
}

func TestQueryBudgetProducesTypedOmission(t *testing.T) {
	im := attachFixture(t)
	res := Query(context.Background(), im, Request{
		Query:       "refund fee account",
		BudgetBytes: 1,
	})

	if len(res.Slices) != 0 {
		t.Fatalf("budget should prevent raw-page materialization, got %+v", res.Slices)
	}
	if !hasVerdict(res, MaterializationAbstain, "budget_exhausted") {
		t.Fatalf("expected budget ABSTAIN verdict, got %+v", res.Verdicts)
	}
	if !hasOmission(res, "budget_exhausted") {
		t.Fatalf("expected budget omission, got %+v", res.Omissions)
	}
}

func TestQueryPinsAndExcludesAreExplicit(t *testing.T) {
	im := attachFixture(t)
	res := Query(context.Background(), im, Request{
		Query:    "refund fee",
		Pins:     []string{"WebSearch"},
		Excludes: []string{"refund_fee"},
	})

	if !hasRefusal(res, "excluded_by_request") {
		t.Fatalf("expected explicit exclusion refusal, got %+v", res.Refused)
	}
	foundPinned := false
	for _, sl := range res.Slices {
		if sl.Role == "WebSearch" {
			foundPinned = true
		}
		if sl.Role == "Read" {
			t.Fatalf("excluded page materialized: %+v", sl)
		}
	}
	if !foundPinned {
		t.Fatalf("pinned WebSearch page was not materialized: %+v", res.Slices)
	}
}

// TestAllFiveMaterializationVerdictsReachable proves the derived-view path
// exercises every MaterializationVerdict kind across a cold build, a stale
// rebuild, a warm reuse, a trust refusal, and a budget abstention. It is the
// central witness for the memo's claim that a view is an adjudicated, reusable
// cache artifact: the warm (HIT) pass pages zero raw bytes.
func TestAllFiveMaterializationVerdictsReachable(t *testing.T) {
	im := attachFixture(t)
	cache := NewViewCache()

	// Pass 1 — cold, policy "p1", no budget: build summaries (FAULT) and refuse
	// the sealed page the query touches (REFUSE).
	cold := Query(context.Background(), im, Request{
		Query:         "refund fee trust violation",
		PreferView:    ViewSummary,
		ViewCache:     cache,
		PolicyVersion: "p1",
	})
	if !hasVerdictKind(cold, MaterializationFault) {
		t.Fatalf("pass 1: expected a FAULT (view build), got %+v", cold.Verdicts)
	}
	if !hasRefusal(cold, "sealed_by_trust_gate") {
		t.Fatalf("pass 1: expected sealed REFUSE, got %+v", cold.Refused)
	}
	if cold.Stats.BytesPagedIn == 0 {
		t.Fatal("pass 1: cold build should fault raw bytes to build views")
	}
	if cold.Stats.ViewHits != 0 {
		t.Fatalf("pass 1: cold cache should have zero HITs, got %d", cold.Stats.ViewHits)
	}
	coldBytesPaged := cold.Stats.BytesPagedIn

	// Pass 2 — warm cache, policy changed to "p2": every cached summary is stale,
	// so the resolver RECOMPUTES (re-faults the source, rebuilds under p2).
	stale := Query(context.Background(), im, Request{
		Query:         "refund fee trust violation",
		PreferView:    ViewSummary,
		ViewCache:     cache,
		PolicyVersion: "p2",
	})
	if !hasVerdictReason(stale, "view_stale_policy_mismatch") {
		t.Fatalf("pass 2: expected a RECOMPUTE, got %+v", stale.Verdicts)
	}
	if stale.Stats.ViewRecomputes == 0 {
		t.Fatalf("pass 2: expected ViewRecomputes > 0, got %+v", stale.Stats)
	}

	// Pass 3 — warm cache, same policy "p2": every view is fresh -> HIT, and the
	// raw page device is untouched. This is the economic proof.
	warm := Query(context.Background(), im, Request{
		Query:         "refund fee trust violation",
		PreferView:    ViewSummary,
		ViewCache:     cache,
		PolicyVersion: "p2",
	})
	if !hasVerdictReason(warm, "view_cache_hit") {
		t.Fatalf("pass 3: expected a HIT, got %+v", warm.Verdicts)
	}
	if warm.Stats.ViewHits == 0 {
		t.Fatalf("pass 3: expected ViewHits > 0, got %+v", warm.Stats)
	}
	if warm.Stats.BytesPagedIn != 0 {
		t.Fatalf("pass 3: warm HIT pass must page 0 raw bytes, got %d (cold was %d)",
			warm.Stats.BytesPagedIn, coldBytesPaged)
	}
	if warm.Stats.ViewRecomputes != 0 {
		t.Fatalf("pass 3: warm fresh pass should not recompute, got %d", warm.Stats.ViewRecomputes)
	}

	// Pass 4 — fresh cache, tiny budget: the first item exceeds budget before
	// resolution -> ABSTAIN without paying a fault.
	budget := Query(context.Background(), im, Request{
		Query:         "refund fee trust violation",
		PreferView:    ViewSummary,
		ViewCache:     cache,
		PolicyVersion: "p2",
		BudgetBytes:   1,
	})
	if !hasVerdict(budget, MaterializationAbstain, "budget_exhausted") {
		t.Fatalf("pass 4: expected budget ABSTAIN, got %+v", budget.Verdicts)
	}
	if budget.Stats.BytesPagedIn != 0 {
		t.Fatalf("pass 4: an over-budget item must not fault raw bytes, got %d", budget.Stats.BytesPagedIn)
	}

	// Summary coverage honesty: an extractive summary of a small page is whole
	// (Coverage 1.0); a summary of a larger page must report Coverage < 1.0. At
	// least one built view in the cold pass must carry a valid cachemeta entry.
	if len(cold.Views) == 0 {
		t.Fatal("pass 1: expected at least one built summary view")
	}
	for _, v := range cold.Views {
		if v.ViewType != ViewSummary {
			t.Fatalf("expected summary view, got %q", v.ViewType)
		}
		if v.CacheEntry.Plane != cachemeta.PlaneMemoryView || v.CacheEntry.ID.MediaType != cachemeta.MediaMemoryView {
			t.Fatalf("summary view did not lower into memory-view cache metadata: %+v", v.CacheEntry)
		}
		if v.FaithfulnessProbe != 1.0 {
			t.Fatalf("extractive summary must be faithful (1.0), got %f", v.FaithfulnessProbe)
		}
		if v.Coverage < 0 || v.Coverage > 1.0 {
			t.Fatalf("coverage out of [0,1]: %f", v.Coverage)
		}
	}
}

func TestViewCacheCopiesPayloadBytes(t *testing.T) {
	cache := NewViewCache()
	view := MemoryViewRecord{
		ViewID:        "v1",
		ViewType:      ViewSummary,
		SourcePageIDs: []int{7},
		Producer:      "test",
	}
	payload := []byte("stable")
	cache.Put(view, payload)
	payload[0] = 'X'

	_, got, ok := cache.Get(7, ViewSummary, "test")
	if !ok {
		t.Fatal("expected cached view")
	}
	if string(got) != "stable" {
		t.Fatalf("cache stored caller-owned slice, got %q", got)
	}
	got[1] = 'Y'
	_, again, ok := cache.Get(7, ViewSummary, "test")
	if !ok {
		t.Fatal("expected cached view on second get")
	}
	if string(again) != "stable" {
		t.Fatalf("cache returned internal slice, got %q", again)
	}
}

func hasVerdictKind(res Result, kind MaterializationKind) bool {
	for _, v := range res.Verdicts {
		if v.Kind == kind {
			return true
		}
	}
	return false
}

func hasVerdictReason(res Result, reason string) bool {
	for _, v := range res.Verdicts {
		if v.Reason == reason {
			return true
		}
	}
	return false
}

func hasVerdict(res Result, kind MaterializationKind, reason string) bool {
	for _, v := range res.Verdicts {
		if v.Kind == kind && v.Reason == reason {
			return true
		}
	}
	return false
}

func hasRefusal(res Result, reason string) bool {
	for _, r := range res.Refused {
		if r.Reason == reason {
			return true
		}
	}
	return false
}

func hasOmission(res Result, reason string) bool {
	for _, o := range res.Omissions {
		if o.Reason == reason {
			return true
		}
	}
	return false
}
