package sessionimage

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/cachemeta"
	"github.com/anthony-chaudhary/fak/internal/recall"
	"github.com/anthony-chaudhary/fak/internal/session"
)

// TestRehydrateCarriesCachemetaInvalidationState is the #1536 witness: a dumped image that
// carries recall content pages, when rehydrated, yields not only the ctxplan Index (as today)
// but ALSO cachemeta.Entry records with EXPLICIT cache-invalidation state — the archived-work
// "queryable history AND explicit invalidation state" resume target of Next-50 item 18.
//
// The page is recorded under an external trust witness, so its lowered entry must carry a set
// Coherence.InvalidationMode (external refutation) and a set Validity (witness/trust epoch),
// not the zero value. It also asserts cold-path correctness stays explicit: the record sits on
// TierDisk, never a resident hot tier, so it can never be mistaken for a live cache hit.
func TestRehydrateCarriesCachemetaInvalidationState(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	const id = "sess-cachemeta"

	rec := recall.NewRecorder(id)
	// A benign page admitted under an external trust witness — FromContextPage maps a
	// witnessed page to InvalidationExternalRefutation with a populated Validity.
	rec.RecordWithWitness(ctx, "get_user_details", []byte(benignAccount), "git:abc123")
	rec.Record(ctx, "search_flights", []byte(benignFlights)) // a second content page

	in := Input{
		SessionID: id,
		Drive:     session.DefaultState(id),
		Recorder:  rec,
		Now:       1_700_000_000,
	}
	if _, err := DumpDir(dir, in); err != nil {
		t.Fatalf("DumpDir: %v", err)
	}
	img, err := LoadDir(dir)
	if err != nil {
		t.Fatalf("LoadDir: %v", err)
	}

	res, err := img.Rehydrate(ctx, RehydrateOptions{Table: session.NewTable()})
	if err != nil {
		t.Fatalf("Rehydrate: %v", err)
	}

	// The ctxplan half is unchanged: the Index must still be restored alongside.
	if res.Index == nil {
		t.Fatal("rehydrate dropped the ctxplan Index — item 18 adds the cachemeta record, it must NOT replace the history path")
	}

	// The cachemeta half: one record per content page, with explicit invalidation state.
	if len(res.CacheEntries) != 2 {
		t.Fatalf("CacheEntries = %d, want 2 (one per rehydrated content page)", len(res.CacheEntries))
	}

	// Find the witnessed page's entry and assert its invalidation state is SET (not zero).
	var witnessed *cachemeta.Entry
	for i := range res.CacheEntries {
		if res.CacheEntries[i].Validity.Witness == "git:abc123" {
			witnessed = &res.CacheEntries[i]
		}
	}
	if witnessed == nil {
		t.Fatalf("no rehydrated entry carried the recorded witness: %+v", res.CacheEntries)
	}
	if witnessed.Coherence.InvalidationMode != cachemeta.InvalidationExternalRefutation {
		t.Fatalf("witnessed entry InvalidationMode = %q, want external_refutation (invalidation state must be set, not zero)",
			witnessed.Coherence.InvalidationMode)
	}
	if witnessed.Validity == (cachemeta.Validity{}) {
		t.Fatal("witnessed entry Validity is zero-value — rehydrate must carry explicit freshness/integrity bounds")
	}
	// The record is a lowered content page, provenance intact: the adapter ran over the
	// rehydrated recall pages (Producer "recall", PlaneContextPage), not a fabricated entry.
	if witnessed.Plane != cachemeta.PlaneContextPage {
		t.Fatalf("witnessed entry Plane = %q, want context_page", witnessed.Plane)
	}
	if witnessed.Derivation.Producer != "recall" {
		t.Fatalf("witnessed entry Producer = %q, want recall", witnessed.Derivation.Producer)
	}
	if witnessed.Labels["session_id"] != id {
		t.Fatalf("witnessed entry session_id label = %q, want %q", witnessed.Labels["session_id"], id)
	}

	// Cold-path correctness stays explicit: a rehydrated record is on disk, never a resident
	// hot tier, so it cannot be served as a live hit before the first post-wake re-prefill.
	for _, e := range res.CacheEntries {
		if e.Residency.Tier != cachemeta.TierDisk {
			t.Fatalf("rehydrated entry residency Tier = %q, want disk (a rehydrated record must not imply a hot hit)", e.Residency.Tier)
		}
	}
}

// TestRehydrateDriveOnlyHasNoCacheEntries: a drive-only image (no content pages) carries no
// cachemeta records — the wiring is strictly additive and never fabricates a record.
func TestRehydrateDriveOnlyHasNoCacheEntries(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	const id = "sess-drive-only"
	in := Input{SessionID: id, Drive: session.DefaultState(id), Now: 1_700_000_000}
	if _, err := DumpDir(dir, in); err != nil {
		t.Fatalf("DumpDir: %v", err)
	}
	img, err := LoadDir(dir)
	if err != nil {
		t.Fatalf("LoadDir: %v", err)
	}
	res, err := img.Rehydrate(ctx, RehydrateOptions{Table: session.NewTable()})
	if err != nil {
		t.Fatalf("Rehydrate: %v", err)
	}
	if len(res.CacheEntries) != 0 {
		t.Fatalf("drive-only image produced %d cache entries, want 0", len(res.CacheEntries))
	}
}
