package toolproc

// repeatarm_test.go — failure-class proofs for the ARMED byte-serving cache
// (#5122): an immutable repeat is served with its actual bytes and a receipt; a
// mutation eagerly evicts the replaced body; query coalescing byte-serves ONLY when
// opted in and only inside the freshness window; writes/unknowns are never
// retained; the byte budget evicts oldest-first and an over-cap payload is never
// held; a decision hit without bytes is honest (BodyServed=false → caller fetches).

import (
	"bytes"
	"testing"
)

// TestArmedCacheServesImmutableRepeatBytes: fetch, offer, then an equivalent
// re-read is byte-served with the exact payload and a keyed-hit receipt.
func TestArmedCacheServesImmutableRepeatBytes(t *testing.T) {
	a := NewArmedCache(ArmedConfig{})
	body := []byte("skill file content")
	first := a.Admit(CallRecord{Tool: "shell_command", Raw: "cat C:/x/SKILL.md", AtMS: 0, OutputBytes: int64(len(body)), Digest: "d1"})
	if first.BodyServed || first.Receipt.Served {
		t.Fatalf("first read must be a real fetch, got %+v", first)
	}
	if !a.Offer(CallRecord{Tool: "shell_command", Raw: "cat C:/x/SKILL.md", AtMS: 0, OutputBytes: int64(len(body)), Digest: "d1"}, body) {
		t.Fatal("offer of an immutable-read body must be retained")
	}
	hit := a.Admit(CallRecord{Tool: "shell_command", Raw: `Get-Content -Raw C:\x\SKILL.md`, AtMS: 20, OutputBytes: int64(len(body)), Digest: "d1"})
	if !hit.BodyServed || !bytes.Equal(hit.Body, body) {
		t.Fatalf("equivalent re-read must byte-serve the deposited payload, got %+v", hit)
	}
	if hit.Receipt.Source != SourceImmutable || hit.Receipt.Reason != ReasonKeyedHit {
		t.Errorf("hit receipt must expose immutable provenance, got %+v", hit.Receipt)
	}
	if a.BodyHits() != 1 || a.ServedBytes() != int64(len(body)) {
		t.Errorf("tally: want 1 body hit / %d served bytes, got %d / %d", len(body), a.BodyHits(), a.ServedBytes())
	}
}

// TestArmedCacheMutationEvictsStaleBody: a new digest for the same path must not
// only miss (decision layer) but eagerly release the replaced body's budget.
func TestArmedCacheMutationEvictsStaleBody(t *testing.T) {
	a := NewArmedCache(ArmedConfig{})
	old := []byte("v1 content")
	a.Admit(CallRecord{Tool: "shell_command", Raw: "cat C:/x/f", AtMS: 0, Digest: "d1", OutputBytes: int64(len(old))})
	a.Offer(CallRecord{Tool: "shell_command", Raw: "cat C:/x/f", AtMS: 0, Digest: "d1", OutputBytes: int64(len(old))}, old)
	if a.RetainedBytes() != int64(len(old)) {
		t.Fatalf("retained bytes: want %d, got %d", len(old), a.RetainedBytes())
	}
	mut := a.Admit(CallRecord{Tool: "shell_command", Raw: "cat C:/x/f", AtMS: 10, Digest: "d2", OutputBytes: 12})
	if mut.BodyServed || mut.Receipt.Served || mut.Receipt.Reason != ReasonDigestChanged {
		t.Fatalf("post-mutation read must be a fresh fetch, got %+v", mut)
	}
	if a.RetainedBytes() != 0 {
		t.Errorf("mutation must eagerly evict the stale body, retained=%d", a.RetainedBytes())
	}
}

// TestArmedCacheQueryCoalescingIsOptIn: without CoalesceQueries a status repeat is
// a decision-layer hit but is NOT byte-served and its body is never retained; with
// it, the repeat inside the freshness window serves bytes with stale-age exposed.
func TestArmedCacheQueryCoalescingIsOptIn(t *testing.T) {
	out := []byte("## main...origin/main\n M x.go\n")
	rec := func(at int64) CallRecord {
		return CallRecord{Tool: "shell_command", Raw: "git status --short", AtMS: at, OutputBytes: int64(len(out))}
	}

	off := NewArmedCache(ArmedConfig{})
	off.Admit(rec(0))
	if off.Offer(rec(0), out) {
		t.Fatal("query body must be refused when coalescing is not opted in")
	}
	hit := off.Admit(rec(500))
	if hit.BodyServed || hit.Body != nil {
		t.Fatalf("opt-out cache must never byte-serve a query, got %+v", hit)
	}
	if !hit.Receipt.Served {
		t.Errorf("decision receipt still flows on opt-out, got %+v", hit.Receipt)
	}

	on := NewArmedCache(ArmedConfig{CoalesceQueries: true})
	on.Admit(rec(0))
	if !on.Offer(rec(0), out) {
		t.Fatal("opted-in query body must be retained")
	}
	fresh := on.Admit(rec(500))
	if !fresh.BodyServed || !bytes.Equal(fresh.Body, out) {
		t.Fatalf("in-window repeat must byte-serve, got %+v", fresh)
	}
	if fresh.Receipt.Source != SourceFreshness || fresh.Receipt.StaleAgeMS != 500 {
		t.Errorf("hit must expose freshness source + stale-age, got %+v", fresh.Receipt)
	}
	stale := on.Admit(rec(10_000)) // far past DefaultFreshnessWindowMS
	if stale.BodyServed || stale.Receipt.Served {
		t.Errorf("past-window repeat must fetch fresh, got %+v", stale)
	}
}

// TestArmedCacheNeverRetainsWritesOrUnknowns: Offer is fail-closed for a push and
// for an unrecognized command, whatever the caller claims.
func TestArmedCacheNeverRetainsWritesOrUnknowns(t *testing.T) {
	a := NewArmedCache(ArmedConfig{CoalesceQueries: true})
	if a.Offer(CallRecord{Tool: "shell_command", Raw: "git push origin main", AtMS: 0}, []byte("ok")) {
		t.Error("a write's output must never be retained")
	}
	if a.Offer(CallRecord{Tool: "shell_command", Raw: "frobnicate --wibble", AtMS: 0}, []byte("x")) {
		t.Error("an unknown command's output must never be retained")
	}
	if a.RetainedBytes() != 0 {
		t.Errorf("nothing may be retained, got %d bytes", a.RetainedBytes())
	}
}

// TestArmedCacheBudgetEvictsOldestAndCapsEntries: an over-cap payload is never
// held, and budget pressure evicts the oldest body first — after which a decision
// hit is honestly BodyServed=false so the caller re-fetches.
func TestArmedCacheBudgetEvictsOldestAndCapsEntries(t *testing.T) {
	a := NewArmedCache(ArmedConfig{MaxBytes: 100, MaxEntryBytes: 60})
	if a.Offer(CallRecord{Tool: "shell_command", Raw: "cat C:/x/huge", Digest: "dh"}, make([]byte, 61)) {
		t.Fatal("an over-cap payload must not be retained")
	}
	offer := func(path, digest string, n int) {
		rec := CallRecord{Tool: "shell_command", Raw: "cat " + path, Digest: digest, OutputBytes: int64(n)}
		a.Admit(rec)
		if !a.Offer(rec, make([]byte, n)) {
			t.Fatalf("offer %s must be retained", path)
		}
	}
	offer("C:/x/a", "da", 60)
	offer("C:/x/b", "db", 60) // 60+60 > 100 → evicts a
	if a.RetainedBytes() != 60 {
		t.Fatalf("budget must hold one 60B body, retained=%d", a.RetainedBytes())
	}
	evicted := a.Admit(CallRecord{Tool: "shell_command", Raw: "cat C:/x/a", Digest: "da", OutputBytes: 60})
	if evicted.BodyServed {
		t.Fatal("an evicted body must not be served")
	}
	if !evicted.Receipt.Served {
		t.Errorf("the decision layer may still hit; arming must downgrade honestly, got %+v", evicted.Receipt)
	}
	kept := a.Admit(CallRecord{Tool: "shell_command", Raw: "cat C:/x/b", Digest: "db", OutputBytes: 60})
	if !kept.BodyServed {
		t.Errorf("the newest body must survive eviction, got %+v", kept)
	}
}
