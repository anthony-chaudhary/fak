package toolproc

// repeatcache_test.go — failure-class proofs for the LIVE reuse engine (#4764 DoD
// #3/#4). The offline classifier is proven in repeatclass_test.go; here the same
// safety contract is proven as STATE, per-call: immutable hits serve, a mutation
// invalidates, writes/unknowns are fail-closed, mutable status is freshness-bounded
// with stale-age + source exposed, and no output body is ever retained.

import (
	"reflect"
	"testing"
)

// TestReuseStoreImmutableHitServesAfterFirstFetch: the first read is a real fetch;
// an identical re-read (same path+digest) is served from the immutable cache with a
// keyed_hit receipt and the saved output size.
func TestReuseStoreImmutableHitServesAfterFirstFetch(t *testing.T) {
	c := NewReuseStore(RepeatConfig{})
	first := c.Admit(CallRecord{Tool: "shell_command", Raw: "cat C:/x/SKILL.md", AtMS: 0, OutputBytes: 1000, Digest: "d1"})
	if first.Served || first.Reason != ReasonFirstFetch || first.Source != SourceMiss {
		t.Fatalf("first read must be a miss/first_fetch, got %+v", first)
	}
	hit := c.Admit(CallRecord{Tool: "shell_command", Raw: `Get-Content -Raw C:\x\SKILL.md`, AtMS: 20, OutputBytes: 1000, Digest: "d1"})
	if !hit.Served || hit.Source != SourceImmutable || hit.Reason != ReasonKeyedHit {
		t.Errorf("equivalent re-read must be a keyed immutable hit, got %+v", hit)
	}
	if hit.SavedBytes != 1000 {
		t.Errorf("hit must report saved bytes = first fetch size, got %d", hit.SavedBytes)
	}
	if c.Hits() != 1 || c.Misses() != 1 {
		t.Errorf("tally: want 1 hit / 1 miss, got %d / %d", c.Hits(), c.Misses())
	}
}

// TestReuseStoreMutationInvalidates: a new content digest for the same path forces a
// fresh fetch (digest_changed), never a stale serve — the invalidation contract.
func TestReuseStoreMutationInvalidates(t *testing.T) {
	c := NewReuseStore(RepeatConfig{})
	recs := []CallRecord{
		{Tool: "shell_command", Raw: "cat C:/x/f", AtMS: 0, OutputBytes: 100, Digest: "d1"},
		{Tool: "shell_command", Raw: "cat C:/x/f", AtMS: 10, OutputBytes: 100, Digest: "d1"}, // hit
		{Tool: "shell_command", Raw: "cat C:/x/f", AtMS: 20, OutputBytes: 100, Digest: "d2"}, // mutated → miss
		{Tool: "shell_command", Raw: "cat C:/x/f", AtMS: 30, OutputBytes: 100, Digest: "d2"}, // hit on new identity
	}
	rc := c.Replay(recs)
	wantServed := []bool{false, true, false, true}
	wantReason := []ReuseReason{ReasonFirstFetch, ReasonKeyedHit, ReasonDigestChanged, ReasonKeyedHit}
	for i := range recs {
		if rc[i].Served != wantServed[i] || rc[i].Reason != wantReason[i] {
			t.Errorf("obs %d: want served=%v reason=%s, got served=%v reason=%s",
				i, wantServed[i], wantReason[i], rc[i].Served, rc[i].Reason)
		}
	}
	// The post-mutation read (obs 2) must NOT be served — a stale serve here is the
	// exact unsafe behaviour the digest key exists to prevent.
	if rc[2].Served {
		t.Fatalf("post-mutation read was served stale: %+v", rc[2])
	}
}

// TestReuseStoreNeverServesWritesOrUnknowns: a push and an unrecognized command are
// fail-closed — never served, whatever the repeat count.
func TestReuseStoreNeverServesWritesOrUnknowns(t *testing.T) {
	c := NewReuseStore(RepeatConfig{})
	recs := []CallRecord{
		{Tool: "shell_command", Raw: "git push origin main", AtMS: 0, OutputBytes: 40},
		{Tool: "shell_command", Raw: "git push origin main", AtMS: 100, OutputBytes: 40},
		{Tool: "shell_command", Raw: "frobnicate --wibble", AtMS: 200, OutputBytes: 9},
		{Tool: "shell_command", Raw: "frobnicate --wibble", AtMS: 300, OutputBytes: 9},
	}
	for i, rec := range recs {
		got := c.Admit(rec)
		if got.Served || got.Reuse != ReuseNever || got.Reason != ReasonNeverReused {
			t.Errorf("obs %d (%q) must be never-reused, got %+v", i, rec.Raw, got)
		}
	}
	if c.Hits() != 0 {
		t.Errorf("writes/unknowns must never hit, got %d hits", c.Hits())
	}
}

// TestReuseStoreStatusIsFreshnessBoundedWithStaleAge: a status poll is served only
// inside its freshness window of the last real fetch; a poll past the window is a
// fresh fetch (window_expired), and every hit exposes stale-age + freshness source.
func TestReuseStoreStatusIsFreshnessBoundedWithStaleAge(t *testing.T) {
	c := NewReuseStore(RepeatConfig{}) // 2s default window
	recs := []CallRecord{
		{Tool: "shell_command", Raw: "git status --short --branch", AtMS: 0, OutputBytes: 200},
		{Tool: "shell_command", Raw: "git status --branch --short", AtMS: 1000, OutputBytes: 200}, // hit (folds flag order)
		{Tool: "shell_command", Raw: "git status --short --branch", AtMS: 5000, OutputBytes: 200}, // past window → miss
		{Tool: "shell_command", Raw: "git status --short --branch", AtMS: 6000, OutputBytes: 200}, // hit
	}
	rc := c.Replay(recs)
	wantServed := []bool{false, true, false, true}
	wantReason := []ReuseReason{ReasonFirstFetch, ReasonFreshnessHit, ReasonWindowExpired, ReasonFreshnessHit}
	for i := range recs {
		if rc[i].Served != wantServed[i] || rc[i].Reason != wantReason[i] {
			t.Fatalf("obs %d: want served=%v reason=%s, got %+v", i, wantServed[i], wantReason[i], rc[i])
		}
	}
	// The in-window hit must expose a positive stale-age and the freshness source.
	if rc[1].StaleAgeMS != 1000 || rc[1].Source != SourceFreshness {
		t.Errorf("freshness hit must expose stale_age=1000 + freshness source, got age=%d source=%s", rc[1].StaleAgeMS, rc[1].Source)
	}
	if c.Hits() != 2 || c.Misses() != 2 {
		t.Errorf("bounded, not blanket: want 2 hits / 2 misses, got %d / %d", c.Hits(), c.Misses())
	}
}

// TestReuseStoreRetainsNoBody: the engine keeps only sizes + digests, never a
// payload — the cached state for a read is the {digest,size} pair, with no command
// text and no output body anywhere in the struct.
func TestReuseStoreRetainsNoBody(t *testing.T) {
	c := NewReuseStore(RepeatConfig{})
	c.Admit(CallRecord{Tool: "shell_command", Raw: "cat C:/x/SKILL.md", AtMS: 0, OutputBytes: 4096, Digest: "d"})
	st, ok := c.reads["C:/x/SKILL.md"]
	if !ok {
		t.Fatalf("read state not recorded for path")
	}
	if st.digest != "d" || st.size != 4096 {
		t.Errorf("cache must retain only {digest,size}, got %+v", st)
	}
}

// TestReuseStoreReplayIsDeterministic: same time-ordered stream ⇒ identical receipts
// across two fresh engines.
func TestReuseStoreReplayIsDeterministic(t *testing.T) {
	recs := []CallRecord{
		{Tool: "shell_command", Raw: "cat C:/x/SKILL.md", AtMS: 0, OutputBytes: 100, Digest: "d"},
		{Tool: "shell_command", Raw: "cat C:/x/SKILL.md", AtMS: 10, OutputBytes: 100, Digest: "d"},
		{Tool: "shell_command", Raw: "git status", AtMS: 20, OutputBytes: 50},
		{Tool: "shell_command", Raw: "git status", AtMS: 500, OutputBytes: 50},
		{Tool: "shell_command", Raw: "git push origin main", AtMS: 30, OutputBytes: 40},
	}
	a := NewReuseStore(RepeatConfig{}).Replay(recs)
	b := NewReuseStore(RepeatConfig{}).Replay(recs)
	if !reflect.DeepEqual(a, b) {
		t.Errorf("replay not deterministic:\n a=%+v\n b=%+v", a, b)
	}
}
