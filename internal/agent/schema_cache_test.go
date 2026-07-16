package agent

import (
	"encoding/json"
	"fmt"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/leakcheck"
)

// TestSchemaNormCacheWitnessOpenAI proves the #796 OpenAI memo is sound: a cache hit
// yields bytes identical to a fresh normalize for the same (root, raw), and a changed
// schema misses (new content key) and recomputes its own distinct result.
func TestSchemaNormCacheWitnessOpenAI(t *testing.T) {
	raw := json.RawMessage(`{"type":"object","properties":{"q":{"type":"string"}}}`)

	// First call computes; the compute path is the source of truth.
	want := openAICompatibleSchemaCompute(raw, true)
	got1 := openAICompatibleSchema(raw, true) // populates the cache
	got2 := openAICompatibleSchema(raw, true) // must be served from cache
	if string(got1) != string(want) {
		t.Fatalf("first cached result %s != recompute %s", got1, want)
	}
	if string(got2) != string(want) {
		t.Fatalf("cache hit %s != recompute %s", got2, want)
	}

	// A returned slice must not alias the cached entry — mutating it must not corrupt a
	// later hit (the cache copies on load).
	if len(got1) > 0 {
		got1[0] = 'X'
	}
	got3 := openAICompatibleSchema(raw, true)
	if string(got3) != string(want) {
		t.Fatalf("cache corrupted by caller mutation: %s != %s", got3, want)
	}

	// A different schema is a different key — its result reflects ITS content, not the
	// first entry's.
	raw2 := json.RawMessage(`{"type":"object","properties":{"n":{"type":"integer"}}}`)
	want2 := openAICompatibleSchemaCompute(raw2, true)
	got4 := openAICompatibleSchema(raw2, true)
	if string(got4) != string(want2) {
		t.Fatalf("distinct schema served wrong result: %s != %s", got4, want2)
	}
	if string(got4) == string(want) {
		t.Fatal("distinct schema collided with the first cache entry")
	}

	// The root flag is part of the key: the same bytes at non-root normalize differently
	// and must not be served the root entry.
	wantNonRoot := openAICompatibleSchemaCompute(raw, false)
	gotNonRoot := openAICompatibleSchema(raw, false)
	if string(gotNonRoot) != string(wantNonRoot) {
		t.Fatalf("non-root served %s, want %s", gotNonRoot, wantNonRoot)
	}
}

// TestSchemaNormCacheWitnessGemini proves the Gemini memo is sound and provider-scoped:
// a hit equals a fresh uppercase-normalize, and the Gemini entry is distinct from the
// OpenAI entry for the same raw bytes (OpenAI lowercases/fills type, Gemini uppercases).
func TestSchemaNormCacheWitnessGemini(t *testing.T) {
	raw := json.RawMessage(`{"type":"object","properties":{"q":{"type":"string"}}}`)

	wantAny := geminiSchemaCompute(raw)
	want, ok := wantAny.(json.RawMessage)
	if !ok {
		t.Fatalf("geminiSchemaCompute returned %T, want json.RawMessage", wantAny)
	}

	gotAny1 := geminiSchema(raw) // populates the cache
	gotAny2 := geminiSchema(raw) // served from cache
	got1, ok := gotAny1.(json.RawMessage)
	if !ok {
		t.Fatalf("geminiSchema returned %T, want json.RawMessage", gotAny1)
	}
	got2, ok := gotAny2.(json.RawMessage)
	if !ok {
		t.Fatalf("geminiSchema (hit) returned %T, want json.RawMessage", gotAny2)
	}
	if string(got1) != string(want) || string(got2) != string(want) {
		t.Fatalf("gemini cache %s / %s != recompute %s", got1, got2, want)
	}

	// Provider isolation: the SAME raw bytes normalize to UPPERCASE for Gemini and
	// lowercase for OpenAI, so the two caches must not cross-serve.
	oaWant := openAICompatibleSchemaCompute(raw, true)
	if string(want) == string(oaWant) {
		t.Fatal("test fixture too weak: gemini and openai normalize identically — provider key untested")
	}
	if string(got2) == string(oaWant) {
		t.Fatal("gemini hit served the OpenAI normalization — provider not in the key")
	}
}

// TestNormalizedSchemaCacheBounded is the #3297 regression proof: the gateway keys this
// cache on sha256(raw pre-normalization bytes), so a gateway fronting heterogeneous
// clients mints a fresh permanent key per distinct schema. Drive many distinct keys
// through the cache and assert the retained-entry count never exceeds the cap — a return
// to the pre-fix unbounded sync.Map would fail here.
func TestNormalizedSchemaCacheBounded(t *testing.T) {
	const cap = 16
	c := &normalizedSchemaLRU{cap: cap}
	step := func(i int) {
		raw := json.RawMessage(fmt.Sprintf(`{"type":"object","id":%d}`, i))
		c.store(normalizedSchemaKey(schemaCacheKeyOpenAI, true, raw), []byte(`{}`))
	}
	leakcheck.BoundedSize(t, 1000, cap, step, c.len)
}

// TestNormalizedSchemaCacheLenObservable proves the global cache's footprint is visible
// through the public store path (#3297 DoD: a count metric so growth is observable) and
// stays within the shipped cap.
func TestNormalizedSchemaCacheLenObservable(t *testing.T) {
	before := normalizedSchemaCacheLen()
	for i := 0; i < 50; i++ {
		raw := json.RawMessage(fmt.Sprintf(`{"type":"object","obs":%d}`, i))
		storeNormalizedSchema(schemaCacheKeyOpenAI, true, raw, json.RawMessage(`{}`))
	}
	after := normalizedSchemaCacheLen()
	if after <= before {
		t.Fatalf("cache len did not grow through the public store path: before=%d after=%d", before, after)
	}
	if after > normalizedSchemaCacheCap {
		t.Fatalf("global cache len %d exceeds cap %d", after, normalizedSchemaCacheCap)
	}
}

// TestNormalizedSchemaCacheEvictsLRU proves eviction is least-recently-used, not arbitrary:
// after overflowing the cap, a key touched between inserts survives while an untouched
// older key is dropped.
func TestNormalizedSchemaCacheEvictsLRU(t *testing.T) {
	const cap = 4
	c := &normalizedSchemaLRU{cap: cap}
	key := func(i int) normalizedSchemaCacheKey {
		return normalizedSchemaKey(schemaCacheKeyOpenAI, true, json.RawMessage(fmt.Sprintf(`{"id":%d}`, i)))
	}
	for i := 0; i < cap; i++ {
		c.store(key(i), []byte(`{}`))
	}
	// Touch key(0) so it becomes most-recently-used, then insert a fresh key: the LRU
	// victim must be key(1) (the oldest untouched), not key(0).
	if _, ok := c.load(key(0)); !ok {
		t.Fatal("key(0) should still be resident before overflow")
	}
	c.store(key(cap), []byte(`{}`)) // forces one eviction
	if _, ok := c.load(key(0)); !ok {
		t.Fatal("recently-touched key(0) was wrongly evicted")
	}
	if _, ok := c.load(key(1)); ok {
		t.Fatal("least-recently-used key(1) should have been evicted")
	}
	if got := c.len(); got != cap {
		t.Fatalf("len=%d after overflow, want cap=%d", got, cap)
	}
}
