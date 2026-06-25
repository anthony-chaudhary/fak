package agent

import (
	"encoding/json"
	"testing"
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
