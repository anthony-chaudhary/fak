package recall

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/cachemeta"
)

// prefix_deps_test.go — witnesses for the §A4 recall-side connector.

// The LIVE path: a Recorder mid-session (not a finished Manifest) yields the same
// coherence deps via PrefixDeps -> Manifest() snapshot. This proves the live OpenAI-proxy
// path has a clean deps source, removing the "no Manifest mid-session" gap.
func TestRecorderPrefixDepsLivePath(t *testing.T) {
	r := NewRecorder("live-session")
	ctx := context.Background()
	r.RecordWithWitness(ctx, "read", []byte("contents of src/a.go — several bytes of context here"), "git_sha:src/a.go")
	r.Record(ctx, "think", []byte("an un-witnessed reasoning step"))
	r.RecordWithWitness(ctx, "read", []byte("doc1 body bytes"), "blob_hash:doc1")
	deps := r.PrefixDeps()
	if len(deps) != 2 {
		t.Fatalf("live recorder: got %d deps, want 2 (the two witnessed pages)", len(deps))
	}
	if deps[0].Witness != "git_sha:src/a.go" || deps[1].Witness != "blob_hash:doc1" {
		t.Fatalf("live deps witnesses = %q,%q", deps[0].Witness, deps[1].Witness)
	}
	if !(deps[1].TokenStart > deps[0].TokenStart) {
		t.Fatalf("live deps offsets not monotonic: %d then %d", deps[0].TokenStart, deps[1].TokenStart)
	}
	// And it drives the shape call: revoke doc1 -> break at its offset.
	_, dir, _ := cachemeta.ShapeGLMTurnRevoked(
		[]cachemeta.PromptSegment{{Kind: cachemeta.SegStable, Tokens: deps[1].TokenStart + 50, Content: []byte("prefix")}},
		deps, func(w string) bool { return w == "blob_hash:doc1" })
	if !dir.Break || dir.BreakAtToken != deps[1].TokenStart {
		t.Fatalf("live revoke should break at doc1 offset %d; got break=%v at=%d", deps[1].TokenStart, dir.Break, dir.BreakAtToken)
	}
}

func TestPrefixDepsExtractsWitnessedPagesAtTokenOffsets(t *testing.T) {
	m := Manifest{Pages: []Page{
		{Step: 0, Len: 400, Witness: "git_sha:src/a.go"}, // offset 0
		{Step: 1, Len: 200},                            // no witness; advances offset by 200/4 = 50
		{Step: 2, Len: 800, Witness: "blob_hash:doc1"}, // offset 150 (400/4 + 200/4)
	}}
	deps := m.PrefixDeps()
	if len(deps) != 2 {
		t.Fatalf("got %d deps, want 2 (only witnessed pages)", len(deps))
	}
	if deps[0].Witness != "git_sha:src/a.go" || deps[0].TokenStart != 0 {
		t.Fatalf("dep0 = %+v, want git_sha:src/a.go @0", deps[0])
	}
	if deps[1].Witness != "blob_hash:doc1" || deps[1].TokenStart != 150 {
		t.Fatalf("dep1 = %+v, want blob_hash:doc1 @150 (400/4 + 200/4)", deps[1])
	}
}

func TestPrefixDepsNoWitnessesIsEmpty(t *testing.T) {
	m := Manifest{Pages: []Page{{Len: 100}, {Len: 200}}}
	if d := m.PrefixDeps(); len(d) != 0 {
		t.Fatalf("no witnessed pages but got %d deps", len(d))
	}
}

// End-to-end: the connector feeds cachemeta.ShapeGLMTurnRevoked, and a revoked witness
// breaks the hosted GLM prefix at that page's token offset — the whole live-path data
// flow (recorded pages -> deps -> revocation check -> shaped turn) minus the gateway
// call-site.
func TestPrefixDepsFeedsShapeGLMTurnRevoked(t *testing.T) {
	m := Manifest{Pages: []Page{
		{Len: 400, Witness: "git_sha:src/a.go"}, // @0
		{Len: 200, Witness: "blob_hash:doc1"},   // @100
	}}
	deps := m.PrefixDeps()
	// The assembled prompt: a 100-token stable system span, then the doc span at 100.
	turn := []cachemeta.PromptSegment{
		{Kind: cachemeta.SegStable, Tokens: 100, Content: []byte("...src/a.go context...")},
		{Kind: cachemeta.SegStable, Tokens: 50, Content: []byte("...doc1 contents...")},
	}
	// doc1's witness is revoked (the blob changed); a.go's still holds.
	revoked := func(w string) bool { return w == "blob_hash:doc1" }
	shaped, dir, _ := cachemeta.ShapeGLMTurnRevoked(turn, deps, revoked)
	if !dir.Break {
		t.Fatalf("doc1 revoked but no break")
	}
	if dir.BreakAtToken != 100 {
		t.Fatalf("break at %d, want 100 (doc1's page offset)", dir.BreakAtToken)
	}
	if len(shaped) != 3 || shaped[1].Kind != cachemeta.SegVolatile {
		t.Fatalf("shaped turn missing the break marker ahead of the stale doc: %+v", shaped)
	}
}
