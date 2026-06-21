package cachemeta

import (
	"bytes"
	"testing"
)

// prefix_coherence_test.go — witnesses for the §A4 witness-gated prefix-break.

func TestPrefixCoherenceAllWitnessesHoldNoBreak(t *testing.T) {
	deps := []PrefixDependency{
		{Witness: "git_sha:src/foo.go", Value: "abc123", TokenStart: 100},
		{Witness: "lease_epoch:job-42", Value: "7", TokenStart: 250},
	}
	cur := map[string]string{"git_sha:src/foo.go": "abc123", "lease_epoch:job-42": "7"}
	d := EvaluatePrefixCoherence(deps, cur)
	if d.Break {
		t.Fatalf("all witnesses hold but Break=true: %+v", d)
	}
}

func TestPrefixCoherenceRefutedWitnessBreaksAtItsSpan(t *testing.T) {
	deps := []PrefixDependency{
		{Witness: "git_sha:src/foo.go", Value: "abc123", TokenStart: 100},
		{Witness: "lease_epoch:job-42", Value: "7", TokenStart: 250},
	}
	cur := map[string]string{"git_sha:src/foo.go": "def456", "lease_epoch:job-42": "7"}
	d := EvaluatePrefixCoherence(deps, cur)
	if !d.Break {
		t.Fatalf("refuted git sha but Break=false")
	}
	if d.BreakAtToken != 100 {
		t.Fatalf("BreakAtToken %d, want 100 (the refuted span)", d.BreakAtToken)
	}
	if len(d.RefutedKeys) != 1 || d.RefutedKeys[0] != "git_sha:src/foo.go" {
		t.Fatalf("RefutedKeys %v, want [git_sha:src/foo.go]", d.RefutedKeys)
	}
	if d.Marker.Kind != SegVolatile {
		t.Fatalf("break marker kind %q, want volatile", d.Marker.Kind)
	}
}

func TestPrefixCoherenceBreaksAtEarliestStaleSpan(t *testing.T) {
	deps := []PrefixDependency{
		{Witness: "blob_hash:doc1", Value: "h1", TokenStart: 400},
		{Witness: "git_sha:src/foo.go", Value: "abc", TokenStart: 120},
		{Witness: "lease_epoch:job-9", Value: "3", TokenStart: 800},
	}
	cur := map[string]string{"blob_hash:doc1": "h2", "git_sha:src/foo.go": "zzz", "lease_epoch:job-9": "3"}
	d := EvaluatePrefixCoherence(deps, cur)
	if !d.Break || d.BreakAtToken != 120 {
		t.Fatalf("BreakAtToken %d, want 120 (earliest of two stale spans); Break=%v", d.BreakAtToken, d.Break)
	}
	if len(d.RefutedKeys) != 2 {
		t.Fatalf("RefutedKeys %v, want 2 entries", d.RefutedKeys)
	}
}

func TestPrefixCoherenceUnconfirmableWitnessFailsClosed(t *testing.T) {
	deps := []PrefixDependency{{Witness: "lease_epoch:job-1", Value: "5", TokenStart: 60}}
	d := EvaluatePrefixCoherence(deps, map[string]string{})
	if !d.Break {
		t.Fatalf("unconfirmable witness must fail closed (Break=true), got %+v", d)
	}
	if d.BreakAtToken != 60 {
		t.Fatalf("BreakAtToken %d, want 60", d.BreakAtToken)
	}
}

func TestPrefixCoherenceMarkerIsDeterministicButWorldSensitive(t *testing.T) {
	deps := []PrefixDependency{{Witness: "git_sha:a", Value: "v0", TokenStart: 10}}
	m1 := EvaluatePrefixCoherence(deps, map[string]string{"git_sha:a": "v1"}).Marker.Content
	m1b := EvaluatePrefixCoherence(deps, map[string]string{"git_sha:a": "v1"}).Marker.Content
	m2 := EvaluatePrefixCoherence(deps, map[string]string{"git_sha:a": "v2"}).Marker.Content
	if !bytes.Equal(m1, m1b) {
		t.Fatalf("same refutation produced different markers: %q vs %q", m1, m1b)
	}
	if bytes.Equal(m1, m2) {
		t.Fatalf("different world values produced the SAME marker %q — provider cache could still hit the stale span", m1)
	}
}

func TestInjectBreakMarkerAppliesTheBreakAheadOfTheStaleSpan(t *testing.T) {
	// Cached prefix: system (100 tok) then a doc span starting at token 100.
	turn := []PromptSegment{
		seg(SegStable, 100, "system"),
		seg(SegStable, 50, "doc-v2 contents"),
	}
	deps := []PrefixDependency{{Witness: "blob_hash:doc", Value: "v1", TokenStart: 100}}
	d := EvaluatePrefixCoherence(deps, map[string]string{"blob_hash:doc": "v2"})
	out := InjectBreakMarker(turn, d)
	if len(out) != 3 {
		t.Fatalf("inject: len %d, want 3 (marker inserted)", len(out))
	}
	if out[1].Kind != SegVolatile {
		t.Fatalf("inject: segment 1 kind %q, want volatile (the marker ahead of the stale doc)", out[1].Kind)
	}
	// The injected turn, compared to the cached prefix, must break right at the marker:
	// system (100) still cacheable, everything from the marker on is re-billed.
	div := Diverge(turn, out)
	if div.StableTokens != 100 {
		t.Fatalf("inject: cacheable after break %d, want 100", div.StableTokens)
	}
	if div.FirstDivergeSeg != 1 {
		t.Fatalf("inject: break at seg %d, want 1 (the marker)", div.FirstDivergeSeg)
	}
}

func TestInjectBreakMarkerNoBreakIsUnchanged(t *testing.T) {
	turn := stablePrefix()
	out := InjectBreakMarker(turn, PrefixBreakDirective{Break: false})
	if len(out) != len(turn) {
		t.Fatalf("no-break inject changed the turn: len %d != %d", len(out), len(turn))
	}
}

func TestEvaluatePrefixCoherenceRevokedWiresToRevocationBus(t *testing.T) {
	deps := []PrefixDependency{
		{Witness: "git_sha:foo", TokenStart: 100},
		{Witness: "lease:job-1", TokenStart: 300},
	}
	// vdso-style predicate: only git_sha:foo is revoked.
	revoked := func(w string) bool { return w == "git_sha:foo" }
	d := EvaluatePrefixCoherenceRevoked(deps, revoked)
	if !d.Break || d.BreakAtToken != 100 {
		t.Fatalf("revoked git_sha should break at 100; got break=%v at=%d", d.Break, d.BreakAtToken)
	}
	if len(d.RefutedKeys) != 1 || d.RefutedKeys[0] != "git_sha:foo" {
		t.Fatalf("RefutedKeys %v, want [git_sha:foo]", d.RefutedKeys)
	}
	// Nothing revoked => no break.
	if d2 := EvaluatePrefixCoherenceRevoked(deps, func(string) bool { return false }); d2.Break {
		t.Fatalf("nothing revoked but Break=true")
	}
	// Nil predicate (no bus) => fail-open, no break.
	if d3 := EvaluatePrefixCoherenceRevoked(deps, nil); d3.Break {
		t.Fatalf("nil revocation predicate should not break")
	}
}

func TestShapeGLMTurnRevokedIsTheLiveProxySeam(t *testing.T) {
	turn := []PromptSegment{seg(SegStable, 100, "system"), seg(SegStable, 50, "doc")}
	deps := []PrefixDependency{{Witness: "blob:doc", TokenStart: 100}}
	// Live path: the gateway hands in vdso.Default.Revoked; here the doc's witness is revoked.
	shaped, dir, _ := ShapeGLMTurnRevoked(turn, deps, func(w string) bool { return w == "blob:doc" })
	if !dir.Break || len(shaped) != 3 || shaped[1].Kind != SegVolatile {
		t.Fatalf("revoked doc witness should inject a break marker; got break=%v len=%d", dir.Break, len(shaped))
	}
	if div := Diverge(turn, shaped); div.StableTokens != 100 || div.FirstDivergeSeg != 1 {
		t.Fatalf("shaped turn must break at the marker: stable=%d seg=%d", div.StableTokens, div.FirstDivergeSeg)
	}
}

func TestSegmentCoherenceBreaksAtRevokedSegmentNoCoordinateNeeded(t *testing.T) {
	// Witnesses ride ON the segments — the live-path form that dissolves the
	// page-vs-prompt coordinate problem. doc1 (at token offset 100) is revoked.
	turn := []PromptSegment{
		{Kind: SegStable, Tokens: 100, Content: []byte("system"), Witness: "git_sha:a.go"},
		{Kind: SegToolResult, Tokens: 50, Content: []byte("doc1 body"), Witness: "blob:doc1"},
		{Kind: SegMessage, Tokens: 20, Content: []byte("user")},
	}
	revoked := func(w string) bool { return w == "blob:doc1" }
	d := EvaluateSegmentCoherence(turn, revoked)
	if !d.Break {
		t.Fatalf("doc1 segment revoked but no break")
	}
	if d.BreakAtToken != 100 {
		t.Fatalf("break at %d, want 100 (doc1 segment's offset, summed from segment tokens)", d.BreakAtToken)
	}
	if len(d.RefutedKeys) != 1 || d.RefutedKeys[0] != "blob:doc1" {
		t.Fatalf("RefutedKeys %v, want [blob:doc1]", d.RefutedKeys)
	}
	// nil predicate / nothing revoked => no break.
	if EvaluateSegmentCoherence(turn, nil).Break {
		t.Fatalf("nil predicate should not break")
	}
	if EvaluateSegmentCoherence(turn, func(string) bool { return false }).Break {
		t.Fatalf("nothing revoked should not break")
	}
}

func TestShapeGLMTurnSegmentWitnessedIsTheSimpleLiveSeam(t *testing.T) {
	turn := []PromptSegment{
		{Kind: SegStable, Tokens: 100, Content: []byte("system"), Witness: "git_sha:a.go"},
		{Kind: SegToolResult, Tokens: 50, Content: []byte("doc1"), Witness: "blob:doc1"},
	}
	shaped, dir, _ := ShapeGLMTurnSegmentWitnessed(turn, func(w string) bool { return w == "blob:doc1" })
	if !dir.Break || len(shaped) != 3 || shaped[1].Kind != SegVolatile {
		t.Fatalf("revoked segment should inject a break marker; got break=%v len=%d", dir.Break, len(shaped))
	}
	if div := Diverge(turn, shaped); div.StableTokens != 100 || div.FirstDivergeSeg != 1 {
		t.Fatalf("shaped turn must break at the marker ahead of doc1: stable=%d seg=%d", div.StableTokens, div.FirstDivergeSeg)
	}
}

func TestShapeGLMTurnIsTheGatewaySeam(t *testing.T) {
	turn := []PromptSegment{seg(SegStable, 100, "system"), seg(SegStable, 50, "doc-v2")}
	deps := []PrefixDependency{{Witness: "blob_hash:doc", Value: "v1", TokenStart: 100}}

	// World holds: no break, turn unchanged, whole prefix cacheable.
	shaped, dir, _ := ShapeGLMTurn(turn, deps, map[string]string{"blob_hash:doc": "v1"})
	if dir.Break || len(shaped) != len(turn) {
		t.Fatalf("witness holds: expected no shaping, got break=%v len=%d", dir.Break, len(shaped))
	}

	// World refuted: break injected ahead of the stale span; the shaped turn's stability
	// shows only the fresh prefix (system, 100) is recoverable-cacheable, the rest broken.
	shaped, dir, _ = ShapeGLMTurn(turn, deps, map[string]string{"blob_hash:doc": "v2"})
	if !dir.Break {
		t.Fatalf("witness refuted but no break")
	}
	if len(shaped) != 3 || shaped[1].Kind != SegVolatile {
		t.Fatalf("shaped turn missing the injected break marker: %+v", shaped)
	}
	// vs the cached (pre-shape) turn, the shaped turn breaks at the marker.
	if div := Diverge(turn, shaped); div.StableTokens != 100 || div.FirstDivergeSeg != 1 {
		t.Fatalf("shaped turn does not break at the marker: stable=%d seg=%d", div.StableTokens, div.FirstDivergeSeg)
	}
}

func TestPrefixCoherenceMarkerBreaksA3Prefix(t *testing.T) {
	cached := []PromptSegment{
		seg(SegStable, 100, "system"),
		seg(SegStable, 50, "doc-v1 contents"),
	}
	deps := []PrefixDependency{{Witness: "blob_hash:doc", Value: "v1", TokenStart: 100}}
	d := EvaluatePrefixCoherence(deps, map[string]string{"blob_hash:doc": "v2"})
	if !d.Break {
		t.Fatalf("doc changed but no break")
	}
	next := []PromptSegment{
		seg(SegStable, 100, "system"),
		d.Marker,
		seg(SegStable, 50, "doc-v2 contents"),
	}
	div := Diverge(cached, next)
	if div.StableTokens != 100 {
		t.Fatalf("after coherence break, cacheable prefix %d, want 100 (only system; the marker breaks the rest)", div.StableTokens)
	}
	if div.FirstDivergeSeg != 1 {
		t.Fatalf("prefix should break at the injected marker (seg 1), got %d", div.FirstDivergeSeg)
	}
}
