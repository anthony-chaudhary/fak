package cachemeta_test

// serve_path_governance_test.go is the cross-path governance witness for cache-frontier
// Next-50 item 19 (#1537): "Local reuse remains governed by scope, freshness, taint, and
// quality evidence." The row-19 serve-path audit (docs/cache-frontier/
// DEFAULT-ENABLEMENT-NEXT-50.md) asserts in prose that every local cache serve path is
// governed by the shared cachemeta MaterializeVerdict discipline — either the umbrella
// function itself or the exact key/quality primitives it composes. This test makes that
// audit EXECUTABLE: it proves the SAME materialization key that a token-only prefix match
// would happily serve is REFUSED at every local serve path, while the request that owns
// the span is admitted. It lives in an external test package so it can drive the three
// downstream serve surfaces (radixkv, contextq) alongside the cachemeta gate they consume,
// witnessing the uniformity across package boundaries in one place.

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/cachemeta"
	"github.com/anthony-chaudhary/fak/internal/contextq"
	"github.com/anthony-chaudhary/fak/internal/radixkv"
)

// storedModelAKVSpan builds a fully-keyed local KV span under model-A so its
// MaterializationKey is Complete on every axis the #1537 fail-closed rule enumerates
// (model / tokenizer / serializer / position regime / policy / admitter).
func storedModelAKVSpan() cachemeta.Entry {
	return cachemeta.FromKVPrefix(
		cachemeta.KVPrefix{Tokens: []int{1, 2, 3, 4}, ModelID: "model-A", TokenizerID: "tok-A"},
		cachemeta.WithSerializer("serde-1"),
		cachemeta.WithPolicyVersion("policy-7"),
		cachemeta.WithLabel("position_regime", "rope:theta=1e4"),
		cachemeta.WithLabel("admitter_version", "admitter-3"),
	)
}

// TestEveryLocalServePathRefusesAMismatchedKey witnesses cache-frontier item 19 (#1537):
// a model-A KV span is fixed once, and a model-B request that is identical on every OTHER
// binding axis (and shares the token prefix, so a token-only match would WRONGLY serve)
// must fail closed at EVERY local serve path — the umbrella MaterializeVerdict, the pool
// reuse path that calls it, the radixkv bound prefix tree, and the contextq KV view gate —
// while the model-A request that owns the span is admitted. Local reuse stays governed by
// scope, freshness, taint, and quality, uniformly, with cold-path correctness explicit.
func TestEveryLocalServePathRefusesAMismatchedKey(t *testing.T) {
	storedA := storedModelAKVSpan()
	keyA := cachemeta.MaterializationKeyOf(storedA)
	if !keyA.Complete() {
		t.Fatalf("the model-A span key must be complete: %+v", keyA)
	}
	// model-B diverges ONLY on the model axis — the silent cross-model reuse #1537 forbids.
	keyB := keyA
	keyB.ModelID = "model-B"
	if ok, reason := keyA.Matches(keyB); ok || reason != cachemeta.ReasonModelMismatch {
		t.Fatalf("model-B must diverge on the model axis alone, got ok=%v reason=%q", ok, reason)
	}

	// Path 1 — the umbrella verdict itself (the shared function every path composes).
	if v := cachemeta.MaterializeVerdict(cachemeta.MatKVSpan, storedA, keyB, cachemeta.QualityEvidence{}); v.CanServe() {
		t.Fatalf("MaterializeVerdict must refuse a model-B key against a model-A span: %+v", v)
	}
	if v := cachemeta.MaterializeVerdict(cachemeta.MatKVSpan, storedA, keyA, cachemeta.QualityEvidence{}); !v.CanServe() {
		t.Fatalf("MaterializeVerdict must serve a model-A span for its own key: %+v", v)
	}

	// Path 2 — the cross-tenant pool reuse path. It calls MaterializeVerdict at its final
	// step, AFTER its own taint/scope/trust fences — so it witnesses the "scope, freshness,
	// taint" governance directly. Reach the key gate with a shareable, trusted cell.
	pooled := storedA
	pooled.Security.Scope = abi.ScopeTenant
	pooled.Security.Taint = abi.TaintTrusted
	if v := cachemeta.PoolReuseVerdict(pooled, keyB); v.CanServe() {
		t.Fatalf("PoolReuseVerdict must refuse a model-B key at the materialization gate: %+v", v)
	}
	if v := cachemeta.PoolReuseVerdict(pooled, keyA); !v.CanServe() {
		t.Fatalf("a shareable, trusted, key-matched cell must serve from the pool: %+v", v)
	}
	// The scope/taint fence is load-bearing too: a private (ScopeAgent) cell refuses
	// BEFORE the key gate — scope governs cross-tenant local reuse, not only the key.
	private := storedA
	private.Security.Scope = abi.ScopeAgent
	if v := cachemeta.PoolReuseVerdict(private, keyA); v.CanServe() {
		t.Fatalf("a private (ScopeAgent) cell must not serve across the pool even on a key match: %+v", v)
	}

	// Path 3 — the radixkv bound prefix tree. A model-B request that SHARES the token
	// prefix must fail closed WITHOUT taking a lease or splitting an edge.
	bt := radixkv.NewBound(0, keyA)
	root, matched := bt.Tree.Lookup([]int{1, 2, 3, 4, 5})
	seededFrom := []int{1, 2, 3, 4, 5}
	leaf := bt.Tree.Insert(root, seededFrom[matched:], nil)
	bt.Tree.Done(leaf)
	before := bt.Tree.Stats()
	if _, _, ok := bt.Lookup(keyB, []int{1, 2, 3, 9}); ok {
		t.Fatal("radixkv BoundTree.Lookup must refuse a model-B request that shares the token prefix")
	}
	if after := bt.Tree.Stats(); after.Splits != before.Splits || after.Nodes != before.Nodes {
		t.Fatalf("a refused cross-binding lookup must not mutate the tree: before=%+v after=%+v", before, after)
	}
	n2, matched2, ok2 := bt.Lookup(keyA, []int{1, 2, 3, 9})
	if !ok2 || matched2 != 3 {
		t.Fatalf("the model-A owner must reuse its shared [1,2,3] prefix: ok=%v matched=%d", ok2, matched2)
	}
	bt.Tree.Done(n2)

	// Path 4 — the contextq KV view gate. The same model-B key surfaces as a typed REFUSE
	// in the view-verdict stream, never a silent HIT.
	view := contextq.NewKVView("contextq-kv", 3, storedA, abi.ScopeAgent, cachemeta.MatKVPrefix, keyA, cachemeta.QualityEvidence{})
	if got := contextq.GateKVView(view, keyB); got.Kind != contextq.MaterializationRefuse {
		t.Fatalf("contextq GateKVView must refuse a model-B key, got kind=%v reason=%q", got.Kind, got.Reason)
	}
	if got := contextq.GateKVView(view, keyA); got.Kind != contextq.MaterializationHit {
		t.Fatalf("contextq GateKVView must hit for the owning model-A key, got kind=%v reason=%q", got.Kind, got.Reason)
	}
}

// TestApproximateLocalServeNeedsQualityEvidence witnesses the "quality evidence" axis of
// #1537: an approximate (compressed_kv) span with a full KEY MATCH is STILL refused until
// measured quality bounds its error — cold-path correctness stays explicit, an unproven
// span is never served on benefit of the doubt.
func TestApproximateLocalServeNeedsQualityEvidence(t *testing.T) {
	storedA := storedModelAKVSpan()
	keyA := cachemeta.MaterializationKeyOf(storedA)

	// No quality evidence (the fail-closed default): refused despite a full key match.
	if v := cachemeta.MaterializeVerdict(cachemeta.MatCompressedKV, storedA, keyA, cachemeta.QualityEvidence{}); v.CanServe() {
		t.Fatalf("an unmeasured approximate span must not serve on a key match alone: %+v", v)
	}
	// Measured quality within its admission bound: the same key-matched span now serves.
	good := cachemeta.QualityEvidence{Measured: true, QualityDelta: 0.01, MaxQualityDelta: 0.05}
	if v := cachemeta.MaterializeVerdict(cachemeta.MatCompressedKV, storedA, keyA, good); !v.CanServe() {
		t.Fatalf("a measured, in-bound approximate span must serve: %+v", v)
	}
}
