package cachemeta

import "testing"

// semanticView is a model-agnostic source-linked view fixture: one summary view
// over one recall page, carrying a world witness.
func semanticView() MemoryView {
	return MemoryView{
		ViewID:   "view-summary-1",
		ViewType: "summary",
		Length:   96,
		SourceRefs: []EntryID{
			{Digest: "src-page-1", MediaType: MediaRecallPage, Length: 128, Unit: UnitBytes},
		},
		Producer:          "contextq",
		PolicyVersion:     "policy-7",
		Coverage:          0.75,
		FaithfulnessProbe: 1.0,
		Witness:           "git_sha:abc123",
	}
}

// kvSpanUnder builds a fully-keyed local KV span entry under a backend so its
// MaterializationKey is Complete (every axis the issue enumerates is populated).
func kvSpanUnder(model, tok string) Entry {
	return FromKVPrefix(
		KVPrefix{Tokens: []int{1, 2, 3, 4}, ModelID: model, TokenizerID: tok},
		WithSerializer("serde-1"),
		WithPolicyVersion("policy-7"),
		WithLabel("position_regime", "rope:theta=1e4,scale=linear"),
		WithLabel("admitter_version", "admitter-3"),
	)
}

// TestSemanticViewSharesAcrossBackendsButKVDoesNot witnesses acceptance #5 (and
// exercises #1, #2, #4): a semantic view is reusable under any backend, while its
// KV materialization is refused the moment any key axis diverges.
func TestSemanticViewSharesAcrossBackendsButKVDoesNot(t *testing.T) {
	view := semanticView()

	// The SEMANTIC view is model-agnostic: built under two different backends it
	// keeps ONE identity, so it can be shared across model backends.
	semA := FromMemoryView(view, WithModel("model-A", "tok-A"))
	semB := FromMemoryView(view, WithModel("model-B", "tok-B"))
	if !SemanticShareable(semA) || !SemanticShareable(semB) {
		t.Fatalf("a memory_view must be cross-backend shareable: %+v / %+v", semA.ID, semB.ID)
	}
	if semA.ID.Digest != semB.ID.Digest {
		t.Fatalf("semantic view identity must NOT depend on the backend: A=%s B=%s",
			semA.ID.Digest, semB.ID.Digest)
	}

	// The KV MATERIALIZATION is backend-specific. Build a span under model A and
	// ask whether it may be served for a request keyed to model B.
	storedA := kvSpanUnder("model-A", "tok-A")
	keyA := MaterializationKeyOf(storedA)
	keyB := MaterializationKeyOf(kvSpanUnder("model-B", "tok-B"))
	if !keyA.Complete() || !keyB.Complete() {
		t.Fatalf("materialization keys must be complete: A=%+v B=%+v", keyA, keyB)
	}
	if ok, _ := keyA.Matches(keyB); ok {
		t.Fatal("a model-A KV span must NOT match a model-B key — KV is not cross-backend")
	}

	// The cross-backend KV serve is refused with the typed model-mismatch reason.
	cross := MaterializeVerdict(MatKVSpan, storedA, keyB, QualityEvidence{})
	if cross.CanServe() {
		t.Fatalf("a KV span built under model A must NOT serve under model B: %+v", cross)
	}
	if cross.Reason != ReasonModelMismatch {
		t.Fatalf("cross-backend KV refusal should be model_mismatch, got %q", cross.Reason)
	}

	// Same backend: the exact-key span serves.
	same := MaterializeVerdict(MatKVSpan, storedA, keyA, QualityEvidence{})
	if !same.CanServe() {
		t.Fatalf("a KV span must serve for its own backend key: %+v", same)
	}
}

// TestMaterializationKeyEnforcesEveryAxis witnesses acceptance #2: each enumerated
// axis (model, tokenizer, serializer, RoPE/position regime, policy, admitter) is
// load-bearing — flipping any one alone refuses the reuse with its typed reason.
func TestMaterializationKeyEnforcesEveryAxis(t *testing.T) {
	base := MaterializationKey{
		ModelID: "m", TokenizerID: "t", SerializerID: "s",
		PositionRegime: "rope-1", PolicyVersion: "p", AdmitterVersion: "a",
	}
	if !base.Complete() {
		t.Fatalf("base key should be complete: %+v", base)
	}
	if ok, r := base.Matches(base); !ok || r != ReasonNone {
		t.Fatalf("an identical key must match with ReasonNone, got ok=%v r=%q", ok, r)
	}

	cases := []struct {
		name   string
		mutate func(MaterializationKey) MaterializationKey
		want   LookupReason
	}{
		{"model", func(k MaterializationKey) MaterializationKey { k.ModelID = "x"; return k }, ReasonModelMismatch},
		{"tokenizer", func(k MaterializationKey) MaterializationKey { k.TokenizerID = "x"; return k }, ReasonTokenizerMismatch},
		{"serializer", func(k MaterializationKey) MaterializationKey { k.SerializerID = "x"; return k }, ReasonSerializerMismatch},
		{"position", func(k MaterializationKey) MaterializationKey { k.PositionRegime = "x"; return k }, ReasonPositionMismatch},
		{"policy", func(k MaterializationKey) MaterializationKey { k.PolicyVersion = "x"; return k }, ReasonPolicyMismatch},
		{"admitter", func(k MaterializationKey) MaterializationKey { k.AdmitterVersion = "x"; return k }, ReasonAdmitterMismatch},
	}
	for _, tc := range cases {
		other := tc.mutate(base)
		ok, r := base.Matches(other)
		if ok {
			t.Fatalf("%s axis: a divergent key must NOT match", tc.name)
		}
		if r != tc.want {
			t.Fatalf("%s axis: want reason %q, got %q", tc.name, tc.want, r)
		}
		// A key missing this axis is incomplete and fails closed.
		missing := tc.mutate(base)
		switch tc.name {
		case "model":
			missing.ModelID = ""
		case "tokenizer":
			missing.TokenizerID = ""
		case "serializer":
			missing.SerializerID = ""
		case "position":
			missing.PositionRegime = ""
		case "policy":
			missing.PolicyVersion = ""
		case "admitter":
			missing.AdmitterVersion = ""
		}
		if missing.Complete() {
			t.Fatalf("%s axis: clearing it must make the key incomplete", tc.name)
		}
	}
}

// TestCompressedKVRequiresQualityEvidence witnesses acceptance #4: an approximate
// (compressed) KV view with a MATCHING key is still refused until quality/fault
// metrics are present and clear the admission bound.
func TestCompressedKVRequiresQualityEvidence(t *testing.T) {
	stored := kvSpanUnder("model-A", "tok-A")
	key := MaterializationKeyOf(stored)

	// Exact KV span with a matching key serves with no quality evidence needed.
	if v := MaterializeVerdict(MatKVSpan, stored, key, QualityEvidence{}); !v.CanServe() {
		t.Fatalf("exact kv_span with matching key must serve: %+v", v)
	}

	// Compressed KV, matching key, but UNMEASURED -> refused as approximate fault.
	unmeasured := MaterializeVerdict(MatCompressedKV, stored, key, QualityEvidence{})
	if unmeasured.CanServe() {
		t.Fatal("compressed_kv must NOT be reported as a hit without quality metrics")
	}
	if unmeasured.Reason != ReasonApproxFault {
		t.Fatalf("unmeasured compressed_kv should be approximate_fault, got %q", unmeasured.Reason)
	}

	// Measured but OVER the admission bound -> still refused.
	over := MaterializeVerdict(MatCompressedKV, stored, key,
		QualityEvidence{Measured: true, QualityDelta: 0.20, MaxQualityDelta: 0.05})
	if over.CanServe() {
		t.Fatalf("compressed_kv over the quality bound must not serve: %+v", over)
	}

	// Measured AND within bound -> serves.
	ok := MaterializeVerdict(MatCompressedKV, stored, key,
		QualityEvidence{Measured: true, QualityDelta: 0.01, MaxQualityDelta: 0.05})
	if !ok.CanServe() {
		t.Fatalf("compressed_kv with acceptable quality evidence must serve: %+v", ok)
	}
}

// TestProviderPrefixIsTelemetryNotTrust witnesses acceptance #3 through the bridge:
// a provider_prefix materialization is never a serveable local hit, regardless of
// any key, so its cached tokens are performance evidence only.
func TestProviderPrefixIsTelemetryNotTrust(t *testing.T) {
	prov := FromProviderCache(ProviderCache{
		Provider: "anthropic", ModelID: "claude-opus", CachedTokens: 1500, PromptTokens: 1800,
	})
	v := MaterializeVerdict(MatProviderPrefix, prov, MaterializationKey{}, QualityEvidence{})
	if v.CanServe() {
		t.Fatalf("provider_prefix must never serve as a local hit: %+v", v)
	}
	if v.Meta["provider_cache"] != "cost_latency_only" {
		t.Fatalf("provider_prefix verdict should mark cost/latency-only telemetry: %+v", v)
	}
}

// TestCrossModelPrefillShareLiftsModelIDBarrierOnly witnesses #534: the exact-ModelID
// barrier is lifted for a declared-compatible family, but ONLY opt-in, ONLY on the
// ModelID axis, and the HIT carries the KVCache.Clone-splice audit trail. Without the
// policy the verdict keeps its pre-#534 exact-ModelID refusal.
func TestCrossModelPrefillShareLiftsModelIDBarrierOnly(t *testing.T) {
	storedA := kvSpanUnder("model-A", "tok-A")
	keyA := MaterializationKeyOf(storedA)
	// A compatible-family consumer: identical EXCEPT for ModelID (the only axis a
	// same-Family+same-PrefixDigest pair can differ on), with the caller asserting a
	// lossless share.
	keyB := keyA
	keyB.ModelID = "model-B"
	share := PrefillSharePolicy{Allowed: true, Family: "qwen", PrefixDigest: "sha-AAA"}

	// (1) With the policy, the cross-model serve is a HIT, stamped as a Clone splice.
	hit := MaterializeVerdict(MatKVSpan, storedA, keyB, QualityEvidence{}, WithPrefillShare(share))
	if !hit.CanServe() {
		t.Fatalf("a share-compatible consumer must HIT across ModelID: %+v", hit)
	}
	if hit.Meta["cross_model_share"] != "true" || hit.Meta["splice"] != "model.KVCache.Clone" {
		t.Fatalf("cross-model HIT must mark the Clone splice + share: %+v", hit.Meta)
	}
	if hit.Entry.Labels["share_family"] != "qwen" {
		t.Fatalf("cross-model HIT must carry the share family for audit: %+v", hit.Entry.Labels)
	}

	// (2) WITHOUT the policy, the same cross-model serve is still refused — the lift is
	//     opt-in, byte-identical to pre-#534.
	refuse := MaterializeVerdict(MatKVSpan, storedA, keyB, QualityEvidence{})
	if refuse.CanServe() || refuse.Reason != ReasonModelMismatch {
		t.Fatalf("without a share policy a cross-model serve must be model_mismatch: %+v", refuse)
	}

	// (3) The lift is ModelID-ONLY: a consumer that also differs in tokenizer is refused
	//     even with a share policy — a lying CanShare can never relax the rest of the
	//     binding (matchesExceptModel).
	keyBadTok := keyB
	keyBadTok.TokenizerID = "tok-Z"
	bad := MaterializeVerdict(MatKVSpan, storedA, keyBadTok, QualityEvidence{}, WithPrefillShare(share))
	if bad.CanServe() {
		t.Fatalf("a tokenizer mismatch must NOT be lifted by a share policy: %+v", bad)
	}

	// (4) An Allowed:false policy never lifts (the caller did not assert a lossless share).
	noShare := MaterializeVerdict(MatKVSpan, storedA, keyB, QualityEvidence{},
		WithPrefillShare(PrefillSharePolicy{Allowed: false, Family: "qwen"}))
	if noShare.CanServe() {
		t.Fatalf("Allowed:false must not lift the barrier: %+v", noShare)
	}

	// (5) An approximate view still needs quality evidence even on a cross-model share —
	//     the lift is exact-KV only; it does not waive the approximate-fault gate.
	unmeasured := MaterializeVerdict(MatCompressedKV, storedA, keyB, QualityEvidence{}, WithPrefillShare(share))
	if unmeasured.CanServe() || unmeasured.Reason != ReasonApproxFault {
		t.Fatalf("a compressed cross-model share still needs quality evidence: %+v", unmeasured)
	}
}

// TestMaterializationBenchRowCarriesProvenance witnesses acceptance #6: one
// benchmark row reports provider cached tokens / local KV hits ALONGSIDE the
// semantic-view witness that drove the materialization — never a bare number.
func TestMaterializationBenchRowCarriesProvenance(t *testing.T) {
	sem := FromMemoryView(semanticView(), WithModel("model-A", "tok-A"))

	var split SavingsSplit
	split.Add(FromProviderCache(ProviderCache{Provider: "anthropic", CachedTokens: 1200}))
	// A local KV entry crediting reuse tokens (provider tokens must NOT leak here).
	local := FromKVPrefix(KVPrefix{Tokens: []int{1, 2}, ModelID: "model-A"})
	local.Metrics.PrefillTokensSaved = 300
	split.Add(local)

	row := NewMaterializationBenchRow(sem, MatKVSpan, split, 5)

	if row.SemanticWitness != "git_sha:abc123" {
		t.Fatalf("row must carry the source-linked semantic witness, got %q", row.SemanticWitness)
	}
	if row.SemanticViewDigest == "" || row.SemanticViewID != "view-summary-1" {
		t.Fatalf("row must identify the underlying semantic view: %+v", row)
	}
	if row.ProviderReadTokens != 1200 {
		t.Fatalf("row must report provider cached tokens, got %d", row.ProviderReadTokens)
	}
	if row.LocalReuseTokens != 300 {
		t.Fatalf("provider tokens must not leak into local reuse; got %d", row.LocalReuseTokens)
	}
	if row.LocalKVHits != 5 {
		t.Fatalf("row must report local KV hit count, got %d", row.LocalKVHits)
	}
}
