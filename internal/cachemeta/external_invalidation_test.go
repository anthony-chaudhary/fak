package cachemeta

import "testing"

func TestPlanExternalInvalidationsDropsRemoteKVAndReferencingAttentionIndex(t *testing.T) {
	remoteKV := FromKVPrefix(
		KVPrefix{Tokens: []int{10, 20, 30, 40}, ModelID: "glm-5.2", TokenizerID: "glm-tokenizer", Owner: "radixkv"},
		WithResidency(TierProvider, "sglang", "session-7"),
		WithLabel("provider", "sglang"),
		WithLabel("engine", "glm-moe-dsa"),
	)
	idx := FromAttentionIndex(
		AttentionIndex{
			Tokens:         []int{10, 20, 30, 40},
			ModelID:        "glm-5.2",
			TokenizerID:    "glm-tokenizer",
			IndexerID:      "glm52-dsa-indexer:v1",
			LayerGroup:     "layers-0-3",
			Layers:         []int{0, 1, 2, 3},
			DecisionDigest: DigestBytes([]byte("topk")),
			ParentKV:       remoteKV.ID,
			Owner:          "sglang-dsa",
			Lease:          "session-7",
			Causal:         true,
		},
		WithResidency(TierProvider, "sglang", "session-7"),
		WithLabel("provider", "sglang"),
		WithLabel("engine", "glm-moe-dsa"),
	)
	other := FromKVPrefix(
		KVPrefix{Tokens: []int{99}, ModelID: "glm-5.2", TokenizerID: "glm-tokenizer"},
		WithResidency(TierProvider, "sglang", "session-8"),
	)
	telemetry := FromProviderCache(ProviderCache{
		Provider:     "sglang",
		ModelID:      "glm-5.2",
		CachedTokens: 128,
		PromptTokens: 256,
	})

	dirs := PlanExternalInvalidations(remoteKV.ID, []Entry{remoteKV, idx, other, telemetry})
	if len(dirs) != 2 {
		t.Fatalf("directives = %d, want remote K/V + attention index: %+v", len(dirs), dirs)
	}
	byKind := map[ExternalInvalidationKind]ExternalInvalidationDirective{}
	for _, d := range dirs {
		byKind[d.Kind] = d
		if d.Provider != "sglang" || d.Engine != "glm-moe-dsa" {
			t.Fatalf("directive lost provider/engine labels: %+v", d)
		}
		if d.Residency.Tier != TierProvider || d.Residency.Owner != "sglang" || d.Residency.Lease != "session-7" {
			t.Fatalf("directive lost external residency: %+v", d)
		}
	}
	if d := byKind[ExternalInvalidateKVSpan]; d.Entry != remoteKV.ID || d.Reason != "poisoned_kv" {
		t.Fatalf("bad K/V directive: %+v", d)
	}
	if d := byKind[ExternalInvalidateAttentionIndex]; d.Entry != idx.ID || d.Reason != "parent_kv_poisoned" {
		t.Fatalf("bad attention-index directive: %+v", d)
	}
	for _, d := range dirs {
		if d.Entry == telemetry.ID {
			t.Fatalf("provider telemetry must not become an invalidation directive: %+v", dirs)
		}
	}
}

func TestPlanExternalInvalidationsRejectsEmptyPoisonedKV(t *testing.T) {
	if dirs := PlanExternalInvalidations(EntryID{}, []Entry{FromKVPrefix(KVPrefix{Tokens: []int{1}})}); len(dirs) != 0 {
		t.Fatalf("empty poisoned K/V should produce no directives: %+v", dirs)
	}
}
