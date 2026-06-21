package cachemeta

import (
	"strconv"
	"testing"
)

func sampleAttentionIndex() (AttentionIndex, EntryID) {
	kv := FromKVPrefix(KVPrefix{
		Tokens:      []int{10, 20, 30, 40},
		ModelID:     "glm-5.2",
		TokenizerID: "glm-tokenizer",
		Owner:       "radixkv",
	})
	idx := AttentionIndex{
		Tokens:           []int{10, 20, 30, 40},
		ModelID:          "glm-5.2",
		TokenizerID:      "glm-tokenizer",
		IndexerID:        "glm52-dsa-indexer:v1",
		LayerGroup:       "layers-0-3",
		Layers:           []int{0, 1, 2, 3},
		DecisionDigest:   DigestBytes([]byte("query-topk-decisions")),
		ParentKV:         kv.ID,
		Owner:            "glm-dsa",
		Lease:            "session-7",
		Causal:           true,
		CausalityWitness: "oracle:causal-indexer",
	}
	return idx, kv.ID
}

func TestFromAttentionIndexRecordsDSAPlaneAndIndexShareConsumers(t *testing.T) {
	idx, parent := sampleAttentionIndex()
	e := FromAttentionIndex(idx)
	if e.Plane != PlaneAttentionIndex || e.ID.MediaType != MediaAttentionIndex {
		t.Fatalf("bad attention-index identity: %+v", e)
	}
	if e.ID.Length != int64(len(idx.Tokens)) || e.ID.Unit != UnitPositions {
		t.Fatalf("bad attention-index length/unit: %+v", e.ID)
	}
	if e.Derivation.ModelID != "glm-5.2" || e.Derivation.TokenizerID != "glm-tokenizer" ||
		e.Derivation.PositionMode != PositionPrefixAligned {
		t.Fatalf("bad derivation axes: %+v", e.Derivation)
	}
	if e.Labels["prefix_digest"] != DigestTokenIDs(idx.Tokens) ||
		e.Labels["decision_digest"] != idx.DecisionDigest ||
		e.Labels["layer_group"] != "layers-0-3" ||
		e.Labels["causal"] != "true" {
		t.Fatalf("DSA labels missing: %+v", e.Labels)
	}
	if len(e.Coherence.Consumers) != 4 {
		t.Fatalf("IndexShare consumers = %d, want 4: %+v", len(e.Coherence.Consumers), e.Coherence.Consumers)
	}
	for i, c := range e.Coherence.Consumers {
		if c.Kind != "attention_layer" || c.ID != strconv.Itoa(i) {
			t.Fatalf("consumer[%d] = %+v, want attention_layer/%d", i, c, i)
		}
	}
	if len(e.Coherence.Parents) != 1 || e.Coherence.Parents[0] != parent {
		t.Fatalf("attention index must parent the KV span it indexes: %+v", e.Coherence.Parents)
	}
	if !AttentionIndexReferences(e, parent) {
		t.Fatalf("AttentionIndexReferences did not find parent KV span")
	}
	if e.Security.Reason != "dsa_attention_index_not_semantic_proof" {
		t.Fatalf("attention-index hit must not be semantic proof, got %q", e.Security.Reason)
	}
}

func TestAttentionIndexLookupRequiresPrefixDecisionAndCausality(t *testing.T) {
	idx, _ := sampleAttentionIndex()
	req := AttentionIndexRequest{
		Tokens:         idx.Tokens,
		ModelID:        idx.ModelID,
		TokenizerID:    idx.TokenizerID,
		IndexerID:      idx.IndexerID,
		LayerGroup:     idx.LayerGroup,
		DecisionDigest: idx.DecisionDigest,
	}
	if v := AttentionIndexLookup(req, idx); v.Kind != LookupHit || !v.CanServe() {
		t.Fatalf("matching DSA index should HIT, got %+v", v)
	}

	badDecision := req
	badDecision.DecisionDigest = DigestBytes([]byte("different-topk"))
	if v := AttentionIndexLookup(badDecision, idx); v.Kind != LookupMiss || v.Reason != ReasonIndexMismatch {
		t.Fatalf("decision mismatch should MISS(index_mismatch), got %+v", v)
	}

	missingDecision := req
	missingDecision.DecisionDigest = ""
	if v := AttentionIndexLookup(missingDecision, idx); v.Kind != LookupMiss || v.Reason != ReasonIndexMismatch {
		t.Fatalf("missing request decision digest should MISS(index_mismatch), got %+v", v)
	}

	badPrefix := req
	badPrefix.Tokens = []int{10, 20, 99, 40}
	badPrefix.PrefixDigest = ""
	if v := AttentionIndexLookup(badPrefix, idx); v.Kind != LookupMiss || v.Reason != ReasonIndexMismatch {
		t.Fatalf("prefix mismatch should MISS(index_mismatch), got %+v", v)
	}

	badModel := req
	badModel.ModelID = "other-model"
	if v := AttentionIndexLookup(badModel, idx); v.Kind != LookupMiss || v.Reason != ReasonModelMismatch {
		t.Fatalf("model mismatch should MISS(model_mismatch), got %+v", v)
	}

	nonCausal := idx
	nonCausal.Causal = false
	if v := AttentionIndexLookup(req, nonCausal); v.Kind != LookupFault || v.Reason != ReasonNonCausalIndex {
		t.Fatalf("non-causal DSA index should FAULT(non_causal_index), got %+v", v)
	}

	incompleteCandidate := idx
	incompleteCandidate.Tokens = nil
	incompleteCandidate.PrefixDigest = ""
	if v := AttentionIndexLookup(req, incompleteCandidate); v.Kind != LookupFault || v.Reason != ReasonIndexMismatch {
		t.Fatalf("candidate without prefix binding should FAULT(index_mismatch), got %+v", v)
	}
}

func TestAttentionIndexLookupBindsParentAndQualityBudget(t *testing.T) {
	idx, parent := sampleAttentionIndex()
	idx.QualityDeltaProbe = 0.05
	req := AttentionIndexRequest{
		Tokens:          idx.Tokens,
		ModelID:         idx.ModelID,
		TokenizerID:     idx.TokenizerID,
		IndexerID:       idx.IndexerID,
		LayerGroup:      idx.LayerGroup,
		DecisionDigest:  idx.DecisionDigest,
		ParentKV:        parent,
		MaxQualityDelta: 0.1,
	}
	if v := AttentionIndexLookup(req, idx); v.Kind != LookupHit {
		t.Fatalf("matching parent and quality budget should HIT, got %+v", v)
	}

	tightBudget := req
	tightBudget.MaxQualityDelta = 0.01
	if v := AttentionIndexLookup(tightBudget, idx); v.Kind != LookupMiss || v.Reason != ReasonApproxFault {
		t.Fatalf("quality budget miss should be approximate_fault, got %+v", v)
	}

	wrongParent := req
	wrongParent.ParentKV = EntryID{Digest: "other", MediaType: MediaKVSpan, Length: parent.Length, Unit: UnitPositions}
	if v := AttentionIndexLookup(wrongParent, idx); v.Kind != LookupMiss || v.Reason != ReasonIndexMismatch {
		t.Fatalf("parent mismatch should MISS(index_mismatch), got %+v", v)
	}
}

func TestAttentionIndexDigestIncludesIndexShareLayerSet(t *testing.T) {
	idx, _ := sampleAttentionIndex()
	other := idx
	other.Layers = []int{4, 5, 6, 7}
	if AttentionIndexBindingDigest(idx) == AttentionIndexBindingDigest(other) {
		t.Fatal("attention-index digest ignored layer consumers")
	}
}
