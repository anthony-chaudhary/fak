package cachemeta

import (
	"crypto/sha256"
	"encoding/hex"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// AttentionIndex is the payload-free metadata for a dynamic-sparse-attention
// index artifact. GLM-5.2's DSA/IndexShare path computes content-dependent key
// selections that can be shared across a layer group; cachemeta records the
// binding axes and dependency graph, never the index payload itself.
type AttentionIndex struct {
	PrefixDigest      string
	Tokens            []int
	PrefixLength      int64
	ModelID           string
	TokenizerID       string
	PositionMode      PositionMode
	IndexerID         string
	LayerGroup        string
	Layers            []int
	DecisionDigest    string
	ParentKV          EntryID
	Owner             string
	Lease             string
	Causal            bool
	CausalityWitness  string
	QualityDeltaProbe float64
}

// AttentionIndexRequest is the set of axes a lookup must match before a DSA
// index may be reused. A non-causal candidate fails closed unless AllowNonCausal
// is explicit, because prefix exactness depends on the indexer being determined
// only by tokens at positions <= the query position.
type AttentionIndexRequest struct {
	PrefixDigest    string
	Tokens          []int
	PrefixLength    int64
	ModelID         string
	TokenizerID     string
	PositionMode    PositionMode
	IndexerID       string
	LayerGroup      string
	DecisionDigest  string
	ParentKV        EntryID
	MaxQualityDelta float64
	AllowNonCausal  bool
}

// AttentionIndexBindingDigest is the deterministic identity for the DSA index
// binding. It covers the GLM-5.2 axes called out in the architecture memo:
// model, tokenizer, position convention, prefix digest, indexer version, layer
// group, layer consumers, and the digest of the computed index decisions.
func AttentionIndexBindingDigest(a AttentionIndex) string {
	a = normalizeAttentionIndex(a)
	h := sha256.New()
	writeField := func(s string) {
		_, _ = h.Write([]byte(s))
		_, _ = h.Write([]byte{0})
	}
	writeField(a.PrefixDigest)
	writeField(strconv.FormatInt(a.PrefixLength, 10))
	writeField(a.ModelID)
	writeField(a.TokenizerID)
	writeField(string(a.PositionMode))
	writeField(a.IndexerID)
	writeField(a.LayerGroup)
	writeField(a.DecisionDigest)
	for _, layer := range a.Layers {
		writeField(strconv.Itoa(layer))
	}
	return hex.EncodeToString(h.Sum(nil))
}

// FromAttentionIndex lowers a DSA/IndexShare index artifact into the
// attention_index plane. Consumers name the layers that share this index; ParentKV
// points at the K/V span whose eviction must invalidate the index too.
func FromAttentionIndex(a AttentionIndex, opts ...Option) Entry {
	a = normalizeAttentionIndex(a)
	owner := a.Owner
	if owner == "" {
		owner = "attention-index"
	}
	e := Entry{
		ID: EntryID{
			Digest:    AttentionIndexBindingDigest(a),
			MediaType: MediaAttentionIndex,
			Length:    a.PrefixLength,
			Unit:      UnitPositions,
		},
		Plane: PlaneAttentionIndex,
		Derivation: Derivation{
			Producer:     owner,
			ModelID:      a.ModelID,
			TokenizerID:  a.TokenizerID,
			PositionMode: a.PositionMode,
		},
		Validity: Validity{
			Witness: a.CausalityWitness,
		},
		Security: Security{
			Taint:            abi.TaintTrusted,
			Scope:            abi.ScopeFleet,
			AdmissionVerdict: AdmissionAllow,
			AdmittedBy:       owner,
			Reason:           "dsa_attention_index_not_semantic_proof",
		},
		Residency: Residency{Tier: TierDRAM, Owner: owner, Lease: a.Lease},
		Coherence: Coherence{
			Consumers:        attentionIndexConsumers(a.Layers),
			InvalidationMode: InvalidationPolicy,
		},
		Metrics: Metrics{
			QualityDeltaProbe: a.QualityDeltaProbe,
		},
		Labels: map[string]string{
			"prefix_digest":   a.PrefixDigest,
			"indexer_id":      a.IndexerID,
			"layer_group":     a.LayerGroup,
			"decision_digest": a.DecisionDigest,
			"layers":          joinInts(a.Layers),
			"causal":          strconv.FormatBool(a.Causal),
		},
	}
	if a.ParentKV.Valid() {
		e.Coherence.Parents = []EntryID{a.ParentKV}
	}
	if a.CausalityWitness != "" {
		e.Labels["causality_witness"] = a.CausalityWitness
	}
	apply(&e, opts)
	return e
}

// AttentionIndexLookup checks whether a candidate DSA index may serve a request.
// It is deliberately stricter than a plain KV-prefix lookup: both the token
// prefix and the computed decision digest must match, and non-causal candidates
// fault by default.
func AttentionIndexLookup(req AttentionIndexRequest, candidate AttentionIndex) LookupVerdict {
	req = normalizeAttentionIndexRequest(req)
	candidate = normalizeAttentionIndex(candidate)
	e := FromAttentionIndex(candidate)
	if !candidate.Causal && !req.AllowNonCausal {
		return Fault(e, ReasonNonCausalIndex)
	}
	if req.ModelID == "" || req.TokenizerID == "" || req.PrefixDigest == "" ||
		req.IndexerID == "" || req.DecisionDigest == "" {
		return Miss(ReasonIndexMismatch)
	}
	if candidate.ModelID == "" || candidate.TokenizerID == "" || candidate.PrefixDigest == "" ||
		candidate.IndexerID == "" || candidate.DecisionDigest == "" {
		return Fault(e, ReasonIndexMismatch)
	}
	if req.ModelID != "" && req.ModelID != candidate.ModelID {
		return Miss(ReasonModelMismatch)
	}
	if req.TokenizerID != "" && req.TokenizerID != candidate.TokenizerID {
		return Miss(ReasonTokenizerMismatch)
	}
	if req.PositionMode != "" && req.PositionMode != candidate.PositionMode {
		return Miss(ReasonPositionMismatch)
	}
	if req.PrefixDigest != "" && req.PrefixDigest != candidate.PrefixDigest {
		return Miss(ReasonIndexMismatch)
	}
	if req.PrefixLength > 0 && req.PrefixLength != candidate.PrefixLength {
		return Miss(ReasonIndexMismatch)
	}
	if req.IndexerID != "" && req.IndexerID != candidate.IndexerID {
		return Miss(ReasonIndexMismatch)
	}
	if req.LayerGroup != "" && req.LayerGroup != candidate.LayerGroup {
		return Miss(ReasonIndexMismatch)
	}
	if req.DecisionDigest != "" && req.DecisionDigest != candidate.DecisionDigest {
		return Miss(ReasonIndexMismatch)
	}
	if req.ParentKV.Valid() && !sameEntryID(req.ParentKV, candidate.ParentKV) {
		return Miss(ReasonIndexMismatch)
	}
	if req.MaxQualityDelta > 0 && candidate.QualityDeltaProbe > req.MaxQualityDelta {
		return Miss(ReasonApproxFault)
	}
	return Hit(e)
}

// AttentionIndexReferences reports whether an attention_index entry depends on a
// K/V span. kvmmu/external-engine bridges can use this predicate when a poisoned
// span is quarantined: any referencing DSA index must be invalidated with it.
func AttentionIndexReferences(e Entry, kv EntryID) bool {
	if e.Plane != PlaneAttentionIndex || !kv.Valid() {
		return false
	}
	for _, p := range e.Coherence.Parents {
		if sameEntryID(p, kv) {
			return true
		}
	}
	return false
}

func normalizeAttentionIndex(a AttentionIndex) AttentionIndex {
	if a.PrefixDigest == "" && len(a.Tokens) > 0 {
		a.PrefixDigest = DigestTokenIDs(a.Tokens)
	}
	if a.PrefixLength == 0 && len(a.Tokens) > 0 {
		a.PrefixLength = int64(len(a.Tokens))
	}
	if a.PositionMode == "" {
		a.PositionMode = PositionPrefixAligned
	}
	return a
}

func normalizeAttentionIndexRequest(req AttentionIndexRequest) AttentionIndexRequest {
	if req.PrefixDigest == "" && len(req.Tokens) > 0 {
		req.PrefixDigest = DigestTokenIDs(req.Tokens)
	}
	if req.PrefixLength == 0 && len(req.Tokens) > 0 {
		req.PrefixLength = int64(len(req.Tokens))
	}
	if req.PositionMode == "" {
		req.PositionMode = PositionPrefixAligned
	}
	return req
}

func attentionIndexConsumers(layers []int) []Consumer {
	if len(layers) == 0 {
		return nil
	}
	out := make([]Consumer, 0, len(layers))
	for _, layer := range layers {
		out = append(out, Consumer{Kind: "attention_layer", ID: strconv.Itoa(layer)})
	}
	return out
}

func joinInts(xs []int) string {
	if len(xs) == 0 {
		return ""
	}
	parts := make([]string, len(xs))
	for i, x := range xs {
		parts[i] = strconv.Itoa(x)
	}
	return strings.Join(parts, ",")
}

func sameEntryID(a, b EntryID) bool {
	return a.Digest == b.Digest && a.MediaType == b.MediaType && a.Length == b.Length && a.Unit == b.Unit
}
