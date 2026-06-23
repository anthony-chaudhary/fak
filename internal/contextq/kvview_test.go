package contextq

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/cachemeta"
)

// fullKey is a complete model/tokenizer/position binding identity — every axis the
// KV gate requires is populated, so it is a valid reuse key.
func fullKey() cachemeta.MaterializationKey {
	return cachemeta.MaterializationKey{
		ModelID:         "llama-3.1-8b",
		TokenizerID:     "llama-bpe-v3",
		SerializerID:    "chatml-v1",
		PositionRegime:  "rope-theta-500000",
		PolicyVersion:   "p1",
		AdmitterVersion: "a1",
	}
}

func kvSourceEntry() cachemeta.Entry {
	return cachemeta.Entry{
		ID: cachemeta.EntryID{
			Digest:    "deadbeefcafef00d",
			MediaType: cachemeta.MediaMemoryView,
			Length:    4096,
			Unit:      cachemeta.UnitBytes,
		},
	}
}

// TestGateKVView_CrossModelFailsClosed is the load-bearing acceptance for #514 /
// epic #437 #7: a KV span materialized under one model/tokenizer/position regime is
// REFUSED under another — never served as a silent HIT.
func TestGateKVView_CrossModelFailsClosed(t *testing.T) {
	src := kvSourceEntry()
	view := NewKVView("contextq-kv", 3, src, abi.ScopeAgent, cachemeta.MatKVPrefix, fullKey(), cachemeta.QualityEvidence{})

	cases := []struct {
		name       string
		mutate     func(k cachemeta.MaterializationKey) cachemeta.MaterializationKey
		wantKind   MaterializationKind
		wantReason string
	}{
		{
			name:     "exact match hits",
			mutate:   func(k cachemeta.MaterializationKey) cachemeta.MaterializationKey { return k },
			wantKind: MaterializationHit, wantReason: "kv_materialization_match",
		},
		{
			name: "different model refuses",
			mutate: func(k cachemeta.MaterializationKey) cachemeta.MaterializationKey {
				k.ModelID = "qwen-2.5-7b"
				return k
			},
			wantKind: MaterializationRefuse, wantReason: "kv_model_mismatch",
		},
		{
			name: "different tokenizer refuses",
			mutate: func(k cachemeta.MaterializationKey) cachemeta.MaterializationKey {
				k.TokenizerID = "qwen-bpe-v2"
				return k
			},
			wantKind: MaterializationRefuse, wantReason: "kv_tokenizer_mismatch",
		},
		{
			name: "different position regime refuses",
			mutate: func(k cachemeta.MaterializationKey) cachemeta.MaterializationKey {
				k.PositionRegime = "rope-theta-10000"
				return k
			},
			wantKind: MaterializationRefuse, wantReason: "kv_position_mismatch",
		},
		{
			name: "policy drift refuses",
			mutate: func(k cachemeta.MaterializationKey) cachemeta.MaterializationKey {
				k.PolicyVersion = "p2"
				return k
			},
			wantKind: MaterializationRefuse, wantReason: "kv_policy_mismatch",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			want := tc.mutate(fullKey())
			got := GateKVView(view, want)
			if got.Kind != tc.wantKind {
				t.Fatalf("kind = %q, want %q (reason %q)", got.Kind, tc.wantKind, got.Reason)
			}
			if got.Reason != tc.wantReason {
				t.Fatalf("reason = %q, want %q", got.Reason, tc.wantReason)
			}
			// The non-negotiable invariant: a mismatch must NEVER read as a HIT.
			if tc.wantKind == MaterializationRefuse && got.Kind == MaterializationHit {
				t.Fatalf("cross-model reuse leaked a silent HIT")
			}
			// Provenance is preserved onto the verdict regardless of outcome.
			if got.ViewID != view.ViewID || got.Step != 3 {
				t.Fatalf("verdict lost provenance: view=%q step=%d", got.ViewID, got.Step)
			}
		})
	}
}

// TestGateKVView_IncompleteKeyFailsClosed: a missing binding axis cannot prove a
// match, so the gate refuses rather than serving a possibly-mismatched span.
func TestGateKVView_IncompleteKeyFailsClosed(t *testing.T) {
	src := kvSourceEntry()
	partial := fullKey()
	partial.SerializerID = "" // an unproven axis
	view := NewKVView("", 1, src, abi.ScopeAgent, cachemeta.MatKVSpan, partial, cachemeta.QualityEvidence{})

	if got := GateKVView(view, fullKey()); got.Kind != MaterializationRefuse || got.Reason != "kv_key_incomplete" {
		t.Fatalf("incomplete stored key: kind=%q reason=%q, want REFUSE/kv_key_incomplete", got.Kind, got.Reason)
	}

	// A complete stored key but an incomplete WANT key also fails closed.
	full := NewKVView("", 1, src, abi.ScopeAgent, cachemeta.MatKVSpan, fullKey(), cachemeta.QualityEvidence{})
	wantPartial := fullKey()
	wantPartial.AdmitterVersion = ""
	if got := GateKVView(full, wantPartial); got.Kind != MaterializationRefuse || got.Reason != "kv_key_incomplete" {
		t.Fatalf("incomplete want key: kind=%q reason=%q, want REFUSE/kv_key_incomplete", got.Kind, got.Reason)
	}
}

// TestGateKVView_ProviderPrefixNeverLocalTrust: a remote provider prefix is
// cost/latency telemetry and can never be promoted to a local-trust HIT.
func TestGateKVView_ProviderPrefixNeverLocalTrust(t *testing.T) {
	src := kvSourceEntry()
	view := NewKVView("", 2, src, abi.ScopeFleet, cachemeta.MatProviderPrefix, fullKey(), cachemeta.QualityEvidence{})
	got := GateKVView(view, fullKey()) // identical key — would HIT if it were local
	if got.Kind != MaterializationRefuse || got.Reason != "kv_provider_prefix_not_local_trust" {
		t.Fatalf("provider_prefix: kind=%q reason=%q, want REFUSE/kv_provider_prefix_not_local_trust", got.Kind, got.Reason)
	}
}

// TestGateKVView_ApproximateNeedsQuality: a compressed/approximate span is a HIT
// only once measured quality clears its admission bound; unmeasured -> REFUSE.
func TestGateKVView_ApproximateNeedsQuality(t *testing.T) {
	src := kvSourceEntry()

	unproven := NewKVView("", 4, src, abi.ScopeAgent, cachemeta.MatCompressedKV, fullKey(), cachemeta.QualityEvidence{})
	if got := GateKVView(unproven, fullKey()); got.Kind != MaterializationRefuse || got.Reason != "kv_approx_unproven" {
		t.Fatalf("unmeasured approx: kind=%q reason=%q, want REFUSE/kv_approx_unproven", got.Kind, got.Reason)
	}
	if unproven.FaithfulnessProbe != 0.0 {
		t.Fatalf("unproven approximate view should not assert faithfulness 1.0, got %v", unproven.FaithfulnessProbe)
	}

	measured := cachemeta.QualityEvidence{Measured: true, QualityDelta: 0.01, MaxQualityDelta: 0.05}
	proven := NewKVView("", 4, src, abi.ScopeAgent, cachemeta.MatCompressedKV, fullKey(), measured)
	if got := GateKVView(proven, fullKey()); got.Kind != MaterializationHit || got.Reason != "kv_materialization_match" {
		t.Fatalf("measured approx within bound: kind=%q reason=%q, want HIT", got.Kind, got.Reason)
	}

	overBound := cachemeta.QualityEvidence{Measured: true, QualityDelta: 0.10, MaxQualityDelta: 0.05}
	bad := NewKVView("", 4, src, abi.ScopeAgent, cachemeta.MatCompressedKV, fullKey(), overBound)
	if got := GateKVView(bad, fullKey()); got.Kind != MaterializationRefuse || got.Reason != "kv_approx_unproven" {
		t.Fatalf("approx over quality bound: kind=%q reason=%q, want REFUSE", got.Kind, got.Reason)
	}
}

// TestNewKVView_StampsBindingAndProvenance: the view carries both its semantic
// provenance (source page + digest) and its model-bound runtime identity (labels).
func TestNewKVView_StampsBindingAndProvenance(t *testing.T) {
	src := kvSourceEntry()
	view := NewKVView("", 7, src, abi.ScopeAgent, cachemeta.MatKVPrefix, fullKey(), cachemeta.QualityEvidence{})

	if view.ViewType != ViewKV {
		t.Fatalf("view type = %q, want %q", view.ViewType, ViewKV)
	}
	if len(view.SourcePageIDs) != 1 || view.SourcePageIDs[0] != 7 {
		t.Fatalf("source page ids = %v, want [7]", view.SourcePageIDs)
	}
	if len(view.SourceDigests) != 1 || view.SourceDigests[0] != src.ID.Digest {
		t.Fatalf("source digests = %v, want [%s]", view.SourceDigests, src.ID.Digest)
	}
	if view.Labels["model"] != "llama-3.1-8b" || view.Labels["tokenizer"] != "llama-bpe-v3" {
		t.Fatalf("binding not stamped into labels: %v", view.Labels)
	}
	if view.Labels["materialization"] != string(cachemeta.MatKVPrefix) {
		t.Fatalf("materialization label = %q, want %q", view.Labels["materialization"], cachemeta.MatKVPrefix)
	}
	// An exact local KV form is bit-faithful.
	if view.FaithfulnessProbe != 1.0 {
		t.Fatalf("exact KV faithfulness = %v, want 1.0", view.FaithfulnessProbe)
	}
}
