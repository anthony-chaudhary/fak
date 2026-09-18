package model

import "testing"

// v41_expert_placement_seam_test.go — #13271 residency/placement predicates for the
// native non-MLA DeepSeek-V4.1 routed experts, whose canonical names are
// model.layers.<L>.ffn.experts.<e>.{w1,w3,w2}.weight (ffn, not mlp).

// TestV41RoutedExpertTensorClassification pins isRoutedExpertTensor (the
// MoEResidentWeightBytes / EP-fit / spill-budget partition predicate): a V4.1
// routed expert must count as the SHARDABLE expert term, not the replicated
// remainder, or a 753B-class V4.1 MoE is mis-sized as fully replicated.
func TestV41RoutedExpertTensorClassification(t *testing.T) {
	routed := []string{
		"model.layers.0.ffn.experts.0.w1.weight",
		"model.layers.3.ffn.experts.7.w3.weight",
		"model.layers.1.ffn.experts.2.w2.weight",
		// unchanged GLM spelling stays routed.
		"model.layers.0.mlp.experts.0.gate_proj.weight",
	}
	for _, n := range routed {
		if !isRoutedExpertTensor(n) {
			t.Errorf("isRoutedExpertTensor(%q)=false, want true (must partition as the shardable expert term)", n)
		}
	}
	replicated := []string{
		"model.layers.0.ffn.shared_experts.w1.weight", // always-on shared expert is replicated
		"model.layers.0.ffn.gate.weight",
		"model.layers.0.attn.wq_a.weight",
		"model.embed_tokens.weight",
	}
	for _, n := range replicated {
		if isRoutedExpertTensor(n) {
			t.Errorf("isRoutedExpertTensor(%q)=true, want false (belongs to the replicated remainder)", n)
		}
	}
}

// TestV41RoutedExpertIdentityParses pins routedExpertIdentity on the V4.1 spelling:
// the expert-ring pin/usage path keys on the (layer, expert) pair, so a V4.1 name
// that fails to parse silently loses pinning/usage for every routed expert.
func TestV41RoutedExpertIdentityParses(t *testing.T) {
	cases := []struct {
		name       string
		layer, exp int
	}{
		{"model.layers.0.ffn.experts.0.w1.weight", 0, 0},
		{"model.layers.7.ffn.experts.129.w3.weight", 7, 129},
		{"model.layers.39.ffn.experts.255.w2.weight", 39, 255},
	}
	for _, c := range cases {
		l, e, ok := routedExpertIdentity(c.name)
		if !ok {
			t.Errorf("routedExpertIdentity(%q) ok=false, want (%d,%d)", c.name, c.layer, c.exp)
			continue
		}
		if l != c.layer || e != c.exp {
			t.Errorf("routedExpertIdentity(%q) = (%d,%d), want (%d,%d)", c.name, l, e, c.layer, c.exp)
		}
	}
	// The shared V4.1 expert must NOT be routed-identified (disjointness).
	if _, _, ok := routedExpertIdentity("model.layers.0.ffn.shared_experts.w1.weight"); ok {
		t.Error("routedExpertIdentity(ffn.shared_experts.w1.weight) ok=true — shared must not parse as routed")
	}
}
