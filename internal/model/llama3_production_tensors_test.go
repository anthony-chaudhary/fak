package model

import "testing"

// llama3_production_tensors_test.go — the WEIGHT-FREE "all tensors" witness for #298
// ("Production Llama 3.x Checkpoints [A-004]"), scope item 1: "Verify full checkpoint
// loading (all tensors)".
//
// The sibling llama3_production_config_test.go proves the loader derives the production
// 8B/70B CONFIG axes (geometry, GQA, llama3 RoPE, the scalar-or-list EOS) and that the 70B
// GQA geometry shards across multi-GPU rank counts. This file proves the next rung: that the
// family-aware tensor-name resolver (ResolveTensorNames, tensor_resolver.go) maps the
// COMPLETE canonical tensor set of a FULL-DEPTH production checkpoint — every one of the 8B's
// 32 layers and the 70B's 80 layers — and fails closed if any single tensor anywhere in that
// stack is absent. The existing resolver fixtures (tensor_resolver_resolve_test.go) only
// exercise single-layer (NumLayers=1) synthetic manifests; none proves the resolver stays
// complete across a real checkpoint's full layer count, which is what "all tensors" means.
//
// Weight-free by construction: the resolver inspects manifest KEYS only (presence), never
// tensor bytes, so the manifest here is just the set of canonical tensor NAMES — no 16 GB of
// 8B weights, no 140 GB of 70B weights, no GPU. Byte-exact forward parity / argmax-exactness /
// within-2x-llama.cpp throughput stay host-gated in the weight-backed oracle and GPU benchmark
// lanes (see the sibling's note); they cannot be, and are not, faked here.

// Identity-spec resolved-tensor accounting for a Llama checkpoint with no biases and an
// untied lm_head — the Llama-3.1 8B/70B shape (tie_word_embeddings=false; RMSNorm carries no
// bias). These mirror identitySpec + baseGlobals in tensor_resolver.go:
//
//	globals : model.embed_tokens.weight, model.norm.weight, lm_head.weight             => 3
//	perLayer: input_layernorm + {q,k,v,o}_proj + post_attention_layernorm + gate/up/down => 9
const (
	llamaResolvedGlobals  = 3
	llamaResolvedPerLayer = 9
)

// fullLlamaManifestNames lists every canonical tensor a full-depth Llama checkpoint stores
// under HF-standard (identity) naming: the three globals plus the nine core-forward tensors
// for each of numLayers layers. It is the presence-only analog of a real checkpoint's
// manifest.json enumerated over all tensors.
func fullLlamaManifestNames(numLayers int) []string {
	names := []string{
		"model.embed_tokens.weight",
		"model.norm.weight",
		"lm_head.weight", // untied for Llama-3.1 (tie_word_embeddings=false)
	}
	for l := 0; l < numLayers; l++ {
		p := layerPrefix(l)
		names = append(names,
			p+"input_layernorm.weight",
			p+"self_attn.q_proj.weight",
			p+"self_attn.k_proj.weight",
			p+"self_attn.v_proj.weight",
			p+"self_attn.o_proj.weight",
			p+"post_attention_layernorm.weight",
			p+"mlp.gate_proj.weight",
			p+"mlp.up_proj.weight",
			p+"mlp.down_proj.weight",
		)
	}
	return names
}

// assertFullCheckpointResolves runs the resolver over a full-depth production manifest and
// asserts (a) every required canonical tensor resolves to ITSELF (identity, zero renames),
// (b) the resolved count equals the exact all-tensors accounting for numLayers — proving no
// layer's tensors were silently dropped, and (c) a tensor deleted from the DEEPEST layer is
// caught with a precise error, proving the resolver scans the whole stack, not just layer 0.
func assertFullCheckpointResolves(t *testing.T, cfg Config, numLayers int) {
	t.Helper()
	if cfg.NumLayers != numLayers {
		t.Fatalf("config NumLayers = %d, want production depth %d", cfg.NumLayers, numLayers)
	}
	man := manifestKeys(fullLlamaManifestNames(numLayers)...)
	res, err := ResolveTensorNames(cfg, man)
	if err != nil {
		t.Fatalf("full %d-layer production checkpoint must resolve every tensor: %v", numLayers, err)
	}
	if res.Family != "llama" {
		t.Fatalf("family = %q, want llama", res.Family)
	}
	// Every required canonical tensor resolves to itself — the Llama load path is unchanged at
	// the name level (zero renames) across ALL layers, not just the shallow ones.
	for canonical, src := range res.Resolved {
		if src != canonical {
			t.Errorf("identity broken at production scale: %q resolved to %q", canonical, src)
		}
	}
	// "all tensors": the resolved set is exactly the full-checkpoint accounting, so none of the
	// numLayers layers' tensors went silently unresolved.
	want := llamaResolvedGlobals + llamaResolvedPerLayer*numLayers
	if len(res.Resolved) != want {
		t.Fatalf("resolved %d tensors for %d layers, want %d (%d globals + %d/layer)",
			len(res.Resolved), numLayers, want, llamaResolvedGlobals, llamaResolvedPerLayer)
	}
	// Spot-check the DEEPEST layer's full block actually resolved — proves completeness reaches
	// layer numLayers-1, not just the first few layers.
	deep := layerPrefix(numLayers - 1)
	for _, name := range []string{
		deep + "input_layernorm.weight",
		deep + "self_attn.q_proj.weight",
		deep + "self_attn.o_proj.weight",
		deep + "post_attention_layernorm.weight",
		deep + "mlp.down_proj.weight",
	} {
		if res.SourceFor(name) != name {
			t.Errorf("deepest layer tensor %q did not resolve (got %q)", name, res.SourceFor(name))
		}
	}
	// Fail closed: drop the deepest layer's down_proj and the resolver names the exact missing
	// tensor, the family, and the searched candidates — a checkpoint missing a tensor ANYWHERE
	// in the full stack is refused, never silently mis-loaded.
	missing := deep + "mlp.down_proj.weight"
	delete(man, missing)
	assertResolveError(t, cfg, man, "llama family", missing, "searched:")
}

// TestLlama31_8BInstructResolvesAllTensors proves the resolver maps the COMPLETE canonical
// tensor set of a full 32-layer Llama-3.1-8B-Instruct checkpoint (all 291 core-forward
// tensors: 3 globals + 9*32) and fails closed on a missing tensor — the structural "all
// tensors" rung of "Load Llama 3.1 8B Instruct" (#298), at production depth, weight-free.
func TestLlama31_8BInstructResolvesAllTensors(t *testing.T) {
	cfg := decodeProdConfig(t, `{
		"architectures": ["LlamaForCausalLM"],
		"model_type": "llama",
		"hidden_size": 4096,
		"num_hidden_layers": 32,
		"num_attention_heads": 32,
		"num_key_value_heads": 8,
		"intermediate_size": 14336,
		"vocab_size": 128256
	}`)
	assertFullCheckpointResolves(t, cfg, 32)
}

// TestLlama31_70BResolvesAllTensors proves the same for a full 80-layer Llama-3.1-70B
// checkpoint (all 723 core-forward tensors: 3 globals + 9*80) — the "all tensors" rung of
// "Load Llama 3.1 70B" (#298). Together with the sibling's NewTPPlan shard proof, this is the
// GPU-free half of loading the 70B: the loader knows every tensor it must find, and how those
// tensors' 8 KV-head groups shard across devices.
func TestLlama31_70BResolvesAllTensors(t *testing.T) {
	cfg := decodeProdConfig(t, `{
		"architectures": ["LlamaForCausalLM"],
		"model_type": "llama",
		"hidden_size": 8192,
		"num_hidden_layers": 80,
		"num_attention_heads": 64,
		"num_key_value_heads": 8,
		"intermediate_size": 28672,
		"vocab_size": 128256
	}`)
	assertFullCheckpointResolves(t, cfg, 80)
}
