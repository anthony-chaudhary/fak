package ggufload

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

// Kimi-K3.1 ("kimi-k31") GGUF tensor map, required-family census, and map-level
// routed-expert shard census (ticket 1611-kimi-k31 leaf L2).
//
// Synthetic-fixture witnesses only: no network, no real weights. The fixtures
// ride a REDUCED geometry (4 layers [kda,kda,kda,dsa], 8 experts, tiny dims)
// whose family STRUCTURE mirrors the measured K3.1-Pro inventory (L0 artifact
// census): every arch-defining tensor family appears, in the converter's own
// spellings, so the map, the fail-closed census, and the multi-shard merge
// seam are exercised end-to-end at the map level. The one full-scale witness
// (TestKimiK31ExpertBandCensus96ShardRouting) builds the real 96-shard
// official-checkpoint organization in memory: 896 experts x 92 MoE layers x 6
// packed/scale components = 494,592 routed-expert occurrences, names only.

const (
	k31TestLayers  = 4 // [kda, kda, kda, dsa]; layer 0 is the dense-FFN layer
	k31TestExperts = 8
)

// kimiK31Pair pins one GGUF spelling to the canonical HF-suffixed name the
// native K3.1 forward will read.
type kimiK31Pair struct{ gguf, want string }

// synthKimiK31ReducedMeta is the minimal K3.1-shaped metadata for the reduced
// 4-layer fixture, keyed on the file's own "kimi-k31." raw prefix (the L1
// contract: recognition normalizes the arch, metadata reads stay raw).
func synthKimiK31ReducedMeta() map[string]Value {
	const arch = "kimi-k31"
	p := arch + "."
	meta := map[string]Value{
		"general.architecture":                 {Type: TypeString, Value: arch},
		p + "context_length":                   {Type: TypeUint64, Value: uint64(4096)},
		p + "embedding_length":                 {Type: TypeUint64, Value: uint64(256)},
		p + "block_count":                      {Type: TypeUint64, Value: uint64(k31TestLayers)},
		p + "feed_forward_length":              {Type: TypeUint64, Value: uint64(128)},
		p + "attention.head_count":             {Type: TypeUint64, Value: uint64(4)},
		p + "attention.layer_norm_rms_epsilon": {Type: TypeFloat32, Value: float32(1e-5)},
		p + "rope.dimension_count":             {Type: TypeUint64, Value: uint64(8)},
		// Per-layer KDA/DSA schedule: 0 marks KDA, positive marks DSA.
		p + "attention.head_count_kv": Value{Type: TypeArray, Value: []Value{
			{Type: TypeUint32, Value: uint32(0)},
			{Type: TypeUint32, Value: uint32(0)},
			{Type: TypeUint32, Value: uint32(0)},
			{Type: TypeUint32, Value: uint32(1)},
		}},
		// MoE axis: 8 routed / 2 active / 2 shared, expert FFN 64, latent 96.
		p + "expert_count":               {Type: TypeUint64, Value: uint64(k31TestExperts)},
		p + "expert_used_count":          {Type: TypeUint64, Value: uint64(2)},
		p + "expert_feed_forward_length": {Type: TypeUint64, Value: uint64(64)},
		p + "expert_latent_length":       {Type: TypeUint64, Value: uint64(96)},
		p + "expert_shared_count":        {Type: TypeUint64, Value: uint64(2)},
		p + "leading_dense_block_count":  {Type: TypeUint64, Value: uint64(1)},
		// MLA latent-projection ranks + per-head widths (qk = nope 16 + rope 8).
		p + "attention.q_lora_rank":      {Type: TypeUint64, Value: uint64(64)},
		p + "attention.kv_lora_rank":     {Type: TypeUint64, Value: uint64(32)},
		p + "attention.key_length_mla":   {Type: TypeUint64, Value: uint64(24)},
		p + "attention.value_length_mla": {Type: TypeUint64, Value: uint64(16)},
		// Gated DSA learned-indexer axis.
		p + "attention.indexer.head_count": {Type: TypeUint64, Value: uint64(2)},
		p + "attention.indexer.key_length": {Type: TypeUint64, Value: uint64(16)},
		p + "attention.indexer.top_k":      {Type: TypeUint64, Value: uint64(8)},
		// KDA axis.
		p + "ssm.conv_kernel": {Type: TypeUint64, Value: uint64(4)},
		p + "kda.head_dim":    {Type: TypeUint64, Value: uint64(32)},
	}
	// Leaf L3 pin: an admitted kimi-k31 header must carry the K3-inherited
	// embedded BPE table the shared GGMLTokenizer -> tokenizer.FromGGML path
	// rides; the reduced complete-file fixture carries it too.
	for k, v := range kimiK31SynthTokenizerTable(true) {
		meta[k] = v
	}
	return meta
}

// k31LayerPrefix / k31CanonPrefix compose the per-layer name prefixes.
func k31LayerPrefix(l int) string { return "blk." + strconv.Itoa(l) + "." }
func k31CanonPrefix(l int) string { return "model.layers." + strconv.Itoa(l) + "." }

// k31CommonLayerPairs are the families EVERY decoder layer carries.
func k31CommonLayerPairs(l int) []kimiK31Pair {
	p, m := k31LayerPrefix(l), k31CanonPrefix(l)
	return []kimiK31Pair{
		{p + "attn_norm.weight", m + "input_layernorm.weight"},
		{p + "ffn_norm.weight", m + "post_attention_layernorm.weight"},
		{p + "attn_res_score", m + "self_attention_res_proj"},
		{p + "ffn_res_score", m + "mlp_res_proj"},
	}
}

// k31KDAPairs are the KDA (Kimi Delta Attention) linear-attention families.
func k31KDAPairs(l int) []kimiK31Pair {
	p, m := k31LayerPrefix(l), k31CanonPrefix(l)
	return []kimiK31Pair{
		{p + "ssm_g.weight", m + "self_attn.g_proj.weight"},
		{p + "ssm_conv1d_q.weight", m + "self_attn.q_conv1d.weight"},
		{p + "ssm_conv1d_k.weight", m + "self_attn.k_conv1d.weight"},
		{p + "ssm_conv1d_v.weight", m + "self_attn.v_conv1d.weight"},
		{p + "ssm_f_a.weight", m + "self_attn.f_a_proj.weight"},
		{p + "ssm_f_b.weight", m + "self_attn.f_b_proj.weight"},
		{p + "ssm_beta.weight", m + "self_attn.b_proj.weight"},
		{p + "ssm_a", m + "self_attn.A_log"},
		{p + "ssm_dt.bias", m + "self_attn.dt_bias"},
		{p + "ssm_norm.weight", m + "self_attn.o_norm"},
		{p + "attn_q.weight", m + "self_attn.q_proj.weight"},
		{p + "attn_k.weight", m + "self_attn.k_proj.weight"},
		{p + "attn_v.weight", m + "self_attn.v_proj.weight"},
		{p + "attn_output.weight", m + "self_attn.o_proj.weight"},
	}
}

// k31DSAPairs are the Gated-DSA (sparse MLA + learned indexer) families.
func k31DSAPairs(l int) []kimiK31Pair {
	p, m := k31LayerPrefix(l), k31CanonPrefix(l)
	return []kimiK31Pair{
		{p + "attn_q_a.weight", m + "self_attn.q_a_proj.weight"},
		{p + "attn_q_a_norm.weight", m + "self_attn.q_a_layernorm.weight"},
		{p + "attn_q_b.weight", m + "self_attn.q_b_proj.weight"},
		{p + "attn_kv_a_mqa.weight", m + "self_attn.kv_a_proj_with_mqa.weight"},
		{p + "attn_kv_a_norm.weight", m + "self_attn.kv_a_layernorm.weight"},
		{p + "attn_kv_b.weight", m + "self_attn.kv_b_proj.weight"},
		{p + "attn_gate.weight", m + "self_attn.o_gate.weight"},
		{p + "attn_output.weight", m + "self_attn.o_proj.weight"},
		{p + "indexer.attn_q_b.weight", m + "self_attn.indexer.wq_b.weight"},
		{p + "indexer.attn_k.weight", m + "self_attn.indexer.wk.weight"},
		{p + "indexer.proj.weight", m + "self_attn.indexer.weights_proj.weight"},
		{p + "indexer.k_norm.weight", m + "self_attn.indexer.k_norm.weight"},
		{p + "indexer.k_norm.bias", m + "self_attn.indexer.k_norm.bias"},
	}
}

// k31MoePairs are the latent-MoE families the router, routed projections, and
// shared experts carry on every non-dense layer. The batched routed slabs
// (ffn_*_exps) are deliberately EXCLUDED: they are unmapped by design, owned
// by the 1->E splitter (see TestKimiK31TensorMapResolvesCompleteFixture).
func k31MoePairs(l int) []kimiK31Pair {
	p, m := k31LayerPrefix(l), k31CanonPrefix(l)
	return []kimiK31Pair{
		{p + "ffn_gate_inp.weight", m + "block_sparse_moe.gate.weight"},
		{p + "exp_probs_b.bias", m + "block_sparse_moe.gate.e_score_correction_bias"},
		{p + "ffn_routed_down.weight", m + "block_sparse_moe.routed_expert_down_proj.weight"},
		{p + "ffn_routed_up.weight", m + "block_sparse_moe.routed_expert_up_proj.weight"},
		{p + "ffn_routed_norm.weight", m + "block_sparse_moe.routed_expert_norm.weight"},
		{p + "ffn_gate_shexp.weight", m + "block_sparse_moe.shared_experts.gate_proj.weight"},
		{p + "ffn_up_shexp.weight", m + "block_sparse_moe.shared_experts.up_proj.weight"},
		{p + "ffn_down_shexp.weight", m + "block_sparse_moe.shared_experts.down_proj.weight"},
	}
}

// k31DensePairs are the plain dense-FFN families of the leading dense layer.
func k31DensePairs(l int) []kimiK31Pair {
	p, m := k31LayerPrefix(l), k31CanonPrefix(l)
	return []kimiK31Pair{
		{p + "ffn_gate.weight", m + "mlp.gate_proj.weight"},
		{p + "ffn_up.weight", m + "mlp.up_proj.weight"},
		{p + "ffn_down.weight", m + "mlp.down_proj.weight"},
	}
}

// k31GlobalPairs are the model-global families.
var k31GlobalPairs = []kimiK31Pair{
	{"token_embd.weight", "model.embed_tokens.weight"},
	{"output_norm.weight", "model.norm.weight"},
	{"output.weight", "lm_head.weight"},
	{"output_res_score", "model.output_attn_res_proj"},
}

// k31SlabNames are the three batched routed-expert slabs of one MoE layer.
func k31SlabNames(l int) []string {
	return []string{
		k31LayerPrefix(l) + "ffn_gate_exps.weight",
		k31LayerPrefix(l) + "ffn_down_exps.weight",
		k31LayerPrefix(l) + "ffn_up_exps.weight",
	}
}

// synthKimiK31ReducedPairs is the complete pinned spelling table for the
// reduced fixture: globals, then per layer [norms/res-score, attention, FFN].
func synthKimiK31ReducedPairs() []kimiK31Pair {
	var pairs []kimiK31Pair
	pairs = append(pairs, k31GlobalPairs...)
	for l := 0; l < k31TestLayers; l++ {
		pairs = append(pairs, k31CommonLayerPairs(l)...)
		if l == k31TestLayers-1 {
			pairs = append(pairs, k31DSAPairs(l)...)
		} else {
			pairs = append(pairs, k31KDAPairs(l)...)
		}
		if l < 1 { // FirstKDenseReplace = 1: layer 0 is dense
			pairs = append(pairs, k31DensePairs(l)...)
		} else {
			pairs = append(pairs, k31MoePairs(l)...)
		}
	}
	return pairs
}

// synthKimiK31ReducedTensors is the synthetic tensor DIRECTORY for the reduced
// fixture: every mapped pair plus the three batched slabs per MoE layer. Dims
// are tiny on purpose (map-level witness; payload shapes are out of scope here)
// except the slabs, whose trailing dim carries the declared expert count the
// census reads.
func synthKimiK31ReducedTensors() []TensorInfo {
	tensors := make([]TensorInfo, 0, len(synthKimiK31ReducedPairs())+3*(k31TestLayers-1))
	for _, p := range synthKimiK31ReducedPairs() {
		tensors = append(tensors, TensorInfo{Name: p.gguf, Dims: []uint64{4}, Type: TensorF32})
	}
	for l := 1; l < k31TestLayers; l++ {
		for _, n := range k31SlabNames(l) {
			tensors = append(tensors, TensorInfo{Name: n, Dims: []uint64{4, 4, uint64(k31TestExperts)}, Type: TensorF32})
		}
	}
	return tensors
}

// k31DropTensors returns a copy of tensors without the named entries.
func k31DropTensors(tensors []TensorInfo, drop ...string) []TensorInfo {
	skip := make(map[string]bool, len(drop))
	for _, d := range drop {
		skip[d] = true
	}
	out := make([]TensorInfo, 0, len(tensors)-len(drop))
	for _, t := range tensors {
		if !skip[t.Name] {
			out = append(out, t)
		}
	}
	return out
}

// k31AddTensors appends extra entries to a copy of tensors.
func k31AddTensors(tensors []TensorInfo, extra ...TensorInfo) []TensorInfo {
	out := make([]TensorInfo, 0, len(tensors)+len(extra))
	out = append(out, tensors...)
	out = append(out, extra...)
	return out
}

// k31ConfigErr runs Config() (and thus the L2 tail census) on a synthetic
// reduced-geometry file carrying the given tensor directory.
func k31ConfigErr(tensors []TensorInfo) error {
	_, err := (&File{Metadata: synthKimiK31ReducedMeta(), Tensors: tensors}).Config()
	return err
}

// k31AssertFamilyRefusal asserts err is the typed sentinel wrapping a refusal
// that names BOTH the offending family and the first missing tensor.
func k31AssertFamilyRefusal(t *testing.T, err error, family, firstMissing string) {
	t.Helper()
	if err == nil {
		t.Fatalf("Config succeeded, want the typed ErrKimiK31MissingTensorFamily refusal")
	}
	if !errors.Is(err, ErrKimiK31MissingTensorFamily) {
		t.Fatalf("refusal %v is not wrapped in ErrKimiK31MissingTensorFamily", err)
	}
	msg := err.Error()
	if !strings.Contains(msg, fmt.Sprintf("family %q", family)) {
		t.Errorf("refusal %q does not name family %q", msg, family)
	}
	if !strings.Contains(msg, firstMissing) {
		t.Errorf("refusal %q does not name the first missing tensor %q", msg, firstMissing)
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestKimiK31TensorMapResolvesCompleteFixture(t *testing.T) {
	pairs := synthKimiK31ReducedPairs()
	f := &File{Metadata: synthKimiK31ReducedMeta(), Tensors: synthKimiK31ReducedTensors()}

	// (a-pre) The complete directory admits: the L2 tail census finds every
	// required family, so Config() succeeds on the merged-shape directory.
	cfg, err := f.Config()
	if err != nil {
		t.Fatalf("Config on the complete reduced fixture: %v, want clean admission", err)
	}
	if cfg.ModelType != "kimik31" || cfg.NumLayers != k31TestLayers || cfg.NumExperts != k31TestExperts {
		t.Fatalf("derived axes = type:%q layers:%d experts:%d, want kimik31/%d/%d",
			cfg.ModelType, cfg.NumLayers, cfg.NumExperts, k31TestLayers, k31TestExperts)
	}
	if cfg.FirstKDenseReplace != 1 || cfg.IndexTopK != 8 || cfg.QLoraRank != 64 || cfg.KVLoraRank != 32 {
		t.Fatalf("derived axes = dense:%d topk:%d qlora:%d kvlora:%d, want 1/8/64/32",
			cfg.FirstKDenseReplace, cfg.IndexTopK, cfg.QLoraRank, cfg.KVLoraRank)
	}
	wantTypes := []string{kimiK31LayerKDA, kimiK31LayerKDA, kimiK31LayerKDA, kimiK31LayerDSA}
	for l, want := range wantTypes {
		if cfg.LayerTypes[l] != want {
			t.Fatalf("LayerTypes[%d] = %q, want %q", l, cfg.LayerTypes[l], want)
		}
	}

	// (a) Every pinned GGUF spelling resolves under kimik31 to its canonical
	// leaf — the full tensor map, walked in one pass.
	for _, p := range pairs {
		got, ok := CanonicalTensorNameArch(p.gguf, "kimik31")
		if !ok {
			t.Errorf("%s: no canonical mapping under kimik31, want %s", p.gguf, p.want)
			continue
		}
		if got != p.want {
			t.Errorf("%s = %s, want %s", p.gguf, got, p.want)
		}
	}

	// The batched routed slabs stay UNMAPPED by design: one GGUF name cannot
	// become E per-expert names, so the 1->E splitter owns them before this
	// map is consulted.
	for l := 1; l < k31TestLayers; l++ {
		for _, n := range k31SlabNames(l) {
			if _, ok := CanonicalTensorNameArch(n, "kimik31"); ok {
				t.Errorf("%s resolved through the 1:1 map; the batched slabs must stay splitter-owned", n)
			}
		}
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestKimiK31PerExpertMXFP4DialectMaps(t *testing.T) {
	// The official-checkpoint per-expert dialect (packed/scale component pairs)
	// resolves onto the same block_sparse_moe.experts.<e>.wX leaves the batched
	// slab splitter emits, keeping both dialects on one canonical leaf set.
	cases := []kimiK31Pair{
		{"blk.3.block_sparse_moe.experts.0.w1.weight_packed", "model.layers.3.block_sparse_moe.experts.0.w1.weight_packed"},
		{"blk.3.block_sparse_moe.experts.0.w2.weight_scale", "model.layers.3.block_sparse_moe.experts.0.w2.weight_scale"},
		{"blk.3.block_sparse_moe.experts.7.w3.weight_packed", "model.layers.3.block_sparse_moe.experts.7.w3.weight_packed"},
		{"blk.0.block_sparse_moe.experts.895.w1.weight_scale", "model.layers.0.block_sparse_moe.experts.895.w1.weight_scale"},
	}
	for _, p := range cases {
		got, ok := CanonicalTensorNameArch(p.gguf, "kimik31")
		if !ok {
			t.Errorf("%s: no canonical mapping under kimik31, want %s", p.gguf, p.want)
			continue
		}
		if got != p.want {
			t.Errorf("%s = %s, want %s", p.gguf, got, p.want)
		}
	}
	// Non-expert members under the same prefix stay unmapped (no over-match):
	// unknown projection, unknown component, non-numeric expert id.
	for _, bad := range []string{
		"blk.3.block_sparse_moe.experts.0.w4.weight_packed",
		"blk.3.block_sparse_moe.experts.0.w1.weight",
		"blk.3.block_sparse_moe.experts.x.w1.weight_packed",
		"blk.3.block_sparse_moe.experts.-1.w1.weight_packed",
	} {
		if _, ok := CanonicalTensorNameArch(bad, "kimik31"); ok {
			t.Errorf("%s resolved; the per-expert dialect must match exactly", bad)
		}
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestKimiK31MissingIndexerFamilyRefused(t *testing.T) {
	all := synthKimiK31ReducedTensors()

	// (b-i) The first load-bearing indexer member missing: the refusal names
	// the dsa-indexer family AND the exact tensor.
	err := k31ConfigErr(k31DropTensors(all, "blk.3.indexer.attn_q_b.weight"))
	k31AssertFamilyRefusal(t, err, "dsa-indexer", "blk.3.indexer.attn_q_b.weight")

	// (b-ii) Only the k_norm weight missing: still dsa-indexer, naming k_norm
	// (stream order within the family decides the first-missing report).
	err = k31ConfigErr(k31DropTensors(all, "blk.3.indexer.k_norm.weight"))
	k31AssertFamilyRefusal(t, err, "dsa-indexer", "blk.3.indexer.k_norm.weight")

	// Optional companions never trip the census: dropping the optional
	// k_norm.bias (and the optional attn_q_a_norm / attn_gate) still admits.
	err = k31ConfigErr(k31DropTensors(all,
		"blk.3.indexer.k_norm.bias", "blk.3.attn_q_a_norm.weight", "blk.3.attn_gate.weight"))
	if err != nil {
		t.Fatalf("dropping optional indexer/MLA members must admit, got: %v", err)
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestKimiK31MissingExpertBandRefused(t *testing.T) {
	all := synthKimiK31ReducedTensors()

	// (c-i) All three batched slabs stripped from a MoE layer: routed-experts
	// family, naming the first slab in stream order.
	err := k31ConfigErr(k31DropTensors(all, k31SlabNames(1)...))
	k31AssertFamilyRefusal(t, err, "routed-experts", "blk.1.ffn_gate_exps.weight")

	// (c-ii) A PARTIAL per-expert band (4 of the 24 official-dialect entries
	// for layer 2, slabs removed): still refused, naming the first expected
	// per-expert entry — a stripped artifact is not admitted halfway.
	base := k31DropTensors(all, k31SlabNames(2)...)
	var extra []TensorInfo
	for e := 0; e < 2; e++ {
		extra = append(extra,
			TensorInfo{Name: fmt.Sprintf("blk.2.block_sparse_moe.experts.%d.w1.weight_packed", e), Dims: []uint64{4}, Type: TensorF32},
			TensorInfo{Name: fmt.Sprintf("blk.2.block_sparse_moe.experts.%d.w1.weight_scale", e), Dims: []uint64{4}, Type: TensorF32},
		)
	}
	err = k31ConfigErr(k31AddTensors(base, extra...))
	k31AssertFamilyRefusal(t, err, "routed-experts", "blk.2.block_sparse_moe.experts.0.w1.weight_packed")
}

// fak-test:runtime fast est=100ms lane=default
func TestKimiK31KVBGroupRefusedWhenUnpaired(t *testing.T) {
	all := synthKimiK31ReducedTensors()

	// The DSA layer's KV-b up-projection is required as a GROUP: combined
	// attn_kv_b OR the split attn_k_b + attn_v_b pair. Neither form present ->
	// dsa-attention refusal naming the group.
	err := k31ConfigErr(k31DropTensors(all, "blk.3.attn_kv_b.weight"))
	k31AssertFamilyRefusal(t, err, "dsa-attention", "blk.3.attn_kv_b.weight")

	// Only ONE split half present: still an unpaired group -> refused.
	half := k31AddTensors(k31DropTensors(all, "blk.3.attn_kv_b.weight"),
		TensorInfo{Name: "blk.3.attn_k_b.weight", Dims: []uint64{4}, Type: TensorF32})
	err = k31ConfigErr(half)
	k31AssertFamilyRefusal(t, err, "dsa-attention", "blk.3.attn_kv_b.weight")

	// BOTH split halves present instead of the combined tensor: admitted —
	// the converter emits exactly this pre-split dialect.
	both := k31AddTensors(k31DropTensors(all, "blk.3.attn_kv_b.weight"),
		TensorInfo{Name: "blk.3.attn_k_b.weight", Dims: []uint64{4}, Type: TensorF32},
		TensorInfo{Name: "blk.3.attn_v_b.weight", Dims: []uint64{4}, Type: TensorF32})
	if err := k31ConfigErr(both); err != nil {
		t.Fatalf("the split KV-b half pair must admit, got: %v", err)
	}
}

// kimiK31TestShardOf routes a per-expert MXFP4 entry name onto one of `shards`
// shards by the same contiguous-block rule the census test asserts analytically:
// shard = (layer*(experts*6) + expert*6 + proj*2 + comp) % shards, where the
// component index is 0 for weight_packed, 1 for weight_scale.
func kimiK31TestShardOf(name string, experts, shards int) int {
	rest := strings.TrimPrefix(name, "blk.")
	dot := strings.IndexByte(rest, '.')
	l, _ := strconv.Atoi(rest[:dot])
	rest = rest[dot+1:]
	i := strings.Index(rest, ".experts.")
	rest = rest[i+len(".experts."):]
	dot = strings.IndexByte(rest, '.')
	e, _ := strconv.Atoi(rest[:dot])
	rest = rest[dot+1:] // wX.component
	dot = strings.IndexByte(rest, '.')
	wIdx := 2
	switch rest[:dot] {
	case "w1":
		wIdx = 0
	case "w2":
		wIdx = 1
	}
	cIdx := 0
	if rest[dot+1:] == "weight_scale" {
		cIdx = 1
	}
	return (l*(experts*6) + e*6 + wIdx*2 + cIdx) % shards
}

// fak-test:runtime medium est=1s lane=default
func TestKimiK31ExpertBandCensus96ShardRouting(t *testing.T) {
	// The real official-checkpoint organization at the map level: 93 layers
	// (layer 0 dense, no band), 896 experts x 92 MoE layers x 6 packed/scale
	// components = 494,592 routed-expert occurrences, names only.
	const layers, experts, shards = 93, 896, 96
	tensors := make([]TensorInfo, 0, (layers-1)*experts*6)
	for l := 1; l < layers; l++ {
		for e := 0; e < experts; e++ {
			for _, w := range []string{"w1", "w2", "w3"} {
				for _, c := range []string{"weight_packed", "weight_scale"} {
					tensors = append(tensors, TensorInfo{
						Name: fmt.Sprintf("blk.%d.block_sparse_moe.experts.%d.%s.%s", l, e, w, c),
						Dims: []uint64{4},
						Type: TensorF32,
					})
				}
			}
		}
	}
	f := &File{Tensors: tensors}
	c, err := CensusKimiK31ExpertBand(f, layers, shards, func(name string) int {
		return kimiK31TestShardOf(name, experts, shards)
	})
	if err != nil {
		t.Fatalf("CensusKimiK31ExpertBand: %v", err)
	}

	// Total occurrences: 896 x 92 x 6 = 494,592.
	const wantTotal = experts * (layers - 1) * 6
	if c.Occurrences != wantTotal {
		t.Fatalf("Occurrences = %d, want %d (896 experts x 92 MoE layers x 6 components)", c.Occurrences, wantTotal)
	}
	if c.Layers != layers-1 {
		t.Errorf("Layers = %d, want %d (the dense layer 0 carries no band)", c.Layers, layers-1)
	}
	if c.Experts != 0 {
		t.Errorf("Experts = %d, want 0 (the per-expert dialect declares no slab count)", c.Experts)
	}
	for l := 1; l < layers; l++ {
		if c.PerLayer[l] != experts*6 {
			t.Errorf("PerLayer[%d] = %d, want %d", l, c.PerLayer[l], experts*6)
			break
		}
	}
	if c.PerLayer[0] != 0 {
		t.Errorf("PerLayer[0] = %d, want 0 (dense layer)", c.PerLayer[0])
	}
	// Per-shard routing: experts*6 = 5376 = 96*56 divides evenly, so every
	// layer contributes EXACTLY 56 occurrences to every shard — assert the
	// analytic value rather than re-simulating the routing.
	const wantPerShard = (layers - 1) * 56
	sum := 0
	for s := 0; s < shards; s++ {
		if c.PerShard[s] != wantPerShard {
			t.Errorf("PerShard[%d] = %d, want %d", s, c.PerShard[s], wantPerShard)
		}
		sum += c.PerShard[s]
	}
	if sum != wantTotal {
		t.Errorf("sum(PerShard) = %d, want %d (every occurrence routes to exactly one shard)", sum, wantTotal)
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestKimiK31ExpertBandCensusGuardRails(t *testing.T) {
	f := &File{Tensors: []TensorInfo{
		{Name: "blk.1.block_sparse_moe.experts.0.w1.weight_packed", Dims: []uint64{4}, Type: TensorF32},
	}}
	shardOf := func(string) int { return 0 }

	if _, err := CensusKimiK31ExpertBand(f, 0, 1, shardOf); err == nil {
		t.Error("numLayers=0 accepted, want refusal")
	}
	if _, err := CensusKimiK31ExpertBand(f, 4, 0, shardOf); err == nil {
		t.Error("shardCount=0 accepted, want refusal (read split.count from the config shard)")
	}
	// An expert-band member naming a layer outside block_count is refused.
	if _, err := CensusKimiK31ExpertBand(f, 1, 1, shardOf); err == nil || !strings.Contains(err.Error(), "outside block_count") {
		t.Errorf("out-of-range layer census err = %v, want 'outside block_count'", err)
	}
	// A shardOf routing outside the shard set is refused.
	if _, err := CensusKimiK31ExpertBand(f, 4, 1, func(string) int { return 1 }); err == nil || !strings.Contains(err.Error(), "routes to shard") {
		t.Errorf("out-of-range shard census err = %v, want 'routes to shard'", err)
	}
	// Duplicate tensor names across shards are refused (the merge seam already
	// refuses them; the census stays sound on hand-built directories too).
	dup := &File{Tensors: []TensorInfo{
		{Name: "blk.1.block_sparse_moe.experts.0.w1.weight_packed", Dims: []uint64{4}, Type: TensorF32},
		{Name: "blk.1.block_sparse_moe.experts.0.w1.weight_packed", Dims: []uint64{4}, Type: TensorF32},
	}}
	if _, err := CensusKimiK31ExpertBand(dup, 4, 1, shardOf); err == nil || !strings.Contains(err.Error(), "duplicate") {
		t.Errorf("duplicate-name census err = %v, want 'duplicate'", err)
	}
}

// k31PatternF32 writes n little-endian f32 values 0..n-1 — a deterministic
// payload pattern for the synthetic split shards.
func k31PatternF32(n int) []byte {
	b := make([]byte, 0, n*4)
	var tmp [4]byte
	for i := 0; i < n; i++ {
		binary.LittleEndian.PutUint32(tmp[:], math.Float32bits(float32(i)))
		b = append(b, tmp[:]...)
	}
	return b
}

// writeKimiK31SplitShard builds one synthetic shard of a Kimi-K3.1 split,
// mirroring writeDeepSeek41SplitShard but declaring kimi-k31 (with the reduced
// axes Config() needs, including the per-layer head_count_kv array). Only the
// config-carrying shard passes config=true.
func writeKimiK31SplitShard(t *testing.T, shardNo, count uint32, config bool, tensors []splitTensor) []byte {
	t.Helper()
	const align = 32

	var kvs []func(*bytes.Buffer)
	if config {
		p := "kimi-k31."
		kvs = append(kvs,
			func(b *bytes.Buffer) { writeKVUint32(b, "general.alignment", align) },
			func(b *bytes.Buffer) { writeKVString(b, "general.architecture", "kimi-k31") },
			func(b *bytes.Buffer) { writeKVUint64(b, p+"embedding_length", 256) },
			func(b *bytes.Buffer) { writeKVUint64(b, p+"block_count", uint64(k31TestLayers)) },
			func(b *bytes.Buffer) { writeKVUint64(b, p+"attention.head_count", 4) },
			func(b *bytes.Buffer) { writeKVUint64(b, p+"feed_forward_length", 128) },
			func(b *bytes.Buffer) { writeKVFloat32(b, p+"attention.layer_norm_rms_epsilon", 1e-5) },
			func(b *bytes.Buffer) { writeKVUint64(b, p+"rope.dimension_count", 8) },
			func(b *bytes.Buffer) { writeKVIntArrayForTest(b, p+"attention.head_count_kv", []int32{0, 0, 0, 1}) },
			func(b *bytes.Buffer) { writeKVUint64(b, p+"expert_count", uint64(k31TestExperts)) },
			func(b *bytes.Buffer) { writeKVUint64(b, p+"expert_used_count", 2) },
			func(b *bytes.Buffer) { writeKVUint64(b, p+"expert_feed_forward_length", 64) },
			func(b *bytes.Buffer) { writeKVUint64(b, p+"expert_shared_count", 2) },
			func(b *bytes.Buffer) { writeKVUint64(b, p+"leading_dense_block_count", 1) },
			func(b *bytes.Buffer) { writeKVUint64(b, p+"attention.q_lora_rank", 64) },
			func(b *bytes.Buffer) { writeKVUint64(b, p+"attention.kv_lora_rank", 32) },
			func(b *bytes.Buffer) { writeKVUint64(b, p+"attention.key_length_mla", 24) },
			func(b *bytes.Buffer) { writeKVUint64(b, p+"attention.value_length_mla", 16) },
			func(b *bytes.Buffer) { writeKVUint64(b, p+"attention.indexer.head_count", 2) },
			func(b *bytes.Buffer) { writeKVUint64(b, p+"attention.indexer.key_length", 16) },
			func(b *bytes.Buffer) { writeKVUint64(b, p+"attention.indexer.top_k", 8) },
			func(b *bytes.Buffer) { writeKVUint64(b, p+"ssm.conv_kernel", 4) },
			func(b *bytes.Buffer) { writeKVUint64(b, p+"kda.head_dim", 32) },
			// Leaf L3 pin: the K3-inherited embedded BPE table the merged
			// Config() must carry to ride the shared GGML loader path.
			func(b *bytes.Buffer) { writeKVString(b, "tokenizer.ggml.model", "gpt2") },
			func(b *bytes.Buffer) { writeKVString(b, "tokenizer.ggml.pre", "kimi-k2") },
			func(b *bytes.Buffer) {
				writeKVStringArray(b, "tokenizer.ggml.tokens", []string{"!", "!!", "<|end_of_msg|>", "[PAD3]"})
			},
			func(b *bytes.Buffer) { writeKVStringArray(b, "tokenizer.ggml.merges", []string{"! !"}) },
			func(b *bytes.Buffer) { writeKVIntArrayForTest(b, "tokenizer.ggml.token_type", []int32{0, 0, 3, 2}) },
			func(b *bytes.Buffer) { writeKVUint32(b, "tokenizer.ggml.eos_token_id", 2) },
		)
	}
	kvs = append(kvs,
		func(b *bytes.Buffer) { writeKVUint32(b, "split.no", shardNo) },
		func(b *bytes.Buffer) { writeKVUint32(b, "split.count", count) },
		func(b *bytes.Buffer) { writeKVUint32(b, "split.tensors.count", uint32(len(tensors))) },
	)

	offsets := make([]uint64, len(tensors))
	off := 0
	for i, tt := range tensors {
		offsets[i] = uint64(off)
		n := 1
		for _, d := range tt.dims {
			n *= int(d)
		}
		off = (off + n*4 + align - 1) / align * align
	}

	var b bytes.Buffer
	writeMinimalHeader(&b, uint64(len(tensors)), uint64(len(kvs)))
	for _, kv := range kvs {
		kv(&b)
	}
	for i, tt := range tensors {
		writeTensorInfoForTest(&b, tt.name, tt.dims, tt.typ, offsets[i])
	}
	padToAlignment(&b, align)
	for _, tt := range tensors {
		dataStart := b.Len()
		b.Write(tt.data)
		padToLen(&b, dataStart+align)
	}
	return b.Bytes()
}

// fak-test:runtime medium est=1s lane=default
func TestKimiK31SplitShardsMergeSeam(t *testing.T) {
	// The complete reduced directory (mapped families + the three batched
	// slabs per MoE layer) distributed round-robin over THREE on-disk shards,
	// opened through the real OpenWeights merge seam: the shard count comes
	// from the config shard's split.count metadata, and every tensor lands in
	// the merged directory regardless of which shard carried it.
	pairs := synthKimiK31ReducedPairs()
	tensors := make([]splitTensor, 0, len(pairs)+3*(k31TestLayers-1))
	for _, p := range pairs {
		tensors = append(tensors, splitTensor{name: p.gguf, dims: []uint64{4}, typ: TensorF32, data: f32Payload(1, 2, 3, 4)})
	}
	for l := 1; l < k31TestLayers; l++ {
		for _, n := range k31SlabNames(l) {
			tensors = append(tensors, splitTensor{name: n, dims: []uint64{4, 4, uint64(k31TestExperts)}, typ: TensorF32, data: k31PatternF32(4 * 4 * k31TestExperts)})
		}
	}
	const shardCount = 3
	route := make(map[string]int, len(tensors))
	shards := make([][]splitTensor, shardCount)
	for i, tt := range tensors {
		s := i % shardCount
		route[tt.name] = s
		shards[s] = append(shards[s], tt)
	}

	dir := t.TempDir()
	paths := make([]string, shardCount)
	for i := 0; i < shardCount; i++ {
		paths[i] = filepath.Join(dir, fmt.Sprintf("k31-%05d-of-0000%d.gguf", i+1, shardCount))
		if err := os.WriteFile(paths[i], writeKimiK31SplitShard(t, uint32(i+1), uint32(shardCount), i == 0, shards[i]), 0o644); err != nil {
			t.Fatalf("write shard %d: %v", i+1, err)
		}
	}

	ws, err := OpenWeights(paths[0])
	if err != nil {
		t.Fatalf("OpenWeights(shard1): %v", err)
	}
	defer ws.Close()

	// (a) Every tensor from all three shards is merged into one directory.
	if len(ws.File.Tensors) != len(tensors) {
		t.Fatalf("merged directory has %d tensors, want %d (all three shards)", len(ws.File.Tensors), len(tensors))
	}
	for _, probe := range []string{"token_embd.weight", "output_res_score", "blk.0.ssm_g.weight", "blk.1.ffn_gate_exps.weight", "blk.3.indexer.proj.weight"} {
		if _, ok := ws.Tensor(probe); !ok {
			t.Errorf("merged source is missing tensor %q", probe)
		}
	}

	// (b) The merged Config resolves from shard 1's kimi-k31 metadata AND the
	// L2 tail census passes on the cross-shard directory (Config succeeding IS
	// the witness that every required family survived the merge).
	cfg, err := ws.File.Config()
	if err != nil {
		t.Fatalf("Config on merged split: %v", err)
	}
	if cfg.ModelType != "kimik31" || cfg.NumLayers != k31TestLayers || cfg.NumExperts != k31TestExperts {
		t.Fatalf("merged axes = type:%q layers:%d experts:%d, want kimik31/%d/%d",
			cfg.ModelType, cfg.NumLayers, cfg.NumExperts, k31TestLayers, k31TestExperts)
	}
	wantTypes := []string{kimiK31LayerKDA, kimiK31LayerKDA, kimiK31LayerKDA, kimiK31LayerDSA}
	for l, want := range wantTypes {
		if cfg.LayerTypes[l] != want {
			t.Fatalf("merged LayerTypes[%d] = %q, want %q", l, cfg.LayerTypes[l], want)
		}
	}

	// (c) The shard count comes from the config shard's split.count metadata,
	// and the map-level census over the MERGED directory accounts the routed-
	// expert band per layer and per shard under the round-robin routing.
	sc, ok := ws.File.Uint64("split.count")
	if !ok || sc != uint64(shardCount) {
		t.Fatalf("split.count = %d (ok=%v), want %d from the config shard metadata", sc, ok, shardCount)
	}
	c, err := CensusKimiK31ExpertBand(ws.File, cfg.NumLayers, int(sc), func(name string) int { return route[name] })
	if err != nil {
		t.Fatalf("CensusKimiK31ExpertBand on merged directory: %v", err)
	}
	if c.Occurrences != 3*k31TestExperts*3 { // 3 MoE layers x 3 slabs x E
		t.Errorf("Occurrences = %d, want %d", c.Occurrences, 3*k31TestExperts*3)
	}
	if c.Layers != 3 || c.Experts != k31TestExperts {
		t.Errorf("band = layers:%d experts:%d, want 3/%d", c.Layers, c.Experts, k31TestExperts)
	}
	for l := 1; l < k31TestLayers; l++ {
		if c.PerLayer[l] != 3*k31TestExperts {
			t.Errorf("PerLayer[%d] = %d, want %d", l, c.PerLayer[l], 3*k31TestExperts)
		}
	}
	if c.PerLayer[0] != 0 {
		t.Errorf("PerLayer[0] = %d, want 0 (dense layer)", c.PerLayer[0])
	}
	sum := 0
	for _, n := range c.PerShard {
		sum += n
	}
	if sum != c.Occurrences {
		t.Errorf("sum(PerShard) = %d, want Occurrences %d", sum, c.Occurrences)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestKimiK31OnlySuffixesDoNotLeakToSiblings(t *testing.T) {
	// The K3.1-only families must NOT resolve under any sibling arch: they
	// belong to the kimik31 arm alone. (Shared spellings like ssm_a /
	// ssm_beta / ssm_dt.bias / ssm_norm DO resolve under siblings via the base
	// linear_attn.* map, and the MLA/indexer/router spellings resolve under
	// glm_moe_dsa/deepseek2/deepseek41 — those are pinned elsewhere.)
	only := []string{
		"blk.0.ssm_g.weight",
		"blk.0.ssm_conv1d_q.weight",
		"blk.0.ssm_conv1d_k.weight",
		"blk.0.ssm_conv1d_v.weight",
		"blk.0.ssm_f_a.weight",
		"blk.0.ssm_f_b.weight",
		"blk.0.attn_res_score",
		"blk.0.ffn_res_score",
		"blk.0.ffn_routed_down.weight",
		"blk.0.ffn_routed_up.weight",
		"blk.0.ffn_routed_norm.weight",
		"blk.0.block_sparse_moe.experts.0.w1.weight_packed",
		"blk.0.block_sparse_moe.experts.0.w2.weight_scale",
		"output_res_score",
	}
	for _, arch := range []string{"deepseek2", "glm_moe_dsa", "deepseek41", "qwen35", ""} {
		for _, n := range only {
			if _, ok := CanonicalTensorNameArch(n, arch); ok {
				t.Errorf("%s resolved under sibling arch %q: a K3.1-only mapping leaked", n, arch)
			}
		}
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestKimiK31BatchedExpertCanonicalNamePinned(t *testing.T) {
	// The splitter's per-expert name composition, pinned per arch: kimik31
	// follows the converter's w1=gate / w2=down / w3=up order onto the census
	// leaf set; deepseek41 keeps its V4 w1/w3/w2 spelling; everything else
	// keeps the deepseek2/GLM mlp.experts spelling.
	cases := []struct {
		arch, proj, want string
	}{
		{"kimik31", "gate_proj", "model.layers.3.block_sparse_moe.experts.7.w1.weight"},
		{"kimik31", "down_proj", "model.layers.3.block_sparse_moe.experts.7.w2.weight"},
		{"kimik31", "up_proj", "model.layers.3.block_sparse_moe.experts.7.w3.weight"},
		{"deepseek41", "gate_proj", "model.layers.3.ffn.experts.7.w1.weight"},
		{"deepseek41", "up_proj", "model.layers.3.ffn.experts.7.w3.weight"},
		{"deepseek41", "down_proj", "model.layers.3.ffn.experts.7.w2.weight"},
		{"glm_moe_dsa", "gate_proj", "model.layers.3.mlp.experts.7.gate_proj.weight"},
		{"glm_moe_dsa", "up_proj", "model.layers.3.mlp.experts.7.up_proj.weight"},
		{"", "down_proj", "model.layers.3.mlp.experts.7.down_proj.weight"},
	}
	for _, tc := range cases {
		if got := batchedExpertCanonicalName(tc.arch, 3, 7, tc.proj); got != tc.want {
			t.Errorf("batchedExpertCanonicalName(%q, gate/up/down=%s) = %s, want %s", tc.arch, tc.proj, got, tc.want)
		}
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestKimiK31KVBHalvesMapToDistinctLeaves(t *testing.T) {
	// Under kimik31 the three KV-b forms resolve onto THREE distinct leaves
	// (combined or either half, whatever the artifact shipped — L6 owns the
	// merge decision). Contrast pin: under glm_moe_dsa the split halves stay
	// UNMAPPED because that arch's loader merges them 2->1 before this map.
	cases := []kimiK31Pair{
		{"blk.3.attn_kv_b.weight", "model.layers.3.self_attn.kv_b_proj.weight"},
		{"blk.3.attn_k_b.weight", "model.layers.3.self_attn.k_b_proj.weight"},
		{"blk.3.attn_v_b.weight", "model.layers.3.self_attn.v_b_proj.weight"},
	}
	for _, p := range cases {
		got, ok := CanonicalTensorNameArch(p.gguf, "kimik31")
		if !ok || got != p.want {
			t.Errorf("kimik31 %s = (%s, ok=%v), want %s", p.gguf, got, ok, p.want)
		}
	}
	for _, n := range []string{"blk.3.attn_k_b.weight", "blk.3.attn_v_b.weight"} {
		if _, ok := CanonicalTensorNameArch(n, "glm_moe_dsa"); ok {
			t.Errorf("glm_moe_dsa %s resolved; the split halves must stay merge-pre-pass-owned there", n)
		}
	}
}
