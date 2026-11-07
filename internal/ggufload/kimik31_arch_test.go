package ggufload

import (
	"strings"
	"testing"
)

// Kimi K3.1 ("kimi-k31") arch admission: recognition, config applier, and the
// fail-closed typed refusals (ticket 1611-kimi-k31 leaf L1).
//
// These are HEADER-ONLY synthetic fixtures (no network, no real weights): they
// pin the arch-string normalization, the raw "<arch>." metadata-prefix reads,
// and the exact axis values the loader/model layer needs, following the
// deepseek41 admission-test pattern. Execution is deliberately NOT asserted:
// model.ClassifyForwardPath has no K3.1 path yet (L5–L8 own internal/model),
// so a loaded kimi-k31 checkpoint is admitted at the header and stays
// execution-gated until the native forward lands.
//
// Fixture geometry is the MEASURED Kimi-K3.1-Pro shape (L0 artifact census,
// docs/research/KIMI-K31-L0-ARTIFACT-CENSUS-2026-11-07.md in fak-private):
// 93 layers = 24 Gated-DSA (0-based 3,7,…,91,92 — config dsa_layers
// [4,8,…,92,93], 1-based) + 69 KDA + 1 dense-FFN layer (layer 0, whose
// attention is still KDA); 896 routed experts x 92 MoE layers; DSA indexer
// 32 heads x 128 dim, top-2048; MLA q_lora 1536 / kv_lora 512, per-head qk
// 192 = qk_nope 128 + qk_rope 64, per-head v 128; latent routed space 3584,
// expert FFN 3072; KDA conv kernel 4, head dim 128.

// kimiK31HeadCountKVArray builds the per-layer attention.head_count_kv integer
// array for the measured K3.1-Pro schedule: entry 0 marks a KDA layer, a
// positive entry a DSA layer (the llama.cpp K3 recurrent marker). DSA layers
// are 0-based 3,7,…,91,92 (24 total); every other layer is KDA.
func kimiK31HeadCountKVArray() Value {
	items := make([]Value, 93)
	for l := range items {
		if l%4 == 3 || l == 92 {
			items[l] = Value{Type: TypeUint32, Value: uint32(1)}
		} else {
			items[l] = Value{Type: TypeUint32, Value: uint32(0)}
		}
	}
	return Value{Type: TypeArray, Value: items}
}

// kimiK31SynthTokenizerTable returns the pinned K3-inherited embedded BPE table
// as global tokenizer.ggml.* keys (leaf L3 fixture), shaped like the community
// K3/K3.1 converter output (kimi-linear set_vocab): model "gpt2", pre "kimi-k2",
// CONTROL-marked special at id 2, UNUSED pad slot at id 3. The four-entry vocab
// is byte-level BPE: ids 0/1 spell "!" / "!!" and the "! !" merge fires inside
// one pre-tokenization chunk (punctuation never splits per rune), so Encode("!!")
// resolves to [1] through the shared loader path; ASCII passes the byte-level
// encode unchanged. withPre=false drops tokenizer.ggml.pre to exercise the
// general.architecture fallback the legacy K3 files rely on.
func kimiK31SynthTokenizerTable(withPre bool) map[string]Value {
	table := map[string]Value{
		"tokenizer.ggml.model": {Type: TypeString, Value: "gpt2"},
		"tokenizer.ggml.tokens": {Type: TypeArray, Value: []Value{
			{Type: TypeString, Value: "!"},
			{Type: TypeString, Value: "!!"},
			{Type: TypeString, Value: "<|end_of_msg|>"},
			{Type: TypeString, Value: "[PAD3]"},
		}},
		"tokenizer.ggml.merges": {Type: TypeArray, Value: []Value{
			{Type: TypeString, Value: "! !"},
		}},
		"tokenizer.ggml.token_type": {Type: TypeArray, Value: []Value{
			{Type: TypeInt32, Value: int32(0)}, // NORMAL
			{Type: TypeInt32, Value: int32(0)}, // NORMAL
			{Type: TypeInt32, Value: int32(3)}, // CONTROL (<|end_of_msg|>)
			{Type: TypeInt32, Value: int32(2)}, // UNUSED ([PAD3])
		}},
		"tokenizer.ggml.eos_token_id": {Type: TypeUint32, Value: uint32(2)},
	}
	if withPre {
		table["tokenizer.ggml.pre"] = Value{Type: TypeString, Value: "kimi-k2"}
	}
	return table
}

// synthKimiK31Meta is the minimal Kimi-K3.1-Pro-shaped MLA+DSA+MoE metadata,
// keyed on the file's OWN "<arch>." prefix (not "kimik31.") to prove the
// metadata reads use the raw arch prefix while recognition normalizes the arch
// string — the same contract synthKimiMeta pins for K2. It carries the complete
// pinned K3-inherited tokenizer table (leaf L3): an admitted kimi-k31 header
// must expose the embedded BPE table the K3 loader path rides.
func synthKimiK31Meta(arch string) map[string]Value {
	p := arch + "."
	meta := map[string]Value{
		"general.architecture":                 {Type: TypeString, Value: arch},
		p + "context_length":                   {Type: TypeUint64, Value: uint64(1048576)},
		p + "embedding_length":                 {Type: TypeUint64, Value: uint64(7168)},
		p + "block_count":                      {Type: TypeUint64, Value: uint64(93)},
		p + "feed_forward_length":              {Type: TypeUint64, Value: uint64(33792)}, // dense layer 0 mlp width
		p + "attention.head_count":             {Type: TypeUint64, Value: uint64(96)},
		p + "attention.layer_norm_rms_epsilon": {Type: TypeFloat32, Value: float32(1e-5)},
		p + "rope.dimension_count":             {Type: TypeUint64, Value: uint64(64)},
		// Per-layer KDA/DSA composition: 0 = KDA, 1 = DSA (measured schedule).
		p + "attention.head_count_kv": kimiK31HeadCountKVArray(),
		// MoE FFN axis: 896 routed / 16 active / 2 shared, expert FFN 3072.
		p + "expert_count":                      {Type: TypeUint64, Value: uint64(896)},
		p + "expert_used_count":                 {Type: TypeUint64, Value: uint64(16)},
		p + "expert_feed_forward_length":        {Type: TypeUint64, Value: uint64(3072)},
		p + "expert_shared_count":               {Type: TypeUint64, Value: uint64(2)},
		p + "expert_shared_feed_forward_length": {Type: TypeUint64, Value: uint64(3072)},
		p + "leading_dense_block_count":         {Type: TypeUint64, Value: uint64(1)},
		p + "expert_weights_norm":               {Type: TypeBool, Value: true},
		// Deferred-axis keys present in real headers: unread at L1 (no
		// model.Config field yet) but must not disturb admission.
		p + "expert_latent_length":        {Type: TypeUint64, Value: uint64(3584)},
		p + "attn_res_block_size":         {Type: TypeUint64, Value: uint64(12)},
		p + "activation.situ_beta":        {Type: TypeFloat32, Value: float32(4.0)},
		p + "activation.situ_linear_beta": {Type: TypeFloat32, Value: float32(25.0)},
		// MLA latent-projection ranks + per-head widths (qk = nope 128 + rope 64).
		p + "attention.q_lora_rank":      {Type: TypeUint64, Value: uint64(1536)},
		p + "attention.kv_lora_rank":     {Type: TypeUint64, Value: uint64(512)},
		p + "attention.key_length_mla":   {Type: TypeUint64, Value: uint64(192)},
		p + "attention.value_length_mla": {Type: TypeUint64, Value: uint64(128)},
		// Gated DSA learned-indexer axis: 32 index heads x 128 dim, top-2048.
		p + "attention.indexer.head_count": {Type: TypeUint64, Value: uint64(32)},
		p + "attention.indexer.key_length": {Type: TypeUint64, Value: uint64(128)},
		p + "attention.indexer.top_k":      {Type: TypeUint64, Value: uint64(2048)},
		// KDA (Kimi Delta Attention) axis.
		p + "ssm.conv_kernel":      {Type: TypeUint64, Value: uint64(4)},
		p + "kda.head_dim":         {Type: TypeUint64, Value: uint64(128)},
		p + "kda.gate_lower_bound": {Type: TypeFloat32, Value: float32(-5.0)},
	}
	for k, v := range kimiK31SynthTokenizerTable(true) {
		meta[k] = v
	}
	return meta
}

// TestCanonicalGGUFArchNormalizesKimiK31 pins the arch-string normalization AND
// the alias precedence: the planned K3.1 spellings normalize onto the distinct
// internal "kimik31", while every pre-existing Moonshot spelling keeps routing
// to "deepseek2" untouched (the old alias is exact-spelling only and must not
// capture "kimi-k31").
func TestCanonicalGGUFArchNormalizesKimiK31(t *testing.T) {
	for _, spelling := range []string{"kimi-k31", "kimi_k31"} {
		if got := canonicalGGUFArch(spelling); got != "kimik31" {
			t.Errorf("canonicalGGUFArch(%q) = %q, want kimik31", spelling, got)
		}
	}
	// Sibling-leak: the K2/K3 alias must stay byte-for-byte intact.
	for _, spelling := range []string{"kimi-k2", "kimi_k2", "kimi2", "kimi-k3", "kimi_k3", "kimi3", "kimi"} {
		if got := canonicalGGUFArch(spelling); got != "deepseek2" {
			t.Errorf("canonicalGGUFArch(%q) = %q, want deepseek2 (K2/K3 alias leaked)", spelling, got)
		}
	}
	// Precedence pin: K3.1 must NOT fall into the deepseek2 alias.
	if got := canonicalGGUFArch("kimi-k31"); got == "deepseek2" {
		t.Fatalf("canonicalGGUFArch(kimi-k31) = deepseek2: the old no-indexer alias captured K3.1")
	}
	// Unrelated arches pass through untouched (no over-normalization).
	if got := canonicalGGUFArch("llama"); got != "llama" {
		t.Errorf("canonicalGGUFArch(llama) = %q, want llama", got)
	}
}

// TestKimiK31ConfigDerivesDistinctArch is the done-condition witness: a
// kimi-k31-branded GGUF header derives a Config whose model_type is the
// distinct "kimik31" (NOT deepseek2) and whose MoE + MLA + DSA-indexer + KDA
// axes are populated from the raw-prefixed keys with the exact measured
// K3.1-Pro values.
func TestKimiK31ConfigDerivesDistinctArch(t *testing.T) {
	f := &File{Metadata: synthKimiK31Meta("kimi-k31")}
	cfg, err := f.Config()
	if err != nil {
		t.Fatalf("Config: %v", err)
	}
	if cfg.ModelType != "kimik31" {
		t.Fatalf("ModelType = %q, want kimik31 (distinct from the no-indexer deepseek2 alias)", cfg.ModelType)
	}
	if !cfg.IsMoE() {
		t.Fatalf("IsMoE() = false; want true (NumExperts=%d)", cfg.NumExperts)
	}
	// Generic block axes.
	if cfg.HiddenSize != 7168 || cfg.NumLayers != 93 || cfg.NumHeads != 96 {
		t.Errorf("core geometry = hidden:%d layers:%d heads:%d, want 7168/93/96",
			cfg.HiddenSize, cfg.NumLayers, cfg.NumHeads)
	}
	if cfg.IntermediateSize != 33792 {
		t.Errorf("IntermediateSize = %d, want 33792 (dense layer 0 mlp width)", cfg.IntermediateSize)
	}
	if cfg.MaxPositionEmbeddings != 1048576 {
		t.Errorf("MaxPositionEmbeddings = %d, want 1048576", cfg.MaxPositionEmbeddings)
	}
	// MoE FFN axis: 896 routed / 16 active / 2 shared, expert FFN 3072.
	if cfg.NumExperts != 896 || cfg.NumExpertsPerTok != 16 {
		t.Fatalf("MoE counts = experts:%d topk:%d, want 896/16", cfg.NumExperts, cfg.NumExpertsPerTok)
	}
	if cfg.MoEIntermediateSize != 3072 || cfg.SharedIntermediateSize != 3072 {
		t.Errorf("MoE ffn = moe:%d shared:%d, want 3072/3072", cfg.MoEIntermediateSize, cfg.SharedIntermediateSize)
	}
	if cfg.NSharedExperts != 2 || cfg.FirstKDenseReplace != 1 {
		t.Errorf("MoE structure = shared:%d firstKDense:%d, want 2/1", cfg.NSharedExperts, cfg.FirstKDenseReplace)
	}
	if !cfg.NormTopKProb {
		t.Errorf("NormTopKProb = false, want true (expert_weights_norm)")
	}
	// MLA latent-projection ranks + per-head widths (qk = nope 128 + rope 64 = 192).
	if cfg.QLoraRank != 1536 || cfg.KVLoraRank != 512 {
		t.Fatalf("MLA ranks = q:%d kv:%d, want 1536/512", cfg.QLoraRank, cfg.KVLoraRank)
	}
	if cfg.QKRopeHeadDim != 64 || cfg.QKNopeHeadDim != 128 || cfg.VHeadDim != 128 {
		t.Errorf("MLA head dims = rope:%d nope:%d v:%d, want 64/128/128",
			cfg.QKRopeHeadDim, cfg.QKNopeHeadDim, cfg.VHeadDim)
	}
	if cfg.HeadDim != 192 {
		t.Errorf("HeadDim = %d, want 192 (qk_nope + qk_rope, not hidden/heads = 74)", cfg.HeadDim)
	}
	// Gated DSA learned-indexer axis: 32 index heads x 128 dim, top-2048.
	if cfg.IndexNHeads != 32 || cfg.IndexHeadDim != 128 || cfg.IndexTopK != 2048 {
		t.Fatalf("DSA indexer = heads:%d dim:%d topk:%d, want 32/128/2048",
			cfg.IndexNHeads, cfg.IndexHeadDim, cfg.IndexTopK)
	}
	// KDA (Kimi Delta Attention) axis.
	if cfg.LinearConvKernelDim != 4 || cfg.LinearKeyHeadDim != 128 || cfg.LinearValueHeadDim != 128 {
		t.Errorf("KDA axis = conv:%d keyDim:%d valDim:%d, want 4/128/128",
			cfg.LinearConvKernelDim, cfg.LinearKeyHeadDim, cfg.LinearValueHeadDim)
	}
	if cfg.LinearGateLowerBound != -5.0 {
		t.Errorf("LinearGateLowerBound = %v, want -5.0", cfg.LinearGateLowerBound)
	}
	// SiTU-GLU activation family recorded when the situ keys are present.
	if cfg.HiddenAct != "situ" {
		t.Errorf("HiddenAct = %q, want situ", cfg.HiddenAct)
	}
}

// TestKimiK31LayerCompositionExact pins the per-layer KDA/DSA schedule the
// applier derives from the head_count_kv array: exactly 24 DSA layers at the
// measured 0-based positions 3,7,…,91,92, 69 KDA layers, and layer 0 marked
// KDA (its attention is KDA even though its FFN is the dense one — density
// rides FirstKDenseReplace, not a third token).
func TestKimiK31LayerCompositionExact(t *testing.T) {
	f := &File{Metadata: synthKimiK31Meta("kimi-k31")}
	cfg, err := f.Config()
	if err != nil {
		t.Fatalf("Config: %v", err)
	}
	types := cfg.LayerTypes
	// The derived schedule must cover every decoder layer exactly once.
	if len(types) != cfg.NumLayers {
		t.Fatalf("len(LayerTypes) = %d, want NumLayers=%d (one entry per decoder layer)", len(types), cfg.NumLayers)
	}
	dsa, kda := 0, 0
	for l, got := range types {
		isDSA := l%4 == 3 || l == 92
		if isDSA {
			dsa++
		} else {
			kda++
		}
		want := kimiK31LayerKDA
		if isDSA {
			want = kimiK31LayerDSA
		}
		if got != want {
			t.Errorf("LayerTypes[%d] = %q, want %q", l, got, want)
		}
	}
	if dsa != 24 || kda != 69 {
		t.Fatalf("composition = %d DSA + %d KDA, want 24 DSA + 69 KDA", dsa, kda)
	}
	// Spot-check the boundary layers: dense-FFN layer 0 is KDA attention; the
	// strided DSA band starts at 3; the final layer 92 is DSA.
	for l, want := range map[int]string{0: kimiK31LayerKDA, 1: kimiK31LayerKDA, 2: kimiK31LayerKDA,
		3: kimiK31LayerDSA, 4: kimiK31LayerKDA, 7: kimiK31LayerDSA, 91: kimiK31LayerDSA, 92: kimiK31LayerDSA} {
		if types[l] != want {
			t.Errorf("LayerTypes[%d] = %q, want %q", l, types[l], want)
		}
	}
}

// TestKimiK31HeaderMissingRequiredKeyRefused is the fail-closed witness: a
// kimi-k31-branded header missing (or zeroing) any arch-defining key refuses at
// the header with a typed error that names the exact "<arch>.<key>" — never a
// silent fall-through to the no-indexer deepseek2 semantics.
func TestKimiK31HeaderMissingRequiredKeyRefused(t *testing.T) {
	required := []string{
		"attention.indexer.head_count",
		"attention.indexer.key_length",
		"attention.indexer.top_k",
		"expert_count",
		"expert_used_count",
		"attention.q_lora_rank",
		"attention.kv_lora_rank",
	}
	for _, key := range required {
		for _, mode := range []string{"absent", "zero"} {
			meta := synthKimiK31Meta("kimi-k31")
			fullKey := "kimi-k31." + key
			switch mode {
			case "absent":
				delete(meta, fullKey)
			case "zero":
				meta[fullKey] = Value{Type: TypeUint64, Value: uint64(0)}
			}
			_, err := (&File{Metadata: meta}).Config()
			if err == nil {
				t.Errorf("%s (%s): Config succeeded, want typed refusal", key, mode)
				continue
			}
			if !strings.Contains(err.Error(), fullKey) {
				t.Errorf("%s (%s): refusal %q does not name the offending key %q", key, mode, err, fullKey)
			}
		}
	}
}

// TestKimiK31ScheduleUnresolvedRefused pins the per-layer-schedule guard: a
// kimi-k31 file that SHIPS TENSORS but leaves the KDA/DSA composition
// unresolvable (head_count_kv absent or the wrong length) refuses naming the
// key, while a header-only fixture without the array is a valid no-op — the
// same load-bearing distinction the glm_moe_dsa unresolved-schedule guard makes.
func TestKimiK31ScheduleUnresolvedRefused(t *testing.T) {
	withTensors := func(meta map[string]Value) *File {
		return &File{Metadata: meta, Tensors: []TensorInfo{{Name: "blk.0.attn_q_a.weight"}}}
	}
	// (a) Tensors present, schedule key absent.
	meta := synthKimiK31Meta("kimi-k31")
	delete(meta, "kimi-k31.attention.head_count_kv")
	_, err := withTensors(meta).Config()
	if err == nil || !strings.Contains(err.Error(), "kimi-k31.attention.head_count_kv") {
		t.Fatalf("(a) absent schedule with tensors: err = %v, want refusal naming kimi-k31.attention.head_count_kv", err)
	}
	// (b) Tensors present, schedule array the wrong length.
	bad := synthKimiK31Meta("kimi-k31")
	bad["kimi-k31.attention.head_count_kv"] = Value{Type: TypeArray, Value: []Value{
		{Type: TypeUint32, Value: uint32(0)},
		{Type: TypeUint32, Value: uint32(1)},
	}}
	_, err = withTensors(bad).Config()
	if err == nil || !strings.Contains(err.Error(), "kimi-k31.attention.head_count_kv") {
		t.Fatalf("(b) short schedule with tensors: err = %v, want refusal naming kimi-k31.attention.head_count_kv", err)
	}
	// (c) Header-only fixture without the schedule: valid no-op, empty LayerTypes.
	noSchedule := synthKimiK31Meta("kimi-k31")
	delete(noSchedule, "kimi-k31.attention.head_count_kv")
	cfg, err := (&File{Metadata: noSchedule}).Config()
	if err != nil {
		t.Fatalf("(c) header-only without schedule: %v, want clean admission", err)
	}
	if len(cfg.LayerTypes) != 0 {
		t.Fatalf("(c) header-only LayerTypes = %v, want empty (schedule unresolved but unsolicited)", cfg.LayerTypes)
	}
}

// TestKimiK31SpellingVariantRoutesSame proves the underscore spelling admits
// identically: same distinct model_type, same axes, metadata reads staying on
// the file's own "kimi_k31." prefix.
func TestKimiK31SpellingVariantRoutesSame(t *testing.T) {
	f := &File{Metadata: synthKimiK31Meta("kimi_k31")}
	cfg, err := f.Config()
	if err != nil {
		t.Fatalf("Config: %v", err)
	}
	if cfg.ModelType != "kimik31" {
		t.Fatalf("ModelType = %q, want kimik31", cfg.ModelType)
	}
	if cfg.NumExperts != 896 || cfg.IndexTopK != 2048 || len(cfg.LayerTypes) != cfg.NumLayers {
		t.Fatalf("axes = experts:%d topk:%d schedule:%d/%d layers, want 896/2048/full-coverage",
			cfg.NumExperts, cfg.IndexTopK, len(cfg.LayerTypes), cfg.NumLayers)
	}
}

// TestMislabeledK3BrandedKimiK31Refused is the alias-precedence end-to-end
// witness: a genuine K2/K3-shaped header (no DSA indexer — exactly what the
// old "kimi*" -> deepseek2 alias assumes) REBRANDED as "kimi-k31" must refuse
// at the header naming the missing indexer key, instead of silently riding the
// deepseek2 no-indexer semantics. This is the failure mode the distinct arch
// exists to prevent.
func TestMislabeledK3BrandedKimiK31Refused(t *testing.T) {
	f := &File{Metadata: synthKimiMeta("kimi-k31")} // K2-shaped metadata, K3.1 branding
	_, err := f.Config()
	if err == nil {
		t.Fatal("a K2/K3-shaped header branded kimi-k31 was admitted; want typed refusal")
	}
	if !strings.Contains(err.Error(), "kimi-k31.attention.indexer.head_count") {
		t.Fatalf("refusal %q does not name the missing indexer key", err)
	}
}

// TestKimiK31DoesNotClassifyAsSiblingForward guards against accidental
// absorption into an existing forward predicate at the CONFIG level: the
// derived Config must not look like a qwen35 hybrid (no "linear_attention"
// token) nor carry the deepseek2 model_type, so no sibling loader gate fires
// on it. (The forward-selection seam itself is owned by L5–L8 in
// internal/model; this pins only the config-side non-absorption.)
func TestKimiK31DoesNotClassifyAsSiblingForward(t *testing.T) {
	f := &File{Metadata: synthKimiK31Meta("kimi-k31")}
	cfg, err := f.Config()
	if err != nil {
		t.Fatalf("Config: %v", err)
	}
	if cfg.IsQwen35Hybrid() {
		t.Fatal("IsQwen35Hybrid() = true: K3.1 LayerTypes leaked into the GDN vocabulary")
	}
	if cfg.ModelType == "deepseek2" || cfg.ModelType == "glm_moe_dsa" {
		t.Fatalf("ModelType = %q: K3.1 absorbed into a sibling MLA arch", cfg.ModelType)
	}
}
