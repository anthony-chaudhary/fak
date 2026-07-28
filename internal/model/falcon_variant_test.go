package model

// Teeth for the Falcon-variant support fence (falcon_variant.go).
//
// The two unimplemented Falcon variants are dangerous precisely because they are
// SILENT: their fused self_attention.query_key_value tensor holds exactly the same
// number of rows as the contiguous q|k|v cut fak performs, so splitFusedProjections'
// row-count guard passes and the checkpoint loads clean and decodes wrong. Each
// refusal test below therefore proves BOTH halves:
//
//	(1) the silence — running the real materialize + split passes on the same
//	    fixture returns nil AND produces a q_proj that is demonstrably not HF's q
//	    (a named row-index mismatch, on distinct fixture values), so the fence is
//	    not guarding a defect the existing guard already catches; and
//	(2) the refusal — newModel returns a typed *UnsupportedFalconVariantError
//	    naming the variant.
//
// Everything is weight-free and synthetic: no checkpoint download, no t.Skip.

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"math"
	"strings"
	"testing"
)

const falconFenceHeadDim = 4

// falconFenceFill gives every tensor a distinct, non-zero, deterministic value.
// LayerNorm gains sit near 1 and biases away from 0 so no term can be masked; the
// fused qkv rows get a wide draw so two different row indices cannot collide.
func falconFenceFill(name string, next func() float32) float32 {
	switch {
	case strings.HasSuffix(name, "ln_attn.weight"), strings.HasSuffix(name, "ln_mlp.weight"),
		strings.HasSuffix(name, "layernorm.weight"), name == "transformer.ln_f.weight":
		return 1 + 0.25*next()
	case strings.HasSuffix(name, "ln_attn.bias"), strings.HasSuffix(name, "ln_mlp.bias"),
		strings.HasSuffix(name, "layernorm.bias"), name == "transformer.ln_f.bias":
		return 0.25 + 0.1*next()
	case strings.HasSuffix(name, "query_key_value.weight"):
		return 0.5 + next() // wide and offset: distinct rows separate cleanly
	default:
		return next() * 0.2
	}
}

// falconFenceFixture builds a Falcon-vocabulary synthetic checkpoint at the given
// head geometry. dualNorm selects the new_decoder_architecture block shape
// (ln_attn + ln_mlp, no input_layernorm); otherwise the single shared
// input_layernorm Falcon-7B and Falcon-RW both ship.
//
// The fused qkv width is HF's own qkv_out_dim = (num_heads + 2*num_kv_heads)*head_dim,
// which collapses to (nH+2)*hd for multi_query (nKV==1) and to 3*nH*hd for RW
// (nKV==nH) — the identity that makes the RW mis-cut undetectable by row count.
func falconFenceFixture(t *testing.T, nH, nKV int, dualNorm bool) (Config, map[string]tensorMeta, []byte) {
	t.Helper()
	hd := falconFenceHeadDim
	cfg := Config{
		HiddenSize:        nH * hd,
		NumLayers:         2,
		NumHeads:          nH,
		NumKVHeads:        nKV,
		HeadDim:           hd,
		IntermediateSize:  40,
		VocabSize:         53,
		ModelType:         "falcon",
		Architectures:     []string{"FalconForCausalLM"},
		HiddenAct:         "gelu",
		RMSNormEps:        1e-5,
		RopeTheta:         10000,
		TieWordEmbeddings: true,
	}
	if err := cfg.deriveConfigAxes(configJSONHints{}); err != nil {
		t.Fatalf("deriveConfigAxes: %v", err)
	}
	H, I, V := cfg.HiddenSize, cfg.IntermediateSize, cfg.VocabSize
	qkvRows := (nH + 2*nKV) * hd

	type ts = synthTensor
	tensors := []ts{{"transformer.word_embeddings.weight", []int{V, H}}}
	for l := 0; l < cfg.NumLayers; l++ {
		p := "transformer.h." + itoa(l) + "."
		if dualNorm {
			tensors = append(tensors,
				ts{p + "ln_attn.weight", []int{H}}, ts{p + "ln_attn.bias", []int{H}},
				ts{p + "ln_mlp.weight", []int{H}}, ts{p + "ln_mlp.bias", []int{H}})
		} else {
			tensors = append(tensors,
				ts{p + "input_layernorm.weight", []int{H}}, ts{p + "input_layernorm.bias", []int{H}})
		}
		tensors = append(tensors,
			ts{p + "self_attention.query_key_value.weight", []int{qkvRows, H}},
			ts{p + "self_attention.dense.weight", []int{H, nH * hd}},
			ts{p + "mlp.dense_h_to_4h.weight", []int{I, H}},
			ts{p + "mlp.dense_4h_to_h.weight", []int{H, I}},
		)
	}
	tensors = append(tensors,
		ts{"transformer.ln_f.weight", []int{H}},
		ts{"transformer.ln_f.bias", []int{H}},
	)
	man, raw := synthBuildRaw(tensors, falconFenceFill)
	return cfg, man, raw
}

// falconFenceRawRow reads row r of a [rows, in] f32 tensor straight out of the raw
// blob, so a check can compare the SOURCE bytes against what the split produced.
func falconFenceRawRow(raw []byte, meta tensorMeta, r, in int) []float32 {
	out := make([]float32, in)
	for c := 0; c < in; c++ {
		out[c] = math.Float32frombits(binary.LittleEndian.Uint32(raw[meta.Offset+(r*in+c)*4:]))
	}
	return out
}

// assertFalconSplitIsSilentlyWrong runs the REAL materialize + split passes and
// proves the mis-cut is undetectable by the existing guards: the split returns nil,
// and the q_proj it produced reads the CONTIGUOUS fused row for (head, dim) where
// HF's interleaved layout reads a different row — with different bytes. This is the
// vacuity floor: without it, a refusal test would prove only that the fence fires,
// not that anything was wrong.
// head selects which query head to probe: it must be one the two layouts disagree
// about (for the per-KV-group layout the heads of group 0 coincide with the
// contiguous cut, so the probe has to reach into a later group).
func assertFalconSplitIsSilentlyWrong(t *testing.T, cfg Config, man map[string]tensorMeta, raw []byte, head int, hfRowFor func(head, dim int) int) {
	t.Helper()
	fused := man["transformer.h.0.self_attention.query_key_value.weight"]
	if err := materializeFalconTensors(cfg, man, &raw); err != nil {
		t.Fatalf("materializeFalconTensors: %v", err)
	}
	if err := splitFusedProjections(cfg, man); err != nil {
		t.Fatalf("splitFusedProjections rejected the fixture (%v) — the row-count guard already"+
			" catches this variant, so the fence would be redundant; re-derive the fixture", err)
	}
	q, ok := man["model.layers.0.self_attn.q_proj.weight"]
	if !ok {
		t.Fatal("split produced no model.layers.0.self_attn.q_proj.weight")
	}
	nH, hd, H := cfg.NumHeads, cfg.HeadDim, cfg.HiddenSize
	if want := nH * hd; q.Shape[0] != want {
		t.Fatalf("split q_proj rows = %d, want %d (contiguous cut)", q.Shape[0], want)
	}
	// dim 0 of the probed head: the contiguous cut reads fused row head*hd; HF reads
	// hfRowFor(head, 0).
	const dim = 0
	gotRow, wantRow := head*hd+dim, hfRowFor(head, dim)
	if gotRow == wantRow {
		t.Fatalf("fixture is degenerate: contiguous row %d == HF row %d, so the layouts do not differ", gotRow, wantRow)
	}
	got := falconFenceRawRow(raw, q, gotRow, H)
	hf := falconFenceRawRow(raw, fused, wantRow, H)
	if got[0] == hf[0] {
		t.Fatalf("fixture is degenerate: fused rows %d and %d hold the same value %v", gotRow, wantRow, got[0])
	}
	t.Logf("silent mis-cut confirmed: split q head=%d dim=%d took fused row %d (%.6f), HF takes row %d (%.6f);"+
		" splitFusedProjections returned nil", head, dim, gotRow, got[0], wantRow, hf[0])
}

// assertRefusedFalconVariant asserts newModel refuses with the named variant and
// that the message is operator-actionable: it must name the variant AND say what IS
// supported, unlike the generic "required canonical tensor ... has no source".
func assertRefusedFalconVariant(t *testing.T, cfg Config, man map[string]tensorMeta, raw []byte, want FalconVariant) {
	t.Helper()
	m, err := newModel(cfg, man, raw)
	if err == nil {
		t.Fatalf("newModel ACCEPTED an unimplemented Falcon %s checkpoint (model=%p) — it would load clean"+
			" and decode wrong; want a typed *UnsupportedFalconVariantError", want, m)
	}
	var ufe *UnsupportedFalconVariantError
	if !errors.As(err, &ufe) {
		t.Fatalf("newModel error = %v (%T), want *UnsupportedFalconVariantError", err, err)
	}
	if ufe.Variant != want {
		t.Fatalf("refusal Variant = %q, want %q", ufe.Variant, want)
	}
	msg := ufe.Error()
	for _, need := range []string{string(want), "multi_query", "Falcon-7B", "INTERLEAVED"} {
		if !strings.Contains(msg, need) {
			t.Errorf("refusal message does not mention %q; got: %s", need, msg)
		}
	}
	if ufe.Witness == "" {
		t.Error("refusal carries no structural witness tensor")
	}
	t.Logf("refusal: %s", msg)
}

// TestFalconRWVariantRefusedAtLoad — Falcon-RW-1B / RW-7B (neither multi_query nor
// new_decoder_architecture) is plain MHA with a PER-HEAD INTERLEAVED fused qkv
// ((num_heads, 3, head_dim), so q of head h starts at fused row h*3*head_dim). The
// contiguous cut takes row h*head_dim instead. Both layouts hold 3*nH*hd rows.
func TestFalconRWVariantRefusedAtLoad(t *testing.T) {
	const nH = 6
	hfRow := func(head, dim int) int { return head*3*falconFenceHeadDim + dim }

	cfg, man, raw := falconFenceFixture(t, nH, nH, false)
	if cfg.NumKVHeads != cfg.NumHeads {
		t.Fatalf("RW fixture NumKVHeads = %d, want %d (plain MHA)", cfg.NumKVHeads, cfg.NumHeads)
	}
	assertFalconSplitIsSilentlyWrong(t, cfg, man, raw, 1, hfRow)

	cfg2, man2, raw2 := falconFenceFixture(t, nH, nH, false)
	assertRefusedFalconVariant(t, cfg2, man2, raw2, FalconRW)
}

// TestFalconNewDecoderArchVariantRefusedAtLoad — Falcon-40B / 180B
// (new_decoder_architecture=true) interleaves PER KV GROUP: group g holds
// (nH/nKV) q-heads then 1 k-head then 1 v-head, so q of head h starts at fused row
// (h/g)*(g+2)*hd + (h%g)*hd. The contiguous cut takes h*hd. Both layouts hold
// (nH+2*nKV)*hd rows. The structural tell is the ln_attn/ln_mlp block-norm pair.
func TestFalconNewDecoderArchVariantRefusedAtLoad(t *testing.T) {
	const nH, nKV = 6, 2
	g := nH / nKV
	hfRow := func(head, dim int) int {
		return (head/g)*(g+2)*falconFenceHeadDim + (head%g)*falconFenceHeadDim + dim
	}

	cfg, man, raw := falconFenceFixture(t, nH, nKV, true)
	if _, ok := man["transformer.h.0.input_layernorm.weight"]; ok {
		t.Fatal("new_decoder_architecture fixture must NOT carry input_layernorm")
	}
	// Probe the first head of GROUP 1 (head index g): group 0's q-heads happen to
	// land on the same rows under both layouts, so only a later group diverges.
	assertFalconSplitIsSilentlyWrong(t, cfg, man, raw, g, hfRow)

	cfg2, man2, raw2 := falconFenceFixture(t, nH, nKV, true)
	assertRefusedFalconVariant(t, cfg2, man2, raw2, FalconNewDecoderArch)
}

// TestFalconMultiQueryVariantStillLoads is the over-firing guard: the ONE variant
// fak implements (Falcon-7B, multi_query=true, num_kv_heads=1) must load exactly as
// before and produce the canonical contiguous q/k/v views. A fence that also refused
// the supported variant would be a regression, not a fence.
func TestFalconMultiQueryVariantStillLoads(t *testing.T) {
	cfg, man, raw := falconFenceFixture(t, 6, 1, false)
	if err := refuseUnsupportedFalconVariant(cfg, man); err != nil {
		t.Fatalf("fence fired on the SUPPORTED multi_query variant: %v", err)
	}
	m, err := newModel(cfg, man, raw)
	if err != nil {
		t.Fatalf("newModel(falcon multi_query): %v", err)
	}
	for _, name := range []string{
		"model.layers.0.self_attn.q_proj.weight",
		"model.layers.0.self_attn.k_proj.weight",
		"model.layers.0.self_attn.v_proj.weight",
		"model.layers.0.input_layernorm.weight",
	} {
		if _, ok := m.manifest[name]; !ok {
			t.Errorf("multi_query load is missing %q", name)
		}
	}
	if got := m.manifest["model.layers.0.self_attn.k_proj.weight"].Shape[0]; got != cfg.HeadDim {
		t.Errorf("multi_query k_proj rows = %d, want %d (one shared kv head)", got, cfg.HeadDim)
	}
}

// TestFalconFenceIgnoresNonFalcon proves the fence is scoped by the fused-qkv
// Falcon source vocabulary and not by anything broader: a manifest with no
// transformer.h.*.self_attention.query_key_value tensor is never refused, even at
// num_kv_heads == num_attention_heads (which is the ordinary MHA case for Llama).
func TestFalconFenceIgnoresNonFalcon(t *testing.T) {
	cfg := Config{NumLayers: 2, NumHeads: 8, NumKVHeads: 8, HeadDim: 4, ModelType: "llama"}
	man := map[string]tensorMeta{
		"model.layers.0.self_attn.q_proj.weight": {Dtype: "F32", Shape: []int{32, 32}, Nbytes: 32 * 32 * 4},
		"model.embed_tokens.weight":              {Dtype: "F32", Shape: []int{16, 32}, Nbytes: 16 * 32 * 4},
	}
	if err := refuseUnsupportedFalconVariant(cfg, man); err != nil {
		t.Fatalf("fence fired on a non-Falcon MHA manifest: %v", err)
	}
	// A single-head Falcon is also not refused: at nH==1 the per-head interleaved
	// layout and the contiguous layout are the same bytes, so there is no divergence.
	cfg1, man1, raw1 := falconFenceFixture(t, 1, 1, false)
	_ = raw1
	if err := refuseUnsupportedFalconVariant(cfg1, man1); err != nil {
		t.Fatalf("fence fired on a single-head Falcon (no layout divergence exists): %v", err)
	}
}

// TestFalconRWConfigDerivesMHAAndAlibi pins the config half of the RW divergence
// straight from a Falcon-RW-shaped config.json, so the derivation cannot drift out
// from under the fence:
//
//   - multi_query=false leaves the multi_query branch of deriveConfigAxes unfired and
//     num_kv_heads absent, so the fallback sets NumKVHeads = NumHeads (config.go) —
//     which is exactly the structural tell refuseUnsupportedFalconVariant reads.
//   - "alibi": true DOES reach Config.Alibi: the field carries json:"alibi" and
//     Config.UnmarshalJSON decodes the whole blob into the embedded alias, so ALiBi is
//     not silently dropped in favour of RoPE. (The remaining RW/ALiBi divergence is the
//     SCALING convention, named in the refusal message: HF Falcon folds the bias into
//     the logits as (scores+alibi)*1/sqrt(head_dim) while fak adds the MPT-convention
//     bias after the scale.)
//   - parallel_attn=false keeps the block off ParallelResidual, so RW is not the
//     Falcon-7B parallel block either.
func TestFalconRWConfigDerivesMHAAndAlibi(t *testing.T) {
	const rwConfigJSON = `{
		"model_type": "falcon",
		"architectures": ["FalconForCausalLM"],
		"hidden_size": 2048,
		"num_hidden_layers": 24,
		"num_attention_heads": 32,
		"multi_query": false,
		"new_decoder_architecture": false,
		"parallel_attn": false,
		"alibi": true,
		"bias": true,
		"layer_norm_epsilon": 1e-5,
		"vocab_size": 50304
	}`
	var cfg Config
	if err := json.Unmarshal([]byte(rwConfigJSON), &cfg); err != nil {
		t.Fatalf("Unmarshal falcon-rw config: %v", err)
	}
	if cfg.NumHeads != 32 {
		t.Fatalf("NumHeads = %d, want 32", cfg.NumHeads)
	}
	if cfg.NumKVHeads != cfg.NumHeads {
		t.Errorf("NumKVHeads = %d, want %d: multi_query=false must fall through to plain MHA,"+
			" which is the tell the variant fence reads", cfg.NumKVHeads, cfg.NumHeads)
	}
	if !cfg.Alibi {
		t.Error("Config.Alibi = false for a config.json carrying \"alibi\": true —" +
			" Falcon-RW would silently run RoPE where HF runs ALiBi")
	}
	if cfg.BlockTopology == ParallelResidual {
		t.Error("BlockTopology = ParallelResidual for parallel_attn=false (Falcon-RW is the serial block)")
	}
}
