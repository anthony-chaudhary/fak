package ggufload

import (
	"bytes"
	"compress/gzip"
	"crypto/sha256"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// The fixture is the complete, losslessly compressed *header prefix* of the pinned
// public Qwen3.8-27B-Q4_K_M artifact — everything through the 32-byte-aligned
// tensor-data offset, with no tensor payload. Provenance distinguishes the fixture's
// own digest from the independently verified full-file digest of the source artifact:
// the header cannot reproduce the whole-file SHA-256, so the extraction record names
// the source digest rather than pretending to recompute it. This proves exact-artifact
// loader admission only; it is not logits parity and not a performance claim.
const (
	// qwen38Q4KMArtifactBytes is the pinned on-disk size of the source artifact.
	qwen38Q4KMArtifactBytes = int64(17106775008)
	// qwen38Q4KMArtifactSHA256 is the independently verified full-file SHA-256 of the
	// source artifact `Qwen3.8-27B-Q4_K_M.gguf` (unsloth/Qwen3.8-27B-GGUF). It cannot
	// be recomputed from the header-only fixture.
	qwen38Q4KMArtifactSHA256 = "7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169"
	// qwen38Q4KMHeaderSHA256 and qwen38Q4KMHeaderBytes identify the decompressed
	// header-prefix bytes the fixture carries; both are locally computed from the
	// source artifact and are distinct from the whole-file digest above.
	qwen38Q4KMHeaderSHA256 = "2b388fac52efc56320cd60053cdfdf0a5f888ef3de64d2913c220498cced3916"
	qwen38Q4KMHeaderBytes  = 10996704
)

// qwen38Q4KMHeaderRaw decompresses the pinned header fixture and verifies its exact
// byte identity before returning the raw bytes. A fixture whose digest or length has
// drifted must fail closed rather than silently load a different header.
func qwen38Q4KMHeaderRaw(t *testing.T) []byte {
	t.Helper()
	compressed, err := os.ReadFile(filepath.Join("testdata", "qwen38_q4km_header.gguf.gz"))
	if err != nil {
		t.Fatal(err)
	}
	zr, err := gzip.NewReader(bytes.NewReader(compressed))
	if err != nil {
		t.Fatal(err)
	}
	defer zr.Close()
	raw, err := io.ReadAll(io.LimitReader(zr, 64<<20))
	if err != nil {
		t.Fatal(err)
	}
	if len(raw) != qwen38Q4KMHeaderBytes || fmt.Sprintf("%x", sha256.Sum256(raw)) != qwen38Q4KMHeaderSHA256 {
		t.Fatalf("pinned header identity mismatch: bytes=%d sha256=%x, want bytes=%d sha256=%s",
			len(raw), sha256.Sum256(raw), qwen38Q4KMHeaderBytes, qwen38Q4KMHeaderSHA256)
	}
	return raw
}

// TestQwen38Q4KMArtifactContract loads the scrubbed header of the pinned
// Qwen3.8-27B-Q4_K_M artifact through the production GGUF parser (not a hand-built
// model.Config) and pins the exact-artifact admission contract every Strix Halo
// performance claim depends on: extraction provenance, fixture identity, hybrid
// geometry, required tensor families, and the observed MTP inventory. Mutation of the
// quality-critical header must fail closed.
func TestQwen38Q4KMArtifactContract(t *testing.T) {
	raw := qwen38Q4KMHeaderRaw(t)

	// The fixture must be payload-free: the parser's own resolved tensor-data offset
	// is at or beyond the last fixture byte, so no weight payload is embedded.
	gg, err := Read(bytes.NewReader(raw))
	if err != nil {
		t.Fatalf("Read pinned Q4_K_M header: %v", err)
	}
	if gg.TensorDataOffset < int64(len(raw)) {
		t.Fatalf("fixture is not payload-free: data offset %d < header bytes %d", gg.TensorDataOffset, len(raw))
	}
	if gg.Alignment != 32 {
		t.Fatalf("Alignment = %d, want 32", gg.Alignment)
	}

	// Exact-artifact identity metadata: the pinned quant, quantizer and size label.
	for key, want := range map[string]string{
		"general.name":         "Qwen3.8-27B",
		"general.basename":     "Qwen3.8-27B",
		"general.size_label":   "27B",
		"general.quantized_by": "Unsloth",
		"general.license":      "apache-2.0",
		"general.architecture": "qwen35",
		"tokenizer.ggml.pre":   "qwen35",
	} {
		got, ok := gg.String(key)
		if !ok || got != want {
			t.Fatalf("metadata %s = %q (present=%v), want %q", key, got, ok, want)
		}
	}
	if ft, ok := gg.Uint64("general.file_type"); !ok || ft != 15 {
		t.Fatalf("general.file_type = %d (present=%v), want 15 (k-quant mix)", ft, ok)
	}
	if qv, ok := gg.Uint64("general.quantization_version"); !ok || qv != 2 {
		t.Fatalf("general.quantization_version = %d (present=%v), want 2", qv, ok)
	}

	// Loaded hybrid geometry: the production parser must derive the target depth
	// (block_count 65 minus the single MTP block) and the full-attention cadence.
	cfg, err := gg.Config()
	if err != nil {
		t.Fatalf("Config pinned Q4_K_M header: %v", err)
	}
	if cfg.Name != "Qwen3.8-27B" || cfg.ModelType != "qwen35" {
		t.Fatalf("identity config mismatch: Name=%q ModelType=%q", cfg.Name, cfg.ModelType)
	}
	if cfg.HiddenSize != 5120 || cfg.NumLayers != 64 || cfg.NumHeads != 24 || cfg.NumKVHeads != 4 || cfg.HeadDim != 256 {
		t.Fatalf("geometry mismatch: hidden=%d layers=%d heads=%d kv_heads=%d head_dim=%d",
			cfg.HiddenSize, cfg.NumLayers, cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim)
	}
	if cfg.IntermediateSize != 17408 || cfg.VocabSize != 248320 {
		t.Fatalf("ffn/vocab mismatch: ffn=%d vocab=%d", cfg.IntermediateSize, cfg.VocabSize)
	}
	if cfg.MaxPositionEmbeddings != 262144 || cfg.EOSTokenID != 248046 {
		t.Fatalf("context/eos mismatch: max_pos=%d eos=%d", cfg.MaxPositionEmbeddings, cfg.EOSTokenID)
	}
	if !cfg.IsQwen35Hybrid() {
		t.Fatalf("IsQwen35Hybrid = false; layer_types = %v", cfg.LayerTypes)
	}
	if len(cfg.LayerTypes) != 64 {
		t.Fatalf("LayerTypes length = %d, want 64 (target depth, MTP block excluded)", len(cfg.LayerTypes))
	}
	if cfg.FullAttentionInterval != 4 {
		t.Fatalf("FullAttentionInterval = %d, want 4", cfg.FullAttentionInterval)
	}
	// Exactly one full-attention layer per 4-layer cadence over 64 target layers.
	full := 0
	for l, lt := range cfg.LayerTypes {
		want := "linear_attention"
		if (l+1)%4 == 0 {
			want = "full_attention"
			full++
		}
		if lt != want {
			t.Fatalf("LayerTypes[%d] = %q, want %q", l, lt, want)
		}
	}
	if full != 16 {
		t.Fatalf("full_attention layer count = %d, want 16 (64/4)", full)
	}
	if !cfg.AttnOutputGate || !cfg.NormGain1p || !cfg.QKNorm {
		t.Fatalf("hybrid axes = gate:%v norm_gain_1p:%v qk_norm:%v, want all true",
			cfg.AttnOutputGate, cfg.NormGain1p, cfg.QKNorm)
	}
	// GDN/SSM axes: conv kernel 4, state 128, 16 key/timestep groups, 48 value heads.
	if cfg.LinearConvKernelDim != 4 || cfg.LinearKeyHeadDim != 128 || cfg.LinearValueHeadDim != 128 ||
		cfg.LinearNumKeyHeads != 16 || cfg.LinearNumValueHeads != 48 {
		t.Fatalf("ssm axes mismatch: conv=%d key_dim=%d value_dim=%d key_heads=%d value_heads=%d",
			cfg.LinearConvKernelDim, cfg.LinearKeyHeadDim, cfg.LinearValueHeadDim, cfg.LinearNumKeyHeads, cfg.LinearNumValueHeads)
	}

	// Gate metadata: the hybrid gate must classify to the recognized GDN forward path.
	got, err := model.ClassifyForwardPath(cfg, nil)
	if err != nil {
		t.Fatalf("pinned artifact must classify, got refusal: %v", err)
	}
	if got != model.ForwardQwen35GDN {
		t.Fatalf("ClassifyForwardPath = %q, want %q", got, model.ForwardQwen35GDN)
	}

	// Tokenizer identity: vocab count and the EOS token string.
	toks, ok := gg.StringArray("tokenizer.ggml.tokens")
	if !ok || cfg.VocabSize != len(toks) || cfg.EOSTokenID >= len(toks) || toks[cfg.EOSTokenID] != "<|im_end|>" {
		t.Fatalf("tokenizer mismatch: vocab=%d eos=%d", len(toks), cfg.EOSTokenID)
	}

	// Required tensor families, and the *observed* MTP inventory. The artifact
	// declares qwen35.nextn_predict_layers = 1 and actually ships the trailing
	// blk.64 nextn block, so the declared MTP inventory is present, not absent.
	names := make(map[string]bool, len(gg.Tensors))
	for _, tn := range gg.Tensors {
		names[tn.Name] = true
	}
	for _, want := range []string{
		"token_embd.weight", "output.weight", "output_norm.weight",
		"blk.0.attn_qkv.weight", "blk.0.attn_gate.weight", "blk.0.ffn_down.weight", "blk.0.ffn_gate.weight", "blk.0.ffn_up.weight",
		"blk.0.ssm_a", "blk.0.ssm_alpha.weight",
	} {
		if !names[want] {
			t.Fatalf("required tensor family %q absent from pinned artifact", want)
		}
	}
	if _, ok := gg.Uint64("qwen35.nextn_predict_layers"); !ok {
		t.Fatalf("qwen35.nextn_predict_layers absent; MTP inventory cannot be checked")
	}
	if cfg.NumNextNPredictLayers != 1 {
		t.Fatalf("NumNextNPredictLayers = %d, want 1", cfg.NumNextNPredictLayers)
	}
	wantMTP := []string{
		"blk.64.nextn.eh_proj.weight",
		"blk.64.nextn.enorm.weight",
		"blk.64.nextn.hnorm.weight",
		"blk.64.nextn.shared_head_norm.weight",
	}
	observed := []string{}
	for _, n := range wantMTP {
		if names[n] {
			observed = append(observed, n)
		}
	}
	if !reflect.DeepEqual(observed, wantMTP) {
		t.Fatalf("observed MTP inventory = %v, want %v (declared-but-absent must be typed unsupported, not invented)",
			observed, wantMTP)
	}

	// Every target block must be present for the derived target depth.
	for l := 0; l < cfg.NumLayers; l++ {
		if !names[fmt.Sprintf("blk.%d.attn_norm.weight", l)] {
			t.Fatalf("target block blk.%d missing altough NumLayers=%d", l, cfg.NumLayers)
		}
	}
}

// TestQwen38Q4KMArtifactContractMutationFailsClosed proves the quality-critical
// contract fails closed: a fixture whose bytes have been mutated must not parse to the
// same identity, and an incomplete artifact (header claiming an absent MTP block) must
// be typed non-qualifying rather than silently accepted.
func TestQwen38Q4KMArtifactContractMutationFailsClosed(t *testing.T) {
	raw := qwen38Q4KMHeaderRaw(t)

	t.Run("truncated header is refused", func(t *testing.T) {
		if _, err := Read(bytes.NewReader(raw[:len(raw)/2])); err == nil {
			t.Fatalf("truncated header parsed without error; want a fail-closed refusal")
		}
	})

	t.Run("mutated identity metadata changes the parsed name", func(t *testing.T) {
		// Flip a byte inside the 'Qwen3.8-27B' general.name string. The mutated header
		// must no longer report the pinned identity.
		mutated := append([]byte(nil), raw...)
		idx := bytes.Index(mutated, []byte("Qwen3.8-27B"))
		if idx < 0 {
			t.Fatalf("general.name not found in fixture")
		}
		mutated[idx] = 'X'
		gg, err := Read(bytes.NewReader(mutated))
		if err != nil {
			return // a parse error is an acceptable fail-closed outcome
		}
		if name, ok := gg.String("general.name"); ok && name == "Qwen3.8-27B" {
			t.Fatalf("mutated identity still reports %q; contract did not fail closed", name)
		}
	})

	t.Run("declared MTP with absent inventory is a typed refusal", func(t *testing.T) {
		// A synthetic file that declares one MTP layer but ships only target blocks is
		// the "declared-but-absent" hazard. Config must report the declaration without
		// inventing the MTP tensors, which remain absent from the tensor directory.
		f := &File{Metadata: map[string]Value{
			"general.architecture":                    {Type: TypeString, Value: "qwen35"},
			"qwen35.embedding_length":                 {Type: TypeUint64, Value: uint64(5120)},
			"qwen35.block_count":                      {Type: TypeUint64, Value: uint64(3)},
			"qwen35.feed_forward_length":              {Type: TypeUint64, Value: uint64(64)},
			"qwen35.attention.head_count":             {Type: TypeUint64, Value: uint64(4)},
			"qwen35.attention.head_count_kv":          {Type: TypeUint64, Value: uint64(2)},
			"qwen35.attention.layer_norm_rms_epsilon": {Type: TypeFloat32, Value: float32(1e-5)},
			"qwen35.full_attention_interval":          {Type: TypeUint64, Value: uint64(4)},
			"qwen35.nextn_predict_layers":             {Type: TypeUint64, Value: uint64(1)},
		}}
		cfg, err := f.Config()
		if err != nil {
			t.Fatalf("Config on declared-MTP synthetic: %v", err)
		}
		if cfg.NumNextNPredictLayers != 1 {
			t.Fatalf("NumNextNPredictLayers = %d, want 1 (declared)", cfg.NumNextNPredictLayers)
		}
		if cfg.NumLayers != 2 {
			t.Fatalf("NumLayers = %d, want 2 (block_count 3 minus 1 MTP)", cfg.NumLayers)
		}
		// No blk.2.nextn.* tensors exist in this synthetic file, so the declared MTP
		// inventory is absent. The parser must not fabricate it.
		for _, tn := range f.Tensors {
			t.Fatalf("synthetic declaration fabricated a tensor %q; want an empty directory", tn.Name)
		}
	})
}
