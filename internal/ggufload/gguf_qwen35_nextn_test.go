package ggufload

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
)

// synthQwen35Meta is the minimal qwen35 metadata (*File).Config() accepts, shaped like the
// real Qwen3.6-27B Q4_K_M scaled down: block_count counts the trailing NextN/MTP draft
// block(s), and nextn_predict_layers declares how many ride at the tail.
func synthQwen35Meta(blocks, nextn uint64) map[string]Value {
	return map[string]Value{
		"general.architecture":                    {Type: TypeString, Value: "qwen35"},
		"qwen35.context_length":                   {Type: TypeUint64, Value: uint64(16)},
		"qwen35.embedding_length":                 {Type: TypeUint64, Value: uint64(32)},
		"qwen35.block_count":                      {Type: TypeUint64, Value: blocks},
		"qwen35.feed_forward_length":              {Type: TypeUint64, Value: uint64(64)},
		"qwen35.attention.head_count":             {Type: TypeUint64, Value: uint64(4)},
		"qwen35.attention.head_count_kv":          {Type: TypeUint64, Value: uint64(2)},
		"qwen35.attention.layer_norm_rms_epsilon": {Type: TypeFloat32, Value: float32(1e-5)},
		"qwen35.full_attention_interval":          {Type: TypeUint64, Value: uint64(4)},
		"qwen35.nextn_predict_layers":             {Type: TypeUint64, Value: nextn},
	}
}

// TestQwen35NextNExcludedFromNumLayers pins the block-count contract the real Qwen3.6-27B
// GGUF witnessed on 2026-07-08: block_count=65 INCLUDES the single NextN/MTP draft block
// (blk.64.nextn.* glue tensors only), which the text forward never runs and the loader
// drops. Config() must therefore subtract nextn_predict_layers from NumLayers before the
// LayerTypes schedule derives, or the built model demands
// model.layers.64.linear_attn.in_proj_qkv.weight and panics ("q8 tensor not built").
func TestQwen35NextNExcludedFromNumLayers(t *testing.T) {
	f := &File{Metadata: synthQwen35Meta(5, 1)}
	cfg, err := f.Config()
	if err != nil {
		t.Fatalf("Config: %v", err)
	}
	if cfg.NumLayers != 4 {
		t.Fatalf("NumLayers=%d, want 4 (block_count=5 minus nextn_predict_layers=1)", cfg.NumLayers)
	}
	if len(cfg.LayerTypes) != 4 {
		t.Fatalf("LayerTypes=%v, want 4 entries (schedule must derive AFTER the subtraction)", cfg.LayerTypes)
	}
	if cfg.LayerTypes[3] != "full_attention" {
		t.Fatalf("LayerTypes[3]=%q, want full_attention (interval 4)", cfg.LayerTypes[3])
	}
}

// TestQwen35NextNAbsurdCountIgnored guards the subtraction: a nextn_predict_layers claim
// that would consume the whole stack (or more) is metadata corruption, not a schedule —
// keep block_count as-is rather than deriving a zero/negative-layer model.
func TestQwen35NextNAbsurdCountIgnored(t *testing.T) {
	f := &File{Metadata: synthQwen35Meta(4, 4)}
	cfg, err := f.Config()
	if err != nil {
		t.Fatalf("Config: %v", err)
	}
	if cfg.NumLayers != 4 {
		t.Fatalf("NumLayers=%d, want 4 (nextn_predict_layers >= block_count must be ignored)", cfg.NumLayers)
	}
}

// TestQwen35NextNTensorsSkippedFromLoadAndClassify pins the OTHER half of the NextN contract,
// witnessed failing on da33 2026-07-10: the real Qwen3.6-27B Q4_K_M died at 97% of load with
// "gguf: no canonical mapping for tensor blk.64.nextn.eh_proj.weight". Subtracting the draft
// block from NumLayers (above) is not enough — the materializing loader and the CPU-offload
// classifier must also SKIP the trailing blk.<L>.nextn.* glue tensors for the qwen35 family,
// exactly as they do for GLM-5.2/DeepSeek, instead of rejecting the checkpoint.
func TestQwen35NextNTensorsSkippedFromLoadAndClassify(t *testing.T) {
	const nextn = "blk.64.nextn.eh_proj.weight"
	for _, arch := range []string{"qwen35", "qwen35moe"} {
		if !archShipsMTPOrVisionSidecar(arch) {
			t.Fatalf("archShipsMTPOrVisionSidecar(%q)=false, want true", arch)
		}
		if !glmMoeDsaSkipGGUFTensorForType(arch, nextn) {
			t.Fatalf("glmMoeDsaSkipGGUFTensorForType(%q, %s)=false, want true (byte-accounting drop)", arch, nextn)
		}
		hostExpert, err := tensorCPUOffloadExpert(nextn, arch)
		if err != nil {
			t.Fatalf("tensorCPUOffloadExpert(%s, %q) must classify, not error: %v", nextn, arch, err)
		}
		if hostExpert {
			t.Fatalf("tensorCPUOffloadExpert(%s, %q)=true, want false (device weight, never offloaded)", nextn, arch)
		}
	}
	// A plain dense arch without the sidecar contract keeps the strict mapping error.
	if archShipsMTPOrVisionSidecar("llama") {
		t.Fatalf("archShipsMTPOrVisionSidecar(llama)=true, want false")
	}
}

// writeQwen35NextNFixture builds a minimal on-disk qwen35 GGUF whose ONLY tensor is the
// trailing NextN/MTP glue tensor the real Qwen3.6-27B ships — the exact tensor the
// witnessed load rejected. Metadata mirrors synthQwen35Meta plus general.alignment.
func writeQwen35NextNFixture() []byte {
	const align = 32
	payload := f32Payload(0.25, -0.75)
	var b bytes.Buffer
	writeMinimalHeader(&b, 1, 11)
	writeKVUint32(&b, "general.alignment", align)
	writeKVString(&b, "general.architecture", "qwen35")
	writeKVUint64(&b, "qwen35.context_length", 16)
	writeKVUint64(&b, "qwen35.embedding_length", 32)
	writeKVUint64(&b, "qwen35.block_count", 5)
	writeKVUint64(&b, "qwen35.feed_forward_length", 64)
	writeKVUint64(&b, "qwen35.attention.head_count", 4)
	writeKVUint64(&b, "qwen35.attention.head_count_kv", 2)
	writeKVFloat32(&b, "qwen35.attention.layer_norm_rms_epsilon", 1e-5)
	writeKVUint64(&b, "qwen35.full_attention_interval", 4)
	writeKVUint64(&b, "qwen35.nextn_predict_layers", 1)
	writeTensorInfoForTest(&b, "blk.64.nextn.eh_proj.weight", []uint64{2}, TensorF32, 0)
	padToAlignment(&b, align)
	b.Write(payload)
	return b.Bytes()
}

// TestQwen35NextNTensorDroppedByWeightSource pins the LIVE loader loop the first fix slice
// missed (witnessed q36safe5, 2026-07-10): gguf_weightsource's materializing loops gated the
// .nextn. drop on the MLA+MoE family, so a real qwen35 checkpoint still died on
// "no canonical mapping for tensor blk.64.nextn.eh_proj.weight" AFTER the classifier and the
// q4k computeFn were fixed. A qwen35 file whose only tensor is the NextN glue tensor must
// materialize cleanly to zero tensors — not reject the checkpoint.
func TestQwen35NextNTensorDroppedByWeightSource(t *testing.T) {
	p := filepath.Join(t.TempDir(), "qwen35_nextn.gguf")
	if err := os.WriteFile(p, writeQwen35NextNFixture(), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	ws, err := OpenWeights(p)
	if err != nil {
		t.Fatalf("OpenWeights: %v", err)
	}
	defer ws.Close()
	cfg, tensors, err := ws.F32Tensors()
	if err != nil {
		t.Fatalf("F32Tensors must drop the qwen35 NextN sidecar, not reject the checkpoint: %v", err)
	}
	if cfg.ModelType != "qwen35" {
		t.Fatalf("ModelType=%q, want qwen35", cfg.ModelType)
	}
	if len(tensors) != 0 {
		names := make([]string, 0, len(tensors))
		for _, tt := range tensors {
			names = append(names, tt.Name)
		}
		t.Fatalf("tensors=%v, want none (the NextN glue tensor must be dropped)", names)
	}
}
