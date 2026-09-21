package ggufload

// deepseek41_expert_hal_fixture_test.go — the fixture half of the fak#13359
// witness ("DeepSeek V4.1 mixed-quant loader reaches the shared expert HAL").
//
// It emits a REAL single-file deepseek41 GGUF carrying the reduced V4.1 tensor
// inventory the native non-MLA forward admits (internal/model/v41_forward.go
// v41ForwardAdmitted), with the routed-expert leaves written as a MIXED slate:
// Q2_K gate/up (the DeepSeek-V4.1-Flash routed-expert quant, which the checkpoint
// tier stages) and Q3_K down (the kind with no device kernel, which the checkpoint
// tier stages but the device MatMul cannot serve). The geometry is the
// block-aligned reduced one (H = I = 256 = qkK) so a k-quant super-block divides
// every routed-expert reduction row.
//
// The point of a FIXTURE rather than the #13358 in-memory model: the expert
// bytes must enter through the REAL production loader
// (WeightSource.FusedExpertTensors -> buildExpertCheckpointTier ->
// model.Model.SetExpertCheckpoint), never through private resident-map
// assignment, so the streamed-expert seam is exercised exactly as a serve loads
// the published artifact. The companion
// deepseek41_expert_hal_integration_test.go drives the loaded model.

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// deepseek41ExpertHALGeometry is the reduced, block-aligned V4.1 fixture geometry.
// H and I are 256 (= qkK) so a Q2_K/Q3_K super-block (256 weights) divides every
// routed projection row; every other axis is the small reduced one the native
// forward admits. The routed-expert count is model.V41RouterExperts (the admitted
// MoE envelope) at model.V41RouterTopK picks per token.
const (
	ds41HALHidden    = 256
	ds41HALInterm    = 256
	ds41HALOutGroups = 2
	ds41HALOutRank   = 16
	ds41HALQLora     = 32
	ds41HALKVLora    = 32
	ds41HALQKNope    = 16
	ds41HALQKRope    = 16
	ds41HALHeads     = 2
	ds41HALVocab     = 8
)

// deepseek41ExpertHALGGUF assembles a real single-file deepseek41 GGUF carrying
// the reduced V4.1 inventory with a mixed-quant routed-expert slate. Every dense
// tensor is F32 (so the only quantized bytes are the routed experts, whose
// staged-vs-resident placement is the thing under test); the three routed-expert
// blobs are Q2_K gate, Q2_K up and Q3_K down, shaped [out,in,E] exactly as the
// GGUF batched-expert convention expects.
//
// expertPatternByte offsets each expert's bytes so a faulted expert is
// distinguishable from its neighbours, mirroring the tier's own Q2_K staging
// witness.
func deepseek41ExpertHALGGUF(t *testing.T) []byte {
	t.Helper()

	const E = model.V41RouterExperts
	const H, I = ds41HALHidden, ds41HALInterm

	var tensors []splitTensor
	// ds41HALFill mirrors internal/model.synthBuildRaw's fill so the fixture is a
	// well-conditioned model: norms and the mHC scale are exactly 1.0 (RMSNorm and
	// the mHC mix stay non-degenerate), the mHC base is 0.0, embeddings are wider,
	// and every matmul weight is a deterministic [-0.1,0.1] LCG draw. Without the
	// norm=1.0 rule a zero norm weight produces a non-finite router input.
	var seed = uint64(0x9E3779B97F4A7C15)
	next := func() float32 {
		seed = seed*6364136223846793005 + 1442695040888963407
		u := float32(seed>>40) / float32(1<<24)
		return u*2 - 1
	}
	fill := func(name string) float32 {
		switch {
		case strings.HasSuffix(name, "norm.weight"):
			return 1.0
		case strings.HasSuffix(name, "mhc_scale.weight"):
			return 1.0
		case strings.HasSuffix(name, "mhc_base.weight"):
			return 0.0
		case strings.Contains(name, "embed_tokens"):
			return next() * 0.2
		default:
			return next() * 0.1
		}
	}
	f32 := func(name string, dims ...uint64) {
		n := 1
		for _, d := range dims {
			n *= int(d)
		}
		data := make([]float32, n)
		for i := range data {
			data[i] = fill(name)
		}
		tensors = append(tensors, splitTensor{name: name, dims: dims, typ: TensorF32, data: f32Payload(data...)})
	}

	f32("token_embd.weight", H, ds41HALVocab)
	f32("output.weight", H, ds41HALVocab)
	f32("output_norm.weight", H)

	// Per-layer dense inventory. Names are the raw "blk.<L>.<suffix>" spellings
	// deepSeek41V41RealSuffixes pins; the loader maps each onto the native leaf.
	// GGUF dims are [in, out] (ne0 first); modelShapeFromGGUFDims reverses them to
	// the native [out, in] admission shape.
	const l = 0
	qHeadDim := uint64(ds41HALHeads * (ds41HALQKNope + ds41HALQKRope))
	oDim := uint64(ds41HALOutRank * ds41HALOutGroups)
	pre := "blk." + itoaForTest(l) + "."
	f32(pre+"attn_norm.weight", H)
	f32(pre+"ffn_norm.weight", H)
	// mHC hyper-connection coefficient block: mixes [v41MHCMixWidth, H], base
	// [v41MHCMixWidth], scale [3]. The forward's mHC stage requires all three.
	const mhcMixWidth = 24
	f32(pre+"mhc_mixes.weight", H, mhcMixWidth)
	f32(pre+"mhc_base.weight", mhcMixWidth)
	f32(pre+"mhc_scale.weight", 3)
	f32(pre+"attn_q_a.weight", H, ds41HALQLora)
	f32(pre+"attn_q_b.weight", ds41HALQLora, qHeadDim)
	f32(pre+"attn_kv.weight", H, ds41HALKVLora)
	f32(pre+"attn_kv_a_norm.weight", ds41HALKVLora)
	f32(pre+"attn_output_a.weight", qHeadDim, ds41HALOutRank)
	f32(pre+"attn_output_b.weight", oDim, H)
	f32(pre+"attn_sinks.weight", ds41HALHeads)
	f32(pre+"ffn_gate_inp.weight", H, uint64(E))
	f32(pre+"exp_probs_b.bias", uint64(E))
	f32(pre+"ffn_gate_shexp.weight", H, I)
	f32(pre+"ffn_up_shexp.weight", H, I)
	f32(pre+"ffn_down_shexp.weight", I, H)

	// The mixed-quant routed-expert slate: Q2_K gate, Q2_K up, Q3_K down.
	tensors = append(tensors,
		splitTensor{name: "blk.0.ffn_gate_exps.weight", dims: []uint64{H, I, uint64(E)}, typ: TensorQ2_K,
			data: ds41HALExpertSlab(TensorQ2_K, H, I, E)},
		splitTensor{name: "blk.0.ffn_up_exps.weight", dims: []uint64{H, I, uint64(E)}, typ: TensorQ2_K,
			data: ds41HALExpertSlab(TensorQ2_K, H, I, E)},
		splitTensor{name: "blk.0.ffn_down_exps.weight", dims: []uint64{I, H, uint64(E)}, typ: TensorQ3_K,
			data: ds41HALExpertSlab(TensorQ3_K, I, H, E)},
	)

	return writeDeepSeek41ExpertHALGGUF(t, tensors)
}

// ds41HALExpertSlab builds one batched routed-expert slab of the given k-quant
// kind, [out,in,E] with a distinct per-expert byte pattern whose super-block
// SCALES are pinned finite/nonzero: an arbitrary byte pattern would put Inf/NaN
// exponents in the f16 block scales and the dequant would produce non-finite
// weights. Q2_K pins d = 1.0 (byte 80) and min = 0 (byte 82); Q3_K pins
// d = 2^-6 (the last two bytes). Everything else is a deterministic LCG fill, so
// a faulted expert is distinguishable from its neighbours.
func ds41HALExpertSlab(typ TensorType, out, in, experts int) []byte {
	elems := out * in * experts
	var perBlock int
	switch typ {
	case TensorQ2_K:
		perBlock = blockQ2KBytes
	case TensorQ3_K:
		perBlock = blockQ3KBytes
	default:
		panic("ds41HALExpertSlab: unsupported quant")
	}
	total := elems / qkK * perBlock
	per := total / experts
	payload := make([]byte, total)
	var seed = uint64(0xD5EED1000 + uint64(typ))
	for x := 0; x < experts; x++ {
		for i := 0; i < per; i++ {
			seed = seed*6364136223846793005 + 1442695040888963407
			payload[x*per+i] = byte((seed >> 40) & 0xff)
		}
		// Pin each super-block's scale so the dequant is finite and nonzero.
		for b := 0; b < per/perBlock; b++ {
			blk := payload[x*per+b*perBlock:]
			switch typ {
			case TensorQ2_K:
				blk[80] = 0x00 // f16 1.0 = 0x3C00 little-endian
				blk[81] = 0x3C
				blk[82] = 0x00 // min = 0
				blk[83] = 0x00
			case TensorQ3_K:
				blk[perBlock-2] = 0x00 // f16 2^-6 = 0x2400 little-endian
				blk[perBlock-1] = 0x24
			}
		}
	}
	return payload
}

// writeDeepSeek41ExpertHALGGUF writes a single-file GGUF with the reduced V4.1
// metadata axes that reconstruct deepseek41ExpertHALGeometry through the
// production Config() derivation, plus the supplied tensor directory.
func writeDeepSeek41ExpertHALGGUF(t *testing.T, tensors []splitTensor) []byte {
	t.Helper()
	const align = 32
	const arch = "deepseek41"
	p := arch + "."

	kvs := []func(*bytes.Buffer){
		func(b *bytes.Buffer) { writeKVUint32(b, "general.alignment", align) },
		func(b *bytes.Buffer) { writeKVString(b, "general.architecture", arch) },
		func(b *bytes.Buffer) { writeKVUint64(b, p+"embedding_length", ds41HALHidden) },
		func(b *bytes.Buffer) { writeKVUint64(b, p+"block_count", 1) },
		func(b *bytes.Buffer) { writeKVUint64(b, p+"attention.head_count", ds41HALHeads) },
		func(b *bytes.Buffer) { writeKVFloat32(b, p+"attention.layer_norm_rms_epsilon", 1e-6) },
		func(b *bytes.Buffer) { writeKVUint64(b, p+"expert_count", model.V41RouterExperts) },
		func(b *bytes.Buffer) { writeKVUint64(b, p+"expert_used_count", model.V41RouterTopK) },
		func(b *bytes.Buffer) { writeKVUint64(b, p+"expert_feed_forward_length", ds41HALInterm) },
		func(b *bytes.Buffer) { writeKVUint64(b, p+"expert_shared_count", 1) },
		func(b *bytes.Buffer) { writeKVUint64(b, p+"expert_shared_feed_forward_length", ds41HALInterm) },
		func(b *bytes.Buffer) { writeKVFloat32(b, p+"expert_weights_scale", 1.5) },
		func(b *bytes.Buffer) { writeKVUint64(b, p+"attention.q_lora_rank", ds41HALQLora) },
		func(b *bytes.Buffer) { writeKVUint64(b, p+"attention.kv_lora_rank", ds41HALKVLora) },
		func(b *bytes.Buffer) { writeKVUint64(b, p+"attention.qk_nope_head_dim", ds41HALQKNope) },
		func(b *bytes.Buffer) { writeKVUint64(b, p+"attention.qk_rope_head_dim", ds41HALQKRope) },
		func(b *bytes.Buffer) { writeKVUint64(b, p+"attention.value_length_mla", ds41HALQKNope) },
		func(b *bytes.Buffer) { writeKVUint64(b, p+"attention.output_group_count", ds41HALOutGroups) },
		func(b *bytes.Buffer) { writeKVUint64(b, p+"attention.output_lora_rank", ds41HALOutRank) },
		func(b *bytes.Buffer) { writeKVUint64(b, p+"hyper_connection.count", 4) },
		func(b *bytes.Buffer) { writeKVFloat32(b, p+"hyper_connection.epsilon", 1e-6) },
		func(b *bytes.Buffer) {
			writeKVIntArrayForTest(b, p+"attention.compress_ratios", []int32{0})
		},
		func(b *bytes.Buffer) {
			writeKVStringArray(b, "tokenizer.ggml.tokens", []string{"a", "b", "c", "d", "e", "f", "g", "h"})
		},
	}

	offsets := make([]uint64, len(tensors))
	off := 0
	for i, tt := range tensors {
		offsets[i] = uint64(off)
		off = (off + len(tt.data) + align - 1) / align * align
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

// writeDeepSeek41ExpertHALFile writes the fixture to a temp dir and returns its path.
func writeDeepSeek41ExpertHALFile(t *testing.T) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "deepseek41-mixed-quant-hal.gguf")
	if err := os.WriteFile(path, deepseek41ExpertHALGGUF(t), 0o644); err != nil {
		t.Fatalf("write deepseek41 HAL fixture: %v", err)
	}
	return path
}
