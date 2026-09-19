package ggufload

import (
	"fmt"
	"strings"
	"testing"
)

// ds41AttentionPreflightMeta is the smallest metadata-only deepseek41 header that
// carries the V4.1 attention axes the descriptor check derives its expectations
// from. It mirrors the axes applyDeepSeek41Config reads off the real artifact
// (vcruz Q2_K) but ships no payload reader.
func ds41AttentionPreflightMeta(numLayers int) map[string]Value {
	const arch = "deepseek41"
	p := arch + "."
	return map[string]Value{
		"general.architecture":                  {Type: TypeString, Value: arch},
		p + "embedding_length":                  {Type: TypeUint64, Value: uint64(5120)},
		p + "block_count":                       {Type: TypeUint64, Value: uint64(numLayers)},
		p + "attention.head_count":              {Type: TypeUint64, Value: uint64(128)},
		p + "attention.layer_norm_rms_epsilon":  {Type: TypeFloat32, Value: float32(1e-6)},
		p + "expert_count":                      {Type: TypeUint64, Value: uint64(384)},
		p + "expert_used_count":                 {Type: TypeUint64, Value: uint64(6)},
		p + "expert_feed_forward_length":        {Type: TypeUint64, Value: uint64(2304)},
		p + "expert_shared_count":               {Type: TypeUint64, Value: uint64(1)},
		p + "expert_shared_feed_forward_length": {Type: TypeUint64, Value: uint64(2304)},

		p + "attention.q_lora_rank":      {Type: TypeUint64, Value: uint64(1280)},
		p + "attention.kv_lora_rank":     {Type: TypeUint64, Value: uint64(512)},
		p + "attention.key_length_mla":   {Type: TypeUint64, Value: uint64(512)},
		p + "attention.value_length_mla": {Type: TypeUint64, Value: uint64(512)},
		p + "attention.qk_nope_head_dim": {Type: TypeUint64, Value: uint64(448)},
		p + "attention.qk_rope_head_dim": {Type: TypeUint64, Value: uint64(64)},

		p + "attention.output_group_count": {Type: TypeUint64, Value: uint64(16)},
		p + "attention.output_lora_rank":   {Type: TypeUint64, Value: uint64(256)},
	}
}

// ds41AttentionDims returns the GGUF dims (outer-first, i.e. the order a real GGUF
// header stores) for the named V4.1 attention operand given the derived config.
// The GGUF directory lists dims innermost-first, so the caller passes the logical
// [out, in] and this helper reverses it.
func ds41AttentionGGUFDims(out, in int) []uint64 {
	return []uint64{uint64(in), uint64(out)}
}

// ds41AttentionOperands builds the full per-layer attention operand inventory for a
// fixture with numLayers layers. groupMAJOR selects the artifact's group-major wo_a
// declaration; the flat form is the reduced-fixture declaration.
func ds41AttentionOperands(numLayers int, groupMajor bool) []TensorInfo {
	const (
		H       = 5120
		nH      = 128
		hd      = 512 // qk_nope 448 + qk_rope 64
		qLora   = 1280
		kvLora  = 512
		oGroups = 16
		oLora   = 256
	)
	var out []TensorInfo
	for l := 0; l < numLayers; l++ {
		pfx := fmt.Sprintf("blk.%d.", l)
		out = append(out,
			TensorInfo{Name: pfx + "attn_q_a.weight", Dims: ds41AttentionGGUFDims(qLora, H), Type: TensorQ2_K},
			TensorInfo{Name: pfx + "attn_q_b.weight", Dims: ds41AttentionGGUFDims(nH*hd, qLora), Type: TensorQ2_K},
			TensorInfo{Name: pfx + "attn_q_a_norm.weight", Dims: []uint64{uint64(qLora)}, Type: TensorF32},
			TensorInfo{Name: pfx + "attn_kv.weight", Dims: ds41AttentionGGUFDims(kvLora, H), Type: TensorQ2_K},
			TensorInfo{Name: pfx + "attn_kv_a_norm.weight", Dims: []uint64{uint64(kvLora)}, Type: TensorF32},
			TensorInfo{Name: pfx + "attn_sinks.weight", Dims: []uint64{uint64(nH)}, Type: TensorF32},
			TensorInfo{Name: pfx + "attn_output_b.weight", Dims: ds41AttentionGGUFDims(H, oGroups*oLora), Type: TensorQ2_K},
		)
		if groupMajor {
			out = append(out, TensorInfo{
				Name: pfx + "attn_output_a.weight",
				Dims: ds41AttentionGGUFDims(oGroups*oLora, (nH/oGroups)*hd),
				Type: TensorQ2_K,
			})
		} else {
			out = append(out, TensorInfo{
				Name: pfx + "attn_output_a.weight",
				Dims: ds41AttentionGGUFDims(oLora, nH*hd),
				Type: TensorQ2_K,
			})
		}
	}
	return out
}

// ds41AttentionWeightSource assembles a metadata-only WeightSource over the given
// tensors, wiring the package's countingReaderAt over a nil inner reader. Any
// payload ReadAt would dereference the nil reader and panic, so the metadata-only
// property is enforced by construction, not merely asserted; the read counter is
// returned for an explicit zero-reads check.
func ds41AttentionWeightSource(t *testing.T, meta map[string]Value, tensors []TensorInfo) (*WeightSource, *countingReaderAt) {
	t.Helper()
	r := &countingReaderAt{}
	ws, err := NewWeightSource(&File{Metadata: meta, Tensors: tensors}, r, 0)
	if err != nil {
		t.Fatalf("NewWeightSource: %v", err)
	}
	return ws, r
}

// TestDeepSeek41AttentionDescriptorPreflight is the acceptance witness for #13320:
// BuildModelPreflight aggregates per-layer V4.1 attention descriptor failures into
// one stable, layer/name-sorted report BEFORE any payload byte is read.
func TestDeepSeek41AttentionDescriptorPreflight(t *testing.T) {
	t.Run("intact artifact metadata is accepted with zero payload reads", func(t *testing.T) {
		const layers = 40
		ws, r := ds41AttentionWeightSource(t, ds41AttentionPreflightMeta(layers), ds41AttentionOperands(layers, true))
		pf := BuildModelPreflight(PreflightInput{Path: "v41.gguf", Source: ws, Lean: true})
		if pf.Refused() {
			t.Fatalf("intact V4.1 attention descriptors refused: verdict=%s reason=%q", pf.Verdict, pf.Reason)
		}
		if pf.Verdict != PreflightReady {
			t.Fatalf("verdict = %s, want READY (reason: %s)", pf.Verdict, pf.Reason)
		}
		if r.reads.Load() != 0 {
			t.Fatalf("payload bytes read during attention descriptor preflight: %d ReadAt call(s)", r.reads.Load())
		}
	})

	t.Run("mutations at layer 0 and layer 39 aggregate into one sorted report", func(t *testing.T) {
		const layers = 40
		operands := ds41AttentionOperands(layers, true)
		// Mutate layer 0's wq_a to the wrong in-dim and layer 39's wkv to the wrong
		// out-dim. Both must land in ONE report before the byte estimate.
		mutate := func(name string, dims []uint64) {
			for i := range operands {
				if operands[i].Name == name {
					operands[i].Dims = dims
					return
				}
			}
			t.Fatalf("fixture is missing %s to mutate", name)
		}
		mutate("blk.0.attn_q_a.weight", ds41AttentionGGUFDims(1280, 4096))
		mutate("blk.39.attn_kv.weight", ds41AttentionGGUFDims(512, 4096))

		ws, r := ds41AttentionWeightSource(t, ds41AttentionPreflightMeta(layers), operands)
		pf := BuildModelPreflight(PreflightInput{Path: "v41.gguf", Source: ws, Lean: true})
		if pf.Verdict != PreflightRefuseHeader {
			t.Fatalf("verdict = %s, want REFUSE_BAD_HEADER", pf.Verdict)
		}
		if !pf.Refused() {
			t.Fatalf("a descriptor refusal must report Refused()")
		}
		for _, want := range []string{
			"model.layers.0.attn.wq_a.weight",
			"model.layers.39.attn.wkv.weight",
		} {
			if !strings.Contains(pf.Reason, want) {
				t.Fatalf("reason must name %q; got:\n%s", want, pf.Reason)
			}
		}
		// The report is stable and sorted: the layer-0 diagnostic precedes layer-39.
		i0 := strings.Index(pf.Reason, "model.layers.0.attn.wq_a.weight")
		i39 := strings.Index(pf.Reason, "model.layers.39.attn.wkv.weight")
		if i0 < 0 || i39 < 0 || i0 > i39 {
			t.Fatalf("report must be layer-sorted; got:\n%s", pf.Reason)
		}
		if r.reads.Load() != 0 {
			t.Fatalf("payload bytes read while reporting descriptor failures: %d ReadAt call(s)", r.reads.Load())
		}

		// Determinism: the same input yields the byte-identical report.
		ws2, _ := ds41AttentionWeightSource(t, ds41AttentionPreflightMeta(layers), ds41AttentionOperands(layers, true))
		_ = ws2
		pf2 := BuildModelPreflight(PreflightInput{Path: "v41.gguf", Source: ws, Lean: true})
		if pf2.Reason != pf.Reason {
			t.Fatalf("attention descriptor report is not deterministic:\n a=%q\n b=%q", pf.Reason, pf2.Reason)
		}
	})

	t.Run("grouped and flat wo_a declarations are both admitted; a bad one is named", func(t *testing.T) {
		const layers = 2
		// Flat declaration (reduced-fixture form) must be accepted.
		wsFlat, _ := ds41AttentionWeightSource(t, ds41AttentionPreflightMeta(layers), ds41AttentionOperands(layers, false))
		if pf := BuildModelPreflight(PreflightInput{Source: wsFlat, Lean: true}); pf.Refused() {
			t.Fatalf("flat wo_a declaration refused: verdict=%s reason=%q", pf.Verdict, pf.Reason)
		}
		// A wo_a whose axes are individually consistent but whose total disagrees
		// must be named, not silently accepted.
		bad := ds41AttentionOperands(layers, false)
		for i := range bad {
			if bad[i].Name == "blk.0.attn_output_a.weight" {
				bad[i].Dims = ds41AttentionGGUFDims(256, 128*511) // one head short
			}
		}
		wsBad, _ := ds41AttentionWeightSource(t, ds41AttentionPreflightMeta(layers), bad)
		pf := BuildModelPreflight(PreflightInput{Source: wsBad, Lean: true})
		if pf.Verdict != PreflightRefuseHeader {
			t.Fatalf("a grouped wo_a total mismatch must refuse; verdict=%s reason=%q", pf.Verdict, pf.Reason)
		}
		if !strings.Contains(pf.Reason, "model.layers.0.attn.wo_a.weight") {
			t.Fatalf("reason must name the offending wo_a tensor; got:\n%s", pf.Reason)
		}
	})

	t.Run("a missing attention operand is named", func(t *testing.T) {
		const layers = 2
		operands := ds41AttentionOperands(layers, true)
		trimmed := operands[:0]
		for _, ti := range operands {
			// The GGUF name for the compressed-KV projection; its canonical
			// name is model.layers.1.attn.wkv.weight.
			if ti.Name == "blk.1.attn_kv.weight" {
				continue
			}
			trimmed = append(trimmed, ti)
		}
		if len(trimmed) != len(operands)-1 {
			t.Fatalf("fixture trim removed %d tensors, want 1", len(operands)-len(trimmed))
		}
		ws, _ := ds41AttentionWeightSource(t, ds41AttentionPreflightMeta(layers), trimmed)
		pf := BuildModelPreflight(PreflightInput{Source: ws, Lean: true})
		if pf.Verdict != PreflightRefuseHeader {
			t.Fatalf("a missing attention operand must refuse; verdict=%s reason=%q", pf.Verdict, pf.Reason)
		}
		if !strings.Contains(pf.Reason, "model.layers.1.attn.wkv.weight") {
			t.Fatalf("reason must name the missing tensor; got:\n%s", pf.Reason)
		}
	})

	t.Run("non-V4.1 preflight is unchanged", func(t *testing.T) {
		ws := readyWeightSource(t)
		pf := BuildModelPreflight(PreflightInput{Path: "x.gguf", Source: ws, Lean: true})
		if pf.Verdict != PreflightReady {
			t.Fatalf("llama preflight must stay READY; verdict=%s reason=%q", pf.Verdict, pf.Reason)
		}
	})
}
