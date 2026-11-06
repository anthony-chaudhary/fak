package ggufload

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
)

// deepseek41_cross_shard_layer_test.go — host-only guard for the DS4.1 Q2_K
// "layer 16" shard geometry (fak-private ds41-5090 DIVERGENCE.md).
//
// In the published 7-shard DS4.1 Flash Q2_K artifact, layers 16/25/34 split one
// layer's MoE tensors across two shard files: ffn_down_exps, ffn_down_shexp and
// exp_probs_b (router bias) sit in shard N while ffn_gate_inp, ffn_gate_exps,
// ffn_up_exps, the shared gate/up and ffn_norm sit in shard N+1. The native
// forward first leaves the reference at layer 16 moe_sum, so a loader that cached
// one reader/base offset per layer (from the first tensor it saw) would have been
// a plausible cause. This pins that every such tensor resolves through its OWN
// shard reader at its OWN offset, both by name (TensorBytes, the dense/router
// path) and on the streamed routed-expert path (FusedExpertTensors descriptors ->
// ExpertCheckpointTier), with distinct per-tensor payloads so a reader or offset
// swap between the two shards cannot read back equal bytes.
func TestDeepSeek41CrossShardLayerResolvesEveryTensorFromItsOwnShard(t *testing.T) {
	const E, I, H = 4, 256, 256
	dimsGate := []uint64{uint64(H), uint64(I), uint64(E)} // [in, out, E] GGUF order
	dimsDown := []uint64{uint64(I), uint64(H), uint64(E)}

	// patterned fills a payload with a tensor-specific byte stream. salt keeps two
	// tensors at the same in-shard offset (shard 2 offset 0 vs shard 3 offset 0)
	// from carrying equal bytes, and the per-expert term keeps experts distinct.
	patterned := func(n, experts, salt int) []byte {
		b := make([]byte, n)
		per := n / experts
		for i := range b {
			b[i] = byte((salt*197 + (i/per)*131 + i*7 + 17) & 0xff)
		}
		return b
	}
	q4k := func(dims []uint64, salt int) []byte {
		return patterned(qwen3MoEPayloadBytes(TensorQ4_K, dims), E, salt)
	}
	f32n := func(n int, base float32) []byte {
		vs := make([]float32, n)
		for i := range vs {
			vs[i] = base + float32(i)*0.25
		}
		return f32Payload(vs...)
	}

	// Shard 2 ("00004"): down projections + router bias, exactly as layer 16.
	shard2 := []splitTensor{
		{name: "blk.0.ffn_down_exps.weight", dims: dimsDown, typ: TensorQ4_K, data: q4k(dimsDown, 1)},
		{name: "blk.0.ffn_down_shexp.weight", dims: []uint64{8}, typ: TensorF32, data: f32n(8, -3)},
		{name: "blk.0.exp_probs_b.bias", dims: []uint64{E}, typ: TensorF32, data: f32n(E, 0.5)},
	}
	// Shard 3 ("00005"): router weight, gate/up experts, shared gate/up, ffn_norm.
	// ffn_gate_exps is FIRST so it shares in-shard offset 0 with shard 2's down slab.
	shard3 := []splitTensor{
		{name: "blk.0.ffn_gate_exps.weight", dims: dimsGate, typ: TensorQ4_K, data: q4k(dimsGate, 2)},
		{name: "blk.0.ffn_up_exps.weight", dims: dimsGate, typ: TensorQ4_K, data: q4k(dimsGate, 3)},
		{name: "blk.0.ffn_gate_inp.weight", dims: []uint64{8, E}, typ: TensorF32, data: f32n(8*E, 7)},
		{name: "blk.0.ffn_gate_shexp.weight", dims: []uint64{8}, typ: TensorF32, data: f32n(8, 11)},
		{name: "blk.0.ffn_up_shexp.weight", dims: []uint64{8}, typ: TensorF32, data: f32n(8, 13)},
		{name: "blk.0.ffn_norm.weight", dims: []uint64{4}, typ: TensorF32, data: f32n(4, 17)},
	}
	shard1 := []splitTensor{
		{name: "blk.0.attn_norm.weight", dims: []uint64{4}, typ: TensorF32, data: f32Payload(1.5, 2.5, 3.5, 4.5)},
	}
	total := uint32(len(shard1) + len(shard2) + len(shard3))

	dir := t.TempDir()
	p := func(n int) string { return filepath.Join(dir, "ds41-0000"+itoaForTest(n)+"-of-00003.gguf") }
	for n, body := range map[int][]byte{
		1: writeDeepSeek41SplitShard(t, 1, 3, total, true, shard1),
		2: writeDeepSeek41SplitShard(t, 2, 3, total, false, shard2),
		3: writeDeepSeek41SplitShard(t, 3, 3, total, false, shard3),
	} {
		if err := os.WriteFile(p(n), body, 0o644); err != nil {
			t.Fatalf("write shard %d: %v", n, err)
		}
	}

	ws, err := OpenWeights(p(1))
	if err != nil {
		t.Fatalf("OpenWeights: %v", err)
	}
	defer ws.Close()

	// (a) Name-addressed reads (router bias, router weight, shared expert, norms,
	// and the whole expert slabs) return exactly the bytes their shard holds.
	want := make(map[string][]byte)
	for _, set := range [][]splitTensor{shard1, shard2, shard3} {
		for _, tt := range set {
			want[tt.name] = tt.data
			got, _, err := ws.TensorBytes(tt.name)
			if err != nil {
				t.Fatalf("TensorBytes(%s): %v", tt.name, err)
			}
			if !bytes.Equal(got, tt.data) {
				t.Errorf("TensorBytes(%s) read %d bytes that differ from the payload its shard holds", tt.name, len(got))
			}
		}
	}

	// (b) Streamed routed-expert descriptors: each slab is grouped under the reader
	// of the shard that holds it, and reading expert e's stride through that
	// descriptor returns expert e's bytes from the right file and offset.
	shards, err := ws.FusedExpertTensors()
	if err != nil {
		t.Fatalf("FusedExpertTensors: %v", err)
	}
	readerOf := make(map[string]any)
	var described int
	for _, sh := range shards {
		for _, f := range sh.Fused {
			described++
			readerOf[f.Name] = sh.Reader
			slab := want[f.Name]
			if slab == nil {
				t.Fatalf("descriptor for unexpected tensor %s", f.Name)
			}
			stride := len(slab) / f.Experts
			for e := 0; e < f.Experts; e++ {
				got := make([]byte, stride)
				if _, err := sh.Reader.ReadAt(got, f.Offset+int64(e*stride)); err != nil {
					t.Fatalf("%s expert %d ReadAt: %v", f.Name, e, err)
				}
				if !bytes.Equal(got, slab[e*stride:(e+1)*stride]) {
					t.Errorf("%s expert %d: descriptor (reader, offset %d) does not address this expert's bytes", f.Name, e, f.Offset)
				}
			}
		}
	}
	if described != 3 {
		t.Fatalf("described %d routed-expert slabs, want 3 (gate/up/down)", described)
	}
	if readerOf["blk.0.ffn_down_exps.weight"] == readerOf["blk.0.ffn_gate_exps.weight"] {
		t.Error("down and gate expert slabs live in different shards but were grouped under one reader")
	}
	if readerOf["blk.0.ffn_gate_exps.weight"] != readerOf["blk.0.ffn_up_exps.weight"] {
		t.Error("gate and up expert slabs share a shard but were grouped under different readers")
	}

	// (c) The tier built over those descriptors indexes every per-expert leaf the
	// V4.1 forward asks for, so no projection silently falls back to another shard.
	tier, err := ws.ExpertCheckpointTier(0)
	if err != nil || tier == nil {
		t.Fatalf("ExpertCheckpointTier: tier=%v err=%v", tier, err)
	}
	for e := 0; e < E; e++ {
		for _, leaf := range []string{"w1", "w3", "w2"} {
			if name := "model.layers.0.ffn.experts." + itoaForTest(e) + "." + leaf + ".weight"; !tier.Has(name) {
				t.Errorf("tier does not index %s", name)
			}
		}
	}
}
