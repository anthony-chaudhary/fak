package ggufload

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
)

func TestQwen3MoEGGUFExpertSplitE2E(t *testing.T) {
	const E, I, H = 2, 3, 2
	perGate := I * H
	perDown := H * I
	gate := sequenceF32ForTest(10, E*perGate)
	up := sequenceF32ForTest(100, E*perGate)
	down := sequenceF32ForTest(200, E*perDown)

	path := filepath.Join(t.TempDir(), "qwen3moe_experts.gguf")
	if err := os.WriteFile(path, qwen3MoEExpertGGUF(E, I, H, TensorF32, gate, up, down), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	ws, err := OpenWeights(path)
	if err != nil {
		t.Fatalf("OpenWeights: %v", err)
	}
	defer ws.Close()

	cfg, tensors, err := ws.F32Tensors()
	if err != nil {
		t.Fatalf("F32Tensors: %v", err)
	}
	if cfg.ModelType != "qwen3moe" {
		t.Fatalf("ModelType=%q, want qwen3moe", cfg.ModelType)
	}
	if !cfg.IsMoE() || cfg.NumExperts != E || cfg.NumExpertsPerTok != 1 || cfg.MoEIntermediateSize != I {
		t.Fatalf("MoE config = experts:%d topk:%d moeI:%d IsMoE:%v, want %d/1/%d/true",
			cfg.NumExperts, cfg.NumExpertsPerTok, cfg.MoEIntermediateSize, cfg.IsMoE(), E, I)
	}
	byName := map[string]modelTensorForTest{}
	for _, tt := range tensors {
		byName[tt.Name] = modelTensorForTest{shape: tt.Shape, data: tt.Data}
	}
	if len(byName) != 1+E*3 {
		t.Fatalf("loaded %d tensors, want router + %d per-expert tensors", len(byName), E*3)
	}
	assertModelTensorShapeForTest(t, byName, "model.layers.0.mlp.gate.weight", []int{E, H})
	for x := 0; x < E; x++ {
		p := "model.layers.0.mlp.experts." + itoaForTest(x) + "."
		assertModelTensorForTest(t, byName, p+"gate_proj.weight", []int{I, H}, gate[x*perGate:(x+1)*perGate])
		assertModelTensorForTest(t, byName, p+"up_proj.weight", []int{I, H}, up[x*perGate:(x+1)*perGate])
		assertModelTensorForTest(t, byName, p+"down_proj.weight", []int{H, I}, down[x*perDown:(x+1)*perDown])
	}
}

func TestQwen3MoEGGUFExpertSplitQ4KResident(t *testing.T) {
	const E, I, H = 2, 256, 256
	path := filepath.Join(t.TempDir(), "qwen3moe_q4k_experts.gguf")
	if err := os.WriteFile(path, qwen3MoEExpertGGUF(E, I, H, TensorQ4_K, nil, nil, nil), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	m, err := LoadModelQ4KProfile(path, nil)
	if err != nil {
		t.Fatalf("LoadModelQ4KProfile: %v", err)
	}
	if !m.Cfg.IsMoE() || m.Cfg.NumExperts != E || m.Cfg.MoEIntermediateSize != I {
		t.Fatalf("loaded cfg = experts:%d moeI:%d IsMoE:%v, want %d/%d/true",
			m.Cfg.NumExperts, m.Cfg.MoEIntermediateSize, m.Cfg.IsMoE(), E, I)
	}
	if got, want := m.Q4KCount(), E*3; got != want {
		t.Fatalf("resident Q4_K tensors = %d, want %d split experts", got, want)
	}
	for x := 0; x < E; x++ {
		for _, proj := range []string{"gate_proj", "up_proj", "down_proj"} {
			name := "model.layers.0.mlp.experts." + itoaForTest(x) + "." + proj + ".weight"
			if !m.HasQ4K(name) {
				t.Fatalf("resident Q4_K expert tensor %q missing", name)
			}
		}
	}
}

func TestQwen3MoEGGUFExpertShardLoadsOnlyResidentBand(t *testing.T) {
	const E, I, H = 4, 256, 256
	path := filepath.Join(t.TempDir(), "qwen3moe_q4k_expert_shard.gguf")
	if err := os.WriteFile(path, qwen3MoEExpertGGUF(E, I, H, TensorQ4_K, nil, nil, nil), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	shard, err := ExpertShardForRank(E, 2, 1)
	if err != nil {
		t.Fatalf("ExpertShardForRank: %v", err)
	}
	if shard.Lo != 2 || shard.Hi != 4 {
		t.Fatalf("rank-1 shard = [%d,%d), want [2,4)", shard.Lo, shard.Hi)
	}
	prof := NewLoadProfiler()
	m, err := LoadModelQ4KProfileOptions(path, prof, WithExpertShard(shard.Lo, shard.Hi))
	if err != nil {
		t.Fatalf("LoadModelQ4KProfileOptions: %v", err)
	}
	if got, want := m.Q4KCount(), 2*3; got != want {
		t.Fatalf("resident Q4_K tensors = %d, want %d split tensors for two local experts", got, want)
	}
	for x := 0; x < E; x++ {
		for _, proj := range []string{"gate_proj", "up_proj", "down_proj"} {
			name := "model.layers.0.mlp.experts." + itoaForTest(x) + "." + proj + ".weight"
			want := x >= shard.Lo && x < shard.Hi
			if got := m.HasQ4K(name); got != want {
				t.Fatalf("HasQ4K(%q)=%v, want %v for shard [%d,%d)", name, got, want, shard.Lo, shard.Hi)
			}
		}
	}
	var row *LoadPathStat
	for i := range prof.loadPathRows() {
		r := prof.loadPathRows()[i]
		if r.QuantType == "Q4_K" && r.Expert {
			row = &r
		}
	}
	if row == nil || row.ResidentTensors != 2*3 || row.DequantTensors != 0 {
		t.Fatalf("load-path breakdown for sharded Q4_K experts = %+v, want resident=%d dequant=0", row, 2*3)
	}
}

type qwen3MoETestTensor struct {
	name string
	dims []uint64
	typ  TensorType
	data []float32
}

func qwen3MoEExpertGGUF(E, I, H int, expertType TensorType, gate, up, down []float32) []byte {
	tensors := []qwen3MoETestTensor{
		{name: "blk.0.ffn_down_exps.weight", dims: []uint64{uint64(I), uint64(H), uint64(E)}, typ: expertType, data: down},
		{name: "blk.0.ffn_gate_exps.weight", dims: []uint64{uint64(H), uint64(I), uint64(E)}, typ: expertType, data: gate},
		{name: "blk.0.ffn_gate_inp.weight", dims: []uint64{uint64(H), uint64(E)}, typ: TensorF32, data: sequenceF32ForTest(300, E*H)},
		{name: "blk.0.ffn_up_exps.weight", dims: []uint64{uint64(H), uint64(I), uint64(E)}, typ: expertType, data: up},
	}
	offsets := make([]uint64, len(tensors))
	off := 0
	for i, tt := range tensors {
		offsets[i] = uint64(off)
		off += qwen3MoEPayloadBytes(tt.typ, tt.dims)
		off = (off + 31) / 32 * 32
	}

	var b bytes.Buffer
	writeMinimalHeader(&b, uint64(len(tensors)), 10)
	writeKVString(&b, "general.architecture", "qwen3moe")
	writeKVUint32(&b, "general.alignment", 32)
	writeKVUint32(&b, "qwen3moe.embedding_length", uint32(H))
	writeKVUint32(&b, "qwen3moe.block_count", 1)
	writeKVUint32(&b, "qwen3moe.attention.head_count", 1)
	writeKVUint32(&b, "qwen3moe.feed_forward_length", uint32(maxForTest(I*2, I)))
	writeKVFloat32(&b, "qwen3moe.attention.layer_norm_rms_epsilon", 1e-6)
	writeKVUint32(&b, "qwen3moe.expert_count", uint32(E))
	writeKVUint32(&b, "qwen3moe.expert_used_count", 1)
	writeKVUint32(&b, "qwen3moe.expert_feed_forward_length", uint32(I))
	for i, tt := range tensors {
		writeTensorInfoForTest(&b, tt.name, tt.dims, tt.typ, offsets[i])
	}
	padToAlignment(&b, 32)
	for _, tt := range tensors {
		switch tt.typ {
		case TensorF32:
			values := tt.data
			if values == nil {
				values = make([]float32, qwen3MoEValueCount(tt.dims))
			}
			if len(values) != qwen3MoEValueCount(tt.dims) {
				panic("bad qwen3moe test tensor data length")
			}
			for _, v := range values {
				writeF32ForTest(&b, v)
			}
		case TensorQ4_K:
			b.Write(make([]byte, qwen3MoEPayloadBytes(tt.typ, tt.dims)))
		default:
			panic("unsupported qwen3moe test tensor type")
		}
		padToAlignment(&b, 32)
	}
	return b.Bytes()
}

func qwen3MoEPayloadBytes(typ TensorType, dims []uint64) int {
	n := qwen3MoEValueCount(dims)
	switch typ {
	case TensorF32:
		return n * 4
	case TensorQ4_K:
		return n / qkK * blockQ4KBytes
	default:
		panic("unsupported qwen3moe test tensor type")
	}
}

func qwen3MoEValueCount(dims []uint64) int {
	n := 1
	for _, d := range dims {
		n *= int(d)
	}
	return n
}

func maxForTest(a, b int) int {
	if a > b {
		return a
	}
	return b
}
