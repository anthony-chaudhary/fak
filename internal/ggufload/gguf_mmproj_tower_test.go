package ggufload

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
)

// gguf_mmproj_tower_test.go covers WeightSource.VisionTower (#4029): reading a
// companion mmproj's v.*/mm.* tensors into a resident model.VisionTower and parsing
// its clip.* geometry. It builds a full one-layer CLIP tower fixture through the same
// byte writers the rest of the package uses — a real open → metadata → tensor read.

// writeMMProjTowerFixture builds a single-file CLIP mmproj GGUF carrying the given
// vision tensors plus a full clip.* geometry block.
func writeMMProjTowerFixture(t *testing.T, tensors []splitTensor) []byte {
	t.Helper()
	const align = 32
	var b bytes.Buffer
	// 7 KV: alignment + architecture + projector_type + embedding_length +
	// block_count + head_count + patch_size.
	writeMinimalHeader(&b, uint64(len(tensors)), 7)
	writeKVUint32(&b, "general.alignment", align)
	writeKVString(&b, "general.architecture", "clip")
	writeKVString(&b, "clip.projector_type", "mlp")
	writeKVUint32(&b, "clip.vision.embedding_length", 4)
	writeKVUint32(&b, "clip.vision.block_count", 1)
	writeKVUint32(&b, "clip.vision.attention.head_count", 2)
	writeKVUint32(&b, "clip.vision.patch_size", 14)
	off := uint64(0)
	for _, tt := range tensors {
		writeTensorInfoForTest(&b, tt.name, tt.dims, tt.typ, off)
		off += uint64((len(tt.data) + align - 1) / align * align)
	}
	padToAlignment(&b, align)
	for _, tt := range tensors {
		dataStart := b.Len()
		b.Write(tt.data)
		padToLen(&b, dataStart+align)
	}
	return b.Bytes()
}

// mmprojTowerTensors is a minimal but complete one-layer CLIP tower: the v.* globals,
// the mm.* projector, and the per-layer v.blk.0.* set.
func mmprojTowerTensors() []splitTensor {
	names := []string{
		"v.patch_embd.weight", "v.position_embd.weight", "mm.0.weight",
		"v.blk.0.attn_q.weight", "v.blk.0.attn_k.weight", "v.blk.0.attn_v.weight",
		"v.blk.0.attn_out.weight", "v.blk.0.ln1.weight", "v.blk.0.ln2.weight",
		"v.blk.0.ffn_up.weight", "v.blk.0.ffn_down.weight",
	}
	out := make([]splitTensor, len(names))
	for i, n := range names {
		out[i] = splitTensor{name: n, dims: []uint64{2}, typ: TensorF32, data: f32Payload(float32(i), float32(-i))}
	}
	return out
}

func TestWeightSourceVisionTower(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "qwen35vl-mmproj-f32.gguf")
	tensors := mmprojTowerTensors()
	if err := os.WriteFile(p, writeMMProjTowerFixture(t, tensors), 0o644); err != nil {
		t.Fatalf("write mmproj: %v", err)
	}
	ws, err := OpenMMProj(p)
	if err != nil {
		t.Fatalf("OpenMMProj: %v", err)
	}
	defer ws.Close()

	tw, err := ws.VisionTower()
	if err != nil {
		t.Fatalf("VisionTower: %v", err)
	}

	cfg := tw.Config()
	if cfg.NumLayers != 1 || cfg.HiddenSize != 4 || cfg.NumHeads != 2 || cfg.PatchSize != 14 {
		t.Fatalf("geometry=%+v, want NumLayers=1 HiddenSize=4 NumHeads=2 PatchSize=14", cfg)
	}
	if cfg.ProjectorType != "mlp" {
		t.Fatalf("ProjectorType=%q, want mlp", cfg.ProjectorType)
	}
	if cfg.MergeSize != 1 {
		t.Fatalf("MergeSize=%d, want default 1", cfg.MergeSize)
	}
	// Every v.*/mm.* tensor is retained: 11 tensors, each 2 f32 -> 88 bytes.
	if got := len(tw.TensorNames()); got != len(tensors) {
		t.Fatalf("tower has %d tensors, want %d", got, len(tensors))
	}
	if got := tw.Bytes(); got != int64(len(tensors)*2*4) {
		t.Fatalf("Bytes()=%d, want %d", got, len(tensors)*2*4)
	}
}
