package ggufload

import (
	"bytes"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// gguf_mmproj_test.go covers OpenMMProj (#4028): a companion CLIP-tower mmproj
// GGUF opened as a standalone weight source. The fixtures are written through the
// same low-level byte builders the rest of this package's tests use, so this is a
// real end-to-end exercise of open → metadata read → v.*/mm.* tensor read with no
// network and no real mmproj download.

// writeMMProjFixture builds a minimal single-file CLIP mmproj GGUF: a clip.*
// metadata block plus one v.* vision tensor and one mm.* projector tensor.
func writeMMProjFixture(t *testing.T) []byte {
	t.Helper()
	const align = 32
	tensors := []splitTensor{
		{name: "v.patch_embd.weight", dims: []uint64{2}, typ: TensorF32, data: f32Payload(0.5, -1.5)},
		{name: "mm.0.weight", dims: []uint64{2}, typ: TensorF32, data: f32Payload(2.0, 4.0)},
	}
	// 4 KV: general.alignment + general.architecture + clip.has_vision_encoder +
	// clip.vision.patch_size.
	var b bytes.Buffer
	writeMinimalHeader(&b, uint64(len(tensors)), 4)
	writeKVUint32(&b, "general.alignment", align)
	writeKVString(&b, "general.architecture", "clip")
	writeKVBool(&b, "clip.has_vision_encoder", true)
	writeKVUint64(&b, "clip.vision.patch_size", 14)
	// Each tensor's payload is padded to `align` in the data section, so tensor i's
	// in-section offset is the cumulative aligned size of its predecessors.
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

// TestOpenMMProjReadsMetadataAndTensors is the happy path: a companion mmproj
// loads, its clip.* metadata resolves through the embedded *File accessors, and
// both the v.* and mm.* tensors read back their exact bytes.
func TestOpenMMProjReadsMetadataAndTensors(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "qwen35vl-mmproj-f32.gguf")
	if err := os.WriteFile(p, writeMMProjFixture(t), 0o644); err != nil {
		t.Fatalf("write mmproj: %v", err)
	}

	ws, err := OpenMMProj(p)
	if err != nil {
		t.Fatalf("OpenMMProj: %v", err)
	}
	defer ws.Close()

	// Metadata via the *File accessors on the WeightSource's parsed header.
	if arch, ok := ws.File.String("general.architecture"); !ok || arch != "clip" {
		t.Fatalf("architecture=%q ok=%v, want clip", arch, ok)
	}
	if v, ok := ws.File.Bool("clip.has_vision_encoder"); !ok || !v {
		t.Fatalf("clip.has_vision_encoder=%v ok=%v, want true", v, ok)
	}
	if v, ok := ws.File.Uint64("clip.vision.patch_size"); !ok || v != 14 {
		t.Fatalf("clip.vision.patch_size=%d ok=%v, want 14", v, ok)
	}

	// Tensor directory + payload reads across both vision namespaces.
	if got := len(ws.File.Tensors); got != 2 {
		t.Fatalf("tensor count=%d, want 2", got)
	}
	assertF32 := func(name string, want ...float32) {
		t.Helper()
		got, _, err := ws.TensorF32(name)
		if err != nil {
			t.Fatalf("TensorF32(%s): %v", name, err)
		}
		if len(got) != len(want) {
			t.Fatalf("%s len=%d, want %d", name, len(got), len(want))
		}
		for i := range got {
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				t.Fatalf("%s[%d]=%v, want %v", name, i, got[i], want[i])
			}
		}
	}
	assertF32("v.patch_embd.weight", 0.5, -1.5)
	assertF32("mm.0.weight", 2.0, 4.0)
}

// TestOpenMMProjRejectsTextModel guards the common misuse: pointing --mmproj at
// the text checkpoint (no v.*/mm.* tensors) must fail loud rather than load an
// empty tower.
func TestOpenMMProjRejectsTextModel(t *testing.T) {
	dir := t.TempDir()
	// A text-only shard: qwen2 config, one ordinary weight, no vision namespace.
	text := writeSplitShard(t, 1, 1, 1, true, []splitTensor{
		{name: "token_embd.weight", dims: []uint64{2}, typ: TensorF32, data: f32Payload(1, 2)},
	})
	p := filepath.Join(dir, "text-only-00001-of-00001.gguf")
	if err := os.WriteFile(p, text, 0o644); err != nil {
		t.Fatalf("write text model: %v", err)
	}
	if _, err := OpenMMProj(p); err == nil {
		t.Fatal("OpenMMProj accepted a text model with no vision tensors")
	}
}

// TestIsMMProjVisionTensor pins the load-side vision-namespace predicate.
func TestIsMMProjVisionTensor(t *testing.T) {
	cases := map[string]bool{
		"v.patch_embd.weight":  true,
		"v.blk.0.attn_q.weight": true,
		"mm.0.weight":           true,
		"mm.merger.ln_q.weight": true,
		"token_embd.weight":     false,
		"blk.0.attn_q.weight":   false,
		"output.weight":         false,
	}
	for name, want := range cases {
		if got := isMMProjVisionTensor(name); got != want {
			t.Errorf("isMMProjVisionTensor(%q)=%v, want %v", name, got, want)
		}
	}
}
