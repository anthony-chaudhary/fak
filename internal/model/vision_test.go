package model

import (
	"encoding/binary"
	"math"
	"strings"
	"testing"
)

// packDecoder builds a decoder-style (manifest, raw f32 LE) pair from named tensors,
// mirroring how Load/LoadSafetensors lay weights out — the input extractQwen35VisionTower
// segregates a retained vision tower out of.
func packDecoder(tensors []NamedTensorF32) (map[string]tensorMeta, []byte) {
	man := make(map[string]tensorMeta, len(tensors))
	var raw []byte
	off := 0
	for _, tsr := range tensors {
		start := len(raw)
		raw = append(raw, make([]byte, len(tsr.Data)*4)...)
		for i, v := range tsr.Data {
			binary.LittleEndian.PutUint32(raw[start+i*4:], math.Float32bits(v))
		}
		man[tsr.Name] = tensorMeta{Dtype: "f32", Shape: append([]int(nil), tsr.Shape...), Offset: off, Nbytes: len(tsr.Data) * 4}
		off += len(tsr.Data) * 4
	}
	return man, raw
}

// vision_test.go covers the VisionTower substrate (#4029): the resident vision
// weight stack (NewVisionTower round-trip + byte accounting) and the vision
// tensor-name resolver proof (canonical v.* scheme + HF model.visual.* aliases +
// a precise missing-tensor error).

// manifestOf builds a presence-only manifest (the resolver inspects keys, not bytes).
func manifestOf(names ...string) map[string]tensorMeta {
	man := make(map[string]tensorMeta, len(names))
	for _, n := range names {
		man[n] = tensorMeta{Dtype: "f32"}
	}
	return man
}

// ggufVisionManifest is a minimal but complete mmproj-scheme vision manifest for an
// N-layer tower: the v.* globals + mm.* projector + per-layer v.blk.<l>.* set the
// resolver requires.
func ggufVisionManifest(layers int) map[string]tensorMeta {
	names := []string{
		"v.patch_embd.weight", "v.patch_embd.bias",
		"v.position_embd.weight", "v.post_ln.weight",
		"mm.0.weight", "mm.2.weight",
	}
	for l := 0; l < layers; l++ {
		p := "v.blk." + itoa(l) + "."
		names = append(names,
			p+"attn_q.weight", p+"attn_q.bias",
			p+"attn_k.weight", p+"attn_k.bias",
			p+"attn_v.weight", p+"attn_v.bias",
			p+"attn_out.weight",
			p+"ln1.weight", p+"ln2.weight",
			p+"ffn_up.weight", p+"ffn_down.weight",
		)
	}
	return manifestOf(names...)
}

func TestNewVisionTowerRoundTrip(t *testing.T) {
	cfg := VisionConfig{HiddenSize: 4, NumLayers: 2, NumHeads: 2, PatchSize: 14, ProjOutDim: 8}
	tensors := []NamedTensorF32{
		{Name: "v.patch_embd.weight", Shape: []int{2, 2}, Data: []float32{1, 2, 3, 4}},
		{Name: "mm.0.weight", Shape: []int{2}, Data: []float32{-1, 5}},
	}
	tw, err := NewVisionTower(cfg, tensors)
	if err != nil {
		t.Fatalf("NewVisionTower: %v", err)
	}
	if got := tw.Config(); got != cfg {
		t.Fatalf("Config()=%+v, want %+v", got, cfg)
	}
	// 6 f32 across the two tensors -> 24 bytes.
	if got := tw.Bytes(); got != 24 {
		t.Fatalf("Bytes()=%d, want 24", got)
	}
	if !tw.has("v.patch_embd.weight") || tw.has("nope") {
		t.Fatalf("has() wrong")
	}
	got := tw.tensor("v.patch_embd.weight")
	want := []float32{1, 2, 3, 4}
	if len(got) != len(want) {
		t.Fatalf("tensor len=%d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("tensor[%d]=%v, want %v", i, got[i], want[i])
		}
	}
	names := tw.TensorNames()
	if len(names) != 2 || names[0] != "mm.0.weight" || names[1] != "v.patch_embd.weight" {
		t.Fatalf("TensorNames()=%v, want sorted [mm.0.weight v.patch_embd.weight]", names)
	}
}

func TestNewVisionTowerRejectsShapeMismatch(t *testing.T) {
	_, err := NewVisionTower(VisionConfig{}, []NamedTensorF32{
		{Name: "v.patch_embd.weight", Shape: []int{2, 2}, Data: []float32{1, 2, 3}}, // 3 != 4
	})
	if err == nil {
		t.Fatal("NewVisionTower accepted a shape/value-count mismatch")
	}
}

func TestResolveVisionTensorNamesGGUF(t *testing.T) {
	cfg := VisionConfig{NumLayers: 3}
	res, err := ResolveVisionTensorNames(cfg, ggufVisionManifest(cfg.NumLayers))
	if err != nil {
		t.Fatalf("ResolveVisionTensorNames: %v", err)
	}
	if res.Family != "vision-clip" {
		t.Fatalf("family=%q, want vision-clip", res.Family)
	}
	// Canonical v.* names resolve to themselves (identity), across globals + all layers.
	if got := res.SourceFor("v.patch_embd.weight"); got != "v.patch_embd.weight" {
		t.Fatalf("patch_embd source=%q, want identity", got)
	}
	if got := res.SourceFor("mm.0.weight"); got != "mm.0.weight" {
		t.Fatalf("projector source=%q, want identity", got)
	}
	if got := res.SourceFor("v.blk.2.ffn_down.weight"); got != "v.blk.2.ffn_down.weight" {
		t.Fatalf("layer-2 ffn_down source=%q, want identity", got)
	}
}

func TestResolveVisionTensorNamesHFAliases(t *testing.T) {
	// An inline HF safetensors vision tower (model.visual.* names) must resolve the
	// canonical v.* set via aliases.
	cfg := VisionConfig{NumLayers: 1}
	man := manifestOf(
		"model.visual.patch_embed.proj.weight",
		"model.visual.merger.mlp.0.weight",
		"model.visual.blocks.0.attn.q.weight",
		"model.visual.blocks.0.attn.k.weight",
		"model.visual.blocks.0.attn.v.weight",
		"model.visual.blocks.0.attn.proj.weight",
		"model.visual.blocks.0.norm1.weight",
		"model.visual.blocks.0.norm2.weight",
		"model.visual.blocks.0.mlp.fc1.weight",
		"model.visual.blocks.0.mlp.fc2.weight",
	)
	res, err := ResolveVisionTensorNames(cfg, man)
	if err != nil {
		t.Fatalf("ResolveVisionTensorNames (HF aliases): %v", err)
	}
	if got := res.SourceFor("v.patch_embd.weight"); got != "model.visual.patch_embed.proj.weight" {
		t.Fatalf("patch_embd alias source=%q", got)
	}
	if got := res.SourceFor("v.blk.0.attn_q.weight"); got != "model.visual.blocks.0.attn.q.weight" {
		t.Fatalf("attn_q alias source=%q", got)
	}
}

func TestResolveVisionTensorNamesMissingErrors(t *testing.T) {
	cfg := VisionConfig{NumLayers: 1}
	man := ggufVisionManifest(cfg.NumLayers)
	delete(man, "v.blk.0.attn_out.weight") // required, no bias fallback
	_, err := ResolveVisionTensorNames(cfg, man)
	if err == nil {
		t.Fatal("ResolveVisionTensorNames accepted a manifest missing attn_out.weight")
	}
	if want := "v.blk.0.attn_out.weight"; !strings.Contains(err.Error(), want) {
		t.Fatalf("error %q does not name the missing tensor %q", err, want)
	}
}

func TestExtractQwen35VisionTower(t *testing.T) {
	man, raw := packDecoder([]NamedTensorF32{
		{Name: "model.embed_tokens.weight", Shape: []int{2, 2}, Data: []float32{1, 2, 3, 4}},
		{Name: "model.visual.patch_embed.proj.weight", Shape: []int{2}, Data: []float32{5, 6}},
		{Name: "model.visual.blocks.0.attn.q.weight", Shape: []int{2}, Data: []float32{7, 8}},
		{Name: "model.visual.blocks.1.attn.q.weight", Shape: []int{2}, Data: []float32{9, 10}},
	})
	tw, err := extractQwen35VisionTower(man, raw)
	if err != nil {
		t.Fatalf("extractQwen35VisionTower: %v", err)
	}
	if tw == nil {
		t.Fatal("nil tower for a manifest carrying model.visual.* tensors")
	}
	if tw.Cfg.NumLayers != 2 { // blocks.0 + blocks.1
		t.Fatalf("NumLayers=%d, want 2 (derived from block indices)", tw.Cfg.NumLayers)
	}
	// The decoder tensor stays; every model.visual.* is segregated out of the manifest.
	if _, ok := man["model.embed_tokens.weight"]; !ok {
		t.Fatal("decoder tensor was removed from the manifest")
	}
	for _, n := range []string{"model.visual.patch_embed.proj.weight", "model.visual.blocks.0.attn.q.weight", "model.visual.blocks.1.attn.q.weight"} {
		if _, ok := man[n]; ok {
			t.Fatalf("vision tensor %s left in the decoder manifest", n)
		}
	}
	// Data round-trips into the tower.
	got := tw.tensor("model.visual.blocks.1.attn.q.weight")
	if len(got) != 2 || got[0] != 9 || got[1] != 10 {
		t.Fatalf("vision tensor data=%v, want [9 10]", got)
	}
}

func TestExtractQwen35VisionTowerNoneReturnsNil(t *testing.T) {
	man, raw := packDecoder([]NamedTensorF32{
		{Name: "model.embed_tokens.weight", Shape: []int{1}, Data: []float32{1}},
	})
	tw, err := extractQwen35VisionTower(man, raw)
	if err != nil || tw != nil {
		t.Fatalf("want (nil, nil) for a text-only manifest, got (%v, %v)", tw, err)
	}
	if _, ok := man["model.embed_tokens.weight"]; !ok {
		t.Fatal("text-only extract disturbed the decoder manifest")
	}
}

func TestCountIndexedChildren(t *testing.T) {
	names := []string{"m.blocks.0.w", "m.blocks.3.w", "m.blocks.1.b", "m.other", "m.blocks.x.w"}
	if got := countIndexedChildren(names, "m.blocks."); got != 4 { // max index 3 -> count 4
		t.Fatalf("countIndexedChildren=%d, want 4", got)
	}
	if got := countIndexedChildren([]string{"x.y"}, "m.blocks."); got != 0 {
		t.Fatalf("countIndexedChildren (no match)=%d, want 0", got)
	}
}
