package model

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"
)

// v41TestReducedConfig narrows the real admitted V4.1 config to small, internally
// consistent axes so fixtures stay tiny while the retained metadata pointer keeps
// IsDeepSeekV41() true. Every narrowed axis is one DeepSeekV41Inventory reads.
func v41TestReducedConfig(t *testing.T, layers, experts int) Config {
	t.Helper()
	_, cfg := readDeepSeekV41Config(t)
	cfg.NumLayers = layers
	cfg.NumExperts = experts
	cfg.HiddenSize = 64
	cfg.NumHeads = 2
	cfg.HeadDim = 32
	cfg.QLoraRank = 32
	cfg.OLoraRank = 16
	cfg.OGroups = 2
	cfg.MoEIntermediateSize = 32
	cfg.VocabSize = 8
	if !cfg.IsDeepSeekV41() || cfg.DeepSeekV41 == nil {
		t.Fatal("reduced config lost V4.1 identity")
	}
	return cfg
}

func v41TestPattern(n int, seed byte) []byte {
	b := make([]byte, n)
	for i := range b {
		b[i] = byte((int(seed) + i) % 251)
	}
	return b
}

// v41TestInventoryTensors lays out deterministic bytes for every weight and
// scale sibling in inv, exactly matching the inventory's dtype/shape.
func v41TestInventoryTensors(t *testing.T, inv *V41Inventory) map[string]tinySTTensor {
	t.Helper()
	out := map[string]tinySTTensor{}
	seed := byte(1)
	add := func(name, dtype string, shape []int) {
		n, ok := checkedShapeProduct(shape...)
		if !ok {
			t.Fatalf("shape %v for %s overflows", shape, name)
		}
		out[name] = tinySTTensor{dtype: dtype, shape: append([]int(nil), shape...), data: v41TestPattern(n, seed)}
		seed++
	}
	for _, ref := range inv.Entries() {
		add(ref.Name, ref.Dtype, ref.Shape)
		if ref.HasScale() {
			add(ref.ScaleName, ref.ScaleDtype, ref.ScaleShape)
		}
	}
	return out
}

func v41TestWriteEntries(t *testing.T, path string, hdr map[string]stEntry, data []byte) {
	t.Helper()
	hb, err := json.Marshal(hdr)
	if err != nil {
		t.Fatal(err)
	}
	blob := make([]byte, 8)
	binary.LittleEndian.PutUint64(blob, uint64(len(hb)))
	blob = append(blob, hb...)
	blob = append(blob, data...)
	if err := os.WriteFile(path, blob, 0o600); err != nil {
		t.Fatal(err)
	}
}

func v41TestShape(t *testing.T, name string, got, want []int) {
	t.Helper()
	if !sameShape(got, want) {
		t.Fatalf("%s shape = %v, want %v", name, got, want)
	}
}

func TestDeepSeekV41InventoryExactShapes(t *testing.T) {
	_, cfg := readDeepSeekV41Config(t)
	inv, err := DeepSeekV41Inventory(cfg)
	if err != nil {
		t.Fatalf("DeepSeekV41Inventory: %v", err)
	}

	samples := []struct {
		name      string
		dtype     string
		shape     []int
		logical   []int
		packed    bool
		scaleName string
		scaleType string
		scale     []int
	}{
		{"model.embed_tokens.weight", "BF16", []int{129280, 5120}, nil, false, "", "", nil},
		{"layers.0.attn.wq_a.weight", "F8_E4M3", []int{1280, 5120}, nil, false, "layers.0.attn.wq_a.scale", "F8_E8M0", []int{40, 160}},
		{"layers.0.attn.wq_b.weight", "F8_E4M3", []int{32768, 1280}, nil, false, "layers.0.attn.wq_b.scale", "F8_E8M0", []int{1024, 40}},
		{"layers.0.attn.wkv.weight", "F8_E4M3", []int{512, 5120}, nil, false, "layers.0.attn.wkv.scale", "F8_E8M0", []int{16, 160}},
		{"layers.0.attn.wo_a.weight", "F8_E4M3", []int{8192, 5120}, nil, false, "layers.0.attn.wo_a.scale", "F8_E8M0", []int{256, 160}},
		{"layers.0.attn.wo_b.weight", "F8_E4M3", []int{5120, 8192}, nil, false, "layers.0.attn.wo_b.scale", "F8_E8M0", []int{160, 256}},
		{"layers.0.ffn.gate.weight", "F8_E4M3", []int{384, 5120}, nil, false, "layers.0.ffn.gate.scale", "F8_E8M0", []int{12, 160}},
		{"layers.0.ffn.gate.e_score_correction_bias", "F32", []int{384}, nil, false, "", "", nil},
		{"layers.0.ffn.shared_experts.w1.weight", "F8_E4M3", []int{2304, 5120}, nil, false, "layers.0.ffn.shared_experts.w1.scale", "F8_E8M0", []int{72, 160}},
		{"layers.0.ffn.experts.0.w1.weight", "I8", []int{2304, 2560}, []int{2304, 5120}, true, "layers.0.ffn.experts.0.w1.scale", "F8_E8M0", []int{2304, 160}},
		{"layers.0.ffn.experts.0.w2.weight", "I8", []int{5120, 1152}, []int{5120, 2304}, true, "layers.0.ffn.experts.0.w2.scale", "F8_E8M0", []int{5120, 72}},
		{"model.norm.weight", "F32", []int{5120}, nil, false, "", "", nil},
		{"lm_head.weight", "F8_E4M3", []int{129280, 5120}, nil, false, "lm_head.scale", "F8_E8M0", []int{4040, 160}},
	}
	for _, s := range samples {
		ref, ok := inv.Lookup(s.name)
		if !ok {
			t.Fatalf("Lookup(%q) missing", s.name)
		}
		if ref.Dtype != s.dtype {
			t.Fatalf("%s dtype = %q, want %q", s.name, ref.Dtype, s.dtype)
		}
		v41TestShape(t, s.name, ref.Shape, s.shape)
		wantLogical := s.logical
		if wantLogical == nil {
			wantLogical = s.shape
		}
		v41TestShape(t, s.name+" logical", ref.LogicalShape, wantLogical)
		if ref.PackedWeight != s.packed {
			t.Fatalf("%s PackedWeight = %v, want %v", s.name, ref.PackedWeight, s.packed)
		}
		if s.scaleName == "" {
			if ref.HasScale() {
				t.Fatalf("%s unexpectedly has scale %q", s.name, ref.ScaleName)
			}
			continue
		}
		if !ref.HasScale() || ref.ScaleName != s.scaleName || ref.ScaleDtype != s.scaleType {
			t.Fatalf("%s scale = %q/%q, want %q/%q", s.name, ref.ScaleName, ref.ScaleDtype, s.scaleName, s.scaleType)
		}
		v41TestShape(t, s.scaleName, ref.ScaleShape, s.scale)
		byScale, ok := inv.Lookup(s.scaleName)
		if !ok || byScale.Name != ref.Name {
			t.Fatalf("Lookup(%q) did not resolve to %s (got %s, ok=%v)", s.scaleName, ref.Name, byScale.Name, ok)
		}
	}

	wantLen := 1 + cfg.NumLayers*(2+5+1+1+3) + cfg.NumLayers*cfg.NumExperts*3 + 2
	if inv.Len() != wantLen {
		t.Fatalf("Len() = %d, want %d", inv.Len(), wantLen)
	}
	if inv.Layers() != cfg.NumLayers {
		t.Fatalf("Layers() = %d, want %d", inv.Layers(), cfg.NumLayers)
	}
	total, err := inv.TotalWeightBytes()
	if err != nil {
		t.Fatalf("TotalWeightBytes: %v", err)
	}
	if total <= 0 {
		t.Fatalf("TotalWeightBytes = %d, want positive", total)
	}
	if len(inv.Names()) != inv.Len() {
		t.Fatalf("Names() len = %d, want %d", len(inv.Names()), inv.Len())
	}
	entries := inv.Entries()
	if len(entries) != inv.Len() {
		t.Fatalf("Entries() len = %d, want %d", len(entries), inv.Len())
	}
	entries[0].Name = "mutated"
	if again, _ := inv.Lookup(inv.Names()[0]); again.Name == "mutated" {
		t.Fatal("Entries() did not return a copy")
	}
}

func TestDeepSeekV41InventoryRejectsNonDivisibleReduction(t *testing.T) {
	_, realCfg := readDeepSeekV41Config(t)
	for _, tc := range []struct {
		name   string
		mutate func(*Config)
	}{
		{"moe_intermediate not /32", func(c *Config) { c.MoEIntermediateSize = 30 }},
		{"hidden not /2", func(c *Config) { c.HiddenSize = 63 }},
		{"hidden not /32", func(c *Config) { c.HiddenSize = 100 }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := realCfg
			tc.mutate(&cfg)
			if _, err := DeepSeekV41Inventory(cfg); !errors.Is(err, ErrV41Inventory) {
				t.Fatalf("error = %v, want ErrV41Inventory", err)
			}
		})
	}
}

func TestDeepSeekV41InventoryRejectsNonV41(t *testing.T) {
	if _, err := DeepSeekV41Inventory(Config{ModelType: "llama", NumLayers: 1, HiddenSize: 64}); !errors.Is(err, ErrV41Inventory) {
		t.Fatalf("non-V4.1 error = %v, want ErrV41Inventory", err)
	}
	_, realCfg := readDeepSeekV41Config(t)
	incomplete := realCfg
	incomplete.MoEIntermediateSize = 0
	if _, err := DeepSeekV41Inventory(incomplete); !errors.Is(err, ErrV41Inventory) {
		t.Fatalf("incomplete geometry error = %v, want ErrV41Inventory", err)
	}
}

func TestDeepSeekV41PackedShardLazyReferenceAndRead(t *testing.T) {
	cfg := v41TestReducedConfig(t, 1, 2)
	inv, err := DeepSeekV41Inventory(cfg)
	if err != nil {
		t.Fatalf("inventory: %v", err)
	}
	tensors := v41TestInventoryTensors(t, inv)
	path := filepath.Join(t.TempDir(), "shard.safetensors")
	writeV4RawShard(t, path, tensors)

	s, err := OpenV41PackedShard(path, cfg)
	if err != nil {
		t.Fatalf("OpenV41PackedShard: %v", err)
	}
	defer s.Close()
	if s.Len() != inv.Len() {
		t.Fatalf("Len() = %d, want %d", s.Len(), inv.Len())
	}

	const target = "layers.0.attn.wq_a.weight"
	invRef, _ := inv.Lookup(target)
	ref, err := s.Reference(target)
	if err != nil {
		t.Fatalf("Reference(%s): %v", target, err)
	}
	if ref.Ref.Name != target || ref.Shaper != path {
		t.Fatalf("ref = %+v, want name=%s shaper=%s", ref, target, path)
	}
	if !(ref.Start < ref.End) {
		t.Fatalf("span = %d..%d, want Start<End", ref.Start, ref.End)
	}
	wantLen, _ := checkedShapeProduct(invRef.Shape...)
	if ref.ByteLen() != int64(wantLen) {
		t.Fatalf("ByteLen() = %d, want %d", ref.ByteLen(), wantLen)
	}
	if !ref.HasScale() {
		t.Fatal("wq_a reference did not pair its scale sibling")
	}
	if ref.ScaleStart < 0 || !(ref.ScaleStart < ref.ScaleEnd) {
		t.Fatalf("scale span = %d..%d, want valid", ref.ScaleStart, ref.ScaleEnd)
	}

	gotBytes, err := s.Read(ref, 1<<30)
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	if !bytes.Equal(gotBytes, tensors[target].data) {
		t.Fatalf("Read returned %d bytes not equal to the packed fixture", len(gotBytes))
	}
	gotScale, err := s.ReadScale(ref, 1<<30)
	if err != nil {
		t.Fatalf("ReadScale: %v", err)
	}
	if !bytes.Equal(gotScale, tensors[invRef.ScaleName].data) {
		t.Fatalf("ReadScale returned %d bytes not equal to the scale fixture", len(gotScale))
	}

	if _, err := s.Reference("model.visual.patch_embed.proj.weight"); !errors.Is(err, ErrV41PackedUnsupported) {
		t.Fatalf("foreign name error = %v, want ErrV41PackedUnsupported", err)
	}
}

func TestDeepSeekV41PackedShardRefusesMissingScale(t *testing.T) {
	cfg := v41TestReducedConfig(t, 1, 1)
	inv, err := DeepSeekV41Inventory(cfg)
	if err != nil {
		t.Fatalf("inventory: %v", err)
	}
	ref, _ := inv.Lookup("layers.0.attn.wq_a.weight")
	n, _ := checkedShapeProduct(ref.Shape...)
	path := filepath.Join(t.TempDir(), "no-scale.safetensors")
	writeV4RawShard(t, path, map[string]tinySTTensor{
		ref.Name: {dtype: ref.Dtype, shape: ref.Shape, data: v41TestPattern(n, 1)},
	})
	s, err := OpenV41PackedShard(path, cfg)
	if s != nil {
		s.Close()
	}
	if !errors.Is(err, ErrV41PackedUnsupported) {
		t.Fatalf("missing scale open error = %v, want ErrV41PackedUnsupported", err)
	}
}

func TestDeepSeekV41PackedShardRefusesMalformedShape(t *testing.T) {
	cfg := v41TestReducedConfig(t, 1, 1)
	inv, err := DeepSeekV41Inventory(cfg)
	if err != nil {
		t.Fatalf("inventory: %v", err)
	}
	ref, _ := inv.Lookup("layers.0.attn.wq_a.weight")
	wn, _ := checkedShapeProduct(ref.Shape...)
	sn, _ := checkedShapeProduct(ref.ScaleShape...)

	base := func() (map[string]stEntry, []byte) {
		data := append(v41TestPattern(wn, 1), v41TestPattern(sn, 2)...)
		hdr := map[string]stEntry{
			ref.Name:      {Dtype: ref.Dtype, Shape: ref.Shape, DataOffsets: []int{0, wn}},
			ref.ScaleName: {Dtype: ref.ScaleDtype, Shape: ref.ScaleShape, DataOffsets: []int{wn, wn + sn}},
		}
		return hdr, data
	}

	cases := map[string]func(hdr map[string]stEntry, data []byte){
		"wrong dtype": func(hdr map[string]stEntry, data []byte) {
			hdr[ref.Name] = stEntry{Dtype: "F32", Shape: ref.Shape, DataOffsets: []int{0, wn}}
		},
		"wrong shape": func(hdr map[string]stEntry, data []byte) {
			hdr[ref.Name] = stEntry{Dtype: ref.Dtype, Shape: []int{ref.Shape[0] + 1, ref.Shape[1]}, DataOffsets: []int{0, wn}}
		},
		"end overruns file": func(hdr map[string]stEntry, data []byte) {
			hdr[ref.Name] = stEntry{Dtype: ref.Dtype, Shape: ref.Shape, DataOffsets: []int{0, wn + 1<<20}}
		},
		"start after end": func(hdr map[string]stEntry, data []byte) {
			hdr[ref.Name] = stEntry{Dtype: ref.Dtype, Shape: ref.Shape, DataOffsets: []int{5, 3}}
		},
		"negative start": func(hdr map[string]stEntry, data []byte) {
			hdr[ref.Name] = stEntry{Dtype: ref.Dtype, Shape: ref.Shape, DataOffsets: []int{-1, wn}}
		},
	}
	for name, mutate := range cases {
		t.Run(name, func(t *testing.T) {
			hdr, data := base()
			mutate(hdr, data)
			path := filepath.Join(t.TempDir(), "malformed.safetensors")
			v41TestWriteEntries(t, path, hdr, data)
			s, err := OpenV41PackedShard(path, cfg)
			if s != nil {
				s.Close()
			}
			if !errors.Is(err, ErrV41PackedUnsupported) {
				t.Fatalf("open error = %v, want ErrV41PackedUnsupported", err)
			}
		})
	}
}

func TestDeepSeekV41PackedShardConfinement(t *testing.T) {
	cfg := v41TestReducedConfig(t, 1, 1)
	inv, err := DeepSeekV41Inventory(cfg)
	if err != nil {
		t.Fatalf("inventory: %v", err)
	}
	dir := t.TempDir()
	pathA := filepath.Join(dir, "a.safetensors")
	pathB := filepath.Join(dir, "b.safetensors")

	wa, _ := inv.Lookup("layers.0.attn.wq_a.weight")
	wn, _ := checkedShapeProduct(wa.Shape...)
	wsn, _ := checkedShapeProduct(wa.ScaleShape...)
	writeV4RawShard(t, pathA, map[string]tinySTTensor{
		wa.Name:      {dtype: wa.Dtype, shape: wa.Shape, data: v41TestPattern(wn, 1)},
		wa.ScaleName: {dtype: wa.ScaleDtype, shape: wa.ScaleShape, data: v41TestPattern(wsn, 2)},
	})
	norm, _ := inv.Lookup("model.norm.weight")
	nn, _ := checkedShapeProduct(norm.Shape...)
	writeV4RawShard(t, pathB, map[string]tinySTTensor{
		norm.Name: {dtype: norm.Dtype, shape: norm.Shape, data: v41TestPattern(nn, 3)},
	})

	a, err := OpenV41PackedShard(pathA, cfg)
	if err != nil {
		t.Fatalf("open A: %v", err)
	}
	defer a.Close()
	b, err := OpenV41PackedShard(pathB, cfg)
	if err != nil {
		t.Fatalf("open B: %v", err)
	}
	defer b.Close()

	if _, err := a.Reference(wa.Name); err != nil {
		t.Fatalf("shard A should serve %s: %v", wa.Name, err)
	}
	if _, err := b.Reference(wa.Name); !errors.Is(err, ErrV41PackedUnsupported) {
		t.Fatalf("shard B confinement error = %v, want ErrV41PackedUnsupported", err)
	}
	if got, err := b.Reference(norm.Name); err != nil || got.Ref.Name != norm.Name {
		t.Fatalf("shard B should serve %s: got=%+v err=%v", norm.Name, got, err)
	}
}

func TestDeepSeekV41PackedShardAllocationBound(t *testing.T) {
	cfg := v41TestReducedConfig(t, 1, 1)
	inv, err := DeepSeekV41Inventory(cfg)
	if err != nil {
		t.Fatalf("inventory: %v", err)
	}
	tensors := v41TestInventoryTensors(t, inv)
	path := filepath.Join(t.TempDir(), "bound.safetensors")
	writeV4RawShard(t, path, tensors)
	s, err := OpenV41PackedShard(path, cfg)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer s.Close()

	ref, err := s.Reference("layers.0.attn.wq_a.weight")
	if err != nil {
		t.Fatalf("Reference: %v", err)
	}
	if ref.ByteLen() <= 0 {
		t.Fatalf("ByteLen() = %d, want positive", ref.ByteLen())
	}
	got, err := s.Read(ref, ref.ByteLen()-1)
	if !errors.Is(err, ErrV41PackedUnsupported) {
		t.Fatalf("over-bound Read error = %v, want ErrV41PackedUnsupported", err)
	}
	if got != nil {
		t.Fatalf("over-bound Read returned %d bytes, want nil", len(got))
	}

	scaleLen := ref.ScaleEnd - ref.ScaleStart
	if scaleLen <= 0 {
		t.Fatalf("scale span = %d..%d, want positive", ref.ScaleStart, ref.ScaleEnd)
	}
	gotScale, err := s.ReadScale(ref, scaleLen-1)
	if !errors.Is(err, ErrV41PackedUnsupported) {
		t.Fatalf("over-bound ReadScale error = %v, want ErrV41PackedUnsupported", err)
	}
	if gotScale != nil {
		t.Fatalf("over-bound ReadScale returned %d bytes, want nil", len(gotScale))
	}
}

func TestDeepSeekV41PackedShardRejectsNonV41Open(t *testing.T) {
	path := filepath.Join(t.TempDir(), "absent.safetensors")
	cfg := Config{ModelType: "llama", NumLayers: 1, HiddenSize: 64}
	s, err := OpenV41PackedShard(path, cfg)
	if s != nil {
		s.Close()
	}
	if !errors.Is(err, ErrV41PackedUnsupported) {
		t.Fatalf("non-V4.1 open error = %v, want ErrV41PackedUnsupported", err)
	}
	if _, statErr := os.Stat(path); !os.IsNotExist(statErr) {
		t.Fatalf("non-V4.1 open created/opened file: stat err = %v", statErr)
	}
}
