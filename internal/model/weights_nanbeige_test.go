package model

import (
	"encoding/binary"
	"math"
	"testing"
	"unsafe"
)

func validNanbeige42TestConfig() Config {
	return Config{
		ModelType:  "nanbeige",
		NumLayers:  22,
		NumLoops:   2,
		NumHeads:   48,
		NumKVHeads: 8,
		HeadDim:    128,
		HiddenSize: 3072,
	}
}

func makeValidNanbeige42Manifest() (Config, map[string]tensorMeta, []byte, int) {
	cfg := validNanbeige42TestConfig()
	man := make(map[string]tensorMeta)

	qBytes := cfg.NumHeads * cfg.HeadDim * cfg.HiddenSize * 4
	kBytes := cfg.NumKVHeads * cfg.HeadDim * cfg.HiddenSize * 4
	vBytes := cfg.NumKVHeads * cfg.HeadDim * cfg.HiddenSize * 4
	normBytes := cfg.HiddenSize * 4

	blockBytes := inNormBytes(normBytes) + postNormBytes(normBytes) + qBytes + kBytes + vBytes
	totalPhysicalBytes := cfg.NumLayers * blockBytes
	raw := make([]byte, totalPhysicalBytes)

	offset := 0
	for l := 0; l < cfg.NumLayers; l++ {
		// Input layernorm
		man[layerName(l, "input_layernorm.weight")] = tensorMeta{
			Dtype:  "f32",
			Shape:  []int{cfg.HiddenSize},
			Offset: offset,
			Nbytes: normBytes,
		}
		// Write distinct marker value
		binary.LittleEndian.PutUint32(raw[offset:], math.Float32bits(float32(100+l)))
		offset += normBytes

		// Post attention layernorm
		man[layerName(l, "post_attention_layernorm.weight")] = tensorMeta{
			Dtype:  "f32",
			Shape:  []int{cfg.HiddenSize},
			Offset: offset,
			Nbytes: normBytes,
		}
		binary.LittleEndian.PutUint32(raw[offset:], math.Float32bits(float32(200+l)))
		offset += normBytes

		// Q proj: [48*128, 3072] = [6144, 3072]
		man[layerName(l, "self_attn.q_proj.weight")] = tensorMeta{
			Dtype:  "f32",
			Shape:  []int{cfg.NumHeads * cfg.HeadDim, cfg.HiddenSize},
			Offset: offset,
			Nbytes: qBytes,
		}
		binary.LittleEndian.PutUint32(raw[offset:], math.Float32bits(float32(1000+l)))
		offset += qBytes

		// K proj: [8*128, 3072] = [1024, 3072]
		man[layerName(l, "self_attn.k_proj.weight")] = tensorMeta{
			Dtype:  "f32",
			Shape:  []int{cfg.NumKVHeads * cfg.HeadDim, cfg.HiddenSize},
			Offset: offset,
			Nbytes: kBytes,
		}
		binary.LittleEndian.PutUint32(raw[offset:], math.Float32bits(float32(2000+l)))
		offset += kBytes

		// V proj: [8*128, 3072] = [1024, 3072]
		man[layerName(l, "self_attn.v_proj.weight")] = tensorMeta{
			Dtype:  "f32",
			Shape:  []int{cfg.NumKVHeads * cfg.HeadDim, cfg.HiddenSize},
			Offset: offset,
			Nbytes: vBytes,
		}
		binary.LittleEndian.PutUint32(raw[offset:], math.Float32bits(float32(3000+l)))
		offset += vBytes
	}

	return cfg, man, raw, totalPhysicalBytes
}

func makeValidNanbeige42ManifestOnly() (Config, map[string]tensorMeta) {
	cfg := validNanbeige42TestConfig()
	man := make(map[string]tensorMeta)

	qBytes := cfg.NumHeads * cfg.HeadDim * cfg.HiddenSize * 4
	kBytes := cfg.NumKVHeads * cfg.HeadDim * cfg.HiddenSize * 4
	vBytes := cfg.NumKVHeads * cfg.HeadDim * cfg.HiddenSize * 4
	normBytes := cfg.HiddenSize * 4

	offset := 0
	for l := 0; l < cfg.NumLayers; l++ {
		man[layerName(l, "input_layernorm.weight")] = tensorMeta{
			Dtype:  "f32",
			Shape:  []int{cfg.HiddenSize},
			Offset: offset,
			Nbytes: normBytes,
		}
		offset += normBytes

		man[layerName(l, "post_attention_layernorm.weight")] = tensorMeta{
			Dtype:  "f32",
			Shape:  []int{cfg.HiddenSize},
			Offset: offset,
			Nbytes: normBytes,
		}
		offset += normBytes

		man[layerName(l, "self_attn.q_proj.weight")] = tensorMeta{
			Dtype:  "f32",
			Shape:  []int{cfg.NumHeads * cfg.HeadDim, cfg.HiddenSize},
			Offset: offset,
			Nbytes: qBytes,
		}
		offset += qBytes

		man[layerName(l, "self_attn.k_proj.weight")] = tensorMeta{
			Dtype:  "f32",
			Shape:  []int{cfg.NumKVHeads * cfg.HeadDim, cfg.HiddenSize},
			Offset: offset,
			Nbytes: kBytes,
		}
		offset += kBytes

		man[layerName(l, "self_attn.v_proj.weight")] = tensorMeta{
			Dtype:  "f32",
			Shape:  []int{cfg.NumKVHeads * cfg.HeadDim, cfg.HiddenSize},
			Offset: offset,
			Nbytes: vBytes,
		}
		offset += vBytes
	}
	return cfg, man
}

func inNormBytes(n int) int   { return n }
func postNormBytes(n int) int { return n }

func TestNanbeige42SharedWeightMapping(t *testing.T) {
	t.Run("maps 22 physical blocks across loops without duplicating weights", func(t *testing.T) {
		cfg, man, raw, totalPhysicalBytes := makeValidNanbeige42Manifest()

		// Initial physical block count in manifest is 22 blocks
		initialTensorCount := len(man)
		if initialTensorCount != 22*5 {
			t.Fatalf("initial tensor count = %d, want %d", initialTensorCount, 22*5)
		}

		m, err := newModel(cfg, man, raw)
		if err != nil {
			t.Fatalf("newModel failed: %v", err)
		}

		// Loop count NumLoops: 2 must NOT duplicate raw memory allocation.
		// Exactly 22 physical blocks allocated in raw, not 44.
		if len(m.raw) != totalPhysicalBytes {
			t.Fatalf("len(m.raw) = %d bytes, want exactly %d bytes for 22 physical blocks (44 blocks would be %d)",
				len(m.raw), totalPhysicalBytes, totalPhysicalBytes*2)
		}

		// Verify that all 22 layers in loop 1 (layers 22 to 43) exist and reference shared physical weights
		for l := 0; l < 22; l++ {
			loop1Layer := l + 22

			for _, suffix := range []string{
				"input_layernorm.weight",
				"post_attention_layernorm.weight",
				"self_attn.q_proj.weight",
				"self_attn.k_proj.weight",
				"self_attn.v_proj.weight",
			} {
				physName := layerName(l, suffix)
				loop1Name := layerName(loop1Layer, suffix)

				physMeta, ok := m.manifest[physName]
				if !ok {
					t.Fatalf("missing physical tensor %s", physName)
				}
				loop1Meta, ok := m.manifest[loop1Name]
				if !ok {
					t.Fatalf("missing mapped loop 1 tensor %s", loop1Name)
				}

				// Must share exact same offset and size in raw storage
				if loop1Meta.Offset != physMeta.Offset {
					t.Fatalf("%s offset %d != %s offset %d (duplicate allocation detected)",
						loop1Name, loop1Meta.Offset, physName, physMeta.Offset)
				}
				if loop1Meta.Nbytes != physMeta.Nbytes {
					t.Fatalf("%s nbytes %d != %s nbytes %d", loop1Name, loop1Meta.Nbytes, physName, physMeta.Nbytes)
				}
				if !sameShape(loop1Meta.Shape, physMeta.Shape) {
					t.Fatalf("%s shape %v != %s shape %v", loop1Name, loop1Meta.Shape, physName, physMeta.Shape)
				}

				// Verify actual pointer identity in memory: layers across loops reference the shared physical weights
				physTensor := m.tensor(physName)
				loop1Tensor := m.tensor(loop1Name)
				if unsafe.SliceData(physTensor) != unsafe.SliceData(loop1Tensor) {
					t.Fatalf("tensor %s and %s point to different memory addresses: %p vs %p",
						physName, loop1Name, unsafe.SliceData(physTensor), unsafe.SliceData(loop1Tensor))
				}

				// Data value match
				if physTensor[0] != loop1Tensor[0] {
					t.Fatalf("tensor %s value %f != %s value %f", physName, physTensor[0], loop1Name, loop1Tensor[0])
				}
			}

			// Verify attentionNorms also resolves identical slices
			physNorms := m.attentionNorms(l)
			loop1Norms := m.attentionNorms(loop1Layer)
			if unsafe.SliceData(physNorms.pre) != unsafe.SliceData(loop1Norms.pre) {
				t.Fatalf("attentionNorms.pre for layer %d and %d do not share memory: %p vs %p",
					l, loop1Layer, unsafe.SliceData(physNorms.pre), unsafe.SliceData(loop1Norms.pre))
			}
			if unsafe.SliceData(physNorms.post) != unsafe.SliceData(loop1Norms.post) {
				t.Fatalf("attentionNorms.post for layer %d and %d do not share memory: %p vs %p",
					l, loop1Layer, unsafe.SliceData(physNorms.post), unsafe.SliceData(loop1Norms.post))
			}

			// Verify physical layer mapping helper
			if phys := m.NanbeigePhysicalLayer(loop1Layer); phys != l {
				t.Fatalf("NanbeigePhysicalLayer(%d) = %d, want %d", loop1Layer, phys, l)
			}
		}

		if m.NanbeigeNumLogicalLayers() != 44 {
			t.Fatalf("NanbeigeNumLogicalLayers() = %d, want 44", m.NanbeigeNumLogicalLayers())
		}
	})

	t.Run("validates actual Q/K/V shapes", func(t *testing.T) {
		cfg := validNanbeige42TestConfig()

		// Verify expected shape values
		wantQ := []int{48 * 128, 3072} // [6144, 3072]
		wantK := []int{8 * 128, 3072}  // [1024, 3072]
		wantV := []int{8 * 128, 3072}  // [1024, 3072]

		t.Run("rejects invalid Q shape", func(t *testing.T) {
			_, man := makeValidNanbeige42ManifestOnly()
			badQ := man[layerName(0, "self_attn.q_proj.weight")]
			badQ.Shape = []int{3072, 3072}
			man[layerName(0, "self_attn.q_proj.weight")] = badQ

			err := ValidateNanbeigeWeights(cfg, man)
			if err == nil {
				t.Fatalf("ValidateNanbeigeWeights accepted invalid Q shape [3072, 3072], want %v", wantQ)
			}
		})

		t.Run("rejects invalid K shape", func(t *testing.T) {
			_, man := makeValidNanbeige42ManifestOnly()
			badK := man[layerName(0, "self_attn.k_proj.weight")]
			badK.Shape = []int{2048, 3072}
			man[layerName(0, "self_attn.k_proj.weight")] = badK

			err := ValidateNanbeigeWeights(cfg, man)
			if err == nil {
				t.Fatalf("ValidateNanbeigeWeights accepted invalid K shape [2048, 3072], want %v", wantK)
			}
		})

		t.Run("rejects invalid V shape", func(t *testing.T) {
			_, man := makeValidNanbeige42ManifestOnly()
			badV := man[layerName(0, "self_attn.v_proj.weight")]
			badV.Shape = []int{2048, 3072}
			man[layerName(0, "self_attn.v_proj.weight")] = badV

			err := ValidateNanbeigeWeights(cfg, man)
			if err == nil {
				t.Fatalf("ValidateNanbeigeWeights accepted invalid V shape [2048, 3072], want %v", wantV)
			}
		})
	})

	t.Run("validates required normalization tensors", func(t *testing.T) {
		cfg := validNanbeige42TestConfig()

		t.Run("rejects missing input_layernorm", func(t *testing.T) {
			_, man := makeValidNanbeige42ManifestOnly()
			delete(man, layerName(7, "input_layernorm.weight"))

			err := ValidateNanbeigeWeights(cfg, man)
			if err == nil {
				t.Fatal("ValidateNanbeigeWeights accepted missing input_layernorm on layer 7")
			}
		})

		t.Run("rejects missing post_attention_layernorm", func(t *testing.T) {
			_, man := makeValidNanbeige42ManifestOnly()
			delete(man, layerName(13, "post_attention_layernorm.weight"))

			err := ValidateNanbeigeWeights(cfg, man)
			if err == nil {
				t.Fatal("ValidateNanbeigeWeights accepted missing post_attention_layernorm on layer 13")
			}
		})

		t.Run("rejects invalid input_layernorm shape", func(t *testing.T) {
			_, man := makeValidNanbeige42ManifestOnly()
			badNorm := man[layerName(3, "input_layernorm.weight")]
			badNorm.Shape = []int{128}
			man[layerName(3, "input_layernorm.weight")] = badNorm

			err := ValidateNanbeigeWeights(cfg, man)
			if err == nil {
				t.Fatal("ValidateNanbeigeWeights accepted invalid input_layernorm shape [128], want [3072]")
			}
		})

		t.Run("rejects invalid post_attention_layernorm shape", func(t *testing.T) {
			_, man := makeValidNanbeige42ManifestOnly()
			badNorm := man[layerName(3, "post_attention_layernorm.weight")]
			badNorm.Shape = []int{128}
			man[layerName(3, "post_attention_layernorm.weight")] = badNorm

			err := ValidateNanbeigeWeights(cfg, man)
			if err == nil {
				t.Fatal("ValidateNanbeigeWeights accepted invalid post_attention_layernorm shape [128], want [3072]")
			}
		})
	})

	t.Run("rejects duplicate weight allocation for loop layers", func(t *testing.T) {
		cfg, man, raw, _ := makeValidNanbeige42Manifest()

		// Artificially inject a duplicate weight tensor for layer 22 with a distinct offset
		duplicateName := layerName(22, "self_attn.q_proj.weight")
		man[duplicateName] = tensorMeta{
			Dtype:  "f32",
			Shape:  []int{cfg.NumHeads * cfg.HeadDim, cfg.HiddenSize},
			Offset: 999999999,
			Nbytes: cfg.NumHeads * cfg.HeadDim * cfg.HiddenSize * 4,
		}

		_, err := newModel(cfg, man, raw)
		if err == nil {
			t.Fatal("newModel accepted manifest with duplicate weight allocation for loop layer 22")
		}
	})
}
