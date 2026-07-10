package ggufload

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// gguf_mmproj_tower.go — building the resident vision tower from a companion mmproj
// (#4029). OpenMMProj (#4028) opens the CLIP file and validates it carries a tower;
// VisionTower here reads its v.*/mm.* tensors into a model.VisionTower and parses the
// clip.* metadata into the geometry the encoder (#4030) forwards. This is the GGUF
// half of the vision-retain seam — the tensors ggufload's glm drop predicate discards
// for a text load are, on this path, RETAINED into a dedicated stack. Nothing calls it
// unless a --mmproj path is supplied (#4032), so text-only loading is untouched.

// VisionTower reads every v.*/mm.* vision tensor from this mmproj source into a
// resident model.VisionTower, dequantizing each to f32 (mmproj towers ship F32/F16),
// and stamps the tower's geometry from the clip.* metadata. The tower's Bytes() is the
// resident vision-weight figure the load estimator accounts when the tower is retained.
// The caller still owns the *WeightSource and must Close it; the tower holds its own
// copy of the weights.
func (s *WeightSource) VisionTower() (*model.VisionTower, error) {
	cfg := s.visionConfig()
	var tensors []model.NamedTensorF32
	for _, ti := range s.File.Tensors {
		if !isMMProjVisionTensor(ti.Name) {
			continue
		}
		data, info, err := s.TensorF32(ti.Name)
		if err != nil {
			return nil, fmt.Errorf("gguf: mmproj vision tensor %s: %w", ti.Name, err)
		}
		shape, err := modelShapeFromGGUFDims(info.Name, info.Dims)
		if err != nil {
			return nil, err
		}
		tensors = append(tensors, model.NamedTensorF32{Name: ti.Name, Shape: shape, Data: data})
	}
	if len(tensors) == 0 {
		// OpenMMProj already refuses a tower-less file, so this is a belt-and-braces
		// guard against a caller building a tower from a non-mmproj source.
		return nil, fmt.Errorf("gguf: mmproj source has no vision tensors to build a tower")
	}
	tw, err := model.NewVisionTower(cfg, tensors)
	if err != nil {
		return nil, fmt.Errorf("gguf: mmproj vision tower: %w", err)
	}
	return tw, nil
}

// visionConfig parses llama.cpp's CLIP-tower metadata (clip.* keys) into a
// model.VisionConfig. Absent keys leave a field zero — the encoder (#4030) pins any
// it requires; MergeSize defaults to 1 (no spatial patch merge) when unset.
func (s *WeightSource) visionConfig() model.VisionConfig {
	u := func(key string) int {
		if v, ok := s.File.Uint64(key); ok {
			return int(v)
		}
		return 0
	}
	cfg := model.VisionConfig{
		HiddenSize: u("clip.vision.embedding_length"),
		NumLayers:  u("clip.vision.block_count"),
		NumHeads:   u("clip.vision.attention.head_count"),
		FFNLength:  u("clip.vision.feed_forward_length"),
		PatchSize:  u("clip.vision.patch_size"),
		ImageSize:  u("clip.vision.image_size"),
		ProjOutDim: u("clip.vision.projection_dim"),
		MergeSize:  1,
	}
	if v, ok := s.File.String("clip.projector_type"); ok {
		cfg.ProjectorType = v
	}
	if v, ok := s.File.Float64("clip.vision.attention.layer_norm_epsilon"); ok {
		cfg.LNEps = float32(v)
	}
	if v, ok := s.File.Uint64("clip.vision.spatial_merge_size"); ok && v > 0 {
		cfg.MergeSize = int(v)
	}
	return cfg
}
