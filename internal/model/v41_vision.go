package model

import (
	"errors"
	"fmt"
)

// ErrV41VisionUnsupported is the typed fail-closed refusal for the V4.1 Flash
// vision path. The official checkpoint ships a "vision_config"
// (model_type "deepseek_v41_vision"), so a config that requests multimodal
// input must be DETECTED and rejected here rather than silently running the
// text-only forward over image placeholder tokens. This is admission and
// routing only: no vision tower, projector, or image-token splice is
// implemented, and none is claimed.
var ErrV41VisionUnsupported = errors.New("model: DeepSeek V4.1 Flash vision path is not implemented")

// DeepSeekV41VisionConfig retains the official checkpoint's vision_config
// markers without admitting an execution path. It is metadata only; its
// presence makes the config a vision request that RefuseDeepSeekV41Vision
// rejects.
type DeepSeekV41VisionConfig struct {
	ModelType      string
	HiddenSize     int
	NumLayers      int
	NumHeads       int
	PatchSize      int
	MaxImageTokens int
}

// HasDeepSeekV41Vision reports whether the config requests the V4.1 vision
// path: either the retained vision_config metadata is present, or the wrapper
// declares an image token. Both are fail-closed markers.
func (c Config) HasDeepSeekV41Vision() bool {
	if c.DeepSeekV41 != nil && c.DeepSeekV41.Vision != nil {
		return true
	}
	return c.ImageTokenID != 0
}

// RefuseDeepSeekV41Vision is the admission hook the V4.1 dispatch seam calls
// before any vision work. It is nil (no-op) for every non-V4.1 family and for
// a V4.1 config that requests no vision, so the text-only path is byte-for-byte
// unchanged. A V4.1 vision request is refused with ErrV41VisionUnsupported.
func (c Config) RefuseDeepSeekV41Vision() error {
	if !c.IsDeepSeekV41() || !c.HasDeepSeekV41Vision() {
		return nil
	}
	detail := "vision_config"
	if c.DeepSeekV41 != nil && c.DeepSeekV41.Vision != nil {
		v := c.DeepSeekV41.Vision
		detail = fmt.Sprintf("%s(depth=%d hidden=%d max_tokens=%d)", v.ModelType, v.NumLayers, v.HiddenSize, v.MaxImageTokens)
	}
	return fmt.Errorf("%w: %s@%s declares %s; only text-only execution is admitted", ErrV41VisionUnsupported, DeepSeekV41FlashModelID, DeepSeekV41FlashRevision, detail)
}
