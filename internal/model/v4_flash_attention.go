package model

import (
	"errors"
	"fmt"
)

// v4_flash_attention.go - the first bounded slice of DeepSeek-V4-Flash-0731
// per-layer attention (parent #12636, epic #12635).
//
// The published contract at revision 7872f01b1d1fe23eabc4c98b48bffcef5a386062
// gives every attention layer a 128-token sliding window and EXACTLY ONE
// compression ratio per layer (Config.CompressRatios):
//
//   - ratio 0   -> window only. This is the smallest token tracer and the only
//     regime implemented here; it runs through the existing SWA window path.
//   - ratio 4   -> overlapping compressor + 64-head indexer (top-k 512).
//   - ratio 128 -> non-overlapping compressor, no indexer.
//
// Ratios 4 and 128 carry compressor/indexer KV state that is not implemented
// yet. They MUST fail closed: a V4 session whose schedule contains either must
// refuse before any prefill/decode, never fall through to the generic Q/K/V
// attention path (the defect #12636 exists to prevent).

// V4FlashWindowSize is the published sliding-window width for every
// DeepSeek-V4-Flash-0731 attention layer ("sliding_window": 128).
const V4FlashWindowSize = 128

var (
	// ErrV4FlashAttentionUnimplemented reports that a DeepSeek-V4 session
	// carries an attention compression ratio whose native forward is not
	// implemented. It is the typed, fail-closed quarantine: the model stays
	// loadable at component level while the full token path is connected.
	ErrV4FlashAttentionUnimplemented = errors.New("model: DeepSeek V4 Flash compressed attention is not implemented")

	// ErrV4FlashAttentionRatioInvalid reports malformed per-layer metadata: a
	// compression ratio outside the published closed set {0,4,128}.
	ErrV4FlashAttentionRatioInvalid = errors.New("model: DeepSeek V4 Flash compression ratio is invalid")
)

// V4FlashAttentionRegime is the closed per-layer attention regime selected by
// one compression ratio.
type V4FlashAttentionRegime int

const (
	// V4FlashAttentionWindowOnly is ratio 0: the 128-token sliding window with
	// no compressor. Implemented (the ratio-0 tracer).
	V4FlashAttentionWindowOnly V4FlashAttentionRegime = iota + 1
	// V4FlashAttentionCompressed is ratio 4 or 128: window plus a compressor
	// plane and, for ratio 4, an indexer. Metadata-valid but unimplemented.
	V4FlashAttentionCompressed
)

// v4FlashAttentionRegime classifies one published compression ratio. ok is
// false for any value outside the closed set {0,4,128}.
func v4FlashAttentionRegime(ratio int) (V4FlashAttentionRegime, bool) {
	switch ratio {
	case 0:
		return V4FlashAttentionWindowOnly, true
	case 4, 128:
		return V4FlashAttentionCompressed, true
	default:
		return 0, false
	}
}

// v4FlashCompressedLayerRatios returns the 1-based layer numbers whose
// compression ratio is metadata-valid but unimplemented, and the layer whose
// ratio is outside the published closed set (0 if none). A non-V4 config, or a
// V4 config without a per-layer schedule (the V4 Pro path), yields no layers.
func v4FlashCompressedLayerRatios(cfg Config) (compressed []int, invalidLayer int) {
	if !cfg.IsDeepSeekV4() || len(cfg.CompressRatios) == 0 {
		return nil, 0
	}
	for l, ratio := range cfg.CompressRatios {
		regime, ok := v4FlashAttentionRegime(ratio)
		switch {
		case !ok:
			if invalidLayer == 0 {
				invalidLayer = l + 1
			}
		case regime == V4FlashAttentionCompressed:
			compressed = append(compressed, l+1)
		}
	}
	return compressed, invalidLayer
}

// refuseUnimplementedV4FlashAttention is the fail-closed admission every V4
// session runs before its first prefill/decode. It returns:
//
//   - ErrV4FlashAttentionRatioInvalid when any layer declares a ratio outside
//     {0,4,128} (malformed metadata);
//   - ErrV4FlashAttentionUnimplemented when any layer declares ratio 4 or 128
//     (metadata-valid, native forward not implemented).
//
// A non-V4 config, and a ratio-0-only schedule, return nil. It never falls
// through to the generic attention path for an unimplemented regime.
func (s *Session) refuseUnimplementedV4FlashAttention() error {
	if s == nil || s.M == nil {
		return nil
	}
	cfg := s.M.Cfg
	compressed, invalid := v4FlashCompressedLayerRatios(cfg)
	if invalid != 0 {
		return fmt.Errorf("%w: layer %d has ratio outside {0,4,128}", ErrV4FlashAttentionRatioInvalid, invalid)
	}
	if len(compressed) > 0 {
		return fmt.Errorf("%w: %d of %d layers carry ratio 4/128 (first layer %d)", ErrV4FlashAttentionUnimplemented, len(compressed), len(cfg.CompressRatios), compressed[0])
	}
	return nil
}
