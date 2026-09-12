package model

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
)

// v4_flash_kv_layout.go - the first bounded slice of DeepSeek-V4-Flash-0731
// per-layer KV state (parent #12637, epic #12635).
//
// The published contract at revision 7872f01b1d1fe23eabc4c98b48bffcef5a386062
// gives EVERY attention layer a 128-token circular window, and additionally one
// per-layer compression regime selected by compress_ratios[layer]:
//
//   - ratio 0   -> window only; NO compressor state.
//   - ratio 4   -> window + overlapping compressor + separate indexer cache.
//   - ratio 128 -> window + one non-overlapping compressor.
//
// This file owns the LAYOUT decision and the shared 128-token circular window.
// A ratio outside the published closed set {0,4,128} fails closed with a typed
// error rather than selecting a generic path. The compressor/indexer BODIES for
// ratios 4 and 128 are deliberately not implemented here (parent #12637 is a
// planning track dispatched as bounded leaves); this slice only guarantees the
// state layout is chosen from the schedule and that state identity is enforced
// on restore.

var (
	// ErrV4FlashKVStateRatioInvalid reports malformed per-layer metadata: a
	// compression ratio outside the published closed set {0,4,128}. It is the
	// fail-closed verdict for a state layout request.
	ErrV4FlashKVStateRatioInvalid = errors.New("model: DeepSeek V4 Flash compression ratio is invalid for KV state")

	// ErrV4FlashKVStateRatioMismatch reports an attempt to restore state that
	// was encoded for one layer ratio into a layer declaring another. State
	// encoded for ratio 4 must never be restored into a ratio-128 layer.
	ErrV4FlashKVStateRatioMismatch = errors.New("model: DeepSeek V4 Flash KV state was encoded for a different layer ratio")
)

// V4FlashKVLayout is the closed per-layer KV state layout selected by one
// compression ratio.
type V4FlashKVLayout int

const (
	// V4FlashKVWindowOnly is ratio 0: the 128-token circular window with no
	// compressor.
	V4FlashKVWindowOnly V4FlashKVLayout = iota + 1
	// V4FlashKVOverlapping is ratio 4: window plus an overlapping compressor
	// and a separate indexer cache.
	V4FlashKVOverlapping
	// V4FlashKVNonOverlapping is ratio 128: window plus one non-overlapping
	// compressor.
	V4FlashKVNonOverlapping
)

// String renders the layout for witnesses and error messages.
func (l V4FlashKVLayout) String() string {
	switch l {
	case V4FlashKVWindowOnly:
		return "window-only"
	case V4FlashKVOverlapping:
		return "overlapping"
	case V4FlashKVNonOverlapping:
		return "non-overlapping"
	default:
		return fmt.Sprintf("layout(%d)", int(l))
	}
}

// v4FlashKVLayoutForRatio maps one published compression ratio onto its KV
// state layout. ok is false for any value outside the closed set {0,4,128}.
func v4FlashKVLayoutForRatio(ratio int) (V4FlashKVLayout, bool) {
	switch ratio {
	case 0:
		return V4FlashKVWindowOnly, true
	case 4:
		return V4FlashKVOverlapping, true
	case 128:
		return V4FlashKVNonOverlapping, true
	default:
		return 0, false
	}
}

// V4FlashKVLayerState is one layer's KV state identity and 128-token circular
// window. Compressor/indexer bodies for ratios 4 and 128 are not attached yet;
// the layout field is the contract this slice fixes.
type V4FlashKVLayerState struct {
	Layer  int
	Ratio  int
	Layout V4FlashKVLayout
	Window *V4FlashCircularWindow
}

// V4FlashKVState is the session-owned per-layer KV state for a Flash schedule.
type V4FlashKVState struct {
	WindowSize int
	Layers     []V4FlashKVLayerState
}

// newV4FlashKVState selects one layer state per compress_ratios entry. It fails
// closed (nil, ErrV4FlashKVStateRatioInvalid) on any ratio outside {0,4,128}
// BEFORE allocating layer state, so a malformed schedule never yields a partial
// state object.
func newV4FlashKVState(cfg Config) (*V4FlashKVState, error) {
	if !cfg.IsDeepSeekV4() || len(cfg.CompressRatios) == 0 {
		return nil, nil
	}
	layouts := make([]V4FlashKVLayout, len(cfg.CompressRatios))
	for l, ratio := range cfg.CompressRatios {
		layout, ok := v4FlashKVLayoutForRatio(ratio)
		if !ok {
			return nil, fmt.Errorf("%w: layer %d has ratio %d outside {0,4,128}", ErrV4FlashKVStateRatioInvalid, l+1, ratio)
		}
		layouts[l] = layout
	}
	state := &V4FlashKVState{
		WindowSize: V4FlashWindowSize,
		Layers:     make([]V4FlashKVLayerState, len(cfg.CompressRatios)),
	}
	for l, ratio := range cfg.CompressRatios {
		state.Layers[l] = V4FlashKVLayerState{
			Layer:  l + 1,
			Ratio:  ratio,
			Layout: layouts[l],
			Window: newV4FlashCircularWindow(V4FlashWindowSize),
		}
	}
	return state, nil
}

// v4FlashKVStateEncoding is the serialized form. The per-layer ratios are the
// identity that must survive a restore; the window rows are the state payload.
type v4FlashKVStateEncoding struct {
	Schema     string                   `json:"schema"`
	WindowSize int                      `json:"window_size"`
	Layers     []v4FlashKVLayerEncoding `json:"layers"`
}

type v4FlashKVLayerEncoding struct {
	Ratio  int         `json:"ratio"`
	Layout int         `json:"layout"`
	Rows   [][]float32 `json:"rows"`
}

// Encode serializes the state so its layer ratios and window rows can be
// restored, and so a mismatch can be detected before adoption.
func (s *V4FlashKVState) Encode() []byte {
	enc := v4FlashKVStateEncoding{Schema: "fak.v4flash-kv-state.v1", WindowSize: s.WindowSize}
	for _, l := range s.Layers {
		enc.Layers = append(enc.Layers, v4FlashKVLayerEncoding{
			Ratio:  l.Ratio,
			Layout: int(l.Layout),
			Rows:   l.Window.Rows(),
		})
	}
	blob, _ := json.Marshal(enc)
	return blob
}

// RestoreV4FlashKVState adopts an encoded state only when the encoding's
// per-layer ratios match the target config exactly. A state encoded for one
// ratio can never be restored into a layer declaring another.
func RestoreV4FlashKVState(blob []byte, cfg Config) error {
	var enc v4FlashKVStateEncoding
	if err := json.Unmarshal(blob, &enc); err != nil {
		return fmt.Errorf("%w: %v", ErrV4FlashKVStateRatioMismatch, err)
	}
	if enc.Schema != "fak.v4flash-kv-state.v1" {
		return fmt.Errorf("%w: schema %q", ErrV4FlashKVStateRatioMismatch, enc.Schema)
	}
	if len(enc.Layers) != len(cfg.CompressRatios) {
		return fmt.Errorf("%w: %d encoded layers vs %d configured", ErrV4FlashKVStateRatioMismatch, len(enc.Layers), len(cfg.CompressRatios))
	}
	for l, layer := range enc.Layers {
		if layer.Ratio != cfg.CompressRatios[l] {
			return fmt.Errorf("%w: layer %d encoded ratio %d, config ratio %d", ErrV4FlashKVStateRatioMismatch, l+1, layer.Ratio, cfg.CompressRatios[l])
		}
		layout, ok := v4FlashKVLayoutForRatio(layer.Ratio)
		if !ok || int(layout) != layer.Layout {
			return fmt.Errorf("%w: layer %d layout %d does not match ratio %d", ErrV4FlashKVStateRatioMismatch, l+1, layer.Layout, layer.Ratio)
		}
	}
	return nil
}

// V4FlashCircularWindow is the 128-token circular window shared by every Flash
// attention layer. It retains the most recent capacity rows in order; appending
// past capacity discards the oldest row.
type V4FlashCircularWindow struct {
	capacity int
	rows     [][]float32
	start    int
	count    int
}

// newV4FlashCircularWindow allocates a window of the given capacity.
func newV4FlashCircularWindow(capacity int) *V4FlashCircularWindow {
	if capacity < 0 {
		capacity = 0
	}
	return &V4FlashCircularWindow{capacity: capacity, rows: make([][]float32, capacity)}
}

// Capacity is the maximum number of rows the window retains.
func (w *V4FlashCircularWindow) Capacity() int { return w.capacity }

// Len is the number of rows currently retained.
func (w *V4FlashCircularWindow) Len() int {
	if w == nil {
		return 0
	}
	return w.count
}

// Append inserts one row at the newest end, discarding the oldest row when the
// window is full. The row slice is copied so later mutation cannot alias it.
func (w *V4FlashCircularWindow) Append(row []float32) {
	if w == nil || w.capacity == 0 {
		return
	}
	cp := append([]float32(nil), row...)
	if w.count < w.capacity {
		w.rows[(w.start+w.count)%w.capacity] = cp
		w.count++
		return
	}
	w.rows[w.start] = cp
	w.start = (w.start + 1) % w.capacity
}

// Rows returns the retained rows oldest-first. The returned slice shares storage
// with the window; callers must not mutate it.
func (w *V4FlashCircularWindow) Rows() [][]float32 {
	if w == nil || w.count == 0 {
		return nil
	}
	out := make([][]float32, w.count)
	for i := 0; i < w.count; i++ {
		out[i] = w.rows[(w.start+i)%w.capacity]
	}
	return out
}

// encodeU32LE is a small helper kept beside the state code so byte-order is
// explicit if a later slice persists the window to disk.
func encodeU32LE(v uint32) []byte {
	var b [4]byte
	binary.LittleEndian.PutUint32(b[:], v)
	return b[:]
}
