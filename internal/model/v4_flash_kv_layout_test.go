package model

import (
	"errors"
	"testing"
)

// v4_flash_kv_layout_test.go - the bounded first-slice witness for DeepSeek
// V4-Flash-0731 per-layer KV state (parent #12637, epic #12635).
//
// The published contract at revision 7872f01b1d1fe23eabc4c98b48bffcef5a386062
// gives EVERY attention layer a 128-token circular window, and additionally one
// per-layer compression regime selected by compress_ratios[layer]:
//
//   - ratio 0   -> window only; NO compressor state.
//   - ratio 4   -> window + overlapping compressor + separate indexer cache.
//   - ratio 128 -> window + one non-overlapping compressor.
//
// This file witnesses three invariants:
//
//  1. layout selection: the state layout is chosen from the layer ratio
//     (0 -> window-only, 4 -> overlapping, 128 -> non-overlapping), and any
//     ratio outside {0,4,128} fails closed with a typed error.
//  2. circular window: a 128-token window retains exactly the most recent 128
//     rows across prefill/decode appends, in order, discarding the oldest.
//  3. schema identity: state encoded for one ratio cannot be restored into a
//     layer declaring a different ratio.

func TestV4FlashKVLayoutForRatio(t *testing.T) {
	cases := []struct {
		ratio  int
		want   V4FlashKVLayout
		wantOK bool
	}{
		{ratio: 0, want: V4FlashKVWindowOnly, wantOK: true},
		{ratio: 4, want: V4FlashKVOverlapping, wantOK: true},
		{ratio: 128, want: V4FlashKVNonOverlapping, wantOK: true},
		{ratio: -1, wantOK: false},
		{ratio: 1, wantOK: false},
		{ratio: 3, wantOK: false},
		{ratio: 5, wantOK: false},
		{ratio: 256, wantOK: false},
	}
	for _, tc := range cases {
		got, ok := v4FlashKVLayoutForRatio(tc.ratio)
		if ok != tc.wantOK || (ok && got != tc.want) {
			t.Fatalf("layoutForRatio(%d) = (%v,%v), want (%v,%v)", tc.ratio, got, ok, tc.want, tc.wantOK)
		}
	}
}

func TestV4FlashKVStateRejectsInvalidRatioBeforeAllocation(t *testing.T) {
	cfg := v4FlashTracerConfig([]int{0, 4, 5})
	state, err := newV4FlashKVState(cfg)
	if !errors.Is(err, ErrV4FlashKVStateRatioInvalid) {
		t.Fatalf("newV4FlashKVState error = %v, want ErrV4FlashKVStateRatioInvalid", err)
	}
	if state != nil {
		t.Fatalf("malformed schedule allocated state %v; want nil", state)
	}
}

func TestV4FlashKVStateSelectsPerLayerLayout(t *testing.T) {
	cfg := v4FlashTracerConfig([]int{0, 4, 128})
	state, err := newV4FlashKVState(cfg)
	if err != nil {
		t.Fatalf("newV4FlashKVState error: %v", err)
	}
	if state == nil {
		t.Fatal("newV4FlashKVState returned nil state for a valid schedule")
	}
	want := []V4FlashKVLayout{V4FlashKVWindowOnly, V4FlashKVOverlapping, V4FlashKVNonOverlapping}
	if len(state.Layers) != len(want) {
		t.Fatalf("layers = %d, want %d", len(state.Layers), len(want))
	}
	for l, w := range want {
		if state.Layers[l].Layout != w {
			t.Fatalf("layer %d layout = %v, want %v", l, state.Layers[l].Layout, w)
		}
		if state.Layers[l].Ratio != cfg.CompressRatios[l] {
			t.Fatalf("layer %d ratio = %d, want %d", l, state.Layers[l].Ratio, cfg.CompressRatios[l])
		}
	}
}

func TestV4FlashCircularWindowRetainsMostRecent(t *testing.T) {
	w := newV4FlashCircularWindow(V4FlashWindowSize)
	if w.Capacity() != V4FlashWindowSize {
		t.Fatalf("capacity = %d, want %d", w.Capacity(), V4FlashWindowSize)
	}
	total := V4FlashWindowSize + 7
	for i := 0; i < total; i++ {
		w.Append([]float32{float32(i)})
	}
	if w.Len() != V4FlashWindowSize {
		t.Fatalf("window len = %d after %d appends, want %d", w.Len(), total, V4FlashWindowSize)
	}
	rows := w.Rows()
	if len(rows) != V4FlashWindowSize {
		t.Fatalf("rows = %d, want %d", len(rows), V4FlashWindowSize)
	}
	first := rows[0][0]
	wantFirst := float32(total - V4FlashWindowSize)
	if first != wantFirst {
		t.Fatalf("oldest retained value = %v, want %v", first, wantFirst)
	}
	last := rows[len(rows)-1][0]
	if last != float32(total-1) {
		t.Fatalf("newest retained value = %v, want %v", last, float32(total-1))
	}
	for i := 1; i < len(rows); i++ {
		if rows[i][0] != rows[i-1][0]+1 {
			t.Fatalf("window rows not contiguous at %d: %v then %v", i, rows[i-1][0], rows[i][0])
		}
	}
}

func TestV4FlashKVStateRatioIdentityIsNotRestorable(t *testing.T) {
	encodeCfg := v4FlashTracerConfig([]int{4})
	state, err := newV4FlashKVState(encodeCfg)
	if err != nil {
		t.Fatalf("newV4FlashKVState error: %v", err)
	}
	blob := state.Encode()

	restoreCfg := v4FlashTracerConfig([]int{128})
	if err := RestoreV4FlashKVState(blob, restoreCfg); !errors.Is(err, ErrV4FlashKVStateRatioMismatch) {
		t.Fatalf("restore into ratio 128 layer error = %v, want ErrV4FlashKVStateRatioMismatch", err)
	}
	if err := RestoreV4FlashKVState(blob, encodeCfg); err != nil {
		t.Fatalf("restore into matching ratio 4 layer error: %v", err)
	}
}
