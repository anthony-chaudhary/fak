package compute

import (
	"math"
	"testing"
)

// fak-test:runtime fast est=10ms
func TestFlashAttentionWindowTileStart(t *testing.T) {
	for kv := 1; kv <= 35; kv++ {
		for pos := 0; pos <= 40; pos++ {
			for window := -1; window <= 40; window++ {
				for tile := 1; tile <= 17; tile++ {
					wantStart, wantSkipped := kv, int64(0)
					for kb := 0; kb < kv; kb += tile {
						end := kb + tile
						if end > kv {
							end = kv
						}
						if window > 0 && end <= pos-window+1 {
							wantSkipped++
							continue
						}
						wantStart = kb
						break
					}
					start, skipped := flashAttentionWindowTileStart(kv, pos, window, tile)
					if start != wantStart || skipped != wantSkipped {
						t.Fatalf("kv=%d pos=%d window=%d tile=%d: (%d,%d), want (%d,%d)", kv, pos, window, tile, start, skipped, wantStart, wantSkipped)
					}
				}
			}
		}
	}
}

// fak-test:runtime fast est=10ms
func TestFlashAttentionWindowTileStartOutputsAndCounters(t *testing.T) {
	for _, layout := range []AttentionMemoryLayout{LayoutPosMajor, LayoutHeadMajor} {
		for _, sizes := range [][2]int{{1, 257}, {5, 257}, {8, 3}} {
			for _, window := range []int{-1, 0, 1, 9, 17, 512} {
				for _, causal := range []bool{false, true} {
					qTokens, kvTokens := sizes[0], sizes[1]
					cfg := FlashAttentionConfig{NumQueryHeads: 4, NumKVHeads: 2, HeadDim: 2, ValueDim: 2, Layout: layout, Causal: causal, SlidingWindow: window, BlockSizeR: 3, BlockSizeC: 8}
					e, err := NewFlashAttentionEngine(cfg)
					if err != nil {
						t.Fatal(err)
					}
					q, k, v := make([]float32, qTokens*8), make([]float32, kvTokens*4), make([]float32, kvTokens*4)
					index := func(pos, head, dim, tokens, heads int) int {
						if layout == LayoutHeadMajor {
							return (head*tokens+pos)*2 + dim
						}
						return (pos*heads+head)*2 + dim
					}
					for pos := 0; pos < kvTokens; pos++ {
						for h := 0; h < 2; h++ {
							for d := 0; d < 2; d++ {
								v[index(pos, h, d, kvTokens, 2)] = float32(pos + h*3 + d)
							}
						}
					}
					out, err := e.Execute(q, k, v, qTokens, kvTokens)
					if err != nil {
						t.Fatal(err)
					}
					var total, skippedWindow, skippedCausal int64
					for qi := 0; qi < qTokens; qi++ {
						pos := qi
						if kvTokens >= qTokens {
							pos += kvTokens - qTokens
						}
						// Preserve the original block-count contract independently of
						// the optimized start helper, including the first causal break.
						for kb := 0; kb < kvTokens; kb += 8 {
							total += 4
							if causal && kb > pos {
								skippedCausal += 4
								break
							}
							end := kb + 8
							if end > kvTokens {
								end = kvTokens
							}
							if window > 0 && end <= pos-window+1 {
								skippedWindow += 4
							}
						}
						for h := 0; h < 4; h++ {
							for d := 0; d < 2; d++ {
								var sum, count float32
								for kj := 0; kj < kvTokens; kj++ {
									if (causal && kj > pos) || (window > 0 && kj < pos-window+1) {
										continue
									}
									sum += v[index(kj, h/2, d, kvTokens, 2)]
									count++
								}
								want := float32(0)
								if count > 0 {
									want = sum / count
								}
								got := out[index(qi, h, d, qTokens, 4)]
								if math.Abs(float64(got-want)) > 1e-4 {
									t.Fatalf("layout=%s sizes=%v window=%d causal=%v qi=%d h=%d d=%d: got %g want %g", layout, sizes, window, causal, qi, h, d, got, want)
								}
							}
						}
					}
					if e.stats.TotalBlocks != total || e.stats.SkippedWindowBlocks != skippedWindow || e.stats.SkippedCausalBlocks != skippedCausal {
						t.Fatalf("layout=%s sizes=%v window=%d causal=%v: stats=%+v want total/window/causal=%d/%d/%d", layout, sizes, window, causal, e.stats, total, skippedWindow, skippedCausal)
					}
				}
			}
		}
	}
}
