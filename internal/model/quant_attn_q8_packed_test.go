package model

import (
	"fmt"
	"math"
	"math/rand"
	"reflect"
	"sync"
	"testing"
)

func q8PackedFixture(grp, hd, positions int) (*KVCache, []float32) {
	cfg := Config{NumLayers: 1, NumHeads: 2 * grp, NumKVHeads: 2, HeadDim: hd, HiddenSize: 2 * grp * hd, RopeTheta: 10000}
	cache := NewKVCacheWithPrecision(cfg, KVPrecisionQ8_0)
	rng := rand.New(rand.NewSource(1416801))
	for pos := 0; pos < positions; pos++ {
		k, v := make([]float32, 2*hd), make([]float32, 2*hd)
		for i := range k {
			k[i], v[i] = float32(rng.NormFloat64()), float32(rng.NormFloat64())
		}
		cache.appendKV(0, k, v)
		cache.appendPosition(pos, pos)
	}
	q := make([]float32, cfg.NumHeads*hd)
	for i := range q {
		q[i] = float32(rng.NormFloat64())
	}
	return cache, q
}

// Independent reference: materialize with the old cache reader, and use existing
// score and value helpers. Never use a packed helper to construct expected data.
func q8PackedReference(cache *KVCache, q []float32, grp, hd, window int, scoreDot func([]float32, []float32) float32) ([]float32, [][]float32) {
	keys, values := cache.attentionRows(0)
	nH, w, nPos := len(q)/hd, cache.kvStride(), cache.kvLen(0)
	first := windowLoContig(nPos, nPos-1, window)
	out, scores := make([]float32, len(q)), make([][]float32, nH)
	for h := 0; h < nH; h++ {
		scores[h] = make([]float32, nPos-first)
		fillSoftmaxAttentionScores(scores[h], q[h*hd:(h+1)*hd], keys, first, nPos, w, h/grp, hd, 1/float32(math.Sqrt(float64(hd))), scoreDot)
		for pos := first; pos < nPos; pos++ {
			for d := 0; d < hd; d++ {
				out[h*hd+d] += scores[h][pos-first] * values[pos*w+(h/grp)*hd+d]
			}
		}
	}
	return out, scores
}

func q8PackedNear(t *testing.T, got, want []float32, atol, rtol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("length %d != %d", len(got), len(want))
	}
	for i := range got {
		err := math.Abs(float64(got[i]) - float64(want[i]))
		if !q8PackedFiniteClose(got[i], want[i], atol, rtol) {
			t.Fatalf("element %d got=%g want=%g error=%g", i, got[i], want[i], err)
		}
	}
}

// Reject nonfinite references as well as results: NaN comparisons and an
// infinite relative tolerance must never turn a broken oracle into a pass.
func q8PackedFiniteClose(got, want float32, atol, rtol float64) bool {
	g, w := float64(got), float64(want)
	if math.IsNaN(g) || math.IsInf(g, 0) || math.IsNaN(w) || math.IsInf(w, 0) {
		return false
	}
	return math.Abs(g-w) <= atol+rtol*math.Abs(w)
}

func TestQ8PackedComparisonRejectsNonfinite(t *testing.T) {
	for _, bad := range []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1))} {
		if q8PackedFiniteClose(1, bad, 3e-5, 2e-5) || q8PackedFiniteClose(bad, 1, 3e-5, 2e-5) || q8PackedFiniteClose(bad, bad, 3e-5, 2e-5) {
			t.Fatal("nonfinite result or reference admitted")
		}
	}
	if !q8PackedFiniteClose(1, 1, 0, 0) || q8PackedFiniteClose(1, 2, 3e-5, 2e-5) {
		t.Fatal("finite comparison policy changed")
	}
}

func TestQ8PackedAttentionReference(t *testing.T) {
	for _, grp := range []int{1, 3, 7} {
		for _, hd := range []int{32, 64, 128} {
			for _, positions := range []int{1, 17, 129} {
				for _, window := range []int{-1, 1, 11} {
					t.Run(fmt.Sprintf("g%d_d%d_p%d_w%d", grp, hd, positions, window), func(t *testing.T) {
						cache, q := q8PackedFixture(grp, hd, positions)
						kBefore, vBefore := clonePacked(&cache.kQ8[0], 0), clonePacked(&cache.vQ8[0], 0)
						bytesBefore := cache.KVCacheResidentBytes()
						qBefore := append([]float32(nil), q...)
						want, wantScores := q8PackedReference(cache, q, grp, hd, window, dot)
						wantSIMD, _ := q8PackedReference(cache, q, grp, hd, window, fdot)
						got := make([]float32, len(q))
						for i := range got {
							got[i] = 99
						} // output must be overwritten
						seen := make([][]float32, len(q)/hd)
						var mu sync.Mutex
						obs := func(layer, queryPos, head int, keys []int, weights []float32) {
							mu.Lock()
							defer mu.Unlock()
							if layer != 0 || queryPos != positions-1 || head < 0 || head >= len(seen) {
								t.Errorf("bad observer coordinates")
								return
							}
							if seen[head] != nil {
								t.Errorf("duplicate observer head %d", head)
							}
							first := windowLoContig(positions, positions-1, window)
							for i, key := range keys {
								if key != first+i {
									t.Errorf("wrong key %d", key)
								}
							}
							seen[head] = append([]float32(nil), weights...)
							// Must not corrupt the consumer's post-softmax score scratch.
							for i := range weights {
								weights[i] = -999
								keys[i] = -999
							}
						}
						scratch, ok := attnDecodeQ8Packed(got, q, cache, 0, len(q)/hd, hd, 2*hd, grp, window, 1/float32(math.Sqrt(float64(hd))), nil, obs)
						if !ok {
							t.Fatal("valid packed cache declined")
						}
						q8PackedNear(t, got, want, 2e-6, 2e-6)
						q8PackedNear(t, got, wantSIMD, 3e-5, 2e-5)
						for h := range seen {
							q8PackedNear(t, seen[h], wantScores[h], 2e-6, 2e-6)
						}
						_, ok = attnDecodeQ8Packed(got, q, cache, 0, len(q)/hd, hd, 2*hd, grp, window, 1/float32(math.Sqrt(float64(hd))), scratch, nil)
						if !ok {
							t.Fatal("warm call declined")
						}
						q8PackedNear(t, got, want, 2e-6, 2e-6)
						if !reflect.DeepEqual(kBefore, &cache.kQ8[0]) || !reflect.DeepEqual(vBefore, &cache.vQ8[0]) || bytesBefore != cache.KVCacheResidentBytes() || !reflect.DeepEqual(qBefore, q) {
							t.Fatal("read-only inputs changed")
						}
					})
				}
			}
		}
	}
}

func TestQ8PackedAttentionDeclineNoTouch(t *testing.T) {
	for _, mode := range []string{"nil", "f32", "layer", "head", "width", "scales", "values", "empty", "window"} {
		t.Run(mode, func(t *testing.T) {
			cache, q := q8PackedFixture(3, 64, 17)
			layer, hd, w, window := 0, 64, 128, -1
			switch mode {
			case "nil":
				cache = nil
			case "f32":
				cache.ConvertToPrecision(KVPrecisionFP32)
			case "layer":
				layer = 1
			case "head":
				hd = 63
			case "width":
				w = 64
			case "scales":
				cache.kQ8[0].scales = nil
			case "values":
				cache.vQ8[0].codes = nil
			case "empty":
				cache.kQ8[0].codes = nil
				cache.kQ8[0].scales = nil
				cache.vQ8[0].codes = nil
				cache.vQ8[0].scales = nil
			case "window":
				window = 0
			}
			out := make([]float32, len(q))
			out[0] = 123
			before := append([]float32(nil), out...)
			scratch := [][]float32{{456}}
			called := false
			gotScratch, ok := attnDecodeQ8Packed(out, q, cache, layer, 6, hd, w, 3, window, 1, scratch, func(int, int, int, []int, []float32) { called = true })
			if ok || called || !reflect.DeepEqual(out, before) || !reflect.DeepEqual(gotScratch, [][]float32{{456}}) {
				t.Fatal("decline changed state")
			}
		})
	}
}

func TestQ8PackedHeadDonorBlockMapping(t *testing.T) {
	// Include -128 even though the current producer usually uses [-127,127].
	keys := kvPackedRow{width: 64, codes: make([]int8, 128), scales: []float32{0.5, 0, -0.25, 2}}
	for i := range keys.codes {
		keys.codes[i] = int8(i * 7)
	}
	decoded, tile := make([]float32, 64), make([]float32, 64)
	for row := 0; row < 2; row++ {
		keys.decodeRowInto(decoded, row)
		q8PackedHeadTile(tile, &keys, row*64)
		q8PackedNear(t, tile, decoded, 0, 0)
	}
}

func BenchmarkQ8PackedAttention(b *testing.B) {
	for _, positions := range []int{128, 2048} {
		for _, packed := range []bool{false, true} {
			b.Run(fmt.Sprintf("positions%d_packed%v", positions, packed), func(b *testing.B) {
				cache, q := q8PackedFixture(7, 64, positions)
				out := make([]float32, len(q))
				var scratch [][]float32
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					if packed {
						var ok bool
						scratch, ok = attnDecodeQ8Packed(out, q, cache, 0, 14, 64, 128, 7, -1, 0.125, scratch, nil)
						if !ok {
							b.Fatal("declined")
						}
					} else {
						clear(out)
						scratch = attnDecodeOne(out, q, cache, 0, 14, 64, 128, 7, 0.125, fdot, fdot3scalar, scratch)
					}
				}
			})
		}
	}
}

// The failed scalar consumer passed loose local tolerances yet failed Session
// logits. Pin the actual native attention branch, not a rewritten dot oracle.
func TestQ8PackedAttentionNativeOrder(t *testing.T) {
	for _, grp := range []int{1, 3, 7} {
		for _, hd := range []int{32, 64, 128} {
			for _, positions := range []int{1, 17, 129} {
				for _, window := range []int{-1, 1, 11} {
					cache, q := q8PackedFixture(grp, hd, positions)
					want, got := make([]float32, len(q)), make([]float32, len(q))
					scale := 1 / float32(math.Sqrt(float64(hd)))
					dot3 := fdot3scalar
					if attnFdot3SIMD {
						dot3 = fdot3SIMD
					}
					attnDecodeBatch(want, q, []*KVCache{cache}, 0, 1, 2*grp, hd, 2*hd, grp, window, scale, fdot, dot3, nil, nil)
					_, ok := attnDecodeQ8Packed(got, q, cache, 0, 2*grp, hd, 2*hd, grp, window, scale, nil, nil)
					if !ok {
						t.Fatal("valid cache declined")
					}
					for i := range got {
						if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
							t.Fatalf("native operation-order mismatch grp=%d hd=%d pos=%d window=%d element=%d got=%g want=%g", grp, hd, positions, window, i, got[i], want[i])
						}
					}
				}
			}
		}
	}
}

func TestQ8PackedNativeReductionTripwire(t *testing.T) {
	rows := kvPackedRow{width: 32, codes: make([]int8, 32), scales: []float32{1}}
	for i := range rows.codes {
		rows.codes[i] = 1
	}
	q := make([]float32, 32)
	q[0], q[1], q[2], q[3] = 1e8, 1, -1e8, 1
	decoded, tile := make([]float32, 32), make([]float32, 32)
	rows.decodeRowInto(decoded, 0)
	q8PackedHeadTile(tile, &rows, 0)
	if dot(q, decoded) == fdot(q, decoded) {
		t.Fatal("cancellation fixture did not distinguish serial and native reduction trees")
	}
	if math.Float32bits(fdot(q, tile)) != math.Float32bits(fdot(q, decoded)) {
		t.Fatal("tile failed native reduction contract")
	}
}
