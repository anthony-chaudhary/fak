package model

import (
	"fmt"
	"math"
	"sync"
	"testing"
)

// fak-test:runtime fast est=1s lane=default
func TestPrefillBatchedScoreScratchOwnershipAndParity(t *testing.T) {
	oldWorkers := NumWorkers()
	defer SetWorkers(oldWorkers)
	bits := func(label string, got, want []float32) {
		t.Helper()
		if len(got) != len(want) {
			t.Fatalf("%s lengths %d != %d", label, len(got), len(want))
		}
		for i := range got {
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				t.Fatalf("%s[%d] bits %08x != %08x", label, i, math.Float32bits(got[i]), math.Float32bits(want[i]))
			}
		}
	}
	type row struct {
		positions []int
		weights   []float32
	}
	for _, workers := range []int{1, 2, 4} {
		if err := SetWorkers(workers); err != nil {
			t.Fatal(err)
		}
		for _, window := range []int{-1, 1, 5, 128} {
			for _, count := range []int{1, 37} {
				t.Run(fmt.Sprintf("workers%d/window%d/count%d", workers, window, count), func(t *testing.T) {
					cfg := llamaArchConfig()
					cfg.Window = []int{window, window}
					cfg.AttnSoftcap = 3
					m := NewSynthetic(cfg)
					prompt := make([]int, count)
					for i := range prompt {
						prompt[i] = (i*17 + 3) % cfg.VocabSize
					}
					// A continuation exercises nonzero base; the empty prefix covers
					// growing rows below the local-window cap.
					for _, prefix := range [][]int{nil, {2, 7, 11}} {
						run := func(batched, mutate bool) (*Session, []float32, map[[3]int]row) {
							m.SetAttnObserver(nil)
							s := m.NewSession()
							for _, id := range prefix {
								s.tokenHidden(id, s.Cache.Len())
							}
							rows := make(map[[3]int]row)
							var mu sync.Mutex
							m.SetAttnObserver(func(l, pos, h int, kp []int, weights []float32) {
								mu.Lock()
								defer mu.Unlock()
								// Retain the actual observer-owned slices: later scratch
								// reuse must not overwrite any previously emitted row.
								rows[[3]int{l, pos, h}] = row{kp, weights}
								if mutate {
									for i := range weights {
										weights[i] = float32(math.NaN())
										kp[i] = -123
									}
								}
							})
							var hidden []float32
							if batched {
								hidden = s.prefillBatched(prompt)
							} else {
								for _, id := range prompt {
									hidden = s.tokenHidden(id, s.Cache.Len())
								}
							}
							m.SetAttnObserver(nil)
							return s, s.head(hidden), rows
						}
						ref, want, wantRows := run(false, false)
						bat, got, gotRows := run(true, false)
						bits("logits", got, want)
						for l := 0; l < cfg.NumLayers; l++ {
							bits("K", bat.Cache.K[l], ref.Cache.K[l])
							bits("Kraw", bat.Cache.Kraw[l], ref.Cache.Kraw[l])
							bits("V", bat.Cache.V[l], ref.Cache.V[l])
						}
						if len(gotRows) != count*cfg.NumLayers*cfg.NumHeads || len(gotRows) != len(wantRows) {
							t.Fatalf("observer row counts %d != %d", len(gotRows), len(wantRows))
						}
						for key, wantRow := range wantRows {
							gotRow, ok := gotRows[key]
							if !ok || len(gotRow.positions) != len(wantRow.positions) {
								t.Fatalf("missing or malformed row %v", key)
							}
							bits("observer weights", gotRow.weights, wantRow.weights)
							for i, pos := range wantRow.positions {
								if gotRow.positions[i] != pos {
									t.Fatalf("observer positions row %v index %d", key, i)
								}
							}
						}
						_, mutated, _ := run(true, true)
						bits("mutating observer logits", mutated, want)
					}
				})
			}
		}
	}
}

// Run this unchanged on the norm-adapter baseline and this candidate. End-to-end
// allocation/time results include all prefill work and must not be called isolated
// attention measurements. No benchmark result is asserted by this source packet.
func BenchmarkPrefillBatchedScoreScratch(b *testing.B) {
	oldWorkers := NumWorkers()
	defer SetWorkers(oldWorkers)
	for _, workers := range []int{1, 4} {
		if err := SetWorkers(workers); err != nil {
			b.Fatal(err)
		}
		for _, window := range []int{-1, 8} {
			b.Run(fmt.Sprintf("workers%d/window%d", workers, window), func(b *testing.B) {
				cfg := llamaArchConfig()
				cfg.Window = []int{window, window}
				m := NewSynthetic(cfg)
				prompt := make([]int, 64)
				for i := range prompt {
					prompt[i] = i % cfg.VocabSize
				}
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					m.NewSession().prefillBatched(prompt)
				}
			})
		}
	}
}
