package agent

import (
	"math"
	"math/rand"
	"testing"
)

func fullSoftmaxReconstructionOracle(logits []float32, temp, topP float64, topK int, minP float64, rng *rand.Rand, coin float64, hasCoin bool, w *samplerWorkspace) int {
	nonFinite, finite := false, false
	if temp <= 0 {
		best, bi := float32(-math.MaxFloat32), 0
		for i, x := range logits {
			if x != x || x > math.MaxFloat32 {
				nonFinite = true
			} else if x >= -math.MaxFloat32 {
				finite = true
			}
			if x > best {
				best, bi = x, i
			}
		}
		w.nonFinite = nonFinite || !finite
		return bi
	}
	maxL := float32(-math.MaxFloat32)
	for _, x := range logits {
		if x != x || x > math.MaxFloat32 {
			nonFinite = true
		} else if x >= -math.MaxFloat32 {
			finite = true
		}
		if x > maxL {
			maxL = x
		}
	}
	w.nonFinite = nonFinite || !finite
	w.resize(len(logits))
	probs := w.probs
	var sum float64
	for i, x := range logits {
		p := math.Exp(float64(x-maxL) / temp)
		probs[i] = p
		sum += p
	}
	if minP > 0 {
		sum = minPTruncate(probs, sum, minP)
	}
	sum = w.applyCPUSamplingTruncation(probs, sum, topP, topK)
	w.mass = sum
	if !hasCoin {
		coin = rng.Float64()
	}
	return sampleProbabilityRow(probs, sum, coin)
}

// fak-test:runtime slow est=25s lane=default
func TestSmallTopKSoftmaxReconstructionParity(t *testing.T) {
	rng := rand.New(rand.NewSource(418))
	var gotWS, wantWS samplerWorkspace
	for _, n := range []int{1, 2, 65, 257, 4096, 17} {
		for trial := 0; trial < 8; trial++ {
			row := make([]float32, n)
			for i := range row {
				row[i] = float32(rng.NormFloat64() * 8)
			}
			switch trial {
			case 1:
				for i := range row {
					row[i] = 0
				}
			case 2:
				row[0] = float32(math.Inf(-1))
			case 3:
				row[0] = float32(math.NaN())
			case 4:
				row[0] = float32(math.Inf(1))
			case 5:
				for i := range row {
					row[i] = float32(math.Inf(-1))
				}
			case 6:
				row[0] = math.MaxFloat32
			case 7:
				for i := range row {
					row[i] = float32(i % 3)
				}
			}
			for _, temp := range []float64{0, .7, 1e-300, 1e300, math.Inf(1), math.NaN()} {
				for _, k := range []int{0, 1, 2, 40, 64, 65, n} {
					for _, minP := range []float64{0, .05, 1, 2, math.NaN()} {
						for _, topP := range []float64{0, .9, 1} {
							// Match workspace reuse, including greedy calls that retain old rows.
							got := sampleLogitsMinPCoreWS(row, temp, topP, k, minP, nil, .3, true, &gotWS)
							want := fullSoftmaxReconstructionOracle(row, temp, topP, k, minP, nil, .3, true, &wantWS)
							if got != want || gotWS.nonFinite != wantWS.nonFinite || math.Float64bits(gotWS.mass) != math.Float64bits(wantWS.mass) {
								t.Fatalf("n=%d trial=%d temp=%g k=%d minP=%g topP=%g token=%d/%d mass=%g/%g", n, trial, temp, k, minP, topP, got, want, gotWS.mass, wantWS.mass)
							}
							if len(gotWS.probs) != len(wantWS.probs) {
								t.Fatal("workspace lengths differ")
							}
							for i := range gotWS.probs {
								if math.Float64bits(gotWS.probs[i]) != math.Float64bits(wantWS.probs[i]) {
									t.Fatalf("weight differs n=%d trial=%d temp=%g k=%d minP=%g topP=%g id=%d", n, trial, temp, k, minP, topP, i)
								}
							}
							if temp > 0 && !gotWS.nonFinite && gotWS.mass > 0 && !math.IsInf(gotWS.mass, 0) {
								var cumulative float64
								for i, weight := range wantWS.probs {
									cumulative += weight
									if weight == 0 {
										continue
									}
									boundary := cumulative / wantWS.mass
									for _, coin := range []float64{math.Nextafter(boundary, 0), boundary, math.Nextafter(boundary, 1)} {
										a := sampleProbabilityRow(gotWS.probs, gotWS.mass, coin)
										b := sampleProbabilityRow(wantWS.probs, wantWS.mass, coin)
										if a != b {
											t.Fatalf("CDF differs at %d", i)
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestSmallTopKSoftmaxReconstructionAdmission(t *testing.T) {
	var w samplerWorkspace
	for _, tc := range []struct {
		row    []float32
		temp   float64
		k      int
		accept bool
	}{
		{[]float32{0, 1, 2, 3}, 1, 2, true},
		{[]float32{3, 3, 3, 1}, 1, 2, false},
		{[]float32{3, 3, 2, 1}, 1, 2, true},
		{[]float32{0, 1, 1, 3}, 1, 2, false},
		{[]float32{0, 1, 2, math.MaxFloat32}, 1, 2, false},
		{[]float32{0, 1, 2, 3}, 1e300, 2, false},
		{[]float32{float32(math.Inf(-1)), 1, 2, 3}, 1, 2, true},
	} {
		w.resize(len(tc.row))
		maxL := tc.row[0]
		for _, x := range tc.row {
			if x > maxL {
				maxL = x
			}
		}
		_, ok := w.smallTopKSoftmax(tc.row, maxL, tc.temp, tc.k)
		if ok != tc.accept {
			t.Fatalf("admission=%v want=%v row=%v", ok, tc.accept, tc.row)
		}
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestSmallTopKSoftmaxReconstructionRNGAndAllocs(t *testing.T) {
	row := make([]float32, 4096)
	for i := range row {
		row[i] = float32(i) * .001
	}
	var w, ref samplerWorkspace
	a, b := rand.New(rand.NewSource(55)), rand.New(rand.NewSource(55))
	for i := 0; i < 100; i++ {
		got := sampleLogitsMinPCoreWS(row, .7, .9, 40, .05, a, 0, false, &w)
		want := fullSoftmaxReconstructionOracle(row, .7, .9, 40, .05, b, 0, false, &ref)
		if got != want {
			t.Fatal("seeded token mismatch")
		}
	}
	if a.Int63() != b.Int63() {
		t.Fatal("RNG state differs")
	}
	allocs := testing.AllocsPerRun(100, func() { sampleLogitsMinPCoreWS(row, .7, .9, 40, .05, nil, .5, true, &w) })
	if allocs != 0 {
		t.Fatalf("allocs=%g", allocs)
	}
}

// BenchmarkSmallTopKSoftmaxReconstruction compares the independent candidate
// with its literal baseline, including strict-gap refusal and disabled paths.
func BenchmarkSmallTopKSoftmaxReconstruction(b *testing.B) {
	for _, shape := range []string{"strict-gap", "tied-cutoff", "mixed-tie"} {
		row := make([]float32, 151936)
		rng := rand.New(rand.NewSource(418))
		for i := range row {
			row[i] = float32(rng.NormFloat64() * 2)
		}
		if shape == "tied-cutoff" || shape == "mixed-tie" {
			clear(row)
		}
		if shape == "mixed-tie" {
			row[len(row)-1] = 1
		}
		for _, mode := range []struct {
			name string
			k    int
		}{{"k40", 40}, {"k0", 0}, {"k65", 65}} {
			for _, candidate := range []bool{false, true} {
				impl := "baseline"
				if candidate {
					impl = "candidate"
				}
				b.Run(shape+"/"+mode.name+"/"+impl, func(b *testing.B) {
					var w samplerWorkspace
					sample := fullSoftmaxReconstructionOracle
					if candidate {
						sample = sampleLogitsMinPCoreWS
					}
					sample(row, .7, .9, mode.k, .05, nil, .3, true, &w)
					b.ReportAllocs()
					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						sample(row, .7, .9, mode.k, .05, nil, .3, true, &w)
					}
				})
			}
		}
	}
}
