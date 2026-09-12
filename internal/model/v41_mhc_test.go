package model

import (
	"math"
	"testing"
)

func TestV41MHCSplitSinkhornMatchesScalarOracle(t *testing.T) {
	const hc = 4
	mixes, base := make([]float32, 24), make([]float32, 24)
	for i := range mixes {
		mixes[i] = float32(i-9) / 7
		base[i] = float32((i%5)-2) / 11
	}
	scale := []float32{0.75, -0.5, 1.25}
	got, err := v41MHCSplit(mixes, scale, base, hc, 3, 1e-6)
	if err != nil {
		t.Fatal(err)
	}
	want := oracleV41MHCSplit(mixes, scale, base, hc, 3, 1e-6)
	for _, pair := range [][2][]float32{{got.pre, want.pre}, {got.post, want.post}, {got.comb, want.comb}} {
		for i := range pair[0] {
			if math.Abs(float64(pair[0][i]-pair[1][i])) > 1e-7 {
				t.Fatalf("coefficient[%d]=%g want %g", i, pair[0][i], pair[1][i])
			}
		}
	}
}

func TestV41MHCPrePostAndHeadCollapse(t *testing.T) {
	streams := [][]float32{{1, 2}, {3, 5}, {7, 11}, {13, 17}}
	pre := []float32{1, 2, 3, 4}
	collapsed, err := v41MHCPre(streams, pre)
	if err != nil {
		t.Fatal(err)
	}
	if collapsed[0] != 80 || collapsed[1] != 113 {
		t.Fatalf("head collapse=%v", collapsed)
	}
	post := []float32{1, 2, 3, 4}
	// Deliberately asymmetric: rows are source streams and columns are
	// destinations, matching sum_j comb[j,k]*residual[j].
	comb := []float32{1, 2, 0, 0, 0, 1, 3, 0, 0, 0, 1, 4, 5, 0, 0, 1}
	got, err := v41MHCPost([]float32{2, 3}, streams, post, comb)
	if err != nil {
		t.Fatal(err)
	}
	want := [][]float32{{68, 90}, {9, 15}, {22, 35}, {49, 73}}
	for h := range want {
		for d := range want[h] {
			if got[h][d] != want[h][d] {
				t.Fatalf("post[%d][%d]=%g want %g", h, d, got[h][d], want[h][d])
			}
		}
	}
}

func oracleV41MHCSplit(mixes, scale, base []float32, hc, iters int, eps float32) v41MHCMix {
	o := v41MHCMix{pre: make([]float32, hc), post: make([]float32, hc), comb: make([]float32, hc*hc)}
	for j := 0; j < hc; j++ {
		o.pre[j] = float32(1/(1+math.Exp(float64(-(mixes[j]*scale[0]+base[j]))))) + eps
		o.post[j] = 2 * float32(1/(1+math.Exp(float64(-(mixes[hc+j]*scale[1]+base[hc+j])))))
	}
	for row := 0; row < hc; row++ {
		var maxv float32 = -math.MaxFloat32
		for col := 0; col < hc; col++ {
			i := row*hc + col
			o.comb[i] = mixes[2*hc+i]*scale[2] + base[2*hc+i]
			if o.comb[i] > maxv {
				maxv = o.comb[i]
			}
		}
		var sum float32
		for col := 0; col < hc; col++ {
			i := row*hc + col
			o.comb[i] = float32(math.Exp(float64(o.comb[i] - maxv)))
			sum += o.comb[i]
		}
		for col := 0; col < hc; col++ {
			i := row*hc + col
			o.comb[i] = o.comb[i]/sum + eps
		}
	}
	for iter := 0; iter < iters; iter++ {
		if iter > 0 {
			for r := 0; r < hc; r++ {
				var s float32
				for c := 0; c < hc; c++ {
					s += o.comb[r*hc+c]
				}
				for c := 0; c < hc; c++ {
					o.comb[r*hc+c] /= s + eps
				}
			}
		}
		for c := 0; c < hc; c++ {
			var s float32
			for r := 0; r < hc; r++ {
				s += o.comb[r*hc+c]
			}
			for r := 0; r < hc; r++ {
				o.comb[r*hc+c] /= s + eps
			}
		}
	}
	return o
}
