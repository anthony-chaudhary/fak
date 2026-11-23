package compute

import "fmt"

// PackedMatVecAlign is the float alignment every input and output segment of a
// packed MatVec batch must start at: 64 floats keeps each segment as aligned
// as a standalone pooled allocation for every GEMV kernel.
const PackedMatVecAlign = 64

// PackedMatVecer runs a batch of independent decode GEMVs in one host round
// trip. x packs every input segment and the returned slice packs every output:
// y[yOff[i] : yOff[i]+ws[i].Shape[0]] = ws[i] · x[xOff[i] : xOff[i]+ws[i].Shape[1]].
// Offsets must be multiples of PackedMatVecAlign.
type PackedMatVecer interface {
	MatVecPacked(ws []Tensor, x []float32, xOff, yOff []int, yLen int) ([]float32, error)
}

// CheckPackedMatVec validates a packed batch's geometry.
func CheckPackedMatVec(ws []Tensor, x []float32, xOff, yOff []int, yLen int) error {
	if len(ws) == 0 || len(xOff) != len(ws) || len(yOff) != len(ws) {
		return fmt.Errorf("compute: packed matvec needs one x and y offset per weight (%d weights, %d, %d)", len(ws), len(xOff), len(yOff))
	}
	for i, w := range ws {
		if len(w.Shape) != 2 {
			return fmt.Errorf("compute: packed matvec weight %d has shape %v", i, w.Shape)
		}
		if xOff[i]%PackedMatVecAlign != 0 || yOff[i]%PackedMatVecAlign != 0 {
			return fmt.Errorf("compute: packed matvec weight %d offsets (%d, %d) not %d-aligned", i, xOff[i], yOff[i], PackedMatVecAlign)
		}
		if xOff[i] < 0 || xOff[i]+w.Shape[1] > len(x) || yOff[i] < 0 || yOff[i]+w.Shape[0] > yLen {
			return fmt.Errorf("compute: packed matvec weight %d [%d %d] out of range (x %d at %d, y %d at %d)",
				i, w.Shape[0], w.Shape[1], len(x), xOff[i], yLen, yOff[i])
		}
	}
	return nil
}

// SituMoE runs a routed expert MLP with the situ activation in one host round
// trip. For pick p, with g, u the halves of guScale[p]*gateUp[p]·x,
//
//	act_p = beta*tanh(g/beta)*sigmoid(g) * (linBeta > 0 ? linBeta*tanh(u/linBeta) : u)
//	out   = sum_p weight[p] * (dnScale[p] * down[p]·act_p)
//
// rounding in that order (float32 scales and sum, float64 activation).
type SituMoE interface {
	SituMoE(gateUp, down []Tensor, x []float32, weight, guScale, dnScale []float32, beta, linBeta float32) ([]float32, error)
}

// CheckSituMoE validates a situ MoE call's geometry: every gate-up shares one
// [2*width, in] shape and every down one [hidden, >=width] shape.
func CheckSituMoE(gateUp, down []Tensor, x []float32, weight, guScale, dnScale []float32) error {
	p := len(gateUp)
	if p == 0 || len(down) != p || len(weight) != p || len(guScale) != p || len(dnScale) != p {
		return fmt.Errorf("compute: situ moe needs equal gate-up, down, weight and scale counts (%d, %d, %d, %d, %d)",
			p, len(down), len(weight), len(guScale), len(dnScale))
	}
	g0, d0 := gateUp[0].Shape, down[0].Shape
	if len(g0) != 2 || len(d0) != 2 || g0[0]%2 != 0 || d0[1] < g0[0]/2 || len(x) > g0[1] {
		return fmt.Errorf("compute: situ moe gate-up %v / down %v / x %d do not chain", g0, d0, len(x))
	}
	for i := range gateUp {
		g, d := gateUp[i].Shape, down[i].Shape
		if len(g) != 2 || len(d) != 2 || g[0] != g0[0] || g[1] != g0[1] || d[0] != d0[0] || d[1] != d0[1] {
			return fmt.Errorf("compute: situ moe pick %d shapes %v / %v differ from %v / %v", i, g, d, g0, d0)
		}
	}
	return nil
}
