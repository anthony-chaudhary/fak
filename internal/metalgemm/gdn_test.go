package metalgemm

import (
	"math"
	"testing"
)

// The oracle deliberately owns its types and arithmetic instead of calling the
// production primitive. It runs on every platform and is consumed by the Darwin
// witness as the independent expected result/state implementation.
type oracleGDNGeometry struct{ nK, nV, kHd, vHd, kernel int }

func (g oracleGDNGeometry) keyDim() int   { return g.nK * g.kHd }
func (g oracleGDNGeometry) valueDim() int { return g.nV * g.vHd }
func (g oracleGDNGeometry) convDim() int  { return 2*g.keyDim() + g.valueDim() }

type oracleGDNPanel struct {
	tokens                   int
	mixed, z, b, a           []float32
	conv, aLog, dtBias, norm []float32
	eps                      float32
}

type oracleGDNState struct{ conv, recurrent []float32 }

func newOracleGDNState(g oracleGDNGeometry) *oracleGDNState {
	return &oracleGDNState{
		conv:      make([]float32, (g.kernel-1)*g.convDim()),
		recurrent: make([]float32, g.nV*g.kHd*g.vHd),
	}
}

func oracleSilu(x float32) float32 { return x / (1 + float32(math.Exp(float64(-x)))) }

func oracleSoftplus(x float32) float32 {
	if x > 20 {
		return x
	}
	return float32(math.Log1p(math.Exp(float64(x))))
}

func oracleGDNRun(g oracleGDNGeometry, p oracleGDNPanel, state *oracleGDNState) []float32 {
	keyDim, valueDim, convDim := g.keyDim(), g.valueDim(), g.convDim()
	convOut := make([]float32, p.tokens*convDim)
	for token := 0; token < p.tokens; token++ {
		for channel := 0; channel < convDim; channel++ {
			acc := float32(0)
			for j := 0; j < g.kernel-1; j++ {
				acc += p.conv[channel*g.kernel+j] * state.conv[j*convDim+channel]
			}
			current := p.mixed[token*convDim+channel]
			acc += p.conv[channel*g.kernel+g.kernel-1] * current
			convOut[token*convDim+channel] = oracleSilu(acc)
		}
		if g.kernel > 1 {
			copy(state.conv, state.conv[convDim:])
			copy(state.conv[(g.kernel-2)*convDim:], p.mixed[token*convDim:(token+1)*convDim])
		}
	}

	qNorm := make([]float32, p.tokens*keyDim)
	kNorm := make([]float32, p.tokens*keyDim)
	qScale := float32(1 / math.Sqrt(float64(g.kHd)))
	for token := 0; token < p.tokens; token++ {
		row := convOut[token*convDim:]
		for head := 0; head < g.nK; head++ {
			q, k := row[head*g.kHd:(head+1)*g.kHd], row[keyDim+head*g.kHd:keyDim+(head+1)*g.kHd]
			qSum, kSum := float32(0), float32(0)
			for i := 0; i < g.kHd; i++ {
				qSum += q[i] * q[i]
				kSum += k[i] * k[i]
			}
			qInv := float32(1 / math.Sqrt(float64(qSum)+1e-6))
			kInv := float32(1 / math.Sqrt(float64(kSum)+1e-6))
			for i := 0; i < g.kHd; i++ {
				qNorm[token*keyDim+head*g.kHd+i] = q[i] * qInv * qScale
				kNorm[token*keyDim+head*g.kHd+i] = k[i] * kInv
			}
		}
	}

	core := make([]float32, p.tokens*valueDim)
	repeat := g.nV / g.nK
	for token := 0; token < p.tokens; token++ {
		for head := 0; head < g.nV; head++ {
			keyHead := head / repeat
			q := qNorm[token*keyDim+keyHead*g.kHd:]
			k := kNorm[token*keyDim+keyHead*g.kHd:]
			beta := float32(1 / (1 + math.Exp(float64(-p.b[token*g.nV+head]))))
			decay := float32(math.Exp(float64(-float32(math.Exp(float64(p.aLog[head]))) * oracleSoftplus(p.a[token*g.nV+head]+p.dtBias[head]))))
			readout := make([]float32, g.vHd)
			for d := 0; d < g.vHd; d++ {
				kvmem := float32(0)
				for i := 0; i < g.kHd; i++ {
					index := (head*g.kHd+i)*g.vHd + d
					state.recurrent[index] *= decay
					kvmem += state.recurrent[index] * k[i]
				}
				v := convOut[token*convDim+2*keyDim+head*g.vHd+d]
				delta := (v - kvmem) * beta
				for i := 0; i < g.kHd; i++ {
					index := (head*g.kHd+i)*g.vHd + d
					state.recurrent[index] += k[i] * delta
					readout[d] += state.recurrent[index] * q[i]
				}
			}
			squares := float32(0)
			for _, value := range readout {
				squares += value * value
			}
			inv := float32(1 / math.Sqrt(float64(squares/float32(g.vHd)+p.eps)))
			for d, value := range readout {
				vd := head*g.vHd + d
				core[token*valueDim+vd] = p.norm[d] * value * inv * oracleSilu(p.z[token*valueDim+vd])
			}
		}
	}
	return core
}

func oracleGDNFixture(g oracleGDNGeometry, tokens int, phase float32) oracleGDNPanel {
	fill := func(n int, scale, offset float32) []float32 {
		out := make([]float32, n)
		for i := range out {
			out[i] = float32(math.Sin(float64(float32(i+1)*0.37+phase)))*scale + offset
		}
		return out
	}
	return oracleGDNPanel{
		tokens: tokens,
		mixed:  fill(tokens*g.convDim(), .35, 0), z: fill(tokens*g.valueDim(), .2, .1),
		b: fill(tokens*g.nV, .3, 0), a: fill(tokens*g.nV, .2, -.05),
		conv: fill(g.convDim()*g.kernel, .15, .02), aLog: fill(g.nV, .1, -1.5),
		dtBias: fill(g.nV, .1, -.2), norm: fill(g.vHd, .15, 1), eps: 1e-5,
	}
}

func TestGDNPortableOraclePreservesSplitSequenceState(t *testing.T) {
	g := oracleGDNGeometry{nK: 2, nV: 4, kHd: 4, vHd: 8, kernel: 3}
	first, second := oracleGDNFixture(g, 3, .1), oracleGDNFixture(g, 2, .7)
	state := newOracleGDNState(g)
	one := oracleGDNRun(g, first, state)
	two := oracleGDNRun(g, second, state)
	if len(one) != 3*g.valueDim() || len(two) != 2*g.valueDim() {
		t.Fatalf("oracle output shapes=%d/%d", len(one), len(two))
	}
	if state.conv[0] == 0 || state.recurrent[0] == 0 {
		t.Fatal("oracle did not persist both convolution and recurrent state")
	}
	other := newOracleGDNState(g)
	_ = oracleGDNRun(g, first, other)
	if state.recurrent[0] == other.recurrent[0] {
		t.Fatal("second panel did not advance recurrent ownership")
	}
}
