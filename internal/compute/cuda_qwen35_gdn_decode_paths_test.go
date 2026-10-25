//go:build cuda && cgo

package compute

import (
	"math"
	"testing"
)

// A key head wider than the 128 register slots, or a value head wider than the register
// kernel's 256-thread block, must take the global-state recurrence kernel (no register-state
// launch) and still advance the recurrence correctly over several steps: outputs and both
// states track a float64 host oracle of the same decode.
// fak-test:runtime fast est=1s
func TestCUDAQwen35GDNDecodeWideHeadsTakeGlobalStateFallback(t *testing.T) {
	for _, tc := range []struct {
		name string
		g    cudaGDNGeometry
	}{
		{"wide-key-head", cudaGDNGeometry{hidden: 16, nK: 2, nV: 4, kHd: 160, vHd: 32, kernel: 4, eps: 1e-5}},
		{"wide-value-head", cudaGDNGeometry{hidden: 16, nK: 1, nV: 2, kHd: 16, vHd: 320, kernel: 3, eps: 1e-5}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			be := cudaGDNBackend(t)
			g := tc.g
			o := newCUDAGDNOperands(t, be, g)
			rng := cudaGDNLCG(0x9359f)
			convInit := rng.vector((g.kernel-1)*g.convDim(), 1.0)
			recurrentInit := rng.vector(g.nV*g.kHd*g.vHd, 0.2)
			o.convState = uploadCUDAGDN(t, be, []int{g.kernel - 1, g.convDim()}, convInit, MemoryKVCache, "qwen35-gdn-wide-conv-state")
			o.recurrentState = uploadCUDAGDN(t, be, []int{g.nV, g.kHd, g.vHd}, recurrentInit, MemoryKVCache, "qwen35-gdn-wide-recurrent-state")
			host := qwen35GDNDecodeHostOracle{
				g: g, inQKV: be.Read(o.inQKV), inZ: be.Read(o.inZ), inB: be.Read(o.inB), inA: be.Read(o.inA),
				convW: be.Read(o.convW), aLog: be.Read(o.aLog), dtBias: be.Read(o.dtBias), norm: be.Read(o.norm),
				outW: be.Read(o.outW), conv: qwen35GDNFloat64s(convInit), recurrent: qwen35GDNFloat64s(recurrentInit),
			}
			launches := qwen35GDNDecodeRegisterStateLaunchesForTest()
			const steps = 3
			for step := 0; step < steps; step++ {
				input := rng.vector(g.hidden, 1.0)
				o.x = uploadCUDAGDN(t, be, []int{g.hidden}, input, MemoryActivation, "qwen35-gdn-wide-input")
				out, _, _, err := o.decode(be, g)
				if err != nil {
					t.Fatalf("decode step %d: %v", step, err)
				}
				want := host.step(input)
				requireQwen35GDNClose(t, "output", step, be.Read(out), want)
			}
			if n := qwen35GDNDecodeRegisterStateLaunchesForTest() - launches; n != 0 {
				t.Fatalf("%d register-state launches for a head the register kernel cannot hold, want 0", n)
			}
			requireQwen35GDNClose(t, "conv state", steps, be.Read(o.convState), host.conv)
			requireQwen35GDNClose(t, "recurrent state", steps, be.Read(o.recurrentState), host.recurrent)
		})
	}
}

func qwen35GDNFloat64s(v []float32) []float64 {
	out := make([]float64, len(v))
	for i, x := range v {
		out[i] = float64(x)
	}
	return out
}

func requireQwen35GDNClose(t *testing.T, name string, step int, got []float32, want []float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s after step %d: len %d, want %d", name, step, len(got), len(want))
	}
	var scale float64
	for _, w := range want {
		scale = math.Max(scale, math.Abs(w))
	}
	for i := range want {
		if d := math.Abs(float64(got[i]) - want[i]); !(d <= 1e-4*(1+scale)) {
			t.Fatalf("%s after step %d: [%d] = %g, host oracle %g (|d|=%.3g, scale %.3g)", name, step, i, got[i], want[i], d, scale)
		}
	}
}

// qwen35GDNDecodeHostOracle is a float64 transcription of one fcuda_qwen35_gdn_decode_f32
// step with F32 weights: fused in-projection, causal conv + SiLU, q/k L2 norm, gated delta
// recurrence, gated RMS out-norm and out-projection.
type qwen35GDNDecodeHostOracle struct {
	g                                         cudaGDNGeometry
	inQKV, inZ, inB, inA, convW, aLog, dtBias []float32
	norm, outW                                []float32
	conv, recurrent                           []float64
}

func (o *qwen35GDNDecodeHostOracle) step(x []float32) []float64 {
	g := o.g
	matvec := func(w []float32, rows, cols int, in []float64) []float64 {
		out := make([]float64, rows)
		for r := range out {
			for c := 0; c < cols; c++ {
				out[r] += float64(w[r*cols+c]) * in[c]
			}
		}
		return out
	}
	silu := func(v float64) float64 { return v / (1 + math.Exp(-v)) }
	xs := qwen35GDNFloat64s(x)
	mixed := matvec(o.inQKV, g.convDim(), g.hidden, xs)
	z := matvec(o.inZ, g.valueDim(), g.hidden, xs)
	b := matvec(o.inB, g.nV, g.hidden, xs)
	a := matvec(o.inA, g.nV, g.hidden, xs)
	convOut := make([]float64, g.convDim())
	for c := range convOut {
		acc := float64(o.convW[c*g.kernel+g.kernel-1]) * mixed[c]
		for j := 0; j < g.kernel-1; j++ {
			acc += float64(o.convW[c*g.kernel+j]) * o.conv[j*g.convDim()+c]
		}
		convOut[c] = silu(acc)
		if g.kernel > 1 {
			for j := 0; j < g.kernel-2; j++ {
				o.conv[j*g.convDim()+c] = o.conv[(j+1)*g.convDim()+c]
			}
			o.conv[(g.kernel-2)*g.convDim()+c] = mixed[c]
		}
	}
	q := make([]float64, g.keyDim())
	k := make([]float64, g.keyDim())
	for h := 0; h < g.nK; h++ {
		var qss, kss float64
		for d := 0; d < g.kHd; d++ {
			qv, kv := convOut[h*g.kHd+d], convOut[g.keyDim()+h*g.kHd+d]
			qss += qv * qv
			kss += kv * kv
		}
		for d := 0; d < g.kHd; d++ {
			q[h*g.kHd+d] = convOut[h*g.kHd+d] / math.Sqrt(qss+1e-6) / math.Sqrt(float64(g.kHd))
			k[h*g.kHd+d] = convOut[g.keyDim()+h*g.kHd+d] / math.Sqrt(kss+1e-6)
		}
	}
	core := make([]float64, g.valueDim())
	for h := 0; h < g.nV; h++ {
		kh := h / (g.nV / g.nK)
		beta := 1 / (1 + math.Exp(-b[h]))
		gate := a[h] + float64(o.dtBias[h])
		if gate <= 20 {
			gate = math.Log1p(math.Exp(gate))
		}
		decay := math.Exp(-math.Exp(float64(o.aLog[h])) * gate)
		var ss float64
		for d := 0; d < g.vHd; d++ {
			state := func(i int) *float64 { return &o.recurrent[(h*g.kHd+i)*g.vHd+d] }
			var kvmem float64
			for i := 0; i < g.kHd; i++ {
				*state(i) *= decay
				kvmem += *state(i) * k[kh*g.kHd+i]
			}
			delta := (convOut[2*g.keyDim()+h*g.vHd+d] - kvmem) * beta
			var readout float64
			for i := 0; i < g.kHd; i++ {
				*state(i) += k[kh*g.kHd+i] * delta
				readout += *state(i) * q[kh*g.kHd+i]
			}
			core[h*g.vHd+d] = readout
			ss += readout * readout
		}
		inv := 1 / math.Sqrt(ss/float64(g.vHd)+float64(g.eps))
		for d := 0; d < g.vHd; d++ {
			vd := h*g.vHd + d
			core[vd] = float64(o.norm[d]) * core[vd] * inv * silu(z[vd])
		}
	}
	return matvec(o.outW, g.hidden, g.valueDim(), core)
}
